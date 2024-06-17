#
# Copyright 2023 The LLM-on-Ray Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# ===========================================================================
#
# This file is adapted from
# https://github.com/HabanaAI/optimum-habana-fork/blob/aa64051ae92d8940a49322b29d1dfa933b2d3250/examples/text-generation/utils.py
# Copyright 2023 Huggingface
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import tempfile
from typing import List, Union
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import (
    placement_group,
)
import torch
from optimum.habana.checkpoint_utils import (
    get_ds_injection_policy,
    model_on_meta,
    write_checkpoints_json,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TextIteratorStreamer,
)
from llm_on_ray.inference.torch_dist import (
    TorchDistributedWorker,
    init_torch_dist_process_group,
)
from llm_on_ray.inference.inference_config import (
    InferenceConfig,
    ModelGenerateResult,
)
from llm_on_ray.inference.predictor import (
    GenerateInput,
    GenerateOutput,
    Predictor,
    SinglePromptInput,
    MultiplePromptInput,
    MllmPromptInput,
)
from llm_on_ray.inference.utils import decide_torch_dtype


class HPUPredictor(Predictor):
    def __init__(self, infer_conf: InferenceConfig):
        super().__init__(infer_conf)

        model_desc = infer_conf.model_description
        # decide correct torch dtype for loading HF model
        decide_torch_dtype(infer_conf)

        self.use_lazy_mode = not infer_conf.hpu_model_config.torch_compile
        self.use_hpu_graphs = infer_conf.hpu_model_config.use_hpu_graphs

        if infer_conf.deepspeed:
            # DeepSpeed is enabled, start worker group
            # Prepare placement group
            resource_bundles = [
                {
                    "CPU": infer_conf.cpus_per_worker,
                    "HPU": infer_conf.hpus_per_worker,
                }
            ] * infer_conf.workers_per_group
            self._workers_pg = placement_group(resource_bundles, strategy="PACK")

            self.init_worker_group()
        else:
            # Not using DeepSpeed, load model locally
            # if using quantization
            if infer_conf.hpu_model_config.quant_config:
                # memory issue in hqt prep_model
                os.environ.setdefault("EXPERIMENTAL_WEIGHT_SHARING", "FALSE")
                # enable inference optimizations
                import habana_frameworks.torch.core as htcore

                htcore.hpu_set_env()

            # Tweak transformer to optimize performance on Gaudi
            from optimum.habana.transformers.modeling_utils import (
                adapt_transformers_to_gaudi,
            )

            adapt_transformers_to_gaudi()

            self.device = torch.device("hpu")
            model = AutoModelForCausalLM.from_pretrained(
                model_desc.model_id_or_path, **model_desc.config.dict()
            )

            # quantization
            if infer_conf.hpu_model_config.quant_config:
                import habana_quantization_toolkit

                habana_quantization_toolkit.prep_model(
                    model, config_path=infer_conf.hpu_model_config.quant_config
                )

            self.model = model.eval().to(self.device)
            # HPU graph runtime
            if self.use_hpu_graphs:
                from habana_frameworks.torch.hpu import (
                    wrap_in_hpu_graph,
                )  # pylint: disable=E0401

                self.model = wrap_in_hpu_graph(self.model)
            else:
                print("Warning: use_hpu_graphs is set to False. This will hurt the performance.")
            # torch compile
            if (
                infer_conf.hpu_model_config.torch_compile
                and self.model.config.model_type == "llama"
            ):
                self.model = get_torch_compiled_model(self.model)
            # Enable quantization inference mode
            if infer_conf.hpu_model_config.quant_config:
                htcore.hpu_initialize(model)

            self.tokenizer = load_tokenizer(model, model_desc.tokenizer_name_or_path)

    def init_worker_group(self):
        deepspeed_worker_cls = HPUDeepSpeedWorker.options(
            num_cpus=self.infer_conf.cpus_per_worker,
            resources={"HPU": self.infer_conf.hpus_per_worker},
            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=self._workers_pg),
        )
        self.deepspeed_workers = [
            deepspeed_worker_cls.remote(self.infer_conf)
            for _ in range(self.infer_conf.workers_per_group)
        ]
        init_torch_dist_process_group(self.deepspeed_workers, backend="hccl")
        futures = [worker.load_model_and_tokenizer.remote() for worker in self.deepspeed_workers]
        ray.wait(futures)

    # Use dummy streamer to ignore other workers' ouputs
    def _create_dummy_streamer(self):
        class DummyStreamer:
            def put(self, value):
                pass

            def end(self):
                pass

        return DummyStreamer()

    def _process_config(self, config):
        config["lazy_mode"] = self.use_lazy_mode
        config["hpu_graphs"] = self.use_hpu_graphs
        # max_new_tokens is required for hpu
        if config.get("max_new_tokens", None) is None:
            config["max_new_tokens"] = 128

    def get_streamer(self):
        if self.infer_conf.deepspeed:
            return ray.get(self.deepspeed_workers[0].get_streamer.remote())
        else:
            return TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, timeout=0, skip_special_tokens=True
            )

    # FIXME: support MllmPromptInput
    def generate(self, input: GenerateInput, **config) -> GenerateOutput:
        if isinstance(input, tuple):
            raise TypeError("HPUPredictor doesn't support MLLM prompts for now!")

        prompt = input

        self._process_config(config)

        if self.infer_conf.deepspeed:
            return ray.get(
                [worker.generate.remote(prompt, **config) for worker in self.deepspeed_workers]
            )[0]
        else:
            input_ids, input_length = self.tokenize_inputs(prompt)
            gen_tokens = self.model.generate(
                input_ids, stopping_criteria=self.stopping_criteria, ignore_eos=False, **config
            )
            decode_result = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            # FIXME generate_length is not correct
            results = [
                ModelGenerateResult(
                    text=decode_result[i],
                    input_length=input_length,
                    generate_length=gen_tokens.size()[1] - input_length,
                )
                for i in range(len(prompt))
            ]
            return results[0] if isinstance(input, SinglePromptInput) else results

    def streaming_generate(self, prompt, streamer, **config):
        self._process_config(config)
        if self.infer_conf.deepspeed:
            self.deepspeed_workers[0].streaming_generate.remote(prompt, streamer, **config)
            for worker in self.deepspeed_workers[1:]:
                worker.streaming_generate.remote(prompt, self._create_dummy_streamer(), **config)
        else:
            input_ids, _ = self.tokenize_inputs(prompt)
            self.model.generate(
                input_ids,
                stopping_criteria=self.stopping_criteria,
                streamer=streamer,
                ignore_eos=False,
                **config,
            )


def load_tokenizer(model, tokenizer_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if not model.config.is_encoder_decoder:
        tokenizer.padding_side = "left"
    # Some models like GPT2 do not have a PAD token so we have to set it if necessary
    if model.config.model_type == "llama":
        # unwind broken decapoda-research config
        model.generation_config.pad_token_id = 0
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2
        tokenizer.bos_token_id = model.generation_config.bos_token_id
        tokenizer.eos_token_id = model.generation_config.eos_token_id
        tokenizer.pad_token_id = model.generation_config.pad_token_id
        tokenizer.pad_token = tokenizer.decode(tokenizer.pad_token_id)
        tokenizer.eos_token = tokenizer.decode(tokenizer.eos_token_id)
        tokenizer.bos_token = tokenizer.decode(tokenizer.bos_token_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    return tokenizer


def get_torch_compiled_model(model):
    model.model = torch.compile(model.model, backend="hpu_backend")
    return model


# FIXME: consolidate DeepSpeedPredictor's PredictionWorker and HPUDeepSpeedWorker
@ray.remote
class HPUDeepSpeedWorker(TorchDistributedWorker):
    def __init__(self, infer_conf: InferenceConfig):
        self.infer_conf = infer_conf
        os.environ.setdefault("PT_HPU_LAZY_ACC_PAR_MODE", "0")
        os.environ.setdefault("PT_HPU_ENABLE_LAZY_COLLECTIVES", "true")
        # if using quantization
        if infer_conf.hpu_model_config.quant_config:
            # memory issue in hqt prep_model
            os.environ.setdefault("EXPERIMENTAL_WEIGHT_SHARING", "FALSE")
            # enable inference optimizations
            import habana_frameworks.torch.core as htcore

            htcore.hpu_set_env()

    def load_model_and_tokenizer(self):
        # after init_worker_group, these envvar should be set
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.device = torch.device("hpu")
        # optimize transformers for gaudi
        from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

        adapt_transformers_to_gaudi()
        self.load_model()
        model_desc = self.infer_conf.model_description
        self.tokenizer = load_tokenizer(self.model, model_desc.tokenizer_name_or_path)
        if self.infer_conf.hpu_model_config.quant_config:
            import habana_frameworks.torch.core as htcore

            htcore.hpu_initialize(self.model)

    def get_streamer(self):
        from llm_on_ray.inference.utils import RayTextIteratorStreamer

        return RayTextIteratorStreamer(self.tokenizer, skip_special_tokens=True)

    def load_model(self):
        import deepspeed

        model_desc = self.infer_conf.model_description
        model_dtype = model_desc.config.torch_dtype
        config = AutoConfig.from_pretrained(model_desc.model_id_or_path, **model_desc.config.dict())
        load_to_meta = model_on_meta(config)

        if load_to_meta:
            with deepspeed.OnDevice(dtype=model_dtype, device="meta"):
                model = AutoModelForCausalLM.from_config(config, torch_dtype=model_dtype)

            checkpoints_json = tempfile.NamedTemporaryFile(suffix=".json", mode="+w")
            if model_desc.config.use_auth_token:
                auth_token = model_desc.config.use_auth_token
            else:
                auth_token = None
            write_checkpoints_json(
                model_desc.model_id_or_path,
                self.local_rank,
                checkpoints_json,
                token=auth_token,
            )
        else:
            with deepspeed.OnDevice(dtype=model_dtype, device="cpu"):
                model = AutoModelForCausalLM.from_pretrained(
                    model_desc.model_id_or_path, **model_desc.config.dict()
                )
        model.eval()

        ds_inference_kwargs = {"dtype": model_dtype}
        ds_inference_kwargs["tensor_parallel"] = {"tp_size": self.world_size}
        ds_inference_kwargs["enable_cuda_graph"] = self.infer_conf.hpu_model_config.use_hpu_graphs
        ds_inference_kwargs["injection_policy"] = get_ds_injection_policy(config)
        if load_to_meta:
            ds_inference_kwargs["checkpoint"] = checkpoints_json.name

        engine = deepspeed.init_inference(model, **ds_inference_kwargs)
        self.model = engine.module

        if self.model.config.model_type == "llama":

            def patch_scoped_linear_all_reduce(model):
                from deepspeed.module_inject.layers import LinearAllreduce
                from optimum.habana.transformers.models.modeling_all_models import (
                    ScopedLinearAllReduce,
                )

                for name, module in model.named_children():
                    if type(module) is LinearAllreduce:
                        SL = ScopedLinearAllReduce(mod=module)
                        setattr(model, name, SL)
                    patch_scoped_linear_all_reduce(module)

            patch_scoped_linear_all_reduce(self.model)
        # quantization
        if self.infer_conf.hpu_model_config.quant_config:
            import habana_quantization_toolkit

            habana_quantization_toolkit.prep_model(
                self.model, config_path=self.infer_conf.hpu_model_config.quant_config
            )
        # torch compile
        if (
            self.infer_conf.hpu_model_config.torch_compile
            and self.model.config.model_type == "llama"
        ):
            self.model = get_torch_compiled_model(self.model)

    def tokenize(self, prompt):
        input_tokens = self.tokenizer(prompt, return_tensors="pt", padding=True)
        # TODO reduce_recompile?
        return input_tokens.input_ids.to(device=self.device)

    def generate(self, prompt, **config):
        input_ids = self.tokenize(prompt)
        gen_tokens = self.model.generate(
            input_ids,
            ignore_eos=False,
            **config,
        )
        decode_result = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        results = [
            ModelGenerateResult(
                text=decode_result[i],
                input_length=input_ids.size()[1],
                generate_length=gen_tokens.size()[1] - input_ids.size()[1],
            )
            for i in range(len(prompt))
        ]
        return results[0] if isinstance(prompt, str) else results

    def streaming_generate(self, prompt, streamer, **config):
        input_ids = self.tokenize(prompt)
        self.model.generate(
            input_ids,
            streamer=streamer,
            ignore_eos=False,
            **config,
        )
