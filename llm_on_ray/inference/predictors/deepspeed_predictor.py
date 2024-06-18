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

import ray
import torch
import deepspeed

from transformers import AutoModelForCausalLM, AutoConfig
from llm_on_ray.inference.torch_dist import (
    TorchDistributedWorker,
    init_torch_dist_process_group,
    shutdown_torch_dist_process_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.air import ScalingConfig
from typing import List, Union
import os
from llm_on_ray.inference.predictor import (
    GenerateInput,
    GenerateOutput,
    MllmPromptInput,
    Predictor,
    SinglePromptInput,
)
from llm_on_ray.inference.utils import decide_torch_dtype
from llm_on_ray.inference.inference_config import (
    InferenceConfig,
    ModelGenerateResult,
    DEVICE_CPU,
    DEVICE_GPU,
    PRECISION_BF16,
)


class DSPipeline:
    def __init__(self, infer_conf: InferenceConfig, pad_token_id, stopping_criteria):
        self.device = torch.device(infer_conf.device)
        self.pad_token_id = pad_token_id
        self.stopping_criteria = stopping_criteria

        model_desc = infer_conf.model_description
        model_config = model_desc.config
        if infer_conf.model_description.config.use_auth_token:
            auth_token = infer_conf.model_description.config.use_auth_token
        else:
            auth_token = None
        hf_config = AutoConfig.from_pretrained(
            model_desc.model_id_or_path,
            torchscript=True,
            trust_remote_code=model_config.trust_remote_code,
            use_auth_token=auth_token,
        )

        # decide correct torch type for loading HF model
        decide_torch_dtype(infer_conf, hf_config)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_desc.model_id_or_path,
            config=hf_config,
            low_cpu_mem_usage=True,
            **model_config.dict(),
        )

        if model_desc.peft_model_id_or_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model,
                model_desc.peft_model_id_or_path,
                use_auth_token=auth_token,
            )

            self.model = self.model.merge_and_unload()

        self.model = self.model.eval().to(self.device)
        # to channels last
        self.model = self.model.to(memory_format=torch.channels_last)
        self.model.eval()

    def streaming_generate(self, inputs, streamer, **generate_kwargs):
        self.model.generate(
            inputs,
            pad_token_id=self.pad_token_id,
            stopping_criteria=self.stopping_criteria,
            streamer=streamer,
            **generate_kwargs,
        )

    def generate(self, inputs, **config):
        gen_tokens = self.model.generate(
            inputs,
            pad_token_id=self.pad_token_id,
            stopping_criteria=self.stopping_criteria,
            **config,
        )
        return gen_tokens


@ray.remote
class PredictionWorker(TorchDistributedWorker):
    """A PredictionWorker is a Ray remote actor that runs a single shard
    of a DeepSpeed job.

    Multiple PredictionWorkers of the same WorkerGroup form a PyTorch DDP
    process group and work together under the orchestration of DeepSpeed.
    """

    def __init__(
        self,
        world_size: int,
        infer_conf: InferenceConfig,
        pad_token_id,
        stopping_criteria,
    ):
        self.world_size = world_size
        self.infer_conf = infer_conf
        self.pad_token_id = pad_token_id
        self.stopping_criteria = stopping_criteria

    def init_model(self, local_rank: int):
        """Initialize model for inference."""

        if self.infer_conf.device == DEVICE_CPU:
            replace_with_kernel_inject = False
        elif self.infer_conf.device == DEVICE_GPU:
            replace_with_kernel_inject = False
        else:
            replace_with_kernel_inject = True

        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)

        pipe = DSPipeline(
            self.infer_conf,
            pad_token_id=self.pad_token_id,
            stopping_criteria=self.stopping_criteria,
        )

        pipe.model = deepspeed.init_inference(
            pipe.model,
            mp_size=self.world_size,
            dtype=torch.bfloat16,
            replace_with_kernel_inject=replace_with_kernel_inject,
        )

        if self.infer_conf.ipex.enabled:
            import intel_extension_for_pytorch as ipex

            try:
                ipex._C.disable_jit_linear_repack()
            except Exception:
                pass
            pipe.model = ipex.llm.optimize(
                pipe.model.eval(),
                dtype=torch.bfloat16
                if self.infer_conf.ipex.precision == PRECISION_BF16
                else torch.float32,
                inplace=True,
            )

        self.generator = pipe

    def streaming_generate(self, inputs, streamer, **config):
        self.generator.streaming_generate(inputs, streamer, **config)

    def generate(self, inputs, **config):
        return self.generator.generate(inputs, **config)


class DeepSpeedPredictor(Predictor):
    def __init__(self, infer_conf: InferenceConfig) -> None:
        super().__init__(infer_conf)
        # TODO this should be removed later
        self.pad_token_id = self.tokenizer.pad_token_id
        # Scaling config for one worker group.
        resource = {"CPU": infer_conf.cpus_per_worker}
        use_gpu = True if (infer_conf.device == "cuda") else False
        if use_gpu:
            resource["GPU"] = infer_conf.gpus_per_worker
        scaling_conf = ScalingConfig(
            use_gpu=use_gpu,
            num_workers=infer_conf.workers_per_group,
            resources_per_worker=resource,
        )

        print(scaling_conf)

        self._init_worker_group(scaling_conf)

    def __del__(self):
        shutdown_torch_dist_process_group(self.prediction_workers)

    # Use dummy streamer to ignore other workers' ouputs
    def _create_dummy_streamer(self):
        class DummyStreamer:
            def put(self, value):
                pass

            def end(self):
                pass

        return DummyStreamer()

    def _init_worker_group(self, scaling_config: ScalingConfig):
        """Create the worker group.

        Each worker in the group communicates with other workers through the
        torch distributed backend. The worker group is inelastic (a failure of
        one worker destroys the entire group). Each worker in the group
        recieves the same input data and outputs the same generated text.
        """

        # Start a placement group for the workers.
        self.pg = scaling_config.as_placement_group_factory().to_placement_group()
        prediction_worker_cls = PredictionWorker.options(
            num_cpus=scaling_config.num_cpus_per_worker,
            num_gpus=scaling_config.num_gpus_per_worker,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=self.pg, placement_group_capture_child_tasks=True
            ),
        )

        # Create the prediction workers.
        self.prediction_workers = [
            prediction_worker_cls.remote(
                scaling_config.num_workers,
                self.infer_conf,
                self.pad_token_id,
                self.stopping_criteria,
            )
            for i in range(scaling_config.num_workers)
        ]

        # Initialize torch distributed process group for the workers.
        local_ranks = init_torch_dist_process_group(
            self.prediction_workers,
            backend="ccl" if self.infer_conf.device != "cuda" else "nccl",
        )

        # Initialize the model on each worker.
        ray.get(
            [
                worker.init_model.remote(local_rank)
                for worker, local_rank in zip(self.prediction_workers, local_ranks)
            ]
        )

    def streaming_generate(self, prompt, streamer, **config):
        input_ids, _ = self.tokenize_inputs(prompt)
        inputs_ref = ray.put(input_ids)
        self.prediction_workers[0].streaming_generate.remote(inputs_ref, streamer, **config)
        for worker in self.prediction_workers[1:]:
            worker.streaming_generate.remote(inputs_ref, self._create_dummy_streamer(), **config)

    def generate(self, input: GenerateInput, **config) -> GenerateOutput:
        if isinstance(input, tuple):
            raise TypeError("DeepSpeedPredictor doesn't support MLLM input.")

        # Convert prompts to list
        prompts = [input] if isinstance(input, SinglePromptInput) else input

        input_ids, input_length = self.tokenize_inputs(prompts)
        inputs_ref = ray.put(input_ids)
        gen_tokens = ray.get(
            [worker.generate.remote(inputs_ref, **config) for worker in self.prediction_workers]
        )[0]

        decode_result = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        results = [
            ModelGenerateResult(
                text=decode_result[i],
                input_length=input_length,
                generate_length=gen_tokens.size()[1] - input_length,
            )
            for i in range(len(prompts))
        ]

        return results[0] if isinstance(input, SinglePromptInput) else results

    def get_streamer(self):
        from llm_on_ray.inference.utils import RayTextIteratorStreamer

        return RayTextIteratorStreamer(self.tokenizer, skip_special_tokens=True)

    def predict(self, data: List[str], **kwargs) -> str:
        data_ref = ray.put(data)
        prediction = ray.get(
            [worker.generate.remote(data_ref, **kwargs) for worker in self.prediction_workers]
        )

        return prediction
