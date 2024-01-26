# Reference: https://github.com/huggingface/optimum-habana/blob/main/examples/text-generation/utils.py
import os
import tempfile
import deepspeed
import ray
from ray.air.util.torch_dist import (
    TorchDistributedWorker,
    init_torch_dist_process_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import (
    placement_group,
)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from optimum.habana.checkpoint_utils import (
    get_ds_injection_policy,
    model_on_meta,
    write_checkpoints_json,
)
from predictor import Predictor
from inference.inference_config import (
    InferenceConfig,
)


class HPUDeepSpeedPredictor(Predictor):
    def __init__(self, infer_conf: InferenceConfig):
        super().__init__(infer_conf)

        resource_bundles = [
            {
                "CPU": infer_conf.cpus_per_worker,
                "HPU": infer_conf.hpus_per_worker,
            }
        ] * infer_conf.workers_per_group
        self._workers_pg = placement_group(resource_bundles, strategy="PACK")

        self.init_worker_group()

    def init_worker_group(self):
        deepspeed_worker_cls = HPUDeepSpeedWorker.options(
            num_cpus=self.infer_conf.num_cpus_per_worker,
            resources={"HPU": self.infer_conf.num_hpus_per_worker},
            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=self._workers_pg),
        )
        self.deepspeed_workers = [
            deepspeed_worker_cls.remote(self.infer_conf) for _ in self.infer_conf.workers_per_group
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

    def generate(self, prompt, **config):
        return ray.get(
            [worker.generate.remote(prompt, **config) for worker in self.deepspeed_workers]
        )[0]

    def streaming_generate(self, prompt, streamer, **config):
        pass


@ray.remote
class HPUDeepSpeedWorker(TorchDistributedWorker):
    def __init__(self, infer_conf: InferenceConfig):
        self.infer_conf = infer_conf

    def load_model_and_tokenizer(self):
        # after init_worker_group, these envvar should be set
        self.world_size = os.environ["WORLD_SIZE"]
        self.local_rank = os.environ["LOCAL_RANK"]
        model_desc = self.infer_conf.model_description
        model_config = model_desc.config
        self.load_model(model_desc, model_config)
        self.load_tokenizer(model_desc)

    def load_model(self, model_desc, model_config):
        # bf16 is used for deepspeed
        model_dtype = torch.bfloat16

        config = AutoConfig.from_pretrained(
            model_desc.model_name_or_path,
            torch_dtype=model_dtype,
            trust_remote_code=model_config.trust_remote_code,
        )
        load_to_meta = model_on_meta(config)

        if load_to_meta:
            with deepspeed.OnDevice(dtype=model_dtype, device="meta"):
                model = AutoModelForCausalLM.from_config(config, torch_dtype=model_dtype)

            checkpoints_json = tempfile.NamedTemporaryFile(suffix=".json", mode="+w")
            write_checkpoints_json(
                model_desc.model_id_or_path, self.local_rank, checkpoints_json, token=""
            )
        else:
            with deepspeed.OnDevice(dtype=model_dtype, device="cpu"):
                model = AutoModelForCausalLM.from_pretrained(
                    model_desc.model_id_or_path, torch_dtype=model_dtype, **model_config.dict()
                )
        model.eval()

        ds_inference_kwargs = {"dtype": model_dtype}
        ds_inference_kwargs["tensor_parallel"] = {"tp_size": self.world_size}
        ds_inference_kwargs["enable_cuda_graph"] = model_desc.use_hpu_graphs
        ds_inference_kwargs["injection_policy"] = get_ds_injection_policy(config)
        if load_to_meta:
            ds_inference_kwargs["checkpoint"] = checkpoints_json.name

        engine = deepspeed.init_inference(model, **ds_inference_kwargs)
        self.model = engine.module

        if self.model.config.model_type == "llama":
            from optimum.habana.examples.text_generation.utils import patch_scoped_linear_all_reduce

            patch_scoped_linear_all_reduce(self.model)

    def load_tokenizer(self, model_desc):
        model = self.model
        tokenizer = AutoTokenizer.from_pretrained(model_desc.model_id_or_path)
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

        self.tokenizer = tokenizer

    def tokenize(self, prompt):
        input_tokens = self.tokenizer(prompt, return_tensors="pt", padding=True)
        # reduce_recompile?
        return input_tokens.input_ids.to(device=self.device)

    def generate(self, prompt, **config):
        input_ids = self.tokenize(prompt)
        gen_tokens = self.model.generate(
            input_ids,
            lazy_mode=True,
            hpu_graphs=self.infer_conf.model_description.use_hpu_graphs,
            **config,
        )
        return self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

    def streaming_generate(self, prompt, streamer, **config):
        input_ids = self.tokenize(prompt)
        self.model.generate(
            input_ids,
            streamer,
            lazy_mode=True,
            hpu_graphs=self.infer_conf.model_description.use_hpu_graphs,
            **config,
        )
