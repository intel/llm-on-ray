# Reference: https://github.com/huggingface/optimum-habana/blob/main/examples/text-generation/utils.py
import os
import tempfile
import ray
from torch_dist import (
    TorchDistributedWorker,
    init_torch_dist_process_group,
)
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

    def get_streamer(self):
        return ray.get(self.deepspeed_workers[0].get_streamer.remote())

    def generate(self, prompt, **config):
        return ray.get(
            [worker.generate.remote(prompt, **config) for worker in self.deepspeed_workers]
        )[0]

    def streaming_generate(self, prompt, streamer, **config):
        self.deepspeed_workers[0].streaming_generate.remote(prompt, streamer, **config)
        for worker in self.deepspeed_workers[1:]:
            worker.streaming_generate.remote(prompt, self._create_dummy_streamer(), **config)


@ray.remote
class HPUDeepSpeedWorker(TorchDistributedWorker):
    def __init__(self, infer_conf: InferenceConfig):
        self.infer_conf = infer_conf
        os.environ.setdefault("PT_HPU_LAZY_ACC_PAR_MODE", "0")
        os.environ.setdefault("PT_HPU_ENABLE_LAZY_COLLECTIVES", "true")

    def load_model_and_tokenizer(self):
        # after init_worker_group, these envvar should be set
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.device = torch.device("hpu")
        # optimize transformers for gaudi
        from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

        adapt_transformers_to_gaudi()
        model_desc = self.infer_conf.model_description
        model_config = model_desc.config
        self.load_model(model_desc, model_config)
        self.load_tokenizer(model_desc)

    def get_streamer(self):
        from utils import RayTextIteratorStreamer

        return RayTextIteratorStreamer(self.tokenizer, skip_special_tokens=True)

    def load_model(self, model_desc, model_config):
        import deepspeed
        from transformers import AutoModelForCausalLM, AutoConfig

        # bf16 is used for deepspeed
        model_dtype = torch.bfloat16

        config = AutoConfig.from_pretrained(
            model_desc.model_id_or_path,
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

    def load_tokenizer(self, model_desc):
        from transformers import AutoTokenizer

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
            streamer=streamer,
            lazy_mode=True,
            hpu_graphs=self.infer_conf.model_description.use_hpu_graphs,
            **config,
        )
