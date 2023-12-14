import ray
import torch
import deepspeed

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from ray.air.util.torch_dist import (
    TorchDistributedWorker,
    init_torch_dist_process_group,
    shutdown_torch_dist_process_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.air import ScalingConfig
from typing import List
import os
from predictor import Predictor
from peft import PeftModel
from deltatuner import DeltaTunerModel
from inference_config import InferenceConfig, DEVICE_CPU, DEVICE_XPU, IPEX_PRECISION_BF16

class DSPipeline:
    def __init__(
        self,
        inferenceConfig: InferenceConfig,
        pad_token_id,
        stopping_criteria
    ):
        self.device = torch.device(inferenceConfig.device)
        self.pad_token_id = pad_token_id
        self.stopping_criteria = stopping_criteria

        model_desc = inferenceConfig.model_description
        model_config = model_desc.config
        hf_config = AutoConfig.from_pretrained(model_desc.model_id_or_path, torchscript=True, trust_remote_code=model_config.trust_remote_code)

        # get correct torch type for loading HF model
        torch_dtype = Predictor.get_torch_dtype(inferenceConfig, hf_config)
        self.model = AutoModelForCausalLM.from_pretrained(model_desc.model_id_or_path,
                                                          config=hf_config,
                                                          torch_dtype=torch_dtype,
                                                          low_cpu_mem_usage=True,
                                                          **model_config.dict())
        
        if model_desc.peft_model_id_or_path:
            self.model = PeftModel.from_pretrained(self.model, model_desc.peft_model_id_or_path)
            if model_desc.peft_type == "deltatuner":
                self.model = DeltaTunerModel.from_pretrained(self.model, model_desc.peft_model_id_or_path)
            self.model = self.model.merge_and_unload()

        self.model = self.model.eval().to(self.device)
        # to channels last
        self.model = self.model.to(memory_format=torch.channels_last)
        self.model.eval()

    def streaming_generate(self, inputs, streamer, **generate_kwargs):
        self.model.generate(**inputs,
                    pad_token_id=self.pad_token_id,
                    stopping_criteria=self.stopping_criteria,
                    streamer=streamer,
                    **generate_kwargs)

    def generate(self, inputs, **config):
        gen_tokens = self.model.generate(
            **inputs,
            pad_token_id=self.pad_token_id,
            stopping_criteria=self.stopping_criteria,
            **config
        )
        return gen_tokens

@ray.remote
class PredictionWorker(TorchDistributedWorker):
    """A PredictionWorker is a Ray remote actor that runs a single shard of a DeepSpeed job.

    Multiple PredictionWorkers of the same WorkerGroup form a PyTorch DDP process
    group and work together under the orchestration of DeepSpeed.
    """
    def __init__(self, world_size: int, inferenceConfig: InferenceConfig, pad_token_id, stopping_criteria):
        self.world_size = world_size
        self.inferenceConfig = inferenceConfig
        self.pad_token_id = pad_token_id
        self.stopping_criteria = stopping_criteria

    def init_model(self, local_rank: int):
        """Initialize model for inference."""

        if self.inferenceConfig.device == DEVICE_CPU:
            replace_with_kernel_inject = False
        elif self.inferenceConfig.device == DEVICE_XPU:
            replace_with_kernel_inject = False
        else:
            replace_with_kernel_inject = True

        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(self.world_size)

        pipe = DSPipeline(
            self.inferenceConfig,
            pad_token_id=self.pad_token_id,
            stopping_criteria=self.stopping_criteria,
        )

        pipe.model = deepspeed.init_inference(
            pipe.model,
            mp_size=self.world_size,
            dtype=torch.bfloat16,
            replace_with_kernel_inject=replace_with_kernel_inject
        )

        if self.inferenceConfig.ipex.enabled:
            import intel_extension_for_pytorch as ipex
            try: ipex._C.disable_jit_linear_repack()
            except: pass
            pipe.model = ipex.optimize_transformers(
                pipe.model.eval(),
                dtype=torch.bfloat16 if self.inferenceConfig.ipex.precision == IPEX_PRECISION_BF16 else torch.float32,
                inplace=True)

        self.generator = pipe

    def streaming_generate(self, inputs, streamer, **config):
        self.generator.streaming_generate(inputs, streamer, **config)

    def generate(self, inputs, **config):
        return self.generator.generate(inputs, **config)

class DeepSpeedPredictor(Predictor):
    def __init__(self, inferenceConfig: InferenceConfig, pad_token_id, stopping_criteria) -> None:
        self.inferenceConfig = inferenceConfig
        self.pad_token_id = pad_token_id
        self.stopping_criteria = stopping_criteria

        use_gpu = True if (inferenceConfig.device == "cuda") else False

        # Scaling config for one worker group.
        resource = {"CPU": inferenceConfig.cpus_per_worker}
        if inferenceConfig.device == "cuda":
            resource["GPU"] = inferenceConfig.gpus_per_worker
        scaling_conf = ScalingConfig(
            use_gpu=use_gpu,
            num_workers=inferenceConfig.workers_per_group,
            resources_per_worker=resource
        )

        print(scaling_conf)

        self._init_worker_group(scaling_conf)

    def __del__(self):
        shutdown_torch_dist_process_group(self.prediction_workers)

    # Use dummy streamer to ignore other workers' ouputs
    def _create_dummy_streamer(self):
        class DummyStreamer():
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
            prediction_worker_cls.remote(scaling_config.num_workers, self.inferenceConfig,
                self.pad_token_id, self.stopping_criteria)
            for i in range(scaling_config.num_workers)
        ]

        # Initialize torch distributed process group for the workers.
        local_ranks = init_torch_dist_process_group(self.prediction_workers, backend="ccl" if self.inferenceConfig.device != "cuda" else "nccl")

        # Initialize the model on each worker.
        ray.get([
            worker.init_model.remote(local_rank)
            for worker, local_rank in zip(self.prediction_workers, local_ranks)
        ])

    def streaming_generate(self, inputs, streamer, **config):
        inputs_ref = ray.put(inputs)
        self.prediction_workers[0].streaming_generate.remote(inputs_ref, streamer, **config)
        for worker in self.prediction_workers[1:]:
                worker.streaming_generate.remote(inputs_ref, self._create_dummy_streamer(), **config)

    def generate(self, inputs, **config):
        inputs_ref = ray.put(inputs)
        prediction = ray.get(
            [
                worker.generate.remote(inputs_ref, **config)
                for worker in self.prediction_workers
            ]
        )[0]
        return prediction

    def predict(
        self,
        data: List[str],
        **kwargs
    ) -> str:
        data_ref = ray.put(data)
        prediction = ray.get(
            [
                worker.generate.remote(data_ref, **kwargs)
                for worker in self.prediction_workers
            ]
        )

        return prediction
