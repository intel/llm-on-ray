import ray
import torch
import torch.distributed as dist
import deepspeed

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from ray.air.util.torch_dist import (
    TorchDistributedWorker,
    _get_node_and_gpu_ids,
    _init_torch_distributed,
    _shutdown_torch_distributed,
    shutdown_torch_dist_process_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.air import ScalingConfig
from ray.train._internal.utils import get_address_and_port
from typing import List
import os
from collections import defaultdict
from predictor import Predictor
from utils import get_torch_dtype


from inference_config import InferenceConfig, DEVICE_CPU, DEVICE_XPU, IPEX_PRECISION_BF16

class DSPipeline:
    def __init__(
        self,
        infer_conf: InferenceConfig,
        stopping_criteria
    ):
        self.device = torch.device(infer_conf.device)
        self.stopping_criteria = stopping_criteria

        model_desc = infer_conf.model_description
        model_config = model_desc.config
        hf_config = AutoConfig.from_pretrained(model_desc.model_id_or_path, torchscript=True, trust_remote_code=model_config.trust_remote_code)

        # get correct torch type for loading HF model
        torch_dtype = get_torch_dtype(infer_conf, hf_config)
        self.model = AutoModelForCausalLM.from_pretrained(model_desc.model_id_or_path,
                                                          config=hf_config,
                                                          torch_dtype=torch_dtype,
                                                          low_cpu_mem_usage=True,
                                                          **model_config.dict())
        
        if model_desc.peft_model_id_or_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, model_desc.peft_model_id_or_path)
            if model_desc.peft_type == "deltatuner":
                from deltatuner import DeltaTunerModel
                self.model = DeltaTunerModel.from_pretrained(self.model, model_desc.peft_model_id_or_path)
            self.model = self.model.merge_and_unload()

        self.model = self.model.eval().to(self.device)
        # to channels last
        self.model = self.model.to(memory_format=torch.channels_last)
        self.model.eval()

    def streaming_generate(self, inputs, streamer, **generate_kwargs):
        self.model.generate(inputs,
                    stopping_criteria=self.stopping_criteria,
                    streamer=streamer,
                    **generate_kwargs)

    def generate(self, inputs, **config):
        gen_tokens = self.model.generate(
            inputs,
            stopping_criteria=self.stopping_criteria,
            **config
        )
        return gen_tokens

class PredictionWorker(TorchDistributedWorker):
    """A PredictionWorker is a Ray remote actor that runs a single shard of a DeepSpeed job.

    Multiple PredictionWorkers of the same WorkerGroup form a PyTorch DDP process
    group and work together under the orchestration of DeepSpeed.
    """
    def __init__(self, world_size: int, infer_conf: InferenceConfig, stopping_criteria):
        self.world_size = world_size
        self.infer_conf = infer_conf
        self.stopping_criteria = stopping_criteria

    def init_model(self, local_rank: int):
        """Initialize model for inference."""

        if self.infer_conf.device == DEVICE_CPU:
            replace_with_kernel_inject = False
        elif self.infer_conf.device == DEVICE_XPU:
            replace_with_kernel_inject = False
        else:
            replace_with_kernel_inject = True

        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(self.world_size)

        pipe = DSPipeline(
            self.infer_conf,
            stopping_criteria=self.stopping_criteria,
        )

        pipe.model = deepspeed.init_inference(
            pipe.model,
            mp_size=self.world_size,
            dtype=torch.bfloat16,
            replace_with_kernel_inject=replace_with_kernel_inject
        )

        if self.infer_conf.ipex.enabled:
            import intel_extension_for_pytorch as ipex
            try: ipex._C.disable_jit_linear_repack()
            except: pass
            pipe.model = ipex.optimize_transformers(
                pipe.model.eval(),
                dtype=torch.bfloat16 if self.infer_conf.ipex.precision == IPEX_PRECISION_BF16 else torch.float32,
                inplace=True)

        self.generator = pipe

    def streaming_generate(self, inputs, streamer, **config):
        self.generator.streaming_generate(inputs, streamer, **config)

    def generate(self, inputs, **config):
        return self.generator.generate(inputs, **config)

class DeepSpeedPredictor(Predictor):
    def __init__(self, infer_conf: InferenceConfig) -> None:
        super().__init__(infer_conf)
        # Scaling config for one worker group.
        resource = {"CPU": infer_conf.cpus_per_worker}
        use_gpu = True if (infer_conf.device == "cuda") else False
        if use_gpu:
            resource["GPU"] = infer_conf.gpus_per_worker
        scaling_conf = ScalingConfig(
            use_gpu=use_gpu,
            # spawn N-1 workers because this process is a worker too
            num_workers=infer_conf.workers_per_group - 1,
            resources_per_worker=resource
        )
        print(scaling_conf)

        self._init_worker_group(scaling_conf)
        self.model = self.worker.generator.model
        self.configure_tokenizer(infer_conf.model_description.model_id_or_path)

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
        # For now it does not guarantee the placement group does not contain
        # this process' node if enough resource is available
        self.pg = scaling_config.as_placement_group_factory().to_placement_group()
        prediction_worker_cls = ray.remote(PredictionWorker).options(
            num_cpus=scaling_config.num_cpus_per_worker,
            num_gpus=scaling_config.num_gpus_per_worker,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=self.pg, placement_group_capture_child_tasks=True
            ),
        )

        # this process itself is rank 0 worker
        # create the worker instance
        num_workers = self.infer_conf.workers_per_group
        self.worker = PredictionWorker(num_workers, self.infer_conf, self.stopping_criteria)

        # Create the rest prediction workers.
        self.prediction_workers = [
            prediction_worker_cls.remote(num_workers, self.infer_conf,
                self.stopping_criteria)
            for i in range(scaling_config.num_workers)
        ]

        # Check if dist is available which is needed to establish communication
        if not dist.is_available():
            raise RuntimeError("Distributed torch is not available.")

        # Build a map from node_id to workers on that node.
        node_and_gpu_ids = [_get_node_and_gpu_ids()]
        node_and_gpu_ids.extend(ray.get(
            [w.execute.remote(_get_node_and_gpu_ids) for w in self.prediction_workers]
        ))
        # All the workers on a specific node.
        node_to_workers = defaultdict(list)
        # All the gpu ids visible to all the workers on a specific node.
        node_to_gpu_ids = defaultdict(set)
        for i, (node_id, gpu_ids) in enumerate(node_and_gpu_ids):
            node_to_workers[node_id].append(i)
            # Force list.
            if not isinstance(gpu_ids, list):
                gpu_ids = [gpu_ids]
            # It is possible for a worker to have access to multiple GPUs.
            for gpu_id in gpu_ids:
                node_to_gpu_ids[node_id].add(gpu_id)

        master_addr, master_port = get_address_and_port()
        setup_futures = []
        world_size = num_workers
        local_ranks = [0]
        for i, worker in enumerate(self.prediction_workers):
            rank = i+1
            node_id = node_and_gpu_ids[rank][0]
            local_rank = node_to_workers[node_id].index(rank)
            local_world_size = len(node_to_workers[node_id])
            setup_futures.append(
                worker.execute.remote(
                    _init_torch_distributed,
                    init_method="env",
                    backend="ccl" if self.infer_conf.device != "cuda" else "nccl",
                    rank=rank,
                    world_size=world_size,
                    local_rank=local_rank,
                    local_world_size=local_world_size,
                    master_addr=master_addr,
                    master_port=master_port,
                    # list(set) will sort the gpu ids, so VISIBLE_CUDA_DEVICES
                    # is always sorted.
                    gpu_ids=list(node_to_gpu_ids[node_id]),
                )
            )
            local_ranks.append(local_rank)
        node_id = node_and_gpu_ids[0][0]
        local_rank = node_to_workers[node_id].index(0)
        local_world_size = len(node_to_workers[node_id])
        _init_torch_distributed(
            init_method="env",
            backend="ccl" if self.infer_conf.device != "cuda" else "nccl",
            rank=0,
            world_size=world_size,
            local_rank=local_rank,
            local_world_size=local_world_size,
            master_addr=master_addr,
            master_port=master_port,
            # list(set) will sort the gpu ids, so VISIBLE_CUDA_DEVICES
            # is always sorted.
            gpu_ids=list(node_to_gpu_ids[node_id]),
        )
        ray.get(setup_futures)

        # Initialize the model on each worker.
        futures = [
            worker.init_model.remote(local_rank)
            for worker, local_rank in zip(self.prediction_workers, local_ranks[1:])
        ]
        self.worker.init_model(0)
        ray.get(futures)

    def streaming_generate(self, prompt, streamer, **config):
        input_ids = self.tokenize_inputs(prompt)
        inputs_ref = ray.put(input_ids)
        for worker in self.prediction_workers:
            worker.streaming_generate.remote(inputs_ref, self._create_dummy_streamer(), **config)
        self.worker.streaming_generate(inputs_ids, streamer, **config)

    def generate(self, prompt, **config):
        input_ids = self.tokenize_inputs(prompt)
        inputs_ref = ray.put(input_ids)
        futures = [
            worker.generate.remote(inputs_ref, **config)
            for worker in self.prediction_workers
        ]
        gen_tokens = self.worker.generate(input_ids, **config)
        ray.get(futures)
        return self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]

    def get_streamer(self):
        from transformers import TextStreamer
        from typing import Optional
        from ray.util.queue import Queue

        class RayTextIteratorStreamer(TextStreamer):
            def __init__(
                self, tokenizer: "AutoTokenizer", skip_prompt: bool = False, timeout: Optional[float] = None, **decode_kwargs
            ):
                super().__init__(tokenizer, skip_prompt, **decode_kwargs)
                self.text_queue = Queue()
                self.stop_signal = None
                self.timeout = timeout

            def on_finalized_text(self, text: str, stream_end: bool = False):
                self.text_queue.put(text, timeout=self.timeout)
                if stream_end:
                    self.text_queue.put(self.stop_signal, timeout=self.timeout)

            def __iter__(self):
                return self

            def __next__(self):
                value = self.text_queue.get(timeout=self.timeout)
                if value == self.stop_signal:
                    raise StopIteration()
                else:
                    return value
        return RayTextIteratorStreamer(self.tokenizer, skip_special_tokens=True)

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
