from transformers import StoppingCriteria
import torch

from inference_config import InferenceConfig, DEVICE_CPU

def get_deployment_actor_options(infer_conf: InferenceConfig):
    _ray_env_key = "env_vars"
    # OMP_NUM_THREADS will be set by num_cpus, so not set in env
    _predictor_runtime_env_ipex = {
        "KMP_BLOCKTIME": "1",
        "KMP_SETTINGS": "1",
        "KMP_AFFINITY": "granularity=fine,compact,1,0",
        "MALLOC_CONF": "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
    }
    runtime_env = {_ray_env_key: {}}
    if infer_conf.ipex.enabled:
        runtime_env[_ray_env_key].update(_predictor_runtime_env_ipex)
    if infer_conf.deepspeed:
        runtime_env[_ray_env_key]["DS_ACCELERATOR"] = infer_conf.device
    # now PredictDeployment itself is a worker, we should require resources for it
    ray_actor_options = {"runtime_env": runtime_env}
    if infer_conf.device == "cpu":
        ray_actor_options["num_cpus"] = infer_conf.cpus_per_worker
    elif infer_conf.device == "cuda":
        ray_actor_options["num_gpus"] = infer_conf.gpus_per_worker
    elif infer_conf.device == "hpu":
        ray_actor_options["resources"] = {"HPU": infer_conf.hpus_per_worker}
    else:
        # TODO add xpu
        pass
    return ray_actor_options

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            length = 1  if len(stop.size())==0 else stop.size()[0]
            if torch.all((stop == input_ids[0][-length:])).item():
                return True
        return False

# used in inference with Gaudi
def max_input_len(input_text_length):
    if input_text_length <= 128:
        return 128
    elif input_text_length <= 512:
        return 512
    elif input_text_length <= 2048:
        return 2048
    else:
        print("Max support length is 4096")
        return 4096

def get_torch_dtype(infer_conf: InferenceConfig, hf_config) -> torch.dtype:
    '''
    return torch default dtype, a.k.a float32, if it's cpu only inference without ipex because
    bfloat16 is too slow and float16 is not supported in CPU
    '''
    if hf_config is None or is_cpu_without_ipex(infer_conf):
        return torch.get_default_dtype()
    if hasattr(hf_config, 'torch_dtype'):
        t = hf_config.torch_dtype
        if t:
            return t
    if hasattr(hf_config, '__getitem__'):
        t = hf_config['torch_dtype']
        if t:
            return t
    return torch.get_default_dtype()

def is_cpu_without_ipex(infer_conf: InferenceConfig) -> bool:
    return (not infer_conf.ipex.enabled) and infer_conf.device == DEVICE_CPU
