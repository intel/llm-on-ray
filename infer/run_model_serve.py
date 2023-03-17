import ray
from ray import serve
from starlette.requests import Request
import torch
import time

@serve.deployment
class PredictDeployment:
    def __init__(self, model_id, amp_enabled, amp_dtype, max_new_tokens):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import intel_extension_for_pytorch as ipex
        import os
        self.amp_enabled = amp_enabled
        self.amp_dtype = amp_dtype
        self.max_new_tokens = max_new_tokens
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = model.eval()
        # to channels last
        model = model.to(memory_format=torch.channels_last)
        # to ipex
        self.model = ipex.optimize(model, dtype=amp_dtype, inplace=True)

    def generate(self, text: str) -> str:
        with torch.cpu.amp.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids
            gen_tokens = self.model.generate(
                input_ids,
                # max_new_tokens=self.max_new_tokens,
                max_length=self.max_new_tokens,
                do_sample=True,
                temperature=0.9
            )
            return self.tokenizer.batch_decode(gen_tokens)[0]

    async def __call__(self, http_request: Request) -> str:
        json_request: str = await http_request.json()
        prompts = []
        for prompt in json_request:
            text = prompt["text"]
            if isinstance(text, list):
                prompts.extend(text)
            else:
                prompts.append(text)
        return self.generate(prompts)

if __name__ == "__main__":
    runtime_env = {
        "env_vars": {
            "KMP_BLOCKTIME": "1",
            "KMP_SETTINGS": "1",
            "KMP_AFFINITY": "granularity=fine,compact,1,0",
            "OMP_NUM_THREADS": "56",
            "LD_PRELOAD": "/root/miniconda3/envs/linzhi/lib/libiomp5.so:/root/miniconda3/envs/linzhi/lib/libjemalloc.so",
            "MALLOC_CONF": "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
        }
    }
    ray.init(address="auto", runtime_env=runtime_env)
    model_id = "EleutherAI/gpt-j-6B"

    # args
    import argparse
    parser = argparse.ArgumentParser('GPT-J generation script', add_help=False)
    parser.add_argument('--precision', default='bf16', type=str, help="fp32 or bf16")
    parser.add_argument('--model', default='EleutherAI/gpt-j-6B', type=str, help="model name or path")
    parser.add_argument('--max-new-tokens', default=32, type=int, help="output max new tokens")
    args = parser.parse_args()

    model_id = args.model
    amp_enabled = True if args.precision != "fp32" else False
    amp_dtype = torch.bfloat16 if args.precision != "fp32" else torch.float32

    deployment = PredictDeployment.bind(model_id, amp_enabled, amp_dtype, args.max_new_tokens)
    handle = serve.run(deployment, _blocking=True)
    input("Service is deployed successfully")
    # prompt = "Once upon a time,"
#             " She wanted to go to places and meet new people, and have fun."
    # total_time = 0.0
    # num_iter = 10
    # num_warmup = 3
    # for i in range(num_iter):
    #    tic = time.time()
    #    result = ray.get(handle.generate.remote(prompt))
    #    toc = time.time()
    #    print(result, flush=True)
    #    if i >= num_warmup:
    #        total_time += (toc - tic)

    # print("Inference latency: %.3f ms." % (total_time / (num_iter - num_warmup) * 1000))

    # print("Model is deloyed successfully")
    # input()