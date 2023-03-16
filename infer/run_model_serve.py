import ray
from ray import serve
from starlette.requests import Request
import torch

@serve.deployment
class PredictDeployment:
    def __init__(self, num_cores, model_id, amp_enabled, amp_dtype, max_new_tokens):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import intel_extension_for_pytorch as ipex
        import os
        os.environ["OMP_NUM_THREADS"] = str(num_cores)
        self.amp_enabled = amp_enabled
        self.amp_dtype = amp_dtype
        self.max_new_tokens = max_new_tokens
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            cache_dir="/mnt/DP_disk3/GPTJ_Model"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = model.eval()
        # to channels last
        # model = model.to(memory_format=torch.channels_last)
        # to ipex
        # self.model = ipex.optimize(model, dtype=amp_dtype, inplace=True)

    def generate(self, text: str) -> str:
        with torch.cpu.amp.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):  
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids
            gen_tokens = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.9,
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
    ray.init(address="local")
    model_id = "EleutherAI/gpt-j-6B"

    # args
    import argparse
    parser = argparse.ArgumentParser('GPT-J generation script', add_help=False)
    parser.add_argument('--num-cores', default=32, type=int, help="number of cores to use")
    parser.add_argument('--precision', default='bf16', type=str, help="fp32 or bf16")
    parser.add_argument('--model', default='EleutherAI/gpt-j-6B', type=str, help="model name or path")
    parser.add_argument('--max-new-tokens', default=32, type=int, help="output max new tokens")

    args = parser.parse_args()

    model_id = args.model
    amp_enabled = True if args.precision != "fp32" else False
    amp_dtype = torch.bfloat16 if args.precision != "fp32" else torch.float32

    deployment = PredictDeployment.bind(args.num_cores, model_id, amp_enabled, amp_dtype, args.max_new_tokens)
    serve.run(deployment, _blocking=True)

    print("Model is deployed successfully")
    # block
    input()
