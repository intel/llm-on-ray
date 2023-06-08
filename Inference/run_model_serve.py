import ray
from ray import serve
from starlette.requests import Request
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import intel_extension_for_pytorch as ipex
from config import all_models
from transformers import StoppingCriteria, StoppingCriteriaList


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


@serve.deployment()
class PredictDeployment:
    def __init__(self, model_id, tokenizer_name_or_path, amp_enabled, amp_dtype, stop_words):
        self.amp_enabled = amp_enabled
        self.amp_dtype = amp_dtype
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=amp_dtype,
            low_cpu_mem_usage=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        model = model.eval()
        # to channels last
        model = model.to(memory_format=torch.channels_last)
        # to ipex
        self.model = ipex.optimize(model, dtype=amp_dtype, inplace=True)
        # self.model = model

        stop_words_ids = [self.tokenizer(stop_word, return_tensors='pt').input_ids.squeeze() for stop_word in stop_words]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def generate(self, text: str, **config) -> str:
        # with torch.cpu.amp.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids

        gen_tokens = self.model.generate(
            input_ids,
            pad_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=self.stopping_criteria,
            **config
        )
        output = self.tokenizer.batch_decode(gen_tokens)[0]

        return output

    async def __call__(self, http_request: Request) -> str:
        json_request: str = await http_request.json()
        prompts = []
        for prompt in json_request:
            text = prompt["text"]
            config = prompt["config"]  if "config" in prompt else {}
            if isinstance(text, list):
                prompts.extend(text)
            else:
                prompts.append(text)
        outputs = self.generate(prompts, **config)
        return outputs

if __name__ == "__main__":
    runtime_env = {
        "env_vars": {
            "KMP_BLOCKTIME": "1",
            "KMP_SETTINGS": "1",
            "KMP_AFFINITY": "granularity=fine,compact,1,0",
            "OMP_NUM_THREADS": "56",
            "MALLOC_CONF": "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000",
        }
    }
    ray.init(address="auto", runtime_env=runtime_env)

    # args
    import argparse
    parser = argparse.ArgumentParser("Model Serve Script", add_help=False)
    parser.add_argument("--precision", default="bf16", type=str, help="fp32 or bf16")
    parser.add_argument("--model", default=None, type=str, help="model name or path")
    parser.add_argument("--tokenizer", default=None, type=str, help="tokenizer name or path")

    args = parser.parse_args()

    amp_enabled = True if args.precision != "fp32" else False
    amp_dtype = torch.bfloat16 if args.precision != "fp32" else torch.float32

    stop_words = ["### Instruction", "# Instruction", "### Question", "##", " ="]

    if args.model is None:
        model_list = all_models
    else:
        model_list = {
            "custom_model": {
                "model_id_or_path": args.model,
                "tokenizer_name_or_path": args.tokenizer,
                "port": "8000",
                "name": "custom-model",
                "route_prefix": "/custom-model"
            }
        }

    for model_id, model_config in model_list.items():
        print("deploy model: ", model_id)
        deployment = PredictDeployment.bind(model_config["model_id_or_path"], model_config["tokenizer_name_or_path"], amp_enabled, amp_dtype, stop_words=stop_words)
        handle = serve.run(deployment, _blocking=True, port=model_config["port"], name=model_config["name"], route_prefix=model_config["route_prefix"])
    input("Service is deployed successfully")
