import os
import ray
from ray import serve
from starlette.requests import Request
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
import intel_extension_for_pytorch as ipex
from config import all_models

from typing import Generator, Union
from starlette.responses import StreamingResponse
from threading import Thread

from transformer_predictor import TransformerPredictor
from deepspeed_predictor import DeepSpeedPredictor
from config import all_models

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

@ray.remote(scheduling_strategy="SPREAD")
class Dialogue:
    def __init__(self, model_id, model_load_config, device_name, amp_enabled, amp_dtype, pad_token_id, stopping_criteria, streamer, ipex_enabled,
                 deepspeed_enabled, cpus_per_worker, workers_per_group):
        if deepspeed_enabled:
            self.predictor = DeepSpeedPredictor(model_id, model_load_config, device_name, amp_enabled, amp_dtype, pad_token_id, stopping_criteria, streamer,
                                                ipex_enabled, cpus_per_worker, workers_per_group)
        else:
            self.predictor = TransformerPredictor(model_id, model_load_config, device_name, amp_enabled, amp_dtype, pad_token_id, stopping_criteria, streamer,
                                                  ipex_enabled)

    def streaming_generate(self, input_ids, **config):
        return self.predictor.streaming_generate(input_ids, **config)

    def generate(self, input_ids, **config):
        return self.predictor.generate(input_ids, **config)

    def ping(self):
        pass

@serve.deployment()
class PredictDeployment:
    def __init__(self, model_id, tokenizer_name_or_path, model_load_config, device_name, amp_enabled, amp_dtype, stop_words,
                 ipex_enabled=False, deepspeed_enabled=False, cpus_per_worker=1, workers_per_group=2):
        self.device_name = device_name
        self.device = torch.device(device_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.streamer = self.create_streamer()
        stop_words_ids = [self.tokenizer(stop_word, return_tensors='pt').input_ids.squeeze() for stop_word in stop_words]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        self.dialogue = Dialogue.remote(model_id, model_load_config, device_name, amp_enabled, amp_dtype,
                                        self.tokenizer.eos_token_id,
                                        self.stopping_criteria, self.streamer, ipex_enabled,
                                        deepspeed_enabled, cpus_per_worker, workers_per_group)
        ray.get(self.dialogue.ping.remote())

    def create_streamer(self):
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

    def generate(self, text: str, streaming_response: bool, **config) -> Generator[str, None, None]:
        # with torch.cpu.amp.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

        if not streaming_response:
            response = self.dialogue.generate.remote(input_ids, **config)
            gen_tokens = ray.get(response)
            output = self.tokenizer.batch_decode(
                gen_tokens,
                skip_special_tokens=True
            )
            yield output[0]

        self.dialogue.streaming_generate.remote(input_ids, **config)

        for new_text in self.streamer:
            yield new_text

    async def __call__(self, http_request: Request) -> Union[StreamingResponse, str]:
        json_request: str = await http_request.json()
        prompts = []
        for prompt in json_request:
            text = prompt["text"]
            config = prompt["config"]  if "config" in prompt else {}
            streaming_response = prompt["stream"]
            if isinstance(text, list):
                prompts.extend(text)
            else:
                prompts.append(text)
        outputs = self.generate(prompts, streaming_response, **config)
        if not streaming_response:
            return next(outputs)
        return StreamingResponse(outputs, status_code=200, media_type="text/plain")

if __name__ == "__main__":

    # args
    import argparse
    parser = argparse.ArgumentParser("Model Serve Script", add_help=False)
    parser.add_argument("--precision", default="bf16", type=str, help="fp32 or bf16")
    parser.add_argument("--model", default=None, type=str, help="model name or path")
    parser.add_argument("--tokenizer", default=None, type=str, help="tokenizer name or path")
    parser.add_argument("--cpus_per_worker", default="24", type=int, help="cpus per worker")
    parser.add_argument("--deepspeed", action='store_true', help="enable deepspeed inference")
    parser.add_argument("--workers_per_group", default="2", type=int, help="workers per group, used with --deepspeed")
    parser.add_argument("--ipex", action='store_true', help="enable ipex optimization")
    parser.add_argument("--device", default="cpu", type=str, help="cpu or gpu")

    args = parser.parse_args()

    runtime_env = {
        "env_vars": {
            "KMP_BLOCKTIME": "1",
            "KMP_SETTINGS": "1",
            "KMP_AFFINITY": "granularity=fine,compact,1,0",
            "OMP_NUM_THREADS": str(args.cpus_per_worker),
            "MALLOC_CONF": "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000",
        }
    }
    ray.init(address="auto", runtime_env=runtime_env)

    amp_enabled = True if args.precision != "fp32" else False
    amp_dtype = torch.bfloat16 if args.precision != "fp32" else torch.float32

    if args.model is None:
        model_list = all_models
    else:
        model_list = {
            "custom_model": {
                "model_id_or_path": args.model,
                "tokenizer_name_or_path": args.tokenizer,
                "port": "8000",
                "name": "custom-model",
                "route_prefix": "/custom-model",
                "chat_model": "ChatModelGptJ",
                "prompt": {
                    "intro": "",
                    "human_id": "",
                    "bot_id": "",
                    "stop_words": []
                }
            }
        }

    for model_id, model_config in model_list.items():
        print("deploy model: ", model_id)
        model_load_config = model_config.get("config", {})
        deployment = PredictDeployment.bind(model_config["model_id_or_path"], model_config["tokenizer_name_or_path"], model_load_config,
                                            args.device, amp_enabled, amp_dtype,
                                            stop_words=model_config["prompt"]["stop_words"],
                                            ipex_enabled=args.ipex,
                                            deepspeed_enabled=args.deepspeed, cpus_per_worker=args.cpus_per_worker, workers_per_group=args.workers_per_group)
        handle = serve.run(deployment, _blocking=True, port=model_config["port"], name=model_config["name"], route_prefix=model_config["route_prefix"])

    msg = "Service is deployed successfully"
    env_name = "KEEP_SERVE_TERMINAL"
    if env_name not in os.environ or os.environ[env_name].lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup']:
        input(msg)
    else:
        print(msg)

