import os
import asyncio
import functools
import ray
from ray import serve
from starlette.requests import Request
from queue import Empty
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers import StoppingCriteria, StoppingCriteriaList
import intel_extension_for_pytorch as ipex
from config import all_models
import sys

from typing import Generator, Union, Optional, List
from starlette.responses import StreamingResponse

from transformer_predictor import TransformerPredictor
from deepspeed_predictor import DeepSpeedPredictor

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


@serve.deployment
class PredictDeployment:
    def __init__(self, model_id, tokenizer_name_or_path, model_load_config, device_name, amp_enabled, amp_dtype, chat_processor_name, prompt,
                 ipex_enabled=False, deepspeed_enabled=False, cpus_per_worker=1, gpus_per_worker=0, workers_per_group=2, deltatuner_model_id=None):
        self.device_name = device_name
        self.device = torch.device(device_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.process_tool = None
        if chat_processor_name is not None:
            module = __import__("chat_process")
            chat_processor = getattr(module, chat_processor_name, None)
            if chat_processor is None:
                raise ValueError(model_id + " deployment failed. chat_processor(" + chat_processor_name + ") does not exist.")
            self.process_tool = chat_processor(**prompt)
        stop_words = prompt["stop_words"]
        stop_words_ids = [self.tokenizer(stop_word, return_tensors='pt').input_ids.squeeze() for stop_word in stop_words]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        self.use_deepspeed = deepspeed_enabled
        if self.use_deepspeed:
            self.streamer = self.create_streamer()
            self.predictor = DeepSpeedPredictor(model_id, model_load_config, device_name, amp_enabled, amp_dtype, self.tokenizer.pad_token_id, self.stopping_criteria,
                                                ipex_enabled, cpus_per_worker, gpus_per_worker, workers_per_group, deltatuner_model_id)
        else:
            self.predictor = TransformerPredictor(model_id, model_load_config, device_name, amp_enabled, amp_dtype, self.tokenizer.pad_token_id, self.stopping_criteria,
                                                  ipex_enabled, deltatuner_model_id)
        self.loop = asyncio.get_running_loop()

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

    def tokenize_inputs(self, text: List[str]):
        input_tokens = self.tokenizer.batch_encode_plus(
            text, return_tensors="pt", padding=True
        )
        input_token_len = input_tokens.input_ids.shape[-1]
        inputs = {k: v.to(device=self.device) \
                  for k,v in input_tokens.items() \
                  if torch.is_tensor(v)}
        return inputs, input_token_len

    def predict(self, text: List[str], **config) -> str:
        inputs, _ = self.tokenize_inputs(text)
        gen_tokens = self.predictor.generate(inputs, **config)
        return self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]

    def predict_stream(self, text: List[str], streamer: TextIteratorStreamer, **config) -> Generator[str, None, None]:
        # with torch.cpu.amp.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
        inputs, _ = self.tokenize_inputs(text)
        self.predictor.streaming_generate(inputs, streamer, **config)
    
    def consume_streamer(self):
        for text in self.streamer:
            yield text

    async def consume_streamer_async(self, streamer: TextIteratorStreamer):
        while True:
            try:
                for token in streamer:
                    yield token
                break
            except Empty:
                # The streamer raises an Empty exception if the next token
                # hasn't been generated yet. `await` here to yield control
                # back to the event loop so other coroutines can run.
                await asyncio.sleep(0.1)

    async def __call__(self, http_request: Request) -> Union[StreamingResponse, str]:
        json_request: str = await http_request.json()
        prompts = []
        for prompt in json_request:
            text = prompt["text"]
            config = prompt["config"]  if "config" in prompt else {}
            streaming_response = prompt["stream"]
            if isinstance(text, list):
                if self.process_tool is not None:
                    prompt = self.process_tool.get_prompt(text)
                    prompts.append(prompt)
                else:
                    prompts.extend(text)            
            else:
                prompts.append(text)
        if not streaming_response:
            return self.predict(prompts, **config)
        if self.use_deepspeed:
            self.predict_stream(prompts, self.streamer, **config)
            return StreamingResponse(self.consume_streamer(), status_code=200, media_type="text/plain")
        else:
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=0, skip_special_tokens=True)
            self.loop.run_in_executor(None, functools.partial(self.predict_stream, prompts, streamer, **config))
            return StreamingResponse(self.consume_streamer_async(streamer), status_code=200, media_type="text/plain")

if __name__ == "__main__":

    # args
    import argparse
    parser = argparse.ArgumentParser("Model Serve Script", add_help=False)
    parser.add_argument("--precision", default="bf16", type=str, help="fp32 or bf16")
    parser.add_argument("--model", default=None, type=str, help="model name or path")
    parser.add_argument("--deltatuner_model", default=None, type=str, help="deltatuner model name or path")
    parser.add_argument("--tokenizer", default=None, type=str, help="tokenizer name or path")
    parser.add_argument("--port", default="8000", type=str, help="the port of deployment address")
    parser.add_argument("--route_prefix", default="custom_model", type=str, help="the route prefix for HTTP requests.")
    parser.add_argument("--chat_processor", default=None, type=str, help="how to process the prompt and output, \
                        the value corresponds to class names in chat_process.py, only for chat version.")
    parser.add_argument("--prompt", default=None, type=str, help="the specific format of prompt.")
    parser.add_argument("--trust_remote_code", action='store_true', 
                        help="Whether or not to allow for custom code defined on the Hub in their own modeling, configuration,\
                            tokenization or even pipeline files.")
    parser.add_argument("--use_auth_token", default=None, type=Optional[Union[bool, str]],
                        help="The token to use as HTTP bearer authorization for remote files.")
    parser.add_argument("--cpus_per_worker", default="24", type=int, help="cpus per worker")
    parser.add_argument("--gpus_per_worker", default=0, type=float, help="gpus per worker, used when --device is cuda")
    parser.add_argument("--deepspeed", action='store_true', help="enable deepspeed inference")
    parser.add_argument("--workers_per_group", default="2", type=int, help="workers per group, used with --deepspeed")
    parser.add_argument("--ipex", action='store_true', help="enable ipex optimization")
    parser.add_argument("--device", default="cpu", type=str, help="cpu, xpu or cuda")

    args = parser.parse_args()

    runtime_env = {
        "env_vars": {
            "KMP_BLOCKTIME": "1",
            "KMP_SETTINGS": "1",
            "KMP_AFFINITY": "granularity=fine,compact,1,0",
            "OMP_NUM_THREADS": str(args.cpus_per_worker),
            "DS_ACCELERATOR": args.device,
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
                "tokenizer_name_or_path": args.tokenizer if args.tokenizer is not None else args.model,
                "port": args.port,
                "name": args.route_prefix,
                "route_prefix": "/{}".format(args.route_prefix),
                "chat_processor": args.chat_processor,
                "prompt": {
                    "intro": "",
                    "human_id": "",
                    "bot_id": "",
                    "stop_words": []
                } if args.prompt is None else eval(args.prompt),
                "config": {
                    "trust_remote_code": args.trust_remote_code,
                    "use_auth_token": args.use_auth_token
                }
            }
        }

    for model_id, model_config in model_list.items():
        print("deploy model: ", model_id)
        model_config["deltatuner_model_id_or_path"] = args.deltatuner_model
        model_load_config = model_config.get("config", {})
        deployment = PredictDeployment.options(ray_actor_options={"num_gpus": min(args.gpus_per_worker, 1)}).bind(
                                            model_config["model_id_or_path"], model_config["tokenizer_name_or_path"], model_load_config,
                                            args.device, amp_enabled, amp_dtype,
                                            chat_processor_name = model_config["chat_processor"], prompt=model_config["prompt"],
                                            ipex_enabled=args.ipex,
                                            deepspeed_enabled=args.deepspeed, cpus_per_worker=args.cpus_per_worker,
                                            gpus_per_worker=args.gpus_per_worker, workers_per_group=args.workers_per_group,
                                            deltatuner_model_id=model_config["deltatuner_model_id_or_path"])
        handle = serve.run(deployment, _blocking=True, port=model_config["port"], name=model_config["name"], route_prefix=model_config["route_prefix"])

    msg = "Service is deployed successfully"
    env_name = "KEEP_SERVE_TERMINAL"
    if env_name not in os.environ or os.environ[env_name].lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup']:
        input(msg)
    else:
        print(msg)

