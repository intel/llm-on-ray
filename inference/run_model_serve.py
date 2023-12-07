import os
import asyncio
import functools
import ray
from ray import serve
from starlette.requests import Request
from queue import Empty
import torch
from transformers import AutoTokenizer, TextIteratorStreamer
from transformers import StoppingCriteria, StoppingCriteriaList
from inference_config import ModelDescription, InferenceConfig, all_models
import sys

from typing import Generator, Union, Optional, List
from starlette.responses import StreamingResponse

from pydantic_yaml import parse_yaml_raw_as

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

@serve.deployment
class PredictDeployment:
    def __init__(self, inferenceConfig: InferenceConfig):
        self.device = torch.device(inferenceConfig.device)
        self.tokenizer = AutoTokenizer.from_pretrained(inferenceConfig.model_description.tokenizer_name_or_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.process_tool = None
        chat_processor_name = inferenceConfig.model_description.chat_processor
        prompt = inferenceConfig.model_description.prompt
        if chat_processor_name:
            module = __import__("chat_process")
            chat_processor = getattr(module, chat_processor_name, None)
            if chat_processor is None:
                raise ValueError(inferenceConfig.name + " deployment failed. chat_processor(" + chat_processor_name + ") does not exist.")
            self.process_tool = chat_processor(**prompt.dict())
        stop_words = prompt.stop_words
        stop_words_ids = [self.tokenizer(stop_word, return_tensors='pt').input_ids.squeeze() for stop_word in stop_words]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        self.use_deepspeed = inferenceConfig.deepspeed
        self.amp_dtype = torch.bfloat16 if inferenceConfig.precision != "fp32" else torch.float32
        if self.use_deepspeed:
            from deepspeed_predictor import DeepSpeedPredictor
            self.streamer = self.create_streamer()
            # now deepspeed predictor don't have the model
            # this should be solved in the next pr
            # where it is also a worker
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.predictor = DeepSpeedPredictor(inferenceConfig, self.amp_dtype, self.tokenizer.pad_token_id, self.stopping_criteria)
        else:
            from transformer_predictor import TransformerPredictor
            self.predictor = TransformerPredictor(inferenceConfig, self.amp_dtype, self.stopping_criteria)
            self.predictor.configure_tokenizer(inferenceConfig.model_description.model_id_or_path, self.tokenizer)
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
        if self.device.type == "hpu":
            input_tokens_no_pad = self.tokenizer(text, return_tensors="pt")
            input_token_len = input_tokens_no_pad.input_ids.shape[-1]
            input_tokens = self.tokenizer.batch_encode_plus(
                text,
                return_tensors="pt",
                padding="max_length",
                max_length=max_input_len(input_token_len),
            )
        else:
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
                await asyncio.sleep(0.001)

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

_ray_env_key = "env_vars"
# OMP_NUM_THREADS will be set by num_cpus, so not set in env
_predictor_runtime_env_ipex = {
    "KMP_BLOCKTIME": "1",
    "KMP_SETTINGS": "1",
    "KMP_AFFINITY": "granularity=fine,compact,1,0",
    "MALLOC_CONF": "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
}

# make it unittest friendly
def main(argv=None):
    # args
    import argparse
    parser = argparse.ArgumentParser("Model Serve Script", add_help=False)
    parser.add_argument("--config_file", type=str, help="inference configuration file in YAML. If specified, all other arguments are ignored")
    parser.add_argument("--model", default=None, type=str, help="model name or path")
    parser.add_argument("--tokenizer", default=None, type=str, help="tokenizer name or path")
    parser.add_argument("--port", default=8000, type=int, help="the port of deployment address")
    parser.add_argument("--route_prefix", default="custom_model", type=str, help="the route prefix for HTTP requests.")
    parser.add_argument("--cpus_per_worker", default="24", type=int, help="cpus per worker")
    parser.add_argument("--gpus_per_worker", default=0, type=float, help="gpus per worker, used when --device is cuda")
    parser.add_argument("--hpus_per_worker", default=0, type=float, help="hpus per worker, used when --device is hpu")
    parser.add_argument("--deepspeed", action='store_true', help="enable deepspeed inference")
    parser.add_argument("--workers_per_group", default="2", type=int, help="workers per group, used with --deepspeed")
    parser.add_argument("--ipex", action='store_true', help="enable ipex optimization")
    parser.add_argument("--device", default="cpu", type=str, help="cpu, xpu, hpu or cuda")
    parser.add_argument("--precision", default="bf16", type=str, help="fp32 or bf16")
    parser.add_argument("--serve_local_only", action="store_true", help="Only support local access to url")

    args = parser.parse_args(argv)

    # serve all pre-defined models, or model from MODEL_TO_SERVE env, if no model argument specified
    if args.model is None and args.config_file is None:
        model_list = all_models
    else:
        # config_file has precedence over others
        if args.config_file:
            print("reading from config file, " + args.config_file)
            with open(args.config_file, "r") as f:
                inferenceConfig = parse_yaml_raw_as(InferenceConfig, f)
        else: # args.model should be set
            print("reading from command line, " + args.model)
            model_desc = ModelDescription()
            model_desc.model_id_or_path = args.model
            model_desc.tokenizer_name_or_path = args.tokenizer if args.tokenizer is not None else args.model
            inferenceConfig = InferenceConfig(model_description=model_desc)
            inferenceConfig.host = "127.0.0.1" if args.serve_local_only else "0.0.0.0"
            inferenceConfig.port = args.port
            rp = args.route_prefix if args.route_prefix else "custom_model"
            inferenceConfig.route_prefix = "/{}".format(rp)
            inferenceConfig.name = rp
            inferenceConfig.ipex = args.ipex
        model_list = {}
        model_list[inferenceConfig.name] = inferenceConfig

    ray.init(address="auto")

    deployments = []
    for model_id, inferCfg in model_list.items():
        print("deploy model: ", model_id)
        runtime_env = {_ray_env_key: {}}
        if inferCfg.ipex:
            runtime_env[_ray_env_key].update(_predictor_runtime_env_ipex)
        if inferCfg.deepspeed:
            runtime_env[_ray_env_key]["DS_ACCELERATOR"] = inferCfg.device
        # now PredictDeployment itself is a worker, we should require resources for it
        ray_actor_options = {"runtime_env": runtime_env}
        if inferCfg.device == "cpu":
            ray_actor_options["num_cpus"] = inferCfg.cpus_per_worker
        elif inferCfg.device == "cuda":
            ray_actor_options["num_gpus"] = inferCfg.gpus_per_worker
        elif inferCfg.device == "hpu":
            ray_actor_options["resources"] = {"HPU": inferCfg.hpus_per_worker}
        else:
            # TODO add xpu
            pass
        deployment = PredictDeployment.options(ray_actor_options=ray_actor_options).bind(inferCfg)
        handle = serve.run(deployment, _blocking=True, host=inferCfg.host, port=inferCfg.port, name=inferCfg.name, route_prefix=inferCfg.route_prefix)
        deployment_name = inferCfg.name
        if inferCfg.host == "0.0.0.0":
            all_nodes = ray.nodes()
            for node in all_nodes:
                if "node:__internal_head__" in node["Resources"]:
                    host_ip = node["NodeManagerAddress"]
                    break
        else:
            host_ip = inferCfg.host
        url = f"http://{host_ip}:{inferCfg.port}{inferCfg.route_prefix}"
        print(f"Deployment '{deployment_name}' is ready at `{url}`.")
        deployments.append(handle)

    msg = "Service is deployed successfully"
    env_name = "KEEP_SERVE_TERMINAL"
    if env_name not in os.environ or os.environ[env_name].lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup']:
        input(msg)
    else:
        print(msg)
    return deployments

if __name__ == "__main__":
    main(sys.argv[1:])