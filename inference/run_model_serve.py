import os
import asyncio
import functools
import ray
from ray import serve
from starlette.requests import Request
from queue import Empty
import torch
from transformers import TextIteratorStreamer
from inference_config import ModelDescription, InferenceConfig, all_models
import sys
from utils import get_deployment_actor_options
from typing import Generator, Union, Optional, List
from starlette.responses import StreamingResponse

from pydantic_yaml import parse_yaml_raw_as

@serve.deployment
class PredictDeployment:
    def __init__(self, infer_conf: InferenceConfig):
        self.device = torch.device(infer_conf.device)
        self.process_tool = None
        chat_processor_name = infer_conf.model_description.chat_processor
        prompt = infer_conf.model_description.prompt
        if chat_processor_name:
            module = __import__("chat_process")
            chat_processor = getattr(module, chat_processor_name, None)
            if chat_processor is None:
                raise ValueError(infer_conf.name + " deployment failed. chat_processor(" + chat_processor_name + ") does not exist.")
            self.process_tool = chat_processor(**prompt.dict())
        
        self.use_deepspeed = infer_conf.deepspeed
        if self.use_deepspeed:
            from deepspeed_predictor import DeepSpeedPredictor
            self.predictor = DeepSpeedPredictor(infer_conf)
            self.streamer = self.predictor.get_streamer()
        else:
            from transformer_predictor import TransformerPredictor
            self.predictor = TransformerPredictor(infer_conf)
        self.loop = asyncio.get_running_loop()
    
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
        text = json_request["text"]
        config = json_request["config"]  if "config" in json_request else {}
        streaming_response = json_request["stream"]
        if isinstance(text, list):
            if self.process_tool is not None:
                prompt = self.process_tool.get_prompt(text)
                prompts.append(prompt)
            else:
                prompts.extend(text)
        else:
            prompts.append(text)
        if not streaming_response:
            return self.predictor.generate(prompts, **config)
        if self.use_deepspeed:
            self.predictor.streaming_generate(prompts, self.streamer, **config)
            return StreamingResponse(self.consume_streamer(), status_code=200, media_type="text/plain")
        else:
            streamer = self.predictor.get_streamer()
            self.loop.run_in_executor(None, functools.partial(self.predictor.streaming_generate, prompts, streamer, **config))
            return StreamingResponse(self.consume_streamer_async(streamer), status_code=200, media_type="text/plain")

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
                infer_conf = parse_yaml_raw_as(InferenceConfig, f)
        else: # args.model should be set
            print("reading from command line, " + args.model)
            model_desc = ModelDescription()
            model_desc.model_id_or_path = args.model
            model_desc.tokenizer_name_or_path = args.tokenizer if args.tokenizer is not None else args.model
            infer_conf = InferenceConfig(model_description=model_desc)
            infer_conf.host = "127.0.0.1" if args.serve_local_only else "0.0.0.0"
            infer_conf.port = args.port
            rp = args.route_prefix if args.route_prefix else "custom_model"
            infer_conf.route_prefix = "/{}".format(rp)
            infer_conf.name = rp
            infer_conf.ipex.enabled = args.ipex
        model_list = {}
        model_list[infer_conf.name] = infer_conf

    ray.init(address="auto")

    deployments = []
    for model_id, infer_conf in model_list.items():
        print("deploy model: ", model_id)
        ray_actor_options = get_deployment_actor_options(infer_conf)
        deployment = PredictDeployment.options(ray_actor_options=ray_actor_options).bind(infer_conf)
        handle = serve.run(deployment, _blocking=True, host=infer_conf.host, port=infer_conf.port, name=infer_conf.name, route_prefix=infer_conf.route_prefix)
        deployment_name = infer_conf.name
        if infer_conf.host == "0.0.0.0":
            all_nodes = ray.nodes()
            for node in all_nodes:
                if "node:__internal_head__" in node["Resources"]:
                    host_ip = node["NodeManagerAddress"]
                    break
        else:
            host_ip = infer_conf.host
        url = f"http://{host_ip}:{infer_conf.port}{infer_conf.route_prefix}"
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