import os
import asyncio
import functools
import ray
from ray import serve
from starlette.requests import Request
from queue import Empty
import torch
from transformers import TextIteratorStreamer
from inference_config import InferenceConfig
from typing import Union
from starlette.responses import StreamingResponse
from api_openai_backend.openai_protocol import ModelResponse


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
        
    async def stream_response(self, content: str, **config):
        prompt = content.prompt
        prompts = []
        if isinstance(prompt, list):
            if self.process_tool is not None:
                prompt = self.process_tool.get_prompt(prompt)
                prompts.append(prompt)
            else:
                prompts.extend(prompt)
        else:
            prompts.append(prompt)

        if self.use_deepspeed:
            self.predictor.streaming_generate(prompts, self.streamer, **config)
            response_handle = self.consume_streamer()
        else:
            streamer = self.predictor.get_streamer()
            self.loop.run_in_executor(None, functools.partial(self.predictor.streaming_generate, prompts, streamer, **config))
            response_handle = self.consume_streamer_async(streamer)
        async for output in response_handle:
            model_response = ModelResponse(
                generated_text=output,
                num_input_tokens=len(prompts[0]),
                num_input_tokens_batch=len(prompts[0]),
                num_generated_tokens=1,
                preprocessing_time=0,
            )
            yield model_response

def serve_run(model_list, deployments):
    for model_id, infer_conf in model_list.items():
        print("deploy model: ", model_id)
        deployment = deployments[model_id]
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

    msg = "Service is deployed successfully"
    env_name = "KEEP_SERVE_TERMINAL"
    if env_name not in os.environ or os.environ[env_name].lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup']:
        input(msg)
    else:
        print(msg)
    return deployments
