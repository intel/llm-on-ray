#
# Copyright 2023 The LLM-on-Ray Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import sys
import asyncio
import functools
from ray import serve
from starlette.requests import Request
from queue import Empty
import torch
from transformers import TextIteratorStreamer
from inference.inference_config import InferenceConfig
from typing import Union, Dict, Any
from starlette.responses import StreamingResponse, JSONResponse
from inference.api_openai_backend.openai_protocol import ModelResponse
from inference.utils import get_input_format


@serve.deployment
class PredictorDeployment:
    def __init__(self, infer_conf: InferenceConfig):
        self.device = torch.device(infer_conf.device)
        self.process_tool = None
        chat_processor_name = infer_conf.model_description.chat_processor
        prompt = infer_conf.model_description.prompt
        if chat_processor_name:
            try:
                module = __import__("chat_process")
            except Exception:
                sys.path.append(os.path.dirname(__file__))
                module = __import__("chat_process")
            chat_processor = getattr(module, chat_processor_name, None)
            if chat_processor is None:
                raise ValueError(
                    infer_conf.name
                    + " deployment failed. chat_processor("
                    + chat_processor_name
                    + ") does not exist."
                )
            self.process_tool = chat_processor(**prompt.dict())

        self.use_deepspeed = infer_conf.deepspeed
        self.use_vllm = infer_conf.vllm.enabled

        if self.use_deepspeed:
            from deepspeed_predictor import DeepSpeedPredictor

            self.predictor = DeepSpeedPredictor(infer_conf)
            self.streamer = self.predictor.get_streamer()
        elif self.use_vllm:
            from vllm_predictor import VllmPredictor

            self.predictor = VllmPredictor(infer_conf)
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

    async def __call__(self, http_request: Request) -> Union[StreamingResponse, JSONResponse, str]:
        json_request: Dict[str, Any] = await http_request.json()
        prompts = []
        text = json_request["text"]
        config = json_request["config"] if "config" in json_request else {}
        streaming_response = json_request["stream"]
        if isinstance(text, list):
            is_chat, is_prompts = get_input_format(text)
            if is_chat:
                if self.process_tool is not None:
                    prompt = self.process_tool.get_prompt(text)
                    prompts.append(prompt)
                else:
                    prompts.extend(text)
            elif is_prompts:
                if streaming_response:
                    return JSONResponse(
                        status_code=400,
                        content="multiple prompts are not supported when streaming response is enabled.",
                    )
                prompts.extend(text)
            else:
                return JSONResponse(status_code=400, content="invalid prompt format.")
        else:
            prompts.append(text)

        if not streaming_response:
            if self.use_vllm:
                return await self.predictor.generate_async(prompts, **config)
            else:
                return self.predictor.generate(prompts, **config)

        if self.use_deepspeed:
            self.predictor.streaming_generate(prompts, self.streamer, **config)
            return StreamingResponse(
                self.consume_streamer(), status_code=200, media_type="text/plain"
            )
        elif self.use_vllm:
            # TODO: streaming only support single prompt
            # It's a wordaround for current situation, need another PR to address this
            if isinstance(prompts, list):
                prompt = prompts[0]
            results_generator = await self.predictor.streaming_generate_async(prompt, **config)
            return StreamingResponse(
                self.predictor.stream_results(results_generator),
                status_code=200,
                media_type="text/plain",
            )
        else:
            streamer = self.predictor.get_streamer()
            self.loop.run_in_executor(
                None,
                functools.partial(self.predictor.streaming_generate, prompts, streamer, **config),
            )
            return StreamingResponse(
                self.consume_streamer_async(streamer), status_code=200, media_type="text/plain"
            )

    async def stream_response(self, prompt, config):
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
            self.loop.run_in_executor(
                None,
                functools.partial(self.predictor.streaming_generate, prompts, streamer, **config),
            )
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
