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
from queue import Empty
import torch
from transformers import TextIteratorStreamer
from typing import AsyncGenerator, List, Tuple, Union, Dict, Any
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse
from fastapi import HTTPException
from llm_on_ray.inference.inference_config import InferenceConfig
from llm_on_ray.inference.api_openai_backend.openai_protocol import (
    ChatMessage,
    ErrorResponse,
    ModelResponse,
)
from llm_on_ray.inference.utils import get_prompt_format, PromptFormat
from llm_on_ray.inference.api_openai_backend.tools import OpenAIToolsPrompter, ChatPromptCapture
from llm_on_ray.inference.logger import get_logger

logger = get_logger(__name__)


@serve.deployment
class PredictorDeployment:
    def __init__(self, infer_conf: InferenceConfig, max_num_seqs):
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
        self.is_mllm = True if chat_processor_name in ["ChatModelwithImage"] else False

        # Used to determine if openai backend is used
        self.use_openai = False

        if infer_conf.device == "hpu":
            from llm_on_ray.inference.hpu_predictor import HPUPredictor

            self.predictor = HPUPredictor(infer_conf)
        elif self.use_deepspeed:
            from llm_on_ray.inference.deepspeed_predictor import DeepSpeedPredictor

            self.predictor = DeepSpeedPredictor(infer_conf)
        elif self.use_vllm:
            from llm_on_ray.inference.vllm_predictor import VllmPredictor

            self.predictor = VllmPredictor(infer_conf, max_num_seqs)
        elif self.is_mllm:
            from llm_on_ray.inference.mllm_predictor import MllmPredictor

            self.predictor = MllmPredictor(infer_conf)
        else:
            from llm_on_ray.inference.transformer_predictor import TransformerPredictor

            self.predictor = TransformerPredictor(infer_conf)

        self.loop = asyncio.get_running_loop()

    def consume_streamer(self, streamer):
        for text in streamer:
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

    # Handle streaming, only support single prompt
    async def handle_streaming(self, prompt: Union[str, List[str]], config: Dict[str, Any]):
        if isinstance(prompt, List):
            error_message = (
                "Streaming response is not supported when multiple prompts are provided."
            )
            if not self.use_openai:
                yield JSONResponse(
                    status_code=400,
                    content=error_message,
                )
            else:
                yield ModelResponse(
                    error=ErrorResponse(
                        message=error_message,
                        code=400,
                        internal_message=error_message,
                        type="InternalServerError",
                    )
                )
        if self.use_deepspeed:
            streamer = self.predictor.get_streamer()
            self.predictor.streaming_generate(prompt, streamer, **config)
            if not self.use_openai:
                yield StreamingResponse(
                    self.consume_streamer(streamer), status_code=200, media_type="text/plain"
                )
            else:
                async for output in self.consume_streamer_async(streamer):
                    model_response = ModelResponse(
                        generated_text=output,
                        num_input_tokens=self.predictor.input_length,
                        num_input_tokens_batch=self.predictor.input_length,
                        num_generated_tokens=1,
                        preprocessing_time=0,
                    )
                    yield model_response
        elif self.use_vllm:
            results_generator = await self.predictor.streaming_generate_async(prompt, **config)
            if not self.use_openai:
                yield StreamingResponse(
                    self.predictor.stream_results(results_generator),
                    status_code=200,
                    media_type="text/plain",
                )
            else:
                async for output in self.predictor.stream_results(results_generator):
                    model_response = ModelResponse(
                        generated_text=output,
                        num_input_tokens=self.predictor.input_length,
                        num_input_tokens_batch=self.predictor.input_length,
                        num_generated_tokens=1,
                        preprocessing_time=0,
                    )
                    yield model_response
        else:  # use transformers predictor
            streamer = self.predictor.get_streamer()
            self.loop.run_in_executor(
                None,
                functools.partial(self.predictor.streaming_generate, prompt, streamer, **config),
            )

            if not self.use_openai:
                yield StreamingResponse(
                    self.consume_streamer_async(streamer), status_code=200, media_type="text/plain"
                )
            else:
                async for output in self.consume_streamer_async(streamer):
                    model_response = ModelResponse(
                        generated_text=output,
                        num_input_tokens=self.predictor.input_length,
                        num_input_tokens_batch=self.predictor.input_length,
                        num_generated_tokens=1,
                        preprocessing_time=0,
                    )
                    yield model_response

    # Handle non-streaming, support single and multiple prompts
    async def handle_non_streaming(self, prompts, config) -> Union[JSONResponse, str]:
        # Use vllm for continuous batching
        if self.use_vllm:
            return await self.predictor.generate_async(prompts, **config)
        else:
            # static batching
            if isinstance(prompts, list):
                return await self.handle_static_batch(prompts, **config)
            # dynamic batching
            return await self.handle_dynamic_batch((prompts, config))

        return JSONResponse(status_code=400, content="Error when handling non-streaming request.")

    # TODO: get max_batch_size from the serve config
    @serve.batch(max_batch_size=4)
    async def handle_dynamic_batch(self, requests):
        logger.info(f"Handling dynamic batch (size={len(requests)}) ...")

        # batch prompts that configs are the same and also save request index
        batched_prompts: Dict[str, Tuple[Union[str, List[str]], List[int]]] = {}
        for i, request in enumerate(requests):
            prompt = request[0]
            config = request[1]
            # sort the config by key and convert to str to ensure the key is unique
            key = str(dict(sorted(config.items())))
            batched_prompts.setdefault(key, ([], []))
            batched_prompts[key][0].append(prompt)
            batched_prompts[key][1].append(i)

        logger.debug("Batched prompts: ", batched_prompts)

        # return results of each batch and fill in final results according to the request indices
        results = [None] * len(requests)
        for key, (prompts, indices) in batched_prompts.items():
            config = dict(eval(key))
            # use transformers predictor for batch generation
            batch_results = self.predictor.generate(prompts, **config)
            for index, result in zip(indices, batch_results):
                results[index] = result
        if not self.use_openai:
            return results
        else:
            return [
                ModelResponse(
                    generated_text=result.text,
                    num_input_tokens=self.predictor.input_length,
                    num_input_tokens_batch=self.predictor.input_length,
                    num_generated_tokens=result.generate_length,
                    preprocessing_time=0,
                )
                for result in results
            ]

    async def handle_static_batch(self, prompts: List[str], **config: Dict[str, Any]):
        logger.info(f"Handling static batch (size={len(prompts)}) ...")
        # Use vllm predictor for batch generation
        if self.use_vllm:
            results = await self.predictor.generate_async(prompts, **config)
            if not self.use_openai:
                return results
            else:
                # TODO: Output responses for a batch in openai format
                ModelResponse(
                    generated_text=results[0].text,
                    num_input_tokens=results[0].input_length,
                    num_input_tokens_batch=results[0].input_length,
                    num_generated_tokens=results[0].generate_length,
                    preprocessing_time=0,
                )
        else:
            # TODO: Output responses for a batch in openai format
            results = self.predictor.generate(prompts, **config)
            if not self.use_openai:
                return results
            else:
                return ModelResponse(
                    generated_text=results[0].text,
                    num_input_tokens=results[0].input_length,
                    num_input_tokens_batch=results[0].input_length,
                    num_generated_tokens=results[0].generate_length,
                    preprocessing_time=0,
                )

    def preprocess_prompts(self, input: Union[str, List[str]]):
        """
        Preprocesses the input prompts.

        Args:
            input (Union[str, List[str]]): The input prompt(s) to be preprocessed.

        Returns:
            Union[str, List[str], Tuple[List[str], List[str]]]: The preprocessed prompt(s):

            - str: If the input is a single prompt or
                   if the input is a list of prompts (CHAT_FORMAT) and processed by tool
            - List[str]: If the input is a list of prompts (CHAT_FORMAT) and not processed by tool.
                         If the input is a list of prompts (PROMPTS_FORMAT)
            - Tuple[List[str], List[str]]: If the input is a list of prompts (CHAT_FORMAT) and processed
                                           by MLLM (Multi-Modal Language Model) tool.

        Raises:
            HTTPException: If the input prompt format is invalid or not supported.
        """
        if isinstance(input, str):
            return input
        elif isinstance(input, list):
            prompts = []
            images = []
            prompt_format = get_prompt_format(input)
            if prompt_format == PromptFormat.CHAT_FORMAT:
                if self.process_tool is not None:
                    if self.is_mllm:
                        input, image = self.process_tool.get_prompt(input)
                        prompts.append(input)
                        images.extend(image)
                        return prompts, images
                    else:
                        prompt = self.process_tool.get_prompt(input)
                        return prompt
                else:
                    prompts.extend(input)
            elif prompt_format == PromptFormat.PROMPTS_FORMAT:
                prompts.extend(input)
            else:
                raise HTTPException(400, "Invalid prompt format.")
            return prompts
        else:
            raise HTTPException(400, "Invalid prompt format.")

    async def __call__(self, http_request: Request) -> Union[StreamingResponse, JSONResponse, str]:
        self.use_openai = False

        try:
            json_request: Dict[str, Any] = await http_request.json()
        except ValueError:
            return JSONResponse(
                status_code=400,
                content="Invalid JSON format from http request.",
            )

        streaming_response = json_request["stream"] if "stream" in json_request else False
        input = json_request["text"] if "text" in json_request else ""
        if input == "":
            return JSONResponse(
                status_code=400,
                content="Empty prompt is not supported.",
            )
        config = json_request["config"] if "config" in json_request else {}

        # return prompt or list of prompts preprocessed
        prompts = self.preprocess_prompts(input)

        # Handle streaming response
        if streaming_response:
            async for result in self.handle_streaming(prompts, config):
                return result

        return await self.handle_non_streaming(prompts, config)

    async def openai_call(
        self, input: Union[str, List[ChatMessage]], config: Dict, streaming_response=True
    ):
        self.use_openai = True

        # return prompt or list of prompts preprocessed
        prompts = self.preprocess_prompts(input)

        # Handle streaming response
        if streaming_response:
            async for result in self.handle_streaming(prompts, config):
                yield result
        else:
            yield await self.handle_non_streaming(prompts, config)
