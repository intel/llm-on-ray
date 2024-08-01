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

from llm_on_ray.inference.chat_template_process import ChatTemplatePreprocess
from llm_on_ray.inference.inference_config import InferenceConfig, DEVICE_HPU, DEVICE_CUDA
from llm_on_ray.inference.api_openai_backend.openai_protocol import (
    ChatMessage,
    ErrorResponse,
    ModelResponse,
)
from llm_on_ray.inference.api_simple_backend.simple_protocol import (
    SimpleRequest,
    SimpleModelResponse,
)
from llm_on_ray.inference.predictor import GenerateInput
from llm_on_ray.inference.utils import get_prompt_format, PromptFormat
from llm_on_ray.inference.api_openai_backend.tools import OpenAIToolsPrompter, ChatPromptCapture
from llm_on_ray.inference.logger import get_logger

logger = get_logger(__name__)


@serve.deployment
class PredictorDeployment:
    _DEFAULT_MAX_BATCH_SIZE = 8
    _DEFAULT_MAX_NUM_SEQS = 256

    def __init__(
        self,
        infer_conf: InferenceConfig,
        max_num_seqs=_DEFAULT_MAX_NUM_SEQS,
        max_batch_size=_DEFAULT_MAX_BATCH_SIZE,
    ):
        self.device = torch.device(infer_conf.device)

        self.handle_dynamic_batch.set_max_batch_size(max_batch_size)
        self.use_deepspeed = infer_conf.deepspeed
        self.use_vllm = infer_conf.vllm.enabled
        self.is_mllm = infer_conf.model_description.chat_model_with_image

        # Used to determine if openai backend is used
        self.use_openai = False
        self.vllm_openai_serving_chat = None

        if infer_conf.device == "hpu" and not self.use_vllm:
            from llm_on_ray.inference.predictors.hpu_predictor import HPUPredictor

            self.predictor = HPUPredictor(infer_conf)
        elif self.use_deepspeed:
            from llm_on_ray.inference.predictors.deepspeed_predictor import DeepSpeedPredictor

            self.predictor = DeepSpeedPredictor(infer_conf)
        elif self.use_vllm:
            if infer_conf.device not in [DEVICE_HPU, DEVICE_CUDA]:
                from llm_on_ray.inference.predictors.vllm_predictor import VllmPredictor

                self.predictor = VllmPredictor(infer_conf, max_num_seqs)
            else:
                self.predictor = None
        elif self.is_mllm:
            from llm_on_ray.inference.predictors.mllm_predictor import MllmPredictor

            self.predictor = MllmPredictor(infer_conf)
        else:
            from llm_on_ray.inference.predictors.transformer_predictor import TransformerPredictor

            self.predictor = TransformerPredictor(infer_conf)

        self.loop = asyncio.get_running_loop()
        self.process_tool = ChatTemplatePreprocess(self.predictor)

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
    async def handle_streaming(self, input: GenerateInput, config: Dict[str, Any]):
        if isinstance(input, List):
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
            self.predictor.streaming_generate(input, streamer, **config)
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
            results_generator = await self.predictor.streaming_generate_async(input, **config)
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
                functools.partial(self.predictor.streaming_generate, input, streamer, **config),
            )

            if not self.use_openai:
                yield StreamingResponse(
                    self.consume_streamer_async(streamer), status_code=200, media_type="text/plain"
                )
            else:
                async for output in self.consume_streamer_async(streamer):
                    processed_output = output
                    tool_call_list = None
                    if self.tools_capture_texts is not None:
                        (
                            processed_output,
                            tool_call_list,
                        ) = self.tools_capture_texts.process_full_output(
                            output, self.openai_tools_prompter, input
                        )
                    model_response = ModelResponse(
                        generated_text=processed_output,
                        tool_calls=tool_call_list,
                        num_input_tokens=self.predictor.input_length,
                        num_input_tokens_batch=self.predictor.input_length,
                        num_generated_tokens=1,
                        preprocessing_time=0,
                    )
                    yield model_response

    # Handle non-streaming, support single and multiple prompts
    async def handle_non_streaming(self, input: GenerateInput, config) -> Union[JSONResponse, str]:
        # Use vllm for continuous batching
        if self.use_vllm:
            results = await self.predictor.generate_async(input, **config)
            if isinstance(results, list):
                responses = []
                for result in results:
                    responses.append(
                        ModelResponse(
                            generated_text=result.text,
                            num_input_tokens=self.predictor.input_length,
                            num_input_tokens_batch=self.predictor.input_length,
                            num_generated_tokens=result.generate_length,
                            preprocessing_time=0,
                        )
                    )
                return responses
            else:
                return ModelResponse(
                    generated_text=results.text,
                    num_input_tokens=results.input_length,
                    num_input_tokens_batch=results.input_length,
                    num_generated_tokens=results.generate_length,
                    preprocessing_time=0,
                )
        else:
            # static batching
            if isinstance(input, list):
                return await self.handle_static_batch(input, **config)

            # dynamic batching
            return await self.handle_dynamic_batch((input, config))

        return JSONResponse(status_code=400, content="Error when handling non-streaming request.")

    # TODO: get max_batch_size from the serve config
    @serve.batch(max_batch_size=_DEFAULT_MAX_BATCH_SIZE)
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
            responses = []
            tool_call_list = None
            for result in results:
                if self.tools_capture_texts is not None:
                    result.text, tool_call_list = self.tools_capture_texts.process_full_output(
                        result.text, self.openai_tools_prompter, prompts
                    )
                responses.append(
                    ModelResponse(
                        generated_text=result.text,
                        tool_calls=tool_call_list,
                        num_input_tokens=self.predictor.input_length,
                        num_input_tokens_batch=self.predictor.input_length,
                        num_generated_tokens=result.generate_length,
                        preprocessing_time=0,
                    )
                )
            return responses

    async def handle_static_batch(self, prompts: List[str], **config: Dict[str, Any]):
        logger.info(f"Handling static batch (size={len(prompts)}) ...")
        # Use vllm predictor for batch generation
        if self.use_vllm:
            results = await self.predictor.generate_async(prompts, **config)
            if not self.use_openai:
                return results
            else:
                # TODO: Output responses for a batch in openai format
                tool_call_list = None
                if self.tools_capture_texts is not None:
                    results[0].text, tool_call_list = self.tools_capture_texts.process_full_output(
                        results[0].text, self.openai_tools_prompter, prompts
                    )
                ModelResponse(
                    generated_text=results[0].text,
                    tool_calls=tool_call_list,
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

    def preprocess_prompts(
        self, input: Union[str, List], tools=None, tool_choice=None
    ) -> GenerateInput:
        """
        Preprocesses the input prompts.

        Args:
            input (Union[str, List[str]]): The input prompt(s) to be preprocessed.
            tools (List[str]): The list of tools to be used.
            tool_choice: The choice of tool to be used.

        Returns:
            The preprocessed prompt(s) in the following `GenerateInput` format:

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
        elif isinstance(input, List):
            prompts = []
            images = []

            prompt_format = get_prompt_format(input)
            if prompt_format == PromptFormat.CHAT_FORMAT:
                # Process the input prompts with tools
                self.tool_call_list = None
                self.openai_tools_prompter: OpenAIToolsPrompter = (
                    OpenAIToolsPrompter() if tools is not None else None
                )
                self.tools_capture_texts: ChatPromptCapture = None
                if self.openai_tools_prompter is not None:
                    input = self.openai_tools_prompter.inject_prompt(input, tools, tool_choice)
                    self.tools_capture_texts = ChatPromptCapture()
                    for m in input:
                        if m.tool_calls is not None:  # type: ignore
                            m.content = self.openai_tools_prompter.content_from_assistant(m)  # type: ignore
                        elif m.tool_call_id is not None:  # type: ignore
                            m.content = self.openai_tools_prompter.content_from_tool(m)  # type: ignore
                # Process the input prompts with MLLM tool
                if self.process_tool is not None:
                    if self.is_mllm:
                        input, image = self.process_tool.get_prompt(input, self.is_mllm)
                        prompts.append(input)
                        images.extend(image)
                        return (prompts, images)
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
            request: Dict[str, Any] = await http_request.json()
        except ValueError:
            return JSONResponse(
                status_code=400,
                content="Invalid JSON format from http request.",
            )

        streaming_response = request["stream"]
        input = request["text"]
        config = request["config"]

        prompts = self.preprocess_prompts(input)

        # Handle streaming response
        if streaming_response:
            async for result in self.handle_streaming(prompts, config):
                return result

        return await self.handle_non_streaming(prompts, config)

    async def openai_call(
        self,
        input: Union[str, List[ChatMessage]],
        config: Dict,
        streaming_response=True,
        tools=None,
        tool_choice=None,
    ):
        self.use_openai = True

        # return prompt or list of prompts preprocessed
        input = self.preprocess_prompts(input, tools, tool_choice)

        # Handle streaming response
        if streaming_response:
            async for result in self.handle_streaming(input, config):
                yield result
        else:
            yield await self.handle_non_streaming(input, config)
