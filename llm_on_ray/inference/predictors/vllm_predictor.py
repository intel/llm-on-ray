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

import asyncio
import os
from typing import AsyncGenerator, List, Union
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from llm_on_ray.inference.predictor import GenerateInput, GenerateOutput, Predictor
from llm_on_ray.inference.inference_config import (
    InferenceConfig,
    ModelGenerateResult,
    PRECISION_BF16,
    DEVICE_HPU,
    DEVICE_CUDA,
)


class VllmPredictor(Predictor):
    VLLM_CPU_KVCACHE_SPACE_DEFAULT = 40

    def __init__(self, infer_conf: InferenceConfig, max_num_seqs):
        super().__init__(infer_conf)

        model_desc = infer_conf.model_description
        model_config = model_desc.config
        dtype = "bfloat16" if infer_conf.vllm.precision == PRECISION_BF16 else "float32"

        # Set environment variable VLLM_CPU_KVCACHE_SPACE to control the size of the CPU key-value cache.
        # The default value is 40GB.
        os.environ["VLLM_CPU_KVCACHE_SPACE"] = str(self.VLLM_CPU_KVCACHE_SPACE_DEFAULT)

        engine_args = AsyncEngineArgs(
            model=model_desc.model_id_or_path,
            tokenizer=model_desc.tokenizer_name_or_path,
            trust_remote_code=model_config.trust_remote_code,
            device=infer_conf.device,
            dtype=dtype,
            disable_log_requests=True,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=infer_conf.vllm.gpu_memory_utilization,
            tensor_parallel_size=infer_conf.vllm.tensor_parallel_size,
            block_size=infer_conf.vllm.block_size,
            max_seq_len_to_capture=infer_conf.vllm.max_seq_len_to_capture,
            enforce_eager=infer_conf.vllm.enforce_eager,
        )
        if (
            infer_conf.device in [DEVICE_HPU, DEVICE_CUDA]
            and infer_conf.vllm.tensor_parallel_size > 1
        ):
            engine_args.worker_use_ray = True
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    def update_vllm_config(self, **config):
        # need to update the keys of config if vllm engine is used
        if "max_new_tokens" in config.keys():
            config["max_tokens"] = config.pop("max_new_tokens")
        if "do_sample" in config.keys():
            config.pop("do_sample")
        unused = [k for k, v in config.items() if v is None]
        [config.pop(k) for k in unused]
        return config

    async def _get_generator_output(self, results_generator):
        async for request_output in results_generator:
            if request_output.finished:
                return ModelGenerateResult(
                    text=request_output.outputs[0].text,
                    input_length=len(request_output.prompt_token_ids),
                    generate_length=len(request_output.outputs[0].token_ids),
                )
        return None

    def generate(
        self,
        input: GenerateInput,
        **config,
    ) -> GenerateOutput:
        # This method is not used for VllmPredictor, used generate_async instead
        pass

    async def generate_async(self, prompts: Union[str, List[str]], **config) -> ModelGenerateResult:
        config = self.update_vllm_config(**config)
        sampling_params = SamplingParams(**config)
        if isinstance(prompts, str):
            request_id = random_uuid()
            results_generator = self.engine.generate(prompts, sampling_params, request_id)
            async for request_output in results_generator:
                if request_output.finished:
                    return ModelGenerateResult(
                        text=request_output.outputs[0].text,
                        input_length=len(request_output.prompt_token_ids),
                        generate_length=len(request_output.outputs[0].token_ids),
                    )
        else:
            results_generators = [
                self.engine.generate(prompt, sampling_params, random_uuid()) for prompt in prompts
            ]
            results = [
                self._get_generator_output(results_generator)
                for results_generator in results_generators
            ]
            return await asyncio.gather(*results)

        return ModelGenerateResult()

    async def streaming_generate_async(self, prompt, **config):
        config = self.update_vllm_config(**config)
        sampling_params = SamplingParams(**config)
        request_id = random_uuid()
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        return results_generator

    async def stream_results(self, results_generator) -> AsyncGenerator[str, None]:
        num_returned = 0
        async for request_output in results_generator:
            text_outputs = [output.text for output in request_output.outputs]
            self.input_length = len(request_output.prompt_token_ids)
            assert len(text_outputs) == 1
            text_output = text_outputs[0][num_returned:]
            yield text_output
            num_returned += len(text_output)
