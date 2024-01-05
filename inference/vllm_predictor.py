from typing import AsyncGenerator, List
from predictor import Predictor
from inference_config import InferenceConfig
from transformers import TextIteratorStreamer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
import json
import asyncio

class VllmPredictor(Predictor):
    def __init__(self, infer_conf: InferenceConfig):
        super().__init__(infer_conf)

        model_desc = infer_conf.model_description
        model_config = model_desc.config

        args = AsyncEngineArgs(model = model_desc.model_id_or_path,
                               trust_remote_code=model_config.trust_remote_code,
                               device=infer_conf.device)

        self.engine = AsyncLLMEngine.from_engine_args(args)

    def generate(self, prompt, **config) -> List[str]:
        return asyncio.run(self._generate_async(prompt, **config))

    async def _generate_async(self, prompt, **config):
        sampling_params = SamplingParams(**config)
        request_id = random_uuid()
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        async for request_output in results_generator:
            if request_output.finished:
                return request_output.outputs[0].text

    async def streaming_generate_async(self, prompt, **config):
        sampling_params = SamplingParams(**config)
        request_id = random_uuid()
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        return results_generator

    async def stream_results(self, results_generator) -> AsyncGenerator[str, None]:
        num_returned = 0
        async for request_output in results_generator:
            text_outputs = [output.text for output in request_output.outputs]
            assert len(text_outputs) == 1
            text_output = text_outputs[0][num_returned:]
            yield text_output
            num_returned += len(text_output)
