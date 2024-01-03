from predictor import Predictor
from inference_config import InferenceConfig

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
import json

class VllmPredictor(Predictor):
    def __init__(self, infer_conf: InferenceConfig):
        super().__init__(infer_conf)

        model_desc = infer_conf.model_description
        model_config = model_desc.config

        args = AsyncEngineArgs(model = model_desc.model_id_or_path,
                               trust_remote_code=model_config.trust_remote_code,
                               device=infer_conf.device)
        self.engine = AsyncLLMEngine.from_engine_args(args)

    async def streaming_generate(self, prompt, streamer, **config):
        sampling_params = SamplingParams(**config)
        request_id = random_uuid()
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        async for request_output in results_generator:
            prompt = request_output.prompt
            [
                streamer.put(output.text) for output in request_output.outputs
            ]
        streamer.end()

    async def generate(self, prompt, **config):
        return await self._generate(prompt, **config)

    async def _generate(self, prompt, **config):
        sampling_params = SamplingParams(**config)
        request_id = random_uuid()
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        assert final_output is not None
        prompt = final_output.prompt
        # TODO: return gen_ids or text?
        text_outputs = [prompt + output.text for output in final_output.outputs]
        ret = {"text": text_outputs}
        return json.dumps(ret)