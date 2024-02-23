import asyncio
from typing import AsyncGenerator, List, Union
from predictor import Predictor
from inference.inference_config import InferenceConfig, GenerateResult, PRECISION_BF16
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid


class VllmPredictor(Predictor):
    def __init__(self, infer_conf: InferenceConfig):
        super().__init__(infer_conf)

        model_desc = infer_conf.model_description
        model_config = model_desc.config
        dtype = "bfloat16" if infer_conf.vllm.precision == PRECISION_BF16 else "float32"

        args = AsyncEngineArgs(
            model=model_desc.model_id_or_path,
            trust_remote_code=model_config.trust_remote_code,
            device=infer_conf.device,
            dtype=dtype,
        )

        self.engine = AsyncLLMEngine.from_engine_args(args)

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
                return GenerateResult(
                    text=request_output.outputs[0].text,
                    input_length=len(request_output.prompt_token_ids),
                    generate_length=len(request_output.outputs[0].token_ids),
                )
        return None

    async def generate_async(self, prompts: Union[str, List[str]], **config) -> GenerateResult:
        config = self.update_vllm_config(**config)
        sampling_params = SamplingParams(**config)
        if isinstance(prompts, str):
            request_id = random_uuid()
            results_generator = self.engine.generate(prompts, sampling_params, request_id)
            async for request_output in results_generator:
                if request_output.finished:
                    return GenerateResult(
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

        return GenerateResult()

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
