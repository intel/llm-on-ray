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

from typing import List, Union
import torch
from transformers import TextIteratorStreamer
from llm_on_ray.inference.inference_config import (
    InferenceConfig,
    ModelGenerateResult,
    PRECISION_BF16,
)
from llm_on_ray.inference.utils import decide_torch_dtype, module_import
from llm_on_ray.inference.predictor import GenerateInput, GenerateOutput, MllmPromptInput, Predictor


class MllmPredictor(Predictor):
    def __init__(self, infer_conf: InferenceConfig):
        super().__init__(infer_conf)
        model_desc = infer_conf.model_description

        if self.device.type == "hpu":
            from optimum.habana.transformers.modeling_utils import (
                adapt_transformers_to_gaudi,
            )

            adapt_transformers_to_gaudi()
        # get correct torch type for loading HF model
        decide_torch_dtype(infer_conf)

        model_loader_name = infer_conf.model_description.model_loader
        input_processor_name = infer_conf.model_description.input_processor
        model = module_import("transformers", model_loader_name).from_pretrained(
            model_desc.model_id_or_path,
            **model_desc.config.dict(),
        )
        processor = module_import("transformers", input_processor_name).from_pretrained(
            model_desc.model_id_or_path,
            **model_desc.config.dict(),
        )

        model = model.eval().to(self.device)
        if self.device.type == "hpu":
            self.use_hpu_graphs = model_desc.use_hpu_graphs
            if self.use_hpu_graphs:
                from habana_frameworks.torch.hpu import (
                    wrap_in_hpu_graph,
                )  # pylint: disable=E0401

                model = wrap_in_hpu_graph(model)
            else:
                print("Warning: use_hpu_graphs is set to False. This will hurt the performance.")
        else:
            # # to channels last
            model = model.to(memory_format=torch.channels_last)
            # to ipex
            if infer_conf.ipex.enabled:
                import intel_extension_for_pytorch as ipex

                torch._C._jit_set_texpr_fuser_enabled(False)
                try:
                    ipex._C.disable_jit_linear_repack()
                except Exception:
                    pass
                model = ipex.optimize_transformers(
                    model.eval(),
                    dtype=torch.bfloat16
                    if infer_conf.ipex.precision == PRECISION_BF16
                    else torch.float32,
                    inplace=True,
                )
        self.model = model
        self.processor = processor

    def _process_config(self, config):
        if self.device.type == "hpu":
            if "max_new_tokens" not in config:
                # hpu requires setting max_new_tokens
                config["max_new_tokens"] = 256
            if self.use_hpu_graphs:
                config["hpu_graphs"] = True
                # lazy mode should be True when using hpu graphs
                config["lazy_mode"] = True

    def _tokenize_inputs(self, text_prompt, images):
        input_tokens = self.processor(text=text_prompt, images=images, return_tensors="pt")
        if self.device.type != "cpu":
            input_tokens = input_tokens.to(device=self.device)
        return input_tokens

    def streaming_generate(self, prompts, images, streamer, **config):
        self._process_config(config)
        inputs = self._tokenize_inputs(prompts, images)
        self.model.generate(
            **inputs,
            stopping_criteria=self.stopping_criteria,
            streamer=streamer,
            **config,
        )

    def generate(self, input: GenerateInput, **config) -> GenerateOutput:
        if not isinstance(input, tuple):
            raise TypeError("MllmPredictor should use (prompt, image) as input.")

        prompts, images = input

        self._process_config(config)
        inputs = self._tokenize_inputs(prompts, images)
        input_length = sum([len(i) for i in prompts])
        gen_tokens = self.model.generate(
            **inputs, stopping_criteria=self.stopping_criteria, **config
        )
        decode_result = self.processor.batch_decode(gen_tokens, skip_special_tokens=True)
        output_length = len(decode_result)
        return ModelGenerateResult(
            text=decode_result,
            input_length=input_length,
            generate_length=output_length - input_length,
        )

    def get_streamer(self):
        return TextIteratorStreamer(
            self.processor, skip_prompt=True, timeout=0, skip_special_tokens=True
        )
