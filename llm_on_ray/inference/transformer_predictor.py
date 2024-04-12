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

import torch
from transformers import AutoModelForCausalLM, AutoConfig, TextIteratorStreamer
from llm_on_ray.inference.inference_config import InferenceConfig, GenerateResult, PRECISION_BF16
from llm_on_ray.inference.utils import decide_torch_dtype
from llm_on_ray.inference.predictor import Predictor


class TransformerPredictor(Predictor):
    def __init__(self, infer_conf: InferenceConfig):
        super().__init__(infer_conf)
        model_desc = infer_conf.model_description
        model_config = model_desc.config
        hf_config = AutoConfig.from_pretrained(
            model_desc.model_id_or_path,
            torchscript=True,
            trust_remote_code=model_config.trust_remote_code,
            use_auth_token=infer_conf.model_description.config.use_auth_token,
        )

        # decide correct torch type for loading HF model
        decide_torch_dtype(infer_conf, hf_config)
        if model_desc.bigdl:
            from bigdl.llm.transformers import (
                AutoModelForCausalLM as BigDLAutoModelForCLM,
            )

            bmodel_config = {}
            bmodel_config.update(model_config.dict())
            if model_desc.bigdl_config.load_in_low_bit:
                bmodel_config.update(model_desc.bigdl_config.dict())
            model = BigDLAutoModelForCLM.from_pretrained(
                model_desc.model_id_or_path,
                config=hf_config,
                low_cpu_mem_usage=True,
                **bmodel_config,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_desc.model_id_or_path,
                config=hf_config,
                low_cpu_mem_usage=True,
                **model_config.dict(),
            )
        if model_desc.peft_model_id_or_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model,
                model_desc.peft_model_id_or_path,
                use_auth_token=infer_conf.model_description.config.use_auth_token,
            )
            if model_desc.peft_type == "deltatuner":
                from deltatuner import DeltaTunerModel

                model = DeltaTunerModel.from_pretrained(model, model_desc.peft_model_id_or_path)
            model = model.merge_and_unload()

        model = model.eval().to(self.device)
        # to channels last
        model = model.to(memory_format=torch.channels_last)
        # to ipex
        if infer_conf.ipex.enabled:
            import intel_extension_for_pytorch as ipex

            torch._C._jit_set_texpr_fuser_enabled(False)
            try:
                ipex._C.disable_jit_linear_repack()
            except Exception:
                pass
            model = ipex.llm.optimize(
                model.eval(),
                dtype=torch.bfloat16
                if infer_conf.ipex.precision == PRECISION_BF16
                else torch.float32,
                inplace=True,
            )
        self.model = model

    def streaming_generate(self, prompt, streamer, **config):
        input_ids, _ = self.tokenize_inputs(prompt)
        self.model.generate(
            input_ids,
            stopping_criteria=self.stopping_criteria,
            streamer=streamer,
            **config,
        )

    def generate(self, prompt, **config):
        input_ids, input_length = self.tokenize_inputs(prompt)
        gen_tokens = self.model.generate(
            input_ids, stopping_criteria=self.stopping_criteria, **config
        )
        decode_result = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        if isinstance(prompt, list) and len(prompt) > 1:
            return decode_result
        return GenerateResult(
            text=decode_result,
            input_length=input_length,
            generate_length=gen_tokens.size()[1] - input_length,
        )

    def get_streamer(self):
        return TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, timeout=0, skip_special_tokens=True
        )
