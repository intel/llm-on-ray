import torch
from transformers import AutoModelForCausalLM, AutoConfig
from peft import PeftModel
from deltatuner import DeltaTunerModel
from inference_config import InferenceConfig

class TransformerPredictor:
    def __init__(self, inferenceConfig: InferenceConfig, amp_dtype, pad_token_id, stopping_criteria):
        self.amp_dtype = amp_dtype
        self.device = torch.device(inferenceConfig.device)

        model_desc = inferenceConfig.model_description
        model_config = model_desc.config
        config = AutoConfig.from_pretrained(model_desc.model_id_or_path, torchscript=True, trust_remote_code=model_config.trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_desc.model_id_or_path,
            torch_dtype=amp_dtype,
            config=config,
            low_cpu_mem_usage=True,
            **model_config.dict()
        )
        if model_desc.peft_model_id_or_path:
            model = PeftModel.from_pretrained(model, model_desc.peft_model_id_or_path)
            if model_desc.peft_type == "deltatuner":
                model = DeltaTunerModel.from_pretrained(model, model_desc.peft_model_id_or_path)
            model = model.merge_and_unload()

        model = model.eval().to(self.device)
        # to channels last
        model = model.to(memory_format=torch.channels_last)
        # to ipex
        if inferenceConfig.ipex:
            import intel_extension_for_pytorch as ipex

            torch._C._jit_set_texpr_fuser_enabled(False)
            try: ipex._C.disable_jit_linear_repack()
            except: pass
            self.model = ipex.optimize_transformers(
                model.eval(),
                dtype=amp_dtype,
                inplace=True
            )
        else:
            self.model = model
        self.pad_token_id = pad_token_id
        self.stopping_criteria = stopping_criteria

    def streaming_generate(self, inputs, streamer, **config):
        self.model.generate(**inputs,
                    pad_token_id=self.pad_token_id,
                    stopping_criteria=self.stopping_criteria,
                    streamer=streamer,
                    **config)

    def generate(self, inputs, **config):
        gen_tokens = self.model.generate(
            **inputs,
            pad_token_id=self.pad_token_id,
            stopping_criteria=self.stopping_criteria,
            **config
        )
        return gen_tokens

    def ping(self):
        pass