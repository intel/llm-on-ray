import torch
from transformers import AutoModelForCausalLM, AutoConfig
from peft import PeftModel
from deltatuner import DeltaTunerModel

class TransformerPredictor:
    def __init__(self, model_id, model_load_config, device_name, amp_enabled, amp_dtype, pad_token_id, stopping_criteria, ipex_enabled, deployment_mode, deltatuner_model_id=None):
        self.amp_enabled = amp_enabled
        self.amp_dtype = amp_dtype
        self.device = torch.device(device_name)
        config = None
        if deployment_mode:
            trust_remote_code = model_load_config.get("trust_remote_code", None)
            config = AutoConfig.from_pretrained(
            model_id, torchscript=deployment_mode, trust_remote_code=trust_remote_code
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=amp_dtype,
            config=config,
            low_cpu_mem_usage=True,
            **model_load_config
        )
        if deltatuner_model_id:
            model = PeftModel.from_pretrained(model, deltatuner_model_id)
            model = DeltaTunerModel.from_pretrained(model, deltatuner_model_id)
            model = model.merge_and_unload()

        model = model.eval().to(self.device)
        # to channels last
        model = model.to(memory_format=torch.channels_last)
        # to ipex
        if ipex_enabled:
            import intel_extension_for_pytorch as ipex

            torch._C._jit_set_texpr_fuser_enabled(False)
            try: ipex._C.disable_jit_linear_repack()
            except: pass
            self.model = ipex.optimize_transformers(
                model.eval(),
                dtype=amp_dtype,
                inplace=True,
                deployment_mode=deployment_mode,
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