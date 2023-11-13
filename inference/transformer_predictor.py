import torch
from transformers import AutoModelForCausalLM

class TransformerPredictor:
    def __init__(self, model_id, model_load_config, device_name, amp_enabled, amp_dtype, pad_token_id, stopping_criteria, ipex_enabled):
        self.amp_enabled = amp_enabled
        self.amp_dtype = amp_dtype
        self.device = torch.device(device_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=amp_dtype,
            low_cpu_mem_usage=True,
            **model_load_config
        )

        model = model.eval().to(self.device)
        # to channels last
        model = model.to(memory_format=torch.channels_last)
        # to ipex
        if ipex_enabled:
            import intel_extension_for_pytorch as ipex
            try: ipex._C.disable_jit_linear_repack()
            except: pass
            self.model = ipex.optimize(model, dtype=amp_dtype, inplace=True)
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