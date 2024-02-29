import torch
from transformers import TextIteratorStreamer
from inference.inference_config import InferenceConfig, GenerateResult, PRECISION_BF16
from predictor import Predictor
from inference.utils import module_import
from inference.utils import get_torch_dtype


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
        torch_dtype = get_torch_dtype(infer_conf, None)

        model_loader_name = infer_conf.model_description.model_loader
        input_processor_name = infer_conf.model_description.input_processor
        model = module_import("transformers", model_loader_name).from_pretrained(
            model_desc.model_id_or_path,
            torch_dtype=torch_dtype,
            **model_desc.config.dict(),
        )
        processor = module_import("transformers", input_processor_name).from_pretrained(
            model_desc.model_id_or_path,
            **model_desc.config.dict(),
        )

        model = model.eval().to(self.device)
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

    def _tokenize_inputs(self, image, text_prompt):
        input_tokens = self.processor(text=text_prompt, images=image, return_tensors="pt")
        return input_tokens

    def streaming_generate(self, image, prompt, streamer, **config):
        self._process_config(config)
        inputs = self._tokenize_inputs(image, prompt)
        self.model.generate(
            **inputs,
            stopping_criteria=self.stopping_criteria,
            streamer=streamer,
            **config,
        )

    def generate(self, image, prompt, **config):
        self._process_config(config)
        inputs = self._tokenize_inputs(image, prompt)
        input_length = sum([len(i) for i in prompt])
        gen_tokens = self.model.generate(
            **inputs, stopping_criteria=self.stopping_criteria, **config
        )
        decode_result = self.processor.batch_decode(gen_tokens, skip_special_tokens=True)
        output_length = len(decode_result)
        return GenerateResult(
            text=decode_result,
            input_length=input_length,
            generate_length=output_length - input_length,
        )

    def get_streamer(self):
        return TextIteratorStreamer(
            self.processor, skip_prompt=True, timeout=0, skip_special_tokens=True
        )
