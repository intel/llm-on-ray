from transformers import TextIteratorStreamer
from inference.inference_config import InferenceConfig
from predictor import Predictor
from inference.utils import module_import


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
        # torch_dtype = get_torch_dtype(infer_conf, hf_config)

        # model = FuyuForCausalLM.from_pretrained(model_desc.model_id_or_path)
        # processor = FuyuProcessor.from_pretrained(model_desc.model_id_or_path)
        model_loader_name = infer_conf.model_description.model_loader
        input_processor_name = infer_conf.model_description.input_processor
        model = module_import("transformers", model_loader_name).from_pretrained(
            model_desc.model_id_or_path, local_files_only=True
        )
        processor = module_import("transformers", input_processor_name).from_pretrained(
            model_desc.model_id_or_path, local_files_only=True
        )

        # model = model.to(self.device)
        # # to channels last
        # model = model.to(memory_format=torch.channels_last)
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

    def _process_input(self, image, text_prompt):
        print(image)
        print(text_prompt)
        inputs = self.processor(text=text_prompt, images=image, return_tensors="pt")
        return inputs

    def streaming_generate(self, image, prompt, streamer, **config):
        self._process_config(config)
        inputs = self._process_input(image, prompt)
        self.model.generate(
            **inputs,
            stopping_criteria=self.stopping_criteria,
            streamer=streamer,
            **config,
        )

    def generate(self, image, prompt, **config):
        self._process_config(config)
        inputs = self._process_input(image, prompt)
        gen_tokens = self.model.generate(
            **inputs, stopping_criteria=self.stopping_criteria, **config
        )
        return self.processor.batch_decode(gen_tokens, skip_special_tokens=True)

    def get_streamer(self):
        return TextIteratorStreamer(
            self.processor, skip_prompt=True, timeout=0, skip_special_tokens=True
        )
