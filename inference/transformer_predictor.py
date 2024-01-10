import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import TextIteratorStreamer
from inference.inference_config import InferenceConfig, IPEX_PRECISION_BF16
from predictor import Predictor
from utils import get_torch_dtype

class TransformerPredictor(Predictor):
    def __init__(self, infer_conf: InferenceConfig):
        super().__init__(infer_conf)
        
        model_desc = infer_conf.model_description
        model_config = model_desc.config
        hf_config = AutoConfig.from_pretrained(model_desc.model_id_or_path, torchscript=True, trust_remote_code=model_config.trust_remote_code, **infer_conf.model_description.config.dict())

        if self.device.type == "hpu":
            from optimum.habana.transformers.modeling_utils import (
                adapt_transformers_to_gaudi,
            )

            adapt_transformers_to_gaudi()
        # get correct torch type for loading HF model
        torch_dtype = get_torch_dtype(infer_conf, hf_config)
        if model_desc.bigdl:
            from bigdl.llm.transformers import AutoModelForCausalLM as BigDLAutoModelForCLM
            bmodel_config = {}
            bmodel_config.update(model_config.dict())
            if model_desc.bigdl_config.load_in_low_bit:
                bmodel_config.update(model_desc.bigdl_config.dict())
            model = BigDLAutoModelForCLM.from_pretrained(
                model_desc.model_id_or_path,
                torch_dtype=torch_dtype,
                config=hf_config,
                low_cpu_mem_usage=True,
                **bmodel_config
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_desc.model_id_or_path,
                torch_dtype=torch_dtype,
                config=hf_config,
                low_cpu_mem_usage=True,
                **model_config.dict()
            )
        if model_desc.peft_model_id_or_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, model_desc.peft_model_id_or_path)
            if model_desc.peft_type == "deltatuner":
                from deltatuner import DeltaTunerModel
                model = DeltaTunerModel.from_pretrained(model, model_desc.peft_model_id_or_path)
            model = model.merge_and_unload()

        model = model.eval().to(self.device)
        if self.device.type == "hpu":
            self.use_hpu_graphs = model_desc.use_hpu_graphs
            if self.use_hpu_graphs:
                from habana_frameworks.torch.hpu import wrap_in_hpu_graph # pylint: disable=E0401
                model = wrap_in_hpu_graph(model)
            else:
                print("Warning: use_hpu_graphs is set to False. This will hurt the performance.")
        else:
            # to channels last
            model = model.to(memory_format=torch.channels_last)
            # to ipex
            if infer_conf.ipex.enabled:
                import intel_extension_for_pytorch as ipex

                torch._C._jit_set_texpr_fuser_enabled(False)
                try: ipex._C.disable_jit_linear_repack()
                except: pass
                model = ipex.optimize_transformers(
                    model.eval(),
                    dtype=torch.bfloat16 if infer_conf.ipex.precision == IPEX_PRECISION_BF16 else torch.float32,
                    inplace=True
                )
        self.model = model

    def _process_config(self, config):
        if self.device.type == "hpu":
            if "max_new_tokens" not in config:
                # hpu requires setting max_new_tokens
                config["max_new_tokens"] = 256
            if self.use_hpu_graphs:
                config["hpu_graphs"] = True
                # lazy mode should be True when using hpu graphs
                config["lazy_mode"] = True

    def streaming_generate(self, prompt, streamer, **config):
        self._process_config(config)
        input_ids = self.tokenize_inputs(prompt)
        self.model.generate(input_ids,
                    stopping_criteria=self.stopping_criteria,
                    streamer=streamer,
                    **config)

    def generate(self, prompt, **config):
        self._process_config(config)
        input_ids = self.tokenize_inputs(prompt)
        gen_tokens = self.model.generate(
            input_ids,
            stopping_criteria=self.stopping_criteria,
            **config
        )
        return self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]

    def get_streamer(self):
        return TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=0, skip_special_tokens=True)
