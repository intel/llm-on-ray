from typing import Optional, List
import time
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from transformers import TextIteratorStreamer
from inference.inference_config import InferenceConfig, PRECISION_BF16
from predictor import Predictor
from utils import get_torch_dtype
from benchmark_utils import BenchmarkWrapper

class TextStreamerWithBench(TextIteratorStreamer):
    def __init__(self, tokenizer: "AutoTokenizer", skip_prompt: bool = False, timeout: Optional[float] = None, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, timeout, **decode_kwargs)
        self.times: List[int] = []

    def tick(self):
        self.times.append(time.perf_counter_ns())
        
    def on_finalized_text(self, text: str, stream_end: bool = False):
        if (not stream_end):
            self.times.append(time.perf_counter_ns())
        else:
            size = len(self.times)            
            diff_list = [(self.times[i + 1] - self.times[i])*1.0/1000_000 for i in range(0, size) if i < size - 1]
            print(len(diff_list))
            print(diff_list)
            first_token = diff_list.pop(0)
            next_token = sum(diff_list)*1.0/len(diff_list)
            print("first_token(ms): %s, next_token(ms): %s" % (first_token, next_token))
        super().on_finalized_text(text, stream_end)

class TransformerPredictor(Predictor):
    def __init__(self, infer_conf: InferenceConfig):
        super().__init__(infer_conf)
        model_desc = infer_conf.model_description
        model_config = model_desc.config
        hf_config = AutoConfig.from_pretrained(
            model_desc.model_id_or_path,
            torchscript=True,
            trust_remote_code=model_config.trust_remote_code,
        )

        if self.device.type == "hpu":
            from optimum.habana.transformers.modeling_utils import (
                adapt_transformers_to_gaudi,
            )

            adapt_transformers_to_gaudi()
        # get correct torch type for loading HF model
        torch_dtype = get_torch_dtype(infer_conf, hf_config)
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
                torch_dtype=torch_dtype,
                config=hf_config,
                low_cpu_mem_usage=True,
                **bmodel_config,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_desc.model_id_or_path,
                torch_dtype=torch_dtype,
                config=hf_config,
                low_cpu_mem_usage=True,
                **model_config.dict(),
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
                from habana_frameworks.torch.hpu import (
                    wrap_in_hpu_graph,
                )  # pylint: disable=E0401

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
        self.model = BenchmarkWrapper(model, do_print=True)

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
        streamer.tick()
        self.model.generate(
            input_ids,
            stopping_criteria=self.stopping_criteria,
            streamer=streamer,
            **config,
        )

    def generate(self, prompt, **config):
        self._process_config(config)
        input_ids = self.tokenize_inputs(prompt)
        start = time.perf_counter_ns()
        gen_tokens = self.model.generate(
            input_ids, stopping_criteria=self.stopping_criteria, **config
        )
        end = time.perf_counter_ns()
        print("total_time: %s" % ((end - start)*1.0/1000_000))
        ret = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        end = time.perf_counter_ns()
        print("total_time_with_decode: %s" % ((end - start)*1.0/1000_000))
        return ret

    def get_streamer(self):
        return TextStreamerWithBench(
            self.tokenizer, skip_prompt=True, timeout=0, skip_special_tokens=True
        )

