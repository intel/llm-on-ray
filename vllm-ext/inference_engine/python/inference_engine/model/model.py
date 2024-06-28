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

import os
from filelock import FileLock
from typing import List
from pathlib import Path
import ctypes
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer, PretrainedConfig
from inference_engine.quant import convert_model

_VOCAB_SIZE_MAP = {"llama3": 128256}
_DEFAULT_QUANT_DIR = "runtime_outs/quantized_models"

_CTYPES_STR_ENCODE = "utf-8"

_TORCH_DTYPE_TO_INT = {
    "torch.float32": 1,
    "torch.float": 1,
    "torch.float64": 2,
    "torch.int8": 3,
    "torch.int16": 4,
    "torch.short": 4,
    "torch.int32": 5,
    "torch.int": 5,
    "torch.int64": 6,
    "torch.long": 6,
    "torch.uint8": 7,
}


class Model:
    def __init__(
        self,
        model_id,
        tokenizer: PreTrainedTokenizer = None,
        max_new_tokens: int = 512,
        max_batch_size: int = 512,
        ctx_size: int = 512,
        pad_token: int = -1,
        memory_dtype: str = "auto",  # auto, fp16, fp32
        scratch_size_ratio: float = 1.0,
        threads: int = 56,
        seed: int = 1234,
    ):
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.generation_args = {}
        self.generation_args["max_new_tokens"] = max_new_tokens
        self.generation_args["max_batch_size"] = max_batch_size
        self.generation_args["ctx_size"] = ctx_size
        self.generation_args["seed"] = seed
        self.generation_args["threads"] = threads
        self.generation_args["pad_token"] = pad_token
        self.generation_args["memory_dtype"] = memory_dtype
        self.generation_args["scratch_size_ratio"] = scratch_size_ratio

        self.cpp_module = None
        self.native_model_ptr = None
        self.quantized_model_path: str = None
        self.fp32_model_path: str = None
        self.config: PretrainedConfig = None

    def _init_cpp_model(self, config):
        if self.cpp_module is not None:
            return
        from inference_engine.model.model_cpp import load_engine_lib

        self.cpp_module = load_engine_lib(config.model_type)

    def _get_model_target_path(self, quant_dir, for_quantized: bool = False, **quant_kwargs):
        weight_type = quant_kwargs["weight_dtype"]
        group_size = quant_kwargs["group_size"]
        alg = quant_kwargs["alg"]
        compute_dtype = quant_kwargs["compute_dtype"]
        if for_quantized:
            desc = f"{self.model_id}_{self.config.model_type}_{weight_type}_{group_size}_{alg}_{compute_dtype}_quantized.bin"
        else:
            desc = f"{self.model_id}_{self.config.model_type}_fp32.bin"
        t_path = os.path.join(quant_dir, desc)
        return t_path

    def __set_pad_token(self, tokenizer):
        if self.generation_args["pad_token"] == -1:
            if tokenizer.pad_token:
                self.generation_args["pad_token"] = tokenizer.pad_token_id
            else:
                if tokenizer.unk_token:
                    tokenizer.pad_token_id = tokenizer.unk_token_id
                elif tokenizer.eos_token:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                else:
                    raise ValueError("Tokenizer has no pad, unk, or eos token")
                self.generation_args["pad_token"] = tokenizer.pad_token_id
        else:
            if tokenizer.pad_token_id:
                assert (
                    tokenizer.pad_token_id == self.generation_args["pad_token"]
                ), "different pad token ids in tokenizer and model init"
            else:
                tokenizer.pad_token_id = self.generation_args["pad_token"]

    def check_and_quantize(self, **quant_kwargs):
        self.config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
        self._init_cpp_model(self.config)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.__set_pad_token(self.tokenizer)
        # check if quantized model exists, quantize on the fly if not existed
        quant_dir = quant_kwargs.get("quant_dir", _DEFAULT_QUANT_DIR)
        if quant_dir == _DEFAULT_QUANT_DIR:
            os.makedirs(quant_dir, exist_ok=True)
        self.quantized_model_path = self._get_model_target_path(
            quant_dir, for_quantized=True, **quant_kwargs
        )
        parent_path = Path(self.quantized_model_path).parent
        os.makedirs(
            parent_path, exist_ok=True
        )  # in case model id is ***/***, e.g., "meta-llama/Llama-7b.."
        if os.path.exists(self.quantized_model_path):  # existed?
            return
        # check if intermediate fp32 model exists, convert on the fly if not existed
        self.fp32_model_path = self._get_model_target_path(
            quant_dir, for_quantized=False, **quant_kwargs
        )
        if not os.path.exists(self.fp32_model_path):
            # lock to prevent being messed up with mutiple processes
            fp32_lock_path = self.fp32_model_path + ".lock"
            with FileLock(fp32_lock_path):
                if not os.path.exists(self.fp32_model_path):  # converted by other process already?
                    fp32_path_tmp = self.fp32_model_path + ".tmp"
                    convert_model(self.model_id, fp32_path_tmp, "f32", model_hub="huggingface")
                    os.rename(fp32_path_tmp, self.fp32_model_path)
                    assert os.path.exists(
                        self.fp32_model_path
                    ), "Failed to convert model to intermediate fp32 model"
        # quantize
        quant_output_tmp = self.quantized_model_path + ".tmp"
        fp32_model_path = self.fp32_model_path.encode(_CTYPES_STR_ENCODE)
        quant_output_path = quant_output_tmp.encode(_CTYPES_STR_ENCODE)
        weight_dtype = quant_kwargs.get("weight_dtype", "int4").encode(_CTYPES_STR_ENCODE)
        alg = quant_kwargs.get("alg", "sym").encode(_CTYPES_STR_ENCODE)
        group_size = quant_kwargs.get("group_size", 32)
        scale_dtype = quant_kwargs.get("scale_dtype", "fp32").encode(_CTYPES_STR_ENCODE)
        compute_dtype = quant_kwargs.get("compute_dtype", "int8").encode(_CTYPES_STR_ENCODE)
        use_ggml = ctypes.c_bool(quant_kwargs.get("use_ggml", False))
        threads = quant_kwargs.get("threads", 8)

        # lock to prevent being messed up with mutiple processes
        quant_lock_path = self.quantized_model_path + ".lock"
        with FileLock(quant_lock_path):
            if not os.path.exists(self.quantized_model_path):  # quantized by other process already?
                self.cpp_module.quantize_model(
                    fp32_model_path,
                    quant_output_path,
                    weight_dtype,
                    alg,
                    group_size,
                    scale_dtype,
                    compute_dtype,
                    use_ggml,
                    threads,
                )
                os.rename(quant_output_tmp, self.quantized_model_path)

        assert os.path.exists(self.quantized_model_path), "Failed to quantize model"

    def load_model(self):
        self.native_model_ptr = ctypes.c_void_p(self.cpp_module.create_new_model())
        quantized_model_path = self.quantized_model_path.encode(_CTYPES_STR_ENCODE)
        memory_dtype = self.generation_args["memory_dtype"].encode(_CTYPES_STR_ENCODE)

        ok = self.cpp_module.init_model(
            self.native_model_ptr,
            quantized_model_path,
            self.generation_args["max_new_tokens"],
            self.generation_args["max_batch_size"],
            self.generation_args["ctx_size"],
            self.generation_args["pad_token"],
            memory_dtype,
            self.generation_args["scratch_size_ratio"],
            self.generation_args["threads"],
            self.generation_args["seed"],
        )
        if not ok:
            raise RuntimeError("Failed to initialize model. Please check native log for details.")

    def __call__(
        self,
        input_ids_dataptr: int,
        positions_dataptr: int,
        is_prompt: bool,
        block_tables_dataptr: int,
        slot_mapping_dataptr: int,
        context_lens: List[int],
    ) -> ctypes.c_void_p:
        context_lens_arr = (ctypes.c_int * len(context_lens))(*context_lens)
        return self.cpp_module.generate(
            self.native_model_ptr,
            ctypes.c_void_p(input_ids_dataptr),
            ctypes.c_void_p(positions_dataptr),
            ctypes.c_bool(is_prompt),
            ctypes.c_void_p(block_tables_dataptr),
            ctypes.c_void_p(slot_mapping_dataptr),
            context_lens_arr,
            len(context_lens),
        )

    def set_block_size(self, block_size: int):
        self.cpp_module.set_block_size(self.native_model_ptr, block_size)

    def free_slots(self, seq_ids: List[int]) -> bool:
        id_len = len(seq_ids)
        seq_id_array = None if id_len == 0 else (ctypes.c_int64 * id_len)(*seq_ids)
        return self.cpp_module.free_slots(self.native_model_ptr, seq_id_array, id_len // 2)

    def set_kv_caches_ptr(self, kv_caches_dataptr: int):
        self.cpp_module.set_kv_caches_ptr(self.native_model_ptr, ctypes.c_void_p(kv_caches_dataptr))

    def get_last_error(self) -> str:
        self.cpp_module.get_last_error(self.native_model_ptr)
