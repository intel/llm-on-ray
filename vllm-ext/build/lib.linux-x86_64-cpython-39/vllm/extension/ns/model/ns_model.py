from typing import List, Optional
import sys
import ctypes
import torch
import numpy as np
from torch import nn

from transformers import PretrainedConfig
from transformers import LlamaConfig

from vllm.config import CacheConfig, LoRAConfig, ModelConfig, SchedulerConfig, ParallelConfig
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.attention import AttentionMetadata

from inference_engine import Model as IE_Model

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.sequence import SamplerOutput, SequenceGroupMetadata

from vllm.extension import ns

class NSModel(nn.Module):
    def __init__(self,
                 config: PretrainedConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 lora_config: Optional[LoRAConfig] = None):
        super(NSModel, self).__init__()
        self.config = config
        self.quant_config = quant_config
        self.lora_config = lora_config
        self.ie_model = None
        self.tokenizer = None

        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List[torch.Tensor],
                attn_metadata: AttentionMetadata):
        assert len(kv_caches) == 1, "kv_caches should have 1 element here"
        # use data_ptr and torch type in str to avoid inference engine model depends on pytorch and vllm types
        # kv_cache pointer is set in NSBlockManager in advance
        return self.ie_model(input_ids.data_ptr(),
                             positions.data_ptr(),
                             attn_metadata.is_prompt,
                             attn_metadata.block_tables.data_ptr(),
                             attn_metadata.slot_mapping.data_ptr(),
                             attn_metadata.seq_lens
                             )
    
    def init_inference_engine(self, model_config: ModelConfig, parallel_config: ParallelConfig, scheduler_config: SchedulerConfig):
        self.ie_model = IE_Model(self.config.name_or_path, max_batch_size=scheduler_config.max_num_seqs, ctx_size=model_config.max_model_len, max_new_tokens=model_config.max_model_len)
        self.tokenizer = self.ie_model.tokenizer

    def load_weights(self, weights):
        assert sum(1 for _ in weights) > 0
        
        qc = self.quant_config
        self.ie_model.check_and_quantize(weight_dtype=qc.weight_dtype,
                                         alg=qc.alg,
                                         group_size=qc.group_size,
                                         scale_dtype=qc.scale_dtype,
                                         compute_dtype=qc.compute_dtype,
                                        )
        self.ie_model.load_model()
        if ns._IE_MODEL:
            raise ValueError("vllm.extension.ns._IE_model should be empty")
        ns._IE_MODEL = self.ie_model
    
class NSLLamaModel(NSModel):
    def __init__(self,
                 config: LlamaConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 lora_config: Optional[LoRAConfig] = None):
        super().__init__(config, cache_config, quant_config, lora_config)

def native_ptr_to_tensor(hidden_states_ptr, seq_len_sum, hidden_size):
    data = ctypes.cast(hidden_states_ptr, ctypes.POINTER(ctypes.c_float))
    data_array = np.ctypeslib.as_array(data, shape=(seq_len_sum * hidden_size,))
    return torch.frombuffer(data_array, dtype=torch.float).view(seq_len_sum, hidden_size)

def get_vllm_reqidx(request_id: str):
    id = 0
    if len(request_id) == 32: # 128-bit UUID
        id = int(request_id, 16)
        # take the first 63 bit to make sure it's less than sys.maxsize. There is very tiny chance id is not unique.
        # But we have logic in native code to check if the id is unique in running requests when prefill.
        id = id >> 65
    else:
        id = int(request_id)
        if id > sys.maxsize:
            raise ValueError(f"request_id {id} exeeds max size of int64 {sys.maxsize}. Please reset the request_id counter and resubmit.")
        
    return id

def set_vllm_reqidx(kv_cache: torch.Tensor, beam_search: bool, block_nbr: int, best_of: int, request_id: str):
    # for assign pre-allocated slots to reduce kv cache copy length
    if beam_search:
        if kv_cache[block_nbr][3][0] == 0: # not set yet
            kv_cache[block_nbr][3][0] = best_of
            kv_cache[block_nbr][3][1] = get_vllm_reqidx(request_id)
    else:
        kv_cache[block_nbr][3][0] = 0 # beam size
        kv_cache[block_nbr][3][1] = -1

def set_more_metadata(attn_metadata, kv_cache: torch.Tensor, seq_group_metadata_list: List[SequenceGroupMetadata]):
    # kv cache usage
    # 0 0 -> seq_id
    # 0 1 -> slot_id, will be set in native
    # 1 0 -> has parent sequence, yes: -1, no: 0
    # 1 1 -> if has parent sequence (-1), parent seq_id
    # 2 0 -> if kv cache copied, yes: -1, no: 0
    # 2 1 -> if has parent sequence (-1), parent slot_id
    # 3 0 -> beam_size if it's beam search. should be greater than 1
    # 3 1 -> vllm request idx

    # set seq id to first element of block in kv cache
    # one sequence one block
    if attn_metadata.is_prompt:
        block_tables = torch.zeros((len(seq_group_metadata_list)), dtype=torch.int)
        i = 0
        for seq_g_meta in seq_group_metadata_list:
            beam_search = seq_g_meta.sampling_params.use_beam_search
            for seq_id, block_nbrs in seq_g_meta.block_tables.items():
                block_nbr = block_nbrs[0]
                kv_cache[block_nbr][0][0] = seq_id
                set_vllm_reqidx(kv_cache, beam_search, block_nbr, seq_g_meta.sampling_params.best_of, seq_g_meta.request_id)
                block_tables[i] = block_nbr
                i = i + 1
        assert i == block_tables.shape[0], "inconsistent block tables and sequences"
        attn_metadata.block_tables = block_tables
    else:
        prompt_lens: List[int] = []
        for seq_g_meta in seq_group_metadata_list:
            beam_search = seq_g_meta.sampling_params.use_beam_search
            for seq_id, block_nbrs in seq_g_meta.block_tables.items():
                block_nbr = block_nbrs[0]
                set_vllm_reqidx(kv_cache, beam_search, block_nbr, seq_g_meta.sampling_params.best_of, seq_g_meta.request_id)
                # check if seq_id matches
                assert seq_id == kv_cache[block_nbr][0][0], "seq_ids in metadata and kv_caches not match"
                prompt_lens.append(seq_g_meta.seq_data[seq_id].get_prompt_len())
        attn_metadata.block_tables = attn_metadata.block_tables.squeeze(1) # we only have one block per sequence
        attn_metadata.seq_lens = prompt_lens

# modified execute_model in cpu_model_runner.py to pass sequence_id and convert tensor to int32 for now
def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        kv_caches: List[torch.Tensor],
    ) -> Optional[SamplerOutput]:
        (input_tokens, input_positions, attn_metadata, sampling_metadata, multi_modal_input
         ) = self.prepare_input_tensors(seq_group_metadata_list)

        set_more_metadata(attn_metadata, kv_caches[0], seq_group_metadata_list)

        model_executable = self.model
        execute_model_kwargs = {
            "input_ids": input_tokens.to(torch.int32),
            "positions": input_positions,
            "kv_caches": kv_caches,
            "attn_metadata": attn_metadata,
        }

        if self.vision_language_config:
            execute_model_kwargs.update({"image_input": multi_modal_input})

        hidden_states_ptr = model_executable(**execute_model_kwargs)
        if not hidden_states_ptr:
            raise RuntimeError("Failed to execute model. Details: " + model_executable.ie_model.get_last_error())

        hidden_states = native_ptr_to_tensor(hidden_states_ptr, input_tokens.shape[0], self.model_config.hf_config.hidden_size)

        hidden_states = hidden_states.to(self.model_config.dtype)

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states, sampling_metadata)

        # Only perform sampling in the driver worker.
        if not self.is_driver_worker:
            return None

        # Sample the next token.
        output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )

        return output