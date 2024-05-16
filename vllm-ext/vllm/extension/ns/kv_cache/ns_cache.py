from typing import List, Dict, Optional, Tuple, Set
import os
import torch
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, DeviceConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus
from vllm.sequence import Sequence, SequenceGroup
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE

from vllm.core.block_manager_v1 import BlockSpaceManagerV1

from vllm.extension import ns

from vllm.logger import init_logger

logger = init_logger(__name__)

_KV_CACHES: List[torch.Tensor] = None

class NSCPUCacheEngine:
    """Origin:
    Manages the KV cache for CPU backend.

    This class is responsible for initializing and managing CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as copying.

    New:
    ======Change to map vllm seq_id to native KV cache slot_id======
    KV cache is managed in native. 
    """

    def __init__(self, cache_config: CacheConfig, model_config: ModelConfig,
                 parallel_config: ParallelConfig,
                 device_config: DeviceConfig) -> None:
        assert device_config.device_type == "cpu"
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_kv_heads(parallel_config)

        # set 
        self.block_size = cache_config.block_size
        # Note: In CacheConfig, num_gpu_blocks actual is num_cpu_blocks
        # for CPU backend, because we want to reuse KV cache management
        # in the scheduler.
        self.num_cpu_blocks = cache_config.num_gpu_blocks

        # if cache_config.cache_dtype == "auto":
        #     self.dtype = model_config.dtype
        # else:
        #     self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        # int64 to hold slot_id, seq_id, vllm_reqeust_id ...
        self.dtype = torch.int64

        # Get attention backend.
        # self.attn_backend = get_attn_backend(model_config.dtype)

        # Initialize the cache.
        # Note: fake kv cache here. We only store native KV cache slot_id here
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks)
        global _KV_CACHES
        if _KV_CACHES:
            raise ValueError("KV cache is already initialized")
        _KV_CACHES = self.cpu_cache

    def _allocate_kv_cache(
        self,
        num_blocks: int,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on CPU."""
        # kv_cache_shape = self.attn_backend.get_kv_cache_shape(
        #     num_blocks, self.block_size, self.num_heads, self.head_size)

        # single tensor would be enough to store the sequence id/slot_id
        kv_cache_shape = (num_blocks, self.block_size, ns._KV_CACHE_LAST_DIM)
        kv_cache: List[torch.Tensor] = []
        # for _ in range(self.num_layers):
        kv_cache.append(
            torch.empty(kv_cache_shape, dtype=self.dtype, device="cpu"))

        return kv_cache

    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        raise NotImplementedError("Swap is not supported in CPUCacheEngine.")

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        raise NotImplementedError("Swap is not supported in CPUCacheEngine.")

    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        # self.attn_backend.copy_blocks(self.cpu_cache, src_to_dsts)
        pass

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig
    ) -> int:
        # head_size = model_config.get_head_size()
        # num_heads = model_config.get_num_kv_heads(parallel_config)
        # num_layers = model_config.get_num_layers(parallel_config)

        # key_cache_block = block_size * num_heads * head_size
        # value_cache_block = key_cache_block
        # total = num_layers * (key_cache_block + value_cache_block)

        # VLLM_CPU_KVCACHE_SPACE env and block_size are used to calculate number of cpu blocks
        # model_config.max_model_len must be no greater than block_size,
        # and  VLLM_CPU_KVCACHE_SPACE *_GB/(block_size*element_size) = num_cpu_blocks <= max_num_seqs as verfied in NSCPUCacheEngine
        # Otherwise, native kv cache may run out of slots.
        # Set VLLM_CPU_KVCACHE_SPACE to at least 1GB here before number of cpu blocks being caclulated in vllm. block_size
        # may be adjusted to meet the requirement.
        _GB = 1 << 30
        block_size = cache_config.block_size
        assert block_size >= model_config.max_model_len, "kv cache block size should be equal to max_model_len"
        space_key = "VLLM_CPU_KVCACHE_SPACE"
        space_value = block_size * scheduler_config.max_num_seqs * 4 # int32
        cache_config.cpu_kvcache_space_bytes = space_value
        os.environ[space_key] = str(float(space_value)/_GB)
        logger.info("reset cache_config.cpu_kvcache_space_bytes to %s GB", os.environ[space_key])
        
        total = block_size
        # if cache_dtype == "auto":
        #     dtype = model_config.dtype
        # else:
        #     dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        # dtype_size = torch.tensor([], dtype=dtype).element_size()

        # we use int32 to store native kv cache slot_id
        return 4 * total
    

class NSBlockSpaceManagerV1(BlockSpaceManagerV1):
    def __init__(self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = False):
        super().__init__(block_size, num_gpu_blocks, num_cpu_blocks, watermark, sliding_window, enable_caching)

        global _KV_CACHES
        if _KV_CACHES is None:
            raise ValueError("KV cache should be set in " + NSCPUCacheEngine.__name__)

        from vllm.extension import ns
        if ns._IE_MODEL is None:
            raise ValueError("vllm.extension.ns._IE_model should not be empty")
        self.ie_model = ns._IE_MODEL
        self.ie_model.set_block_size(block_size)
        self.ie_model.set_kv_caches_ptr(_KV_CACHES[0].data_ptr())
        
        # corresponding kv cache to be copied from. parent_seq_id -> (parent_seq, [child_seq])
        self.kv_cache_copy_waiting: Dict[int, Tuple[Sequence, List[Sequence]]] = {}
        # for quick lookup
        self.cp_child_id_to_parent_id: Dict[int, int] = {}
        # for deferred parent seq native slot free, also for checking if block table and its native slot can be reused
        # the parent seq id also in kv_cache_copy_waiting, but not vice-versa
        self.free_native_waiting: List[int] = []

    def allocate_block_table_for_seq(self, seq_id: int):
        block_table = self.gpu_allocator.allocate()
        self.block_tables[seq_id] = [block_table]
        return block_table.block_number

    # always return true since we use native slots which are preallocated
    # also do some bookkeeping for block table assignment and kv cache copy and free native slot of deferred parent seqs
    def can_append_slots(self,
                         seq_group: SequenceGroup,
                         num_lookahead_slots: int = 0) -> bool:
        # collect all parent seq ids
        parent_seq_ids: Set[int] = set()
        for seq_id in seq_group.seqs_dict.keys():
            if seq_id in self.cp_child_id_to_parent_id:
                parent_seq_ids.add(self.cp_child_id_to_parent_id.pop(seq_id))
        # copy kv cache if needed
        # kv cache usage
        # 0 0 -> seq_id
        # 0 1 -> slot_id, will be set in native
        # 1 0 -> has parent sequence, yes: -1, no: 0
        # 1 1 -> if has parent sequence (-1), parent seq_id
        # 2 0 -> if kv cache copied, yes: -1, no: 0
        # 2 1 -> if has parent sequence (-1), parent slot_id
        # 3 0 -> beam_size if it's beam search. should be greater than 1
        # 3 1 -> vllm request idx

        global _KV_CACHES
        kv_cache = _KV_CACHES[0]
        for parent_seq_id in parent_seq_ids:
            if parent_seq_id not in self.kv_cache_copy_waiting:
                continue
            # free native slot of deferred parent_seq and copy kv cache if needed
            _, child_seqs = self.kv_cache_copy_waiting.pop(parent_seq_id)
            parent_block_nbr = None
            parent_slot_id = None
            for child_seq in child_seqs:
                if child_seq.seq_id not in self.block_tables: # make sure child seq is still valid
                    continue
                # reuse parent block nbr and slot id if prarent seq's native slot going to be freed
                # otherwise, get new block table
                child_block_nbr = self.block_tables[child_seq.seq_id][0].block_number
                # child seq and parent seq share same block nbr initially
                if parent_block_nbr is None:
                    parent_block_nbr = child_block_nbr
                    parent_slot_id = kv_cache[child_block_nbr][0][1]
                # reuse parent block nbr, no copy needed, only free parent seq
                if parent_seq_id in self.free_native_waiting:
                    # no need to free native slot since the slot will be reused here
                    self.free_native_waiting.remove(parent_seq_id)
                    # reuse parent_block_nbr, no copy needed
                    kv_cache[child_block_nbr][0][1] = parent_slot_id
                    kv_cache[child_block_nbr][2][0] = ns._KV_CACHE_MARK_YES
                else: # assign new block nbr and copy kv cache
                    # call parent class's free since it's new child sequence without native slot associated
                    super().free(child_seq)
                    # allocate new one
                    child_block_nbr = self.allocate_block_table_for_seq(child_seq.seq_id)
                    # need to copy kv cache
                    kv_cache[child_block_nbr][0][1] = 0
                    kv_cache[child_block_nbr][2][0] = ns._KV_CACHE_MARK_NO

                # check seq_id is correct in execute_model
                kv_cache[child_block_nbr][0][0] = child_seq.seq_id
                kv_cache[child_block_nbr][1][0] = ns._KV_CACHE_MARK_YES
                kv_cache[child_block_nbr][1][1] = parent_seq_id
                kv_cache[child_block_nbr][2][1] = parent_slot_id # need to copy kv cache in native
            # cannot reset kv cache here since the blocks may be already used by child seqs
            # it's ok not resetting kv cache since all control elements are set specifically in above logic and in execute_model (beam size and vllm group request idx)
        return True
    
    # return empty dict since no slot to append
    def append_slots(self, seq: Sequence, num_lookahead_slots: int = 0) -> Dict[int, List[int]]:
        if seq.seq_id in self.cp_child_id_to_parent_id or seq.seq_id in self.kv_cache_copy_waiting:
            raise ValueError("seq should not be in cp_child_id_to_parent_id or kv_cache_copy_waiting")
        return {}

    def remove_seq_from_block_tables(self, seq_id):
        block_table = self.block_tables[seq_id][0]
        block_table.ref_count -= 1
        del self.block_tables[seq_id]

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        super().fork(parent_seq, child_seq)

        # one block per sequence
        parent_block_nbr = self.block_tables[parent_seq.seq_id][0].block_number
        child_block_nbr = self.block_tables[child_seq.seq_id][0].block_number
        assert parent_block_nbr == child_block_nbr, "child block nbr should be equal to parent block nbr"

        # add to kv cache copy waiting which will be copied in "can_append_slots"
        self.cp_child_id_to_parent_id[child_seq.seq_id] = parent_seq.seq_id
        if parent_seq.seq_id in self.kv_cache_copy_waiting:
            self.kv_cache_copy_waiting[parent_seq.seq_id][1].append(child_seq)
        else:
            self.kv_cache_copy_waiting[parent_seq.seq_id] = (parent_seq, [child_seq])

    def free(self, seq: Sequence) -> None:
        # free native slot when no need to copy kv cache
        # otherwise, do actual native slot free in "can_append_slot" which is called in scheduling running seqs.
        if seq.seq_id not in self.kv_cache_copy_waiting:
            block_nbr = self.block_tables[seq.seq_id][0].block_number
            global _KV_CACHES
            kv_cache = _KV_CACHES[0]
            # free native slot
            if not self.ie_model.free_slots([seq.seq_id, kv_cache[block_nbr][3][1]]):
                raise ValueError("cannot free slot for seq")
            kv_cache[block_nbr][0:ns._KV_CACHE_ELEMENT_USED][:] = 0 # other elements are not used
        else:
            self.free_native_waiting.append(seq.seq_id)
        
        # free block table
        super().free(seq)
