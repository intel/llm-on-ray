//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
// Defines sigaction on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <random>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <utility>
#include <vector>
#include <memory>

#include "common.h"
#include "core/layers/bestla_common.hpp"
#include "core/layers/bestla_gemm.h"
#include "bestla/bestla_parallel.h"
#include "models/model_utils/model_types.h"
#include "models/model_utils/model_config.h"
#include "models/model_utils/model_utils.h"
#include "models/model_utils/quant_utils.h"

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <signal.h>
#include <windows.h>
#endif


static const char * ERROR_MESSAGE_SLOT = "ERROR: no free slot available or illegal state for seq id: ";
static const char * ERROR_MESSAGE_DUPLICATE_REQID = "ERROR: duplicate request index!\n";

using namespace std;

static std::set<model_archs> cont_batching_model_archs = {MODEL_GPTJ, MODEL_LLAMA};
void init_model_params(model_params* params, const std::string& model_path, int max_new_tokens = -1, int max_batch_size = 512,
                     int ctx_size = 512, model_vocab::id pad_token = -1, const std::string& memory_dtype = "auto",
                     const float& scratch_size_ratio = 1.0f, int threads = 8, int seed = -1) {
  MODEL_ASSERT(params != nullptr);
#ifdef MODEL_NAME
  params->model_name = MODEL_NAME;
#endif
  params->model_arch = model_name_to_arch::init().find(params->model_name);
  params->model = model_path;
  params->n_predict = max_new_tokens;
  params->max_batch_size = max_batch_size;
  params->n_ctx = ctx_size;
  params->seed = seed;
  params->n_threads = threads;
  
  if (memory_dtype == "f32")
    params->memory_type = KV_MEM_TYPE_F32;
  else if (memory_dtype == "f16")
    params->memory_type = KV_MEM_TYPE_F16;
  else if (memory_dtype == "auto")
    params->memory_type = KV_MEM_TYPE_AUTO;
  else
    fprintf(stderr, "Unexpected memory dtype %s!", memory_dtype.c_str());

  if (cont_batching_model_archs.count(params->model_arch) == 0) {
    fprintf(stderr, "unsupported model %d!", params->model_arch);
    exit(0);
  }
  params->do_early_stopping = true;
  params->scratch_size_ratio = scratch_size_ratio;

  // TODO(Yi): MHA FOR LONG TOKENS
  int32_t tokens_length = 6144;
  if (params->n_ctx > tokens_length) {
    params->memory_type = KV_MEM_TYPE_F16;
  }

  if (params->seed < 0) {
    params->seed = time(nullptr);
  }

  printf(
      "Model Parameters in cpp. model_name: %s, model_arch: %d, n_predict: %d, max_batch_size: %d, n_ctx: %d, memory type: %d, "
      "early_stopping: %d, scratch_size_ratio: %.3f, threads: %d, seed: %d\n",
      params->model_name.c_str(), params->model_arch, params->n_predict, params->max_batch_size, params->n_ctx, params->memory_type,
      params->do_early_stopping, params->scratch_size_ratio, params->n_threads, params->seed);
}

std::shared_ptr<quant_layer_base> get_model_quant_layer(const std::string model_name) {
  return ql_registry::create_ql(model_name);
}

#define STATIC_INPUT_HEAD_IDX 0

#define KV_CACHE_MARK_YES         -1
#define KV_CACHE_MARK_NO          0
#define KV_CACHE_LAST_DIM         2
#define KV_CACHE_CPY_PARAMS_SIZE  5
#define KV_CACHE_ELEMENT_USED     4


// kv cache usage
// 0 0 -> seq_id
// 0 1 -> slot_id, will be set in native
// 1 0 -> has parent sequence, yes: -1, no: 0
// 1 1 -> if has parent sequence (-1), parent seq_id
// 2 0 -> if kv cache copied, yes: -1, no: 0
// 2 1 -> if has parent sequence (-1), parent slot_id
// 3 0 -> beam_size if it's beam search. should be greater than 1
// 3 1 -> vllm request idx

struct slot_mapping {
  int64_t seq_id; // -1 if not used
  int slot_id; // AKA. req_idx
};

class SlotManager {
 public:
  SlotManager(struct model_context * ctx, int max_batch_size) : ctx(ctx), max_batch_size(max_batch_size) {
    free_req_idx.resize(max_batch_size, true);
  }

  int64_t calculate_kv_cache_idx(int block_nbr) {
    return kv_cache_block_size * block_nbr * KV_CACHE_LAST_DIM;
  }

  /**
   * @brief Free a slot by request index as well as vllmgroup request index association if any
   * @param seq_id: sequence id
   * @param vllmgroup_reqidx: vllmgroup request index
   * @return 0 if success, -1 if failed
  */
  int free_slot(int64_t seq_id, int64_t vllmgroup_reqidx) {
    if (vllmseqid_to_reqidx.find(seq_id) == vllmseqid_to_reqidx.end()) {
      fprintf(stderr, "ERROR: seq_id %ld not found in vllmseqid_to_reqidx, nothing to free!\n", seq_id);
      return -1;
    }
    int req_idx = vllmseqid_to_reqidx[seq_id];
    // if vllmgroup_reqidx not exists
    if (vllmgroup_reqidx_to_reqindices.find(vllmgroup_reqidx) == vllmgroup_reqidx_to_reqindices.end()) {
      free_req_idx[req_idx] = true;
      vllmseqid_to_reqidx.erase(seq_id);
      return 0;
    }
    // group assigned slots
    std::vector<slot_mapping>& slots = vllmgroup_reqidx_to_reqindices[vllmgroup_reqidx];
    bool all_freed = true;
    for (auto & ss : slots) {
      if (ss.slot_id == req_idx) {
        ss.seq_id = -1;
      }
      all_freed &= (ss.seq_id == -1);
    }
    if (all_freed) { // beam search done. free all slots. If beam search is not done, at lease one slot occupied
      for (auto & ss : slots) {
        free_req_idx[ss.slot_id] = true;
        vllmseqid_to_reqidx.erase(ss.seq_id);
      }
      vllmgroup_reqidx_to_reqindices.erase(vllmgroup_reqidx);
    }
    return 0;
  }

  /**
   * @brief Assign a slot for a given seq_id or multiple slots for vllmgroup reqidx for beam search
   * @param kv_cache_idx: kv cache index based on block_nbr
   * @param is_prompt: if it's prompt decoding, true; otherwise, false. It's for checking if there is duplicate vllmgroup_reqidx
   * @return slot index or first assigned slot index if it's beam search in prefill phase
  */
  int assign_slot(int64_t kv_cache_idx, bool is_prompt) {
    int64_t kv_cache_beam_idx = kv_cache_idx + 3 * KV_CACHE_LAST_DIM; // is beam search, kv_caches[block_nbr][3][0]

    int64_t seq_id = kv_caches[kv_cache_idx];
    int beam_size = (int)kv_caches[kv_cache_beam_idx];
    int64_t vllmgroup_reqidx = kv_caches[kv_cache_beam_idx + 1]; // kv_caches[block_nbr][3][1] is the vllmgroup_reqidx
    if (beam_size > 1) { // beam search case
      return assign_group_slots(vllmgroup_reqidx, seq_id, kv_cache_idx, beam_size, is_prompt);
    }
    int req_idx = query_free_req_idx();
    if (req_idx == -1) {
      fprintf(stderr, "ERROR: no free slot available for seq id %ld!\n", seq_id);
      return -1;
    }
    vllmseqid_to_reqidx.emplace(seq_id, req_idx); // kv_caches[block_nbr][0][0] is the vllmseqid
    kv_caches[kv_cache_idx + 1] = req_idx; // kv_caches[block_nbr][0][1] is the req_idx
    return req_idx;
  }

  /**
   * @brief Get slot for a sequence which has some metadata stored in kv cache entry identified by kv_cache_idx.
   * If it's prompt decoding, a new slot or multiple slots (beam search) will be assigned. Otherwise, get assigned slot from seq_id -> slot_id map if it's non-beam search.
   * For beam search in next decoding phase, get slot from group assigned slots to avoid copying all kv cache content from parent cache since prompt tokens kv caches are same across the group.
  */
  int get_slot(int64_t kv_cache_idx, int prompt_len, int total_len) {
    int64_t seq_id = kv_caches[kv_cache_idx]; // kv_caches[block_nbr][0][0] is the vllmseqid
    int req_idx;
    // get existing slot id or assign new slot
    if (vllmseqid_to_reqidx.find(seq_id) != vllmseqid_to_reqidx.end()) { // existing slot
      req_idx = vllmseqid_to_reqidx[seq_id];
      MODEL_ASSERT(req_idx == kv_caches[kv_cache_idx + 1]);
      return req_idx;
    }
    int64_t kv_cache_parent_idx = kv_cache_idx + 1 * KV_CACHE_LAST_DIM; // has parent_seq, kv_caches[block_nbr][1][0]
    int64_t kv_cache_cpy_idx = kv_cache_idx + 2 * KV_CACHE_LAST_DIM; // is kv cache copied, kv_caches[block_nbr][2][0]
    if ((kv_caches[kv_cache_parent_idx] == KV_CACHE_MARK_YES) && (kv_caches[kv_cache_cpy_idx] == KV_CACHE_MARK_NO)) { // has parent seq, but kv cache not copied yet
      // parent seqs were not freed from vllm call. so it's ok to copy from parent.
      return copy_kv_cache(kv_cache_idx, prompt_len, total_len);
    }
    
    if ((kv_caches[kv_cache_parent_idx] == KV_CACHE_MARK_YES) && (kv_caches[kv_cache_cpy_idx] == KV_CACHE_MARK_YES)) { // slot of parent seq reused
      int64_t parent_seq_id = kv_caches[kv_cache_parent_idx + 1]; // kv_caches[block_nbr][1][1] is the parent_seq_id
      return transfer_slot(parent_seq_id, seq_id, kv_caches[kv_cache_idx + 3 * KV_CACHE_LAST_DIM + 1], kv_cache_idx); // kv_caches[block_nbr][3][1] is the vllmgroup_reqidx
    }
    return assign_slot(kv_cache_idx, false);
  }

  int64_t get_seq_id(int block_nbr) {
    return kv_caches[block_nbr * kv_cache_block_size * KV_CACHE_LAST_DIM];
  }

  void set_block_size(int64_t block_size) {
    kv_cache_block_size = block_size;
  }

  void set_kv_caches_ptr(void * kv_caches_ptr) {
    kv_caches = static_cast<int64_t *>(kv_caches_ptr);
  }

 private:
  int max_batch_size;
  struct model_context * ctx = nullptr;
  std::vector<bool> free_req_idx;
  std::unordered_map<int64_t, int> vllmseqid_to_reqidx;
  // for fixed beam-width slot allocation
  std::unordered_map<int64_t, std::vector<slot_mapping>> vllmgroup_reqidx_to_reqindices; // via vllmreqidx -> multiple seqs -> reqidices

  int64_t kv_cache_block_size;
  int64_t * kv_caches;

  /**
   * @brief Query a free request index and set it to be used
  */
  int query_free_req_idx() {
    auto iter = std::find_if(free_req_idx.begin(), free_req_idx.end(), [](const bool flag) { return flag; });
    if (iter != free_req_idx.end()) {
      int idx = std::distance(free_req_idx.begin(), iter);
      free_req_idx[idx] = false;
      return idx;
    }
    return -1;
  }

  /**
   * @brief Assign slots for a group of seqs for beam search in prefill phase. Or assign a slot from group assinged slots for a given seq_id in decoding phase
   * @param vllmgroup_reqidx: vllmgroup request index
   * @param seq_id: sequence id
   * @param kv_cache_idx: kv cache index based on block_nbr
   * @param beam_size: beam size
   * @param is_prompt: if it's prompt decoding, true; otherwise, false. It's for checking if there is duplicate vllmgroup_reqidx
   * @return slot index or first assigned slot index if it's beam search in prefill phase
  */
  int assign_group_slots(int64_t vllmgroup_reqidx, int64_t seq_id, int64_t kv_cache_idx, int beam_size, bool is_prompt) {
    if (vllmgroup_reqidx_to_reqindices.find(vllmgroup_reqidx) == vllmgroup_reqidx_to_reqindices.end()) { // prefill phase
      // avoid vector copy
      vllmgroup_reqidx_to_reqindices.emplace(vllmgroup_reqidx, std::vector<slot_mapping>());
      std::vector<slot_mapping> & slots = vllmgroup_reqidx_to_reqindices[vllmgroup_reqidx];
      slots.reserve(beam_size);
      for (int i = 0; i < beam_size; i++) {
        int req_idx = query_free_req_idx();
        if (req_idx == -1) { // no enough slots, rollback assignments
          rollback_group_assignment(vllmgroup_reqidx);
          fprintf(stderr, "ERROR: no free slot available for group slots assignment, %ld!\n", vllmgroup_reqidx);
          return -1;
        }
        slots.push_back(slot_mapping{-1, req_idx});
      }
      // return first available slot id
      slot_mapping & ss = slots[0];
      occupy_group_slot(ss, seq_id, kv_cache_idx);
      return ss.slot_id;
    }
    // decoding phase
    if (is_prompt) {
      fprintf(stderr, ERROR_MESSAGE_DUPLICATE_REQID);
      return -2;
    }
    std::vector<slot_mapping>& slots = vllmgroup_reqidx_to_reqindices[vllmgroup_reqidx];
    for (auto & slot : slots) {
      if (slot.seq_id == -1) {
        occupy_group_slot(slot, seq_id, kv_cache_idx);
        return slot.slot_id;
      }
    }
    fprintf(stderr, "ERROR: no free slot available from already assigned slots, seq id is %ld, vllmgroup reqidx is %ld!\n", seq_id, vllmgroup_reqidx);
    return -1;
  }

  void occupy_group_slot(slot_mapping & ss, int64_t seq_id, int64_t kv_cache_idx) {
    ss.seq_id = seq_id;
    // int kv_cache_idx = block_nbr * kv_cache_block_size * KV_CACHE_LAST_DIM;
    vllmseqid_to_reqidx.emplace(seq_id, ss.slot_id); // kv_caches[block_nbr][0][0] is the vllmseqid
    kv_caches[kv_cache_idx + 1] = ss.slot_id; // kv_caches[block_nbr][0][1] is the req_idx
  }

  void rollback_group_assignment(int64_t vllmgroup_reqidx) {
    std::vector<slot_mapping>& slots = vllmgroup_reqidx_to_reqindices[vllmgroup_reqidx];
    for (auto & ss : slots) {
      free_req_idx[ss.slot_id] = true;
    }
    vllmgroup_reqidx_to_reqindices.erase(vllmgroup_reqidx);
  }

  /**
   * @brief Transfer slot from parent seq to current seq
   * @param parent_seq_id: parent sequence id
   * @param seq_id: sequence id
   * @param vllmgroup_reqidx: vllmgroup request index
   * @param kv_cache_idx: kv cache index based on block_nbr
   * @return slot index
  */
  int transfer_slot(int64_t parent_seq_id, int64_t seq_id, int64_t vllmgroup_reqidx, int64_t kv_cache_idx) {
    if (vllmseqid_to_reqidx.find(parent_seq_id) == vllmseqid_to_reqidx.end()) {
      fprintf(stderr, "ERROR: parent_seq_id %ld not found in vllmseqid_to_reqidx, nothing to transfer!\n", parent_seq_id);
      return -1;
    }
    if (vllmgroup_reqidx_to_reqindices.find(vllmgroup_reqidx) == vllmgroup_reqidx_to_reqindices.end()) {
      fprintf(stderr, "ERROR: vllmgroup_reqidx %ld not existed, illegal state!\n", vllmgroup_reqidx);
      return -1;
    }
    // slot_mapping are not changed, only vllmseqid_to_reqidx is updated
    int req_idx = vllmseqid_to_reqidx[parent_seq_id];
    vllmseqid_to_reqidx.erase(parent_seq_id);
    vllmseqid_to_reqidx.emplace(seq_id, req_idx);
    kv_caches[kv_cache_idx + 1] = req_idx;
    return req_idx;
  }

  /**
   * @brief Copy kv cache from parent seq to current seq. If it's prompt decoding, all kv cache content will be copied. Otherwise, only the kv caches of new tokens will be copied.
   * @param kv_cache_idx: kv cache index based on block_nbr
   * @param prompt_len: prompt length
   * @param total_len: total length
   * @return slot index
  */
  int copy_kv_cache(int64_t kv_cache_idx, int prompt_len, int total_len) {
    int64_t kv_cache_cpy_idx = kv_cache_idx + 2 * KV_CACHE_LAST_DIM; // is kv cache copied, kv_caches[block_nbr][2][0]
    MODEL_ASSERT(kv_caches[kv_cache_cpy_idx] == KV_CACHE_MARK_NO);

    // assign slot and copy kv cache
    int req_idx = assign_slot(kv_cache_idx, false);
    if (req_idx >= 0) {
      int parent_req_idx = kv_caches[kv_cache_cpy_idx + 1]; // kv_caches[block_nbr][2][1] is the parent slot id
      // copy kv cache
      int start_pos = prompt_len == total_len ? 0 : prompt_len;
      model_kv_cache_seq_cpy(ctx, parent_req_idx, req_idx, start_pos, total_len);
      // mark kv cache copied
      kv_caches[kv_cache_cpy_idx] = KV_CACHE_MARK_YES;
    }

    return req_idx;
  }
};

class Model {

 public:
  Model() { model_init_backend(); }

  ~Model() {
    if (ctx) model_free(ctx);
    if (slot_manager) delete slot_manager;
  }

  bool init_model(const std::string& model_path, int max_new_tokens, int max_batch_size, int ctx_size, model_vocab::id pad_token,
                  const std::string& memory_dtype, const float& scratch_size_ratio, int threads, int seed);

  model_token get_eos_id() { return ctx->vocab.eos_token_id; }

  void * generate(void * input_ids_ptr,
                  void * positions_dataptr,
                  bool is_prompt,
                  void * block_tables_ptr,
                  void * slot_mapping_ptr,
                  int * prompt_lens_arr, int n_prompts);

  void set_block_size(int64_t block_size) { slot_manager->set_block_size(block_size); }

  void set_kv_caches_ptr(void * kv_caches_ptr) { slot_manager->set_kv_caches_ptr(kv_caches_ptr); }

  bool free_slots(int64_t * seq_id, int nbr_of_ids);

  const char * get_last_error() const {
    return last_error.c_str();
  }

  static int quantize_model(const std::string& model_path, const std::string& out_path, const std::string& weight_dtype,
                         const std::string& alg, int group_size, const std::string& scale_dtype,
                         const std::string& compute_dtype, bool use_ggml, int threads);

  void print_time() { model_print_timings(ctx); }

  void reset_time() { model_reset_timings(ctx); }

 private:
  model_context* ctx = nullptr;
  model_params params;
  SlotManager * slot_manager = nullptr;

  std::string last_error;
};

bool Model::init_model(const std::string& model_path, int max_new_tokens, int max_batch_size, int ctx_size, model_vocab::id pad_token,
                       const std::string& memory_dtype, const float& scratch_size_ratio, int threads, int seed) {
  try {
    init_model_params(&params, model_path, max_new_tokens, max_batch_size, ctx_size, pad_token, memory_dtype, scratch_size_ratio,
                      threads, seed);
    ctx = create_model_context(params);
    if (pad_token != -1) ctx->vocab.pad_token_id = pad_token;

    slot_manager = new SlotManager(ctx, max_batch_size);
  } catch (std::exception & e) {
    last_error = "ERROR: model initialization failed! ";
    fprintf(stderr, "%s %s\n", last_error, e.what());
    return false;
  }
  return true;
}
 
void * Model::generate(void * input_ids_ptr,
                void * positions_ptr,
                bool is_prompt,
                void * block_tables_ptr,
                void * slot_mapping_ptr,
                int * n_contexts_arr, int n_contexts) {
  int32_t * input_ids = static_cast<int32_t *>(input_ids_ptr);
  int64_t * positions_data = static_cast<int64_t *>(positions_ptr);
  int * block_tables = static_cast<int *>(block_tables_ptr);
  int64_t * slot_mapping = static_cast<int64_t *>(slot_mapping_ptr);

  // prepare inputs
  std::vector<model_input> inputs;
  inputs.reserve(n_contexts);
  if (is_prompt) {
    int input_offset = 0;
    for (int i = 0; i < n_contexts; ++i) {
      // assign slot based on parameters in kv_cache, like seq_id, beam_size, vllmgroup_reqidx
      // must assign slot here since it's prompt decoding
      int64_t kv_cache_idx = slot_manager->calculate_kv_cache_idx(block_tables[i]); // kv_caches[block_nbr][0][0] is the vllmseqid
      int req_idx = slot_manager->assign_slot(kv_cache_idx, true);
      if (req_idx < 0) {
        std::ostringstream oss;
        construct_message(oss, ERROR_MESSAGE_SLOT, slot_manager->get_seq_id(block_tables[i]), "!\n",
                          req_idx == -2 ? ERROR_MESSAGE_DUPLICATE_REQID : "");
        last_error = oss.str();
        fprintf(stderr, last_error.c_str());
        return nullptr;
      }
      inputs.push_back(model_input{
          /*.tokens           =*/input_ids + input_offset,
          /*.n_tokens         =*/(uint32_t)n_contexts_arr[i],
          /*.n_prompt_tokens  =*/(uint32_t)n_contexts_arr[i],
          /*.n_past           =*/0,
          /*.n_total          =*/0,
          /*.request_idx      =*/req_idx,
          /*.beam_idx         =*/0,
          /*.padding_side     =*/0,
          /*n_padding         =*/0,
      });
      input_offset += n_contexts_arr[i];
    }
  } else { // next decoding
    for (int i = 0; i < n_contexts; ++i) {
      int64_t kv_cache_idx = slot_manager->calculate_kv_cache_idx(block_tables[i]); // kv_caches[block_nbr][0][0] is the vllmseqid
      int req_idx = slot_manager->get_slot(kv_cache_idx, n_contexts_arr[i], positions_data[i]);
      if (req_idx < 0) {
        std::ostringstream oss;
        construct_message(oss, ERROR_MESSAGE_SLOT, slot_manager->get_seq_id(block_tables[i]), "!\n",
                          req_idx == -2 ? ERROR_MESSAGE_DUPLICATE_REQID : "");
        last_error = oss.str();
        fprintf(stderr, last_error.c_str());
        return nullptr;
      }
      inputs.push_back(model_input{
          /*.tokens           =*/input_ids + i,
          /*.n_tokens         =*/1,
          /*.n_prompt_tokens  =*/(uint32_t)n_contexts_arr[i],
          /*.n_past           =*/(uint32_t)positions_data[i],
          /*.n_total          =*/(uint32_t)positions_data[i],
          /*.request_idx      =*/req_idx,
          /*.beam_idx         =*/0,
          /*.padding_side     =*/0,
          /*n_padding         =*/0,
      });
    }
  }
  ctx->batch_size = inputs.size();
  if (model_eval(ctx, inputs.data(), inputs.size(), params.n_threads) > 0) {
    last_error = "ERROR: model_eval failed!\n";
    fprintf(stderr, last_error.c_str());
    return nullptr;
  }
  return ctx->last_hidden_states.data();
}

bool Model::free_slots(int64_t * seq_ids, int nbr_of_ids) {
  int sum = 0;
  for (int i = 0; i < nbr_of_ids; i++) {
    int64_t seq_id = seq_ids[i];
    int64_t vllmgroup_reqidx = seq_ids[i + 1];
    sum += slot_manager->free_slot(seq_id, vllmgroup_reqidx);
  }
  return sum == 0;
}

int Model::quantize_model(const std::string& model_path, const std::string& out_path, const std::string& weight_dtype,
                       const std::string& alg, int group_size, const std::string& scale_dtype,
                       const std::string& compute_dtype, bool use_ggml, int threads) {
  quant_params q_params;
#ifdef MODEL_NAME
  q_params.model_name = MODEL_NAME;
#endif
  model_archs mt = model_name_to_arch::init().find(q_params.model_name);
  if (mt == MODEL_UNKNOWN) {
    fprintf(stderr, "error, please set model_name \n");
    exit(0);
  }
  q_params.model_arch = mt;
  q_params.model_file = model_path;
  q_params.out_file = out_path;
  q_params.weight_dtype = weight_dtype;
  q_params.alg = alg;
  q_params.group_size = group_size;
  q_params.scale_dtype = scale_dtype;
  q_params.compute_dtype = compute_dtype;
  q_params.use_ggml = use_ggml;
  q_params.nthread = threads;

  ne_ftype ftype = quant_params_to_ftype(q_params);
  printf("ne_ftype: %d\n", ftype);

  auto quant_layer = get_model_quant_layer(q_params.model_name);
  if (model_quantize(q_params, quant_layer)) {
    fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, q_params.model_file.c_str());
    return 1;
  }
  return 0;
}

extern "C" {
  void * create_new_model() {
    return new(std::nothrow) Model();
  }

  void destroy_model(void * model_ptr) {
    delete static_cast<Model*>(model_ptr);
  }

  bool init_model(void * model_ptr, char* model_path, int max_new_tokens, int max_batch_size,
                  int ctx_size, int32_t pad_token, char* memory_dtype,
                  float scratch_size_ratio, int threads, int seed) {
    Model* model = static_cast<Model*>(model_ptr);
    return model->init_model(model_path, max_new_tokens, max_batch_size, ctx_size, pad_token, memory_dtype, scratch_size_ratio, threads, seed);
  }

  int quantize_model(char* model_path, char * out_path, char * weight_dtype,
                     char * alg, int group_size, char * scale_dtype, char * compute_dtype, bool use_ggml, int threads) {
    return Model::quantize_model(model_path, out_path, weight_dtype, alg, group_size, scale_dtype,
                          compute_dtype, use_ggml, threads);
  }

  void * generate(void * model_ptr,
                  void * input_ids_ptr,
                  void * positions_dataptr,
                  bool is_prompt,
                  void * block_tables_ptr,
                  void * slot_mapping_ptr,
                  int * context_lens_arr, int n_contexts) {
    Model* model = static_cast<Model*>(model_ptr);
    return model->generate(input_ids_ptr, positions_dataptr, is_prompt,
            block_tables_ptr, slot_mapping_ptr, context_lens_arr, n_contexts);
  }

  void set_block_size(void * model_ptr, int64_t block_size) {
    Model* model = static_cast<Model*>(model_ptr);
    model->set_block_size(block_size);
  }

  bool free_slots(void * model_ptr, int64_t * seq_ids, int nbr_of_ids) {
    Model* model = static_cast<Model*>(model_ptr);
    return model->free_slots(seq_ids, nbr_of_ids);
  }

  void set_kv_caches_ptr(void * model_ptr, void * kv_caches_ptr) {
    Model* model = static_cast<Model*>(model_ptr);
    model->set_kv_caches_ptr(kv_caches_ptr);
  }

  const char * get_last_error(void * model_ptr) {
    Model* model = static_cast<Model*>(model_ptr);
    return model->get_last_error();
  }

}  // extern "C"
