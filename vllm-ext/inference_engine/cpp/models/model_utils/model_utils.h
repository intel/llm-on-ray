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
#ifndef MODEL_H
#define MODEL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <limits>

#include "models/application/common.h"
#include "models/model_utils/model_config.h"
#include "models/model_utils/model_types.h"

#ifdef MODEL_SHARED
#if defined(_WIN32) && !defined(__MINGW32__)
#ifdef MODEL_BUILD
#define MODEL_API __declspec(dllexport)
#else
#define MODEL_API __declspec(dllimport)
#endif
#else
#define MODEL_API __attribute__((visibility("default")))
#endif
#else
#define MODEL_API
#endif

#define MODEL_FILE_MAGIC_GGJT 0x67676a74u  // 'ggjt'
#define MODEL_FILE_MAGIC_GGLA 0x67676c61u  // 'ggla'
#define MODEL_FILE_MAGIC_GGMF 0x67676d66u  // 'ggmf'
#define MODEL_FILE_MAGIC_NE 0x67676d6cu    // 'ne'
#define MODEL_FILE_MAGIC_GGSN 0x6767736eu  // 'ggsn'

#define MODEL_FILE_VERSION 3
#define MODEL_FILE_MAGIC MODEL_FILE_MAGIC_GGJT
#define MODEL_FILE_MAGIC_UNVERSIONED MODEL_FILE_MAGIC_NE
#define MODEL_SESSION_MAGIC MODEL_FILE_MAGIC_GGSN
#define MODEL_SESSION_VERSION 1

inline int64_t ns_log_level() {
  static int64_t log_level = -1;
  if (log_level == -1) {
    const char* log_level_env = getenv("IE_VERBOSE");
    if (log_level_env != nullptr)
      log_level = std::stoi(log_level_env);
    else
      log_level = -2;
  }
  return log_level;
}

void load_model_internal(const struct model_params& params, model_context& lctx,
                         model_progress_callback progress_callback, void* progress_ctx);

#ifdef __cplusplus
extern "C" {
#endif

MODEL_API bool model_mmap_supported();
MODEL_API bool model_mlock_supported();

// TODO: not great API - very likely to change
// Initialize the model + ne backend
// Call once at the start of the program
MODEL_API void model_init_backend();

MODEL_API int64_t model_time_us();

// Frees all allocated memory
MODEL_API void model_free(struct model_context* ctx);

// Apply a LoRA adapter to a loaded model
// path_base_model is the path to a higher quality model to use as a base for
// the layers modified by the adapter. Can be nullptr to use the current loaded
// model. The model needs to be reloaded before applying a new adapter,
// otherwise the adapter will be applied on top of the previous one Returns 0 on
// success
MODEL_API int model_apply_lora_from_file(struct model_context* ctx, const char* path_lora, const char* path_base_model,
                                         int n_threads);

// Returns the number of tokens in the KV cache
MODEL_API int model_get_kv_cache_token_count(const struct model_context* ctx);

// Sets the current rng seed.
MODEL_API void model_set_rng_seed(struct model_context* ctx, int seed);

// Returns the maximum size in bytes of the state (rng, logits, embedding
// and kv_cache) - will often be smaller after compacting tokens
MODEL_API size_t model_get_state_size(const struct model_context* ctx);

// Copies the state to the specified destination address.
// Destination needs to have allocated enough memory.
// Returns the number of bytes copied
MODEL_API size_t model_copy_state_data(struct model_context* ctx, uint8_t* dst);

// Set the state reading from the specified address
// Returns the number of bytes read
MODEL_API size_t model_set_state_data(struct model_context* ctx, uint8_t* src);

// Save/load session file
MODEL_API bool model_load_session_file(struct model_context* ctx, const char* path_session, model_token* tokens_out,
                                       size_t n_token_capacity, size_t* n_token_total_out);
MODEL_API bool model_save_session_file(struct model_context* ctx, const char* path_session, const model_token* tokens,
                                       size_t n_token_total);

// Run the model inference to obtain the logits and probabilities for the next
// model_input has some necessary members for inference (more details please see model_types.h):
// token. tokens + n_tokens is the provided batch of new tokens to process
// n_past is the offset to which the kv is cached to
// n_total is the number of tokens evaluated in previous eval calls
// Returns 0 on success
MODEL_API int model_eval(struct model_context* ctx, const model_input* inputs, const int n_input, int n_threads);

// Convert the provided text into tokens.
// The tokens pointer must be large enough to hold the resulting tokens.
// Returns the number of tokens on success, no more than n_max_tokens
// Returns a negative number on failure - the number of tokens that would have
// been returned
// TODO: not sure if correct
MODEL_API int model_tokenize(struct model_context* ctx, const char* text, model_token* tokens, int n_max_tokens,
                             bool add_bos);

MODEL_API int model_n_vocab(const struct model_context* ctx);
MODEL_API int model_n_ctx(const struct model_context* ctx);
MODEL_API int model_n_embd(const struct model_context* ctx);

// Token logits obtained from the last call to model_eval()
// The logits for the last token are stored in the last row
// Can be mutated in order to change the probabilities of the next token
// Rows: n_tokens
// Cols: n_vocab
MODEL_API float* model_get_logits(struct model_context* ctx);

// Get the embeddings for the input
// shape: [n_embd] (1-dimensional)
MODEL_API float* model_get_embeddings(struct model_context* ctx);

// Token Id -> String. Uses the vocabulary in the provided context
MODEL_API const char* model_token_to_str(const struct model_context* ctx, model_token token);

// Special tokens
MODEL_API model_token model_token_nl();

// Performance information
MODEL_API void model_print_timings(struct model_context* ctx);
MODEL_API void model_reset_timings(struct model_context* ctx);

// Print system information
MODEL_API const char* model_print_system_info(void);

#ifdef __cplusplus
}
#endif

/* kv cache utils */
// kv cache both stores permuted tensor
// k shape is [head_dim, N, n_head]
// v shape is [N, head_dim, n_head] or [N, n_embd]
/* kv cache utils */

// copy consecutive tokens from one seq to another
MODEL_API void model_kv_cache_seq_cpy(struct model_context* ctx, const model_seq_id& seq_id_src,
                                      const model_seq_id& seq_id_dst, const model_pos& p0, const model_pos& p1);

// concat several seqs into a continuous batch from kv cache
MODEL_API ne_tensor* model_kv_cache_seq_concat(struct ne_cgraph* cgraph, struct model_context* moctx,
                                               struct ne_context* nectx, const int64_t& ne0, const int64_t& ne1,
                                               const int64_t& ne2, const int64_t& ne3,
                                               const std::vector<int>& block_ids, const int& layer_idx,
                                               const bool& concat_k = true);

MODEL_API std::vector<std::vector<int>> split_inputs_into_groups(const model_input* inputs, const int n_input);

#endif  // MODEL_H
