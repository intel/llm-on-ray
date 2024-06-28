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
// Defines fileno on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#include <cstddef>
#include <cstdint>
#include <cstdio>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cinttypes>
#include <climits>
#include <cstring>
#include <ctime>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>  //NOLINT
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <thread>  //NOLINT
#include <unordered_map>

#include "models/application/common.h"
#include "core/layers/bestla_common.hpp"
#include "core/layers/mha_dense.h"
#include "core/ne_layers.h"
#include "core/layers/bestla_gemm.h"
#include "bestla/bestla_parallel.h"

#include "models/model_utils/model_config.h"
#include "models/model_utils/model_files.h"
#include "models/model_utils/model_utils.h"
#include "models/model_utils/util.h"
#include "models/models.h"

//
// kv cache
//

// non-null pointer of model for kv-cache as components of model->layers[il] (e.g. chatglm)
static bool kv_cache_init(const struct model_hparams& hparams, struct model_kv_cache& cache,  // NOLINT
                          const ne_type wtype, const int n_ctx, const int max_batch_size, model_struct* model) {
  const auto n_layer = hparams.n_layer;
  auto heads_kv = hparams.n_head_kv > 0 ? hparams.n_head_kv : hparams.n_head;
  const auto head_size = hparams.n_embd_head_k == 0 ? hparams.n_embd / hparams.n_head : hparams.n_embd_head_k;

  // 64bit to avoid overflow in later calculation
  int64_t k_size, v_size;
  get_batch_kv_elements_from_model_params(heads_kv, head_size, n_ctx, wtype, &k_size, &v_size);

  int64_t layer_ne_k = max_batch_size * k_size;
  int64_t layer_ne_v = max_batch_size * v_size;
  const auto wsize = wtype == NE_TYPE_BTLA ? 1 : ne_type_size(wtype);

  cache.buf.resize(n_layer * (layer_ne_k + layer_ne_v) * wsize + 2u * MB);
  cache.seq_cells.resize(max_batch_size);
  for (int i = 0; i < cache.seq_cells.size(); ++i) {
    cache.seq_cells[i].token_cells.resize(n_ctx);
  }

  struct ne_init_params params;
  params.mem_size = cache.buf.size;
  params.mem_buffer = cache.buf.addr;
  params.no_alloc = false;

  cache.ctx = ne_init(params);

  if (!cache.ctx) {
    fprintf(stderr, "%s: failed to allocate memory for kv cache\n", __func__);
    return false;
  }

  // NE_TYPE_BTLA can not be allocated memory
  const auto wtype_alloc = wtype == NE_TYPE_BTLA ? NE_TYPE_I8 : wtype;

  if (model) {  // non-null param of model for kv-cache as components of model->layers[il]
    for (int il = 0; il < n_layer; ++il) {
      auto& k_cache = model->layers[il].k_cache;
      auto& v_cache = model->layers[il].v_cache;
      if (wtype == NE_TYPE_F16) {  // chatglm does not support fp32 kv-cache in original impl of chatglm_util.cpp
        const auto head_size = hparams.n_embd_head_k == 0 ? hparams.n_embd / hparams.n_head : hparams.n_embd_head_k;
        const int heads_kv = hparams.multi_query_group_num > 0 ? hparams.multi_query_group_num : hparams.n_head;
        k_cache = d_ne_new_tensor_4d(model->ctx, NE_TYPE_F16, head_size, n_ctx, heads_kv, max_batch_size);
        v_cache = d_ne_new_tensor_4d(model->ctx, NE_TYPE_F16, n_ctx, head_size, heads_kv, max_batch_size);
      } else if (wtype == NE_TYPE_BTLA) {
        k_cache = ne_new_tensor_1d(model->ctx, wtype_alloc, layer_ne_k + NE_ALIGNMENT, NE_SIZE_CALC);
        const auto k_align_off = reinterpret_cast<uintptr_t>(k_cache->data) % NE_ALIGNMENT;
        k_cache = ne_view_1d(model->ctx, k_cache, layer_ne_k, NE_ALIGNMENT - k_align_off);
        k_cache->type = wtype;
        v_cache = ne_new_tensor_1d(model->ctx, wtype_alloc, layer_ne_v + NE_ALIGNMENT, NE_SIZE_CALC);
        const auto v_align_off = reinterpret_cast<uintptr_t>(v_cache->data) % NE_ALIGNMENT;
        v_cache = ne_view_1d(model->ctx, v_cache, layer_ne_v, NE_ALIGNMENT - v_align_off);
        v_cache->type = wtype;
      } else {
        NE_ASSERT(("Unexpected ne dtype for kv-cache", false));
      }
      ne_set_name(k_cache, "cache_k");
      ne_set_name(v_cache, "cache_v");
    }
    const bool run_mha_reordered = model->layers[0].k_cache->type == NE_TYPE_BTLA;
    fprintf(stderr, "%s: run_mha_reordered = %d\n", __func__, run_mha_reordered);
  } else {
    cache.k = ne_new_tensor_1d(cache.ctx, wtype_alloc, n_layer * layer_ne_k + NE_ALIGNMENT, NE_SIZE_CALC);
    const auto k_align_off = reinterpret_cast<uintptr_t>(cache.k->data) % NE_ALIGNMENT;
    cache.k = ne_view_1d(cache.ctx, cache.k, n_layer * layer_ne_k, NE_ALIGNMENT - k_align_off);
    cache.k->type = wtype;
    cache.v = ne_new_tensor_1d(cache.ctx, wtype_alloc, n_layer * layer_ne_v + NE_ALIGNMENT, NE_SIZE_CALC);
    const auto v_align_off = reinterpret_cast<uintptr_t>(cache.v->data) % NE_ALIGNMENT;
    cache.v = ne_view_1d(cache.ctx, cache.v, n_layer * layer_ne_v, NE_ALIGNMENT - v_align_off);
    cache.v->type = wtype;
    ne_set_name(cache.k, "cache_k");
    ne_set_name(cache.v, "cache_v");
  }

  return true;
}

bool model_mmap_supported() { return model_mmap::SUPPORTED; }

bool model_mlock_supported() { return model_mlock::SUPPORTED; }

void model_init_backend() {
  ne_time_init();

  // needed to initialize f16 tables
  {
    struct ne_init_params params = {0, nullptr, false};
    struct ne_context* ctx = ne_init(params);
    ne_free(ctx);
  }
}

int64_t model_time_us() { return ne_time_us(); }

//
// model loading
//

static bool load_model(const struct model_params& params, model_context& lctx, model_progress_callback progress_callback,
                       void* progress_ctx) {
  try {
    lctx.t_start_us = ne_time_us();
    lctx.model.arch = params.model_arch;
    load_model_internal(params, lctx, progress_callback, progress_ctx);
    lctx.t_load_us = ne_time_us() - lctx.t_start_us;
    return true;
  } catch (const std::string& err) {
    fprintf(stderr, "error loading model: %s\n", err.c_str());
    return false;
  }
}

//
// tokenizer
//

static size_t utf8_len(char src) {
  const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
  uint8_t highbits = static_cast<uint8_t>(src) >> 4;
  return lookup[highbits];
}

struct model_sp_symbol_t {
  using index = int;
  index prev;
  index next;
  const char* text;
  size_t n;
};

static_assert(std::is_trivially_copyable<model_sp_symbol_t>::value, "model_sp_symbol_t is not trivially copyable");

struct model_sp_bigram_t {
  struct comparator_t {
    bool operator()(model_sp_bigram_t& l, model_sp_bigram_t& r) {  // NOLINT
      return (l.score < r.score) || (l.score == r.score && l.left > r.left);
    }
  };
  using queue_storage = std::vector<model_sp_bigram_t>;
  using queue = std::priority_queue<model_sp_bigram_t, queue_storage, comparator_t>;
  model_sp_symbol_t::index left;
  model_sp_symbol_t::index right;
  float score;
  size_t size;
};

// original implementation:
// https://github.com/ggerganov/model.cpp/commit/074bea2eb1f1349a0118239c4152914aecaa1be4
struct model_tokenizer_t {
  model_tokenizer_t(const model_vocab& vocab) : vocab_(vocab) {}  // NOLINT

  void tokenize(const std::string& text, std::vector<model_vocab::id>& output) {
    // split string into utf8 chars
    int index = 0;
    size_t offs = 0;
    while (offs < text.size()) {
      model_sp_symbol_t sym;
      size_t char_len = std::min(text.size() - offs, utf8_len(text[offs]));
      sym.text = text.c_str() + offs;
      sym.n = char_len;
      offs += char_len;
      sym.prev = index - 1;
      sym.next = offs == text.size() ? -1 : index + 1;
      index++;
      symbols_.emplace_back(sym);
    }

    // seed the work queue with all possible 2-character tokens.
    for (size_t i = 1; i < symbols_.size(); ++i) {
      try_add_bigram(i - 1, i);
    }

    // keep substituting the highest frequency pairs for as long as we can.
    while (!work_queue_.empty()) {
      auto bigram = work_queue_.top();
      work_queue_.pop();

      auto& left_sym = symbols_[bigram.left];
      auto& right_sym = symbols_[bigram.right];

      // if one of the symbols already got merged, skip it.
      if (left_sym.n == 0 || right_sym.n == 0 || left_sym.n + right_sym.n != bigram.size) {
        continue;
      }

      // merge the right sym into the left one
      left_sym.n += right_sym.n;
      right_sym.n = 0;

      // printf("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text,
      // bigram.size);

      // remove the right sym from the chain
      left_sym.next = right_sym.next;
      if (right_sym.next >= 0) {
        symbols_[right_sym.next].prev = bigram.left;
      }

      // find more substitutions
      try_add_bigram(left_sym.prev, bigram.left);
      try_add_bigram(bigram.left, left_sym.next);
    }

    for (int i = 0; i != -1; i = symbols_[i].next) {
      auto& symbol = symbols_[i];
      auto symbol_text = std::string(symbol.text, symbol.n);
      auto token = vocab_.token_to_id.find(symbol_text);

      if (token == vocab_.token_to_id.end()) {
        // output any symbols that did not form tokens as bytes.
        for (int j = 0; j < static_cast<int>(symbol.n); ++j) {
          model_vocab::id token_id = static_cast<uint8_t>(symbol.text[j]) + 3;
          output.push_back(token_id);
        }
      } else {
        output.push_back((*token).second);
      }
    }
  }

 private:
  void try_add_bigram(int left, int right) {
    if (left == -1 || right == -1) {
      return;
    }

    const std::string text = std::string(symbols_[left].text, symbols_[left].n + symbols_[right].n);
    auto token = vocab_.token_to_id.find(text);

    if (token == vocab_.token_to_id.end()) {
      return;
    }

    if (static_cast<size_t>((*token).second) >= vocab_.id_to_token.size()) {
      return;
    }

    const auto& tok_score = vocab_.id_to_token[(*token).second];

    model_sp_bigram_t bigram;
    bigram.left = left;
    bigram.right = right;
    bigram.score = tok_score.score;
    bigram.size = text.size();
    work_queue_.push(bigram);
  }

  const model_vocab& vocab_;
  std::vector<model_sp_symbol_t> symbols_;
  model_sp_bigram_t::queue work_queue_;
};

static std::vector<model_vocab::id> model_tokenize(const model_vocab& vocab, const std::string& text, bool bos) {
  model_tokenizer_t tokenizer(vocab);
  std::vector<model_vocab::id> output;

  if (text.empty()) {
    return output;
  }

  if (bos) {
    output.push_back(vocab.bos_token_id);
  }

  tokenizer.tokenize(text, output);
  return output;
}

//
// interface implementation
//

struct model_context* create_model_context_from_file(const struct model_params& params) {
  ne_time_init();

  model_context* ctx = new model_context;
  
  unsigned cur_percentage = 0;
  model_progress_callback progress_callback = [](float progress, void* ctx) {
    unsigned* cur_percentage_p = reinterpret_cast<unsigned*>(ctx);
    unsigned percentage = (unsigned)(100 * progress);
    while (percentage > *cur_percentage_p) {
      *cur_percentage_p = percentage;
      fprintf(stderr, ".");
      fflush(stderr);
      if (percentage >= 100) {
        fprintf(stderr, "\n");
      }
    }
  };

  ctx->rng = std::mt19937(params.seed);
  ctx->max_batch_size = params.max_batch_size;
  ctx->n_ctx = params.n_ctx;
  ctx->kv_n_ctx_block = ctx->max_batch_size;

  // ctx->scratch_size_ratio = params.scratch_size_ratio * ctx->max_batch_size;
  ctx->scratch_size_ratio = params.scratch_size_ratio * 2.0f;  // 1.2f to be protective

  const model_archs arch = params.model_arch;

  // the type so that kv-cache allocated according to this type must be large enough
  if (!load_model(params, *ctx, progress_callback, &cur_percentage)) {
    fprintf(stderr, "%s: failed to load model\n", __func__);
    model_free(ctx);
    return nullptr;
  }

  // reserve memory for context buffers
  const auto& hparams = ctx->model.hparams;

  const attn_shape_t attn_shape = {
      /* .batch_size = */ ctx->max_batch_size,
      /* .head_num = */ static_cast<int>(hparams.n_head),
      /* .heads_kv = */ static_cast<int>(hparams.n_head_kv),
      /* .head_size = */ static_cast<int>(hparams.n_embd / hparams.n_head),
      /* .sl_q = */ 1,  // for next-token inference
      /* .sl_kv = */ static_cast<int>(ctx->n_ctx),
  };
  const bool support_bestla_kv = ctx->support_bestla_kv && bestla_reordered_attn_fp32_support(&attn_shape);
  fprintf(stderr, "%s: support_bestla_kv = %d\n", __func__, support_bestla_kv);

  const ne_type memory_type = params.memory_type == KV_MEM_TYPE_F16   ? NE_TYPE_F16
                              : params.memory_type == KV_MEM_TYPE_F32 ? NE_TYPE_F32
                              : params.memory_type == KV_MEM_TYPE_AUTO
                                  ? (support_bestla_kv ? NE_TYPE_BTLA : NE_TYPE_F16)  // fall back to fp16
                                  : NE_TYPE_COUNT;
  NE_ASSERT(memory_type != NE_TYPE_COUNT);

  const bool kv_in_layers =
      (arch == MODEL_CHATGLM3 || arch == MODEL_CHATGLM2 || arch == MODEL_CHATGLM || arch == MODEL_BAICHUAN);
  if (!kv_cache_init(ctx->model.hparams, ctx->model.kv_self, memory_type, ctx->n_ctx, ctx->max_batch_size, (kv_in_layers ? &ctx->model : nullptr))) {
    fprintf(stderr, "%s: kv_cache_init() failed for self-attention cache\n", __func__);
    model_free(ctx);
    return nullptr;
  }

  if (ctx->model.kv_self.k != nullptr) {
    const size_t memory_size = params.memory_type == KV_MEM_TYPE_AUTO
                                    ? ne_nelements(ctx->model.kv_self.k) + ne_nelements(ctx->model.kv_self.v)
                                    : ne_nbytes(ctx->model.kv_self.k) + ne_nbytes(ctx->model.kv_self.v);
    fprintf(stderr, "%s: kv self size = %7.2f MB\n", __func__, memory_size / 1024.0 / 1024.0);
  } else if (ctx->model.layers[0].k_cache != nullptr) {
    const auto k_cache = ctx->model.layers[0].k_cache;
    const auto v_cache = ctx->model.layers[0].v_cache;
    const size_t layer_memory_size = params.memory_type == KV_MEM_TYPE_AUTO
                                          ? ne_nelements(k_cache) + ne_nelements(v_cache)
                                          : ne_nbytes(k_cache) + ne_nbytes(v_cache);
    fprintf(stderr, "%s: kv self size = %7.2f MB\n", __func__, layer_memory_size / 1024.0 / 1024.0 * hparams.n_layer);
  } else {
    NE_ASSERT(("KV-cache not allocated!", false));
  }

  ctx->last_hidden_states.resize(hparams.n_embd);

  ctx->buf_compute.resize(ctx->model.scratchs.eval);

  ctx->buf_scratch[0].resize(ctx->model.scratchs.scratch0);
  ctx->buf_scratch[1].resize(ctx->model.scratchs.scratch1);

  return ctx;
}

void model_free(struct model_context* ctx) { delete ctx; }

int model_apply_lora_from_file_internal(struct model_context* ctx, const char* path_lora, const char* path_base_model,
                                        int n_threads) {
  fprintf(stderr, "%s: applying lora adapter from '%s' - please wait ...\n", __func__, path_lora);

  auto& model = ctx->model;

  const int64_t t_start_lora_us = ne_time_us();

  auto fin = std::ifstream(path_lora, std::ios::binary);
  if (!fin) {
    fprintf(stderr, "%s: failed to open '%s'\n", __func__, path_lora);
    return 1;
  }

  // verify magic and version
  {
    uint32_t magic;
    fin.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != MODEL_FILE_MAGIC_GGLA) {
      fprintf(stderr, "%s: bad file magic\n", __func__);
      return 1;
    }
    uint32_t format_version;
    fin.read(reinterpret_cast<char*>(&format_version), sizeof(format_version));

    if (format_version != 1) {
      fprintf(stderr, "%s: unsupported file version\n", __func__);
      return 1;
    }
  }

  int32_t lora_r;
  int32_t lora_alpha;
  fin.read(reinterpret_cast<char*>(&lora_r), sizeof(lora_r));
  fin.read(reinterpret_cast<char*>(&lora_alpha), sizeof(lora_alpha));
  float scaling = static_cast<float>(lora_alpha) / static_cast<float>(lora_r);

  fprintf(stderr, "%s: r = %d, alpha = %d, scaling = %.2f\n", __func__, lora_r, lora_alpha, scaling);

  // create a temporary ne context to store the lora tensors
  // todo: calculate size from biggest possible tensor
  std::vector<uint8_t> lora_buf(1024ull * 1024ull * 1024ull);
  struct ne_init_params params;
  params.mem_size = lora_buf.size();
  params.mem_buffer = lora_buf.data();
  params.no_alloc = false;

  ne_context* lora_ctx = ne_init(params);
  std::unordered_map<std::string, struct ne_tensor*> lora_tensors;

  // create a name -> tensor map of the model to accelerate lookups
  std::unordered_map<std::string, struct ne_tensor*> model_tensors;
  for (auto& kv : model.tensors_by_name) {
    model_tensors.insert(kv);
  }

  // load base model
  std::unique_ptr<model_model_loader> model_loader;
  ne_context* base_ctx = nullptr;
  model_buffer base_buf;
  if (path_base_model) {
    fprintf(stderr, "%s: loading base model from '%s'\n", __func__, path_base_model);
    model_loader.reset(new model_model_loader(path_base_model, /*use_mmap*/ true, /*vocab_only*/ false));

    size_t ctx_size;
    size_t mmapped_size;
    model_loader->calc_sizes(&ctx_size, &mmapped_size);
    base_buf.resize(ctx_size);

    ne_init_params base_params;
    base_params.mem_size = base_buf.size;
    base_params.mem_buffer = base_buf.addr;
    base_params.no_alloc = model_loader->use_mmap;

    base_ctx = ne_init(base_params);

    model_loader->ne_ctx = base_ctx;

    // maybe this should in model_model_loader
    if (model_loader->use_mmap) {
      model_loader->mapping.reset(new model_mmap(&model_loader->file_loaders.at(0)->file, /* prefetch */ 0));
    }
  }

  // read tensors and apply
  bool warned = false;
  int n_tensors = 0;
  while (true) {
    int32_t n_dims;
    int32_t length;
    int32_t ftype;

    fin.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
    fin.read(reinterpret_cast<char*>(&length), sizeof(length));
    fin.read(reinterpret_cast<char*>(&ftype), sizeof(ftype));
    if (fin.eof()) {
      break;
    }

    int32_t ne[2] = {1, 1};
    for (int i = 0; i < n_dims; ++i) {
      fin.read(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
    }

    std::string name;
    {
      char buf[1024];
      fin.read(buf, length);
      name = std::string(buf, length);
    }

    // check for lora suffix and get the type of tensor
    const std::string lora_suffix = ".lora";
    size_t pos = name.rfind(lora_suffix);
    if (pos == std::string::npos) {
      fprintf(stderr, "%s: error: '%s' is not a lora tensor\n", __func__, name.c_str());
      return 1;
    }

    std::string lora_type = name.substr(pos + lora_suffix.length());
    std::string base_name = name;
    base_name.erase(pos);
    // fprintf(stderr, "%s: %s => %s (lora type %s) ", __func__,
    // name.c_str(),base_name.c_str(), lora_type.c_str());

    if (model_tensors.find(base_name) == model_tensors.end()) {
      fprintf(stderr, "%s: unknown tensor '%s' in lora adapter\n", __func__, name.data());
      return 1;
    }

    // create ne tensor
    ne_type wtype;
    switch (ftype) {
      case 0:
        wtype = NE_TYPE_F32;
        break;
      case 1:
        wtype = NE_TYPE_F16;
        break;
      default: {
        fprintf(stderr, "%s: invalid tensor data type '%d'\n", __func__, ftype);
        return false;
      }
    }
    ne_tensor* lora_tensor;
    if (n_dims == 2) {
      lora_tensor = ne_new_tensor_2d(lora_ctx, wtype, ne[0], ne[1], NE_SIZE_CALC);
    } else {
      fprintf(stderr, "%s: unsupported tensor dimension %d\n", __func__, n_dims);
      return 1;
    }

    // load tensor data
    size_t offset = fin.tellg();
    size_t tensor_data_size = ne_nbytes(lora_tensor);
    offset = (offset + 31) & -32;
    fin.seekg(offset);
    fin.read(reinterpret_cast<char*>(lora_tensor->data), tensor_data_size);

    lora_tensors[name] = lora_tensor;

    // check if we have both A and B tensors and apply
    if (lora_tensors.find(base_name + ".loraA") != lora_tensors.end() &&
        lora_tensors.find(base_name + ".loraB") != lora_tensors.end()) {
      ne_tensor* dest_t = model_tensors[base_name];
      ne_tensor* base_t;
      if (model_loader) {
        // load from base model
        if (model_loader->tensors_map.name_to_idx.find(base_name) == model_loader->tensors_map.name_to_idx.end()) {
          fprintf(stderr, "%s: error: tensor '%s' not found in base model\n", __func__, base_name.c_str());
          return 1;
        }
        size_t idx = model_loader->tensors_map.name_to_idx[base_name];
        model_load_tensor& lt = model_loader->tensors_map.tensors[idx];
        base_t =
            model_loader->get_tensor(base_name, {(uint32_t)dest_t->ne[0], (uint32_t)dest_t->ne[1]}, NE_BACKEND_CPU);
        lt.data = reinterpret_cast<uint8_t*>(lt.ne_tensor->data);
        model_loader->load_data_for(lt);
        lt.ne_tensor->data = lt.data;
      } else {
        base_t = dest_t;
      }

      if (ne_is_quantized(base_t->type)) {
        if (!warned) {
          fprintf(stderr,
                  "%s: warning: using a lora adapter with a quantized model "
                  "may result in poor quality, "
                  "use a f16 or f32 base model with --lora-base\n",
                  __func__);
          warned = true;
        }
      }

      ne_tensor* loraA = lora_tensors[base_name + ".loraA"];
      ne_tensor* loraB = lora_tensors[base_name + ".loraB"];

      if (base_t->ne[0] != loraA->ne[1] || base_t->ne[1] != loraB->ne[1]) {
        fprintf(stderr,
                "%s: incompatible tensor dimensions (%" PRId64 " and %" PRId64
                ");"
                " are you sure that this adapter is for this model?\n",
                __func__, base_t->ne[0], loraA->ne[1]);
        return 1;
      }

      // w = w + BA*s
      ne_tensor* BA = ne_mul_mat(lora_ctx, loraA, loraB);

      if (scaling != 1.0f) {
        ne_tensor* scale_tensor = ne_new_f32(lora_ctx, scaling);
        BA = ne_scale_inplace(lora_ctx, BA, scale_tensor);
      }

      ne_tensor* r;
      if (base_t == dest_t) {
        r = ne_add_inplace(lora_ctx, dest_t, BA);
      } else {
        r = ne_add(lora_ctx, base_t, BA);
        r = ne_cpy(lora_ctx, r, dest_t);
      }

      struct ne_cgraph gf = ne_build_forward(r);
      gf.n_threads = n_threads;
      ne_graph_compute(lora_ctx, &gf);

      // we won't need these tensors again, reset the context to save memory
      ne_free(lora_ctx);
      lora_ctx = ne_init(params);
      lora_tensors.clear();

      n_tensors++;
      if (n_tensors % 4 == 0) {
        fprintf(stderr, ".");
      }
    }
  }

  // this should be in a destructor, it will leak on failure
  ne_free(lora_ctx);
  if (base_ctx) {
    ne_free(base_ctx);
  }

  const int64_t t_lora_us = ne_time_us() - t_start_lora_us;
  fprintf(stderr, " done (%.2f ms)\n", t_lora_us / 1000.0);

  return 0;
}

int model_apply_lora_from_file(struct model_context* ctx, const char* path_lora, const char* path_base_model,
                               int n_threads) {
  try {
    return model_apply_lora_from_file_internal(ctx, path_lora, path_base_model, n_threads);
  } catch (const std::string& err) {
    fprintf(stderr, "%s: failed to apply lora adapter: %s\n", __func__, err.c_str());
    return 1;
  }
}

struct model_context* create_model_context(const model_params& params) {
  if (params.model_arch == MODEL_UNKNOWN) {
    fprintf(stderr, "error, please set model_name \n");
    exit(0);
  }

  model_context* lctx = create_model_context_from_file(params);

  const auto& model_hparams = lctx->model.hparams;
  // printf("n_head_kv=%s,multi_query_group_num=%s",model_hparams.n_head_kv,model_hparams.multi_query_group_num);
  NE_ASSERT(("Can not set n_head_kv and multi_query_group_num at the same time",
             model_hparams.n_head_kv == 0 || model_hparams.multi_query_group_num == 0 ||
                 model_hparams.n_head_kv == model_hparams.multi_query_group_num));
  attn_shape_t attn_shape = {
      /* .batch_size = */ params.max_batch_size,
      /* .head_num = */ static_cast<int>(model_hparams.n_head),
      /* .heads_kv = */ static_cast<int>(model_hparams.n_head_kv + model_hparams.multi_query_group_num),
      /* .head_size = */ static_cast<int>(model_hparams.n_embd / model_hparams.n_head),
      /* .sl_q = */ 1,  // Note: make sure that bestla reordered attn supports next token inferencing
      /* .sl_kv = */ static_cast<int>(params.n_ctx),
  };
  const auto k_cache_example = lctx->model.kv_self.k != nullptr ? lctx->model.kv_self.k           // llama.cpp style
                                                                : lctx->model.layers[0].k_cache;  // chatglm style
  NE_ASSERT(k_cache_example->type != NE_TYPE_BTLA || bestla_reordered_attn_fp32_support(&attn_shape));

  if (lctx == nullptr) {
    fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
    return nullptr;
  }

  if (!params.lora_adapter.empty()) {
    int err =
        model_apply_lora_from_file(lctx, params.lora_adapter.c_str(),
                                   params.lora_base.empty() ? nullptr : params.lora_base.c_str(), params.n_threads);
    if (err != 0) {
      fprintf(stderr, "%s: error: failed to apply lora adapter\n", __func__);
      return nullptr;
    }
  }

  return lctx;
}

void get_batch_kv_elements_from_model_params(int heads_kv, int head_size, int n_ctx, ne_type wtype, int64_t* k_size,
                                           int64_t* v_size) {
  if (wtype == NE_TYPE_F16 || wtype == NE_TYPE_F32) {
    *k_size = n_ctx * heads_kv * head_size;
    *v_size = n_ctx * heads_kv * head_size;
  } else if (wtype == NE_TYPE_BTLA) {
    kv_shape_t kv_shape = {
        /* .heads_kv = */ static_cast<uint32_t>(heads_kv),
        /* .head_size = */ static_cast<uint32_t>(head_size),
        /* .sl_kv_max = */ static_cast<uint32_t>(n_ctx),
    };
    kv_cache_info_t kv_cache_info;
    bestla_reordered_attn_fp32_batch_kv_info(&kv_shape, &kv_cache_info);
    *k_size = kv_cache_info.k_bytes;
    *v_size = kv_cache_info.v_bytes;
  } else {
    assert(false);
  }
}

int model_get_kv_cache_token_count(const struct model_context* ctx) { return ctx->model.kv_self.n; }

#define MODEL_MAX_RNG_STATE (64 * 1024)

void model_set_rng_seed(struct model_context* ctx, int seed) {
  if (seed < 0) {
    seed = time(nullptr);
  }
  ctx->rng.seed(seed);
}

// Returns the *maximum* size of the state
size_t model_get_state_size(const struct model_context* ctx) {
  // we don't know size of rng until we actually serialize it. so reserve more
  // than enough memory for its serialized state. for reference,
  // std::mt19937(1337) serializes to 6701 bytes.
  const size_t s_rng_size = sizeof(size_t);
  const size_t s_rng = MODEL_MAX_RNG_STATE;
  const size_t s_hidden_state_size = sizeof(size_t);
  const size_t s_hidden_states = ctx->last_hidden_states.size() * sizeof(float);
  const size_t s_kv_size = sizeof(size_t);
  const size_t s_kv_ntok = sizeof(int);
  const size_t s_kv = ctx->model.kv_self.buf.size;

  const size_t s_total = (+s_rng_size + s_rng + s_hidden_state_size + s_hidden_states +
                          s_kv_size + s_kv_ntok + s_kv);

  return s_total;
}

// Copies the state to the specified destination address
size_t model_copy_state_data(struct model_context* ctx, uint8_t* dst) {
  uint8_t* out = dst;

  // copy rng
  {
    std::stringstream rng_ss;
    rng_ss << ctx->rng;

    const size_t rng_size = rng_ss.str().size();
    char rng_buf[MODEL_MAX_RNG_STATE];

    memset(&rng_buf[0], 0, MODEL_MAX_RNG_STATE);
    memcpy(&rng_buf[0], rng_ss.str().data(), rng_ss.str().size());

    memcpy(out, &rng_size, sizeof(rng_size));
    out += sizeof(rng_size);
    memcpy(out, &rng_buf[0], MODEL_MAX_RNG_STATE);
    out += MODEL_MAX_RNG_STATE;
  }

  // copy hidden states
  {
    const size_t hidden_state_size = ctx->last_hidden_states.size();

    memcpy(out, &hidden_state_size, sizeof(hidden_state_size));
    out += sizeof(hidden_state_size);

    if (hidden_state_size) {
      memcpy(out, ctx->last_hidden_states.data(), hidden_state_size * sizeof(float));
      out += hidden_state_size * sizeof(float);
    }
  }

  // copy kv cache
  {
    const auto& kv_self = ctx->model.kv_self;
    const auto& hparams = ctx->model.hparams;
    const int n_layer = hparams.n_layer;
    const int n_embd = hparams.n_embd;
    const int n_ctx = ctx->n_ctx;

    const size_t kv_size = kv_self.buf.size;
    const int kv_ntok = model_get_kv_cache_token_count(ctx);

    memcpy(out, &kv_size, sizeof(kv_size));
    out += sizeof(kv_size);
    memcpy(out, &kv_ntok, sizeof(kv_ntok));
    out += sizeof(kv_ntok);

    if (kv_size) {
      const size_t elt_size = ne_element_size(kv_self.k);

      char buffer[4096];

      ne_context* cpy_ctx = ne_init({sizeof(buffer), buffer, /* no_alloc */ true});
      ne_cgraph gf{};
      gf.n_threads = 1;

      ne_tensor* kout3d = ne_new_tensor_3d(cpy_ctx, kv_self.k->type, n_embd, kv_ntok, n_layer, NE_SIZE_CALC);
      kout3d->data = out;
      out += ne_nbytes(kout3d);

      ne_tensor* vout3d = ne_new_tensor_3d(cpy_ctx, kv_self.v->type, kv_ntok, n_embd, n_layer, NE_SIZE_CALC);
      vout3d->data = out;
      out += ne_nbytes(vout3d);

      ne_tensor* k3d =
          ne_view_3d(cpy_ctx, kv_self.k, n_embd, kv_ntok, n_layer, elt_size * n_embd, elt_size * n_embd * n_ctx, 0);

      ne_tensor* v3d =
          ne_view_3d(cpy_ctx, kv_self.v, kv_ntok, n_embd, n_layer, elt_size * n_ctx, elt_size * n_ctx * n_embd, 0);

      ne_build_forward_expand(&gf, ne_cpy(cpy_ctx, k3d, kout3d));
      ne_build_forward_expand(&gf, ne_cpy(cpy_ctx, v3d, vout3d));
      ne_graph_compute(cpy_ctx, &gf);

      ne_free(cpy_ctx);
    }
  }

  const size_t written = out - dst;
  const size_t max_size = model_get_state_size(ctx);

  MODEL_ASSERT(written <= max_size);

  return written;
}

// Sets the state reading from the specified source address
size_t model_set_state_data(struct model_context* ctx, uint8_t* src) {
  uint8_t* inp = src;

  // set rng
  {
    size_t rng_size;
    char rng_buf[MODEL_MAX_RNG_STATE];

    memcpy(&rng_size, inp, sizeof(rng_size));
    inp += sizeof(rng_size);
    memcpy(&rng_buf[0], inp, MODEL_MAX_RNG_STATE);
    inp += MODEL_MAX_RNG_STATE;

    std::stringstream rng_ss;
    rng_ss.str(std::string(&rng_buf[0], rng_size));
    rng_ss >> ctx->rng;

    MODEL_ASSERT(rng_ss.fail() == false);
  }

  // set hidden states
  {
    size_t hidden_state_size;

    memcpy(&hidden_state_size, inp, sizeof(hidden_state_size));
    inp += sizeof(hidden_state_size);

    MODEL_ASSERT(ctx->last_hidden_states.capacity() == hidden_state_size);

    if (hidden_state_size) {
      memcpy(ctx->last_hidden_states.data(), inp, hidden_state_size * sizeof(float));
      inp += hidden_state_size * sizeof(float);
    }
  }

  // set kv cache
  {
    const auto& kv_self = ctx->model.kv_self;
    const auto& hparams = ctx->model.hparams;
    const int n_layer = hparams.n_layer;
    const int n_embd = hparams.n_embd;
    const int n_ctx = ctx->n_ctx;

    size_t kv_size;
    int kv_ntok;

    memcpy(&kv_size, inp, sizeof(kv_size));
    inp += sizeof(kv_size);
    memcpy(&kv_ntok, inp, sizeof(kv_ntok));
    inp += sizeof(kv_ntok);

    if (kv_size) {
      MODEL_ASSERT(kv_self.buf.size == kv_size);

      const size_t elt_size = ne_element_size(kv_self.k);

      char buffer[4096];

      ne_context* cpy_ctx = ne_init({sizeof(buffer), buffer, /* no_alloc */ true});
      ne_cgraph gf{};
      gf.n_threads = 1;

      ne_tensor* kin3d = ne_new_tensor_3d(cpy_ctx, kv_self.k->type, n_embd, kv_ntok, n_layer, NE_SIZE_CALC);
      kin3d->data = reinterpret_cast<void*>(inp);
      inp += ne_nbytes(kin3d);

      ne_tensor* vin3d = ne_new_tensor_3d(cpy_ctx, kv_self.v->type, kv_ntok, n_embd, n_layer, NE_SIZE_CALC);
      vin3d->data = reinterpret_cast<void*>(inp);
      inp += ne_nbytes(vin3d);

      ne_tensor* k3d =
          ne_view_3d(cpy_ctx, kv_self.k, n_embd, kv_ntok, n_layer, elt_size * n_embd, elt_size * n_embd * n_ctx, 0);

      ne_tensor* v3d =
          ne_view_3d(cpy_ctx, kv_self.v, kv_ntok, n_embd, n_layer, elt_size * n_ctx, elt_size * n_ctx * n_embd, 0);

      ne_build_forward_expand(&gf, ne_cpy(cpy_ctx, kin3d, k3d));
      ne_build_forward_expand(&gf, ne_cpy(cpy_ctx, vin3d, v3d));
      ne_graph_compute(cpy_ctx, &gf);

      ne_free(cpy_ctx);
    }

    ctx->model.kv_self.n = kv_ntok;
  }

  const size_t nread = inp - src;
  const size_t max_size = model_get_state_size(ctx);

  MODEL_ASSERT(nread <= max_size);

  return nread;
}

bool model_load_session_file(struct model_context* ctx, const char* path_session, model_token* tokens_out,
                             size_t n_token_capacity, size_t* n_token_count_out) {
  model_file file(path_session, "rb");

  // sanity checks
  {
    const uint32_t magic = file.read_u32();
    const uint32_t version = file.read_u32();

    if (magic != MODEL_SESSION_MAGIC || version != MODEL_SESSION_VERSION) {
      fprintf(stderr, "%s : unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
      return false;
    }

    model_hparams session_hparams;
    file.read_raw(&session_hparams, sizeof(model_hparams));

    if (session_hparams != ctx->model.hparams) {
      fprintf(stderr, "%s : model hparams didn't match from session file!\n", __func__);
      return false;
    }
  }

  // load the prompt
  {
    const uint32_t n_token_count = file.read_u32();

    if (n_token_count > n_token_capacity) {
      fprintf(stderr, "%s : token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count,
              n_token_capacity);
      return false;
    }

    file.read_raw(tokens_out, sizeof(model_token) * n_token_count);
    *n_token_count_out = n_token_count;
  }

  // restore the context state
  {
    const size_t n_state_size_cur = file.size - file.tell();
    const size_t n_state_size_max = model_get_state_size(ctx);

    if (n_state_size_cur > n_state_size_max) {
      fprintf(stderr, "%s : the state size in session file is too big! max %zu, got %zu\n", __func__, n_state_size_max,
              n_state_size_cur);
      return false;
    }

    std::vector<uint8_t> state_data(n_state_size_max);
    file.read_raw(state_data.data(), n_state_size_cur);

    model_set_state_data(ctx, state_data.data());
  }

  return true;
}

bool model_save_session_file(struct model_context* ctx, const char* path_session, const model_token* tokens,
                             size_t n_token_count) {
  model_file file(path_session, "wb");

  file.write_u32(MODEL_SESSION_MAGIC);
  file.write_u32(MODEL_SESSION_VERSION);

  file.write_raw(&ctx->model.hparams, sizeof(model_hparams));

  // save the prompt
  file.write_u32((uint32_t)n_token_count);
  file.write_raw(tokens, sizeof(model_token) * n_token_count);

  // save the context state
  {
    const size_t n_state_size_max = model_get_state_size(ctx);

    std::vector<uint8_t> state_data(n_state_size_max);
    const size_t n_state_size_cur = model_copy_state_data(ctx, state_data.data());

    file.write_raw(state_data.data(), n_state_size_cur);
  }

  return true;
}

int model_tokenize(struct model_context* ctx, const char* text, model_token* tokens, int n_max_tokens, bool add_bos) {
  auto res = model_tokenize(ctx->vocab, text, add_bos);

  if (n_max_tokens < static_cast<int>(res.size())) {
    fprintf(stderr, "%s: too many tokens\n", __func__);
    return -(static_cast<int>(res.size()));
  }

  for (size_t i = 0; i < res.size(); i++) {
    tokens[i] = res[i];
  }

  return res.size();
}

std::vector<model_token> model_tokenize(struct model_context* ctx, const std::string& text, bool add_bos) {
  // initialize to prompt number of chars, since n_tokens <= n_prompt_chars
  std::vector<model_token> res(text.size() + static_cast<int>(add_bos));
  const int n = model_tokenize(ctx, text.c_str(), res.data(), res.size(), add_bos);
  assert(n >= 0);
  res.resize(n);

  return res;
}

int model_n_vocab(const struct model_context* ctx) { return ctx->vocab.id_to_token.size(); }

int model_n_ctx(const struct model_context* ctx) { return ctx->n_ctx; }

int model_n_embd(const struct model_context* ctx) { return ctx->model.hparams.n_embd; }

float* model_get_last_hidden_states(struct model_context* ctx) { return ctx->last_hidden_states.data(); }

const char* model_token_to_str(const struct model_context* ctx, model_token token) {
  if (token >= model_n_vocab(ctx)) {
    return nullptr;
  }

  return ctx->vocab.id_to_token[token].tok.c_str();
}

model_token model_token_nl() { return 13; }

void model_print_timings(struct model_context* ctx) {
  const int64_t t_end_us = ne_time_us();

  const int32_t n_eval = std::max(1, ctx->n_eval);
  const int32_t n_p_eval = std::max(1, ctx->n_p_eval);

  fprintf(stderr, "\n");
  fprintf(stderr, "%s:        load time = %8.2f ms\n", __func__, ctx->t_load_us / 1000.0);
  fprintf(stderr, "%s: prompt eval time = %8.2f ms / %5d tokens (%8.2f ms per token)\n", __func__,
          1e-3 * ctx->t_p_eval_us, n_p_eval, 1e-3 * ctx->t_p_eval_us / n_p_eval);
  fprintf(stderr, "%s:        eval time = %8.2f ms / %5d runs   (%8.2f ms per token)\n", __func__,
          1e-3 * ctx->t_eval_us, n_eval, 1e-3 * ctx->t_eval_us / n_eval);
  fprintf(stderr, "%s:       total time = %8.2f ms\n", __func__, (t_end_us - ctx->t_start_us) / 1000.0);
  fflush(stderr);
  printf("========== eval time log of each prediction ==========\n");
  for (int i = 0; i < ctx->eval_times.size(); ++i) {
    printf("prediction %3d, time: %.2fms\n", i, ctx->eval_times[i] / 1000.0f);
  }
  fflush(stdout);
}

void model_reset_timings(struct model_context* ctx) {
  ctx->t_start_us = ne_time_us();
  ctx->has_evaluated_once = false;
  ctx->eval_times.clear();
  ctx->t_eval_us = ctx->n_eval = 0;
  ctx->t_p_eval_us = ctx->n_p_eval = 0;
}

const char* model_print_system_info(void) {
  static std::string s;

  s = "";
  s += "AVX = " + std::to_string(ne_cpu_has_avx()) + " | ";
  s += "AVX2 = " + std::to_string(ne_cpu_has_avx2()) + " | ";
  s += "AVX512 = " + std::to_string(ne_cpu_has_avx512()) + " | ";
  s += "AVX512_VBMI = " + std::to_string(ne_cpu_has_avx512_vbmi()) + " | ";
  s += "AVX512_VNNI = " + std::to_string(ne_cpu_has_avx512_vnni()) + " | ";
  s += "FMA = " + std::to_string(ne_cpu_has_fma()) + " | ";
  s += "F16C = " + std::to_string(ne_cpu_has_f16c()) + " | ";
  s += "BLAS = " + std::to_string(ne_cpu_has_blas()) + " | ";
  s += "SSE3 = " + std::to_string(ne_cpu_has_sse3()) + " | ";
  s += "VSX = " + std::to_string(ne_cpu_has_vsx()) + " | ";

  return s.c_str();
}

// For internal test use
std::vector<std::pair<std::string, struct ne_tensor*>>& model_internal_get_tensor_map(struct model_context* ctx) {
  return ctx->model.tensors_by_name;
}

static void ne_model_kv_cache_seq_cpy(struct model_context* ctx, const model_seq_id& seq_id_src,
                                      const model_seq_id& seq_id_dst, const model_pos& p0, const model_pos& p1) {
  const uint32_t kv_n_ctx_block = ctx->kv_n_ctx_block;
  uint32_t n_head = 0;
  auto h_n_head_kv = ctx->model.hparams.n_head_kv;
  auto h_multi_query_group_num = ctx->model.hparams.multi_query_group_num;
  if (h_n_head_kv > 0) {
    n_head = h_n_head_kv;
    MODEL_ASSERT(("Invalid: multi_query_group_num > 0 and n_head_kv >0 !\n", (!h_multi_query_group_num > 0)));
  } else if (h_multi_query_group_num > 0) {
    n_head = h_multi_query_group_num;
  } else {
    n_head = ctx->model.hparams.n_head;
  }
  const uint32_t head_dim = ctx->model.hparams.n_embd / ctx->model.hparams.n_head;
  const uint32_t n_embd = n_head * head_dim;
  const uint32_t n_ctx = ctx->n_ctx;
  const size_t k_elem_size = ne_element_size(ctx->model.kv_self.k);
  const size_t v_elem_size = ne_element_size(ctx->model.kv_self.v);
#pragma omp parallel for collapse(2)
  for (int i = 0; i < ctx->model.layers.size(); ++i) {  // K
    // [head_dim, N, n_head]
    for (int nh = 0; nh < n_head; ++nh) {
      memcpy(static_cast<char*>(ctx->model.kv_self.k->data) + i * n_ctx * k_elem_size * n_embd * kv_n_ctx_block +
                 seq_id_dst * n_ctx * k_elem_size * n_embd + k_elem_size * nh * head_dim * n_ctx +
                 p0 * k_elem_size * head_dim,
             static_cast<char*>(ctx->model.kv_self.k->data) + i * n_ctx * k_elem_size * n_embd * kv_n_ctx_block +
                 seq_id_src * n_ctx * k_elem_size * n_embd + k_elem_size * nh * head_dim * n_ctx +
                 p0 * k_elem_size * head_dim,
             k_elem_size * head_dim * (p1 - p0 + 1));
    }
  }
#pragma omp parallel for collapse(2)
  for (int i = 0; i < ctx->model.layers.size(); ++i) {  // V
    // [N, head_dim, n_head] or [N, n_embd]
    for (int nm = 0; nm < n_embd; ++nm) {
      memcpy(static_cast<char*>(ctx->model.kv_self.v->data) + i * n_ctx * v_elem_size * n_embd * kv_n_ctx_block +
                 seq_id_dst * n_ctx * v_elem_size * n_embd + n_ctx * nm * v_elem_size + p0 * v_elem_size,
             static_cast<char*>(ctx->model.kv_self.v->data) + i * n_ctx * v_elem_size * n_embd * kv_n_ctx_block +
                 seq_id_src * n_ctx * v_elem_size * n_embd + n_ctx * nm * v_elem_size + p0 * v_elem_size,
             v_elem_size * (p1 - p0 + 1));
    }
  }
}

static void bestla_model_kv_cache_seq_cpy(struct model_context* ctx, const model_seq_id& seq_id_src,
                                          const model_seq_id& seq_id_dst, const model_pos& p0, const model_pos& p1) {
  const auto& kv_self = ctx->model.kv_self;
  const auto& hparams = ctx->model.hparams;
  int heads_kv = 0;
  auto h_n_head_kv = hparams.n_head_kv;
  auto h_multi_query_group_num = hparams.multi_query_group_num;
  if (h_n_head_kv > 0) {
    heads_kv = h_n_head_kv;
    MODEL_ASSERT(("Invalid: multi_query_group_num > 0 and n_head_kv >0 !\n", (!h_multi_query_group_num > 0)));
  } else if (h_multi_query_group_num > 0) {
    heads_kv = h_multi_query_group_num;
  } else {
    heads_kv = hparams.n_head;
  }
  const auto head_size = hparams.n_embd_head_k == 0 ? hparams.n_embd / hparams.n_head : hparams.n_embd_head_k;
  const int n_ctx = ctx->n_ctx;
  const auto kv_n_ctx_block = ctx->kv_n_ctx_block;
  NE_ASSERT(("Invalid end position!", n_ctx >= p1));
  kv_cache_info_t kv_cache_info;
  kv_shape_t kv_shape{
      /* .head_num = */ static_cast<uint32_t>(heads_kv),
      /* .head_size = */ static_cast<uint32_t>(head_size),
      /* .sl_kv_max = */ static_cast<uint32_t>(n_ctx),
  };
  bestla_reordered_attn_fp32_batch_kv_info(&kv_shape, &kv_cache_info);
  const auto k_bytes = kv_cache_info.k_bytes;
  const auto v_bytes = kv_cache_info.v_bytes;

  bestla_fusion_attn_fp32_batch_cpy_kv_args_t seq_cpy_param{
      /* .src = */ nullptr,
      /* .dst = */ nullptr,
      /* .heads_kv = */ heads_kv,
      /* .head_size = */ static_cast<int>(head_size),
      /* .seq_off = */ p0,
      /* .seq_size = */ p1 - p0,
      /* .seq_max = */ n_ctx,
      /* .no_zeroing = */ false,
  };
  for (int il = 0; il < ctx->model.layers.size(); ++il) {
    const auto k_data = reinterpret_cast<char*>(kv_self.k->data) + il * kv_n_ctx_block * k_bytes;
    seq_cpy_param.src = k_data + seq_id_src * k_bytes;
    seq_cpy_param.dst = k_data + seq_id_dst * k_bytes;
    bestla_fusion_attn_fp32_batch_cpy_k(&seq_cpy_param);

    const auto v_data = reinterpret_cast<char*>(kv_self.v->data) + il * kv_n_ctx_block * v_bytes;
    seq_cpy_param.src = v_data + seq_id_src * v_bytes;
    seq_cpy_param.dst = v_data + seq_id_dst * v_bytes;
    bestla_fusion_attn_fp32_batch_cpy_v(&seq_cpy_param);
  }
}

void model_kv_cache_seq_cpy(struct model_context* ctx, const model_seq_id& seq_id_src, const model_seq_id& seq_id_dst,
                            const model_pos& p0, const model_pos& p1) {
  if (ctx->model.kv_self.k->type != NE_TYPE_BTLA)
    ne_model_kv_cache_seq_cpy(ctx, seq_id_src, seq_id_dst, p0, p1);
  else
    bestla_model_kv_cache_seq_cpy(ctx, seq_id_src, seq_id_dst, p0, p1);
}

static ne_tensor* ne_model_kv_cache_seq_concat(struct ne_cgraph* cgraph, struct model_context* moctx,
                                               struct ne_context* nectx, const int64_t& ne0, const int64_t& ne1,
                                               const int64_t& ne2, const int64_t& ne3,
                                               const std::vector<int>& block_ids, const int& layer_idx,
                                               const bool& concat_k) {
  MODEL_ASSERT(ne3 == block_ids.size());
  struct ne_tensor* cache = concat_k ? moctx->model.kv_self.k : moctx->model.kv_self.v;
  // K = [head_dim, n_past+N, n_head, batch_size]
  // V = [N_past+N, head_dim, n_head, batch_size]
  const uint32_t n_embd_kv = concat_k ? ne0 * ne2 : ne1 * ne2;
  struct ne_tensor* dst = nullptr;
  if (concat_k) {
    MODEL_ASSERT(ne1 <= moctx->n_ctx);
  } else {
    MODEL_ASSERT(ne0 <= moctx->n_ctx);
  }
  const size_t elem_size = ne_element_size(cache);
  const size_t nb1 = concat_k ? elem_size * ne0 : elem_size * moctx->n_ctx;
  const size_t nb2 = concat_k ? nb1 * moctx->n_ctx : nb1 * ne1;
  const size_t nb3 = nb2 * ne2;
  int cont_bs = 1;
  int start_idx = block_ids[0];
  int id = 1;
  size_t dst_off = 0;
  while (id < block_ids.size()) {
    if (block_ids[id] - block_ids[id - 1] <= 1) {
      cont_bs++;
      id++;
      continue;
    } else {
      if (dst == nullptr) {
        dst = ne_new_tensor_4d(nectx, cache->type, ne0, ne1, ne2, ne3, NE_SIZE_CALC);
      }
      struct ne_tensor* dst_i = ne_view_4d(nectx, dst, ne0, ne1, ne2, cont_bs, elem_size * ne0, elem_size * ne0 * ne1,
                                           elem_size * ne0 * ne1 * ne2, dst_off);
      dst_off += elem_size * ne0 * ne1 * ne2 * cont_bs;
      size_t off = layer_idx * moctx->n_ctx * elem_size * n_embd_kv * moctx->kv_n_ctx_block +
                   start_idx * moctx->n_ctx * elem_size * n_embd_kv;
      ne_build_forward_expand(
          cgraph, ne_cpy(nectx, ne_view_4d(nectx, cache, ne0, ne1, ne2, cont_bs, nb1, nb2, nb3, off), dst_i));
      start_idx = block_ids[id];
      cont_bs = 1;
      id++;
    }
  }

  size_t off = layer_idx * moctx->n_ctx * elem_size * n_embd_kv * moctx->kv_n_ctx_block +
               start_idx * moctx->n_ctx * elem_size * n_embd_kv;
  if (start_idx == block_ids[0]) {
    // continuous among all batch tokens
    return ne_view_4d(nectx, cache, ne0, ne1, ne2, ne3, nb1, nb2, nb3, off);
  } else {
    // last cont batch
    struct ne_tensor* dst_i = ne_view_4d(nectx, dst, ne0, ne1, ne2, cont_bs, elem_size * ne0, elem_size * ne0 * ne1,
                                         elem_size * ne0 * ne1 * ne2, dst_off);
    ne_build_forward_expand(cgraph,
                            ne_cpy(nectx, ne_view_4d(nectx, cache, ne0, ne1, ne2, cont_bs, nb1, nb2, nb3, off), dst_i));
    return dst;
  }
}

ne_tensor* model_kv_cache_seq_concat(struct ne_cgraph* cgraph, struct model_context* moctx, struct ne_context* nectx,
                                     const int64_t& ne0, const int64_t& ne1, const int64_t& ne2, const int64_t& ne3,
                                     const std::vector<int>& block_ids, const int& layer_idx, const bool& concat_k) {
  if (moctx->model.kv_self.k->type != NE_TYPE_BTLA) {
    return ne_model_kv_cache_seq_concat(cgraph, moctx, nectx, ne0, ne1, ne2, ne3, block_ids, layer_idx, concat_k);
  } else {
    return nullptr;  // bestla
  }
}

std::vector<std::vector<int>> split_inputs_into_groups(const model_input* inputs, const int n_input) {
  MODEL_ASSERT(("There should be some input!", n_input > 0));
  std::vector<std::vector<int>> groups{{0}};
  for (int i = 1; i < n_input; ++i) {
    const auto last_idx = inputs[i - 1].request_idx;
    const auto curr_idx = inputs[i].request_idx;
    if (curr_idx != last_idx) {
      groups.emplace_back();  // Here is the beginning of a new group
    } else {
      MODEL_ASSERT(("n_tokens should be same", inputs[i - 1].n_tokens == inputs[i].n_tokens));
      MODEL_ASSERT(("n_past should be same", inputs[i - 1].n_past == inputs[i].n_past));
    }
    groups.back().push_back(i);
  }
  return groups;
}
