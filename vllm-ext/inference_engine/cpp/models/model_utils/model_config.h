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
// Various helper functions and utilities

#pragma once

#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "models/model_utils/model_types.h"

#if !defined(_WIN32)
#include <stdio.h>
#include <termios.h>
#endif

struct model_params {
  std::string model_name = "";
  model_archs model_arch = MODEL_UNKNOWN;
  int n_layers;
  int32_t seed = -1;  // RNG seed
  int32_t n_threads = get_num_physical_cores();
  int32_t n_predict = -1;  // new tokens to predict
  int32_t n_ctx = 512;     // context size

  int32_t max_batch_size = 512;     // batch size for prompt processing (must be >=32 to use BLAS)

  std::string model = "";  // model path

  std::string lora_adapter = "";  // lora adapter path
  std::string lora_base = "";     // base model path for the lora adapter

  KV_MEM_TYPE memory_type = KV_MEM_TYPE_AUTO;  // Memory kv data type
  bool shift_roped_k = false;                  // whether to store non-RoPEd K cache
 
  bool do_early_stopping = false;  // whether to do early stopping
  
  float scratch_size_ratio = 1.0f;  // model memory scratch enlarge scale
};

bool gpt_params_parse(int argc, char** argv, model_params& params);

void gpt_print_usage(int argc, char** argv, const model_params& params);

//
// Vocab utils
//

std::vector<model_token> model_tokenize(struct model_context* ctx, const std::string& text, bool add_bos);

//
// Model utils
//

struct model_context* create_model_context(const model_params& params);

// KV cache elements per layer per batch per beam
void get_batch_kv_elements_from_model_params(int heads_kv, int head_size, int n_ctx, ne_type wtype, int64_t* k_size,
                                           int64_t* v_size);
