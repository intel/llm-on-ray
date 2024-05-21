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

# Please input your path
model_base_path = ""
vs_path = ""


emb_model_name = "deepseek-coder-33b-instruct"
emb_model_path = os.path.join(model_base_path, emb_model_name)

index_name = "velox-doc_deepseek-coder-33b-instruct"
index_path = os.path.join(vs_path, index_name)
