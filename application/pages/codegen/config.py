import os

# Please input your path
model_base_path = ""
vs_path = ""


emb_model_name = "deepseek-coder-33b-instruct"
emb_model_path = os.path.join(model_base_path, emb_model_name)

index_name = "velox-doc_deepseek-coder-33b-instruct"
index_path = os.path.join(vs_path, index_name)
