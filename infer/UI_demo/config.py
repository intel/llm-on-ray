# format
model1 = {
    "model_id_or_path": "",
    "port": "",     # port for HTTP server
    "name": "",
    "route_prefix": ""
}

all_models = {}

gpt_j = {
    "model_id_or_path": "/mnt/DP_disk3/ykp/huggingface/gpt-j-6B",
    "port": "8000",
    "name": "model1",
    "route_prefix": "/model1"
}

gpt_j_finetuned_2K = {
    "model_id_or_path": "/mnt/DP_disk3/ykp/huggingface/gpt-j-6B-finetuned-2K",
    "port": "8000",
    "name": "model2",
    "route_prefix": "/model2"
}

gpt_j_finetuned_52K = {
    "model_id_or_path": "/mnt/DP_disk3/ykp/huggingface/gpt-j-6B-finetuned-52K",
    "port": "8000",
    "name": "model3",
    "route_prefix": "/model3"
}

all_models["gpt_j"] = gpt_j
all_models["gpt_j_finetuned_2K"] = gpt_j_finetuned_2K
all_models["gpt_j_finetuned_52K"] = gpt_j_finetuned_52K
