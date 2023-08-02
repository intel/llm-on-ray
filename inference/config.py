import os

# format
model1 = {
    "model_id_or_path": "",
    "port": "",     # port for HTTP server
    "name": "",
    "route_prefix": "",
    "chat_model": "ChatModelGptJ", # prompt processing, corresponding to chat_process.py
    "prompt": {
        "intro": "",
        "human_id": "",
        "bot_id": "",
        "stop_words": []
    }
}

all_models = {}
base_models = {}

gpt_j_finetuned_52K = {
    "model_id_or_path": "/mnt/DP_disk3/ykp/huggingface/gpt-j-6B-finetuned-52K",
    "tokenizer_name_or_path": "/mnt/DP_disk3/ykp/huggingface/gpt-j-6B-finetuned-52K",
    "port": "8000",
    "name": "gpt_j_finetuned_52K",
    "route_prefix": "/gpt-j-6B-finetuned-52K",
    "chat_model": "ChatModelGptJ",
    "prompt": {
        "intro": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n",
        "human_id": "\n### Instruction",
        "bot_id": "\n### Response",
        "stop_words": []
    }
}

gpt_j_6B = {
    "model_id_or_path": "EleutherAI/gpt-j-6b",
    "tokenizer_name_or_path": "EleutherAI/gpt-j-6b",
    "port": "8000",
    "name": "gpt-j-6B",
    "route_prefix": "/gpt-j-6B",
    "chat_model": "ChatModelGptJ",
    "prompt": {
        "intro": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n",
        "human_id": "\n### Instruction",
        "bot_id": "\n### Response",
        "stop_words": []
    }
}

gpt2 = {
    "model_id_or_path": "gpt2",
    "tokenizer_name_or_path": "gpt2",
    "port": "8000",
    "name": "gpt2",
    "route_prefix": "/gpt2",
    "chat_model": "ChatModelGptJ",
    "prompt": {
        "intro": "",
        "human_id": "",
        "bot_id": "",
        "stop_words": []
    }
}


bloom = {
    "model_id_or_path": "bigscience/bloom-560m",
    "tokenizer_name_or_path": "bigscience/bloom-560m",
    "port": "8000",
    "name": "bloom",
    "route_prefix": "/bloom",
    "chat_model": "ChatModelGptJ",
    "prompt": {
        "intro": "",
        "human_id": "",
        "bot_id": "",
        "stop_words": []
    }
}

opt = {
    "model_id_or_path": "facebook/opt-125m",
    "tokenizer_name_or_path": "facebook/opt-125m",
    "port": "8000",
    "name": "opt",
    "route_prefix": "/opt",
    "chat_model": "ChatModelGptJ",
    "prompt": {
        "intro": "",
        "human_id": "",
        "bot_id": "",
        "stop_words": []
    }
}

mpt = {
    "model_id_or_path": "mosaicml/mpt-7b",
    "tokenizer_name_or_path": "EleutherAI/gpt-neox-20b",
    "port": "8000",
    "name": "mpt",
    "route_prefix": "/mpt",
    "chat_model": "ChatModelGptJ",
    "prompt": {
        "intro": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n",
        "human_id": "\n### Instruction",
        "bot_id": "\n### Response",
        "stop_words": []
    },
    "trust_remote_code": True,
}

_models = {
    "gpt-j-6B": gpt_j_6B,
    "gpt2": gpt2,
    "bloom": bloom,
    "opt": opt,
    "mpt": mpt
}

env_model = "MODEL_TO_SERVE"
if env_model in os.environ:
    all_models[os.environ[env_model]] = _models[os.environ[env_model]]
else:
    # all_models["gpt-j-6B-finetuned-52K"] = gpt_j_finetuned_52K
    all_models = _models.copy()

base_models["gpt2"] = gpt2
base_models["gpt-j-6B"] = gpt_j_6B
