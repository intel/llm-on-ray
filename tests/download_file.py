from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Llama-2-7b-chat-hf",
    local_dir="./hg_cache/meta-llama/Llama-2-7b-chat-hf",
    local_dir_use_symlinks=False,
)
# snapshot_download(repo_id="meta-llama/Llama-2-7b-chat-hf")
