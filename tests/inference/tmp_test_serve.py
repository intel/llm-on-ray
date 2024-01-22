import pytest
from serve import main

CONFIG_FILE = "../inference/models/llama-2-7b-chat-hf.yaml"
model_name = "llama-2-7b-chat-hf"


# 测试不同的命令行参数
@pytest.mark.parametrize(
    "argv",
    [
        ["--config_file", "../inference/models/llama-2-7b-chat-hf.yaml", "--serve_simple"],
    ],
)
def test_main(argv):
    print(argv)
    main(argv)
