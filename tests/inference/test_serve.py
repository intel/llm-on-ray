import pytest
from unittest.mock import patch, MagicMock
from serve import main,get_deployed_models

CONFIG_FILE="../inference/models/llama-2-7b-chat-hf.yaml"
model_name="llama-2-7b-chat-hf"


# # 模拟外部依赖
# @pytest.fixture
# def mock_ray_init():
#     with patch('script.ray.init') as mock:
#         yield mock

# @pytest.fixture
# def mock_get_deployed_models():
#     with patch('script.get_deployed_models') as mock:
#         mock.return_value = ([], [])  # 返回空的部署列表和模型列表
#         yield mock

# @pytest.fixture
# def mock_serve_run():
#     with patch('script.serve_run') as mock:
#         yield mock

# @pytest.fixture
# def mock_openai_serve_run():
#     with patch('script.openai_serve_run') as mock:
#         yield mock

# 测试不同的命令行参数
@pytest.mark.parametrize("argv", [
    ['--config_file', '../inference/models/llama-2-7b-chat-hf.yaml', '--serve_simple'],
])
def test_main(argv):
    print(argv)
    main(argv)
    # captured = capsys.readouterr()  # 捕获输出

    # # 验证是否调用了 ray.init
    # mock_ray_init.assert_called_once_with(address="auto")

    # # 验证是否调用了 get_deployed_models
    # mock_get_deployed_models.assert_called_once()

    # # 根据参数验证是否调用了正确的 serve 函数
    # if '--serve_simple' in argv:
    #     mock_serve_run.assert_called_once()
    # else:
    #     mock_openai_serve_run.assert_called_once()

    # # 验证输出信息
    # if '--keep_serve_terminal' in argv:
    #     assert captured.out == ""  # 没有输出，因为应该等待用户输入
    # else:
    #     assert "Service is deployed successfully." in captured.out

# 运行测试
# if __name__ == "__main__":
#     pytest.main()


# @pytest.mark.parametrize("argvs",("--config_file", "../inference/models/llama-2-7b-chat-hf.yaml","--serve_simple"))
# def test_main(argvs, monkeypatch):
#     monkeypatch.setattr('builtins.input', lambda _: None)
#     argv=argvs
#     print(argv)
#     parser = argparse.ArgumentParser(description="Model Serve Script", add_help=False)
#     args = parser.parse_args(argv)
#     print(args.serve_simple)
#     # 调用main函数
#     main(argv)