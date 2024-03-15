import subprocess
import pytest

# Config matrix
# config_file_array = ["inference/models/gpt2.yaml", None]
# model_id_or_path_array = ["gpt2", None]
# models_array = ["gpt2", "gpt2 gpt-j-6b", "gpt2 bloom-560m falcon-7b"]
# port_array = [8000, None]
# route_prefix_array = [None]
# cpus_per_worker_array = [24, None]
# gpus_per_worker_array = [0, None]
# hpus_per_worker_array = [0, None]
# deepspeed_array = [True, False]
# workers_per_group_array = [2, None]
# ipex_array = [True, False]
# device_array = ["cpu", None]
# serve_local_only_array = [True, False]
# simple_array = [True, False]
# keep_serve_termimal_array = [True, False]


# Parametrize the test function with different combinations of parameters
@pytest.mark.parametrize(
    "config_file, model_id_or_path, models, port, route_prefix, cpus_per_worker, gpus_per_worker, hpus_per_worker, deepspeed, workers_per_group, ipex, device, serve_local_only, simple, keep_serve_termimal",
    [
        (
            config_file,
            model_id_or_path,
            models,
            port,
            route_prefix,
            cpus_per_worker,
            gpus_per_worker,
            hpus_per_worker,
            deepspeed,
            workers_per_group,
            ipex,
            device,
            serve_local_only,
            simple,
            keep_serve_termimal,
        )
        for config_file in ["../.github/workflows/config/gpt2-ci.yaml"]
        for model_id_or_path in ["gpt2"]
        for models in ["gpt2"]
        for port in [8000]
        for route_prefix in [None]
        for cpus_per_worker in [24]
        for gpus_per_worker in [0]
        for hpus_per_worker in [0]
        for deepspeed in [False]
        for workers_per_group in [None]
        for ipex in [False]
        for device in ["cpu"]
        for serve_local_only in [False]
        for simple in [False]
        for keep_serve_termimal in [False]
    ],
)
def test_script(
    config_file,
    model_id_or_path,
    models,
    port,
    route_prefix,
    cpus_per_worker,
    gpus_per_worker,
    hpus_per_worker,
    deepspeed,
    workers_per_group,
    ipex,
    device,
    serve_local_only,
    simple,
    keep_serve_termimal,
):
    cmd_serve = ["python", "../llm_on_ray/inference/serve.py"]
    if config_file is not None:
        cmd_serve.append("--config_file")
        cmd_serve.append(str(config_file))
    if model_id_or_path is not None:
        cmd_serve.append("--model_id_or_path")
        cmd_serve.append(str(model_id_or_path))
    if models is not None:
        cmd_serve.append("--models")
        cmd_serve.append(str(models))
    if port is not None:
        cmd_serve.append("--port")
        cmd_serve.append(str(port))
    if route_prefix is not None:
        cmd_serve.append("--route_prefix")
        cmd_serve.append(str(route_prefix))
    if cpus_per_worker is not None:
        cmd_serve.append("--cpus_per_worker")
        cmd_serve.append(str(cpus_per_worker))
    if gpus_per_worker is not None:
        cmd_serve.append("--gpus_per_worker")
        cmd_serve.append(str(gpus_per_worker))
    if hpus_per_worker is not None:
        cmd_serve.append("--hpus_per_worker")
        cmd_serve.append(str(hpus_per_worker))
    if deepspeed:
        cmd_serve.append("--deepspeed")
    if workers_per_group is not None:
        cmd_serve.append("--workers_per_group")
        cmd_serve.append(str(workers_per_group))
    if ipex:
        cmd_serve.append("--ipex")
    if device is not None:
        cmd_serve.append("--device")
        cmd_serve.append(str(device))
    if serve_local_only:
        cmd_serve.append("--serve_local_only")
    if simple:
        cmd_serve.append("--simple")
    if keep_serve_termimal:
        cmd_serve.append("--keep_serve_termimal")

    result_serve = subprocess.run(cmd_serve, capture_output=True, text=True)

    # TODO: Find a better way to assert the result, like checking processes etc.
    assert "Error" not in result_serve.stderr
    assert result_serve == 0
    print("Asserted no erros in the result log, which is:")
    print(result_serve.stderr)
