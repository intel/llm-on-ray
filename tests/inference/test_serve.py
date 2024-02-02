import subprocess
import pytest

# Config matrix
config_file_array = ["inference/models/gpt2.yaml", None]
model_id_or_path_array = ["gpt2", None]
models_array = ["gpt2", "gpt2 gpt-j-6b", "gpt2 bloom-560m falcon-7b"]
port_array = [8000, None]
route_prefix_array = [None]
cpus_per_worker_array = [24, None]
gpus_per_worker_array = [0, None]
hpus_per_worker_array = [0, None]
deepspeed_array = [True, False]
workers_per_group_array = [2, None]
ipex_array = [True, False]
device_array = ["cpu", None]
serve_local_only_array = [True, False]
simple_array = [True, False]
keep_serve_termimal_array = [True, False]


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
        for config_file in config_file_array
        for model_id_or_path in model_id_or_path_array
        for models in models_array
        for port in port_array
        for route_prefix in route_prefix_array
        for cpus_per_worker in cpus_per_worker_array
        for gpus_per_worker in gpus_per_worker_array
        for hpus_per_worker in hpus_per_worker_array
        for deepspeed in deepspeed_array
        for workers_per_group in workers_per_group_array
        for ipex in ipex_array
        for device in device_array
        for serve_local_only in serve_local_only_array
        for simple in simple_array
        for keep_serve_termimal in keep_serve_termimal_array
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
    cmd_serve = ["python", "../inference/serve.py"]
    if config_file is not None:
        cmd_serve.append["--config_file", config_file]
    if model_id_or_path is not None:
        cmd_serve.append["--model_id_or_path", model_id_or_path]
    if models is not None:
        cmd_serve.append["--models", models]
    if port is not None:
        cmd_serve.append["--port", port]
    if route_prefix is not None:
        cmd_serve.append["--route_prefix", route_prefix]
    if cpus_per_worker is not None:
        cmd_serve.append["--cpus_per_worker", cpus_per_worker]
    if gpus_per_worker is not None:
        cmd_serve.append["--gpus_per_worker", gpus_per_worker]
    if hpus_per_worker is not None:
        cmd_serve.append["--hpus_per_worker", hpus_per_worker]
    if deepspeed:
        cmd_serve.append["--deepspeed"]
    if workers_per_group is not None:
        cmd_serve.append["--workers_per_group", workers_per_group]
    if ipex:
        cmd_serve.append["--ipex"]
    if device is not None:
        cmd_serve.append["--device", device]
    if serve_local_only:
        cmd_serve.append["--serve_local_only"]
    if simple:
        cmd_serve.append["--simple"]
    if keep_serve_termimal:
        cmd_serve.append["--keep_serve_termimal"]

    result_serve = subprocess.run(cmd_serve, capture_output=True, text=True)

    # TODO: Find a better way to assert the result, like checking processes etc.
    assert "Error" not in result_serve.stderr
