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

import ray
import sys
from inference.utils import get_deployment_actor_options
from pydantic_yaml import parse_yaml_raw_as
from api_server_simple import serve_run
from api_server_openai import openai_serve_run
from predictor_deployment import PredictorDeployment
from inference.inference_config import ModelDescription, InferenceConfig, all_models


def get_deployed_models(args):
    """
    The priority of how to choose models to deploy based on passed parameters:
    1. Use inference configuration file if config_file is set,
    2. Use relevant configuration parameters to generate `InferenceConfig` if model_id_or_path is set,
    3. Serve all pre-defined models in inference/models/*.yaml, or part of them if models is set.
    """
    if args.model_id_or_path is None and args.config_file is None:
        if args.models:
            models = args.models
            all_models_name = list(all_models.keys())
            assert set(models).issubset(
                set(all_models_name)
            ), f"models must be a subset of {all_models_name} predefined by inference/models/*.yaml, but found {models}."
            model_list = {model: all_models[model] for model in models}
        else:
            model_list = all_models
    else:
        # config_file has precedence over others
        if args.config_file:
            print("reading from config file, " + args.config_file)
            with open(args.config_file, "r") as f:
                infer_conf = parse_yaml_raw_as(InferenceConfig, f)
        else:  # args.model_id_or_path should be set
            print("reading from command line, " + args.model_id_or_path)
            model_desc = ModelDescription()
            model_desc.model_id_or_path = args.model_id_or_path
            model_desc.tokenizer_name_or_path = (
                args.tokenizer_id_or_path
                if args.tokenizer_id_or_path is not None
                else args.model_id_or_path
            )
            infer_conf = InferenceConfig(model_description=model_desc)
            infer_conf.host = "127.0.0.1" if args.serve_local_only else "0.0.0.0"
            infer_conf.port = args.port
            rp = args.route_prefix if args.route_prefix else ""
            infer_conf.route_prefix = "/{}".format(rp)
            infer_conf.num_replicas = args.num_replicas
            infer_conf.name = rp
            infer_conf.ipex.enabled = args.ipex
        model_list = {}
        model_list[infer_conf.name] = infer_conf

    deployments = {}
    for model_id, infer_conf in model_list.items():
        ray_actor_options = get_deployment_actor_options(infer_conf)
        deployments[model_id] = PredictorDeployment.options(
            num_replicas=infer_conf.num_replicas, ray_actor_options=ray_actor_options
        ).bind(infer_conf)
    return deployments, model_list


# make it unittest friendly
def main(argv=None):
    # args
    import argparse

    parser = argparse.ArgumentParser(description="Model Serve Script", add_help=True)
    parser.add_argument(
        "--config_file",
        type=str,
        help="Inference configuration file in YAML. If specified, all other arguments will be ignored.",
    )
    parser.add_argument("--model_id_or_path", default=None, type=str, help="Model name or path.")
    parser.add_argument(
        "--tokenizer_id_or_path", default=None, type=str, help="Tokenizer name or path."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["gpt2"],
        type=str,
        help=f"Only used when config_file and model_id_or_path are both None, valid values can be any items in {list(all_models.keys())}.",
    )
    parser.add_argument("--port", default=8000, type=int, help="The port of deployment address.")
    parser.add_argument(
        "--route_prefix",
        default=None,
        type=str,
        help="The route prefix for HTTP requests.",
    )
    parser.add_argument(
        "--num_replicas",
        default=1,
        type=int,
        help="The number of replicas used to respond to HTTP requests.",
    )
    parser.add_argument("--cpus_per_worker", default="24", type=int, help="CPUs per worker.")
    parser.add_argument(
        "--gpus_per_worker",
        default=0,
        type=float,
        help="GPUs per worker, used when --device is cuda.",
    )
    parser.add_argument(
        "--hpus_per_worker",
        default=0,
        type=float,
        help="HPUs per worker, used when --device is hpu.",
    )
    parser.add_argument("--deepspeed", action="store_true", help="Enable deepspeed inference.")
    parser.add_argument(
        "--workers_per_group",
        default="2",
        type=int,
        help="Workers per group, used with --deepspeed.",
    )
    parser.add_argument("--ipex", action="store_true", help="Enable ipex optimization.")
    parser.add_argument("--device", default="cpu", type=str, help="cpu, xpu, hpu or cuda.")
    parser.add_argument(
        "--serve_local_only",
        action="store_true",
        help="Only support local access to url.",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Whether to serve OpenAI-compatible API for all models or serve simple endpoint based on model conf files.",
    )
    parser.add_argument(
        "--keep_serve_terminal",
        action="store_true",
        help="Whether to keep serve terminal.",
    )

    args = parser.parse_args(argv)

    ray.init(address="auto")
    deployments, model_list = get_deployed_models(args)
    if args.simple:
        # provide simple model endpoint
        # models can be served to customed URLs according to configuration files.
        serve_run(deployments, model_list)
    else:
        # provide OpenAI compatible api to run LLM models
        # all models are served under the same URL and then accessed
        # through model_id, so it needs to pass in a unified URL.
        host = "127.0.0.1" if args.serve_local_only else "0.0.0.0"
        rp = args.route_prefix if args.route_prefix else ""
        route_prefix = "/{}".format(rp)
        openai_serve_run(deployments, host, route_prefix, args.port)

    msg = "Service is deployed successfully."
    if args.keep_serve_terminal:
        input(msg)
    else:
        print(msg)


if __name__ == "__main__":
    main(sys.argv[1:])
