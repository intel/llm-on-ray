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
from pydantic_yaml import parse_yaml_raw_as
from llm_on_ray.inference.utils import get_deployment_actor_options
from llm_on_ray.inference.api_server_simple import serve_run
from llm_on_ray.inference.api_server_openai import openai_serve_run
from llm_on_ray.inference.predictor_deployment import PredictorDeployment
from llm_on_ray.inference.inference_config import (
    ModelDescription,
    InferenceConfig,
    all_models,
)


def get_deployed_models(args):
    """
    The priority of how to choose models to deploy based on passed parameters:
    1. Use inference configuration file if config_file is set,
    2. If config_file is unset, serve all of the models if models is set, or GPT2 by default if it is unset.
    """
    if args.config_file is None:
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
            print("Reading from config file, " + args.config_file)
            with open(args.config_file, "r") as f:
                infer_conf = parse_yaml_raw_as(InferenceConfig, f)
        model_list = {}
        model_list[infer_conf.name] = infer_conf

    deployments = {}
    for model_id, infer_conf in model_list.items():
        ray_actor_options = get_deployment_actor_options(infer_conf)
        depolyment_config = {
            "ray_actor_options": ray_actor_options,
            "max_ongoing_requests": infer_conf.max_ongoing_requests
            if not args.max_ongoing_requests
            else args.max_ongoing_requests,
        }
        if infer_conf.autoscaling_config:
            depolyment_config["autoscaling_config"] = infer_conf.autoscaling_config.dict()
        elif infer_conf.num_replicas:
            depolyment_config["num_replicas"] = infer_conf.num_replicas
        max_num_seqs = infer_conf.vllm.max_num_seqs if not args.max_num_seqs else args.max_num_seqs
        dynamic_max_batch_size = (
            infer_conf.dynamic_max_batch_size if not args.max_batch_size else args.max_batch_size
        )
        deployments[model_id] = PredictorDeployment.options(**depolyment_config).bind(
            infer_conf, max_num_seqs, dynamic_max_batch_size
        )

    return deployments, model_list


# make it unittest friendly
def main(argv=None):
    # args
    import argparse

    parser = argparse.ArgumentParser(description="Model Serve Script", add_help=True)
    parser.add_argument(
        "--config_file",
        type=str,
        help="Inference configuration file in YAML. If specified, it will be prioritized and other configs like --models will be ignored.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["gpt2"],
        type=str,
        help=f"Only used when config_file is None, valid values can be any items in {list(all_models.keys())}.",
    )
    parser.add_argument(
        "--list_model_ids",
        action="store_true",
        help="List all supported model IDs with config file path",
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
    parser.add_argument(
        "--max_ongoing_requests",
        default=None,
        type=int,
        help="The max concurrent requests ray serve can process for all models.",
    )
    parser.add_argument(
        "--serve_local_only",
        action="store_true",
        help="Only support local access to url.",
    )
    parser.add_argument("--port", default=8000, type=int, help="The port of deployment address.")
    parser.add_argument(
        "--max_num_seqs",
        default=None,
        type=int,
        help="The batch size for vLLM. Used when vLLM is enabled.",
    )

    parser.add_argument(
        "--max_batch_size", default=None, type=int, help="The max batch size for dynamic batching."
    )

    # Print help if no arguments were provided
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args(argv)

    all_models_name = list(all_models.keys())
    if args.list_model_ids:
        for model in all_models_name:
            print(f"{model}: \tllm_on_ray/inference/models/{model}.yaml")
        sys.exit(0)

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
        print("Service is running with deployments:" + str(deployments))
        print("Service is running models:" + str(model_list))
        openai_serve_run(deployments, model_list, host, "/", args.port, args.max_ongoing_requests)

    msg = "Service is deployed successfully."
    if args.keep_serve_terminal:
        try:
            input(msg)
        except EOFError:
            return
    else:
        print(msg)


if __name__ == "__main__":
    main(sys.argv[1:])
