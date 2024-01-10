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
from utils import get_deployment_actor_options
from pydantic_yaml import parse_yaml_raw_as
from api_server_simple import serve_run
from api_server_openai import openai_serve_run
from predictor_deployment import PredictorDeployment
from inference.inference_config import ModelDescription, InferenceConfig, all_models


def get_deployed_models(args):
    # serve all pre-defined models, or model from MODEL_TO_SERVE env, if no model argument specified
    if args.model is None and args.config_file is None:
        model_list = all_models
    else:
        # config_file has precedence over others
        if args.config_file:
            print("reading from config file, " + args.config_file)
            with open(args.config_file, "r") as f:
                infer_conf = parse_yaml_raw_as(InferenceConfig, f)
        else:  # args.model should be set
            print("reading from command line, " + args.model)
            model_desc = ModelDescription()
            model_desc.model_id_or_path = args.model
            model_desc.tokenizer_name_or_path = (
                args.tokenizer if args.tokenizer is not None else args.model
            )
            infer_conf = InferenceConfig(model_description=model_desc)
            infer_conf.host = "127.0.0.1" if args.serve_local_only else "0.0.0.0"
            infer_conf.port = args.port
            rp = args.route_prefix if args.route_prefix else ""
            infer_conf.route_prefix = "/{}".format(rp)
            infer_conf.name = rp
            infer_conf.ipex.enabled = args.ipex
        model_list = {}
        model_list[infer_conf.name] = infer_conf

    deployments = {}
    for model_id, infer_conf in model_list.items():
        ray_actor_options = get_deployment_actor_options(infer_conf)
        deployments[model_id] = PredictorDeployment.options(
            ray_actor_options=ray_actor_options
        ).bind(infer_conf)
    return deployments, model_list


# make it unittest friendly
def main(argv=None):
    # args
    import argparse

    parser = argparse.ArgumentParser(description="Model Serve Script", add_help=False)
    parser.add_argument(
        "--config_file",
        type=str,
        help="inference configuration file in YAML. If specified, all other arguments are ignored",
    )
    parser.add_argument("--model", default=None, type=str, help="model name or path")
    parser.add_argument("--tokenizer", default=None, type=str, help="tokenizer name or path")
    parser.add_argument("--port", default=8000, type=int, help="the port of deployment address")
    parser.add_argument(
        "--route_prefix",
        default=None,
        type=str,
        help="the route prefix for HTTP requests.",
    )
    parser.add_argument("--cpus_per_worker", default="24", type=int, help="cpus per worker")
    parser.add_argument(
        "--gpus_per_worker",
        default=0,
        type=float,
        help="gpus per worker, used when --device is cuda",
    )
    parser.add_argument(
        "--hpus_per_worker",
        default=0,
        type=float,
        help="hpus per worker, used when --device is hpu",
    )
    parser.add_argument("--deepspeed", action="store_true", help="enable deepspeed inference")
    parser.add_argument(
        "--workers_per_group",
        default="2",
        type=int,
        help="workers per group, used with --deepspeed",
    )
    parser.add_argument("--ipex", action="store_true", help="enable ipex optimization")
    parser.add_argument("--device", default="cpu", type=str, help="cpu, xpu, hpu or cuda")
    parser.add_argument(
        "--serve_local_only",
        action="store_true",
        help="only support local access to url",
    )
    parser.add_argument(
        "--serve_simple",
        action="store_true",
        help="whether to serve OpenAI-compatible API for all models or serve simple endpoint based on model conf files.",
    )
    parser.add_argument(
        "--keep_serve_terminal",
        action="store_true",
        help="whether to keep serve terminal.",
    )

    args = parser.parse_args(argv)

    ray.init(address="auto")
    deployments, model_list = get_deployed_models(args)
    if args.serve_simple:
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
