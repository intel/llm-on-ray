import sys
import os
inference_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(inference_path)
import ray
from ray import serve
from run_model_serve import PredictDeployment, _ray_env_key, _predictor_runtime_env_ipex
from inference_config import ModelDescription, InferenceConfig, all_models
from plugin.query_client import RouterQueryClient
from routers.router_app import Router, router_app
from pydantic_yaml import parse_yaml_raw_as


def router_application(model_ids, model_list, hooks=None):
    """Create a Router Deployment.

    Router Deployment will point to a Serve Deployment for each specified base model,
    and have a client to query each one.
    """

    deployment_map = {}
    for model_id in model_ids:
        if model_id not in model_list.keys():
            raise ValueError(f"The config file of {model_id} doesn't exist, please add it.")
        model_config = model_list[model_id]
        runtime_env = {_ray_env_key: {}}
        if model_config.ipex:
            runtime_env[_ray_env_key].update(_predictor_runtime_env_ipex)
        if model_config.deepspeed:
            runtime_env[_ray_env_key]["DS_ACCELERATOR"] = model_config.device
        # now PredictDeployment itself is a worker, we should require resources for it
        ray_actor_options = {"runtime_env": runtime_env}
        if model_config.device == "cpu":
            ray_actor_options["num_cpus"] = model_config.cpus_per_worker
        elif model_config.device == "cuda":
            ray_actor_options["num_gpus"] = model_config.gpus_per_worker
        elif model_config.device == "hpu":
            ray_actor_options["resources"] = {"HPU": model_config.hpus_per_worker}
        else:
            # TODO add xpu
            pass
        deployment_map[model_id] = PredictDeployment.options(ray_actor_options=ray_actor_options).bind(model_config)
    merged_client = RouterQueryClient(deployment_map)

    RouterDeployment = serve.deployment(
        route_prefix="/",
        autoscaling_config={
            "min_replicas": int(os.environ.get("ROUTER_MIN_REPLICAS", 2)),
            "initial_replicas": int(os.environ.get("ROUTER_INITIAL_REPLICAS", 2)),
            "max_replicas": int(os.environ.get("ROUTER_MAX_REPLICAS", 16)),
            "target_num_ongoing_requests_per_replica": int(
                os.environ.get("ROUTER_TARGET_NUM_ONGOING_REQUESTS_PER_REPLICA", 200)
            ),
        },
        max_concurrent_queries=1000,  # Maximum backlog for a single replica
    )(serve.ingress(router_app)(Router))

    return RouterDeployment.options(
        autoscaling_config={
            "min_replicas": 1,
            "initial_replicas": 1,
            "max_replicas": 1,
            "target_num_ongoing_requests_per_replica": 1,
        }
    ).bind(merged_client)

def main(argv=None, model_ids=None, hooks=None):
    """Run the LLM Server on the local Ray Cluster
    """
    import argparse
    parser = argparse.ArgumentParser("Model Serve Script", add_help=False)
    parser.add_argument("--config_file", type=str, help="inference configuration file in YAML. If specified, all other arguments are ignored")
    parser.add_argument("--model", default=None, type=str, help="model name or path")
    parser.add_argument("--tokenizer", default=None, type=str, help="tokenizer name or path")
    parser.add_argument("--port", default=8000, type=int, help="the port of deployment address")
    parser.add_argument("--route_prefix", default="custom_model", type=str, help="the route prefix for HTTP requests.")
    parser.add_argument("--cpus_per_worker", default="24", type=int, help="cpus per worker")
    parser.add_argument("--gpus_per_worker", default=0, type=float, help="gpus per worker, used when --device is cuda")
    parser.add_argument("--hpus_per_worker", default=0, type=float, help="hpus per worker, used when --device is hpu")
    parser.add_argument("--deepspeed", action='store_true', help="enable deepspeed inference")
    parser.add_argument("--workers_per_group", default="2", type=int, help="workers per group, used with --deepspeed")
    parser.add_argument("--ipex", action='store_true', help="enable ipex optimization")
    parser.add_argument("--device", default="cpu", type=str, help="cpu, xpu, hpu or cuda")
    parser.add_argument("--precision", default="bf16", type=str, help="fp32 or bf16")
    parser.add_argument("--serve_local_only", action="store_true", help="Only support local access to url")

    args = parser.parse_args(argv)

    # serve all pre-defined models, or model from MODEL_TO_SERVE env, if no model argument specified
    if args.model is None and args.config_file is None:
        model_list = all_models
    else:
        # config_file has precedence over others
        if args.config_file:
            print("reading from config file, " + args.config_file)
            with open(args.config_file, "r") as f:
                inferenceConfig = parse_yaml_raw_as(InferenceConfig, f)
        else: # args.model should be set
            print("reading from command line, " + args.model)
            model_desc = ModelDescription()
            model_desc.model_id_or_path = args.model
            model_desc.tokenizer_name_or_path = args.tokenizer if args.tokenizer is not None else args.model
            inferenceConfig = InferenceConfig(model_description=model_desc)
            inferenceConfig.host = "127.0.0.1" if args.serve_local_only else "0.0.0.0"
            inferenceConfig.port = args.port
            rp = args.route_prefix if args.route_prefix else "custom_model"
            inferenceConfig.route_prefix = "/{}".format(rp)
            inferenceConfig.name = rp
            inferenceConfig.ipex = args.ipex
        model_list = {}
        model_list[inferenceConfig.name] = inferenceConfig

    ray.init(address="auto")

    router_app = router_application(model_ids, model_list, hooks=hooks)

    host = "127.0.0.1" if args.serve_local_only else "0.0.0.0"
    rp = args.route_prefix if args.route_prefix else "custom_model"
    route_prefix = "/{}".format(rp)
    print("route_prefix: ", route_prefix)
    serve.run(
          router_app,
          name="router",
          route_prefix=route_prefix,
          host=host,
          _blocking=True,
      ).options(
        stream=True,
        use_new_handle_api=True,
    )
    deployment_address = f"http://{host}:{args.port}/{rp}"
    return deployment_address

if __name__ == "__main__":
    main(sys.argv[1:], model_ids=["gpt2"])
    input()
