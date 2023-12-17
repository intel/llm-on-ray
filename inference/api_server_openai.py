import sys
import os
import ray
from ray import serve
from run_model_serve import PredictDeployment, _ray_env_key, _predictor_runtime_env_ipex
from api_backend.plugin.query_client import RouterQueryClient
from api_backend.routers.router_app import Router, router_app
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

def openai_serve_run(model_ids, model_list, host, route_prefix, port, hooks=None):
    router_app = router_application(model_ids, model_list, hooks=hooks)

    # host = "127.0.0.1" if args.serve_local_only else "0.0.0.0"
    # rp = args.route_prefix if args.route_prefix else "custom_model"
    # route_prefix = "/{}".format(rp)
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
    deployment_address = f"http://{host}:{port}{route_prefix}"
    print(f"Deployment is ready at `{deployment_address}`.")
    return deployment_address