from ray import serve
from app import RouterDeployment
from run_model_serve import PredictDeployment, _ray_env_key, _predictor_runtime_env_ipex
from inference_config import all_models
from api_backend.common.query_client import RouterQueryClient


def router_application(model_ids, hooks=None):
    """Create a Router Deployment.

    Router Deployment will point to a Serve Deployment for each specified base model,
    and have a client to query each one.
    """

    deployment_map = {}
    for model_id in model_ids:
        if model_id not in all_models.keys():
            raise(f"The config file of {model_id} doesn't exist, please add it.")
        model_config = all_models[model_id]
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



    return RouterDeployment.options(
        autoscaling_config={
            "min_replicas": 1,
            "initial_replicas": 1,
            "max_replicas": 1,
            "target_num_ongoing_requests_per_replica": 1,
        }
    ).bind(merged_client)


def run(model_ids, route_prefix="/", hooks=None):
    """Run the LLM Server on the local Ray Cluster
    """
    router_app = router_application(model_ids, hooks=hooks)

    host = "0.0.0.0"
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
    deployment_address = f"http://{host}:8000"
    return deployment_address



if __name__ == "__main__":
    run(model_ids=["gpt2"])
    input()
