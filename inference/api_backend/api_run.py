from ray import serve
from app import RouterDeployment
from run_model_serve import PredictDeployment
from config import all_models
import torch
from api_backend.common.query_client import RouterQueryClient


def router_application(model_ids, hooks=None):
    """Create a Router Deployment.

    Router Deployment will point to a Serve Deployment for each specified base model,
    and have a client to query each one.
    """

    deployment_map = {}
    for model_id in model_ids:
        model_config = all_models[model_id]
        trust_remote_code = model_config.get("trust_remote_code")
        deployment_map[model_id]\
            = PredictDeployment\
            .bind(
            model_config["model_id_or_path"], model_config["tokenizer_name_or_path"], trust_remote_code,
                                            "cpu", True, torch.bfloat16,
                                            stop_words=model_config["prompt"]["stop_words"],
                                            ipex_enabled=False, deepspeed_enabled=False, cpus_per_worker=24, workers_per_group=2)

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
