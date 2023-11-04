from ray import serve
from app import RouterDeployment
from run_model_serve import PredictDeployment
from config import all_models
import torch
from common.query_client import RouterQueryClient


model_app_def = """
deployment_config:
  autoscaling_config:
    min_replicas: 1
    initial_replicas: 1
    max_replicas: 8
    target_num_ongoing_requests_per_replica: 5
    metrics_interval_s: 10.0
    look_back_period_s: 30.0
    smoothing_factor: 1.0
    downscale_delay_s: 300.0
    upscale_delay_s: 60.0
  max_concurrent_queries: 15
  ray_actor_options:
    resources: {}
multiplex_config:
    max_num_models_per_replica: 16
model_config:
  model_id: meta-llama/Llama-2-7b-hf
  model_kwargs:
    trust_remote_code: True
  max_total_tokens: 4096
  generation:
    prompt_format:
      system: "[INST] <<SYS>>\n{instruction}\n<</SYS>>\n\n"
      assistant: " {instruction} </s><s> [INST] "
      trailing_assistant: " "
      user: "{instruction} [/INST]"
      default_system_message: "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share False information."
    stopping_sequences: ["<unk>"]
scaling_config:
  num_workers: 1
  num_gpus_per_worker: 1
  num_cpus_per_worker: 8
  placement_strategy: "STRICT_PACK"
  resources_per_worker: {}
"""  # noqa




def router_application(model_id, num_deployments: int = 1, hooks=None):
    """Create a Router Deployment.

    Router Deployment will point to a Serve Deployment for each specified base model,
    and have a client to query each one.
    """

    deployment_map = {}
    model_config = all_models[model_id]
    trust_remote_code = model_config.get("trust_remote_code")
        #     .options(
        #     name=f"Deployment:model_{i}",
        # )\
    for i in range(num_deployments):
        # todo: fix
        # deployment_map[f"model_{i}"] 
        deployment_map[model_id]\
            = PredictDeployment\
            .bind(
            model_config["model_id_or_path"], model_config["tokenizer_name_or_path"], trust_remote_code,
                                            "cpu", True, torch.bfloat16,
                                            stop_words=model_config["prompt"]["stop_words"],
                                            ipex_enabled=False, deepspeed_enabled=False, cpus_per_worker=24, workers_per_group=2)

    # Merged client
    # todo: what is hooks? 
    merged_client = RouterQueryClient(deployment_map)



    return RouterDeployment.options(
        autoscaling_config={
            "min_replicas": 1,
            "initial_replicas": 1,
            "max_replicas": 1,
            "target_num_ongoing_requests_per_replica": 1,
        }
    ).bind(merged_client)


def run(model_id, num_deployments: int = 1, route_prefix="/", hooks=None):
    """Run the Mock LLM Server on the local Ray Cluster
    Args:
        num_deployments: Number of deployments to run

    """
    router_app = router_application(model_id, num_deployments, hooks=hooks)
    print("success build router app, start serve run.....")

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
    run(model_id="gpt2")
    input()
