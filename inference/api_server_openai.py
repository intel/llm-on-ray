"""
Reference:
  - https://github.com/ray-project/ray-llm/blob/b3560aa55dadf6978f0de0a6f8f91002a5d2bed1/aviary/backend/server/run.py
"""
import os
from ray import serve
from api_openai_backend.query_client import RouterQueryClient
from api_openai_backend.router_app import Router, router_app


def router_application(deployments):
    """Create a Router Deployment.

    Router Deployment will point to a Serve Deployment for each specified base model,
    and have a client to query each one.
    """
    merged_client = RouterQueryClient(deployments)

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

def openai_serve_run(deployments, host, route_prefix, port):
    router_app = router_application(deployments)

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
    msg = f"Deployment is ready at `{deployment_address}`. Service is deployed successfully."
    env_name = "KEEP_SERVE_TERMINAL"
    if env_name not in os.environ or os.environ[env_name].lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup']:
        input(msg)
    else:
        print(msg)
    return deployment_address