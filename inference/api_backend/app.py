import os
from ray import serve
from routers.router_app import Router, router_app



RouterDeployment = serve.deployment(
    route_prefix="/",
    # TODO make this configurable in aviary run
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
