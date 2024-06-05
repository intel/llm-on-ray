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
#
# ===========================================================================
#
# This file is adapted from
# https://github.com/ray-project/ray-llm/blob/b3560aa55dadf6978f0de0a6f8f91002a5d2bed1/aviary/backend/server/run.py
# Copyright 2023 Anyscale
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

from ray import serve
from llm_on_ray.inference.api_openai_backend.query_client import RouterQueryClient
from llm_on_ray.inference.api_openai_backend.router_app import Router, router_app


def router_application(deployments, model_list, max_ongoing_requests):
    """Create a Router Deployment.

    Router Deployment will point to a Serve Deployment for each specified base model,
    and have a client to query each one.
    """
    merged_client = RouterQueryClient(deployments)

    # get the value of max_ongoing_requests based on configuration of all models
    total_num_replica = 0
    max_num_concurrent_query = 0
    for _, infer_conf in model_list.items():
        if infer_conf.autoscaling_config:
            config_num_replicas = infer_conf.autoscaling_config.max_replicas
        else:
            config_num_replicas = infer_conf.num_replicas if infer_conf.num_replicas else 1
        total_num_replica += config_num_replicas
        max_num_concurrent_query = max(
            max_num_concurrent_query,
            infer_conf.max_ongoing_requests if infer_conf.max_ongoing_requests else 100,
        )

    RouterDeployment = serve.deployment(
        route_prefix="/",
        max_ongoing_requests=total_num_replica
        * (
            (max_ongoing_requests if max_ongoing_requests else max_num_concurrent_query) + 1
        ),  # Maximum backlog for a single replica
    )(serve.ingress(router_app)(Router))

    return RouterDeployment.bind(merged_client)


def openai_serve_run(deployments, model_list, host, route_prefix, port, max_ongoing_requests):
    router_app = router_application(deployments, model_list, max_ongoing_requests)

    serve.start(http_options={"host": host, "port": port})
    serve.run(
        router_app,
        name="router",
        route_prefix=route_prefix,
    ).options(
        stream=True,
        use_new_handle_api=True,
    )
    deployment_address = f"http://{host}:{port}{route_prefix}"
    print(f"Deployment is ready at `{deployment_address}`.")
    return deployment_address
