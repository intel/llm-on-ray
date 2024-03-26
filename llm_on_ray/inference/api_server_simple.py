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
from ray import serve


def serve_run(deployments, model_list):
    for model_id, infer_conf in model_list.items():
        print("deploy model: ", model_id)
        deployment = deployments[model_id]

        serve.start(http_options={"host": infer_conf.host, "port": infer_conf.port})
        serve.run(
            deployment,
            name=infer_conf.name,
            route_prefix=infer_conf.route_prefix,
        )
        deployment_name = infer_conf.name
        if infer_conf.host == "0.0.0.0":
            all_nodes = ray.nodes()
            for node in all_nodes:
                if "node:__internal_head__" in node["Resources"]:
                    host_ip = node["NodeManagerAddress"]
                    break
        else:
            host_ip = infer_conf.host
        url = f"http://{host_ip}:{infer_conf.port}{infer_conf.route_prefix}"
        print(f"Deployment '{deployment_name}' is ready at `{url}`.")

    return deployments
