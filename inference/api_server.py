import os
import ray
from ray import serve


def serve_run(deployments, model_list):
    for model_id, infer_conf in model_list.items():
        print("deploy model: ", model_id)
        deployment = deployments[model_id]
        handle = serve.run(deployment, _blocking=True, host=infer_conf.host, port=infer_conf.port, name=infer_conf.name, route_prefix=infer_conf.route_prefix)
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

    msg = "Service is deployed successfully"
    env_name = "KEEP_SERVE_TERMINAL"
    if env_name not in os.environ or os.environ[env_name].lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup']:
        input(msg)
    else:
        print(msg)
    return deployments
