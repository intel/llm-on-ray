With this guide, you can run llm-ray with Ray clusters on Kubernetes. The deployment on single node and multi-nodes are both supported, so you can choose it according to your needs. For more details about KubeRay, please refer to [Ray on Kubernetes](https://docs.ray.io/en/latest/cluster/kubernetes/index.html).
# Prepare code and resources
Clone repository
```bash
git clone https://github.com/intel/llm-on-ray.git
```

# Deployment
## Deployment on single node
### Build image
```bash
docker build -t llm-ray . -f llm-ray/scripts/k8s/Dockerfile  --build-arg http_proxy --build-arg https_proxy
```
### Start a kind ([install kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)) cluster and load the docker image
```bash
kind create cluster --config=llm-ray/scripts/k8s/kind_config.yaml
kind load docker-image llm-ray:latest --name kind-cluster
```
### Deploying the KubeRay operator

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm install kuberay-operator kuberay/kuberay-operator --version 0.5.0
# Confirm that the operator is running
$ kubectl get pods
NAME                                READY   STATUS    RESTARTS   AGE
kuberay-operator-86b48c7847-thhh7        1/1     Running   0          8d
```

### Deploying Ray cluster
```bash
kubectl apply -f llm-ray/scripts/k8s/deploy_cluster.yaml
# validate
$ kubectl get pods
NAME                                     READY   STATUS    RESTARTS   AGE
deployment-ray-head-5bb54f9687-k6tqn     1/1     Running   0          17s
deployment-ray-worker-784cfdfb55-dfz67   1/1     Running   0          17s
deployment-ray-worker-784cfdfb55-h5mnk   1/1     Running   0          17s
kuberay-operator-86b48c7847-thhh7        1/1     Running   0          8d
redis-cb9bfb657-mxb7z                    1/1     Running   0          17s
```
### Run
1. Exec directly into the head pod
    ```bash
    kubectl exec -it deployment-ray-head-5bb54f9687-k6tqn /bin/bash
    cd llm-ray
    python Finetune/finetune.py --finetune_config Finetune/finetune.conf
    ```
2. Ray job submission
    ```bash
    # use port-forwarding to access the Ray Dashboard port (8265 by default)
    kubectl port-forward --address 0.0.0.0 service/service-ray-cluster 8265:8265
    # run
    ray job submit --address http://localhost:8265 --working-dir llm-ray -- python Finetune/finetune.py --finetune_config Finetune/finetune.conf
    # stop it
    ray job stop raysubmit_BXKn7PjREiaQhKt2 --address http://localhost:8265
    ```

## Deployment on multi-nodes
> Need to prepare a k8s cluster on multi-nodes in advance.
```bash
$ kubectl get nodes
NAME    STATUS   ROLES                  AGE   VERSION
sr234   Ready    control-plane,master   26d   v1.20.4
sr236   Ready    <none>                 26d   v1.20.4
```
Next steps are the same as [Deployment on single node](#Deploymentonsinglenode), except that kind cluster does not need to be created.


# (Recommended) Persistent storage
An optional method of persistenting storage is provided here, you can mount folders from host when deploying k8s cluster. Without mounting it into pods, files like datasets and models need to be downloaded in pods and will be re-downloaded if pods are restarted, and outputs and checkpoints will also be lost. For more types of persistenting storage, please refer to [kubernetes volume](https://kubernetes.io/docs/concepts/storage/volumes/).

1. Assuming you already have prepared models and datasets on host, Or you want to store the outputs and checkpoints of training persistently.
    ```bash
    | -- /home
    | | -- project
    | | | -- llm-ray
    | | -- resources
    | | | -- dataset
    | | | | -- dataset_1
    | | | | | -- **.jsonl
    | | | -- model
    | | | | -- model_1  # Including model data and corresponding tokenizer
    | | | -- outputs
    | | | -- checkpoints
    ```
    > When deploying on multi-nodes, please make a copy on every node if these files cannot be shared between multi-nodes.
2. Setting mount path in config file of k8s deployment.

    First need to setting the following hostPath in `deploy_cluster.yaml`.
    ```yaml
    - name: repo
      hostPath:
        path: /home/project/llm-ray/
        type: DirectoryOrCreate
    - name: dataset
      hostPath:
        path: /home/resources/dataset/
        type: DirectoryOrCreate
    - name: model
      hostPath:
        path: /home/resources/model
        type: DirectoryOrCreate
    - name: save-path
      hostPath: 
        path: /home/resources/outputs/
        type: DirectoryOrCreate
    - name: checkpoints
      hostPath: 
        path: /home/resources/checkpoints/
        type: DirectoryOrCreate
    ```
    If you use kind to deploy cluster on single node, the hostPaths setting in `kind_config.yaml` also need to be set.
    ```yaml
     # repo
    - hostPath: /home/project/llm-ray/
      containerPath: /home/project/llm-ray/
     # dataset
    - hostPath: /home/resources/dataset/
      containerPath: /home/resources/dataset/
     # model
    - hostPath: /home/resources/model
      containerPath: /home/resources/model
     # save path
    - hostPath: /home/resources/outputs/
      containerPath: /home/resources/outputs/
     # checkpoint path
    - hostPath: /home/resources/checkpoints/
      containerPath: /home/resources/checkpoints/
    ```
4. Setting config file of Finetune
    ```yaml
    # /home/project/llm-ray/Finetune/finetune.conf
    "General": {
        "base_model": "/home/ray/resources/model/model_1",
        "output_dir": "/home/ray/resources/outputs/",
        "checkpoint_dir": "/home/ray/resources/checkpoints"
    },
    "Dataset": {
        "train_file": "/home/ray/resources/dataset/dataset_1",
    }
    ```
