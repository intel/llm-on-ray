# Inference with Intel Habana Gaudi

### Install Dependencies
To run inference on Gaudi, it is recommended to build a docker image first, using the Dockerfile under this folder.
But first, follow this ![guide](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html?highlight=installer#run-using-containers) to install drivers and container runtime.

### Build Docker Image

Use the following script to build it:
```bash
# cd to this folder
cd inference/habana
# you may not need proxy
docker build ../.. -f Dockerfile -t llm-on-ray:hpu --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${https_proxy}
```

### Start a Container

Now you can start a container like this:
```bash
# it might be helpful to mount the huggingface cache dir into the container
# in such case, add -v cache/on/host/:/root/.cache/huggingface/hub/
docker run -it --runtime habana -e HABANA_VISIBILE_MODULES="0,1" llm-on-ray:hpu
```

After the container is started, run `ray start --head` to start a ray cluster. Now you can run an example to verify the installation, remember you need to set the `device` field of a model config to "HPU" first.