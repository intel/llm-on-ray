# Run Inference on Habana Gaudi

### Build Docker Image

Use the Dockerfile in this folder to build the image you need to run inference on Habana Gaudi. Alternatively, you can install the dependecies on a bare metal machine. In this case, please refer to [here](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html#build-docker-bare).

```bash
docker build \
    -f ${dockerfile} ../../ \
    -t llm-ray-habana:latest \
    --network=host \
    --build-arg http_proxy=${http_proxy} \
    --build-arg https_proxy=${https_proxy} \
    --build-arg no_proxy=${no_proxy} \
```

### Run Inference

After the image is built successfully, start a container:

```bash
docker run -it --runtime=habana -v ./llm-ray:/root/llm-ray --name="llm-ray-habana-demo" llm-ray-habana:latest 
```

Inside the container, under `~/llm-ray`, run `MODEL_TO_SERVE=bloom python run_model_serve.py --device hpu`
In addition, `--use_hpu_graphs` can be added to speedup inference. After the service is deployed, run `python run_model_infer.py` to check if work.