# Setting up vLLM For Intel CPU

__NOTICE: The support for vLLM is experimental and subject to change.__

## Install LLM-on-Ray

Please follow [Setting up For Intel CPU/GPU/Gaudi](setup.md) to install LLM-on-Ray

## Install vLLM for Intel CPU

__NOTICE: vLLM should be installed after LLM-on-Ray for now due to vLLM may introduce newer packages.__

vLLM for CPU currently supports Intel® 4th Gen Xeon® Scalable Performance processor (formerly codenamed Sapphire Rapids) for best performance and is runnable with FP32 precision for other Xeon processors.

Currently a GNU C++ compiler with >=12.3 version is required to build and install. You can use the following commands to install GCC 12.3 and dependencies into the conda environment:

```bash
conda install -y -c conda-forge gxx=12.3 gxx_linux-64=12.3 libxcrypt
```

Then please run the following script to install vLLM for CPU into your LLM-on-Ray environment.

```bash
dev/scripts/install-vllm-cpu.sh
```

## Install vLLM Extension for Quantization (Optional)
To further speed up quantized model inference on Intel CPU, we extend vLLM to run the model decoding in own own inference engine, which is based on [https://github.com/intel/neural-speed](neural-speed).
The extension gets 2X performance improvement compared to plain vLLM CPU on Llama-2-7b and Llama-3-8b models.

Neural Speed is an innovative library designed to support the efficient inference of large language models (LLMs) on Intel platforms through the state-of-the-art (SOTA) low-bit quantization powered by
[https://github.com/intel/neural-compressor](Intel Neural Compressor). The work is inspired by [https://github.com/ggerganov/llama.cpp](llama.cpp) and further optimized for Intel platforms with our
innovations in [https://arxiv.org/abs/2311.00502](NeurIPS' 2023).

You need to first install llm-on-ray with "vllm-cpu" extra.

```bash
pip install .[vllm-cpu] --extra-index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
```

Then, install the vLLM extension and the inference engine.
```bash
cd vllm-ext
pip install . && pip install --upgrade protobuf

```

## Run

#### Serving

* Vanilla vLLM
To serve model with vLLM and simple protocol, run the following:

```bash
llm_on_ray-serve --config_file llm_on_ray/inference/models/vllm/llama-2-7b-chat-hf-vllm.yaml --simple --keep_serve_terminal
```

In the above example, `vllm` property is set to `true` in the config file for enabling vLLM.

* vLLM Extension
To serve model with vLLM extension with Intel inference engine, run with following (Note: only Llama-2-7b-chat-hf and Llama-3-8b-instruct are supported for now):

```bash
# copy quantization config file to your specific snapshot dir, for example .../snapshots/f5db02db7.../
# the quant_ns_config.json will be copied from llm_on_ray package with default config if you don't copy your desired one manually.
cp llm_on_ray/inference/models/vllm/quantization/quant_ns_config.json <your model snapshot dir>

# startup ray head node and worker node with promper configurations
# '--num-cpus 0' makes no task or actor run in head node
numactl -N 0 -m 0 ray start --head --num-cpus 0 --include-dashboard=False --disable-usage-stats --node-ip-address <your IP address>
# adjust values of NS_NUM_THREADS and numa core binding accordingly in your machine whilst keep OMP_PROC_BIND and OMP_NUM_THREADS unchanged. The example below is for standard SPR machine.
# typically number of bound physical cores should be equal to value --num-cpus. and NS_NUM_THREADS should be equal to number of bound physical cores - 2
# for deploying multiple replicas, check examples/inference/vllm/run_multiple_ray_node.sh
OMP_PROC_BIND=true NS_NUM_THREADS=44 OMP_NUM_THREADS=1 numactl -N 0 -C 0-45 -m 0 ray start --address='<your IP address>:6379' --num-cpus 46 --resources='{"inference_engine": 1}'
# startup one more ray worker node if it's open-ai compatible deployment
numactl -N 1 -C 110-111 -m 1 ray start --address='10.0.11.8:6379' --num-cpus 1 --resources='{"app_router": 1}'

# deploy model serving. Note: It includes quantizing the model on the fly based on the quant_ns_config.json if it has not been quantized.
llm_on_ray-serve --config_file llm_on_ray/inference/models/vllm/llama-2-7b-chat-hf-vllm-ns.yaml --keep_serve_terminal --max-ongoing-requests 4 --max_num_seqs 4
```

For now, only Llama-2-7b-chat-hf and Llama-3-8b-instruct are supported. But it can quickly extend to support other models.

#### Querying

To start a non-streaming query, run the following:

```bash
python examples/inference/api_server_simple/query_single.py --model_endpoint http://127.0.0.1:8000/llama-2-7b-chat-hf
```

To start a streaming query, run the following:

```bash
python examples/inference/api_server_simple/query_single.py --model_endpoint http://127.0.0.1:8000/llama-2-7b-chat-hf --streaming_response
```

## Further Configuration

Please follow [Deploying and Serving LLMs on Intel CPU/GPU/Gaudi](serve.md) document to for other configurations.
