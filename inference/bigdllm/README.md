## Serving with BigDL-LLM
[bigdl-llm](https://bigdl.readthedocs.io/en/latest/doc/LLM/index.html) is a library for running LLM (large language model) on Intel XPU (from Laptop to GPU to Cloud) using INT4 with very low latency (for any PyTorch model).

### Install llm-on-ray with BigDL-LLM
```bash
pip install .[bigdl-cpu] -f https://developer.intel.com/ipex-whl-stable-cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### Prepare Model Configuration File
You can create your model configuration file by copying either [template](../models/template/inference_config_template.yaml) or existing model files under `../modles/bigdl/`. In order to using BigDL-LLM model, you need to set "model_description :: bigdl" to "true" and "model_description :: config :: load_in_4bit" to "true".

### Deploy and Run Inference
Here, we take `../models/bigdl/mistral-7b-v0.1.yaml` as example.
```bash
cd ..
python run_model_serve.py --config_file ./models/bigdl/mistral-7b-v0.1.yaml
python run_model_infer.py --model_endpoint http://127.0.0.1:8000/mistral-7b-v0.1 --streaming_response
```
