# Ray-chatbot
## Prepare environment
```bash
pip install ray[serve] gradio
```
## Inference
### Method 1: deploy Finetune and Inference through UI
```bash
python start_ui.py
open http://localhost:8081/
```
### Method 2: inference via Ray Serve
```bash
python run_model_serve.py # wait for "Service is deployed successfully" to be printed.
python run_model_infer.py
```