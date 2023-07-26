
## Inference
### Parameter Setting
Update `inference/conf_file/inference.conf` and `inference/config.py` as needed.
### Method 1: Deploy Finetune and Inference through UI
```bash
python start_ui.py

# Running on local URL:  http://0.0.0.0:8080
# Running on public URL: https://180cd5f7c31a1cfd3c.gradio.live
```
Access url and deploy service in a few simple clicks.
### Method 2: Inference via Terminal Execution
You can deploy a custom model by passing parameters.
```bash
python run_model_serve.py --model $model --tokenizer $tokenizer

# INFO - Deployment 'custom-model_PredictDeployment' is ready at `http://127.0.0.1:8000/custom-model`. component=serve deployment=custom-model_PredictDeployment
# Service is deployed successfully

python run_model_infer.py --model_endpoint http://127.0.0.1:8000/custom-model
```
Or you can deploy models configured in `inference/config.py` without passing parameters.
