import requests
from config import all_models, base_models
import time
import os
from chat_process import ChatModelGptJ
import torch
from run_model_serve import PredictDeployment
from ray import serve
import ray
import gradio as gr
import sys
import argparse

class ChatBotUI():

    def __init__(self, all_models: dict, base_models: dict, finetune_model_path: str, config: dict):
        self._all_models = all_models
        self._base_models = base_models
        self.ip_port = "http://127.0.0.1:8000"
        self.process_tool = None
        self.finetuned_model_path = finetune_model_path
        self.config = config
        self._init_ui()

    @staticmethod
    def history_to_messages(history):
        messages = []
        for human_text, bot_text in history:
            messages.append({
                "role": "user",
                "content": human_text,
            })
            if bot_text is not None:
                messages.append({
                    "role": "assistant",
                    "content": bot_text,
                })
        return messages

    def clear(self):
        return None

    def user(self, user_message, history):
        return "", history + [[user_message, None]]


    def model_generate(self, prompt, request_url, config):
        print("prompt: ", prompt)
        prompt = self.process_tool.get_prompt(prompt)
        
        sample_input = {"text": prompt, "config": config}
        proxies = { "http": None, "https": None}
        output = requests.post(request_url, proxies=proxies, json=[sample_input]).text
        # remove context
        output = output[len(prompt):]
        output = self.process_tool.convert_output(output)
        return output

    def bot(self, history, model_endpoint, Max_new_tokens, Temperature, Top_p, Top_k):
        prompt = self.history_to_messages(history)
        request_url = model_endpoint
        time_start = time.time()
        config = {
            "max_new_tokens": Max_new_tokens,
            "temperature": Temperature,
            "top_p": Top_p,
            "top_k": Top_k,
        }
        response = self.model_generate(prompt=prompt, request_url=request_url, config=config)
        time_end = time.time()
        history[-1][1]=response
        time_spend = time_end - time_start
        return [history, time_spend, "model"] 

    def finetune(self, model_name, dataset, new_model_name):
        origin_model_path = self._base_models[model_name]["model_id_or_path"]
        tokenizer_path = self._base_models[model_name]["tokenizer_name_or_path"]
        finetuned_model_path = os.path.join(self.finetuned_model_path, new_model_name)
        config["datasets"]["name"]=dataset
        config["tokenizer"]["name"]=tokenizer_path
        config["model"]["name"]=origin_model_path
        config["trainer"]["output"]=finetuned_model_path
        main(config)
        
        model_config = {
            "model_id_or_path": finetuned_model_path,
            "tokenizer_name_or_path": tokenizer_path,
            "port": "8000",
            "name": new_model_name,
            "route_prefix": "/" + new_model_name
        }
        self._all_models[new_model_name] = model_config
        return gr.Dropdown.update(choices=list(self._all_models.keys()))

    def deploy_func(self, model_name: str):
        print("Deploying model:" + model_name)
        amp_enabled = True
        amp_dtype = torch.bfloat16

        stop_words = ["### Instruction", "# Instruction", "### Question", "##", " ="]
        model_config = self._all_models[model_name]
        print("model path: ", model_config["model_id_or_path"])

        chat_model = getattr(sys.modules[__name__], model_config["chat_model"], None)
        if chat_model is None:
            return model_name + " deployment failed. " + model_config["chat_model"] + " does not exist."
        self.process_tool = chat_model(**model_config["prompt"])
        deployment = PredictDeployment.bind(model_config["model_id_or_path"], model_config["tokenizer_name_or_path"], amp_enabled, amp_dtype, stop_words=stop_words)
        handle = serve.run(deployment, _blocking=True, port=model_config["port"], name=model_config["name"], route_prefix=model_config["route_prefix"])
        return self.ip_port + model_config["route_prefix"]

    def _init_ui(self):
        title = "LLM on Ray Workflow as a Service Demo"
        description = """
        Build your own LLM models with proprietary data, deploy an online inference service in production, all in a few simple clicks.
        """
        with gr.Blocks(css="OpenChatKit .overflow-y-auto{height:500px}", title=title) as gr_chat:
            gr.HTML("<h1 style='text-align: center; margin-bottom: 1rem'>"+ title+ "</h1>")
            gr.HTML("<h3 style='text-align: center; margin-bottom: 1rem'>"+ description + "</h3>")

            step1 = "Step1: Finetune the model with the base model and data"
            gr.HTML("<h3 style='text-align: left; margin-bottom: 1rem'>"+ step1 + "</h3>")
            with gr.Row():
                base_models_list = list(self._base_models.keys())
                base_model_dropdown = gr.Dropdown(base_models_list, value=base_models_list[0],
                                             label="Select Base Model")

            with gr.Row():
                with gr.Column(scale=0.6):
                    data_url = gr.Text(label="Data URL",
                                     value="/mnt/DP_disk3/ykp/dataset/wikitext")
                with gr.Column(scale=0.2):
                    finetuned_model_name = gr.Text(label="New Model Name",
                                     value="my_alpaca")
                with gr.Column(scale=0.2, min_width=0):
                    finetune_btn = gr.Button("Start to Finetune")


            step2 = "Step2: Deploy the finetuned model as an online inference service"
            gr.HTML("<h3 style='text-align: left; margin-bottom: 1rem'>"+ step2 + "</h3>")
            with gr.Row():
                with gr.Column(scale=0.8):
                    all_models_list = list(self._all_models.keys())
                    all_model_dropdown = gr.Dropdown(all_models_list, value=all_models_list[0],
                                                label="Select Model to Deploy")
                with gr.Column(scale=0.2, min_width=0):
                    deploy_btn = gr.Button("Deploy")
            with gr.Row():
                with gr.Column(scale=1):
                    deployed_model_endpoint = gr.Text(label="Deployed Model Endpoint", value="")


            step3 = "Step3: Access the online inference service in your own application"
            gr.HTML("<h3 style='text-align: left; margin-bottom: 1rem'>"+ step3 + "</h3>")
            chatbot = gr.Chatbot(elem_id="chatbot", label="chatbot")

            with gr.Row():
                with gr.Column(scale=0.6):
                    msg = gr.Textbox(show_label=False,
                                     placeholder="Input your question and press Enter").style(container=False)
                with gr.Column(scale=0.2, min_width=0):
                    send_btn = gr.Button("Send")
                with gr.Column(scale=0.2, min_width=0):
                    clear_btn = gr.Button("Clear")


            with gr.Row():
                with gr.Column(scale=0.1):
                    latency = gr.Text(label="Inference latency (s)", value="0", visible=False)
                with gr.Column(scale=0.1):
                    model_used = gr.Text(label="Inference Model", value="", visible=False)


            with gr.Accordion("Configuration", open=False, visible=True):
                max_new_tokens = gr.Slider(1, 2000, 128, step=1, interactive=True, label="Max New Tokens", info="The maximum numbers of tokens to generate.")
                Temperature = gr.Slider(0, 1, 0.7, step=0.01, interactive=True, label="Temperature", info="The value used to modulate the next token probabilities.")
                Top_p = gr.Slider(0, 1, 1.0, step=0.01, interactive=True, label="Top p", info="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to`Top p` or higher are kept for generation.")
                Top_k = gr.Slider(0, 100, 0, step=1, interactive=True, label="Top k", info="The number of highest probability vocabulary tokens to keep for top-k-filtering.")
            
            msg.submit(self.user, [msg, chatbot], [msg, chatbot], queue=False).then(
                self.bot, [chatbot, deployed_model_endpoint, max_new_tokens, Temperature, Top_p, Top_k],
                           [chatbot, latency, model_used]
            )
            clear_btn.click(self.clear, None, chatbot, queue=False)
            send_btn.click(self.user, [msg, chatbot], [msg, chatbot], queue=False).then(
                self.bot, [chatbot, deployed_model_endpoint, max_new_tokens, Temperature, Top_p, Top_k],
                           [chatbot, latency, model_used]
            )

            finetune_btn.click(self.finetune, [base_model_dropdown, data_url, finetuned_model_name], [all_model_dropdown])
            deploy_btn.click(self.deploy_func, [all_model_dropdown], [deployed_model_endpoint])

            powerby_msg = """
            The workflow is powered by Ray to provide infrastructure management, distributed training, model serving with reliability and auto scaling.
            """
            gr.HTML("<h3 style='text-align: left; margin-bottom: 1rem'>"+ powerby_msg + "</h3>")

        self.gr_chat = gr_chat

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Start UI', add_help=False)
    parser.add_argument('--finetune_model_path', default='./', type=str, help="Where to save the finetune model.")
    args = parser.parse_args()

    file_path = os.path.abspath(__file__)
    infer_path = os.path.dirname(file_path)
    finetune_path = os.path.abspath(infer_path + os.path.sep + "../Finetune")
    config_path = os.path.join(infer_path, "conf_file/llm_finetune_template.conf")

    sys.path.append(finetune_path)
    ray.worker.global_worker.run_function_on_all_workers(lambda worker_info: sys.path.append(finetune_path))
    from main import main

    with open(os.path.join(config_path), 'r') as f:
        config = eval(f.read())
    f.close()

    ray_config = config.get("ray_config")
    ray_init_config = ray_config.get("init", {})
    ray.init(**ray_init_config)

    finetune_model_path = args.finetune_model_path
    if not os.path.isabs(finetune_model_path):
        finetune_model_path = os.path.abspath(infer_path + os.path.sep + finetune_model_path)

    ui = ChatBotUI(all_models, base_models, finetune_model_path, config)
    ui.gr_chat.launch(share=True, server_port=8082, server_name="0.0.0.0")
