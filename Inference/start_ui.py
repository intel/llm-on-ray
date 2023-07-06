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
from ray.air import session
from ray.tune import Stopper
from multiprocessing import Process, Queue

class CustomStopper(Stopper):
    def __init__(self):
        self.should_stop = False

    def __call__(self, trial_id: str, result: dict) -> bool:
        return self.should_stop

    def stop_all(self) -> bool:
        """Returns whether to stop trials and prevent new ones from starting."""
        return self.should_stop

    def stop(self, flag):
        self.should_stop = flag
    

class ChatBotUI():

    def __init__(self, all_models: dict, base_models: dict, finetune_model_path: str, finetune_code_path: str, default_data_path: str, config: dict):
        self._all_models = all_models
        self._base_models = base_models
        self.ip_port = "http://127.0.0.1:8000"
        self.process_tool = None
        self.finetune_code_path = finetune_code_path
        self.finetuned_model_path = finetune_model_path
        self.default_data_path = default_data_path
        self.config = config
        self.stopper = CustomStopper()
        self.test_replica = 4
        self.bot_queue = list(range(self.test_replica))
        self.messages = ["What is AI?", "What is Spark?", "What is Ray?", "What is chatbot?"]

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

    def reset(self, id):
        id = int(id)
        return self.messages[id], None

    def user(self, user_message, history):
        return "", history + [[user_message, None]]


    def model_generate(self, prompt, request_url, config):
        print("prompt: ", prompt)
        prompt = self.process_tool.get_prompt(prompt)
        
        sample_input = {"text": prompt, "config": config, "stream": True}
        proxies = { "http": None, "https": None}
        outputs = requests.post(request_url, proxies=proxies, json=[sample_input], stream=True)
        outputs.raise_for_status()
        for output in outputs.iter_content(chunk_size=None, decode_unicode=True):
            # remove context
            output = output[len(prompt):]
            output = self.process_tool.convert_output(output)
            yield output

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
        outputs = self.model_generate(prompt=prompt, request_url=request_url, config=config)

        for output in outputs:
            if len(output) != 0:
                time_end = time.time()
                history[-1][1]=output
                time_spend = time_end - time_start
                yield [history, time_spend] 

    def bot_test(self, bot_queue, queue_id, history, model_endpoint, Max_new_tokens, Temperature, Top_p, Top_k):
        prompt = self.history_to_messages(history)
        request_url = model_endpoint
        time_start = time.time()
        config = {
            "max_new_tokens": Max_new_tokens,
            "temperature": Temperature,
            "top_p": Top_p,
            "top_k": Top_k,
        }
        outputs = self.model_generate(prompt=prompt, request_url=request_url, config=config)

        for output in outputs:
            if len(output) != 0:
                time_end = time.time()
                history[-1][1]=output
                time_spend = time_end - time_start
                bot_queue.put([queue_id, history, time_spend])
        bot_queue.put([queue_id, "", ""])

    def send_all_bot(self, id, history, model_endpoint, Max_new_tokens, Temperature, Top_p, Top_k):
        id = int(id)
        self.bot_queue[id] = Queue()
        p = Process(target=self.bot_test, args=(self.bot_queue[id], id, history, model_endpoint, Max_new_tokens, Temperature, Top_p, Top_k))
        p.start()
        while(True):
            res = self.bot_queue[id].get()
            if res[1] == "":
                break
            yield [res[1], res[2]]

    def finetune(self, model_name, dataset, new_model_name, batch_size, num_epochs, max_train_step, lr, worker_num, cpus_per_worker):
        origin_model_path = self._base_models[model_name]["model_id_or_path"]
        tokenizer_path = self._base_models[model_name]["tokenizer_name_or_path"]
        finetuned_model_path = os.path.join(self.finetuned_model_path, new_model_name)

        ray_config = self.config.get("ray_config")
        exist_worker = int(ray_config["scaling_config"]["num_workers"])
        exist_cpus_per_worker = int(ray_config["scaling_config"]["resources_per_worker"]["CPU"])
        if cpus_per_worker * worker_num + 1 > int(ray.available_resources()["CPU"]):
            raise gr.Error("Resources are not meeting the demand")
        if worker_num != exist_worker or cpus_per_worker != exist_cpus_per_worker:
            ray.shutdown()
            self.config["ray_config"]["init"]["runtime_env"]["env_vars"]["CCL_WORKER_COUNT"] = str(worker_num)
            self.config["ray_config"]["init"]["runtime_env"]["env_vars"]["WORLD_SIZE"] = str(worker_num)
            self.config["ray_config"]["scaling_config"]["num_workers"] = worker_num
            self.config["torch_thread_num"] = cpus_per_worker
            self.config["ray_config"]["init"]["runtime_env"]["env_vars"]["OMP_NUM_THREADS"] = str(cpus_per_worker)
            self.config["ray_config"]["scaling_config"]["resources_per_worker"]["CPU"] = cpus_per_worker
            new_ray_init_config = self.config["ray_config"]["init"]
            path = self.finetune_code_path
            ray.worker.global_worker.run_function_on_all_workers(lambda worker_info: sys.path.append(path))
            ray.init(**new_ray_init_config)
            exist_worker = worker_num
            exist_cpus_per_worker = cpus_per_worker

        self.config["datasets"]["name"]=dataset
        self.config["tokenizer"]["name"]=tokenizer_path
        self.config["model"]["name"]=origin_model_path
        self.config["trainer"]["num_train_epochs"]=num_epochs
        self.config["trainer"]["output"]=finetuned_model_path
        self.config["trainer"]["dataprocesser"]["batch_size"]=batch_size
        self.config["optimizer"]["config"]["lr"]=lr
        if max_train_step==0:
            self.config["trainer"].pop("max_train_step", None)
        else:
            self.config["trainer"]["max_train_step"]=max_train_step
        
        if not hasattr(globals().get("main"), '__call__'):
            from main import main
        self.stopper.stop(False)
        self.config["ray_config"]["run_config"]["stop"] = self.stopper
        main(self.config)
        
        model_config = {
            "model_id_or_path": finetuned_model_path,
            "tokenizer_name_or_path": tokenizer_path,
            "port": "8000",
            "name": new_model_name,
            "route_prefix": "/" + new_model_name,
            "chat_model": self._base_models[model_name]["chat_model"],
            "prompt": {
                "intro": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n",
                "human_id": "\n### Instruction",
                "bot_id": "\n### Response",
                "stop_words": ["### Instruction", "# Instruction", "### Question", "##", " ="]
            }
        }
        self._all_models[new_model_name] = model_config
        return gr.Dropdown.update(choices=list(self._all_models.keys()))

    def deploy_func(self, model_name: str, replica_num: int):
        ray_config = self.config.get("ray_config")
        cpus_per_worker = int(ray_config["scaling_config"]["resources_per_worker"]["CPU"])
        # if cpus_per_worker * replica_num >= int(ray.available_resources()["CPU"]):
        #     raise gr.Error("Resources are not meeting the demand")

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
        trust_remote_code = model_config.get("trust_remote_code")
        deployment = PredictDeployment.options(num_replicas=replica_num, ray_actor_options={"runtime_env": {"pip": ["transformers==4.28.0"]}})\
                                      .bind(model_config["model_id_or_path"], model_config["tokenizer_name_or_path"], trust_remote_code, amp_enabled, amp_dtype, stop_words=stop_words)
        handle = serve.run(deployment, _blocking=True, port=model_config["port"], name=model_config["name"], route_prefix=model_config["route_prefix"])
        return self.ip_port + model_config["route_prefix"]

    def shutdown_finetune(self):
        self.stopper.stop(True)

    def shutdown_deploy(self):
        serve.shutdown()

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

            with gr.Accordion("Parameters", open=False, visible=True):
                batch_size = gr.Slider(0, 1000, 256, step=1, interactive=True, label="Batch Size")
                num_epochs = gr.Slider(1, 100, 1, step=1, interactive=True, label="Epochs")
                max_train_step = gr.Slider(0, 1000, 1, step=1, interactive=True, label="Step per Epoch", info="value 0 means use the entire dataset.")
                lr = gr.Slider(0, 0.001, 0.00001, step=0.00001, interactive=True, label="Learning Rate")
                worker_num = gr.Slider(1, 8, 2, step=1, interactive=True, label="Worker Number", info="the number of workers used for finetuning.")
                cpus_per_worker = gr.Slider(1, 100, 24, step=1, interactive=True, label="Cpus per Worker", info="the number of cpu cores used for every worker.")

            with gr.Row():
                with gr.Column(scale=0.6):
                    data_url = gr.Text(label="Data URL",
                                     value=self.default_data_path)
                with gr.Column(scale=0.2):
                    finetuned_model_name = gr.Text(label="New Model Name",
                                     value="my_alpaca")
                with gr.Column(scale=0.2, min_width=0):
                    finetune_btn = gr.Button("Start to Finetune")
                    stop_finetune_btn = gr.Button("Stop")

            step2 = "Step2: Deploy the finetuned model as an online inference service"
            gr.HTML("<h3 style='text-align: left; margin-bottom: 1rem'>"+ step2 + "</h3>")
            with gr.Row():
                with gr.Column(scale=0.8):
                    all_models_list = list(self._all_models.keys())
                    all_model_dropdown = gr.Dropdown(all_models_list, value=all_models_list[0],
                                                label="Select Model to Deploy")
                with gr.Column(scale=0.2, min_width=0):
                    deploy_btn = gr.Button("Deploy")
                    stop_deploy_btn = gr.Button("Stop")
            
            with gr.Accordion("Parameters", open=False, visible=True):
                replica_num = gr.Slider(1, 8, 1, step=1, interactive=True, label="Session Num")

            with gr.Row():
                with gr.Column(scale=1):
                    deployed_model_endpoint = gr.Text(label="Deployed Model Endpoint", value="")


            step3 = "Step3: Access the online inference service in your own application"
            gr.HTML("<h3 style='text-align: left; margin-bottom: 1rem'>"+ step3 + "</h3>")
            with gr.Accordion("Configuration", open=False, visible=True):
                max_new_tokens = gr.Slider(1, 2000, 128, step=1, interactive=True, label="Max New Tokens", info="The maximum numbers of tokens to generate.")
                Temperature = gr.Slider(0, 1, 0.7, step=0.01, interactive=True, label="Temperature", info="The value used to modulate the next token probabilities.")
                Top_p = gr.Slider(0, 1, 1.0, step=0.01, interactive=True, label="Top p", info="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to`Top p` or higher are kept for generation.")
                Top_k = gr.Slider(0, 100, 0, step=1, interactive=True, label="Top k", info="The number of highest probability vocabulary tokens to keep for top-k-filtering.")
            
            with gr.Tab("Dialogue"):
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

            with gr.Tab("Multi-Session"):
                scale_num = 1 / self.test_replica
                with gr.Row():
                    chatbots = list(range(self.test_replica))
                    msgs = list(range(self.test_replica))
                    for i in range(self.test_replica):
                        with gr.Column(scale=scale_num):
                            chatbots[i] = gr.Chatbot(elem_id="chatbot"+str(i+1), label="chatbot"+str(i+1))
                            msgs[i] = gr.Textbox(show_label=False,
                                            placeholder="Input your question and press Enter", value=self.messages[i]).style(container=False)
                with gr.Row():
                    ids = list(range(self.test_replica))
                    for i in range(self.test_replica):
                        with gr.Column(scale=scale_num):
                            ids[i] = gr.Text(value=str(i), visible=False)

                with gr.Row():
                    with gr.Column(scale=0.5):
                        send_all_btn = gr.Button("Send all requsts")
                    with gr.Column(scale=0.5):
                        reset_all_btn = gr.Button("Reset")

            msg.submit(self.user, [msg, chatbot], [msg, chatbot], queue=False).then(
                self.bot, [chatbot, deployed_model_endpoint, max_new_tokens, Temperature, Top_p, Top_k],
                           [chatbot, latency]
            )
            clear_btn.click(self.clear, None, chatbot, queue=False)
            for i in range(self.test_replica):
                reset_all_btn.click(self.reset, [ids[i]], [msgs[i], chatbots[i]], queue=False)

            send_btn.click(self.user, [msg, chatbot], [msg, chatbot], queue=False).then(
                self.bot, [chatbot, deployed_model_endpoint, max_new_tokens, Temperature, Top_p, Top_k],
                           [chatbot, latency]
            )

            for i in range(self.test_replica):
                send_all_btn.click(self.user, [msgs[i], chatbots[i]], [msgs[i], chatbots[i]], queue=False).then(
                    self.send_all_bot, [ids[i], chatbots[i], deployed_model_endpoint, max_new_tokens, Temperature, Top_p, Top_k],
                    [chatbots[i], latency]
                )

            finetune_event = finetune_btn.click(self.finetune, [base_model_dropdown, data_url, finetuned_model_name, batch_size, num_epochs, max_train_step, lr, worker_num, cpus_per_worker], [all_model_dropdown])
            stop_finetune_btn.click(fn=self.shutdown_finetune, inputs=None, outputs=None, cancels=[finetune_event])
            deploy_event = deploy_btn.click(self.deploy_func, [all_model_dropdown, replica_num], [deployed_model_endpoint])
            stop_deploy_btn.click(fn=self.shutdown_deploy, inputs=None, outputs=None, cancels=[deploy_event])

            powerby_msg = """
            The workflow is powered by Ray to provide infrastructure management, distributed training, model serving with reliability and auto scaling.
            """
            gr.HTML("<h3 style='text-align: left; margin-bottom: 1rem'>"+ powerby_msg + "</h3>")

        self.gr_chat = gr_chat

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Start UI", add_help=False)
    parser.add_argument("--finetune_model_path", default="./", type=str, help="Where to save the finetune model.")
    args = parser.parse_args()

    file_path = os.path.abspath(__file__)
    infer_path = os.path.dirname(file_path)
    finetune_path = os.path.abspath(infer_path + os.path.sep + "../Finetune")
    finetune_config_path = os.path.join(infer_path, "conf_file/finetune.conf")
    default_data_path = os.path.abspath(infer_path + os.path.sep + "../examples/data/sample_finetune_data.jsonl")

    sys.path.append(finetune_path)
    ray.worker.global_worker.run_function_on_all_workers(lambda worker_info: sys.path.append(finetune_path))

    import plugin
    from finetune import CONFIG_MAPPING, TEMPLATE_CONFIG_PATH
    template_config = plugin.parse_config(TEMPLATE_CONFIG_PATH)
    user_config = plugin.parse_config(finetune_config_path)

    config = plugin.Config()
    config.merge(template_config)
    config.merge_with_mapping(user_config, CONFIG_MAPPING, only_in_table=False)

    ray_config = config.get("ray_config")
    ray_init_config = ray_config.get("init", {})
    ray.init(**ray_init_config)

    finetune_model_path = args.finetune_model_path
    if not os.path.isabs(finetune_model_path):
        finetune_model_path = os.path.abspath(infer_path + os.path.sep + finetune_model_path)

    ui = ChatBotUI(all_models, base_models, finetune_model_path, finetune_path, default_data_path, config)
    ui.gr_chat.queue(concurrency_count=10).launch(share=True, server_port=8080, server_name="0.0.0.0")
