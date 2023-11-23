import requests
from inference_config import all_models, base_models, ModelDescription, Prompt
from inference_config import InferenceConfig as FinetunedConfig
import time
import os
from chat_process import ChatModelGptJ, ChatModelLLama
import torch
from run_model_serve import PredictDeployment
from ray import serve
import ray
import gradio as gr
import sys
import argparse
from ray.tune import Stopper
from ray.train.base_trainer import TrainingFailedError
from ray.tune.logger import LoggerCallback
from multiprocessing import Process, Queue
from ray.util import queue
import paramiko
from html_format import cpu_memory_html, ray_status_html, custom_css
from typing import Any, Dict

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


@ray.remote
class Progress_Actor():
    def __init__(self, config) -> None:
        self.config = config

    def track_progress(self):
        if "epoch_value" not in self.config:
            return -1,-1,-1,-1
        if not self.config["epoch_value"].empty():
            total_epochs = self.config["total_epochs"].get(block=False)
            total_steps = self.config["total_steps"].get(block=False)
            value_epoch = self.config["epoch_value"].get(block=False)
            value_step = self.config["step_value"].get(block=False)
            return total_epochs, total_steps, value_epoch, value_step
        return -1, -1, -1, -1


class LoggingCallback(LoggerCallback):
    def __init__(self, config) -> None:
        self.config = config
        self.results = []

    def log_trial_result(self, iteration: int, trial: "Trial", result: Dict):
        if "train_epoch" in trial.last_result:
            self.config["epoch_value"].put(trial.last_result["train_epoch"] + 1, block=False)
            self.config["total_epochs"].put(trial.last_result["total_epochs"], block=False)
            self.config["step_value"].put(trial.last_result["train_step"] + 1, block=False)
            self.config["total_steps"].put(trial.last_result["total_steps"], block=False)

    def get_result(self):
        return self.results


class ChatBotUI():
    def __init__(
        self,
        all_models: dict[str, FinetunedConfig],
        base_models: dict[str, FinetunedConfig],
        finetune_model_path: str,
        finetuned_checkpoint_path: str,
        repo_code_path: str,
        default_data_path: str,
        config: dict,
        head_node_ip: str,
        node_port: str,
        node_user_name: str,
        conda_env_name: str,
        master_ip_port: str
    ):
        self._all_models = all_models
        self._base_models = base_models
        self.finetuned_model_path = finetune_model_path
        self.finetuned_checkpoint_path = finetuned_checkpoint_path
        self.repo_code_path = repo_code_path
        self.default_data_path = default_data_path
        self.config = config
        self.head_node_ip = head_node_ip
        self.node_port = node_port
        self.user_name = node_user_name
        self.conda_env_name = conda_env_name
        self.master_ip_port = master_ip_port
        self.ray_nodes = ray.nodes()
        self.ssh_connect = [None] * (len(self.ray_nodes)+1)
        self.ip_port = "http://127.0.0.1:8000"
        self.stopper = CustomStopper()
        self.test_replica = 4
        self.bot_queue = list(range(self.test_replica))
        self.messages = ["What is AI?", "What is Spark?", "What is Ray?", "What is chatbot?"]
        self.process_tool = None
        self.finetune_actor = None
        self.finetune_status = False

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
            if prompt in output:
                output = output[len(prompt):]
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
                if history[-1][1] is None:
                    history[-1][1]=output
                else:
                    history[-1][1]+=output
                history[-1][1] = self.process_tool.convert_output(history[-1][1])
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
                if history[-1][1] is None:
                    history[-1][1]=output
                else:
                    history[-1][1]+=output
                history[-1][1] = self.process_tool.convert_output(history[-1][1])
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
        model_desc = self._base_models[model_name].model_description
        origin_model_path = model_desc.model_id_or_path
        tokenizer_path = model_desc.tokenizer_name_or_path
        gpt_base_model = model_desc.gpt_base_model
        last_gpt_base_model = False
        finetuned_model_path = os.path.join(self.finetuned_model_path, model_name, new_model_name)
        finetuned_checkpoint_path = os.path.join(self.finetuned_checkpoint_path, model_name, new_model_name) if self.finetuned_checkpoint_path != "" else None

        finetune_config = self.config.copy()
        training_config = finetune_config.get("Training")
        exist_worker = int(training_config["num_training_workers"])
        exist_cpus_per_worker = int(training_config["resources_per_worker"]["CPU"])

        ray_resources = ray.available_resources()
        if "CPU" not in ray_resources or cpus_per_worker * worker_num + 1 > int(ray.available_resources()["CPU"]):
            raise gr.Error("Resources are not meeting the demand")
        if worker_num != exist_worker or cpus_per_worker != exist_cpus_per_worker or not (gpt_base_model and last_gpt_base_model):
            ray.shutdown()
            new_ray_init_config = {
                "runtime_env": {
                    "env_vars": {
                        "OMP_NUM_THREADS": str(cpus_per_worker), 
                        "ACCELERATE_USE_CPU": "True", 
                        "ACCELERATE_MIXED_PRECISION": "no",
                        "CCL_WORKER_COUNT": "1",
                        "CCL_LOG_LEVEL": "info",
                        "WORLD_SIZE": str(worker_num),
                    }
                },
                "address": "auto",
                "_node_ip_address": "127.0.0.1",
            }
            if gpt_base_model:
                new_ray_init_config["runtime_env"]["pip"] = ["transformers==4.26.0"]
            else:
                new_ray_init_config["runtime_env"]["pip"] = ["transformers==4.31.0"]
            last_gpt_base_model = gpt_base_model
            finetune_config["Training"]["num_training_workers"] = int(worker_num)
            finetune_config["Training"]["resources_per_worker"]["CPU"] = int(cpus_per_worker)

            ray.init(**new_ray_init_config)
            exist_worker = worker_num
            exist_cpus_per_worker = cpus_per_worker

        finetune_config["Dataset"]["train_file"] = dataset
        finetune_config["General"]["base_model"] = origin_model_path
        finetune_config["Training"]["epochs"] = num_epochs
        finetune_config["General"]["output_dir"] = finetuned_model_path
        if finetuned_checkpoint_path:
            finetune_config["General"]["checkpoint_dir"] = finetuned_checkpoint_path
        finetune_config["Training"]["batch_size"] = batch_size
        finetune_config["Training"]["learning_rate"] = lr
        if max_train_step != 0:
            finetune_config["Training"]["max_train_steps"] = max_train_step

        if not hasattr(globals().get("main"), '__call__'):
            from finetune.finetune import main
        finetune_config["total_epochs"] = queue.Queue(actor_options={"resources": {"queue_hardware": 1}})
        finetune_config["total_steps"] = queue.Queue(actor_options={"resources": {"queue_hardware": 1}})
        finetune_config["epoch_value"] = queue.Queue(actor_options={"resources": {"queue_hardware": 1}})
        finetune_config["step_value"] = queue.Queue(actor_options={"resources": {"queue_hardware": 1}})
        self.finetune_actor = Progress_Actor.options(resources={"queue_hardware": 1}).remote(finetune_config)

        callback = LoggingCallback(finetune_config)
        finetune_config["run_config"] = {}
        finetune_config["run_config"]["callbacks"] = [callback]
        self.stopper.stop(False)
        finetune_config["run_config"]["stop"] = self.stopper
        self.finetune_status = False
        # todo: a more reasonable solution is needed
        try:
            if main is None:
                raise Exception("An error occurred, main cannot be null")
            results = main(finetune_config)
            if results.metrics["done"]:
                self.finetune_status = True
        except TrainingFailedError as e:
            self.finetune_status = True
            print("An error occurred, possibly due to failed recovery")
            print(e)

        finetune_config["total_epochs"].shutdown(force=True)
        finetune_config["total_steps"].shutdown(force=True)
        finetune_config["epoch_value"].shutdown(force=True)
        finetune_config["step_value"].shutdown(force=True)
        ray.kill(self.finetune_actor)
        self.finetune_actor = None

        new_prompt = Prompt()
        new_prompt.intro = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
        new_prompt.human_id = "\n### Instruction"
        new_prompt.bot_id = "\n### Response"
        new_prompt.stop_words.extend(["### Instruction", "# Instruction", "### Question", "##", " ="])
        new_model_desc = ModelDescription(model_id_or_path=finetuned_model_path,
                                          tokenizer_name_or_path=tokenizer_path,
                                          prompt=new_prompt,
                                          chat_processor=model_desc.chat_processor
                                          )
        new_finetuned = FinetunedConfig(name=new_model_name,
                                  route_prefix="/" + new_model_name,
                                  model_description=new_model_desc
                                  )
        self._all_models[new_model_name] = new_finetuned
        return gr.Dropdown.update(choices=list(self._all_models.keys()))

    def finetune_progress(self, progress=gr.Progress()):
        finetune_flag = False
        while True:
            time.sleep(1)
            if self.finetune_actor is None:
                if finetune_flag == False:
                    continue
                else:
                    break
            if self.finetune_status == True:
                break
            finetune_flag = True
            try:
                total_epochs, total_steps, value_epoch, value_step = ray.get(self.finetune_actor.track_progress.remote())
                if value_epoch == -1:
                    continue
                progress(float(int(value_step)/int(total_steps)), desc="Start Training: epoch "+ str(value_epoch)+" / "+str(total_epochs) +"  "+"step " + str(value_step)+ " / "+ str(total_steps))
            except Exception as e:
                progress(0, "Restarting...")
        self.finetune_status = False
        return "<h4 style='text-align: left; margin-bottom: 1rem'>Completed the fine-tuning process.</h4>"

    def deploy_func(self, model_name: str, replica_num: int, cpus_per_worker: int):
        self.shutdown_deploy()
        if cpus_per_worker * replica_num > int(ray.available_resources()["CPU"]):
            raise gr.Error("Resources are not meeting the demand")

        print("Deploying model:" + model_name)
        amp_dtype = torch.bfloat16

        stop_words = ["### Instruction", "# Instruction", "### Question", "##", " ="]
        finetuned = self._all_models[model_name]
        model_desc = finetuned.model_description
        prompt = model_desc.prompt if model_desc.prompt else {}
        print("model path: ", model_desc.model_id_or_path)

        chat_model = getattr(sys.modules[__name__], model_desc.chat_processor, None)
        if chat_model is None:
            return model_name + " deployment failed. " + model_desc.chat_processor + " does not exist."
        self.process_tool = chat_model(**prompt)

        finetuned_deploy = finetuned.copy(deep=True)
        finetuned_deploy.device = 'cpu'
        finetuned_deploy.precision = 'bf16'
        finetuned_deploy.model_description.prompt.stop_words = stop_words
        finetuned_deploy.cpus_per_worker = cpus_per_worker
        deployment = PredictDeployment.options(num_replicas=replica_num, ray_actor_options={"num_cpus": cpus_per_worker, "runtime_env": {"pip": ["transformers==4.28.0"]}})\
                                      .bind(finetuned_deploy)
        handle = serve.run(deployment, _blocking=True, port=finetuned_deploy.port, name=finetuned_deploy.name, route_prefix=finetuned_deploy.route_prefix)
        return self.ip_port + finetuned_deploy.route_prefix

    def shutdown_finetune(self):
        self.stopper.stop(True)

    def shutdown_deploy(self):
        serve.shutdown()
    
    def get_ray_cluster(self):
        command = 'conda activate ' + self.conda_env_name + '; ray status'
        stdin, stdout, stderr = self.ssh_connect[-1].exec_command(command)
        out = stdout.read().decode('utf-8')
        out_words = [word for word in out.split("\n") if 'CPU' in word][0]
        cpu_info = out_words.split(" ")[1].split("/")
        total_core = int(float(cpu_info[1]))
        used_core = int(float(cpu_info[0]))
        utilization = float(used_core/total_core)
        return ray_status_html.format(str(round(utilization*100, 1)), used_core, total_core)

    def get_cpu_memory(self, index):
        if self.ray_nodes[index]["Alive"] == "False":
            return cpu_memory_html.format(str(round(0, 1)), str(round(0, 1)))
        command = 'export TERM=xterm; echo $(top -n 1 -b | head -n 4 | tail -n 2)'
        stdin, stdout, stderr = self.ssh_connect[index].exec_command(command)
        out = stdout.read().decode('utf-8')
        out_words = out.split(" ")
        cpu_value = 100 - float(out_words[7])
        total_memory = float(out_words[20].split('+')[0])
        free_memory = float(out_words[21].split('+')[0])
        used_memory = 1 - free_memory/total_memory
        return cpu_memory_html.format(str(round(cpu_value, 1)), str(round(used_memory*100, 1)))
    
    def kill_node(self, btn_txt, index):
        serve.shutdown()
        if btn_txt=="Kill":
            index = int(index)
            command = 'conda activate ' + self.conda_env_name + '; ray stop'
            self.ssh_connect[index].exec_command(command)
            self.ray_nodes[index]["Alive"] = "False"
            time.sleep(2)
            return "Start", ""
        elif btn_txt=="Start":
            index = int(index)
            command = "conda activate " + self.conda_env_name + "; RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING=1 ray start --address=" + self.master_ip_port + r""" --resources='{"special_hardware": 2}'"""
            self.ssh_connect[index].exec_command(command)
            self.ray_nodes[index]["Alive"] = "True"
            time.sleep(2)
            return "Kill", ""
    
    def watch_node_status(self, index):
        if self.ray_nodes[index]["Alive"] == "False":
            return "<p style='color: rgb(244, 67, 54); background-color: rgba(244, 67, 54, 0.125);'>DEAD</p>"
        else:
            return "<p style='color: rgb(76, 175, 80); background-color: rgba(76, 175, 80, 0.125);'>ALIVE</p>"
    
    def _init_ui(self):
        mark_alive = None
        for index in range(len(self.ray_nodes)):
            if self.ray_nodes[index]["Alive"] == False:
                continue
            if mark_alive is None:
                mark_alive = index
            node_ip = self.ray_nodes[index]["NodeName"]
            self.ssh_connect[index] = paramiko.SSHClient()
            self.ssh_connect[index].load_system_host_keys()
            self.ssh_connect[index].set_missing_host_key_policy(paramiko.RejectPolicy())
            self.ssh_connect[index].connect(hostname=node_ip, port=self.node_port, username=self.user_name)
        self.ssh_connect[-1] = paramiko.SSHClient()
        self.ssh_connect[-1].load_system_host_keys()
        self.ssh_connect[-1].set_missing_host_key_policy(paramiko.RejectPolicy())
        self.ssh_connect[-1].connect(hostname=self.ray_nodes[mark_alive]["NodeName"], port=self.node_port, username=self.user_name)
        
        title = "LLM on Ray Workflow as a Service Demo"
        with gr.Blocks(css=custom_css,title=title) as gr_chat:
            head_content = """
                <div style="color: #fff;text-align: center;">
                    <div style="position:absolute; left:15px; top:15px; "><img  src="/file=inference/ui_images/logo.png" width="50" height="50"/></div>
                    <p style="color: #fff; font-size: 1.0rem;">LLM on Ray Workflow as a Service Demo</p> 
                    <p style="color: #fff; font-size: 0.8rem;">Build your own LLM models with proprietary data, deploy an online inference service in production, all in a few simple clicks.</p>
                </div>
            """
            foot_content = """
                <div class="footer">
                    <p>The workflow is powered by Ray to provide infrastructure management, distributed training, model serving with reliability and auto scaling.</p>
                </div>
            """
            notice = gr.Markdown(head_content, elem_classes="notice_markdown")

            with gr.Tab("Finetune"):
                step1 = "Finetune the model with the base model and data"
                gr.HTML("<h3 style='text-align: left; margin-bottom: 1rem'>"+ step1 + "</h3>")
                with gr.Row():
                    base_models_list = list(self._base_models.keys())
                    base_model_dropdown = gr.Dropdown(base_models_list, value=base_models_list[0],
                                                label="Select Base Model")

                with gr.Accordion("Parameters", open=False, visible=True):
                    batch_size = gr.Slider(0, 1000, 2, step=1, interactive=True, label="Batch Size", info="train batch size per worker.")
                    num_epochs = gr.Slider(1, 100, 1, step=1, interactive=True, label="Epochs")
                    max_train_step = gr.Slider(0, 1000, 10, step=1, interactive=True, label="Step per Epoch", info="value 0 means use the entire dataset.")
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
                
                with gr.Row():
                    finetune_res = gr.HTML("<h4 style='text-align: left; margin-bottom: 1rem'></h4>")

            with gr.Tab("Deployment"):
                step2 = "Deploy the finetuned model as an online inference service"
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
                    replica_num = gr.Slider(1, 8, 4, step=1, interactive=True, label="Maximum Concurrent Requests")

                with gr.Row():
                    with gr.Column(scale=1):
                        deployed_model_endpoint = gr.Text(label="Deployed Model Endpoint", value="")

            with gr.Tab("Inference"):
                step3 = "Access the online inference service in your own application"
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
                            msg = gr.Textbox(show_label=False, container=False,
                                            placeholder="Input your question and press Enter")
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
                            with gr.Column(scale=scale_num, min_width=1):
                                chatbots[i] = gr.Chatbot(elem_id="chatbot"+str(i+1), label="chatbot"+str(i+1), min_width=1)
                                msgs[i] = gr.Textbox(show_label=False, container=False,
                                                placeholder="Input your question and press Enter",
                                                value=self.messages[i], min_width=1)
                    with gr.Row(visible=False):
                        ids = list(range(self.test_replica))
                        for i in range(self.test_replica):
                            with gr.Column(scale=scale_num):
                                ids[i] = gr.Text(value=str(i), visible=False)

                    with gr.Row():
                        with gr.Column(scale=0.5):
                            send_all_btn = gr.Button("Send all requsts")
                        with gr.Column(scale=0.5):
                            reset_all_btn = gr.Button("Reset")
            
            with gr.Accordion("Cluster Status", open=False, visible=True):
                with gr.Row():
                    with gr.Column(scale=0.1, min_width=45):
                        with gr.Row():
                            node_pic = r"./inference/ui_images/Picture2.png"
                            gr.Image(type="pil", value=node_pic, show_label=False, min_width=45, height=45, width=45, elem_id="notshowimg", container=False)
                        with gr.Row():
                            gr.HTML("<h4 style='text-align: center; margin-bottom: 1rem'> Ray Cluster </h4>")
                    with gr.Column(scale=0.9):
                        with gr.Row():
                            with gr.Column(scale=0.05, min_width=40):
                                gr.HTML("<h4 style='text-align: right;'> cpu core</h4>")
                            with gr.Column():
                                gr.HTML(self.get_ray_cluster, elem_classes="disablegenerating", every=2)
                
                stop_btn = []
                node_status = []
                node_index = []
                for index in range(len(self.ray_nodes)):
                    if self.ray_nodes[index]["Alive"] == False:
                        continue
                    node_ip = self.ray_nodes[index]["NodeName"]
                    with gr.Row():
                        with gr.Column(scale=0.1, min_width=25):
                            with gr.Row():
                                if index==0:
                                    func = lambda: self.watch_node_status(index=0)
                                elif index==1:
                                    func = lambda: self.watch_node_status(index=1)
                                elif index==2:
                                    func = lambda: self.watch_node_status(index=2)
                                elif index==3:
                                    func = lambda: self.watch_node_status(index=3)
                                node_status.append(gr.HTML(func, elem_classes="statusstyle", every=2))
                            with gr.Row():
                                node_index.append(gr.Text(value=len(stop_btn), visible=False))
                                if node_ip == self.head_node_ip:
                                    stop_btn.append(gr.Button("Kill", interactive=False, elem_classes="btn-style"))
                                else:
                                    stop_btn.append(gr.Button("Kill", elem_classes="btn-style"))

                        with gr.Column(scale=0.065, min_width=45):
                            with gr.Row():
                                node_pic = r"./inference/ui_images/Picture1.png"
                                gr.Image(type="pil", value=node_pic, show_label=False, min_width=45, height=45, width=45, elem_id="notshowimg", container=False)
                            with gr.Row():
                                if node_ip == self.head_node_ip:
                                    gr.HTML("<h4 style='text-align: center; margin-bottom: 1rem'> head node </h4>")
                                else:
                                    gr.HTML("<h4 style='text-align: center; margin-bottom: 1rem'> work node </h4>")
                        with gr.Column(scale=0.835):
                            with gr.Row():
                                with gr.Column(scale=0.05, min_width=40):
                                    gr.HTML("<h4 style='text-align: right;'> cpu </h4>")
                                    gr.HTML("<div style='line-height:70%;'></br></div>")
                                    gr.HTML("<h4 style='text-align: right;'> memory </h4>")
                                with gr.Column():
                                    if index==0:
                                        func = lambda: self.get_cpu_memory(index=0)
                                    elif index==1:
                                        func = lambda: self.get_cpu_memory(index=1)
                                    elif index==2:
                                        func = lambda: self.get_cpu_memory(index=2)
                                    elif index==3:
                                        func = lambda: self.get_cpu_memory(index=3)
                                    gr.HTML(func, elem_classes="disablegenerating", every=2)

            msg.submit(self.user, [msg, chatbot], [msg, chatbot], queue=False).then(
                self.bot, [chatbot, deployed_model_endpoint, max_new_tokens, Temperature, Top_p, Top_k],
                           [chatbot, latency]
            )
            clear_btn.click(self.clear, None, chatbot, queue=False)

            send_btn.click(self.user, [msg, chatbot], [msg, chatbot], queue=False).then(
                self.bot, [chatbot, deployed_model_endpoint, max_new_tokens, Temperature, Top_p, Top_k],
                           [chatbot, latency]
            )

            for i in range(self.test_replica):
                send_all_btn.click(self.user, [msgs[i], chatbots[i]], [msgs[i], chatbots[i]], queue=False).then(
                    self.send_all_bot, [ids[i], chatbots[i], deployed_model_endpoint, max_new_tokens, Temperature, Top_p, Top_k],
                    [chatbots[i], latency]
                )
            for i in range(self.test_replica):
                reset_all_btn.click(self.reset, [ids[i]], [msgs[i], chatbots[i]], queue=False)
            
            for i in range(len(stop_btn)):
                stop_btn[i].click(self.kill_node, [stop_btn[i], node_index[i]], [stop_btn[i], deployed_model_endpoint])

            finetune_event = finetune_btn.click(self.finetune, [base_model_dropdown, data_url, finetuned_model_name, batch_size, num_epochs, max_train_step, lr, worker_num, cpus_per_worker], [all_model_dropdown])
            finetune_progress_event = finetune_btn.click(self.finetune_progress, None, [finetune_res])
            stop_finetune_btn.click(fn=self.shutdown_finetune, inputs=None, outputs=None, cancels=[finetune_event, finetune_progress_event])
            deploy_event = deploy_btn.click(self.deploy_func, [all_model_dropdown, replica_num, cpus_per_worker], [deployed_model_endpoint])
            stop_deploy_btn.click(fn=self.shutdown_deploy, inputs=None, outputs=None, cancels=[deploy_event])

            gr.Markdown(foot_content)

        self.gr_chat = gr_chat

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Start UI", add_help=False)
    parser.add_argument("--finetune_model_path", default="./", type=str, help="Where to save the finetune model.")
    parser.add_argument("--finetune_checkpoint_path", default="", type=str, help="Where to save checkpoints.")
    parser.add_argument("--node_port", default="22", type=str, help="The node port that ssh connects.")
    parser.add_argument("--node_user_name", default="root", type=str, help="The node user name that ssh connects.")
    parser.add_argument("--conda_env_name", default="test_gradio", type=str, help="The environment used to execute ssh commands.")
    parser.add_argument("--master_ip_port", default="None", type=str, help="The ip:port of head node to connect when restart a worker node.")
    args = parser.parse_args()

    file_path = os.path.abspath(__file__)
    infer_path = os.path.dirname(file_path)
    repo_path = os.path.abspath(infer_path + os.path.sep + "../")
    default_data_path = os.path.abspath(infer_path + os.path.sep + "../examples/data/sample_finetune_data.jsonl")

    sys.path.append(repo_path)

    finetune_config = {
        "General": {
            "config": {}
        },
        "Dataset": {
            "validation_file": None,
            "validation_split_percentage": 0
        },
        "Training": {
            "optimizer": "AdamW",
            "lr_scheduler": "linear",
            "weight_decay": 0.0,
            "device": "CPU",
            "num_training_workers": 2,
            "resources_per_worker": {
                "CPU": 24
            },
        },
        "failure_config": {
            "max_failures": 5
        }
    }

    ray_init_config = {
        "runtime_env": {
            "env_vars": {
                "OMP_NUM_THREADS": "24", 
                "ACCELERATE_USE_CPU": "True", 
                "ACCELERATE_MIXED_PRECISION": "no",
                "CCL_WORKER_COUNT": "1",
                "CCL_LOG_LEVEL": "info",
                "WORLD_SIZE": "2",
            }
        },
        "address": "auto",
        "_node_ip_address": "127.0.0.1",
    }
    context = ray.init(**ray_init_config)
    head_node_ip = context.get("address").split(":")[0]

    finetune_model_path = args.finetune_model_path
    finetune_checkpoint_path = args.finetune_checkpoint_path

    ui = ChatBotUI(all_models, base_models, finetune_model_path, finetune_checkpoint_path, repo_path, default_data_path, finetune_config, head_node_ip, args.node_port, args.node_user_name, args.conda_env_name, args.master_ip_port)
    ui.gr_chat.queue(concurrency_count=10).launch(share=True, server_port=8080, server_name="0.0.0.0")
