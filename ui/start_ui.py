#
# Copyright 2023 The LLM-on-Ray Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import requests
import time
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from inference.inference_config import all_models, ModelDescription, Prompt
from inference.inference_config import InferenceConfig as FinetunedConfig
from inference.chat_process import ChatModelGptJ, ChatModelLLama, ChatModelwithImage  # noqa: F401
from inference.predictor_deployment import PredictorDeployment
from ray import serve
import ray
import gradio as gr
import argparse
from ray.tune import Stopper
from ray.train.base_trainer import TrainingFailedError
from ray.tune.logger import LoggerCallback
from multiprocessing import Process, Queue
from ray.util import queue
import paramiko
from html_format import cpu_memory_html, ray_status_html, custom_css
from typing import Dict, List, Any
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from pyrecdp.LLM import TextPipeline
from pyrecdp.primitives.operations import (
    UrlLoader,
    DirectoryLoader,
    DocumentSplit,
    DocumentIngestion,
    YoutubeLoader,
    RAGTextFix,
)
from pyrecdp.primitives.document.reader import _default_file_readers
from pyrecdp.core.cache_utils import RECDP_MODELS_CACHE

if ('RECDP_CACHE_HOME' not in os.environ) or (not os.environ['RECDP_CACHE_HOME']):
    os.environ['RECDP_CACHE_HOME'] = os.getcwd()

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

def is_simple_api(request_url, model_name):
    if model_name is None:
        return True
    return model_name in request_url

@ray.remote
class Progress_Actor:
    def __init__(self, config) -> None:
        self.config = config

    def track_progress(self):
        if "epoch_value" not in self.config:
            return -1, -1, -1, -1
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
        self.results: List[Any] = []

    def log_trial_result(self, iteration: int, trial, result: Dict):
        if "train_epoch" in trial.last_result:
            self.config["epoch_value"].put(trial.last_result["train_epoch"] + 1, block=False)
            self.config["total_epochs"].put(trial.last_result["total_epochs"], block=False)
            self.config["step_value"].put(trial.last_result["train_step"] + 1, block=False)
            self.config["total_steps"].put(trial.last_result["total_steps"], block=False)

    def get_result(self):
        return self.results


class ChatBotUI:
    def __init__(
        self,
        all_models: Dict[str, FinetunedConfig],
        base_models: Dict[str, FinetunedConfig],
        finetune_model_path: str,
        finetuned_checkpoint_path: str,
        repo_code_path: str,
        default_data_path: str,
        default_rag_path: str,
        config: dict,
        head_node_ip: str,
        node_port: str,
        node_user_name: str,
        conda_env_name: str,
        master_ip_port: str,
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
        self.ssh_connect = [None] * (len(self.ray_nodes) + 1)
        self.ip_port = "http://127.0.0.1:8000"
        self.stopper = CustomStopper()
        self.test_replica = 4
        self.bot_queue = list(range(self.test_replica))
        self.messages = [
            "What is AI?",
            "What is Spark?",
            "What is Ray?",
            "What is chatbot?",
        ]
        self.process_tool = None
        self.finetune_actor = None
        self.finetune_status = False
        self.default_rag_path = default_rag_path
        self.embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
        self._init_ui()

    @staticmethod
    def history_to_messages(history, image = None):
        messages = []
        for human_text, bot_text in history:
            if image is not None:
                import base64
                from io import BytesIO

                buffered = BytesIO()
                image.save(buffered, format="JPEG")
                base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

                human_text = [
                    {'type': "text", "text": human_text},
                    {
                        'type': "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                ]
            messages.append(
                {
                    "role": "user",
                    "content": human_text,
                }
            )
            if bot_text is not None:
                messages.append(
                    {
                        "role": "assistant",
                        "content": bot_text,
                    }
                )                
        return messages

    @staticmethod
    def add_knowledge(prompt, enhance_knowledge):
        description = "Known knowledge: {knowledge}. Then please answer the question based on follow conversation: {conversation}."
        if not isinstance(prompt[-1]['content'], list):
            prompt[-1]['content'] = description.format(knowledge=enhance_knowledge, conversation=prompt[-1]['content'])
        return prompt    

    def clear(self):
        return (
            None,
            """| <!-- --> | <!-- --> |
                         |---|---|
                         | Total Latency [s] | - |
                         | Tokens | - |""",
        )

    def reset(self, id):
        id = int(id)
        return self.messages[id], None

    def user(self, user_message, history):
        return "", history + [[user_message, None]]

    def model_generate(self, prompt, request_url, model_name, config, simple_api=True):        
        if simple_api:
            prompt = self.process_tool.get_prompt(prompt)
            sample_input = {"text": prompt, "config": config, "stream": True}
        else:
            sample_input = {
                "model": model_name,
                "messages": prompt,
                "stream": True,
                "max_tokens": config["max_new_tokens"],
                "temperature": config["temperature"],
                "top_p": config["top_p"],
                "top_k": config["top_k"]
            }
        proxies = {"http": None, "https": None}
        print(sample_input)
        outputs = requests.post(request_url, proxies=proxies, json=sample_input, stream=True)
        outputs.raise_for_status()
        for output in outputs.iter_lines(chunk_size=None, decode_unicode=True):
            # remove context
            if simple_api:
                if prompt in output:
                    output = output[len(prompt) :]
            else:
                import json
                import re
                chunk_data = re.sub("^data: ", "", output)
                if chunk_data != "[DONE]":
                    # Get message choices from data
                    choices = json.loads(chunk_data)["choices"]
                    # Pick content from first choice
                    output = choices[0]["delta"].get("content", "")
                else:
                    output = ""
            yield output

    def bot(
        self,
        history,
        deploy_model_endpoint,
        model_endpoint,
        Max_new_tokens,
        Temperature,
        Top_p,
        Top_k,
        model_name=None,
        image=None,
        enhance_knowledge=None,
    ):
        print("request submitted!")
        request_url = model_endpoint if model_endpoint != "" else deploy_model_endpoint
        simple_api = is_simple_api(request_url, model_name)
        prompt = self.history_to_messages(history, image)
        if enhance_knowledge:
            prompt = self.add_knowledge(prompt, enhance_knowledge)
        
        time_start = time.time()
        token_num = 0
        config = {
            "max_new_tokens": Max_new_tokens,
            "temperature": Temperature,
            "do_sample": True,
            "top_p": Top_p,
            "top_k": Top_k,
            "model": model_name
        }
        print("request wip to submit")
        outputs = self.model_generate(prompt=prompt, request_url=request_url, model_name=model_name, config=config, simple_api=simple_api)

        if history[-1][1] is None:
            history[-1][1] = ""
        for output in outputs:
            if len(output) != 0:
                time_end = time.time()
                if simple_api:
                    history[-1][1] += output
                    history[-1][1] = self.process_tool.convert_output(history[-1][1])
                else:
                    history[-1][1] += output
                time_spend = round(time_end - time_start, 3)
                token_num += 1
                new_token_latency = f"""
                                    | <!-- --> | <!-- --> |
                                    |---|---|
                                    | Total Latency [s] | {time_spend} |
                                    | Tokens | {token_num} |"""
                yield [history, new_token_latency]

    def bot_test(
        self,
        bot_queue,
        queue_id,
        history,
        model_endpoint,
        Max_new_tokens,
        Temperature,
        Top_p,
        Top_k,
        model_name=None
    ):
        request_url = model_endpoint
        simple_api = is_simple_api(request_url, model_name)
        prompt = self.history_to_messages(history)
        time_start = time.time()
        config = {
            "max_new_tokens": Max_new_tokens,
            "temperature": Temperature,
            "do_sample": True,
            "top_p": Top_p,
            "top_k": Top_k,
            "model": model_name
        }
        outputs = self.model_generate(prompt=prompt, request_url=request_url, model_name=model_name, config=config, simple_api=simple_api)
        history[-1][1] = ""
        for output in outputs:
            if len(output) != 0:
                time_end = time.time()
                if simple_api:
                    history[-1][1] += output
                    history[-1][1] = self.process_tool.convert_output(history[-1][1])
                else:
                    history[-1][1] += output

                time_spend = time_end - time_start
                bot_queue.put([queue_id, history, time_spend])
        bot_queue.put([queue_id, "", ""])

    def bot_rag(
        self,
        history,
        deploy_model_endpoint,
        model_endpoint,
        Max_new_tokens,
        Temperature,
        Top_p,
        Top_k,
        rag_selector,
        rag_path,
        returned_k,
        model_name=None,
        image=None,
    ):
        enhance_knowledge = None
        if os.path.isabs(rag_path):
            tmp_folder = os.getcwd()
            load_dir = os.path.join(tmp_folder, rag_path)
        else:
            load_dir = rag_path
        if not os.path.exists(load_dir):
            raise gr.Error("The specified path does not exist")
        if rag_selector:
            question = history[-1][0]
            print("history: ", history)
            print("question: ", question)
            
            if not hasattr(self, "embeddings"):
                local_embedding_model_path = os.path.join(RECDP_MODELS_CACHE, self.embedding_model_name)
                if os.path.exists(local_embedding_model_path):
                    self.embeddings = HuggingFaceEmbeddings(model_name=local_embedding_model_path)
                else:
                    self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

            vectorstore = FAISS.load_local(load_dir, self.embeddings, index_name="knowledge_db")
            sim_res = vectorstore.similarity_search(question, k=int(returned_k))
            enhance_knowledge = ""
            for doc in sim_res:
                enhance_knowledge = enhance_knowledge + doc.page_content + ". "

        bot_generator = self.bot(
            history,
            deploy_model_endpoint,
            model_endpoint,
            Max_new_tokens,
            Temperature,
            Top_p,
            Top_k,
            model_name=model_name,
            image=image,
            enhance_knowledge=enhance_knowledge,
        )
        for output in bot_generator:
            yield output

    def regenerate(
        self,
        db_dir,
        upload_type,
        input_type,
        input_texts,
        depth,
        upload_files,
        embedding_model,
        splitter_chunk_size,
        cpus_per_worker,
    ):
        if upload_type == "Youtube":
            input_texts = input_texts.split(";")
            target_urls = [url.strip() for url in input_texts if url != ""]
            loader = YoutubeLoader(urls=target_urls)
        elif upload_type == "Web":
            input_texts = input_texts.split(";")
            target_urls = [url.strip() for url in input_texts if url != ""]
            loader = UrlLoader(urls=target_urls, max_depth=int(depth))
        else:
            if input_type == "local":
                input_texts = input_texts.split(";")
                target_folders = [folder.strip() for folder in input_texts if folder != ""]
                info_str = "Load files: "
                for folder in target_folders:
                    files = os.listdir(folder)
                    info_str = info_str + " ".join(files) + " "

                gr.Info(info_str)
                loader = DirectoryLoader(input_dir=target_folders)
            else:
                files_folder = []
                if upload_files:
                    for _, file in enumerate(upload_files):
                        files_folder.append(file.name)
                    loader = DirectoryLoader(input_files=files_folder)
                else:
                    raise gr.Warning("Can't get any uploaded files.")

        if os.path.isabs(db_dir):
            tmp_folder = os.getcwd()
            save_dir = os.path.join(tmp_folder, db_dir)
        else:
            save_dir = db_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        vector_store_type = "FAISS"
        index_name = "knowledge_db"
        text_splitter = "RecursiveCharacterTextSplitter"
        splitter_chunk_size = int(splitter_chunk_size)
        text_splitter_args = {
            "chunk_size": splitter_chunk_size,
            "chunk_overlap": 0,
            "separators": ["\n\n", "\n", " ", ""],
        }
        embeddings_type = "HuggingFaceEmbeddings"
        
        self.embedding_model_name = embedding_model
        local_embedding_model_path = os.path.join(RECDP_MODELS_CACHE, self.embedding_model_name)
        if os.path.exists(local_embedding_model_path):
            self.embeddings = HuggingFaceEmbeddings(model_name=local_embedding_model_path)
            embeddings_args = {"model_name": local_embedding_model_path}
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
            embeddings_args = {"model_name": self.embedding_model_name}
        

        pipeline = TextPipeline()
        ops = [loader]

        ops.extend(
            [
                RAGTextFix(re_sentence=True),
                DocumentSplit(text_splitter=text_splitter, text_splitter_args=text_splitter_args),
                DocumentIngestion(
                    vector_store=vector_store_type,
                    vector_store_args={"output_dir": save_dir, "index": index_name},
                    embeddings=embeddings_type,
                    embeddings_args=embeddings_args,
                    num_cpus=cpus_per_worker,
                ),
            ]
        )
        pipeline.add_operations(ops)
        pipeline.execute()
        return db_dir

    def send_all_bot(self, id, history, model_endpoint, Max_new_tokens, Temperature, Top_p, Top_k):
        id = int(id)
        self.bot_queue[id] = Queue()
        p = Process(
            target=self.bot_test,
            args=(
                self.bot_queue[id],
                id,
                history,
                model_endpoint,
                Max_new_tokens,
                Temperature,
                Top_p,
                Top_k,
            ),
        )
        p.start()
        while True:
            res = self.bot_queue[id].get()
            if res[1] == "":
                break
            yield res[1]

    def finetune(
        self,
        model_name,
        custom_model_name,
        custom_tokenizer_name,
        dataset,
        new_model_name,
        batch_size,
        num_epochs,
        max_train_step,
        lr,
        worker_num,
        cpus_per_worker_ftn,
    ):
        if model_name == "specify other models":
            model_desc = None
            origin_model_path = custom_model_name
            tokenizer_path = custom_tokenizer_name
            if "gpt" in model_name.lower() or "pythia" in model_name.lower():
                gpt_base_model = True
            else:
                gpt_base_model = False
        else:
            model_desc = self._base_models[model_name].model_description
            origin_model_path = model_desc.model_id_or_path
            tokenizer_path = model_desc.tokenizer_name_or_path
            gpt_base_model = model_desc.gpt_base_model
        last_gpt_base_model = False
        finetuned_model_path = os.path.join(self.finetuned_model_path, model_name, new_model_name)
        finetuned_checkpoint_path = (
            os.path.join(self.finetuned_checkpoint_path, model_name, new_model_name)
            if self.finetuned_checkpoint_path != ""
            else None
        )

        finetune_config = self.config.copy()
        training_config = finetune_config.get("Training")
        exist_worker = int(training_config["num_training_workers"])
        exist_cpus_per_worker_ftn = int(training_config["resources_per_worker"]["CPU"])

        ray_resources = ray.available_resources()
        if "CPU" not in ray_resources or cpus_per_worker_ftn * worker_num + 1 > int(
            ray.available_resources()["CPU"]
        ):
            num_req = cpus_per_worker_ftn * worker_num + 1
            num_act = int(ray.available_resources()['CPU'])
            error_msg = f"Resources are not meeting the demand, required num_cpu is {num_req}, actual num_cpu is {num_act}"
            raise gr.Error(error_msg)
        if (
            worker_num != exist_worker
            or cpus_per_worker_ftn != exist_cpus_per_worker_ftn
            or not (gpt_base_model and last_gpt_base_model)
        ):
            ray.shutdown()
            new_ray_init_config = {
                "runtime_env": {
                    "env_vars": {
                        "OMP_NUM_THREADS": str(cpus_per_worker_ftn),
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
            finetune_config["Training"]["resources_per_worker"]["CPU"] = int(cpus_per_worker_ftn)

            ray.init(**new_ray_init_config)
            exist_worker = worker_num
            exist_cpus_per_worker_ftn = cpus_per_worker_ftn

        finetune_config["Dataset"]["train_file"] = dataset
        finetune_config["General"]["base_model"] = origin_model_path
        finetune_config["Training"]["epochs"] = num_epochs
        finetune_config["General"]["output_dir"] = finetuned_model_path
        finetune_config["General"]["config"]["trust_remote_code"] = True
        if finetuned_checkpoint_path:
            finetune_config["General"]["checkpoint_dir"] = finetuned_checkpoint_path
        finetune_config["Training"]["batch_size"] = batch_size
        finetune_config["Training"]["learning_rate"] = lr
        if max_train_step != 0:
            finetune_config["Training"]["max_train_steps"] = max_train_step

        from finetune.finetune import main

        finetune_config["total_epochs"] = queue.Queue(
            actor_options={"resources": {"queue_hardware": 1}}
        )
        finetune_config["total_steps"] = queue.Queue(
            actor_options={"resources": {"queue_hardware": 1}}
        )
        finetune_config["epoch_value"] = queue.Queue(
            actor_options={"resources": {"queue_hardware": 1}}
        )
        finetune_config["step_value"] = queue.Queue(
            actor_options={"resources": {"queue_hardware": 1}}
        )
        self.finetune_actor = Progress_Actor.options(resources={"queue_hardware": 1}).remote(
            finetune_config
        )

        callback = LoggingCallback(finetune_config)
        finetune_config["run_config"] = {}
        finetune_config["run_config"]["callbacks"] = [callback]
        self.stopper.stop(False)
        finetune_config["run_config"]["stop"] = self.stopper
        self.finetune_status = False
        # todo: a more reasonable solution is needed
        try:
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
        new_prompt.stop_words.extend(
            ["### Instruction", "# Instruction", "### Question", "##", " ="]
        )
        new_model_desc = ModelDescription(
            model_id_or_path=finetuned_model_path,
            tokenizer_name_or_path=tokenizer_path,
            prompt=new_prompt,
            chat_processor=model_desc.chat_processor if model_desc is not None else "ChatModelGptJ",
        )
        new_model_desc.config.trust_remote_code = True
        new_finetuned = FinetunedConfig(
            name=new_model_name,
            route_prefix="/" + new_model_name,
            model_description=new_model_desc,
        )
        self._all_models[new_model_name] = new_finetuned
        return gr.Dropdown.update(choices=list(self._all_models.keys()))

    def finetune_progress(self, progress=gr.Progress()):
        finetune_flag = False
        while True:
            time.sleep(1)
            if self.finetune_actor is None:
                if finetune_flag is False:
                    continue
                else:
                    break
            if self.finetune_status is True:
                break
            finetune_flag = True
            try:
                total_epochs, total_steps, value_epoch, value_step = ray.get(
                    self.finetune_actor.track_progress.remote()
                )
                if value_epoch == -1:
                    continue
                progress(
                    float(int(value_step) / int(total_steps)),
                    desc="Start Training: epoch "
                    + str(value_epoch)
                    + " / "
                    + str(total_epochs)
                    + "  "
                    + "step "
                    + str(value_step)
                    + " / "
                    + str(total_steps),
                )
            except Exception:
                pass
        self.finetune_status = False
        return "<h4 style='text-align: left; margin-bottom: 1rem'>Completed the fine-tuning process.</h4>"

    def deploy_func(self, model_name: str, replica_num: int, cpus_per_worker_deploy: int):
        self.shutdown_deploy()
        if cpus_per_worker_deploy * replica_num > int(ray.available_resources()["CPU"]):
            raise gr.Error("Resources are not meeting the demand")

        print("Deploying model:" + model_name)

        finetuned = self._all_models[model_name]
        model_desc = finetuned.model_description
        prompt = model_desc.prompt
        print("model path: ", model_desc.model_id_or_path)

        if model_desc.chat_processor is not None:
            chat_model = getattr(sys.modules[__name__], model_desc.chat_processor, None)
            if chat_model is None:
                return (
                    model_name
                    + " deployment failed. "
                    + model_desc.chat_processor
                    + " does not exist."
                )
            self.process_tool = chat_model(**prompt.dict())

        finetuned_deploy = finetuned.copy(deep=True)
        finetuned_deploy.device = "cpu"
        finetuned_deploy.ipex.precision = "bf16"
        finetuned_deploy.cpus_per_worker = cpus_per_worker_deploy
        # transformers 4.35 is needed for neural-chat-7b-v3-1, will be fixed later
        if "neural-chat" in model_name:
            pip_env = "transformers==4.35.0"
        elif "fuyu-8b" in model_name:
            pip_env = "transformers==4.37.2"
        else:
            pip_env = "transformers==4.31.0"
        deployment = PredictorDeployment.options(  # type: ignore
            num_replicas=replica_num,
            ray_actor_options={
                "num_cpus": cpus_per_worker_deploy,
                "runtime_env": {"pip": [pip_env]},
            },
        ).bind(finetuned_deploy)
        serve.run(
            deployment,
            _blocking=True,
            port=finetuned_deploy.port,
            name=finetuned_deploy.name,
            route_prefix=finetuned_deploy.route_prefix,
        )
        return (
            self.ip_port
            if finetuned_deploy.route_prefix is None
            else self.ip_port + finetuned_deploy.route_prefix
        )

    def shutdown_finetune(self):
        self.stopper.stop(True)

    def shutdown_deploy(self):
        serve.shutdown()

    def get_ray_cluster(self):
        command = "source ~/anaconda3/bin/activate; conda activate " + self.conda_env_name + "; ray status"
        stdin, stdout, stderr = self.ssh_connect[-1].exec_command(command)
        out = stdout.read().decode("utf-8")
        #print(f"out is {out}")
        try:
            out_words = [word for word in out.split("\n") if "CPU" in word][0]
        except Exception:
            raise ValueError(f"Can't connect Ray cluster info: stderr is {stderr.read().decode('utf-8')}")
        cpu_info = out_words.split(" ")[1].split("/")
        total_core = int(float(cpu_info[1]))
        used_core = int(float(cpu_info[0]))
        utilization = float(used_core / total_core)
        return ray_status_html.format(str(round(utilization * 100, 1)), used_core, total_core)

    def get_cpu_memory(self, index):
        if self.ray_nodes[index]["Alive"] == "False":
            return cpu_memory_html.format(str(round(0, 1)), str(round(0, 1)))
        cpu_command = "export TERM=xterm; echo $(top -n 1 -b | head -n 4 | tail -n 2)"
        _, cpu_stdout, _ = self.ssh_connect[index].exec_command(cpu_command)
        cpu_out = cpu_stdout.read().decode("utf-8")
        cpu_out_words = cpu_out.split(" ")
        cpu_value = 100 - float(cpu_out_words[7])
        memory_command = "export TERM=xterm; echo $(free -m)"
        _, memory_stdout, _ = self.ssh_connect[index].exec_command(memory_command)
        memory_out = memory_stdout.read().decode("utf-8")
        memory_out_words = memory_out.split("Mem:")[1].split("Swap")[0].split(" ")
        memory_out_words = [m for m in memory_out_words if m != ""]
        total_memory = float(memory_out_words[0].strip())
        free_memory = float(memory_out_words[2].strip())
        buffer_memory = float(memory_out_words[4].strip())
        used_memory = (total_memory - free_memory - buffer_memory) / total_memory
        return cpu_memory_html.format(str(round(cpu_value, 1)), str(round(used_memory * 100, 1)))

    def kill_node(self, btn_txt, index):
        serve.shutdown()
        if btn_txt == "Kill":
            index = int(index)
            command = "conda activate " + self.conda_env_name + "; ray stop"
            self.ssh_connect[index].exec_command(command)
            self.ray_nodes[index]["Alive"] = "False"
            time.sleep(2)
            return "Start", ""
        elif btn_txt == "Start":
            index = int(index)
            command = (
                "conda activate "
                + self.conda_env_name
                + "; RAY_SERVE_ENABLE_EXPERIMENTAL_STREAMING=1 ray start --address="
                + self.master_ip_port
                + r""" --resources='{"special_hardware": 2}'"""
            )
            self.ssh_connect[index].exec_command(command)
            self.ray_nodes[index]["Alive"] = "True"
            time.sleep(2)
            return "Kill", ""

    def watch_node_status(self, index):
        if self.ray_nodes[index]["Alive"] == "False":
            return "<p style='color: rgb(244, 67, 54); background-color: rgba(244, 67, 54, 0.125);'>DEAD</p>"
        else:
            return "<p style='color: rgb(76, 175, 80); background-color: rgba(76, 175, 80, 0.125);'>ALIVE</p>"

    def set_custom_model(self, base_model_name):
        visible = True if base_model_name == "specify other models" else False
        return gr.Textbox.update(visible=visible), gr.Textbox.update(visible=visible)

    def set_upload_box(self, upload_type):
        if upload_type == "Youtube":
            return (
                gr.Textbox.update(
                    visible=True,
                    label="Youtube urls",
                    info="",
                    placeholder="Support multiple urls separated by ';'",
                    value="https://www.youtube.com/watch?v=843OFFzqp3k",
                ),
                gr.File.update(visible=False),
                gr.Slider.update(visible=False),
                gr.Radio.update(visible=False),
            )
        elif upload_type == "Web":
            return (
                gr.Textbox.update(
                    label="Web urls",
                    placeholder="Support multiple urls separated by ';'",
                    visible=True,
                    value="https://www.intc.com/news-events/press-releases/detail/1655/intel-reports-third-quarter-2023-financial-results",
                    info="",
                ),
                gr.File.update(visible=False),
                gr.Slider.update(visible=True),
                gr.Radio.update(visible=False),
            )
        else:
            return (
                gr.Textbox.update(
                    label="Files path",
                    placeholder="Support multiple path separated by ';'",
                    value="",
                    visible=True,
                ),
                gr.File.update(visible=False),
                gr.Slider.update(visible=False),
                gr.Radio.update(visible=True, value="local"),
            )

    def set_input_radio(self, input_type):
        if input_type == "upload":
            return gr.Textbox.update(visible=True, value=""), gr.File.update(visible=False)
        else:
            return gr.Textbox.update(visible=False), gr.File.update(visible=True)

    def set_rag_default_path(self, selector, rag_path):
        if rag_path:
            return rag_path
        if selector is False:
            return None
        else:
            return self.default_rag_path

    def _init_ui(self):
        mark_alive = None
        for index in range(len(self.ray_nodes)):
            if "node:__internal_head__" in ray.nodes()[index]["Resources"]:
                mark_alive = index
                node_ip = self.ray_nodes[index]["NodeName"]
                self.ssh_connect[index] = paramiko.SSHClient()
                self.ssh_connect[index].load_system_host_keys()
                self.ssh_connect[index].set_missing_host_key_policy(paramiko.AutoAddPolicy())
                self.ssh_connect[index].connect(
                    hostname=node_ip, port=self.node_port, username=self.user_name
                )
        if mark_alive is None:
            print("No alive ray worker found! Exit")
            return
        self.ssh_connect[-1] = paramiko.SSHClient()
        self.ssh_connect[-1].load_system_host_keys()
        self.ssh_connect[-1].set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh_connect[-1].connect(
            hostname=self.ray_nodes[mark_alive]["NodeName"],
            port=self.node_port,
            username=self.user_name,
        )

        title = "Manage LLM Lifecycle"
        with gr.Blocks(css=custom_css, title=title) as gr_chat:
            head_content = """
                <div style="color: #fff;text-align: center;">
                    <div style="position:absolute; left:15px; top:15px; "><img  src="/file=ui/images/logo.png" width="50" height="50"/></div>
                    <p style="color: #fff; font-size: 1.1rem;">Manage LLM Lifecycle</p>
                    <p style="color: #fff; font-size: 0.9rem;">Fine-Tune LLMs using workflow on Ray, Deploy and Inference</p>
                </div>
            """
            foot_content = """
                <div class="footer">
                    <p>The workflow is powered by Ray to provide infrastructure management, distributed training, model serving with reliability and auto scaling.</p>
                </div>
            """
            gr.Markdown(head_content, elem_classes="notice_markdown")

            with gr.Tab("Finetune"):
                step1 = "Finetune the model with the base model and data"
                gr.HTML("<h3 style='text-align: left; margin-bottom: 1rem'>" + step1 + "</h3>")
                with gr.Group():
                    base_models_list = list(self._base_models.keys())
                    # set the default value of finetuning to gpt2
                    model_index = (
                        base_models_list.index("gpt2") if "gpt2" in base_models_list else 0
                    )
                    base_models_list.append("specify other models")
                    base_model_dropdown = gr.Dropdown(
                        base_models_list,
                        value=base_models_list[model_index],
                        label="Select Base Model",
                        allow_custom_value=True,
                    )
                    custom_model_name = gr.Textbox(
                        label="Model id",
                        placeholder="The model id of a pretrained model configuration hosted inside a model repo on huggingface.co",
                        visible=False,
                        interactive=True,
                        elem_classes="disable_status",
                    )
                    custom_tokenizer_name = gr.Textbox(
                        label="Tokenizer id",
                        placeholder="The model id of a predefined tokenizer hosted inside a model repo on huggingface.co",
                        visible=False,
                        interactive=True,
                        elem_classes="disable_status",
                    )

                with gr.Accordion("Parameters", open=False, visible=True):
                    batch_size = gr.Slider(
                        0,
                        1000,
                        2,
                        step=1,
                        interactive=True,
                        label="Batch Size",
                        info="train batch size per worker.",
                    )
                    num_epochs = gr.Slider(1, 100, 1, step=1, interactive=True, label="Epochs")
                    max_train_step = gr.Slider(
                        0,
                        1000,
                        10,
                        step=1,
                        interactive=True,
                        label="Step per Epoch",
                        info="value 0 means use the entire dataset.",
                    )
                    lr = gr.Slider(
                        0,
                        0.001,
                        0.00001,
                        step=0.00001,
                        interactive=True,
                        label="Learning Rate",
                    )
                    worker_num = gr.Slider(
                        1,
                        8,
                        1,
                        step=1,
                        interactive=True,
                        label="Worker Number",
                        info="the number of workers used for finetuning.",
                    )
                    cpus_per_worker_ftn = gr.Slider(
                        1,
                        100,
                        24,
                        step=1,
                        interactive=True,
                        label="Cpus per Worker",
                        info="the number of cpu cores used for every worker.",
                    )
                    gr.Slider(
                        0,
                        16,
                        0,
                        step=1,
                        interactive=True,
                        label="Gpus per Worker",
                        info="the number of gpu used for every worker.",
                    )

                with gr.Row():
                    with gr.Column(scale=0.6):
                        data_url = gr.Text(label="Data URL", value=self.default_data_path)
                    with gr.Column(scale=0.2):
                        finetuned_model_name = gr.Text(label="New Model Name", value="my_alpaca")
                    with gr.Column(scale=0.2, min_width=0):
                        finetune_btn = gr.Button("Start to Finetune")
                        stop_finetune_btn = gr.Button("Stop")

                with gr.Row():
                    finetune_res = gr.HTML(
                        "<h4 style='text-align: left; margin-bottom: 1rem'></h4>",
                        show_label=False,
                        elem_classes="disable_status",
                    )

            with gr.Tab("Deployment"):
                step2 = "Deploy the finetuned model as an online inference service"
                gr.HTML("<h3 style='text-align: left; margin-bottom: 1rem'>" + step2 + "</h3>")
                with gr.Row():
                    with gr.Column(scale=0.8):
                        all_models_list = list(self._all_models.keys())
                        # set the default value of deployment to llama-2-7b-chat-hf
                        model_index = (
                            all_models_list.index("llama-2-7b-chat-hf")
                            if "llama-2-7b-chat-hf" in all_models_list
                            else 0
                        )
                        all_model_dropdown = gr.Dropdown(
                            all_models_list,
                            value=all_models_list[model_index],
                            label="Select Model to Deploy",
                            elem_classes="disable_status",
                            allow_custom_value=True,
                        )
                    with gr.Column(scale=0.2, min_width=0):
                        deploy_btn = gr.Button("Deploy")
                        stop_deploy_btn = gr.Button("Stop")

                with gr.Accordion("Parameters", open=False, visible=True):
                    replica_num = gr.Slider(
                        1, 8, 1, step=1, interactive=True, label="Model Replica Number"
                    )
                    cpus_per_worker_deploy = gr.Slider(
                        1,
                        100,
                        24,
                        step=1,
                        interactive=True,
                        label="Cpus per Worker",
                        info="the number of cpu cores used for every worker.",
                    )
                    gr.Slider(
                        0,
                        16,
                        0,
                        step=1,
                        interactive=True,
                        label="Gpus per Worker",
                        info="the number of gpu used for every worker.",
                    )

                with gr.Row():
                    with gr.Column(scale=1):
                        deployed_model_endpoint = gr.Text(
                            label="Deployed Model Endpoint",
                            value="",
                            elem_classes="disable_status",
                        )

            with gr.Tab("Inference"):
                step3 = "Access the online inference service in your own application"
                gr.HTML("<h3 style='text-align: left; margin-bottom: 1rem'>" + step3 + "</h3>")
                with gr.Accordion("Configuration", open=False, visible=True):
                    max_new_tokens = gr.Slider(
                        1,
                        2000,
                        256,
                        step=1,
                        interactive=True,
                        label="Max New Tokens",
                        info="The maximum numbers of tokens to generate.",
                    )
                    Temperature = gr.Slider(
                        0,
                        1,
                        0.2,
                        step=0.01,
                        interactive=True,
                        label="Temperature",
                        info="The value used to modulate the next token probabilities.",
                    )
                    Top_p = gr.Slider(
                        0,
                        1,
                        0.7,
                        step=0.01,
                        interactive=True,
                        label="Top p",
                        info="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to`Top p` or higher are kept for generation.",
                    )
                    Top_k = gr.Slider(
                        0,
                        100,
                        0,
                        step=1,
                        interactive=True,
                        label="Top k",
                        info="The number of highest probability vocabulary tokens to keep for top-k-filtering.",
                    )
                with gr.Tab("Dialogue"):
                    chatbot = gr.Chatbot(
                        elem_id="chatbot",
                        label="chatbot",
                        elem_classes="disable_status",
                    )

                    with gr.Row():
                        with gr.Column(scale=0.8):
                            with gr.Accordion("image", open=False, visible=True):
                                image = gr.Image(type="pil")
                            with gr.Row():
                                endpoint_value = "http://127.0.0.1:8000/v1/chat/completions"
                                model_endpoint = gr.Text(
                                    label="Model Endpoint",
                                    value=endpoint_value,
                                    scale = 1
                                )
                                model_name = gr.Text(
                                    label="Model Name",
                                    value="llama-2-7b-chat-hf",
                                    scale = 1
                                )
                            msg = gr.Textbox(
                                show_label=False,
                                container=False,
                                placeholder="Input your question and press Enter",
                            )
                        with gr.Column(scale=0.2, min_width=20):
                            latency_status = gr.Markdown(
                                """
                                                | <!-- --> | <!-- --> |
                                                |---|---|
                                                | Total Latency [s] | - |
                                                | Tokens | - |""",
                                elem_classes=[
                                    "disable_status",
                                    "output-stats",
                                    "disablegenerating",
                                    "div_height",
                                ],
                            )
                    with gr.Row():
                        with gr.Column(scale=0.5, min_width=0):
                            send_btn = gr.Button("Send")
                        with gr.Column(scale=0.5, min_width=0):
                            clear_btn = gr.Button("Clear")

                with gr.Tab("Multi-Session"):
                    scale_num = 1 / self.test_replica
                    with gr.Row():
                        chatbots = list(range(self.test_replica))
                        msgs = list(range(self.test_replica))
                        for i in range(self.test_replica):
                            with gr.Column(scale=scale_num, min_width=1):
                                chatbots[i] = gr.Chatbot(
                                    elem_id="chatbot" + str(i + 1),
                                    label="chatbot" + str(i + 1),
                                    min_width=1,
                                    elem_classes="disable_status",
                                )
                                msgs[i] = gr.Textbox(
                                    show_label=False,
                                    container=False,
                                    placeholder="Input your question and press Enter",
                                    value=self.messages[i],
                                    min_width=1,
                                )
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

            with gr.Tab("RAG"):
                step3_rag = "Use RAG to enhance generation capabilities"
                gr.HTML("<h3 style='text-align: left; margin-bottom: 1rem'>" + step3_rag + "</h3>")
                with gr.Accordion("Configuration", open=False, visible=True):
                    max_new_tokens_rag = gr.Slider(
                        1,
                        2000,
                        256,
                        step=1,
                        interactive=True,
                        label="Max New Tokens",
                        info="The maximum numbers of tokens to generate.",
                    )
                    Temperature_rag = gr.Slider(
                        0,
                        1,
                        0.7,
                        step=0.01,
                        interactive=True,
                        label="Temperature",
                        info="The value used to modulate the next token probabilities.",
                    )
                    Top_p_rag = gr.Slider(
                        0,
                        1,
                        1.0,
                        step=0.01,
                        interactive=True,
                        label="Top p",
                        info="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to`Top p` or higher are kept for generation.",
                    )
                    Top_k_rag = gr.Slider(
                        0,
                        100,
                        0,
                        step=1,
                        interactive=True,
                        label="Top k",
                        info="The number of highest probability vocabulary tokens to keep for top-k-filtering.",
                    )

                with gr.Accordion("RAG parameters", open=False, visible=True):
                    with gr.Row():
                        recdp_support_suffix = list(_default_file_readers.keys())
                        recdp_support_types = ["Files"]
                        recdp_support_types.extend(["Youtube", "Web"])
                        with gr.Column(scale=1):
                            rag_upload_file_type = gr.Dropdown(
                                recdp_support_types,
                                value=recdp_support_types[0],
                                label="Select File Type to Upload",
                                elem_classes="disable_status",
                                allow_custom_value=True,
                            )
                        input_type = gr.Radio(
                            choices=["local", "upload"],
                            value="local",
                            label="Input Type",
                            scale=1,
                            visible=True,
                        )
                        web_depth = gr.Slider(
                            minimum=1,
                            maximum=10,
                            step=1,
                            interactive=True,
                            label="Max Depth",
                            visible=False,
                            scale=1,
                            info="The max depth of the recursive loading.",
                        )

                        rag_input_text = gr.Textbox(
                            label="Local file path",
                            placeholder="Support types: "
                            + " ".join(recdp_support_suffix)
                            + ". Support multiple absolute paths, separated by ';'",
                            visible=True,
                            scale=2,
                        )

                        data_files = gr.File(
                            label="Upload Files",
                            file_count="multiple",
                            file_types=recdp_support_suffix,
                            elem_classes="file_height",
                            scale=3,
                            visible=False,
                            info="Support types: " + ", ".join(recdp_support_suffix),
                        )

                    with gr.Row():
                        with gr.Column(scale=4, min_width=100):
                            embedding_model = gr.Textbox(
                                label="Embedding Model",
                                value="sentence-transformers/all-mpnet-base-v2",
                                placeholder="Model name to use",
                                info="Model name to use",
                            )
                        with gr.Column(scale=3, min_width=100):
                            splitter_chunk_size = gr.Textbox(
                                label="Text Chunk Size",
                                value="500",
                                placeholder="Maximum size of chunks to return",
                                info="Maximum size of chunks to return",
                                min_width=100,
                            )
                        with gr.Column(scale=3, min_width=100):
                            returned_k = gr.Textbox(
                                label="Fetch result number",
                                value=1,
                                placeholder="Number of retrieved chunks to return",
                                info="Number of retrieved chunks to return",
                                min_width=100,
                            )

                with gr.Row():
                    with gr.Column(scale=0.2, min_width=100):
                        rag_selector = gr.Checkbox(label="RAG", min_width=0)
                    with gr.Column(scale=0.6, min_width=100):
                        rag_path = gr.Textbox(
                            show_label=False,
                            container=False,
                            placeholder="The path of vectorstore",
                            elem_classes="disable_status",
                        )
                    with gr.Column(scale=0.2, min_width=100):
                        regenerate_btn = gr.Button("Regenerate", min_width=0)

                with gr.Tab("Dialogue"):
                    chatbot_rag = gr.Chatbot(
                        elem_id="chatbot",
                        label="chatbot",
                        elem_classes="disable_status",
                    )

                    with gr.Row():
                        with gr.Column(scale=0.8):
                            with gr.Accordion("image", open=False, visible=True):
                                rag_image = gr.Image(type="pil")
                            with gr.Row():
                                endpoint_value = "http://127.0.0.1:8000/v1/chat/completions"
                                rag_model_endpoint = gr.Text(
                                    label="Model Endpoint",
                                    value=endpoint_value,
                                    scale = 1
                                )
                                rag_model_name = gr.Text(
                                    label="Model Name",
                                    value="llama-2-7b-chat-hf",
                                    scale = 1
                                )
                            msg_rag = gr.Textbox(
                                show_label=False,
                                container=False,
                                placeholder="Input your question and press Enter",
                            )
                        with gr.Column(scale=0.2, min_width=0):
                            latency_status_rag = gr.Markdown(
                                """
                                                | <!-- --> | <!-- --> |
                                                |---|---|
                                                | Total Latency [s] | - |
                                                | Tokens | - |""",
                                elem_classes=[
                                    "disable_status",
                                    "output-stats",
                                    "disablegenerating",
                                    "div_height",
                                ],
                            )
                    with gr.Row():
                        with gr.Column(scale=0.5, min_width=0):
                            send_btn_rag = gr.Button("Send")
                        with gr.Column(scale=0.5, min_width=0):
                            clear_btn_rag = gr.Button("Clear")

            with gr.Accordion("Cluster Status", open=False, visible=True):
                with gr.Row():
                    with gr.Column(scale=0.1, min_width=45):
                        with gr.Row():
                            node_pic = r"./ui/images/Picture2.png"
                            gr.Image(
                                type="pil",
                                value=node_pic,
                                show_label=False,
                                min_width=45,
                                height=45,
                                width=45,
                                elem_id="notshowimg",
                                container=False,
                            )
                        with gr.Row():
                            gr.HTML(
                                "<h4 style='text-align: center; margin-bottom: 1rem'> Ray Cluster </h4>"
                            )
                    with gr.Column(scale=0.9):
                        with gr.Row():
                            with gr.Column(scale=0.05, min_width=40):
                                gr.HTML("<h4 style='text-align: right;'> cpu core</h4>")
                            with gr.Column():
                                gr.HTML(
                                    self.get_ray_cluster,
                                    elem_classes="disablegenerating",
                                    every=2,
                                )

                stop_btn = []
                node_status = []
                node_index = []
                for index in range(len(self.ray_nodes)):
                    if self.ray_nodes[index]["Alive"] is False:
                        continue
                    node_ip = self.ray_nodes[index]["NodeName"]
                    with gr.Row():
                        with gr.Column(scale=0.1, min_width=25):
                            with gr.Row():
                                if index == 0:
                                    func = lambda: self.watch_node_status(index=0)
                                elif index == 1:
                                    func = lambda: self.watch_node_status(index=1)
                                elif index == 2:
                                    func = lambda: self.watch_node_status(index=2)
                                elif index == 3:
                                    func = lambda: self.watch_node_status(index=3)

                                node_status.append(
                                    gr.HTML(func, elem_classes="statusstyle", every=2)
                                )
                            with gr.Row():
                                node_index.append(gr.Text(value=len(stop_btn), visible=False))
                                if node_ip == self.head_node_ip:
                                    stop_btn.append(
                                        gr.Button(
                                            "Kill",
                                            interactive=False,
                                            elem_classes="btn-style",
                                        )
                                    )
                                else:
                                    stop_btn.append(gr.Button("Kill", elem_classes="btn-style"))

                        with gr.Column(scale=0.065, min_width=45):
                            with gr.Row():
                                node_pic = r"./ui/images/Picture1.png"
                                gr.Image(
                                    type="pil",
                                    value=node_pic,
                                    show_label=False,
                                    min_width=45,
                                    height=45,
                                    width=45,
                                    elem_id="notshowimg",
                                    container=False,
                                )
                            with gr.Row():
                                if node_ip == self.head_node_ip:
                                    gr.HTML(
                                        "<h4 style='text-align: center; margin-bottom: 1rem'> head node </h4>"
                                    )
                                else:
                                    gr.HTML(
                                        "<h4 style='text-align: center; margin-bottom: 1rem'> work node </h4>"
                                    )
                        with gr.Column(scale=0.835):
                            with gr.Row():
                                with gr.Column(scale=0.05, min_width=40):
                                    gr.HTML("<h4 style='text-align: right;'> cpu </h4>")
                                    gr.HTML("<div style='line-height:70%;'></br></div>")
                                    gr.HTML("<h4 style='text-align: right;'> memory </h4>")
                                with gr.Column():
                                    if index == 0:
                                        func = lambda: self.get_cpu_memory(index=0)
                                    elif index == 1:
                                        func = lambda: self.get_cpu_memory(index=1)
                                    elif index == 2:
                                        func = lambda: self.get_cpu_memory(index=2)
                                    elif index == 3:
                                        func = lambda: self.get_cpu_memory(index=3)

                                    gr.HTML(func, elem_classes="disablegenerating", every=2)

            msg.submit(self.user, [msg, chatbot], [msg, chatbot], queue=False).then(
                self.bot,
                [
                    chatbot,
                    deployed_model_endpoint,
                    model_endpoint,
                    max_new_tokens,
                    Temperature,
                    Top_p,
                    Top_k,
                    model_name,
                    image,
                ],
                [chatbot, latency_status],
            )
            clear_btn.click(self.clear, None, [chatbot, latency_status], queue=False)

            send_btn.click(self.user, [msg, chatbot], [msg, chatbot], queue=False).then(
                self.bot,
                [
                    chatbot,
                    deployed_model_endpoint,
                    model_endpoint,
                    max_new_tokens,
                    Temperature,
                    Top_p,
                    Top_k,
                    model_name,
                    image,
                ],
                [chatbot, latency_status],
            )

            rag_upload_file_type.select(
                self.set_upload_box,
                [rag_upload_file_type],
                [rag_input_text, data_files, web_depth, input_type],
            )
            input_type.select(self.set_input_radio, [input_type], [rag_input_text, data_files])
            regenerate_btn.click(
                self.regenerate,
                [
                    rag_path,
                    rag_upload_file_type,
                    input_type,
                    rag_input_text,
                    web_depth,
                    data_files,
                    embedding_model,
                    splitter_chunk_size,
                    cpus_per_worker_deploy,
                ],
                [rag_path],
            )

            clear_btn_rag.click(self.clear, None, [chatbot_rag, latency_status_rag], queue=False)
            rag_selector.select(self.set_rag_default_path, [rag_selector, rag_path], rag_path)
            msg_rag.submit(
                self.user, [msg_rag, chatbot_rag], [msg_rag, chatbot_rag], queue=False
            ).then(
                self.bot_rag,
                [
                    chatbot_rag,
                    deployed_model_endpoint,
                    rag_model_endpoint,
                    max_new_tokens_rag,
                    Temperature_rag,
                    Top_p_rag,
                    Top_k_rag,
                    rag_selector,
                    rag_path,
                    returned_k,
                    rag_model_name,
                    rag_image,
                ],
                [chatbot_rag, latency_status_rag],
            )
            send_btn_rag.click(
                self.user, [msg_rag, chatbot_rag], [msg_rag, chatbot_rag], queue=False
            ).then(
                self.bot_rag,
                [
                    chatbot_rag,
                    deployed_model_endpoint,
                    rag_model_endpoint,
                    max_new_tokens_rag,
                    Temperature_rag,
                    Top_p_rag,
                    Top_k_rag,
                    rag_selector,
                    rag_path,
                    returned_k,
                    rag_model_name,
                    rag_image,
                ],
                [chatbot_rag, latency_status_rag],
            )

            for i in range(self.test_replica):
                send_all_btn.click(
                    self.user,
                    [msgs[i], chatbots[i]],
                    [msgs[i], chatbots[i]],
                    queue=False,
                ).then(
                    self.send_all_bot,
                    [
                        ids[i],
                        chatbots[i],
                        deployed_model_endpoint,
                        max_new_tokens,
                        Temperature,
                        Top_p,
                        Top_k,
                    ],
                    chatbots[i],
                )
            for i in range(self.test_replica):
                reset_all_btn.click(self.reset, [ids[i]], [msgs[i], chatbots[i]], queue=False)

            for i in range(len(stop_btn)):
                stop_btn[i].click(
                    self.kill_node,
                    [stop_btn[i], node_index[i]],
                    [stop_btn[i], deployed_model_endpoint],
                )

            base_model_dropdown.select(
                self.set_custom_model,
                [base_model_dropdown],
                [custom_model_name, custom_tokenizer_name],
            )
            finetune_event = finetune_btn.click(
                self.finetune,
                [
                    base_model_dropdown,
                    custom_model_name,
                    custom_tokenizer_name,
                    data_url,
                    finetuned_model_name,
                    batch_size,
                    num_epochs,
                    max_train_step,
                    lr,
                    worker_num,
                    cpus_per_worker_ftn,
                ],
                [all_model_dropdown],
            )
            finetune_progress_event = finetune_btn.click(
                self.finetune_progress, None, [finetune_res]
            )
            stop_finetune_btn.click(
                fn=self.shutdown_finetune,
                inputs=None,
                outputs=None,
                cancels=[finetune_event, finetune_progress_event],
            )
            deploy_event = deploy_btn.click(
                self.deploy_func,
                [all_model_dropdown, replica_num, cpus_per_worker_deploy],
                [deployed_model_endpoint],
            )
            stop_deploy_btn.click(
                fn=self.shutdown_deploy,
                inputs=None,
                outputs=None,
                cancels=[deploy_event],
            )

            gr.Markdown(foot_content)

        self.gr_chat = gr_chat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web UI for LLM on Ray", add_help=True)
    parser.add_argument(
        "--finetune_model_path",
        default="./",
        type=str,
        help="Where to save the finetune model.",
    )
    parser.add_argument(
        "--finetune_checkpoint_path",
        default="",
        type=str,
        help="Where to save checkpoints.",
    )
    parser.add_argument(
        "--default_rag_path",
        default="./vector_store/",
        type=str,
        help="The path of vectorstore used by RAG.",
    )
    parser.add_argument(
        "--node_port", default="22", type=str, help="The node port that ssh connects."
    )
    parser.add_argument(
        "--node_user_name",
        default="root",
        type=str,
        help="The node user name that ssh connects.",
    )
    parser.add_argument(
        "--conda_env_name",
        default="base",
        type=str,
        help="The environment used to execute ssh commands.",
    )
    parser.add_argument(
        "--master_ip_port",
        default="None",
        type=str,
        help="The ip:port of head node to connect when restart a worker node.",
    )
    args = parser.parse_args()

    file_path = os.path.abspath(__file__)
    infer_path = os.path.dirname(file_path)
    repo_path = os.path.abspath(infer_path + os.path.sep + "../")
    default_data_path = os.path.abspath(
        infer_path + os.path.sep + "../examples/data/sample_finetune_data.jsonl"
    )

    sys.path.append(repo_path)
    from finetune.finetune import get_accelerate_environment_variable

    finetune_config: Dict[str, Any] = {
        "General": {"config": {}},
        "Dataset": {"validation_file": None, "validation_split_percentage": 0},
        "Training": {
            "optimizer": "AdamW",
            "lr_scheduler": "linear",
            "weight_decay": 0.0,
            "device": "CPU",
            "num_training_workers": 2,
            "resources_per_worker": {"CPU": 24},
            "accelerate_mode": "CPU_DDP",
        },
        "failure_config": {"max_failures": 5},
    }

    ray_init_config: Dict[str, Any] = {
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
    }
    accelerate_env_vars = get_accelerate_environment_variable(
        finetune_config["Training"]["accelerate_mode"], config=None
    )
    ray_init_config["runtime_env"]["env_vars"].update(accelerate_env_vars)
    print("Start to init Ray connection")
    context = ray.init(**ray_init_config)
    print("Ray connected")
    head_node_ip = context.get("address").split(":")[0]

    finetune_model_path = args.finetune_model_path
    finetune_checkpoint_path = args.finetune_checkpoint_path
    default_rag_path = args.default_rag_path

    initial_model_list = {k: all_models[k] for k in sorted(all_models.keys())}
    ui = ChatBotUI(
        initial_model_list,
        initial_model_list,
        finetune_model_path,
        finetune_checkpoint_path,
        repo_path,
        default_data_path,
        default_rag_path,
        finetune_config,
        head_node_ip,
        args.node_port,
        args.node_user_name,
        args.conda_env_name,
        args.master_ip_port,
    )
    ui.gr_chat.queue(concurrency_count=10).launch(
        share=True, server_port=8080, server_name="0.0.0.0"
    )
