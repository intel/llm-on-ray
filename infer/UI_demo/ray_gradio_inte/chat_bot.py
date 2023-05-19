import ray
from ray import serve
from ray.serve.gradio_integrations import GradioIngress

import gradio as gr

import asyncio
from transformers import pipeline
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import intel_extension_for_pytorch as ipex
from transformers import StoppingCriteria, StoppingCriteriaList
from chat_process import ChatModelGptJ


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            length = 1  if len(stop.size())==0 else stop.size()[0]
            if torch.all((stop == input_ids[0][-length:])).item():
                return True

        return False

@serve.deployment()
class PredictDeployment:
    def __init__(self, model_id, amp_enabled, amp_dtype, stop_words):
        self.amp_enabled = amp_enabled
        self.amp_dtype = amp_dtype
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=amp_dtype,
            low_cpu_mem_usage=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = model.eval()
        # to channels last
        model = model.to(memory_format=torch.channels_last)
        # to ipex
        self.model = ipex.optimize(model, dtype=amp_dtype, inplace=True)
        # self.model = model

        stop_words_ids = [self.tokenizer(stop_word, return_tensors='pt').input_ids.squeeze() for stop_word in stop_words]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def generate(self, text: str, config) -> str:
        # with torch.cpu.amp.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        max_new_tokens = config["max_new_tokens"]
        Temperature = config["Temperature"]
        Top_p = config["Top_p"]
        Top_k = config["Top_k"]
        Beams = config["Beams"]
        do_sample = config["do_sample"]

        gen_tokens = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=Temperature,
            top_p=Top_p,
            top_k=Top_k,
            num_beams=Beams,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=self.stopping_criteria,
        )
        output = self.tokenizer.batch_decode(gen_tokens)[0]

        return output

    async def __call__(self, text, config=None) -> str:
        prompts = []
        if isinstance(text, list):
            prompts.extend(text)
        else:
            prompts.append(text)
        outputs = self.generate(prompts, config)
        return outputs



@serve.deployment
class MyGradioServer(GradioIngress):
    def __init__(self, app_model_all):
        self._models = app_model_all
        self._models_name = list(app_model_all.keys())
        self.process_tool = ChatModelGptJ("### Instruction", "### Response", stop_words=["### Instruction", "# Instruction", "### Question", "##", " ="])
        print("self._models: ", self._models)
        self.init_ui()
        super().__init__(lambda: self.gr_chat)

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

    def init_ui(self):
        import gradio as gr
        # title = "LLM on Ray Fine-tune and Inference workflow"
        title = "Inference Demo of GPT-J-6B based on LLM-Ray Fine-tuning"
        description = """
        The demo showcase GPT-J-6B fine-tuned model on using LLM on Ray workflow. gpt-j-6B is the original model and gpt-j-6B-finetuned-52K is the fine-tuned model. Ray Serve was used for the inference. 
        """
        with gr.Blocks(css="OpenChatKit .overflow-y-auto{height:500px}", title=title) as gr_chat:
            gr.HTML("<h1 style='text-align: center; margin-bottom: 1rem'>"+ title+ "</h1>")
            # gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>"+ title+ "</h1>")
            gr.Markdown(description)
            with gr.Row():
                with gr.Column(scale=0.6):
                    self.msg = gr.Textbox(show_label=False,
                                     placeholder="Input your question").style(container=False)
                with gr.Column(scale=0.2, min_width=0):
                    send_btn = gr.Button("Send")
                with gr.Column(scale=0.2, min_width=0):
                    clear_btn = gr.Button("Clear")
            self.chatbot = gr.Chatbot(elem_id="chatbot", label="ChatLLM")
            with gr.Row():
                self.model_dropdown = gr.Dropdown(self._models_name, value=self._models_name[0],
                                             label="Select Model", default=self._models_name[0])

            with gr.Accordion("Configuration", open=False, visible=True):
                # https://github.com/huggingface/transformers/blob/167aa76cfae7eac01d6cae98aeba4d1e2810a1ce/src/transformers/generation/configuration_utils.py#L69
                self.max_new_tokens = gr.Slider(1, 2000, 128, step=1, interactive=True, label="Max New Tokens", info="The maximum numbers of tokens to generate.")
                self.Temperature = gr.Slider(0, 1, 0.7, step=0.01, interactive=True, label="Temperature", info="The value used to modulate the next token probabilities.")
                self.Top_p = gr.Slider(0, 1, 0.95, step=0.01, interactive=True, label="Top p", info="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to`top_p` or higher are kept for generation.")
                self.Top_k = gr.Slider(0, 100, 50, step=1, interactive=True, label="Top k", info="The number of highest probability vocabulary tokens to keep for top-k-filtering.")
                self.Beams = gr.Slider(1, 10, 1, step=1, interactive=True, label="Beams Number", info="Number of beams for beam search. 1 means no beam search.")
                self.Do_smaple = gr.Checkbox(value=True, interactive=True, label="do sample", info="Whether or not to use sampling.")

            with gr.Row():
                with gr.Column(scale=0.1):
                    self.latency = gr.Text(label="Inference latency (s)", value="0")
                with gr.Column(scale=0.1):
                    self.model_used = gr.Text(label="Inference Model", value="")
            self.msg.submit(self.user, [self.msg, self.chatbot], [self.msg, self.chatbot], queue=False).then(
                self.fanout, [self.chatbot, self.model_dropdown, self.max_new_tokens, self.Temperature, self.Top_p, self.Top_k, self.Beams, self.Do_smaple], 
                           [self.chatbot, self.latency, self.model_used]
            )
            clear_btn.click(self.clear, None, self.chatbot, queue=False)
            send_btn.click(self.user, [self.msg, self.chatbot], [self.msg, self.chatbot], queue=False).then(
                self.fanout, [self.chatbot, self.model_dropdown, self.max_new_tokens, self.Temperature, self.Top_p, self.Top_k, self.Beams, self.Do_smaple], 
                           [self.chatbot, self.latency, self.model_used]
            )
            
        self.gr_chat = gr_chat

    async def fanout(self, chatbot, model_name, max_new_tokens, Temperature, Top_p, Top_k, Beams, Do_smaple):

        print("request model: ", model_name)
        prompt = self.history_to_messages(chatbot)
        config = {
                    "max_new_tokens": max_new_tokens,
                    "Temperature": Temperature,
                    "Top_p": Top_p,
                    "Top_k": Top_k,
                    "Beams": Beams,
                    "do_sample": Do_smaple
                }

        time_start = time.time()
        # process text
        prompt = self.process_tool.get_prompt(prompt)
        downstream_model = self._models[model_name]
        refs = await asyncio.gather(downstream_model.remote(prompt, config))
        result = ray.get(refs)
        print("result: ", result)
        # remove context
        result = result[0][len(prompt):]
        result = self.process_tool.convert_output(result)
        chatbot[-1][1]=result
        time_end = time.time()
        time_spend = time_end-time_start
        return [chatbot, time_spend, model_name]

runtime_env = {
    "env_vars": {
        "KMP_BLOCKTIME": "1",
        "KMP_SETTINGS": "1",
        "KMP_AFFINITY": "granularity=fine,compact,1,0",
        "OMP_NUM_THREADS": "56",
        "MALLOC_CONF": "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000",
    }
}
# ray.init(address="auto", runtime_env=runtime_env)

app_all = {}
app_all["gpt-j-6B-finetuned-52K"] = "/mnt/DP_disk3/ykp/huggingface/gpt-j-6B-finetuned-52K"
app_all["gpt-j-6B-finetuned-2K"] = "/mnt/DP_disk3/ykp/huggingface/gpt-j-6B-finetuned-2K"
app_all["gpt-j-6B"] = "/mnt/DP_disk3/ykp/huggingface/gpt-j-6B"


for model_name, model_id in app_all.items():
    app_all[model_name] = PredictDeployment.options(ray_actor_options={"runtime_env": runtime_env}).bind(model_id, True, torch.bfloat16, stop_words=["### Instruction", "# Instruction", "### Question", "##", " ="])
app = MyGradioServer.bind(app_all)