import requests
from config import all_models
import time

from chat_process import ChatModelGptJ
class ChatBotUI():

    def __init__(self, models: list):
        self._models = models
        self.ip_port = "http://127.0.0.1:8000"
        self.process_tool = ChatModelGptJ("### Instruction", "### Response", stop_words=["### Instruction", "# Instruction", "### Question", "##", " ="])
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
        print("response: ", output)
        return output

    def bot(self, history, model, Max_new_tokens, Temperature, Top_p, Top_k, Beams, do_sample):
        prompt = self.history_to_messages(history)
        request_url = self.ip_port+all_models[model]["route_prefix"]
        time_start = time.time()
        config = {
            "max_new_tokens": Max_new_tokens,
            "Temperature": Temperature,
            "Top_p": Top_p,
            "Top_k": Top_k,
            "Beams": Beams,
            "do_sample": do_sample
        }
        response = self.model_generate(prompt=prompt, request_url=request_url, config=config)
        time_end = time.time()
        history[-1][1]=response
        time_spend = time_end - time_start
        return [history, time_spend, model] 
    

    def _set_model(self, selected_model):
        self._model = selected_model

    def _init_ui(self):
        import gradio as gr
        with gr.Blocks(css="OpenChatKit .overflow-y-auto{height:500px}") as gr_chat:
            chatbot = gr.Chatbot(elem_id="chatbot", label="ChatLLM")
            with gr.Row():
                model_dropdown = gr.Dropdown(self._models, value=self._models[0],
                                             label="Select Model", default=self._models[0])
                model_dropdown.change(self._set_model, [model_dropdown])

            with gr.Accordion("Configuration", open=False, visible=True):
                max_new_tokens = gr.Slider(1, 2000, 40, step=1, interactive=True, label="Max New Tokens")
                Temperature = gr.Slider(0, 1, 1.0, step=0.01, interactive=True, label="Temperature")
                Top_p = gr.Slider(0, 1, 1.0, step=0.01, interactive=True, label="Top p")
                Top_k = gr.Slider(0, 100, 0, step=1, interactive=True, label="Top k")
                Beams = gr.Slider(1, 10, 1, step=1, interactive=True, label="Beams Number")
                Do_smaple = gr.Checkbox(value=False, interactive=True, label="do sample")

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
                    latency = gr.Text(label="Inference latency (s)", value="0")
                with gr.Column(scale=0.1):
                    model_used = gr.Text(label="Inference Model", value="")

            msg.submit(self.user, [msg, chatbot], [msg, chatbot], queue=False).then(
                self.bot, [chatbot, model_dropdown, max_new_tokens, Temperature, Top_p, Top_k, Beams, Do_smaple], 
                           [chatbot, latency, model_used]
            )
            clear_btn.click(self.clear, None, chatbot, queue=False)
            send_btn.click(self.user, [msg, chatbot], [msg, chatbot], queue=False).then(
                self.bot, [chatbot, model_dropdown, max_new_tokens, Temperature, Top_p, Top_k, Beams, Do_smaple], 
                           [chatbot, latency, model_used]
            )

        self.gr_chat = gr_chat

if __name__ == "__main__":

    models = list(all_models.keys())
    ui = ChatBotUI(models)
    ui.gr_chat.launch(share=True, server_port=8081, server_name="0.0.0.0")