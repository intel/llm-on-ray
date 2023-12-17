import os
import asyncio
import functools
import ray
from ray import serve
from starlette.requests import Request
from queue import Empty
import torch
from transformers import AutoTokenizer, TextIteratorStreamer
from transformers import StoppingCriteria, StoppingCriteriaList
from inference_config import InferenceConfig

from typing import Generator, Union, List
from starlette.responses import StreamingResponse
from api_openai_backend.common.openai_protocol import ModelResponse


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

def max_input_len(input_text_length):
    if input_text_length <= 128:
        return 128
    elif input_text_length <= 512:
        return 512
    elif input_text_length <= 2048:
        return 2048
    else:
        print("Max support length is 4096")
        return 4096

@serve.deployment
class PredictDeployment:
    def __init__(self, inferenceConfig: InferenceConfig):
        self.device = torch.device(inferenceConfig.device)
        self.tokenizer = AutoTokenizer.from_pretrained(inferenceConfig.model_description.tokenizer_name_or_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.process_tool = None
        chat_processor_name = inferenceConfig.model_description.chat_processor
        prompt = inferenceConfig.model_description.prompt
        if chat_processor_name:
            module = __import__("chat_process")
            chat_processor = getattr(module, chat_processor_name, None)
            if chat_processor is None:
                raise ValueError(inferenceConfig.name + " deployment failed. chat_processor(" + chat_processor_name + ") does not exist.")
            self.process_tool = chat_processor(**prompt.dict())
        stop_words = prompt.stop_words
        stop_words_ids = [self.tokenizer(stop_word, return_tensors='pt').input_ids.squeeze() for stop_word in stop_words]
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        self.use_deepspeed = inferenceConfig.deepspeed
        self.amp_dtype = torch.bfloat16 if inferenceConfig.precision != "fp32" else torch.float32
        if self.use_deepspeed:
            from deepspeed_predictor import DeepSpeedPredictor
            self.streamer = self.create_streamer()
            # now deepspeed predictor don't have the model
            # this should be solved in the next pr
            # where it is also a worker
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.predictor = DeepSpeedPredictor(inferenceConfig, self.amp_dtype, self.tokenizer.pad_token_id, self.stopping_criteria)
        else:
            from transformer_predictor import TransformerPredictor
            self.predictor = TransformerPredictor(inferenceConfig, self.amp_dtype, self.stopping_criteria)
            self.predictor.configure_tokenizer(inferenceConfig.model_description.model_id_or_path, self.tokenizer)
        self.loop = asyncio.get_running_loop()

    def create_streamer(self):
        from transformers import TextStreamer
        from typing import Optional
        from ray.util.queue import Queue

        class RayTextIteratorStreamer(TextStreamer):
            def __init__(
                self, tokenizer: "AutoTokenizer", skip_prompt: bool = False, timeout: Optional[float] = None, **decode_kwargs
            ):
                super().__init__(tokenizer, skip_prompt, **decode_kwargs)
                self.text_queue = Queue()
                self.stop_signal = None
                self.timeout = timeout

            def on_finalized_text(self, text: str, stream_end: bool = False):
                self.text_queue.put(text, timeout=self.timeout)
                if stream_end:
                    self.text_queue.put(self.stop_signal, timeout=self.timeout)

            def __iter__(self):
                return self

            def __next__(self):
                value = self.text_queue.get(timeout=self.timeout)
                if value == self.stop_signal:
                    raise StopIteration()
                else:
                    return value
        return RayTextIteratorStreamer(self.tokenizer, skip_special_tokens=True)

    def tokenize_inputs(self, text: List[str]):
        if self.device.type == "hpu":
            input_tokens_no_pad = self.tokenizer(text, return_tensors="pt")
            input_token_len = input_tokens_no_pad.input_ids.shape[-1]
            input_tokens = self.tokenizer.batch_encode_plus(
                text,
                return_tensors="pt",
                padding="max_length",
                max_length=max_input_len(input_token_len),
            )
        else:
            input_tokens = self.tokenizer.batch_encode_plus(
                text, return_tensors="pt", padding=True
            )
            input_token_len = input_tokens.input_ids.shape[-1]
        inputs = {k: v.to(device=self.device) \
                  for k,v in input_tokens.items() \
                  if torch.is_tensor(v)}
        return inputs, input_token_len

    def predict(self, text: List[str], **config) -> str:
        inputs, _ = self.tokenize_inputs(text)
        gen_tokens = self.predictor.generate(inputs, **config)
        return self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]

    def predict_stream(self, text: List[str], streamer: TextIteratorStreamer, **config) -> Generator[str, None, None]:
        # with torch.cpu.amp.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
        inputs, _ = self.tokenize_inputs(text)
        self.predictor.streaming_generate(inputs, streamer, **config)
    
    def consume_streamer(self):
        for text in self.streamer:
            yield text

    async def consume_streamer_async(self, streamer: TextIteratorStreamer):
        while True:
            try:
                for token in streamer:
                    yield token
                break
            except Empty:
                # The streamer raises an Empty exception if the next token
                # hasn't been generated yet. `await` here to yield control
                # back to the event loop so other coroutines can run.
                await asyncio.sleep(0.001)

    async def __call__(self, http_request: Request) -> Union[StreamingResponse, str]:
        json_request: str = await http_request.json()
        prompts = []
        for prompt in json_request:
            text = prompt["text"]
            config = prompt["config"]  if "config" in prompt else {}
            streaming_response = prompt["stream"]
            if isinstance(text, list):
                if self.process_tool is not None:
                    prompt = self.process_tool.get_prompt(text)
                    prompts.append(prompt)
                else:
                    prompts.extend(text)
            else:
                prompts.append(text)
        if not streaming_response:
            return self.predict(prompts, **config)
        if self.use_deepspeed:
            self.predict_stream(prompts, self.streamer, **config)
            return StreamingResponse(self.consume_streamer(), status_code=200, media_type="text/plain")
        else:
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=0, skip_special_tokens=True)
            self.loop.run_in_executor(None, functools.partial(self.predict_stream, prompts, streamer, **config))
            return StreamingResponse(self.consume_streamer_async(streamer), status_code=200, media_type="text/plain")
        
    async def stream_response(self, content: str, **config):
        prompt = content.prompt
        prompts = []
        if isinstance(prompt, list):
            for message in prompt:
                prompts.append(message.content)
            prompts = [" ".join(prompts)]
        else:
            prompts.append(prompt)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=0, skip_special_tokens=True)
        self.loop.run_in_executor(None, functools.partial(self.predict_stream, prompts, streamer, **config))
        response_handle = self.consume_streamer_async(streamer)
        async for output in response_handle:
            model_response = ModelResponse(
                generated_text=output,
                num_input_tokens=len(prompts[0]),
                num_input_tokens_batch=len(prompts[0]),
                num_generated_tokens=1,
                preprocessing_time=0,
            )
            yield model_response

def serve_run(model_list, deployments):
    for model_id, inferCfg in model_list.items():
        print("deploy model: ", model_id)
        deployment = deployments[model_id]
        handle = serve.run(deployment, _blocking=True, host=inferCfg.host, port=inferCfg.port, name=inferCfg.name, route_prefix=inferCfg.route_prefix)
        deployment_name = inferCfg.name
        if inferCfg.host == "0.0.0.0":
            all_nodes = ray.nodes()
            for node in all_nodes:
                if "node:__internal_head__" in node["Resources"]:
                    host_ip = node["NodeManagerAddress"]
                    break
        else:
            host_ip = inferCfg.host
        url = f"http://{host_ip}:{inferCfg.port}{inferCfg.route_prefix}"
        print(f"Deployment '{deployment_name}' is ready at `{url}`.")

    msg = "Service is deployed successfully"
    env_name = "KEEP_SERVE_TERMINAL"
    if env_name not in os.environ or os.environ[env_name].lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup']:
        input(msg)
    else:
        print(msg)
    return deployments
