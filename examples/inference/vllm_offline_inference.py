from vllm import LLM, SamplingParams

from vllm.extension import ns as ns

############################
# TODO: TODO: TODO: reset request_id and seq_id before they reach to max value of int64_t
############################

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
# sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=256, use_beam_search=True, best_of=2)
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=256, top_k=40)
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)

# Create an LLM.
# llm = LLM(model="facebook/opt-125m", device="cpu", quantization="awq")
# llm = LLM(model="TheBloke/Llama-2-7b-Chat-AWQ", device="cpu", quantization="AWQ")
# llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", device="cpu", quantization="ns")
# TODO verify block_size >= max_model_len
# TODO set VLLM_CPU_KVCACHE_SPACE to X (GB) so that VLLM_CPU_KVCACHE_SPACE/(block_size*element_size) = num_cpu_blocks <= max_num_seqs. Otherwise, native kv cache may run out of slots.
ctx_size = 512
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", device="cpu", max_num_seqs=20, block_size=ctx_size, max_model_len=ctx_size, quantization="ns")
# llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", device="cpu")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
