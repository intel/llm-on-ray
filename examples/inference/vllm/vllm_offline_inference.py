
from vllm import LLM, SamplingParams

from vllm.extension import ns as ns

from time import perf_counter


############################
# TODO: TODO: TODO: reset request_id and seq_id before they reach to max value of int64_t
############################

# Sample prompts.
# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]
prompts = [
    "Russell Brunson's Perfect Webinar Script is a framework for delivering webinars that are designed to sell a product or service. ",
    "Tony Robbins describes six core human needs that drive our behaviors and motivations. These six needs are:\n\n1. Certainty: The need for safety, stability, and predictability. This includes the need for comfort, security, and control over our environment.\n2.",
    "1. Homogeneity: The segment should consist of customers who share similar characteristics and behaviors.\n2. Distinctiveness: The segment should be different from other segments in terms of their characteristics and behaviors.\n3. Stability: The segment should remain relatively stable over time and not change drastically. The characteristics and behaviors of customers within the segment should not change significantly.",
    "In Java, I want to replace string like \"This is a new {object} at {place}\" with a Map, {object: \"student\", \"point 3, 4\"}, and get a result \"This is a new student at point 3, 4\". How can I do?",
    "You can use the `String.format()` method in Java to replace placeholders in a string with values from a map. Here's an example code snippet that demonstrates how you can achieve this:\n```java\nimport java.util.HashMap;\nimport java.util.Map;\n\npublic class StringReplaceExample {\n    public static void main(String[] args) {\n        String input = \"This is a new {object} at {place}\";\n        Map<String, String> replacements = new HashMap<>();\n        replacements.put(\"object\", \"student\");\n        replacements.put(\"place\", \"point 3, 4\");\n\n ",
    "The language used to describe the addressing modes of these instructions is metaphorical and grandiose, emphasizing the complexity and power of these commands. For example, the use of \"enigmatic\" and \"confounding\" to describe JMP ABCD and MOV AX, [BX+SI], respectively, suggests that these instructions are not easily understood and require a level of expertise to comprehend.\n\nSimilarly, the use of \"inscrutable\" and \"cryptic\" to describe MOV AX, [100] and MOV AX, [BX], respectively, implies that these commands are shrouded in mystery and are difficult to decipher. The speaker's use of \"perplexing\" and \"unfathomable\" to describe MOV AX, [BX\\*2+SI] and MOV AX, BX, respectively, ",
    "Lo and behold! The arcane and elusive art of metaphorical language has been summoned forth to expound upon the enigmatic addressing modes of the instructions at hand. The speakers have wielded grandiose expressions with utmost reverence and awe, extolling the ineffable power and bewildering functionality of these directives. Among the inscrutable commands are the confounding JMP ABCD, the abstruse MOV AX, [BX+SI], the unfathomable MOV AX, [100],",
    "more more perplexity and verbose",
    "By the grace of the gods,"
]
# Create a sampling params object.
# sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=256, use_beam_search=True, best_of=2)
# sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=256, top_k=40)
sampling_params = SamplingParams(max_tokens=32)
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)

#######################################
prompts = ["Hello, how are you?", "What is your name?", "What is your favorite color?", "What is your favorite food?", "What is your favorite movie?", "What is your favorite song?", "What is your favorite book?", "What is your favorite animal", "What is your favorite sport?", "What is your favorite hobby?", "What is your favorite subject?", "What is your favorite game?", "What is your favorite TV show?", "What is your favorite actor?", "What is your favorite actress?", "What is your favorite singer?", "What is your favorite band?", "What is your favorite artist?", "What is your favorite author?", "What is your favorite poet?"]

# contents = []
new_prompts = []

for i in range(4):
    # content = {"prompt": prompts[i%len(prompts)], "stream": False, "max_tokens": 4096, "best_of": 2, "use_beam_search": True, "temperature": 0}
    # content = {"prompt": prompts[i%len(prompts)], "stream": False, "max_tokens": 4096}
    # contents.append(content)
    new_prompts.append(prompts[i%len(prompts)])
######################################
# new_prompts.clear()
# new_prompts.append(prompts[2])


# Create an LLM.
# llm = LLM(model="facebook/opt-125m", device="cpu", quantization="awq")
# llm = LLM(model="TheBloke/Llama-2-7b-Chat-AWQ", device="cpu", quantization="AWQ")
# llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", device="cpu", quantization="ns")
# TODO verify block_size >= max_model_len
# TODO set VLLM_CPU_KVCACHE_SPACE to X (GB) so that VLLM_CPU_KVCACHE_SPACE/(block_size*element_size) = num_cpu_blocks <= max_num_seqs. Otherwise, native kv cache may run out of slots.
ctx_size = 512
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", device="cpu", max_num_seqs=64, block_size=ctx_size, max_model_len=ctx_size, quantization="ns")
# llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", device="cpu")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
for i in range(4):
    t0 = perf_counter()
    outputs = llm.generate(new_prompts, sampling_params)
    total_time = (perf_counter() - t0)
    # Print the outputs.
    total_prompts = 0
    total_generated = 0
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        total_prompts += len(prompt.split(' '))
        total_generated += len(generated_text.split(' '))
        # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    print(total_time, total_prompts, total_generated)
    print(f"prompts per second: {total_prompts/total_time}")
    print(f"tokens per second: {total_generated/total_time}")
