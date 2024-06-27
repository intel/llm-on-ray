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
    "Tony Robbins describes six core human needs that drive our behaviors and motivations. These six needs are:\n\n1. Certainty: The need for safety, stability, and predictability.",
    "1. Homogeneity: The segment should consist of customers who share similar characteristics and behaviors.\n2. Distinctiveness: The segment should be different from other segments in terms of their characteristics and behaviors.\n3. Stability: The segment should remain relatively stable over time and not change drastically. The characteristics and behaviors of customers within the segment should not change significantly.",
    'In Java, I want to replace string like "This is a new {object} at {place}" with a Map, {object: "student", "point 3, 4"}, and get a result "This is a new student at point 3, 4". How can I do?',
    'You can use the `String.format()` method in Java to replace placeholders in a string with values from a map. Here\'s an example code snippet that demonstrates how you can achieve this:\n```java\nimport java.util.HashMap;\nimport java.util.Map;\n\npublic class StringReplaceExample {\n    public static void main(String[] args) {\n        String input = "This is a new {object} at {place}";\n        Map<String, String> replacements = new HashMap<>();\n        replacements.put("object", "student");\n        replacements.put("place", "point 3, 4");\n\n ',
    'The language used to describe the addressing modes of these instructions is metaphorical and grandiose, emphasizing the complexity and power of these commands. For example, the use of "enigmatic" and "confounding" to describe JMP ABCD and MOV AX, [BX+SI], respectively, suggests that these instructions are not easily understood and require a level of expertise to comprehend.\n\nSimilarly, the use of "inscrutable" and "cryptic" to describe MOV AX, [100] and MOV AX, [BX], respectively, implies that these commands are shrouded in mystery and are difficult to decipher. The speaker\'s use of "perplexing" and "unfathomable" to describe MOV AX, [BX\\*2+SI] and MOV AX, BX, respectively, ',
    "Lo and behold! The arcane and elusive art of metaphorical language has been summoned forth to expound upon the enigmatic addressing modes of the instructions at hand. The speakers have wielded grandiose expressions with utmost reverence and awe, extolling the ineffable power and bewildering functionality of these directives. Among the inscrutable commands are the confounding JMP ABCD, the abstruse MOV AX, [BX+SI], the unfathomable MOV AX, [100],",
    "more more perplexity and verbose",
    "By the grace of the gods,",
]
# Create a sampling params object.
# sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=256, use_beam_search=True, best_of=2)
# sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=256, top_k=40)
sampling_params = SamplingParams(max_tokens=128)
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)

#######################################
prompts = [
    "Hello, how are you?",
    "What is your name?",
    "What is your favorite color?",
    "What is your favorite food?",
    "What is your favorite movie?",
    "What is your favorite song?",
    "What is your favorite book?",
    "What is your favorite animal",
    "What is your favorite sport?",
    "What is your favorite hobby?",
    "What is your favorite subject?",
    "What is your favorite game?",
    "What is your favorite TV show?",
    "What is your favorite actor?",
    "What is your favorite actress?",
    "What is your favorite singer?",
    "What is your favorite band?",
    "What is your favorite artist?",
    "What is your favorite author?",
    "What is your favorite poet?",
]
# 32 input
prompts = [
    "Tony Robbins describes six core human needs that drive our behaviors and motivations. These six needs are:\n\n1. Certainty: The need for safety, stability, and predictability."
]
# 1024 input
prompts = [
    "It is done, and submitted. You can play 'Survival of the Tastiest' on Android, and on the web. Playing on the web works, but you have to simulate multiple touch for table moving and that can be a bit confusing. There is a lot I'd like to talk about. I will go through every topic, insted of making the typical what went right/wrong list. Concept Working over the theme was probably one of the hardest tasks which I had to face. Originally, I had an idea of what kind of game I wanted to develop, gameplay wise - something with a lot of enemies/actors, simple graphics, maybe set in space, controlled from a top-down view. I was confident that I could fit any theme around it. In the end, the problem with a theme like 'Evolution' in a game is that evolution is unassisted. It happens through several seemingly random mutations over time, with the most apt permutation surviving. This genetic car simulator is, in my opinion, a great example of actual evolution of a species facing a challenge. But is it a game? In a game, you need to control something to reach an objective. That control goes against what evolution is supposed to be like. If you allow the user to pick how to evolve something, it's not evolution anymore - it's the equivalent of intelligent design, the fable invented by creationists to combat the idea of evolution. Being agnostic and a Pastafarian, that's not something that rubbed me the right way. Hence, my biggest dillema when deciding what to create was not with what I wanted to create, but with what I did not. I didn't want to create an 'intelligent design' simulator and wrongly call it evolution. This is a problem, of course, every other contestant also had to face. And judging by the entries submitted, not many managed to work around it. I'd say the only real solution was through the use of artificial selection, somehow. So far, I have not seen any entry using this at its core gameplay. Alas, this is just a fun competition and after a while I decided not to be as strict with the game idea, and allowed myself to pick whatever I thought would work out. My initial idea was to create something where humanity tried to evolve to a next level but had some kind of foe trying to stop them from doing so. I kind of had this image of human souls flying in space towards a monolith or a space baby (all based in 2001: A Space Odyssey of course) but I couldn't think of compelling (read: serious) mechanics for that. Borgs were my next inspiration, as their whole hypothesis fit pretty well into the evolution theme. But how to make it work? Are you the borg, or fighting the Borg? The third and final idea came to me through my girlfriend, who somehow gave me the idea of making something about the evolution of Pasta. The more I thought about it the more it sounded like it would work, so I decided to go with it. Conversations with my inspiring co-worker Roushey (who also created the 'Mechanical Underdogs' signature logo for my intros) further matured the concept, as it involved into the idea of having individual pieces of pasta flying around and trying to evolve until they became all-powerful. A secondary idea here was that the game would work to explain how the Flying Spaghetti Monster came to exist - by evolving from a normal dinner table. So the idea evolved more or less into this: you are sitting a table. You have your own plate, with is your 'base'. There are 5 other guests at the table, each with their own plate. Your plate can spawn little pieces of pasta. You do so by 'ordering' them through a menu. Some pastas are better than others; some are faster, some are stronger. They have varying 'costs', which are debited from your credits (you start with a number of credits). Once spawned, your pastas start flying around. Their instinct is to fly to other plates, in order to conquer them (the objective of the game is having your pasta conquer all the plates on the table). But they are really autonomous, so after being spawned, you have no control over your pasta (think DotA or LoL creeps). Your pasta doesn't like other people's pasta, so if they meet, they shoot sauce at each other until one dies. You get credits for other pastas your own pasta kill. Once a pasta is in the vicinity of a plate,"
]
# contents = []
new_prompts = []

for i in range(8):
    # content = {"prompt": prompts[i%len(prompts)], "stream": False, "max_tokens": 4096, "best_of": 2, "use_beam_search": True, "temperature": 0}
    # content = {"prompt": prompts[i%len(prompts)], "stream": False, "max_tokens": 4096}
    # contents.append(content)
    new_prompts.append(prompts[i % len(prompts)])
######################################
# new_prompts.clear()
# new_prompts.append(prompts[2])


# Create an LLM.
# llm = LLM(model="facebook/opt-125m", device="cpu", quantization="awq")
# llm = LLM(model="TheBloke/Llama-2-7b-Chat-AWQ", device="cpu", quantization="AWQ")
# llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", device="cpu", quantization="ns")
# TODO verify block_size >= max_model_len
# TODO set VLLM_CPU_KVCACHE_SPACE to X (GB) so that VLLM_CPU_KVCACHE_SPACE/(block_size*element_size) = num_cpu_blocks <= max_num_seqs. Otherwise, native kv cache may run out of slots.
ctx_size = 4096
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    device="cpu",
    max_num_seqs=64,
    block_size=ctx_size,
    max_model_len=ctx_size,
    quantization="ns",
)
# llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", device="cpu")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
for i in range(16):
    t0 = perf_counter()
    outputs = llm.generate(new_prompts, sampling_params)
    total_time = perf_counter() - t0
    # Print the outputs.
    total_prompts = 0
    total_generated = 0
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        total_prompts += len(prompt.split(" "))
        total_generated += len(generated_text.split(" "))
        # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    print(total_time, total_prompts, total_generated)
    print(f"prompts per second: {total_prompts/total_time}")
    print(f"tokens per second: {total_generated/total_time}")
