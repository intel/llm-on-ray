# Finetuning LLMs on Intel CPU/GPU

This guide provides detailed steps for finetuning LLMs on Intel CPU/GPU.

## Setup
Please follow [setup.md](setup.md) to setup the environment first.

## Prepare the Dataset for Finetuning

To finetune the model, your dataset must be in a JSONL (JSON Lines) format, where each line is a separate JSON object. Here’s the structure each line should follow:

``` json
{"instruction":"<User Input>", "context":"<Additional Information>", "response":"<Expected Output>"}
```

- Instruction: This is the user's input, such as a question, command, or prompt for content generation.
- Context: Supplementary information that aids the instruction. This can include previous conversation parts, background details, or specificities influencing the response. It's optional and can be left empty.
- Response: The model's expected output in response to the 'instruction', considering the 'context' if provided.

### Examples:
``` json
{"instruction":"Which is a species of fish? Tope or Rope", "context":"", "response":"Tope"}
{"instruction":"What is the average lifespan of a Golden Retriever?","context":"Golden Retrievers are a generally healthy breed; they have an average lifespan of 12 to 13 years. Irresponsible breeding to meet high demand has led to the prevalence of inherited health problems in some breed lines, including allergic skin conditions, eye problems and sometimes snappiness. These problems are rarely encountered in dogs bred from responsible breeders.","response":"The average lifespan of a Golden Retriever is 12 to 13 years."}
```

An example dataset can be accessed at `examples/data/sample_finetune_data.jsonl`. Ensure each line in your dataset follows the above format.

## Configure Finetuning Parameters

We provide an example configuration file ([CPU version](../llm_on_ray/finetune/finetune.yaml), [GPU version](../examples/finetune/gpt_j_6b/finetune_intel_gpu.yaml)) for finetuning LLMs. You can customize a few configruaitons such as the base model, the train file and the number of training workers, etc.

For CPU,  please make sure you set `device` to CPU, set CPU number for `resources_per_worker` and set `accelerate_mode` to CPU_DDP.
```
Training:
  device: CPU
  resources_per_worker:
    CPU: 32
  accelerate_mode: CPU_DDP
```
For GPU, please make sure you set `device` to GPU, set GPU number for `resources_per_worker` and set `accelerate_mode` to GPU_DDP or GPU_FSDP.
```
Training:
  device: GPU
  resources_per_worker:
    CPU: 1
    GPU: 1
  accelerate_mode: GPU_DDP
```
Please see [finetune_parameters.md](finetune_parameters.md) for details of the parameters.


### Validated Models
The following models have been verified on Intel CPUs or GPUs.
|Model|CPU|GPU|
|---|---|---|
|llama/llama2|✅|✅|
|mistral|✅|✅|
|pythia|✅|✅|
|gpt-j|✅|✅|
|bloom|✅|✅|
|opt|✅|✅|
|mpt|✅||
|gpt2|✅|✅|

## Finetune the model
To finetune your model, execute the following command. The finetuned model will be saved in /tmp/llm-ray/output by default.
``` bash
llm_on_ray-finetune --config_file <your finetuning conf file>
```
