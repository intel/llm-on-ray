The LLM on Ray finetuning workflow supports running with Intel GPU.

The following is an example to setup the environment and finetune a LLM model.

## Hardware and Software requirements

### Hardware Requirements

[Intel® Data Center GPU Max Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/max-series/products.html)

|Product Name|Launch Date|Memory Size|Xe-cores|
|---|---|---|---|
|[Intel® Data Center GPU Max 1550](https://www.intel.com/content/www/us/en/products/sku/232873/intel-data-center-gpu-max-1550/specifications.html)|Q1'23|128 GB|128|
|[Intel® Data Center GPU Max 1100](https://www.intel.com/content/www/us/en/products/sku/232876/intel-data-center-gpu-max-1100/specifications.html)|Q2'23|48 GB|56|

### Software Requirements
Workflow has been tested on SUSE Linux Enterprise Server 15 SP4 (Linux borealis-uan1 5.14.21-150400.22-default)
- conda
- Python 3.9

## Setup

### Build conda environment
``` bash
conda create -n llm_ray_gpu python=3.9
# after install necessary python modules
conda activate llm_ray_gpu
```

### Clone codes
``` bash
git clone https://github.com/intel-sandbox/llm-ray.git llm-ray-gpu
cd llm-ray-gpu
```

### Install dependencies

Suppose you have installed [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html)

``` bash
pip install -r requirements.intel.gpu.txt
# install ipex/oneccl/torch/torchvision with intel gpu version
pip install torch==2.0.1a0 torchvision==0.15.2a0 intel-extension-for-pytorch==2.0.110+xpu oneccl-bind-pt --extra-index-url https://developer.intel.com/ipex-whl-stable-xpu
```
More versions are available [here](https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/torch/). More detail information's are [here](https://intel.github.io/intel-extension-for-pytorch/)

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

## Configurations for Finetuning

The workflow is designed for configure driven.

Detail finetuning configure parameters are [here](../docs/finetune_parameters.md)

To finetune on Intel GPU, following options may be modified:

``` json
{
    "General": {
        "base_model": "EleutherAI/gpt-j-6b",
    },
    "Dataset": {
        "train_file": "examples/data/sample_finetune_data.jsonl",
    },
    "Training": {
        "device": "GPU",
        "num_training_workers": 2,
        "accelerate_mode": "GPU_DDP",
        "resources_per_worker": {
            "CPU": 1,
            "GPU": 1,
        },
    },
}
```

### Models

Finetuning workflow support many models as base model, following models are verified:
- gpt-j-6b
- pythia-70m/160m/410m/1b/1.4b/2.8b/6.9b/12b
- Llama-2-7b

### Example

We provide an [example](../examples/finetune/gpt_j_6b/finetune_intel_gpu.conf) configuration for finetuning `gpt-j-6b` on Intel GPUs.

## Finetuning

``` bash
python finetune/finetune.py --config_path examples/finetune/gpt_j_6b/finetune_intel_gpu.conf
```

