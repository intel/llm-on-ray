# Finetuning Parameters 
The following are the parameters supported in the finetuning workflow.


## General Parameters

|Configuration Name| Default|Meaning|
|-|-|-|
|base_model| EleutherAI/gpt-j-6b|Path to pretrained model or model identifier from huggingface.co/models|
|tokenizer_name|None|Path to pretrained tokenizer from huggingface.co/models. If not provided, the tokenizer will be loaded from the `base_model`.|
|gpt_base_model|True|This parameter is for [Transformers#22482](https://github.com/huggingface/transformers/issues/22482). It needs to be set to True when the pretrained model is realted to gpt, otherwise it is False.|
|output_dir|/tmp/llm-ray/output|The output directory to store the finetuned model.|
|report_to|none|The list of integrations to report the results and logs to. Possible values are: "none", "tensorboard".|
|resume_from_checkpoint|null|The path to a folder with a valid checkpoint for your model.|
|save_strategy|no|The checkpoint save strategy to adopt during training. Possible values are: "no", "epoch", "steps".|
|config|trust_remote_code: False<br> use_auth_token: None|Will be passed to the transformers `from_pretrained()` method|
|lora_config|task_type: CAUSAL_LM<br>r: 8<br>lora_alpha: 32<br>lora_dropout: 0.1|Will be passed to the LoraConfig `__init__()` method, then it'll be used as config to build Peft model object.|
|enable_gradient_checkpointing|False|enable gradient checkpointing to save GPU memory, but will cost more compute runtime|


## Dataset Parameters
|Configuration Name| Default|Meaning|
|-|-|-|
|train_file|examples/data/sample_finetune_data.jsonl|A json file containing the training data.|
|validation_file|None|A json file containing the validation data.|
|validation_split_percentage|5|The percentage of the train set used as validation set in case there's no validation split|
|preprocessing_num_workers|None|The number of processes to use for the preprocessing.|
|max_length|512|Padding sequential data to max length of a batch|
|group|True|Whether to concatenate the sentence for more efficient training|
|block_size|512|The block size of concatenated sentence|
|shuffle|False|Whether shuffle the data at every epoch|


## Training Parameters
|Configuration Name| Default|Meaning|
|-|-|-|
|optimizer|adamw_torch|The optimizer to use: adamw_hf, adamw_torch, adamw_torch_fused, adamw_apex_fused, adamw_anyprecision or adafactor. for more optimizer names, please search OptimizerNames [here](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py)|
|batch_size|4|Batch size per training worker|
|epochs|3|Total number of training epochs to perform.|
|learning_rate|1e-5|Initial learning rate to use.|
|lr_scheduler|linear|The scheduler type to use, supported value: "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"|
|weight_decay|0.0|Weight decay is a regularization technique that adds an L2 norm of all model weights to the loss function while increasing the probability of improving the model generalization.|
|mixed_precision|no|Whether or not to use mixed precision training. Choose from "no", "fp16", "bf16". Default is "no" if not set.|
|device|CPU|The device type used, supported types are {"CPU", "GPU", "HPU"}, here "GPU" is Intel GPU, "HPU" is Habana Gaidu device.|
|num_training_workers|2|The number of the training process.|
|resources_per_worker|{"CPU": 32}|A dict to specify the resources for each worker. If `device` is "GPU", please set it like {"CPU": 1, "GPU": 1}. If `device` is "HPU", please set it like {"CPU": 1, "HPU": 1}.|
|accelerate_mode|DDP|The accelerate mode for training model, available options are: "DDP", "FSDP", "DEEPSPEED".|
|max_train_steps|None|Total number of training steps to perform. If provided, overrides epochs.|
|gradient_accumulation_steps|1|Number of updates steps to accumulate before performing a backward/update pass.|
|seed|None|A seed for reproducible training.|
|logging_steps|10|logging per steps|
|deepspeed_config_file|None|The config file for deepspeed, the format must be json file.|
