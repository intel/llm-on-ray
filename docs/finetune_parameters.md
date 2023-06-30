# Finetuning Parameters 
The following are the parameters supported in the finetuning workflow.


## General Parameters

|Configuration Name| Default|Meaning|
|-|-|-|
|base_model| mosaicml/mpt-7b|Path to pretrained model or model identifier from huggingface.co/models|
|output_dir|/tmp/llm-ray/output|The output directory to store the finetuned model|
|checkpoint_dir|/tmp/llm-ray/checkpoint|The directory to store checkpoint|



## Dataset Parameters
|Configuration Name| Default|Meaning|
|-|-|-|
|train_file|examples/data/sample_finetune_data.jsonl|A json file containing the training data.|
|validation_file|None|A json file containing the validation data.|
|validation_split_percentage|5|The percentage of the train set used as validation set in case there's no validation split|
|preprocessing_num_workers|None|The number of processes to use for the preprocessing.|

## Training Parameters
|Configuration Name| Default|Meaning|
|-|-|-|
|optimizer|AdamW|The optimizer used for model training. Supported values: "Adadelta", "Adagrad", "Adam", "AdamW", "Adamax", "ASGD", "NAdam", "RAdam", "RMSprop", "Rprop", "SGD"|
|batch_size|4|Batch size per training worker|
|epochs|3|Total number of training epochs to perform.|
|learning_rate|1e-5|Initial learning rate to use.|
|lr_scheduler|linear|The scheduler type to use, supported value: "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"|
|weight_decay|0.0|Weight decay is a regularization technique that adds an L2 norm of all model weights to the loss function while increasing the probability of improving the model generalization.|
|num_training_workers|2|The number of the training process|
|resources_per_worker|{"CPU": 32}|A dict to specify the resources for each worker. Currently only CPU is supported.|
|max_train_steps|None|Total number of training steps to perform. If provided, overrides epochs.|
|gradient_accumulation_steps|1|Number of updates steps to accumulate before performing a backward/update pass.|
|seed|None|A seed for reproducible training.|





