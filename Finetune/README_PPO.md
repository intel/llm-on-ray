# Reinforcement Learning by Human Feedback Workflow

## Supervised Fine-Tuning (SFT) Workflow

### 1. Prepare Dataset

The standard data format should be as follows:

```json
{
    "instruction": "Why can camels survive for long without water?",
    "context": "",
    "response": "Camels use the fat in their humps to keep them filled with energy and hydration for long periods of time."
}
// ...
```

If you want to use you customed data, you can convert your data to the format by referring to `Finetune/process_data.py`.

After you have prepared you own data, you can use it by setting `train_file` in `rlhf/reward.conf` as following:

```json
"train_file": "/path/to/your/sft_data.jsonl",
```

Additionally, we provide a simple data example at the path `examples/data/sample_finetune_data.json`.

### 2. Training SFT model

You can run the following command to start a SFT model training:

```bash
python Finetune/finetune.py --config_path Finetune/finetune.conf 
```

Once the model training is completed, based on your settings in the configuration file (such as `"checkpoint_dir": "/tmp/llm-ray/checkpoint/sft"`), you will obtain the final trained model in the `/tmp/llm-ray/checkpoint/sft` directory. This trained model will be utilized in the final stage of the PPO training process.

## Reward Model Workflow

### 1. Prepare Dataset

The standard data format should be as follows:

```json
{
    "instruction":"Explain how potatoes grow",
    "chosen":"Potatoes grow underground, and then they send up stems called sprouts. Those sprouts grow into the familiar potatoes we eat.",
    "rejected":"Potatoes grow underground in large, firm tubers. Once the potatoes have grown large enough, they sprout up above ground."
}

// ...

```

If you want to use you customed data, you can convert your data to the format by referring to `Finetune/process_data_rlhf.py`. 

After you have prepared you own data, you can use it by setting `train_file` in `rlhf/reward.conf` as following:

```json
"train_file": "/path/to/your/rm_data.jsonl",
```

Additionally, we provide a simple data example at the path `examples/data/sample_rm_data.json`.

### 2. Training reward model

You can run the following command to start a reward model training:

```bash
python rlhf/reward_trainer.py --config_path rlhf/reward.conf
```
Once the reward model training is complete, based on your settings in the configuration file (such as `"checkpoint_dir": "/tmp/llm-ray/checkpoint/rm"`), you will obtain the final trained model in the `/tmp/llm-ray/checkpoint/rm` directory. This trained reward model will also be utilized in the final stage of the PPO training process.

### 3. Config Detail

In the `General` configuration, the model should be specified by setting the `base_model` parameter. Additionally, the `output_dir` directory contains the training log, while the `checkpoint_dir` directory is used to save the trained model.

```json
"General": {
    "base_model": "EleutherAI/gpt-j-6b",
    "output_dir": "/tmp/output",
    "checkpoint_dir": "/tmp/checkpoint"
}
```

Secondly, the `train_file` parameter specifies the path to the training data, which should adhere to our specified format. If you have separate validation data, you can set it using the `validation_file` parameter. Alternatively, you can utilize the `validation_split_percentage` parameter to obtain a portion of the validation data from the `train_file`.

```json
"Dataset": {
    "train_file": "/path/to/reward_data.jsonl",
    "validation_file": None,
    "validation_split_percentage": 5
},
```

Finally, we should configure the training parameters as follows:

```json
"Training": {
    "optimizer": "AdamW",
    "batch_size": 2,
    "epochs": 3,
    "learning_rate": 1e-5,
    "lr_scheduler": "linear",
    "weight_decay": 0.0,
    "num_training_workers": 2,
    "resources_per_worker": {
        "CPU": 32
    },
}
```

- `optimizer`: Specifies the optimization algorithm to be used during training. In this case, the "AdamW" optimizer is chosen.
- `batch_size`: Defines the number of training examples processed in a single iteration.
- `epochs`: Determines the number of complete passes through the training dataset during the training process.
- `learning_rate`: Sets the initial learning rate for the optimizer. It determines the step size taken during the parameter updates.
- `lr_scheduler`: Specifies the learning rate scheduler to adjust the learning rate during training. Here, the "linear" scheduler is utilized.
- `weight_decay`: Controls the amount of regularization applied to the model's weights to prevent overfitting.
- `num_training_workers`: Specifies the number of workers (parallel processes) to be used for training.
- `resources_per_worker`: Defines the amount of computing resources allocated per training worker, with "CPU" referring to the number of CPU cores assigned per worker. In this case, each worker is assigned 32 CPU cores.



## PPO (Proximal Policy Optimization) Workflow

### 1. Prepare Dataset

The standard data format should be as follows:

```json
{"prompt":"Explain how potatoes grow"}
{"prompt":"How do I keep my ears warm in the winter?"}
// ...
```

During the PPO training stage, only `prompts` are required as training data. If you want to use you customed data, you can convert your data to the format by referring to `Finetune/process_data_rlhf.py`. 


After you have prepared you own data, you can use it by setting `train_file` in `rlhf/ppo.conf` as following:

```json
"train_file": "/path/to/your/ppo_data.jsonl",
```

Additionally, we provide a simple data example at the path `examples/data/sample_ppo_data.json`.


### 2. PPO Training


You can run the following command to start a ppo training:

```bash
python rlhf/ppo_trainer.py --config_path rlhf/ppo.conf
```

It is important to note that before training, we need to configure the corresponding settings in `ppo.conf` based on the saved paths of the SFT (Structured Fine-Tuning) model and the RM (Reward Model) model. Assuming that we are using pre-trained models of `EleutherAI/gpt2` and considering the previous model save settings, we should configure the following in the `ppo.conf`:

```json
"General": {
    "model_name": "EleutherAI/gpt2",
    "model_pretrain": "/tmp/llm-ray/checkpoint/sft",
    "rm_name": "EleutherAI/gpt2",
    "rm_pretrain": "/tmp/llm-ray/checkpoint/rm"
}
```

Additionally, in the provided `Finetune/ppo.conf` configuration, we have set `model_pretrain` and `rm_pretrain` to `None`. This ensures that the original model parameters are directly used for training, without reloading any model parameters. This avoids redundant loading in cases where the model loaded from Hugging Face is already a trained SFT or RM model, thereby preventing duplicate loading.

### 3. Config Detail

In the `General` configuration, the parameter explanations are as follows:

```json
"General": {
    "model_name": "EleutherAI/gpt2",
    "model_pretrain": "/tmp/llm-ray/checkpoint/sft",
    "rm_name": "EleutherAI/gpt2",
    "rm_pretrain": "/tmp/llm-ray/checkpoint/rm",
}
```

- `model_name`: Specifies the name or identifier of the base model architecture to be used for PPO training. In this case, "EleutherAI/gpt2" is chosen.
- `model_pretrain`: Specifies the path to a pre-trained model checkpoint for initialization. It allows starting the training from a pre-existing model state. 
- `rm_name`: Specifies the name or identifier of the reward model architecture to be used. Here, "EleutherAI/gpt2" is used as the reward model.
- `rm_pretrain`: Specifies the path to a pre-trained reward model checkpoint for initialization. Similar to `model_pretrain`, this parameter allows starting the training with a pre-existing reward model checkpoint. 


The configuration for `Dataset` is the same with previous stage.

```json
"Dataset": {
    "train_file": "examples/data/sample_ppo_data.jsonl",
    "validation_file": None,
    "validation_split_percentage": 5
},
```
In PPO, most of the parameters remain the same as the ones used in the reward training stage. The `kl_coeff` parameter controls the strength of the KL (Kullback-Leibler) divergence regularization term. By adjusting the value of `kl_coeff`, you can control the trade-off between exploration and exploitation in the training process. A higher value of kl_coeff will place more emphasis on maintaining policy stability, while a lower value will allow for more exploration and potentially faster learning.

```json
"Training": {
    "optimizer": "AdamW",
    "experience_batch_size": 2,
    "epochs": 3,
    "learning_rate": 1e-5,
    "kl_coeff": 0.2,
    "num_training_workers": 2,
    "resources_per_worker": {
        "CPU": 32
    },
},
```


