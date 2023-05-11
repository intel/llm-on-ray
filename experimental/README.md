# experimental

## description

This file is based on `Finetune/run_clm_no_trainer_ray.py`. The difference is that the implementation here splits several components into several independent plug-ins including `dataset`, `model`, `optimizer`, `tokenizer` and `trainer`. Users can select the appropriate components through configuration. `trainer` defines two interfaces `prepare` and `train`. `prepare` is to handle data preprocessing. `train` is to handle the training process. `main.py` is a global driver for loading different components and launching tasks.
