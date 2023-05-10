from typing import Any, Dict, List, Tuple, Union
from . import dataset
from . import tokenizer
from . import model
from . import optimizer
from . import trainer
from . import config

def load_dataset(config: Dict[str, Any]):
    if not isinstance(config, Dict):
        raise ValueError("Dataset config type error, config should be a Dict not a %s"%type(config))
    else:
        dataset_type = config.get("type")
        Factory = dataset.Dataset.registory.get(dataset_type)
        if Factory is None:
            raise ValueError("there is no %s dataset."%dataset_type)
        else:
            return Factory()(config)

def load_tokenizer(config: Dict[str, Any]):
    if not isinstance(config, Dict):
        raise ValueError("Tokenizer config type error, config should be a Dict not a %s"%type(config))
    else:
        tokenizer_type = config.get("type")
        Factory = tokenizer.Tokenizer.registory.get(tokenizer_type)
        if Factory is None:
            raise ValueError("there is no %s tokenizer."%tokenizer_type)
        else:
            return Factory()(config)

def load_model(config: Dict[str, Any]):
    if not isinstance(config, Dict):
        raise ValueError("Model config type error, config should be a Dict not a %s"%type(config))
    else:
        model_type = config.get("type")
        Factory = model.Model.registory.get(model_type)
        if Factory is None:
            raise ValueError("there is no %s model."%model_type)
        else:
            return Factory()(config)

def load_optimizer(model, config: Dict[str, Any]):
    if not isinstance(config, Dict):
        raise ValueError("Optimizer config type error, config should be a Dict not a %s"%type(config))
    else:
        optimizer_type = config.get("type")
        Factory = optimizer.Optimizer.registory.get(optimizer_type)
        if Factory is None:
            raise ValueError("there is no %s optimizer."%model_type)
        else:
            return Factory()(model, config)

def get_trainer(config: Dict[str, Any]):
    if not isinstance(config, Dict):
        raise ValueError("Trainer config type error, config should be a Dict not a %s"%type(config))
    else:
        trainer_type = config.get("type")
        Factory = trainer.Trainer.registory.get(trainer_type)
        if Factory is None:
            raise ValueError("there is no %s trainer."%trainer_type)
        return Factory(config)
