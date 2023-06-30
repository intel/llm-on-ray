import os
import datasets

from .dataset import Dataset

def local_load(name, **load_config):
    if os.path.isfile(name):
        file = os.path.basename(os.path.abspath(name))
        path = os.path.dirname(os.path.abspath(name))
        dataset = datasets.load_dataset(path, data_files=file, **load_config)
    else:
        dataset = datasets.load_dataset(name, **load_config)
    return dataset["train"]

class HuggingfaceDataset(Dataset):
    def __call__(self, config):
        name = config.get("name")
        load_from_disk = config.get("load_from_disk", False)
        validation_file = config.get("validation_file", None)
        validation_split_percentage = config.get("validation_split_percentage", 0)
        if os.path.exists(name):
            train_dataset = local_load(name)
            if validation_file is not None:
                validation_dataset = local_load(validation_file)
                return datasets.DatasetDict({"train":train_dataset, "validation_dataset": validation_dataset})
            if validation_split_percentage/100 > 0.0 and validation_split_percentage/100 < 1.0:
                datasets_dict = train_dataset.train_test_split(test_size = validation_split_percentage/100)
                datasets_dict["validation"] = datasets_dict["test"]
                return datasets_dict
            return datasets.DatasetDict({"train":train_dataset})
        else:
            load_config = config.get("load_config", {})
            if load_from_disk:
                return datasets.load_from_disk(name, **load_config)
            else:
                return datasets.load_dataset(name, **load_config)
