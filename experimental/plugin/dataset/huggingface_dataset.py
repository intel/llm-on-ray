import datasets

from .dataset import Dataset

class HuggingfaceDataset(Dataset):
    def __call__(self, config):
        name = config.get("name")
        if name is None:
            raise ValueError("Dataset config error, config should contain a dataset name")
        is_local = config.get("is_local", False)
        load_config = config.get("load_config", {})
        if is_local:
            return datasets.load_from_disk(name, **load_config)
        else:
            return datasets.load_dataset(name, **load_config)