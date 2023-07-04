import numpy as np

from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
from megatron.data.gpt_dataset import GPTDataset

from .dataset import Dataset

class MegatronDataset(Dataset):
    def __call__(self, config):
        name = config.get("name", "test")
        path = config.get("path")
        impl = config.get("impl", "mmap")
        seq_length = config.get("seq_length", 1024)
        seed = config.get("seed", 42)
        use_seq_len_plus_one_tokens = config.get("use_seq_len_plus_one_tokens", True)

        indexed_dataset = make_indexed_dataset(path, impl)
        data_prefix = path
        train_samples = len(indexed_dataset)
        documents = np.arange(start=0, stop=train_samples, step=1, dtype=np.int32)
        dataset = GPTDataset(name, data_prefix,
                                documents, indexed_dataset,
                                train_samples,
                                seq_length, seed, use_seq_len_plus_one_tokens)
        return dataset
