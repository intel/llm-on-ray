import os
import datasets

from common.dataset import Dataset


class GroupDataset(Dataset):
    def __call__(self, config):
        path = config.get("path")
        load_from_disk = config.get("load_from_disk", False)
        load_config = config.get("load_config", {})
        names = self.get_all_file(path)
        if load_from_disk:
            return [datasets.load_from_disk(name, **load_config) for name in names]
        else:
            return [datasets.load_dataset(name, **load_config) for name in names]

    def get_all_file(self, path):
        files = os.listdir(path)
        list.sort(files)
        return [os.path.join(path, file) for file in files]
