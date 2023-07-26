import torch

from megatron.data.data_samplers import MegatronPretrainingSampler

from common.dataprocesser import DataProcesser

class _MegatronPretrainingSampler(MegatronPretrainingSampler):
    def __len__(self):
        return self.total_samples // self.micro_batch_times_data_parallel_size

class MegatronProcesser(DataProcesser):
    def prepare(self, tokenizer, dataset, **kwargs):
        per_device_train_batch_size = self.config.get("per_device_train_batch_size", 2)
        seed = self.config.get("seed", 42)

        data_parallel_rank = kwargs.get("rank", 0)
        data_parallel_size = kwargs.get("size", 1)
        starting_step = kwargs.get("starting_step", 0)

        consumed_samples = starting_step * data_parallel_size * per_device_train_batch_size
        batch_sampler = _MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=per_device_train_batch_size,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size
        )
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            generator=torch.Generator().manual_seed(seed),
            batch_sampler=batch_sampler,
            collate_fn=None,
            pin_memory=True
        )
        eval_dataloader = None
        return [train_dataloader, eval_dataloader]