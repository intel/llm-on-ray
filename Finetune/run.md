## Accelerate + Ray 
(This still has bugs to be fixed)
### 1. Prepare environment
Follow [LLM Finetune](https://wiki.ith.intel.com/pages/viewpage.action?spaceKey=AppliedML&title=LLM+Finetune)

### 2. Install Ray dependency
```bash
pip install ray
```
### 3. Test ray TorchTrainer example
```bash
python -u examples/pytorch/language-modeling/run_clm_no_trainer_ray.py --model_name_or_path  EleutherAI/gpt-j-6B --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1  --per_device_train_batch_size 2  --per_device_eval_batch_size 4  --num_train_epochs 1 --address 10.0.2.140 --num_workers 2
```

## FSDP_CPU + Ray
### 1. Enable fsdp_cpu in Ray
Edit codes in train_loop_utils.py
```python
class _TorchAccelerator(Accelerator):
    def prepare_model(...):
        ...
        # if not torch.cuda.is_available():
        #     raise RuntimeError(
        #         "FSDP is only available with GPU-enabled "
        #         "training. Set "
        #         "`use_gpu=True` in your Trainer to train with "
        #         "GPUs."
        #     )
        parallel_strategy_kwargs = {
                "device_id": device,
                **parallel_strategy_kwargs,
        }
        DataParallel = FullyShardedDataParallel
```
Then enable `parellel_strategy` when running on CPU.
```python
def train_func(config: Dict):
    ...
    model = train.torch.prepare_model(model, parallel_strategy="fsdp")
    ...
```

### 2. enable torch_ccl in Ray
```bash
pip install --pre raydp
```
The codes that need to add:
```python
from raydp.torch.config import TorchConfig

def train_fashion_mnist(...):
    torch_config = TorchConfig(backend="ccl")       # enable ccl
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={"lr": 1e-3, "batch_size": 64, "epochs": 4},
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        torch_config=torch_config,      # pass to TorchTrainer
    )
    ...
```
If this is not done, the following error will appear:
```bash
File "env/lib/python3.7/site-packages/ray/train/_internal/worker_group.py", line 31, in __execute
    raise skipped from exception_cause(skipped)
  File "env/lib/python3.7/site-packages/ray/train/_internal/utils.py", line 129, in discard_return_wrapper
    train_func(*args, **kwargs)
  File "/home/lzhi/LLM_project/LLM_5/transformers/examples/pytorch/language-modeling/run_minist_fsdp.py", line 112, in train_func
    train_epoch(train_dataloader, model, loss_fn, optimizer)
  File "/home/lzhi/LLM_project/LLM_5/transformers/examples/pytorch/language-modeling/run_minist_fsdp.py", line 57, in train_epoch
    pred = model(X)
  File "env/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "env/lib/python3.7/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py", line 2741, in forward
    self._pre_forward(self._handles, unshard_fn, unused, unused)
  File "env/lib/python3.7/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py", line 2773, in _pre_forward
    self._exec_order_data.record_pre_forward(handles, self.training)
  File "env/lib/python3.7/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py", line 535, in record_pre_forward
    self._check_order(handles_key, is_training)
  File "env/lib/python3.7/site-packages/torch/distributed/fsdp/fully_sharded_data_parallel.py", line 578, in _check_order
    group=self.process_group,
  File "env/lib/python3.7/site-packages/torch/distributed/distributed_c10d.py", line 2392, in _all_gather_base
    return all_gather_into_tensor(output_tensor, input_tensor, group, async_op)
  File "env/lib/python3.7/site-packages/torch/distributed/distributed_c10d.py", line 2358, in all_gather_into_tensor
    work = group._allgather_base(output_tensor, input_tensor)
RuntimeError: no support for _allgather_base in Gloo process group
```

### 3. Test Fashion MNIST example
```python
python run_minist_fsdp.py
```
