## Accelerate + Ray 
### 1. Prepare environment
```bash
# on head node 
git clone https://github.com/intel-sandbox/llm-ray.git
cd llm-ray 
./build-image.sh 
docker save -o ray-image.tar ray-llm:latest
# on worker node   
docker load -i ray-image.tar 
```

### 2. Start the containers 
```bash 
# on head node 
./run.sh head 

# on worker node 
./run.sh worker
```

### 3. Enable torch_ccl
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

### 4. Set parameters
- FSDP parameters 
  ```python
  trainer = AccelerateTrainer(
              ...
              # accelerate_config = None,
              accelerate_config={
                "distributed_type": "MULTI_CPU", 
                "fsdp_config": {}, 
                "num_machines": 1, 
                "num_processes": 2, 
                "use_cpu": "true"
              },
              ...
            )
  …
  ```
  You can config FSDP via accelerate_config parameter. It can be a path to a file generated with ``accelerate config``,  a configuration dict or None, in which case it will load the configuration file from the default location as defined by Accelerate.

- Network and fault tolerance parameters can be set in runtime_env
  ```python
  runtime_env = {
        "env_vars": {
            "FI_PROVIDER": "tcp",         # Network setting
            "FI_TCP_IFACE": "***", 
            "JAVA_HOME": os.getenv("JAVA_HOME"),    # HDFS setting
            "CLASSPATH": os.getenv("CLASSPATH"),
            "ARROW_LIBHDFS_DIR": os.getenv("ARROW_LIBHDFS_DIR"),
        }
    }
  ```

### 5. Test Ray TorchTrainer example
```bash
python -u run_clm_no_trainer_ray.py --model_name_or_path  EleutherAI/gpt-j-6B --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1  --per_device_train_batch_size 2  --per_device_eval_batch_size 4  --num_train_epochs 1 --address 10.165.9.166 --num_workers 2
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
The codes that need to be added:
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

## Memory Status
Reference to Applied Machine Learning team ([intel-sandbox/HuggingFace](https://github.com/intel-sandbox/HuggingFace/tree/main/test/memory))
- First run finetune code and get the pid of a Ray worker process.
- Run memory_collect_ray.py to generate memory csv results.
- Run csv_analysis.py to generate comparison result.

## Fault tolerance
The parameter `ray_fault_tolerance` is used to set Ray fault tolerance, this can allow training to recover from failures. The default value 0 is to disable it, -1 will lead to infinite recovery retries, and n will recover a run at least n times. Now, Ray's default checkpoint mechanism can only support ddp training mode. So we use HDFS to support fault tolerance of FSDP training mode. Please set the parameter `checkpoint_hdfs_path`, and also make sure `JAVA_HOME`, `CLASSPATH`, `ARROW_LIBHDFS_DIR` can be taken from env.

## Ray debugging
- You can set parameter `ray_debug_error` to enable Ray post mortem debugging. It can help you debug code when an error happens.
- You can add breakpoint() everywhere in train_func(), then you'll drop into a PDB session to use Ray debugger when this breakpoint is hit.

Please refer to [Ray Debugger](https://docs.ray.io/en/master/ray-observability/ray-debugging.html) for more details.
