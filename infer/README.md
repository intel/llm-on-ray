### Requirements
This module requires to install Ray Serve. Install it as `pip install ray[serve]`

### How to run
Refer to [here](https://github.com/intel/intel-extension-for-transformers/tree/main/examples/optimization/pytorch/huggingface/language-modeling/inference)

First, we need to start the ray cluster via `ray start --head`. Before that, we should export some environmental variables:
```bash
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0

# IOMP
export OMP_NUM_THREADS=56
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

```


Then, we need to start the server as `python run_model_serve.py`.  Other arguments include `--max-new-tokens` and `--precision`.

The deployment will take a few minutes to load the model and initialize, depending on how large the model is. If the specified model has not been cached locally, it needs to be downloaded first.

After it prints "Service is deployed successfully", it's ready to accept http requests. You can run `python run_model_infer.py` to post requests and test the latency.

Example output:
> Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun. Most of all, she wanted to have a long, long life. Unfortunately, the world didn’t always want to grant this wish, so she decided