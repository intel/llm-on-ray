### Requirements
This module requires to install Ray Serve. Install it as `pip install ray[serve]`

### How to run
First, we need to start the server as `python run_model_serve.py`. You can add arguments like `--model` to choose what model to use, by default it is *EleutherAI/gpt-j-6B*. You should specify `--num-cores` to utilize multi-threading to accelerate inferrence.  Other arguments include `--max-new-tokens` and `--precision`.

The deployment will take a few minutes to load the model and initialize, depending on how large the model is. If the specified model has not been cached locally, it needs to be downloaded first.

After it prints "Model is deployed successfully", it's ready to accept http requests. You can run `python run_model_infer.py` to make a request. 

Example output:
> Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun. Most of all, she wanted to have a long, long life. Unfortunately, the world didn’t always want to grant this wish, so she decided