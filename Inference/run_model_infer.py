import requests
import time
import argparse

parser = argparse.ArgumentParser('Model Inference Script', add_help=False)
parser.add_argument('--model_endpoint', default='http://127.0.0.1:8000/', type=str, help="fp32 or bf16")
args = parser.parse_args()
prompt = "Once upon a time,"
sample_input = {"text": prompt}
total_time = 0.0
num_iter = 10
num_warmup = 3
for i in range(num_iter):
    print("iter: ", i)
    tic = time.time()
    proxies = { "http": None, "https": None}
    print("post: ", args.model_endpoint)
    output = requests.post(args.model_endpoint, proxies=proxies, json=[sample_input]).text
    toc = time.time()
    print(output, flush=True)
    if i >= num_warmup:
        total_time += (toc - tic)

print("Inference latency: %.3f ms." % (total_time / (num_iter - num_warmup) * 1000))
