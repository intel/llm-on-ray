import requests
import time

prompt = "Once upon a time,"
sample_input = {"text": prompt}
total_time = 0.0
num_iter = 10
num_warmup = 3
for i in range(num_iter):
    tic = time.time()
    output = requests.post("http://127.0.0.1:8000/", json=[sample_input]).text
    toc = time.time()
    print(output, flush=True)
    if i >= num_warmup:
        total_time += (toc - tic)

print("Inference latency: %.3f ms." % (total_time / (num_iter - num_warmup) * 1000))
