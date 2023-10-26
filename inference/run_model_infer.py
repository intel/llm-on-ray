import requests
import time
import argparse

parser = argparse.ArgumentParser("Model Inference Script", add_help=False)
parser.add_argument("--model_endpoint", default="http://127.0.0.1:8000", type=str, help="deployed model endpoint")
parser.add_argument("--streaming_response", default=False, action="store_true", help="whether to enable streaming response")
parser.add_argument("--max_new_tokens", default=None, help="The maximum numbers of tokens to generate")
parser.add_argument("--temperature", default=None, help="The value used to modulate the next token probabilities")
parser.add_argument("--top_p", default=None, help="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to`Top p` or higher are kept for generation")
parser.add_argument("--top_k", default=None, help="The number of highest probability vocabulary tokens to keep for top-k-filtering")
parser.add_argument("--num_iter", default=10, type=int, help="The Number of inference iterations")

args = parser.parse_args()
prompt = "Once upon a time,"
config = {}
if args.max_new_tokens:
    config["max_new_tokens"] = int(args.max_new_tokens)
if args.temperature:
    config["temperature"] = float(args.temperature)
if args.top_p:
    config["top_p"] = float(args.top_p)
if args.top_k:
    config["top_k"] = float(args.top_k)

sample_input = {"text": prompt, "config": config, "stream": args.streaming_response}

total_time = 0.0
num_iter = args.num_iter
num_warmup = 3
for i in range(num_iter):
    print("iter: ", i)
    tic = time.time()
    proxies = { "http": None, "https": None}
    outputs = requests.post(args.model_endpoint, proxies=proxies, json=[sample_input], stream=args.streaming_response)
    if args.streaming_response:
        outputs.raise_for_status()
        for output in outputs.iter_content(chunk_size=None, decode_unicode=True):
            print(output, end='', flush=True)
        print()
    else:
        print(outputs.text, flush=True)
    toc = time.time()
    if i >= num_warmup:
        total_time += (toc - tic)

print("Inference latency: %.3f ms." % (total_time / (num_iter - num_warmup) * 1000))
