import sys
import re
from typing import Dict, List

if len(sys.argv) < 4:
    raise ValueError(
        "need arguments, file path, number of expected iterations and expected generated token length"
    )

file_path = sys.argv[1]
nbr_iter = int(sys.argv[2])
expected_gen_token_len = int(sys.argv[3])

with open(file_path) as f:
    lines = f.readlines()

PAT_NBR_USERS = re.compile(r"Run num_prompts (\d+) (.+)")
PAT_ITER = re.compile(r"Run iter (\d+)")
PAT_ACTUAL_LEN = re.compile(
    r"Warning: the actual generated length is (\d+), which is different from the expected output length\((\d+)\)\."
)
PAT_TOTAL_TIME = re.compile(r"Total time: ([^ ]+) s")
PAT_PROMPT_LEN = re.compile(r"Prompt Length \(Min/Med/Max\): (\d+).+")
PAT_REQ_TPT = re.compile(r"Request Throughput \(QPS\): ([^ ]+) requests/s")
PAT_INPUT_TPT = re.compile(r"Input Token Throughput: ([^ ]+) tokens/s")
PAT_OUTPUT_TPT = re.compile(r"output Token Throughput: ([^ ]+) tokens/s")
PAT_REQ_LAT = re.compile(r"Average latency per Request: ([^ ]+) s")
PAT_TOK_LAT = re.compile(r"Average latency per Token: ([^ ]+) s")
PAT_FTOK_LAT = re.compile(r"Average latency for First Tokens: ([^ ]+) s")
PAT_NTOK_LAT = re.compile(r"Average latency for Next Tokens: ([^ ]+) s")

nbr_users_perf: Dict[int, List[Dict[str, float]]] = {}

token_lengths: List[int] = []

state = 0
current_nbr_user = -1
current_iter = -1

for no, line in enumerate(lines):
    if state == 0:
        m = PAT_NBR_USERS.match(line)
        if m:
            current_nbr_user = int(m.group(1))
            print("collecting number of users (num_prompts): ", current_nbr_user)
            nbr_users_perf[current_nbr_user] = []
            state = 1
    elif state == 1:
        m = PAT_ITER.match(line)
        if m:
            current_iter = int(m.group(1)) - 1
            nbr_users_perf[current_nbr_user].append({})
            state = 2
    elif state == 2:
        m = PAT_ACTUAL_LEN.match(line)
        if m:
            metrics = nbr_users_perf[current_nbr_user][current_iter]
            print(">>>", line, m.group(1))
            token_lengths.append(int(m.group(1)))
            if expected_gen_token_len != int(m.group(2)):
                raise ValueError(
                    "expected token lengths are not equal", expected_gen_token_len, m.group(2)
                )
        else:
            m = PAT_TOTAL_TIME.match(line)
            if m:
                metrics = nbr_users_perf[current_nbr_user][current_iter]
                full_gen_lens = token_lengths + [512] * (current_nbr_user - len(token_lengths))
                metrics["ACT_GEN_TOKENS"] = float(sum(full_gen_lens)) / current_nbr_user
                metrics["TOTAL_TIME"] = float(m.group(1))
                token_lengths = []
                state = 4
    elif state == 3:
        m = PAT_TOTAL_TIME.match(line)
        if m:
            metrics["TOTAL_TIME"] = float(m.group(1))
            state = 4
    elif state == 4:
        m = PAT_PROMPT_LEN.match(line)
        if m:
            metrics["PROMPT_LEN"] = float(m.group(1))
            state = 5
        else:
            raise ValueError("invalid line: " + line, no)
    elif state == 5:
        m = PAT_REQ_TPT.match(line)
        if m:
            metrics["REQ_TPT"] = float(m.group(1))
            state = 6
        else:
            raise ValueError("invalid line: " + line, no)
    elif state == 6:
        m = PAT_INPUT_TPT.match(line)
        if m:
            metrics["INPUT_TPT"] = float(m.group(1))
            state = 7
        else:
            raise ValueError("invalid line: " + line, no)
    elif state == 7:
        m = PAT_OUTPUT_TPT.match(line)
        if m:
            metrics["OUTPUT_TPT"] = float(m.group(1))
            state = 8
        else:
            raise ValueError("invalid line: " + line, no)
    elif state == 8:
        m = PAT_REQ_LAT.match(line)
        if m:
            metrics["REQ_LAT"] = float(m.group(1))
            state = 9
        else:
            raise ValueError("invalid line: " + line, no)
    elif state == 9:
        m = PAT_TOK_LAT.match(line)
        if m:
            metrics["TOK_LAT"] = float(m.group(1))
            state = 10
        else:
            raise ValueError("invalid line: " + line, no)
    elif state == 10:
        m = PAT_FTOK_LAT.match(line)
        if m:
            metrics["FTOK_LAT"] = float(m.group(1))
            state = 11
        else:
            raise ValueError("invalid line: " + line, no)
    elif state == 11:
        m = PAT_NTOK_LAT.match(line)
        if m:
            metrics["NTOK_LAT"] = float(m.group(1))
            if current_iter == nbr_iter - 1:
                state = 0
                current_iter = -1
                current_nbr_user = -1
            else:
                state = 1
                current_iter = -1
        else:
            raise ValueError("invalid line: " + line, no)

if nbr_users_perf:
    print(nbr_users_perf)
    for k, values in nbr_users_perf.items():
        print("number of users: ", k)
        size = len(values)
        if size != nbr_iter:
            raise ValueError(
                "size should be equal to number of interations, "
                + str(size)
                + " != "
                + str(nbr_iter)
            )
        metrics = {
            "ACT_GEN_TOKENS": 0.0,
            "PROMPT_LEN": 0.0,
            "TOTAL_TIME": 0.0,
            "REQ_TPT": 0.0,
            "INPUT_TPT": 0.0,
            "OUTPUT_TPT": 0.0,
            "REQ_LAT": 0.0,
            "TOK_LAT": 0.0,
            "FTOK_LAT": 0.0,
            "NTOK_LAT": 0.0,
        }
        for v in values:
            for kk in metrics:
                metrics[kk] += v[kk]
        for kk, vv in metrics.items():
            metrics[kk] = vv / size
        print(metrics)
        print("=========================================")


else:
    raise ValueError("Failed to collect metrics")
