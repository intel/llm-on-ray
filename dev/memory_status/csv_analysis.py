#
# Copyright 2023 The LLM-on-Ray Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import csv

import matplotlib.pyplot as plt


csv_w_fdsp = "./res/rss_per_process_w_FSDP.csv"
f = open(csv_w_fdsp, mode="r", encoding="utf-8", newline="")
csv_reader = csv.DictReader(
    f,
    fieldnames=[
        "rss",
        "vms",
        "data",
    ],
)
next(csv_reader)
rss_single = list()
for line in csv_reader:
    rss_single.append(float(line["rss"]))
x = range(len(rss_single))
plt.figure()
plt.subplot(122)
plt.plot(x, rss_single)
plt.title("rss/process w FSDP")
plt.xlabel("second")
plt.ylabel("rss GB")
plt.subplot(121)

csv_wo_fsdp = "./res/rss_per_process_wo_FSDP.csv"
f = open(csv_wo_fsdp, mode="r", encoding="utf-8", newline="")
csv_reader = csv.DictReader(
    f,
    fieldnames=[
        "rss",
        "vms",
        "data",
    ],
)
next(csv_reader)
line = next(csv_reader)
rss_2ddp = list()
for line in csv_reader:
    rss_2ddp.append(float(line["rss"]))
x = range(len(rss_2ddp))
plt.plot(x, rss_2ddp)
plt.title("rss/process wo FSDP")
plt.xlabel("second")
plt.ylabel("rss GB")
plt.savefig("rss.png")
