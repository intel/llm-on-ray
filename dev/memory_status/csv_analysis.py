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
