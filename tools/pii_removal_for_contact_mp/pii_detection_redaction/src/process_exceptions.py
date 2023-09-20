import json
import glob

data_dir = "/home/user/local/processed/refinedweb/exceptions/"
filename = "2af2834dff874b4affffffffffffffffffffffff13000000.json"

with open(data_dir+filename, 'r') as f:
    data = json.load(f)

print(len(data))