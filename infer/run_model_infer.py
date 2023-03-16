import requests
prompt = "Once upon a time, there existed a little girl, who liked to have adventures." + \
             " She wanted to go to places and meet new people, and have fun."
sample_input = {"text": prompt}

output = requests.post("http://127.0.0.1:8000/", json=[sample_input]).text

print(output)
