import os
import json
import requests

s = requests.Session()

api_base = os.getenv("ENDPOINT_URL")
url = f"{api_base}/chat/completions"
body = {
  "model": "gpt2",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me a long story with many words."}
  ],
  "temperature": 0.7,
  "stream": True,
}

with s.post(url, json=body, stream=True) as response:
    for chunk in response.iter_content(decode_unicode=True):
        if chunk is not None:
            try:
                chunk = chunk.decode("utf-8")
                print(chunk, end="")
                # # Get data from reponse chunk
                # chunk_data = chunk.split("data: ")[1]

                # # Get message choices from data
                # choices = json.loads(chunk_data)["choices"]

                # # Pick content from first choice
                # content = choices[0]["delta"]["content"]

                # print(content, end="", flush=True)
            except json.decoder.JSONDecodeError:
                # Chunk was not formatted as expected
                pass
            except KeyError:
                # No message was contained in the chunk
                pass
    print("")