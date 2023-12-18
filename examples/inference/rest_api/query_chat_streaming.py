import os
import json
import requests

s = requests.Session()

api_base = os.getenv("ENDPOINT_URL")
if api_base is None:
    api_base = "http://localhost:8000/custom_model/v1"
url = f"{api_base}/chat/completions"

model_name = os.getenv("MODEL_TO_SERVE", "gpt2")
body = {
  "model": model_name,
  "messages": [
    {"role": "assistant", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me a long story with many words."}
  ],
  "temperature": 0.7,
  "stream": True,
}

with s.post(url, json=body) as response:
    for chunk in response.iter_lines(decode_unicode=True):
        if chunk is not None:
            try:
                # Get data from reponse chunk
                chunk_data = chunk.split("data: ")[1]

                # Get message choices from data
                choices = json.loads(chunk_data)["choices"]

                # Pick content from first choice
                content = choices[0]["delta"]["content"]

                print(content, end="", flush=True)
            except json.decoder.JSONDecodeError:
                # Chunk was not formatted as expected
                pass
            except KeyError:
                # No message was contained in the chunk
                pass
            except:
                pass
    print("")