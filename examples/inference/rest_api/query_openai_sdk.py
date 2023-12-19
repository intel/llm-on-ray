import openai
import os

# List all models.
models = openai.Model.list()
print(models)

# Note: not all arguments are currently supported and will be ignored by the backend.
model_name = os.getenv("MODEL_TO_SERVE", "gpt2")
chat_completion = openai.ChatCompletion.create(
    model=model_name,
    messages=[
      {"role": "assistant", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Tell me a long story with many words."}
    ],
    temperature=0.7,
    stream=False,
)
print(chat_completion)