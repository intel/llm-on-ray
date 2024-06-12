from openai import OpenAI

# Note: Ray Serve doesn't support all OpenAI client arguments and may ignore some.
client = OpenAI(
        # Replace the URL if deploying your app remotely
        # (e.g., on Anyscale or KubeRay).
        base_url="http://localhost:8000/v1",
        api_key="NOT A REAL KEY",
)
chat_completion = client.chat.completions.create(
        model="meta-llama/Llama-2-7b-chat-hf",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "What are some highly rated restaurants in San Francisco?'",
            },
        ],
        temperature=0.01,
        stream=True,
)

for chat in chat_completion:
    if chat.choices[0].delta.content is not None:
        print(chat.choices[0].delta.content, end="")
