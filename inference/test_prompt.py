from pydantic import BaseModel, root_validator, validator
from typing import Any, Dict, List, Literal, Optional, Union

body = {
  "model": "gpt2",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me a long story with many words."}
  ],
  "temperature": 0.7,
  "stream": True,
}


class Message(BaseModel):
    role: Literal["system", "assistant", "user"]
    content: str
    def __str__(self):
        return self.content

class Prompt(BaseModel):
    prompt: Union[str, List[Message]]


prompt = Prompt(prompt=body["messages"])

print(prompt.prompt)
print(prompt.prompt[1])