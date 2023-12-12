from typing import Dict, List, Set, Optional, TypeVar
from pydantic import BaseModel
from common.models import Message

ModelT = TypeVar("ModelT", bound=BaseModel)

class Completions(BaseModel):
    model: str
    prompt: str
    suffix: Optional[str] = None
    stream: bool = False
    echo: Optional[bool] = False
    user: Optional[str] = None
    max_tokens: Optional[int] = 16
    top_k: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: int = 1
    logprobs: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    best_of: int = 1

class ChatCompletions(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = False
    echo: Optional[bool] = False
    user: Optional[str] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: int = 1
    logprobs: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    best_of: int = 1
    _ignored_fields: Set[str] = set()
