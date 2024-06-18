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
from typing import Dict, Optional, Union, Iterator, List
import requests
from pydantic import BaseModel, ValidationError, validator


class SimpleRequest(BaseModel):
    text: str
    config: Dict[str, Union[int, float]] = {}
    stream: Optional[bool] = False

    @validator("text")
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Empty prompt is not supported.")
        return v

    @validator("config", pre=True)
    def check_config_type(cls, value):
        allowed_keys = ["max_new_tokens", "temperature", "top_p", "top_k"]
        allowed_set = set(allowed_keys)
        config_dict = value.keys()
        config_keys = [key for key in config_dict]
        config_set = set(config_keys)

        if not isinstance(value, dict):
            raise ValueError("Config must be a dictionary")

        if not all(isinstance(key, str) for key in value.keys()):
            raise ValueError("All keys in config must be strings")

        if not all(isinstance(val, (int, float)) for val in value.values()):
            raise ValueError("All values in config must be integers or floats")

        if not config_set.issubset(allowed_set):
            invalid_keys = config_set - allowed_set
            raise ValueError(f'Invalid config keys: {", ".join(invalid_keys)}')

        return value

    @validator("stream", pre=True)
    def check_stream_type(cls, value):
        if not isinstance(value, bool) and value is not None:
            raise ValueError("Stream must be a boolean or None")
        return value


class SimpleModelResponse(BaseModel):
    headers: Dict[str, str]
    text: str
    content: bytes
    status_code: int
    url: str

    class Config:
        arbitrary_types_allowed = True

    response: Optional[requests.Response] = None

    @staticmethod
    def from_requests_response(response: requests.Response):
        return SimpleModelResponse(
            headers=dict(response.headers),
            text=response.text,
            content=response.content,
            status_code=response.status_code,
            url=response.url,
            response=response,
        )

    def iter_content(
        self, chunk_size: Optional[int] = 1, decode_unicode: bool = False
    ) -> Iterator[Union[bytes, str]]:
        if self.response is not None:
            return self.response.iter_content(chunk_size=chunk_size, decode_unicode=decode_unicode)
        else:
            return iter([])
