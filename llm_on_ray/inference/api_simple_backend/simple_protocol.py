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
#
# ===========================================================================
#
# This file is adapted from
# https://github.com/ray-project/ray-llm/blob/b3560aa55dadf6978f0de0a6f8f91002a5d2bed1/aviary/common/models.py
# Copyright 2023 Anyscale
#
# https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/protocol/openai_api_protocol.py
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

from typing import Dict, Optional, Union, Iterator
import requests
from pydantic import BaseModel


class SimpleRequest(BaseModel):
    text: str
    config: Dict[str, Union[int, float]]
    stream: Optional[bool] = False


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
