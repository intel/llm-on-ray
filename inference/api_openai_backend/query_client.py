"""
Reference:
  - https://github.com/ray-project/ray-llm/blob/b3560aa55dadf6978f0de0a6f8f91002a5d2bed1/aviary/backend/server/plugins/router_query_engine.py
"""
from typing import Dict
from fastapi import HTTPException
from .openai_protocol import ModelCard, Prompt
from .openai_protocol import Prompt, ModelResponse
from .error_handler import handle_request

class RouterQueryClient():
    def __init__(self, serve_deployments):
        self.serve_deployments = serve_deployments

    async def query(self, model: str, prompt: Prompt, request_id: str):
        response_stream = self.stream(
            model,
            prompt,
            request_id,
        )
        responses = [resp async for resp in response_stream]
        return ModelResponse.merge_stream(*responses)

    async def stream(
        self, model: str, prompt: Prompt, request_id: str
    ):
        if model in self.serve_deployments:
            deploy_handle = self.serve_deployments[model]
        else:
            raise HTTPException(404, f"Could not find model with id {model}")

        async for x in handle_request(
            model=model,
            prompt=prompt,
            request_id=request_id,
            async_iterator=deploy_handle.options(stream=True).stream_response.options(stream=True, use_new_handle_api=True).remote(prompt)
        ):
            yield x

    async def model(self, model_id: str) -> ModelCard:
        """Get configurations for a supported model"""
        return ModelCard(
            id=model_id,
            root=model_id,
        )

    async def models(self) -> Dict[str, ModelCard]:
        """Get configurations for supported models"""
        metadatas = {}
        for model_id in self.serve_deployments:
            metadatas[model_id] = await self.model(model_id)
        return metadatas
