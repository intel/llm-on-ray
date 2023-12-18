from typing import Dict
from fastapi import HTTPException, Request
from .openai_protocol import ModelCard, Prompt, QueuePriority
from .openai_protocol import Prompt, ModelResponse
from .error_handler.streaming_error_handler import StreamingErrorHandler

class RouterQueryClient():
    def __init__(self, serve_deployments, hooks=None):
        self.serve_deployments = serve_deployments
        self.error_hook = StreamingErrorHandler(hooks=hooks)

    async def query(self, model: str, prompt: Prompt, request: Request):
        response_stream = self.stream(
            model,
            prompt,
            request,
        )
        responses = [resp async for resp in response_stream]
        return ModelResponse.merge_stream(*responses)

    async def stream(
        self, model: str, prompt: Prompt, request, priority=None
    ):
        if model in self.serve_deployments:
            deploy_handle = self.serve_deployments[model]
        else:
            raise HTTPException(404, f"Could not find model with id {model}")

        async for x in self.error_hook.handle_failure(
            model=model,
            request=request,
            prompt=prompt,
            async_iterator=deploy_handle.options(stream=True).stream_response.options(stream=True, use_new_handle_api=True).remote(prompt)
        ):
            yield x

    async def model(self, model_id: str) -> ModelCard:
        """Get configurations for a supported model"""
        return ModelCard(
            id=model_id,
            root=model_id,
            permission=["serve permission"],
        )

    async def models(self) -> Dict[str, ModelCard]:
        """Get configurations for supported models"""
        metadatas = {}
        for model_id in self.serve_deployments:
            metadatas[model_id] = await self.model(model_id)
        return metadatas
