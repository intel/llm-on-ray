from abc import ABC, abstractmethod
from run_model_serve import PredictDeployment
from typing import AsyncIterator, Dict, List, Optional
from fastapi import HTTPException, Request
from api_backend.common.models import ModelData, Prompt, QueuePriority
from api_backend.common.models import Prompt, ModelResponse
from api_backend.plugin.router_query_engine import StreamingErrorHandler
from enum import IntEnum

class RouterQueryClient():
    def __init__(self, serve_deployments: PredictDeployment, hooks=None):
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
            async_iterator=deploy_handle.options(stream=True, use_new_handle_api=True).\
                test_generate.options(stream=True, use_new_handle_api=True).remote(prompt, streaming_response=True),
        ):
            yield x

    async def model(self, model_id: str) -> ModelData:
        """Get configurations for a supported model"""
        return ModelData(
            id=model_id,
            object="model",
            owned_by="serve owner",
            permission=["serve permission"],
            aviary_metadata={
                "serve_metadata": "serve_metadata",
                "engine_config": {
                    "model_description": "serve_description",
                    "model_url": "serve_url",
                },
            },
        )

    async def models(self) -> Dict[str, ModelData]:
        """Get configurations for supported models"""
        metadatas = {}
        for model_id in self.serve_deployments:
            metadatas[model_id] = await self.model(model_id)
        return metadatas
