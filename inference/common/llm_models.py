from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union, Literal
from pydantic import BaseModel, Extra, PrivateAttr, root_validator, validator
import yaml
from util.dict_utils import merge_dicts

ModelT = TypeVar("ModelT", bound=BaseModel)

class VLLMCompletions(BaseModel):
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

    @classmethod
    def parse_yaml(cls: Type[ModelT], file, **kwargs) -> ModelT:
        kwargs.setdefault("Loader", yaml.SafeLoader)
        dict_args = yaml.load(file, **kwargs)
        return cls.parse_obj(dict_args)

    def yaml(
        self,
        *,
        stream=None,
        include=None,
        exclude=None,
        by_alias: bool = False,
        skip_defaults: Union[bool, None] = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        **kwargs,
    ):
        """
        Generate a YAML representation of the model, `include` and `exclude` arguments as per `dict()`.
        """
        return yaml.dump(
            self.dict(
                include=include,
                exclude=exclude,
                by_alias=by_alias,
                skip_defaults=skip_defaults,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
            ),
            stream=stream,
            **kwargs,
        )

    def dict(self, **kwargs):
        if kwargs.get("exclude", None) is None:
            kwargs["exclude"] = self._ignored_fields
        return super().dict(**kwargs)

    # @validator("stop", always=True)
    # def validate_stopping_sequences(cls, values):
    #     if not values:
    #         return values

    #     unique_val = sorted(list(set(values)))

    #     if len(unique_val) > MAX_NUM_STOPPING_SEQUENCES:
    #         TooManyStoppingSequences(
    #             len(unique_val), MAX_NUM_STOPPING_SEQUENCES
    #         ).raise_exception()

    #     return unique_val

    # @classmethod
    # def merge_generation_params(
    #     cls: Type[ModelT], prompt: Prompt, generation: GenerationConfig
    # ) -> ModelT:
    #     # Extract parameters object from prompt
    #     parameters = prompt.parameters or {}
    #     if not isinstance(parameters, dict):
    #         parameters = parameters.dict()

    #     # Merge in the generate kwargs
    #     generate_kwargs = merge_dicts(
    #         parameters,
    #         generation.generate_kwargs,
    #     )

    #     # The stoppping sequence needs to be merged manually
    #     generate_kwargs["stop"] = (parameters.get("stop") or []) + (
    #         generation.stopping_sequences or []
    #     )

    #     return cls.parse_obj(generate_kwargs)


class Message(BaseModel):
    role: Literal["system", "assistant", "user"]
    content: str

    def __str__(self):
        return self.content


class VLLMChatCompletions(BaseModel):
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

    @classmethod
    def parse_yaml(cls: Type[ModelT], file, **kwargs) -> ModelT:
        kwargs.setdefault("Loader", yaml.SafeLoader)
        dict_args = yaml.load(file, **kwargs)
        return cls.parse_obj(dict_args)

    def yaml(
        self,
        *,
        stream=None,
        include=None,
        exclude=None,
        by_alias: bool = False,
        skip_defaults: Union[bool, None] = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        **kwargs,
    ):
        """
        Generate a YAML representation of the model, `include` and `exclude` arguments as per `dict()`.
        """
        return yaml.dump(
            self.dict(
                include=include,
                exclude=exclude,
                by_alias=by_alias,
                skip_defaults=skip_defaults,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
            ),
            stream=stream,
            **kwargs,
        )

    def dict(self, **kwargs):
        if kwargs.get("exclude", None) is None:
            kwargs["exclude"] = self._ignored_fields
        return super().dict(**kwargs)