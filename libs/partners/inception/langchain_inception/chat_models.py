"""Inception chat models."""

from __future__ import annotations

from typing import Any, cast

import openai
from langchain_core.language_models import (
    LangSmithParams,
    LanguageModelInput,
    ModelProfile,
    ModelProfileRegistry,
)
from langchain_core.utils import from_env, secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_inception.data._profiles import _PROFILES

DEFAULT_API_BASE = "https://api.inceptionlabs.ai/v1"

_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


class ChatInception(BaseChatOpenAI):
    """Inception chat model integration to access models hosted on Inception's API.

    Setup:
        Install `langchain-inception` and set environment variable `INCEPTION_API_KEY`.

        ```bash
        pip install -U langchain-inception
        export INCEPTION_API_KEY="your-api-key"
        ```

    Key init args — completion params:
        model:
            Name of Inception model to use, e.g. `'mercury-2'`.
        temperature:
            Sampling temperature. Defaults to `0.75`.
        max_tokens:
            Max number of tokens to generate. Defaults to `8192`.
        reasoning_effort:
            Controls thinking depth. One of `'instant'`, `'low'`, `'medium'`, `'high'`.

    Key init args — client params:
        timeout:
            Timeout for requests.
        max_retries:
            Max number of retries.
        api_key:
            Inception API key. If not passed in will be read from env var
            `INCEPTION_API_KEY`.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        ```python
        from langchain_inception import ChatInception

        model = ChatInception(
            model="mercury-2",
            temperature=0.75,
            max_tokens=8192,
            # api_key="...",
            # other params...
        )
        ```

    Invoke:
        ```python
        messages = [
            (
                "system",
                "You are a helpful translator. Translate the user sentence to French.",
            ),
            ("human", "I love programming."),
        ]
        model.invoke(messages)
        ```

    Stream:
        ```python
        for chunk in model.stream(messages):
            print(chunk.text, end="")
        ```
        ```python
        stream = model.stream(messages)
        full = next(stream)
        for chunk in stream:
            full += chunk
        full
        ```

    Async:
        ```python
        await model.ainvoke(messages)

        # stream:
        # async for chunk in (await model.astream(messages))

        # batch:
        # await model.abatch([messages])
        ```

    Tool calling:
        ```python
        from pydantic import BaseModel, Field


        class GetWeather(BaseModel):
            '''Get the current weather in a given location'''

            location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


        class GetPopulation(BaseModel):
            '''Get the current population in a given location'''

            location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


        model_with_tools = model.bind_tools([GetWeather, GetPopulation])
        ai_msg = model_with_tools.invoke("Which city is hotter today and which is bigger: LA or NY?")
        ai_msg.tool_calls
        ```

        See `ChatInception.bind_tools()` method for more.

    Structured output:
        ```python
        from typing import Optional

        from pydantic import BaseModel, Field


        class Joke(BaseModel):
            '''Joke to tell user.'''

            setup: str = Field(description="The setup of the joke")
            punchline: str = Field(description="The punchline to the joke")
            rating: int | None = Field(description="How funny the joke is, from 1 to 10")


        structured_model = model.with_structured_output(Joke)
        structured_model.invoke("Tell me a joke about cats")
        ```

        See `ChatInception.with_structured_output()` for more.

    Token usage:
        ```python
        ai_msg = model.invoke(messages)
        ai_msg.usage_metadata
        ```
        ```python
        {"input_tokens": 28, "output_tokens": 5, "total_tokens": 33}
        ```

    Response metadata:
        ```python
        ai_msg = model.invoke(messages)
        ai_msg.response_metadata
        ```
    """  # noqa: E501

    model_name: str = Field(alias="model")
    """The name of the model."""
    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("INCEPTION_API_KEY", default=None),
    )
    """Inception API key."""
    api_base: str = Field(
        alias="base_url",
        default_factory=from_env("INCEPTION_API_BASE", default=DEFAULT_API_BASE),
    )
    """Inception API base URL.

    Automatically read from env variable `INCEPTION_API_BASE` if not provided.
    """
    temperature: float | None = 0.75
    """Sampling temperature. Defaults to `0.75` per Inception's recommendation."""
    max_tokens: int | None = 8192
    """Max number of tokens to generate. Defaults to `8192`."""
    reasoning_effort: str | None = None
    """Controls thinking depth. One of `'low'`, `'instant'`, `'medium'`, `'high'`."""

    model_config = ConfigDict(populate_by_name=True)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-inception"

    @property
    def lc_secrets(self) -> dict[str, str]:
        """A map of constructor argument names to secret ids."""
        return {"api_key": "INCEPTION_API_KEY"}

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        ls_params = super()._get_ls_params(stop=stop, **kwargs)
        ls_params["ls_provider"] = "inception"
        return ls_params

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate necessary environment vars and client params."""
        if self.api_base == DEFAULT_API_BASE and not (
            self.api_key and self.api_key.get_secret_value()
        ):
            msg = "If using default api base, INCEPTION_API_KEY must be set."
            raise ValueError(msg)
        client_params: dict = {
            k: v
            for k, v in {
                "api_key": self.api_key.get_secret_value() if self.api_key else None,
                "base_url": self.api_base,
                "timeout": self.request_timeout,
                "max_retries": self.max_retries,
                "default_headers": self.default_headers,
                "default_query": self.default_query,
            }.items()
            if v is not None
        }

        if not (self.client or None):
            sync_specific: dict = {"http_client": self.http_client}
            self.root_client = openai.OpenAI(**client_params, **sync_specific)
            self.client = self.root_client.chat.completions
        if not (self.async_client or None):
            async_specific: dict = {"http_client": self.http_async_client}
            self.root_async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,
            )
            self.async_client = self.root_async_client.chat.completions
        return self

    def _resolve_model_profile(self) -> ModelProfile | None:
        return _get_default_model_profile(self.model_name) or None

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        return payload
