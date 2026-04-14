"""Test chat model integration."""

from __future__ import annotations

from langchain_tests.unit_tests import ChatModelUnitTests
from pydantic import SecretStr

from langchain_inception.chat_models import DEFAULT_API_BASE, ChatInception

MODEL_NAME = "mercury-2"


class TestChatInceptionUnit(ChatModelUnitTests):
    """Standard unit tests for `ChatInception` chat model."""

    @property
    def chat_model_class(self) -> type[ChatInception]:
        """Chat model class being tested."""
        return ChatInception

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        """Parameters to initialize from environment variables."""
        return (
            {
                "INCEPTION_API_KEY": "api_key",
                "INCEPTION_API_BASE": "api_base",
            },
            {
                "model": MODEL_NAME,
            },
            {
                "api_key": "api_key",
                "api_base": "api_base",
            },
        )

    @property
    def chat_model_params(self) -> dict:
        """Parameters to create chat model instance for testing."""
        return {
            "model": MODEL_NAME,
            "api_key": "api_key",
        }

    def get_chat_model(self) -> ChatInception:
        """Get a chat model instance for testing."""
        return ChatInception(**self.chat_model_params)


class TestChatInceptionCustomUnit:
    """Custom tests specific to Inception chat model."""

    def test_base_url_alias(self) -> None:
        """Test that `base_url` is accepted as an alias for `api_base`."""
        chat_model = ChatInception(
            model=MODEL_NAME,
            api_key=SecretStr("api_key"),
            base_url="http://example.test/v1",
        )
        assert chat_model.api_base == "http://example.test/v1"

    def test_default_temperature(self) -> None:
        """Test that default temperature is 0.75."""
        chat_model = ChatInception(
            model=MODEL_NAME,
            api_key=SecretStr("api_key"),
        )
        assert chat_model.temperature == 0.75

    def test_default_max_tokens(self) -> None:
        """Test that default max_tokens is 8192."""
        chat_model = ChatInception(
            model=MODEL_NAME,
            api_key=SecretStr("api_key"),
        )
        assert chat_model.max_tokens == 8192

    def test_default_api_base(self) -> None:
        """Test that default api_base points to Inception API."""
        chat_model = ChatInception(
            model=MODEL_NAME,
            api_key=SecretStr("api_key"),
        )
        assert chat_model.api_base == DEFAULT_API_BASE

    def test_reasoning_effort_in_payload(self) -> None:
        """Test that reasoning_effort is included in the request payload."""
        chat_model = ChatInception(
            model=MODEL_NAME,
            api_key=SecretStr("api_key"),
            reasoning_effort="medium",
        )
        payload = chat_model._get_request_payload(
            [("user", "Hello")],
        )
        assert payload["reasoning_effort"] == "medium"

    def test_reasoning_effort_not_in_payload_when_none(self) -> None:
        """Test that reasoning_effort is omitted from payload when not set."""
        chat_model = ChatInception(
            model=MODEL_NAME,
            api_key=SecretStr("api_key"),
        )
        payload = chat_model._get_request_payload(
            [("user", "Hello")],
        )
        assert "reasoning_effort" not in payload


def test_profile() -> None:
    """Test that model profile is loaded correctly."""
    model = ChatInception(model="mercury-2", api_key=SecretStr("test_key"))
    assert model.profile is not None
