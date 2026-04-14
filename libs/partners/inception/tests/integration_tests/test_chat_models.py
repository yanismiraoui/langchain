"""Test ChatInception chat model."""

from __future__ import annotations

from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_inception.chat_models import ChatInception

MODEL_NAME = "mercury-2"


class TestChatInception(ChatModelIntegrationTests):
    """Test `ChatInception` chat model."""

    @property
    def chat_model_class(self) -> type[ChatInception]:
        """Return class of chat model being tested."""
        return ChatInception

    @property
    def chat_model_params(self) -> dict:
        """Parameters to create chat model instance for testing."""
        return {
            "model": MODEL_NAME,
            "temperature": 0,
        }
