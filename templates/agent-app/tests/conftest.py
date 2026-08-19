"""Shared test fixtures.

The key piece is FakeToolModel. langchain-core's GenericFakeChatModel does
not implement bind_tools(), so create_agent() raises NotImplementedError the
moment your agent has any tools. Subclassing it with a no-op bind_tools is
enough to test everything you own - wiring, thread isolation, error handling -
with no API key and no cost.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models import GenericFakeChatModel


class FakeToolModel(GenericFakeChatModel):
    """A fake chat model that can be used with agents that have tools.

    It never actually calls a tool - it replays the canned replies you give
    it. Use it to test your plumbing, not the model's judgment.
    """

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
        return self


@pytest.fixture
def fake_model():
    """Return a factory: fake_model("reply 1", "reply 2", ...)."""

    def _make(*replies: str) -> FakeToolModel:
        return FakeToolModel(messages=iter(list(replies)))

    return _make
