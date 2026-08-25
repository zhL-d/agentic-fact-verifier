"""Shared test fixtures/helpers."""

import asyncio
import json

import pytest


class FakeMCPTool:
    """Stand-in for the langchain-mcp-adapters tool wrapper's
    `.ainvoke(args) -> [{"text": json_str}]` shape, without needing a real
    MCP server/session."""

    def __init__(self, fn):
        self.fn = fn
        self.calls: list[dict] = []

    async def ainvoke(self, args: dict):
        self.calls.append(args)
        result = self.fn(args)
        if asyncio.iscoroutine(result):
            result = await result
        return [{"text": json.dumps(result)}]


@pytest.fixture
def fake_tool():
    return FakeMCPTool
