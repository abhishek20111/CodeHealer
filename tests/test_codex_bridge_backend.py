import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ["LLM_BACKEND"] = "codex-bridge"
os.environ["CODEX_BRIDGE_TOKEN"] = "unit-test-token"
os.environ["CODEX_BRIDGE_URL"] = "http://127.0.0.1:3200/v1"

import agent
from langchain_core.messages import AIMessage, HumanMessage


class CodexBridgeBackendTests(unittest.IsolatedAsyncioTestCase):
    def test_create_agent_selects_direct_codex_backend(self):
        self.assertEqual(agent.create_agent(), {"_backend": "codex-bridge"})

    def test_deterministic_routes_do_not_need_an_llm(self):
        self.assertEqual(agent.deterministic_route("search GitHub code"), "code")
        self.assertEqual(agent.deterministic_route("check the Slack channel"), "slack")
        self.assertEqual(agent.deterministic_route("query the employees table"), "database")
        self.assertEqual(agent.deterministic_route("inspect OBS logs"), "logs")
        self.assertIsNone(agent.deterministic_route("hello"))

    def test_bridge_messages_preserve_roles_and_selected_prompt(self):
        messages = agent._bridge_messages(
            "code",
            [HumanMessage(content="Earlier question"), AIMessage(content="Earlier answer")],
            "Current question",
        )
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant", "user"])
        self.assertIn("read-only code", messages[0]["content"])
        self.assertEqual(messages[-1]["content"], "Current question")

    async def test_streaming_yields_one_completed_codex_answer(self):
        bridge_agent = {"_backend": "codex-bridge"}
        with patch.object(agent, "_ask_codex_bridge", new=AsyncMock(return_value="Completed answer")):
            chunks = [chunk async for chunk in agent.stream_agent(bridge_agent, "hello", "session-1")]
        self.assertEqual(chunks, ["Completed answer"])


if __name__ == "__main__":
    unittest.main()
