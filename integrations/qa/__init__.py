"""QA sub-agent tools that connect the supervisor to the local automation tools.

Frontend Automation Tool (UI Comparator & Recorder) — Playwright dashboard on http://localhost:3737
Backend Automation Tool (DataVeriTest / api-test-tool) — Spring Boot on http://localhost:9090

Both tools must be running locally for these tools to work.
"""

from .tools import create_qa_tools, QA_TOOLS

__all__ = ["create_qa_tools", "QA_TOOLS"]
