"""Slack integration package — Web API client and LangChain tools."""

from .slack_client import (
    SlackClient,
    search_slack as search_slack_impl,
    list_slack_channels as list_slack_channels_impl
)
from .tools import create_slack_tools
from .workflow_tools import create_workflow_tools, WORKFLOW_TOOLS
from .enhanced_workflow_tools import create_enhanced_workflow_tools, ENHANCED_WORKFLOW_TOOLS

__all__ = [
    "SlackClient",
    "search_slack_impl",
    "list_slack_channels_impl",
    "create_slack_tools",
    "create_workflow_tools",
    "WORKFLOW_TOOLS",
    "create_enhanced_workflow_tools",
    "ENHANCED_WORKFLOW_TOOLS",
]
