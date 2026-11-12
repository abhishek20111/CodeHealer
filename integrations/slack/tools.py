"""
Slack Tools — LangChain tools for Slack integration

These tools allow the AI agent to search Slack messages, list channels,
and interact with Slack workspace via the Web API.
"""

from langchain_core.tools import tool
from typing import Optional
from .slack_client import (
    SlackClient, 
    search_slack as _search_slack_impl,
    list_slack_channels as _list_slack_channels_impl
)


@tool
def search_slack_messages(
    query: str,
    count: int = 20,
    channel: Optional[str] = None,
    from_user: Optional[str] = None
) -> str:
    """
    Search Slack messages across channels.
    
    Args:
        query: Search query (keywords, phrases, or Slack search syntax)
        count: Number of results to return (default 20, max 100)
        channel: Optional channel name to narrow search (e.g., "payments-eng" or "#payments-eng")
        from_user: Optional user handle to narrow search (e.g., "alice" or "@alice")
    
    Returns:
        Formatted search results with channel, user, text, and permalink
    
    Examples:
        search_slack_messages("incident post-mortem")
        search_slack_messages("database migration", channel="platform-eng")
        search_slack_messages("deploy failed", from_user="alice")
    """
    return _search_slack_impl(query, count, channel, from_user)


@tool
def list_slack_channels(limit: int = 50) -> str:
    """
    List Slack channels in the workspace.
    
    Args:
        limit: Maximum number of channels to return (default 50, max 1000)
    
    Returns:
        Formatted list of channels with name, ID, privacy, member count, and topic
    
    Examples:
        list_slack_channels()
        list_slack_channels(limit=100)
    """
    result = _list_slack_channels_impl(limit)
    if "not configured" in result or "No channels" in result:
        return "❌ Slack authentication failed or token expired. Update SLACK_TOKEN in .env with a bot token (xoxb-...) to enable Slack integration."
    return result


@tool
def get_slack_channel_history(
    channel: str,
    limit: int = 10
) -> str:
    """
    Get recent messages from a specific Slack channel.
    
    Args:
        channel: Channel ID (e.g., "C01234567") or name (e.g., "app-tech-debt")
        limit: Number of recent messages to retrieve (default 10, max 100)
    
    Returns:
        Formatted channel history with timestamps, users, and message text
    
    Examples:
        get_slack_channel_history("C01234567")
        get_slack_channel_history("app-tech-debt", limit=5)
    """
    client = SlackClient()
    result = client.get_channel_history(channel, limit)
    
    if not result.get("ok"):
        error = result.get("error", "unknown")
        if error == "invalid_auth":
            return "❌ Slack authentication failed. The token may have expired. Update SLACK_TOKEN in .env (bot token xoxb-... recommended)."
        if error == "channel_not_found":
            return f"❌ Channel '{channel}' not found. Check the channel name or use a channel ID (starts with 'C')."
        return f"❌ Slack API error: {error}"
    
    messages = result.get("messages", [])
    if not messages:
        return f"No messages found in channel {channel}."
    
    output = [f"Recent {len(messages)} messages from #{channel}:"]
    output.append("")
    
    for msg in messages:
        user = msg.get("user", "unknown")
        text = msg.get("text", "")[:200]
        ts = msg.get("ts", "")
        
        output.append(f"@{user} at {ts}:")
        output.append(f"  {text}")
        output.append("")
    
    return "\n".join(output)


@tool
def post_slack_message(
    channel: str,
    text: str,
    thread_ts: Optional[str] = None
) -> str:
    """
    Post a message to a Slack channel.
    
    Args:
        channel: Channel ID (e.g., "C01234567") or name (e.g., "#general")
        text: Message text to post
        thread_ts: Optional parent message timestamp for threading
    
    Returns:
        Confirmation message with posted timestamp
    
    Examples:
        post_slack_message("C01234567", "Deployment complete ✅")
        post_slack_message("#platform-eng", "New feature released!")
        post_slack_message("C01234567", "Follow-up comment", thread_ts="1234567890.123456")
    
    Note: Requires SLACK_TOKEN with chat:write scope
    """
    client = SlackClient()
    result = client.post_message(channel, text, thread_ts)
    
    if not result.get("ok"):
        error = result.get("error", "unknown")
        return f"❌ Failed to post message: {error}"
    
    ts = result.get("ts", "")
    channel_posted = result.get("channel", channel)
    
    return f"✅ Message posted to {channel_posted} at {ts}"


@tool
def get_slack_user_info(user_id: str) -> str:
    """
    Get information about a Slack user.
    
    Args:
        user_id: Slack user ID (e.g., "U01234567")
    
    Returns:
        User information including name, real name, title, and status
    
    Examples:
        get_slack_user_info("U01234567")
    """
    client = SlackClient()
    user = client.get_user_info(user_id)
    
    if not user.get("ok", True):
        return f"❌ Failed to get user info: {user.get('error')}"
    
    name = user.get("name", "unknown")
    real_name = user.get("real_name", "")
    title = user.get("profile", {}).get("title", "")
    email = user.get("profile", {}).get("email", "")
    
    output = [f"User: {name}"]
    if real_name:
        output.append(f"Real name: {real_name}")
    if title:
        output.append(f"Title: {title}")
    if email:
        output.append(f"Email: {email}")
    
    return "\n".join(output)


def create_slack_tools() -> list:
    """
    Create and return all Slack tools.
    
    Returns:
        List of LangChain tools for Slack operations
    """
    return [
        search_slack_messages,
        list_slack_channels,
        get_slack_channel_history,
        post_slack_message,
        get_slack_user_info,
    ]
