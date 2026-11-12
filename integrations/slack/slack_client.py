"""
Slack Client — Production Slack API client using curl (similar to OBS approach)

Uses curl --noproxy to bypass local proxy issues, following the same pattern
as OBS integration (ADR-0004: curl-fallback-for-tls-proxy-constrained-calls).
"""

import os
import json
import subprocess
from datetime import datetime
from typing import Optional, Dict, Any, List

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

SLACK_TOKEN = os.getenv("SLACK_TOKEN", "").strip().strip('"').strip("'")
SLACK_COOKIE_D = os.getenv("SLACK_COOKIE_D", "").strip().strip('"').strip("'")
SLACK_WORKSPACE_URL = os.getenv("SLACK_WORKSPACE_URL", "https://slack.com").strip().strip('"').strip("'")

def _log(msg: str):
    """Log with timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [SLACK] {msg}")


class SlackClient:
    """Production Slack client using Slack Web API via curl."""
    
    def __init__(self, token: Optional[str] = None, cookie_d: Optional[str] = None):
        self.token = token or SLACK_TOKEN
        self.cookie_d = cookie_d or SLACK_COOKIE_D
        # Set SLACK_WORKSPACE_URL to a workspace or enterprise Slack URL if slack.com is blocked
        self.api_base = f"{SLACK_WORKSPACE_URL}/api"
        
        if not self.token:
            _log("⚠️  SLACK_TOKEN not configured")
        elif self.token.startswith("xoxc"):
            # Browser token (not recommended, requires cookie)
            _log("⚠️  Using browser token (xoxc) - consider using bot token (xoxb) instead")
            if not self.cookie_d:
                _log("⚠️  SLACK_COOKIE_D not configured (required for xoxc tokens)")
        elif self.token.startswith("xoxb"):
            _log("✅ Using bot token (xoxb)")
        elif self.token.startswith("xoxp"):
            _log("✅ Using user token (xoxp)")
    
    def _call_api(
        self, 
        method: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Call Slack Web API using curl with form-encoded data (matches docs-mcp).
        
        Browser tokens (xoxc) require BOTH token AND cookie_d.
        
        Args:
            method: Slack API method (e.g., 'conversations.list', 'search.messages')
            params: API parameters as dict
        
        Returns:
            Dict with API response
        """
        if not self.token:
            return {"ok": False, "error": "not_configured", "message": "SLACK_TOKEN not set"}
        
        url = f"{self.api_base}/{method}"
        
        # Build curl command
        cmd = [
            "curl",
            "--noproxy", "*",  # Bypass local proxy
            "-X", "POST",
            "-H", f"Authorization: Bearer {self.token}",
            "--silent",
            "--show-error",
        ]
        # TLS verification stays on unless SLACK_INSECURE_SSL=true (self-signed internal certs only).
        if os.getenv("SLACK_INSECURE_SSL", "false").lower() == "true":
            cmd.insert(3, "--insecure")
        
        # Add cookie header ONLY for browser tokens (xoxc)
        # Bot tokens (xoxb) and user tokens (xoxp) don't need cookies
        if self.cookie_d and self.token and self.token.startswith("xoxc"):
            cmd.extend(["-H", f"Cookie: d={self.cookie_d}"])
        
        # Add parameters as form-encoded data (not JSON!)
        # This matches the docs-mcp implementation
        if params:
            for key, value in params.items():
                if value is not None:
                    cmd.extend(["--data-urlencode", f"{key}={value}"])
        
        cmd.append(url)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                _log(f"❌ curl failed (exit {result.returncode}): {result.stderr[:200]}")
                return {
                    "ok": False,
                    "error": "curl_failed",
                    "message": f"curl exit code {result.returncode}"
                }
            
            # Parse JSON response
            try:
                response = json.loads(result.stdout)
                if not response.get("ok"):
                    error = response.get("error", "unknown")
                    _log(f"❌ Slack API error: {error}")
                return response
            except json.JSONDecodeError as e:
                _log(f"❌ Invalid JSON response: {str(e)}")
                return {
                    "ok": False,
                    "error": "invalid_json",
                    "message": f"Response: {result.stdout[:200]}"
                }
        
        except subprocess.TimeoutExpired:
            _log(f"❌ Request timeout after 30s")
            return {"ok": False, "error": "timeout", "message": "Request timeout"}
        
        except Exception as e:
            _log(f"❌ Request failed: {str(e)}")
            return {"ok": False, "error": "request_failed", "message": str(e)}
    
    def list_channels(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List Slack channels (conversations).
        
        Args:
            limit: Maximum number of channels to return
        
        Returns:
            List of channel objects with id, name, is_private, etc.
        """
        result = self._call_api("conversations.list", {
            "limit": limit,
            "types": "public_channel,private_channel",
            "exclude_archived": True
        })
        
        if not result.get("ok"):
            _log(f"❌ Failed to list channels: {result.get('error')}")
            return result  # Return the full error dict so callers can check
        
        channels = result.get("channels", [])
        _log(f"✅ Retrieved {len(channels)} channels")
        return channels

    def resolve_channel_id(self, channel_name: str) -> Optional[str]:
        """Resolve a channel name to its ID by listing channels."""
        channel_name = channel_name.lstrip('#')
        channels = self.list_channels(limit=500)
        if isinstance(channels, dict):  # error dict
            return None
        for ch in channels:
            if ch.get("name") == channel_name:
                return ch["id"]
        return None
    
    def search_messages(
        self,
        query: str,
        count: int = 20,
        channel: Optional[str] = None,
        from_user: Optional[str] = None,
        sort: str = "score"
    ) -> Dict[str, Any]:
        """
        Search Slack messages.
        
        Args:
            query: Search query (can include Slack search syntax)
            count: Number of results (max 100)
            channel: Optional channel name to narrow search (e.g., "payments-eng")
            from_user: Optional user handle to narrow search (e.g., "@alice")
            sort: Sort mode ("score" or "timestamp")
        
        Returns:
            Dict with matches, query, and pagination info
        """
        # Build query with filters
        search_query = query
        if channel:
            channel_name = channel.lstrip('#')
            search_query += f" in:#{channel_name}"
        if from_user:
            user_handle = from_user.lstrip('@')
            search_query += f" from:@{user_handle}"
        
        _log(f"🔍 Searching messages: '{search_query}' (count={count})")
        
        result = self._call_api("search.messages", {
            "query": search_query,
            "count": min(count, 100),
            "sort": sort
        })
        
        if not result.get("ok"):
            _log(f"❌ Search failed: {result.get('error')}")
            return {
                "ok": False,
                "error": result.get("error"),
                "query": search_query,
                "matches": []
            }
        
        messages = result.get("messages", {})
        matches = messages.get("matches", [])
        _log(f"✅ Found {len(matches)} messages")
        
        return {
            "ok": True,
            "query": search_query,
            "matches": matches,
            "total": messages.get("total", 0),
            "pagination": messages.get("pagination", {})
        }
    
    def get_channel_history(
        self,
        channel: str,
        limit: int = 10,
        oldest: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get recent messages from a channel.
        
        Args:
            channel: Slack channel ID (e.g., "C01234567") or name (e.g., "general")
            limit: Number of messages to retrieve
            oldest: Oldest message timestamp (optional)
        
        Returns:
            Dict with ok, messages, and optional error
        """
        # Resolve name to ID if needed
        channel_id = channel
        if not channel.startswith("C"):
            resolved = self.resolve_channel_id(channel)
            if resolved:
                channel_id = resolved
                _log(f"Resolved #{channel} → {channel_id}")
            else:
                _log(f"❌ Could not resolve channel name: #{channel}")
                return {"ok": False, "error": "channel_not_found", "message": f"Could not resolve #{channel}"}

        params = {
            "channel": channel_id,
            "limit": limit
        }
        if oldest:
            params["oldest"] = oldest
        
        result = self._call_api("conversations.history", params)
        
        if not result.get("ok"):
            _log(f"❌ Failed to get history: {result.get('error')}")
            return result
        
        messages = result.get("messages", [])
        _log(f"✅ Retrieved {len(messages)} messages from {channel_id}")
        return {"ok": True, "messages": messages}
    
    def post_message(
        self,
        channel: str,
        text: str,
        thread_ts: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Post a message to a Slack channel.
        
        Args:
            channel: Channel ID or name
            text: Message text
            thread_ts: Optional parent message timestamp for threading
        
        Returns:
            Dict with posting result
        """
        params = {
            "channel": channel,
            "text": text
        }
        if thread_ts:
            params["thread_ts"] = thread_ts
        
        _log(f"📤 Posting message to {channel}")
        
        result = self._call_api("chat.postMessage", params)
        
        if not result.get("ok"):
            _log(f"❌ Failed to post message: {result.get('error')}")
        else:
            _log(f"✅ Message posted: {result.get('ts')}")
        
        return result
    
    def get_user_info(self, user_id: str) -> Dict[str, Any]:
        """
        Get information about a user.
        
        Args:
            user_id: Slack user ID (e.g., "U01234567")
        
        Returns:
            Dict with user information
        """
        result = self._call_api("users.info", {"user": user_id})
        
        if not result.get("ok"):
            _log(f"❌ Failed to get user info: {result.get('error')}")
            return {"ok": False, "error": result.get("error")}
        
        return result.get("user", {})


# ══════════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ══════════════════════════════════════════════════════════════════════════════

def search_slack(
    query: str,
    count: int = 20,
    channel: Optional[str] = None,
    from_user: Optional[str] = None
) -> str:
    """
    Search Slack messages and return formatted results.
    
    Args:
        query: Search query
        count: Number of results
        channel: Optional channel filter
        from_user: Optional user filter
    
    Returns:
        Formatted search results as string
    """
    client = SlackClient()
    result = client.search_messages(query, count, channel, from_user)
    
    if not result.get("ok"):
        return f"❌ Error: {result.get('error')}"
    
    matches = result.get("matches", [])
    if not matches:
        return f"No results found for: {result.get('query')}"
    
    output = [f"Found {len(matches)} messages (total: {result.get('total', 0)}):"]
    output.append("")
    
    for i, match in enumerate(matches, 1):
        channel_name = match.get("channel", {}).get("name", "unknown")
        username = match.get("username", "unknown")
        text = match.get("text", "")[:200]
        permalink = match.get("permalink", "")
        timestamp = match.get("ts", "")
        
        output.append(f"{i}. #{channel_name} @{username}")
        output.append(f"   {text}")
        if permalink:
            output.append(f"   {permalink}")
        output.append("")
    
    return "\n".join(output)


def list_slack_channels(limit: int = 50) -> str:
    """
    List Slack channels and return formatted results.
    
    Args:
        limit: Maximum number of channels
    
    Returns:
        Formatted channel list
    """
    client = SlackClient()
    channels = client.list_channels(limit)
    
    if isinstance(channels, dict) and not channels.get("ok"):
        error = channels.get("error", "unknown")
        if error == "invalid_auth":
            return "❌ Slack authentication failed. The token may have expired. Update SLACK_TOKEN in .env with a bot token (xoxb-...)."
        return f"❌ Slack API error: {error}"
    
    if not channels:
        return "No channels found or SLACK_TOKEN not configured."
    
    output = [f"Found {len(channels)} channels:"]
    output.append("")
    
    for channel in channels:
        name = channel.get("name", "unknown")
        id = channel.get("id", "")
        is_private = channel.get("is_private", False)
        members = channel.get("num_members", 0)
        topic = channel.get("topic", {}).get("value", "")[:100]
        
        privacy = "🔒 Private" if is_private else "🌐 Public"
        output.append(f"#{name} ({id})")
        output.append(f"   {privacy} | {members} members")
        if topic:
            output.append(f"   Topic: {topic}")
        output.append("")
    
    return "\n".join(output)
