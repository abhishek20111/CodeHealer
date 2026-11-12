#!/usr/bin/env python3
"""
Test Slack Integration - Check if we can access and write to app-tech-debt channel
"""

import os
import json
import subprocess

# Get credentials from environment or .env file
def load_env():
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

load_env()

SLACK_TOKEN = os.getenv("SLACK_TOKEN", "").strip()
SLACK_COOKIE_D = os.getenv("SLACK_COOKIE_D", "").strip()
SLACK_WORKSPACE_URL = os.getenv("SLACK_WORKSPACE_URL", "https://slack.com").strip()

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def call_slack_api(method, params=None):
    """Call Slack API using curl (same as slack_client.py)"""
    if not SLACK_TOKEN:
        return {"ok": False, "error": "not_configured"}
    
    url = f"{SLACK_WORKSPACE_URL}/api/{method}"
    
    print(f"   🔗 URL: {url}")
    print(f"   🔑 Token: {SLACK_TOKEN[:20]}...{SLACK_TOKEN[-10:]}")
    print(f"   🍪 Cookie: {SLACK_COOKIE_D[:20] if SLACK_COOKIE_D else 'None'}...{SLACK_COOKIE_D[-10:] if SLACK_COOKIE_D else ''}")
    
    cmd = [
        "curl", "--noproxy", "*", "--insecure", "-X", "POST",
        "-H", f"Authorization: Bearer {SLACK_TOKEN}",
        "--silent", "--show-error",
    ]
    
    if SLACK_COOKIE_D:
        cmd.extend(["-H", f"Cookie: d={SLACK_COOKIE_D}"])
    
    if params:
        for key, value in params.items():
            if value is not None:
                cmd.extend(["--data-urlencode", f"{key}={value}"])
    
    cmd.append(url)
    
    print(f"   📤 Sending request...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return {"ok": False, "error": "curl_failed", "message": result.stderr[:200]}
        
        response = json.loads(result.stdout)
        print(f"   📥 Response: {json.dumps(response, indent=2)[:300]}")
        return response
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except json.JSONDecodeError as e:
        print(f"   ❌ Invalid JSON: {result.stdout[:500]}")
        return {"ok": False, "error": "invalid_json", "message": str(e)}
    except Exception as e:
        return {"ok": False, "error": "unknown", "message": str(e)}

def main():
    print_section("🧪 SLACK API TEST - APP-TECH-DEBT CHANNEL")
    
    # Test 1: List all channels
    print_section("1️⃣ Listing All Channels")
    print("Fetching channels...")
    channels_result = call_slack_api("conversations.list", {"types": "public_channel,private_channel"})
    
    if not channels_result.get("ok"):
        print(f"❌ Failed to list channels: {channels_result.get('error')}")
        print(f"   Message: {channels_result.get('message', 'Unknown error')}")
        return
    
    channels = channels_result.get("channels", [])
    print(f"✅ Found {len(channels)} channels\n")
    
    # Find app-tech-debt
    target_channel = None
    for ch in channels:
        channel_name = ch.get("name", "")
        if "app-tech-debt" in channel_name.lower():
            target_channel = ch
            print(f"🎯 Found target channel: #{channel_name}")
            print(f"   ID: {ch.get('id')}")
            print(f"   Members: {ch.get('num_members', 'N/A')}")
            print(f"   Private: {ch.get('is_private', False)}")
            break
    
    if not target_channel:
        print("❌ Channel 'app-tech-debt' not found!")
        print("\nAvailable channels (first 10):")
        for ch in channels[:10]:
            print(f"   - #{ch.get('name')} (ID: {ch.get('id')})")
        return
    
    channel_id = target_channel.get("id")
    
    # Test 2: Get channel history
    print_section("2️⃣ Reading Channel History")
    print(f"Fetching messages from #{target_channel.get('name')}...")
    history_result = call_slack_api("conversations.history", {"channel": channel_id, "limit": "5"})
    
    if not history_result.get("ok"):
        print(f"❌ Failed to read history: {history_result.get('error')}")
        print(f"   Message: {history_result.get('message', 'Unknown error')}")
    else:
        messages = history_result.get("messages", [])
        print(f"✅ Found {len(messages)} recent messages")
        if messages:
            for i, msg in enumerate(messages, 1):
                text = msg.get("text", "")[:50]
                user = msg.get("user", "unknown")
                print(f"   {i}. [{user}] {text}...")
        else:
            print("   (No messages yet)")
    
    # Test 3: Post a test message
    print_section("3️⃣ Posting Test Message")
    test_message = "🤖 Test from AI Agent - Checking if I can write to this channel!"
    print(f"Posting message: '{test_message}'")
    
    post_result = call_slack_api("chat.postMessage", {"channel": channel_id, "text": test_message})
    
    if post_result.get("ok"):
        print("✅ Successfully posted message!")
        print(f"   Timestamp: {post_result.get('ts')}")
    else:
        print(f"❌ Failed to post message: {post_result.get('error')}")
        print(f"   Message: {post_result.get('message', 'Unknown error')}")
        
        # Additional debugging
        if post_result.get("error") == "not_in_channel":
            print("\n💡 TIP: The bot needs to be added to the channel!")
            print("   1. Go to channel #app-tech-debt in Slack")
            print("   2. Click 'Add apps' or invite the app")
            print("   3. Search for your app and add it")
        
        if post_result.get("error") == "channel_not_found":
            print("\n💡 TIP: Double check the channel name or permissions")
    
    # Test 4: Search for test message
    print_section("4️⃣ Searching Messages (verification)")
    search_result = call_slack_api("search.messages", {"query": "Test from AI Agent", "count": "5"})
    
    if search_result.get("ok"):
        matches = search_result.get("messages", {}).get("matches", [])
        print(f"✅ Search found {len(matches)} matching messages")
    else:
        print(f"⚠️  Search failed: {search_result.get('error')}")
    
    print_section("✨ Test Complete!")
    
    # Summary
    print("\n📊 SUMMARY:")
    print(f"   ✓ List Channels: {'✅' if channels_result.get('ok') else '❌'}")
    print(f"   ✓ Read History: {'✅' if history_result.get('ok') else '❌'}")
    print(f"   ✓ Post Message: {'✅' if post_result.get('ok') else '❌'}")
    print(f"   ✓ Search: {'✅' if search_result.get('ok') else '❌'}")
    
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
