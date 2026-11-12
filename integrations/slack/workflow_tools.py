"""
Slack Workflow Tools for Error Analysis & Proposal System

These tools enable:
1. Analyzing error messages from Slack
2. Creating fix proposals
3. Handling approvals/rejections
4. Full error-to-PR workflow
"""

import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from langchain_core.tools import tool

# Import existing tools
from .slack_client import SlackClient
from ..obs.query_client import ObsQueryClient
from ..github.github_tools import search_github_code, read_github_file


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

def load_config():
    """Load configuration"""
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.json")
    with open(config_path, 'r') as f:
        return json.load(f)


def load_app_mappings():
    """Load application mappings"""
    mappings_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "json", "application_mappings.json"
    )
    with open(mappings_path, 'r') as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# Tool 1: Analyze Last Error from Slack Channel
# ══════════════════════════════════════════════════════════════════════════════

@tool
def analyze_last_slack_error(
    channel_name: Optional[str] = None,
    include_obs_query: bool = True
) -> str:
    """
    Analyze the last error message from a Slack channel.
    
    This tool:
    1. Reads the last message from the error channel
    2. Parses error details (application, endpoint, error type)
    3. Optionally queries OBS for full error details
    4. Returns comprehensive error analysis
    
    Args:
        channel_name: Slack channel name (default: from config)
        include_obs_query: Whether to query OBS for detailed logs
    
    Returns:
        Detailed error analysis with root cause and affected components
    
    Example:
        analyze_last_slack_error("app-errors")
    """
    try:
        # Load config
        config = load_config()
        if not channel_name:
            channel_name = config['slack']['channels']['errors']
        
        # Get Slack client
        slack = SlackClient()
        
        # Get channel history
        history_result = slack.get_channel_history(channel_name, limit=10)
        
        if not history_result.get('ok') or not history_result.get('messages'):
            return f"❌ Could not retrieve messages from #{channel_name}"
        
        # Parse messages for errors
        messages = history_result['messages']
        error_messages = []
        
        for msg in messages:
            text = msg.get('text', '')
            # Look for error indicators
            if any(keyword in text.lower() for keyword in [
                'error', '500', '404', 'exception', 'failed', 'opensearch dashboard'
            ]):
                error_messages.append({
                    'text': text,
                    'timestamp': msg.get('ts', ''),
                    'user': msg.get('user', 'bot')
                })
        
        if not error_messages:
            return f"✅ No error messages found in #{channel_name} (last 10 messages checked)"
        
        # Analyze the most recent error
        latest_error = error_messages[0]
        error_text = latest_error['text']
        
        # Parse error details
        analysis = {
            'channel': channel_name,
            'timestamp': latest_error['timestamp'],
            'raw_message': error_text,
            'parsed_details': {}
        }
        
        # Extract application name
        if 'application.name' in error_text or 'sampleapi' in error_text.lower():
            analysis['parsed_details']['application'] = 'sampleapi'
        
        # Extract error type
        if '500' in error_text:
            analysis['parsed_details']['error_type'] = '500 Internal Server Error'
        elif '404' in error_text:
            analysis['parsed_details']['error_type'] = '404 Not Found'
        elif 'ERROR' in error_text:
            analysis['parsed_details']['error_type'] = 'ERROR'
        
        # Extract endpoint
        import re
        endpoint_match = re.search(r'/[\w/]+(?:/[\w-]+)*', error_text)
        if endpoint_match:
            analysis['parsed_details']['endpoint'] = endpoint_match.group(0)
        
        # Extract OpenSearch link
        opensearch_match = re.search(r'https://logs-[^)]+', error_text)
        if opensearch_match:
            analysis['parsed_details']['opensearch_url'] = opensearch_match.group(0)
        
        # Query OBS if requested
        if include_obs_query and 'application' in analysis['parsed_details']:
            app_name = analysis['parsed_details']['application']
            
            obs_client = ObsQueryClient(
                app_id=os.getenv('OBS_APP_ID', '100000001'),
                username=os.getenv('OBS_USERNAME', 'your-username'),
                password=os.getenv('OBS_PASSWORD', ''),
                proxy_url=os.getenv('OBS_PROXY_URL', 'https://obsproxy-qa.example.com')
            )
            
            # Query OBS for recent errors
            obs_result = obs_client.search(
                query=f'application.name:{app_name} AND tags.status_code:ERROR',
                time_range='1h',
                size=1
            )
            
            if 'hits' in obs_result and obs_result['hits']['hits']:
                log_entry = obs_result['hits']['hits'][0]['_source']
                analysis['obs_details'] = {
                    'trace_id': log_entry.get('traceId', ''),
                    'url_path': log_entry.get('tags', {}).get('url#path', ''),
                    'error_type': log_entry.get('tags', {}).get('error#type', ''),
                    'git_commit': log_entry.get('tags', {}).get('git#commit', ''),
                    'environment': log_entry.get('environment', {}).get('name', ''),
                    'timestamp': log_entry.get('timestamp', ''),
                    'operation': log_entry.get('operation', '')
                }
        
        # Load app mappings to get repo info
        app_mappings = load_app_mappings()
        app_id = os.getenv('OBS_APP_ID', '100000001')
        
        if app_id in app_mappings:
            app_name = analysis['parsed_details'].get('application', '')
            if app_name in app_mappings[app_id]['applications']:
                app_config = app_mappings[app_id]['applications'][app_name]
                analysis['repo_info'] = {
                    'repo': app_config['repo'],
                    'branch': app_config['branch'],
                    'description': app_config['description']
                }
                
                # Try to map endpoint to code file
                endpoint = analysis['parsed_details'].get('endpoint', '')
                if endpoint and 'url_to_code_mapping' in app_config:
                    for pattern, file_info in app_config['url_to_code_mapping'].items():
                        if pattern in endpoint or endpoint in pattern:
                            analysis['likely_file'] = file_info['file']
                            break
        
        # Format response
        result = f"""
📊 **Error Analysis from #{channel_name}**

**Error Details:**
• Type: {analysis['parsed_details'].get('error_type', 'Unknown')}
• Application: {analysis['parsed_details'].get('application', 'Unknown')}
• Endpoint: {analysis['parsed_details'].get('endpoint', 'N/A')}
"""
        
        if 'obs_details' in analysis:
            result += f"""
**OBS Log Details:**
• Trace ID: {analysis['obs_details']['trace_id']}
• URL Path: {analysis['obs_details']['url_path']}
• Git Commit: {analysis['obs_details']['git_commit']}
• Environment: {analysis['obs_details']['environment']}
• Timestamp: {analysis['obs_details']['timestamp']}
"""
        
        if 'repo_info' in analysis:
            result += f"""
**Repository:**
• Repo: {analysis['repo_info']['repo']}
• Branch: {analysis['repo_info']['branch']}
"""
        
        if 'likely_file' in analysis:
            result += f"""
**Likely Affected File:**
• {analysis['likely_file']}
"""
        
        result += f"""
**Raw Slack Message:**
{error_text[:500]}...

**Next Steps:**
1. Use search_github_code() to find the affected code
2. Use read_github_file() to read the file
3. Analyze the code and identify the root cause
4. Propose a fix
"""
        
        return result
        
    except Exception as e:
        return f"❌ Error analyzing Slack message: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# Tool 2: Get Recent Errors with Analysis
# ══════════════════════════════════════════════════════════════════════════════

@tool
def get_recent_errors_from_obs(
    application_name: str,
    time_range: str = "1h",
    max_errors: int = 5
) -> str:
    """
    Query OBS for recent errors and provide detailed analysis.
    
    Args:
        application_name: Name of the application (e.g., "sampleapi")
        time_range: Time range to search (e.g., "1h", "24h")
        max_errors: Maximum number of errors to return
    
    Returns:
        Detailed analysis of recent errors with root causes
    
    Example:
        get_recent_errors_from_obs("sampleapi", "1h", 3)
    """
    try:
        # Initialize OBS client
        obs_client = ObsQueryClient(
            app_id=os.getenv('OBS_APP_ID', '100000001'),
            username=os.getenv('OBS_USERNAME', 'your-username'),
            password=os.getenv('OBS_PASSWORD', ''),
            proxy_url=os.getenv('OBS_PROXY_URL', 'https://obsproxy-qa.example.com')
        )
        
        # Query for errors
        result = obs_client.search(
            query=f'application.name:{application_name} AND tags.status_code:ERROR',
            time_range=time_range,
            size=max_errors
        )
        
        if 'hits' not in result or not result['hits']['hits']:
            return f"✅ No errors found for {application_name} in the last {time_range}"
        
        errors = result['hits']['hits']
        
        # Analyze each error
        analysis_results = []
        app_mappings = load_app_mappings()
        app_id = os.getenv('OBS_APP_ID', '100000001')
        
        for idx, error in enumerate(errors, 1):
            source = error['_source']
            tags = source.get('tags', {})
            
            error_analysis = {
                'number': idx,
                'trace_id': source.get('traceId', ''),
                'timestamp': source.get('timestamp', ''),
                'error_type': tags.get('error#type', ''),
                'url_path': tags.get('url#path', ''),
                'http_status': tags.get('http#response#status_code', ''),
                'operation': source.get('operation', ''),
                'git_commit': tags.get('git#commit', ''),
                'environment': source.get('environment', {}).get('name', '')
            }
            
            # Try to find affected file
            if app_id in app_mappings and application_name in app_mappings[app_id]['applications']:
                app_config = app_mappings[app_id]['applications'][application_name]
                url_path = error_analysis['url_path']
                
                if url_path and 'url_to_code_mapping' in app_config:
                    for pattern, file_info in app_config['url_to_code_mapping'].items():
                        if pattern in url_path or url_path in pattern:
                            error_analysis['likely_file'] = file_info['file']
                            error_analysis['file_description'] = file_info['description']
                            break
            
            analysis_results.append(error_analysis)
        
        # Format response
        response = f"""
📊 **Recent Errors for {application_name}** (Last {time_range})
Found {len(errors)} error(s)

"""
        
        for err in analysis_results:
            response += f"""
**Error #{err['number']}**
• Timestamp: {err['timestamp']}
• Type: {err['error_type']} (HTTP {err['http_status']})
• Endpoint: {err['url_path']}
• Operation: {err['operation']}
• Trace ID: {err['trace_id']}
• Git Commit: {err['git_commit']}
• Environment: {err['environment']}
"""
            if 'likely_file' in err:
                response += f"• Likely Affected File: {err['likely_file']}\n"
                response += f"  ({err['file_description']})\n"
            
            response += "\n"
        
        response += """
**Recommended Actions:**
1. Use search_github_code() to examine the affected files
2. Analyze the code patterns causing these errors
3. Propose a fix that addresses the root cause
"""
        
        return response
        
    except Exception as e:
        return f"❌ Error querying OBS: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# Tool 3: Post Analysis to Slack
# ══════════════════════════════════════════════════════════════════════════════

@tool
def post_error_analysis_to_slack(
    channel: str,
    error_summary: str,
    root_cause: str,
    proposed_fix: str,
    tag_user: bool = True
) -> str:
    """
    Post error analysis results to Slack channel.
    
    Args:
        channel: Slack channel name (e.g., "team-channel")
        error_summary: Brief summary of the error
        root_cause: Identified root cause
        proposed_fix: Proposed solution
        tag_user: Whether to tag the configured user
    
    Returns:
        Confirmation message with Slack message link
    
    Example:
        post_error_analysis_to_slack(
            "team-channel",
            "NullPointerException in PaymentService",
            "Missing null check for clientId",
            "Add validation: if (clientId == null) throw exception"
        )
    """
    try:
        config = load_config()
        slack = SlackClient()
        
        # Build message
        user_tag = ""
        if tag_user and config['slack']['notification_preferences']['tag_users_on_proposal']:
            user_name = config['users']['primary']['name']
            user_tag = f"@{user_name} "
        
        message = f"""{user_tag}🔍 **Error Analysis Complete**

**Error Summary:**
{error_summary}

**Root Cause:**
{root_cause}

**Proposed Fix:**
{proposed_fix}

---
*Analysis generated by AI Error Analysis System*
*Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # Post to Slack
        result = slack.post_message(channel, message)
        
        if result.get('ok'):
            return f"✅ Analysis posted to #{channel} successfully!"
        else:
            error_msg = result.get('error', 'Unknown error')
            return f"❌ Failed to post to Slack: {error_msg}"
        
    except Exception as e:
        return f"❌ Error posting to Slack: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# Export all workflow tools
# ══════════════════════════════════════════════════════════════════════════════

WORKFLOW_TOOLS = [
    analyze_last_slack_error,
    get_recent_errors_from_obs,
    post_error_analysis_to_slack
]


def create_workflow_tools():
    """Create and return all workflow tools"""
    return WORKFLOW_TOOLS
