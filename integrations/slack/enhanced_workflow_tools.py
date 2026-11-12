"""
Enhanced workflow tools with approval and PR creation
"""

import json
import os
from typing import Dict, Optional
from datetime import datetime
from langchain_core.tools import tool

# Import existing modules
from .slack_client import SlackClient
from ..obs.query_client import ObsQueryClient
from ..github.github_tools import (
    search_github_code,
    read_github_file,
    create_branch,
    commit_file,
    create_pull_request
)

# Import approval workflow
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from approval_workflow import ApprovalWorkflowManager, ProposalData


# ══════════════════════════════════════════════════════════════════════════════
# Tool: Create Fix Proposal with Approval Workflow
# ══════════════════════════════════════════════════════════════════════════════

@tool
def create_fix_proposal_for_approval(
    error_summary: str,
    root_cause: str,
    affected_file: str,
    proposed_fix_code: str,
    application_name: str,
    endpoint: str,
    trace_id: Optional[str] = None
) -> str:
    """
    Create a fix proposal and post to Slack for approval.
    
    This creates a proposal that includes approve/reject instructions.
    The system will monitor Slack for responses every 2 minutes.
    
    When user replies:
    - "approve" → Automatically creates PR
    - "reject - I'll do it myself" → Stops and says thanks
    - "reject - use X instead" → Generates new proposal with feedback
    
    Args:
        error_summary: Brief description of the error
        root_cause: Root cause analysis
        affected_file: File path that needs changes
        proposed_fix_code: The actual code fix
        application_name: Name of the application (e.g., "sampleapi")
        endpoint: API endpoint with the error
        trace_id: Optional OBS trace ID
    
    Returns:
        Confirmation with proposal ID
    
    Example:
        create_fix_proposal_for_approval(
            error_summary="NullPointerException in PaymentInputService",
            root_cause="Missing null check for clientId parameter",
            affected_file="src/main/java/com/example/sampleapp/optimizations/service/PaymentInputService.java",
            proposed_fix_code="if (clientId == null) throw new IllegalArgumentException(...)",
            application_name="sampleapi",
            endpoint="/clients/*/optimizations/payment/input",
            trace_id="d0c38de31271afd3b6444ae407667f72"
        )
    """
    try:
        # Load config
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "config.json"
        )
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load app mappings
        mappings_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "json", "application_mappings.json"
        )
        with open(mappings_path, 'r') as f:
            app_mappings = json.load(f)
        
        # Get repo info
        app_id = os.getenv('OBS_APP_ID', '100000001')
        repo_info = None
        if app_id in app_mappings and application_name in app_mappings[app_id]['applications']:
            repo_info = app_mappings[app_id]['applications'][application_name]
        
        if not repo_info:
            return f"❌ Cannot create proposal: Application '{application_name}' not found in mappings"
        
        # Initialize workflow manager
        workflow = ApprovalWorkflowManager()
        
        # Prepare error details
        error_details = {
            'error_type': 'Runtime Error',
            'application_name': application_name,
            'url_path': endpoint,
            'error_message': error_summary,
            'trace_id': trace_id or 'N/A'
        }
        
        # Prepare analysis
        analysis = {
            'root_cause': root_cause,
            'affected_file': affected_file,
            'repo': repo_info['repo']
        }
        
        # Prepare fix
        proposed_fix = {
            'description': f"Fix: {error_summary}",
            'code_changes': {
                affected_file: proposed_fix_code
            }
        }
        
        # Get approval channel from config
        approval_channel = config['slack']['channels'].get('approvals', 'team-channel')
        
        # Create proposal
        proposal_id = workflow.create_proposal(
            error_details=error_details,
            analysis=analysis,
            proposed_fix=proposed_fix,
            slack_channel=approval_channel
        )
        
        return f"""
✅ **Fix Proposal Created!**

**Proposal ID:** {proposal_id}
**Posted to:** #{approval_channel}
**Tagged:** @{config['users']['primary']['name']}

**Next Steps:**
The system will monitor Slack for responses every 2 minutes.

**To approve:** Reply with "approve" or ✅
**To reject (manual):** Reply with "reject - I'll do it myself"
**To reject (feedback):** Reply with "reject - use X instead"

When approved, I'll automatically:
1. Create branch: fix/{proposal_id}
2. Commit the code changes
3. Create a Pull Request
4. Post PR link to Slack
"""
        
    except Exception as e:
        return f"❌ Error creating proposal: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# Export enhanced tools
# ══════════════════════════════════════════════════════════════════════════════

ENHANCED_WORKFLOW_TOOLS = [
    create_fix_proposal_for_approval
]


def create_enhanced_workflow_tools():
    """Create and return enhanced workflow tools"""
    return ENHANCED_WORKFLOW_TOOLS
