"""
Approval Workflow Manager for Automated Error Analysis & PR Creation

This module handles:
1. Error detection from Slack/OBS
2. Proposal generation
3. Approval/rejection handling
4. Automated PR creation
"""

import json
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass, asdict
from enum import Enum

# Import our existing tools
from integrations.slack.slack_client import SlackClient
from integrations.obs.query_client import ObsQueryClient
from integrations.github.github_tools import (
    create_branch,
    commit_file,
    create_pull_request,
    search_github_code
)


# ══════════════════════════════════════════════════════════════════════════════
# State Machine
# ══════════════════════════════════════════════════════════════════════════════

class ProposalStatus(Enum):
    PENDING = "pending"                    # Waiting for approval
    APPROVED = "approved"                  # User approved
    REJECTED_MANUAL = "rejected_manual"    # User will do manually
    REJECTED_FEEDBACK = "rejected_feedback"  # User wants changes
    CREATING_PR = "creating_pr"            # In progress
    COMPLETED = "completed"                # PR created successfully
    FAILED = "failed"                      # Error during PR creation


@dataclass
class ProposalData:
    """Represents a fix proposal"""
    id: str
    timestamp: str
    status: ProposalStatus
    
    # Error details
    error_type: str
    application_name: str
    url_path: str
    error_message: str
    trace_id: str
    
    # Analysis
    root_cause: str
    affected_file: str
    repo: str
    branch: str
    
    # Proposed fix
    fix_description: str
    code_changes: Dict[str, str]  # file_path -> new_content
    
    # Slack interaction
    slack_channel: str
    slack_message_ts: Optional[str] = None
    
    # Approval tracking
    approved_by: Optional[str] = None
    rejection_reason: Optional[str] = None
    feedback_message: Optional[str] = None
    
    # PR details
    pr_url: Optional[str] = None
    pr_number: Optional[int] = None


# ══════════════════════════════════════════════════════════════════════════════
# Proposal Storage
# ══════════════════════════════════════════════════════════════════════════════

class ProposalStore:
    """Manages proposal persistence"""
    
    def __init__(self, file_path: str = "json/proposals.json"):
        self.file_path = file_path
        self.proposals: Dict[str, ProposalData] = {}
        self.load()
    
    def load(self):
        """Load proposals from file"""
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                data = json.load(f)
                for pid, pdata in data.items():
                    pdata['status'] = ProposalStatus(pdata['status'])
                    self.proposals[pid] = ProposalData(**pdata)
    
    def save(self):
        """Save proposals to file"""
        data = {}
        for pid, proposal in self.proposals.items():
            proposal_dict = asdict(proposal)
            proposal_dict['status'] = proposal.status.value
            data[pid] = proposal_dict
        
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def add(self, proposal: ProposalData):
        """Add new proposal"""
        self.proposals[proposal.id] = proposal
        self.save()
    
    def get(self, proposal_id: str) -> Optional[ProposalData]:
        """Get proposal by ID"""
        return self.proposals.get(proposal_id)
    
    def update(self, proposal_id: str, **kwargs):
        """Update proposal fields"""
        if proposal_id in self.proposals:
            proposal = self.proposals[proposal_id]
            for key, value in kwargs.items():
                setattr(proposal, key, value)
            self.save()
    
    def get_pending(self) -> List[ProposalData]:
        """Get all pending proposals"""
        return [p for p in self.proposals.values() 
                if p.status == ProposalStatus.PENDING]


# ══════════════════════════════════════════════════════════════════════════════
# Approval Parser
# ══════════════════════════════════════════════════════════════════════════════

class ApprovalParser:
    """Parse user responses from Slack"""
    
    @staticmethod
    def parse_response(message: str) -> Dict:
        """
        Parse user response and determine intent
        
        Examples:
            "approve" → {"action": "approve"}
            "✅" → {"action": "approve"}
            "reject - I'll do it myself" → {"action": "reject_manual"}
            "reject - use repository pattern instead" → {
                "action": "reject_feedback", 
                "feedback": "use repository pattern instead"
            }
        """
        message = message.strip().lower()
        
        # Approve patterns
        if message in ["approve", "✅", "yes", "looks good", "lgtm"]:
            return {"action": "approve"}
        
        # Reject with feedback
        if message.startswith("reject"):
            # Extract feedback after "reject -"
            if " - " in message:
                feedback = message.split(" - ", 1)[1].strip()
                
                # Check if user wants to do it manually
                if any(phrase in feedback for phrase in [
                    "i'll do", "i will do", "do myself", "manually",
                    "handle it", "no thanks"
                ]):
                    return {"action": "reject_manual"}
                else:
                    return {
                        "action": "reject_feedback",
                        "feedback": feedback
                    }
            else:
                # Generic reject
                return {"action": "reject_manual"}
        
        # Explicit reject
        if message in ["❌", "no", "reject", "skip"]:
            return {"action": "reject_manual"}
        
        # Unknown - treat as feedback
        return {
            "action": "reject_feedback",
            "feedback": message
        }


# ══════════════════════════════════════════════════════════════════════════════
# Workflow Manager
# ══════════════════════════════════════════════════════════════════════════════

class ApprovalWorkflowManager:
    """Manages the full approval workflow"""
    
    def __init__(self):
        self.store = ProposalStore()
        self.slack = SlackClient()
        self.parser = ApprovalParser()
        
        # Load application mappings
        with open('json/application_mappings.json', 'r') as f:
            self.app_mappings = json.load(f)
    
    def create_proposal(
        self,
        error_details: Dict,
        analysis: Dict,
        proposed_fix: Dict,
        slack_channel: str = "team-channel"
    ) -> str:
        """
        Create new proposal and post to Slack
        
        Returns:
            proposal_id
        """
        # Generate unique ID
        proposal_id = f"proposal-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Create proposal object
        proposal = ProposalData(
            id=proposal_id,
            timestamp=datetime.now().isoformat(),
            status=ProposalStatus.PENDING,
            
            # Error details
            error_type=error_details.get('error_type', 'Unknown'),
            application_name=error_details.get('application_name', ''),
            url_path=error_details.get('url_path', ''),
            error_message=error_details.get('error_message', ''),
            trace_id=error_details.get('trace_id', ''),
            
            # Analysis
            root_cause=analysis.get('root_cause', ''),
            affected_file=analysis.get('affected_file', ''),
            repo=analysis.get('repo', ''),
            branch=f"fix/{proposal_id}",
            
            # Fix
            fix_description=proposed_fix.get('description', ''),
            code_changes=proposed_fix.get('code_changes', {}),
            
            # Slack
            slack_channel=slack_channel
        )
        
        # Save proposal
        self.store.add(proposal)
        
        # Post to Slack
        self._post_proposal_to_slack(proposal)
        
        return proposal_id
    
    def _post_proposal_to_slack(self, proposal: ProposalData):
        """Post proposal to Slack with instructions"""
        
        # Build message
        message = f"""
🔧 **Automated Fix Proposal** (ID: `{proposal.id}`)

**Error Detected:**
• Type: {proposal.error_type}
• Application: {proposal.application_name}
• Endpoint: `{proposal.url_path}`
• Trace ID: `{proposal.trace_id}`

**Root Cause Analysis:**
{proposal.root_cause}

**Affected File:**
`{proposal.affected_file}`

**Proposed Fix:**
{proposal.fix_description}

**Code Changes:**
```
{self._format_code_changes(proposal.code_changes)}
```

**Repository:** {proposal.repo}
**Branch:** {proposal.branch}

---

**To proceed, reply with:**
• `approve` or `✅` → I'll create the PR immediately
• `reject - I'll do it myself` → Stop, I'll handle manually
• `reject - <your feedback>` → Generate new proposal with your suggestions

Example: `reject - use try-catch instead of null check`
"""
        
        # Post to Slack
        result = self.slack.post_message(proposal.slack_channel, message)
        
        if result.get('ok'):
            # Store message timestamp for tracking replies
            self.store.update(proposal.id, slack_message_ts=result.get('ts'))
    
    def _format_code_changes(self, code_changes: Dict[str, str]) -> str:
        """Format code changes for display"""
        result = []
        for file_path, content in code_changes.items():
            result.append(f"// {file_path}")
            result.append(content[:500])  # Truncate for Slack
            if len(content) > 500:
                result.append("... (truncated)")
        return "\n".join(result)
    
    async def check_for_approvals(self):
        """
        Check Slack for responses to pending proposals
        
        This should be called periodically (e.g., every 30 seconds)
        """
        pending = self.store.get_pending()
        
        for proposal in pending:
            # Get replies to the proposal message
            replies = self._get_slack_replies(
                proposal.slack_channel,
                proposal.slack_message_ts
            )
            
            if replies:
                # Process the latest reply
                latest_reply = replies[-1]
                await self._handle_response(proposal, latest_reply)
    
    def _get_slack_replies(
        self,
        channel: str,
        message_ts: Optional[str]
    ) -> List[Dict]:
        """Get replies to a Slack message"""
        if not message_ts:
            return []
        
        # Get conversation replies
        # This would use conversations.replies API
        # For now, simplified version
        history = self.slack.get_channel_history(channel, limit=100)
        
        if not history.get('ok'):
            return []
        
        # Filter for replies to our message
        # (In real implementation, use thread_ts filtering)
        return []
    
    async def _handle_response(
        self,
        proposal: ProposalData,
        reply: Dict
    ):
        """Handle user response to proposal"""
        
        message_text = reply.get('text', '')
        user_id = reply.get('user', 'unknown')
        
        # Parse response
        parsed = self.parser.parse_response(message_text)
        action = parsed['action']
        
        if action == "approve":
            await self._handle_approval(proposal, user_id)
        
        elif action == "reject_manual":
            await self._handle_rejection_manual(proposal, user_id)
        
        elif action == "reject_feedback":
            feedback = parsed.get('feedback', '')
            await self._handle_rejection_feedback(proposal, user_id, feedback)
    
    async def _handle_approval(self, proposal: ProposalData, user_id: str):
        """Handle proposal approval - create PR"""
        
        print(f"[WORKFLOW] Proposal {proposal.id} approved by {user_id}")
        
        # Update status
        self.store.update(
            proposal.id,
            status=ProposalStatus.CREATING_PR,
            approved_by=user_id
        )
        
        try:
            # 1. Create branch
            print(f"[WORKFLOW] Creating branch: {proposal.branch}")
            create_branch(proposal.repo, proposal.branch)
            
            # 2. Commit code changes
            for file_path, content in proposal.code_changes.items():
                print(f"[WORKFLOW] Committing: {file_path}")
                commit_file(
                    repo=proposal.repo,
                    branch=proposal.branch,
                    file_path=file_path,
                    content=content,
                    message=f"Fix: {proposal.fix_description}"
                )
            
            # 3. Create PR
            print(f"[WORKFLOW] Creating PR")
            pr_result = create_pull_request(
                repo=proposal.repo,
                branch=proposal.branch,
                title=f"Fix: {proposal.error_type} in {proposal.affected_file}",
                body=f"""
## Automated Fix

**Error:** {proposal.error_type}
**Trace ID:** {proposal.trace_id}

**Root Cause:**
{proposal.root_cause}

**Fix Description:**
{proposal.fix_description}

**Approved by:** <@{proposal.approved_by}>

---
*This PR was automatically generated by the AI Error Analysis System*
*Proposal ID: {proposal.id}*
"""
            )
            
            pr_url = pr_result.get('html_url', '')
            pr_number = pr_result.get('number', 0)
            
            # Update proposal
            self.store.update(
                proposal.id,
                status=ProposalStatus.COMPLETED,
                pr_url=pr_url,
                pr_number=pr_number
            )
            
            # Notify in Slack
            self.slack.post_message(
                proposal.slack_channel,
                f"✅ PR created successfully!\n{pr_url}\n\nApproved by: <@{user_id}>"
            )
            
        except Exception as e:
            print(f"[WORKFLOW] Error creating PR: {e}")
            self.store.update(
                proposal.id,
                status=ProposalStatus.FAILED
            )
            self.slack.post_message(
                proposal.slack_channel,
                f"❌ Failed to create PR: {str(e)}"
            )
    
    async def _handle_rejection_manual(self, proposal: ProposalData, user_id: str):
        """Handle manual rejection - user will do it themselves"""
        
        print(f"[WORKFLOW] Proposal {proposal.id} rejected (manual) by {user_id}")
        
        self.store.update(
            proposal.id,
            status=ProposalStatus.REJECTED_MANUAL,
            rejection_reason="User will handle manually"
        )
        
        # Thank user
        self.slack.post_message(
            proposal.slack_channel,
            f"👍 Got it! I'll let you handle this one. Thanks for letting me know!"
        )
    
    async def _handle_rejection_feedback(
        self,
        proposal: ProposalData,
        user_id: str,
        feedback: str
    ):
        """Handle rejection with feedback - generate new proposal"""
        
        print(f"[WORKFLOW] Proposal {proposal.id} rejected with feedback by {user_id}")
        print(f"[WORKFLOW] Feedback: {feedback}")
        
        self.store.update(
            proposal.id,
            status=ProposalStatus.REJECTED_FEEDBACK,
            rejection_reason=f"User feedback: {feedback}",
            feedback_message=feedback
        )
        
        # Notify user
        self.slack.post_message(
            proposal.slack_channel,
            f"🔄 Got your feedback! Generating a new proposal based on: \"{feedback}\"\n\nThis will take a moment..."
        )
        
        # TODO: Trigger agent to generate new proposal with feedback
        # This would call back to the main agent with the feedback
        # For now, just acknowledge
        await asyncio.sleep(2)
        
        self.slack.post_message(
            proposal.slack_channel,
            f"📝 New proposal generation queued. The agent will analyze your feedback and create an updated fix proposal shortly."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Background Monitor (For Automation)
# ══════════════════════════════════════════════════════════════════════════════

class BackgroundMonitor:
    """
    Monitors Slack channels for new errors and checks for approvals
    """
    
    def __init__(
        self,
        workflow_manager: ApprovalWorkflowManager,
        check_interval: int = 60  # seconds
    ):
        self.workflow = workflow_manager
        self.check_interval = check_interval
        self.running = False
    
    async def start(self):
        """Start background monitoring"""
        self.running = True
        print(f"[MONITOR] Starting background monitoring (interval: {self.check_interval}s)")
        
        while self.running:
            try:
                # Check for approvals
                await self.workflow.check_for_approvals()
                
                # Wait before next check
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                print(f"[MONITOR] Error in background monitor: {e}")
                await asyncio.sleep(self.check_interval)
    
    def stop(self):
        """Stop background monitoring"""
        self.running = False
        print("[MONITOR] Stopping background monitoring")


# ══════════════════════════════════════════════════════════════════════════════
# Example Usage
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example: Create a proposal
    workflow = ApprovalWorkflowManager()
    
    error_details = {
        "error_type": "NullPointerException",
        "application_name": "sampleapi",
        "url_path": "/clients/CHANGE_HEALTHCARE_USA_5475/optimizations/payment/input",
        "error_message": "Cannot invoke method on null object",
        "trace_id": "d0c38de31271afd3b6444ae407667f72"
    }
    
    analysis = {
        "root_cause": "PaymentInputService.processInput() doesn't check for null client ID",
        "affected_file": "src/main/java/com/example/sampleapp/optimizations/service/PaymentInputService.java",
        "repo": "github.example.com/your-username/sample-app"  # fork of your-org/sample-app
    }
    
    proposed_fix = {
        "description": "Add null check before processing client optimization request",
        "code_changes": {
            "src/main/java/com/example/sampleapp/optimizations/service/PaymentInputService.java": '''
public void processInput(OptimizationRequest request) {
    // Add null check
    if (request.getClientId() == null) {
        throw new IllegalArgumentException("Client ID cannot be null");
    }
    
    // Existing logic...
    optimizationEngine.process(request);
}
'''
        }
    }
    
    proposal_id = workflow.create_proposal(
        error_details,
        analysis,
        proposed_fix,
        slack_channel="team-channel"
    )
    
    print(f"Created proposal: {proposal_id}")
