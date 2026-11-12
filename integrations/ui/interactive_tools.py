"""
Interactive Workflow Tools - UI-based question/response workflow

These tools use the session_manager to ask questions in the UI instead of Slack.
Supports the full error-to-PR workflow with user interaction.
"""

from typing import Optional
from langchain_core.tools import tool
from session_manager import session_manager


# Global context to pass session_id to tools
_current_task_session_id = None


def set_task_session(session_id: str):
    """Set the current task session ID for tools to use"""
    global _current_task_session_id
    _current_task_session_id = session_id


def get_task_session() -> Optional[str]:
    """Get the current task session ID"""
    return _current_task_session_id


async def _ask_user_question_impl(question: str, options: list[str]) -> str:
    """Internal implementation - can be called from other tools directly."""
    session_id = get_task_session()
    if not session_id:
        return "Error: No active task session"
    
    print(f"\n🤔 Asking user: {question}")
    print(f"   Options: {options}")
    
    response = await session_manager.ask_question(
        session_id=session_id,
        question_text=question,
        options=options,
        timeout_seconds=300  # 5 minutes
    )
    
    if response is None:
        return "cancelled"
    
    print(f"✅ User responded: {response}")
    return response


@tool
async def ask_user_question(question: str, options: list[str]) -> str:
    """
    Ask the user a question and wait for their response.
    
    Args:
        question: The question to ask the user
        options: List of option buttons to show (e.g., ["Yes", "No", "Cancel"])
    
    Returns:
        The user's selected response
    
    Example:
        response = await ask_user_question(
            "Found error in TraceID abc123. Investigate?",
            ["Investigate", "Skip", "Cancel"]
        )
    """
    return await _ask_user_question_impl(question, options)


@tool
async def investigate_error_interactive(
    error_summary: str,
    trace_id: str,
    application: str
) -> str:
    """
    Ask user if they want to investigate an error.
    
    Args:
        error_summary: Brief description of the error
        trace_id: Trace ID for OBS lookup
        application: Application name
    
    Returns:
        "investigate" if user wants to investigate, "skip" otherwise
    """
    question = f"""🔍 **Error Found**

**Application:** {application}
**Trace ID:** {trace_id}
**Summary:** {error_summary}

Would you like to investigate this error?"""
    
    response = await _ask_user_question_impl(question, ["Investigate", "Skip", "Cancel"])
    return response.lower()


@tool
async def ask_create_pr(
    file_path: str,
    fix_description: str,
    code_changes: str,
    branch_name: str
) -> str:
    """
    Ask user if they want to create a PR with the proposed fix.
    
    Args:
        file_path: Path to the file being fixed
        fix_description: Description of the fix
        code_changes: The actual code changes (diff format)
        branch_name: Proposed branch name
    
    Returns:
        "create_pr", "create_branch", or "cancel"
    """
    question = f"""✅ **Fix Ready**

**File:** `{file_path}`
**Branch:** `{branch_name}`

**Fix Description:**
{fix_description}

**Changes:**
```
{code_changes[:500]}{'...' if len(code_changes) > 500 else ''}
```

What would you like to do?"""
    
    response = await _ask_user_question_impl(
        question,
        ["Create PR", "Create Branch Only", "Cancel"]
    )
    
    response_map = {
        "create pr": "create_pr",
        "create branch only": "create_branch",
        "cancel": "cancel"
    }
    
    return response_map.get(response.lower(), "cancel")


@tool
async def show_analysis_result(
    title: str,
    content: str,
    actions: Optional[list[str]] = None
) -> str:
    """
    Show analysis results to user and optionally ask for next action.
    
    Args:
        title: Title of the analysis
        content: Analysis content
        actions: Optional list of action buttons
    
    Returns:
        User's selected action or "continue"
    """
    if not actions:
        # Just informational, no question
        print(f"\n📊 {title}")
        print(content)
        return "continue"
    
    question = f"""📊 **{title}**

{content}

What would you like to do next?"""
    
    response = await _ask_user_question_impl(question, actions)
    return response.lower()


# Export all interactive tools
INTERACTIVE_WORKFLOW_TOOLS = [
    ask_user_question,
    investigate_error_interactive,
    ask_create_pr,
    show_analysis_result,
]
