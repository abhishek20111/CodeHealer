"""
Session Manager - Tracks active user sessions and task states

Handles:
- Session lifecycle (create, update, cleanup)
- Task state management (IDLE, PROCESSING, AWAITING_INPUT, CANCELLED)
- Question/response mechanism for interactive workflows
- Automatic cleanup of stale sessions
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
import threading


class TaskStatus(Enum):
    """Task execution status"""
    IDLE = "idle"                      # No active task
    PROCESSING = "processing"          # Agent is working
    AWAITING_INPUT = "awaiting_input"  # Waiting for user response
    CANCELLED = "cancelled"            # User cancelled task
    COMPLETED = "completed"            # Task finished successfully
    ERROR = "error"                    # Task failed


@dataclass
class SessionState:
    """State for a single user session"""
    session_id: str
    status: TaskStatus = TaskStatus.IDLE
    current_task_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    pending_question: Optional[Dict[str, Any]] = None
    user_response: Optional[str] = None
    response_event: Optional[asyncio.Event] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
    
    def is_stale(self, timeout_minutes: int = 30) -> bool:
        """Check if session is stale (no activity for timeout_minutes)"""
        return datetime.now() - self.last_activity > timedelta(minutes=timeout_minutes)
    
    def reset(self):
        """Reset session to IDLE state"""
        self.status = TaskStatus.IDLE
        self.current_task_id = None
        self.pending_question = None
        self.user_response = None
        self.response_event = None
        self.error_message = None
        self.update_activity()


class SessionManager:
    """
    Manages user sessions and task states.
    
    Thread-safe for concurrent access.
    Handles session cleanup and stale session removal.
    """
    
    def __init__(self, session_timeout_minutes: int = 30):
        self.sessions: Dict[str, SessionState] = {}
        self.session_timeout_minutes = session_timeout_minutes
        self._lock = threading.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
    
    def create_session(self) -> str:
        """Create a new session and return session ID"""
        session_id = str(uuid.uuid4())
        with self._lock:
            self.sessions[session_id] = SessionState(session_id=session_id)
        return session_id
    
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session by ID, return None if not found"""
        with self._lock:
            session = self.sessions.get(session_id)
            if session:
                session.update_activity()
            return session
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> SessionState:
        """Get existing session or create new one"""
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        
        # Create new session
        new_session_id = self.create_session()
        return self.get_session(new_session_id)
    
    def start_task(self, session_id: str, task_id: Optional[str] = None) -> bool:
        """
        Start a new task for session.
        Returns False if session is already processing.
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return False
            
            if session.status == TaskStatus.PROCESSING:
                return False  # Already processing
            
            session.status = TaskStatus.PROCESSING
            session.current_task_id = task_id or str(uuid.uuid4())
            session.update_activity()
            return True
    
    def complete_task(self, session_id: str):
        """Mark task as completed"""
        with self._lock:
            session = self.sessions.get(session_id)
            if session:
                session.status = TaskStatus.COMPLETED
                session.update_activity()
    
    def cancel_task(self, session_id: str) -> bool:
        """
        Cancel active task.
        Returns False if no active task.
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return False
            
            if session.status not in [TaskStatus.PROCESSING, TaskStatus.AWAITING_INPUT]:
                return False
            
            session.status = TaskStatus.CANCELLED
            session.update_activity()
            
            # Wake up any waiting response event
            if session.response_event:
                session.response_event.set()
            
            return True
    
    def set_error(self, session_id: str, error_message: str):
        """Set error state for session"""
        with self._lock:
            session = self.sessions.get(session_id)
            if session:
                session.status = TaskStatus.ERROR
                session.error_message = error_message
                session.update_activity()
                
                # Wake up any waiting response event
                if session.response_event:
                    session.response_event.set()
    
    async def ask_question(
        self, 
        session_id: str, 
        question_text: str, 
        options: List[str],
        timeout_seconds: int = 300
    ) -> Optional[str]:
        """
        Ask user a question and wait for response.
        
        Args:
            session_id: Session ID
            question_text: Question to ask
            options: List of option buttons to show
            timeout_seconds: Max wait time (default 5 minutes)
        
        Returns:
            User's response or None if cancelled/timeout
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return None
            
            # Check if cancelled
            if session.status == TaskStatus.CANCELLED:
                return None
            
            # Set up question
            session.status = TaskStatus.AWAITING_INPUT
            session.pending_question = {
                "text": question_text,
                "options": options
            }
            session.user_response = None
            session.response_event = asyncio.Event()
            session.update_activity()
            
            response_event = session.response_event
        
        # Wait for response (outside lock to allow other operations)
        try:
            await asyncio.wait_for(response_event.wait(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            with self._lock:
                if session.status == TaskStatus.AWAITING_INPUT:
                    session.status = TaskStatus.ERROR
                    session.error_message = "Question timeout - no response received"
            return None
        
        # Get response
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return None
            
            # Check if cancelled
            if session.status == TaskStatus.CANCELLED:
                return None
            
            response = session.user_response
            
            # Resume processing
            session.status = TaskStatus.PROCESSING
            session.pending_question = None
            session.user_response = None
            session.response_event = None
            session.update_activity()
            
            return response
    
    def provide_response(self, session_id: str, response: str) -> bool:
        """
        Provide user's response to pending question.
        Returns False if no question pending.
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if not session:
                return False
            
            if session.status != TaskStatus.AWAITING_INPUT:
                return False
            
            if not session.pending_question:
                return False
            
            session.user_response = response
            session.update_activity()
            
            # Wake up waiting task
            if session.response_event:
                session.response_event.set()
            
            return True
    
    def cleanup_session(self, session_id: str):
        """Remove session"""
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
    
    def cleanup_stale_sessions(self):
        """Remove stale sessions (no activity for timeout period)"""
        with self._lock:
            stale_sessions = [
                sid for sid, session in self.sessions.items()
                if session.is_stale(self.session_timeout_minutes)
            ]
            
            for sid in stale_sessions:
                del self.sessions[sid]
            
            if stale_sessions:
                print(f"[SessionManager] Cleaned up {len(stale_sessions)} stale sessions")
    
    async def start_cleanup_task(self, interval_seconds: int = 300):
        """Start background task to cleanup stale sessions every interval_seconds"""
        while True:
            await asyncio.sleep(interval_seconds)
            self.cleanup_stale_sessions()
    
    def get_session_count(self) -> int:
        """Get number of active sessions"""
        with self._lock:
            return len(self.sessions)
    
    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all sessions (for debugging)"""
        with self._lock:
            return {
                sid: {
                    "status": session.status.value,
                    "task_id": session.current_task_id,
                    "has_question": session.pending_question is not None,
                    "last_activity": session.last_activity.isoformat()
                }
                for sid, session in self.sessions.items()
            }


# Global session manager instance
session_manager = SessionManager(session_timeout_minutes=30)
