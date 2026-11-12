"""
══════════════════════════════════════════════════════════════════════════════════
WEB SERVER — Vector Memory Edition
══════════════════════════════════════════════════════════════════════════════════

DUAL STORAGE:
- sessions.json → persistent session metadata + messages (for UI display)
- ChromaDB      → vector embeddings of messages (for semantic retrieval)

WHY BOTH:
- sessions.json survives `rm -rf chroma_db` or DB corruption
- ChromaDB provides semantic search that JSON can't
- UI always has reliable data to show in sidebar
- Agent always has smart context retrieval

SCHEMA LOADING:
- Runs as a BACKGROUND asyncio task (doesn't block server startup)
- Progress tracked via schema_status dict
- UI polls /db/schema-status for live progress
══════════════════════════════════════════════════════════════════════════════════
"""

import asyncio
import json
import os
import uuid
import traceback
from datetime import datetime

from starlette.applications import Starlette
from starlette.responses import JSONResponse, FileResponse, StreamingResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.requests import Request

from agent import create_agent, ask_agent, stream_agent, load_schema_context, log, LOG_SEP, set_task_session, MAX_STORED_RESPONSE
from tools import load_databases, ALL_TOOLS
import re
import vector_store
from vector_store import C
from session_manager import session_manager, TaskStatus


def _clean_for_storage(text: str) -> str:
    """Clean assistant response before storing in ChromaDB.
    
    Strips tool-call markers (🔧 Calling...) and caps length so future
    context retrieval returns concise, useful history instead of bloated
    intermediate step noise.
    """
    # Strip tool-call markers injected during streaming
    cleaned = re.sub(r'\n*🔧[^\n]*\n*', '\n', text)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    if len(cleaned) <= MAX_STORED_RESPONSE:
        return cleaned
    return cleaned[:MAX_STORED_RESPONSE] + "\n\n[...response truncated for storage]"


# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
SESSIONS_FILE = os.path.join(BASE_DIR, "json", "sessions.json")

# ── Global State ─────────────────────────────────────────────────────────────

agent = None
schema_cache = ""
schema_status = {"loading": False, "current": 0, "total": 0, "message": "Idle"}
db_tables_cache = {}  # {db_name: [table1, table2, ...]}
_schema_task = None


# ══════════════════════════════════════════════════════════════════════════════
# SESSION MANAGEMENT — sessions.json for persistence + ChromaDB for vectors
#
# sessions.json = source of truth for UI display (titles, message list)
# ChromaDB = source of truth for semantic retrieval (embeddings)
# Both updated together on every message.
# ══════════════════════════════════════════════════════════════════════════════

def load_sessions() -> list:
    if not os.path.exists(SESSIONS_FILE):
        return []
    with open(SESSIONS_FILE, "r") as f:
        return json.load(f)


def save_sessions(sessions: list):
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions, f, indent=2)


def get_session(session_id: str) -> dict | None:
    sessions = load_sessions()
    return next((s for s in sessions if s["id"] == session_id), None)


def update_session(session: dict):
    sessions = load_sessions()
    for i, s in enumerate(sessions):
        if s["id"] == session["id"]:
            sessions[i] = session
            save_sessions(sessions)
            return
    sessions.insert(0, session)
    save_sessions(sessions)


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA LOADING — Background task with progress tracking
#
# This is the key fix: schema loads AFTER server starts.
# The UI polls /db/schema-status to show progress (X/Y tables loaded).
# ══════════════════════════════════════════════════════════════════════════════

async def reload_schema():
    """Load schema from all active databases in the background with progress."""
    global schema_cache, schema_status, db_tables_cache
    from tools import list_tables, describe_table

    schema_status = {"loading": True, "current": 0, "total": 0, "message": "Discovering databases..."}
    log("SCHEMA", "Background schema loading started...")

    dbs = load_databases()
    active = [db for db in dbs if db.get("active")]

    if not active:
        schema_cache = "No active databases."
        schema_status = {"loading": False, "current": 0, "total": 0, "message": "Done"}
        return

    # Phase 1: list tables for each DB
    # NOTE: tool.invoke() is SYNCHRONOUS (blocks the event loop).
    # We wrap in asyncio.to_thread() so the server can still handle HTTP requests.
    db_tables_map = {}
    total_tables = 0
    parts = []

    for db in active:
        db_name = db["name"]
        schema_status["message"] = f"Listing tables in {db_name}..."
        try:
            tables_json = await asyncio.to_thread(list_tables.invoke, {"db_name": db_name})
            tables = json.loads(tables_json)
            if isinstance(tables, dict) and "error" in tables:
                parts.append(f"\n--- {db_name}: UNREACHABLE ---")
                continue
            db_tables_map[db_name] = tables
            db_tables_cache[db_name] = tables
            total_tables += len(tables)
        except Exception as e:
            parts.append(f"\n--- {db_name}: ERROR: {e} ---")
            log("SCHEMA", f"Failed to list tables for {db_name}: {e}")

    schema_status["total"] = total_tables
    schema_status["current"] = 0

    # Phase 2: describe each table (slow, tracked)
    # Each call runs in a thread so the event loop stays free for HTTP requests.
    described = 0
    for db_name, tables in db_tables_map.items():
        db_config = next(d for d in active if d["name"] == db_name)
        parts.append(f"\n--- {db_name} ({db_config['database']}@{db_config['host']}) ---")
        parts.append(f"Tables: {json.dumps(tables)}")

        for table in tables:
            schema_status["message"] = f"Describing {db_name}.{table}..."
            try:
                desc_json = await asyncio.to_thread(describe_table.invoke, {"db_name": db_name, "table_name": table})
                raw = json.loads(desc_json)
                if isinstance(raw, dict) and "columns" in raw:
                    cols = ", ".join(f"{c['column_name']}({c['data_type']})" for c in raw["columns"])
                    parts.append(f"  {table}: [{cols}]")
                else:
                    parts.append(f"  {table}: {desc_json}")
            except Exception as e:
                parts.append(f"  {table}: ERROR: {e}")
            described += 1
            schema_status["current"] = described

    schema_cache = "\n".join(parts)
    schema_status = {"loading": False, "current": total_tables, "total": total_tables, "message": "Done"}
    log("SCHEMA", f"Schema loading complete: {len(schema_cache)} chars, {total_tables} tables")


def start_schema_reload():
    """Start schema loading as a background asyncio task."""
    global _schema_task
    if _schema_task and not _schema_task.done():
        _schema_task.cancel()
    _schema_task = asyncio.create_task(reload_schema())


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

async def homepage(request: Request):
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


async def ask_endpoint(request: Request):
    """POST /ask — Non-streaming endpoint."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid request body"}, status_code=400)

    question = body.get("question", "").strip()
    session_id = body.get("session_id", None)
    log("HTTP", f"POST /ask  question='{question[:80]}'  session={session_id}")

    if not question:
        return JSONResponse({"error": "Empty question"}, status_code=400)

    if not session_id:
        session_id = f"s_{uuid.uuid4().hex[:8]}"

    # Ensure session exists in both stores
    session = get_session(session_id)
    if not session:
        session = {
            "id": session_id,
            "title": question[:40],
            "created_at": datetime.now().isoformat(),
            "messages": [],
        }
        update_session(session)
        vector_store.create_session(session_id, title=question[:40])

    try:
        answer = await ask_agent(agent, question, session_id=session_id)

        # Save to BOTH stores
        # 1. sessions.json (for UI persistence)
        session["messages"].append({"role": "user", "content": question})
        session["messages"].append({"role": "assistant", "content": answer})
        update_session(session)

        # 2. ChromaDB (for vector embeddings / semantic search)
        vector_store.add_message(session_id, "user", question)
        vector_store.add_message(session_id, "assistant", _clean_for_storage(answer))

        log("HTTP", f"POST /ask → 200 OK  answer_len={len(answer)}")
        return JSONResponse({"answer": answer, "session_id": session_id})

    except Exception as e:
        log("HTTP", f"POST /ask → 500 ERROR: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def ask_stream_endpoint(request: Request):
    """POST /ask/stream — Streaming endpoint using SSE with interactive questions."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid request body"}, status_code=400)

    question = body.get("question", "").strip()
    ui_session_id = body.get("session_id", None)
    log("HTTP", f"POST /ask/stream  question='{question[:80]}'  session={ui_session_id}")

    if not question:
        return JSONResponse({"error": "Empty question"}, status_code=400)

    # UI session for message history
    if not ui_session_id:
        ui_session_id = f"s_{uuid.uuid4().hex[:8]}"

    # Task session for interactive workflow
    task_session = session_manager.get_or_create_session(ui_session_id)
    
    # Check if already processing
    if not session_manager.start_task(task_session.session_id, task_id=str(uuid.uuid4())):
        return JSONResponse(
            {"error": "Session already processing a task. Cancel or wait for completion."},
            status_code=409
        )

    # Ensure UI session exists
    session = get_session(ui_session_id)
    if not session:
        session = {
            "id": ui_session_id,
            "title": question[:40],
            "created_at": datetime.now().isoformat(),
            "messages": [],
        }
        update_session(session)
        vector_store.create_session(ui_session_id, title=question[:40])

    async def event_generator():
        full_response = ""
        token_queue = asyncio.Queue()
        stream_done = asyncio.Event()
        stream_error = None

        async def _run_stream():
            nonlocal stream_error
            try:
                async for token in stream_agent(agent, question, session_id=ui_session_id):
                    await token_queue.put(("token", token))
                await token_queue.put(("stream_done", None))
            except Exception as e:
                stream_error = str(e)
                await token_queue.put(("error", str(e)))
            finally:
                stream_done.set()

        try:
            # Send session IDs
            yield f"data: {json.dumps({'type': 'session_id', 'content': ui_session_id})}\n\n"
            yield f"data: {json.dumps({'type': 'task_session_id', 'content': task_session.session_id})}\n\n"
            
            # Store context in task session for questions
            task_session.context['ui_session_id'] = ui_session_id
            task_session.context['question'] = question
            task_session.context['event_generator'] = 'active'
            set_task_session(task_session.session_id)

            # Run stream_agent in background so we can poll session state
            stream_task = asyncio.create_task(_run_stream())
            last_question_sent = None

            while True:
                # Check session state for pending questions
                current_session = session_manager.get_session(task_session.session_id)
                if current_session:
                    if current_session.status == TaskStatus.CANCELLED:
                        yield f"data: {json.dumps({'type': 'cancelled'})}\n\n"
                        stream_task.cancel()
                        return
                    if current_session.status == TaskStatus.AWAITING_INPUT and current_session.pending_question:
                        q = current_session.pending_question
                        q_key = q.get("text", "")
                        if q_key != last_question_sent:
                            last_question_sent = q_key
                            yield f"data: {json.dumps({'type': 'question', 'content': q})}\n\n"

                # Drain available tokens from queue (non-blocking)
                try:
                    msg_type, msg_data = await asyncio.wait_for(token_queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    if stream_done.is_set() and token_queue.empty():
                        break
                    continue

                if msg_type == "stream_done":
                    break
                elif msg_type == "error":
                    yield f"data: {json.dumps({'type': 'error', 'content': msg_data})}\n\n"
                    break
                elif msg_type == "token":
                    token = msg_data
                    full_response += token
                    if token.startswith("\n\n🔧"):
                        yield f"data: {json.dumps({'type': 'tool', 'content': token.strip()})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            # Save to BOTH stores after streaming
            print(f"\n{C.GREEN}  ✅ Stream done. Saving to dual storage...{C.RESET}")
            # 1. sessions.json
            session_now = get_session(ui_session_id) or session
            session_now["messages"].append({"role": "user", "content": question})
            session_now["messages"].append({"role": "assistant", "content": full_response})
            update_session(session_now)
            print(f"  ├─ sessions.json: ✅ saved")

            # 2. ChromaDB (cleaned + capped for future context retrieval)
            vector_store.add_message(ui_session_id, "user", question)
            vector_store.add_message(ui_session_id, "assistant", _clean_for_storage(full_response))

            # Mark task as completed
            session_manager.complete_task(task_session.session_id)
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            log("HTTP", f"POST /ask/stream → DONE  tokens_streamed={len(full_response)}")

        except Exception as e:
            log("HTTP", f"POST /ask/stream → ERROR: {e}")
            session_manager.set_error(task_session.session_id, str(e))
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Session Endpoints — Read from sessions.json ──────────────────────────────

async def sessions_list_endpoint(request: Request):
    """GET /sessions/list — Return all sessions for sidebar (from sessions.json)."""
    sessions = load_sessions()
    safe = [{"id": s["id"], "title": s["title"], "created_at": s.get("created_at", ""),
             "message_count": len(s.get("messages", []))} for s in sessions]
    return JSONResponse(safe)


async def sessions_load_endpoint(request: Request):
    """POST /sessions/load — Load a session with full messages (from sessions.json)."""
    body = await request.json()
    session = get_session(body.get("id"))
    if not session:
        return JSONResponse({"error": "Session not found."}, status_code=404)
    return JSONResponse({
        "id": session["id"],
        "title": session["title"],
        "messages": session.get("messages", []),
    })


async def sessions_delete_endpoint(request: Request):
    """POST /sessions/delete — Delete from both sessions.json and ChromaDB."""
    body = await request.json()
    session_id = body.get("id")

    # Delete from sessions.json
    sessions = load_sessions()
    new_list = [s for s in sessions if s["id"] != session_id]
    if len(new_list) == len(sessions):
        return JSONResponse({"error": "Session not found."}, status_code=404)
    save_sessions(new_list)

    # Delete from ChromaDB
    vector_store.delete_session(session_id)

    return JSONResponse({"success": True})


# ── Schema & DB Endpoints ─────────────────────────────────────────────────────

async def db_info_endpoint(request: Request):
    """Return database info for sidebar display."""
    databases = load_databases()
    result = []
    for db in databases:
        result.append({
            "name": db["name"],
            "database": db["database"],
            "host": db["host"],
            "active": db.get("active", False),
            "tables": db_tables_cache.get(db["name"], []),
        })
    return JSONResponse(result)


async def schema_status_endpoint(request: Request):
    """GET /db/schema-status — Return live schema loading progress."""
    return JSONResponse(schema_status)


# ── Vector Endpoints ──────────────────────────────────────────────────────────

async def vector_stats_endpoint(request: Request):
    """GET /vector/stats — Return vector store statistics."""
    stats = vector_store.get_stats()
    return JSONResponse(stats)


# ══════════════════════════════════════════════════════════════════════════════
# CHART CONFIG — LLM suggests best chart configuration for data
# ══════════════════════════════════════════════════════════════════════════════

CHART_PROMPT = """You are a data visualization expert. Given table data, suggest the best chart configuration.

INPUT: Column headers, data types, sample values, and the user's original question.
OUTPUT: A JSON object with chart configuration. ONLY return valid JSON, nothing else.

JSON format:
{
  "recommended_type": "bar|pie|line|doughnut",
  "reason": "Brief reason why this chart type is best",
  "xAxis": "column_name for labels/categories",
  "yAxis": ["column_name(s) for values"],
  "title": "Short chart title",
  "format": {
    "yPrefix": "$" or "" (prefix for Y values, e.g. $ for currency),
    "ySuffix": "%" or "" (suffix for Y values),
    "decimals": 0 or 2 (decimal places for values)
  },
  "colors": ["#667eea", "#764ba2", "#f093fb", "#4facfe", "#43e97b"],
  "sortBy": "value_desc|value_asc|label|none",
  "showLegend": true or false,
  "alternatives": ["other chart types that could work"]
}

Rules:
- For comparing categories (departments, names): use "bar"
- For proportions/percentages of a whole: use "pie" or "doughnut"
- For time series or trends: use "line"
- For large numbers: suggest appropriate format (M for millions, K for thousands)
- If only 1 numeric column: showLegend = false
- If more than 8 categories in pie: suggest bar instead
- ONLY return the JSON object, no explanation text.
"""

async def chart_config_endpoint(request: Request):
    """POST /chart-config — LLM suggests best chart config for given data."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid request"}, status_code=400)

    headers = body.get("headers", [])
    rows = body.get("rows", [])
    question = body.get("question", "")

    if not headers or not rows:
        return JSONResponse({"error": "No data provided"}, status_code=400)

    # Build a compact data summary for the LLM
    sample_rows = rows[:5]  # Only send first 5 rows to save tokens
    data_summary = f"Question: {question}\n\nColumns: {headers}\n\nSample data ({len(rows)} total rows):\n"
    for row in sample_rows:
        data_summary += f"  {row}\n"

    # Call LLM with chart prompt
    from langchain_openai import ChatOpenAI
    from agent import LLM_BASE_URL, LLM_API_KEY, MODEL

    llm = ChatOpenAI(
        model=MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.3,  # Slightly creative for viz suggestions
    )

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": CHART_PROMPT},
            {"role": "user", "content": data_summary},
        ])

        # Parse JSON from response
        import re
        text = response.content.strip()
        # Extract JSON if wrapped in markdown code block
        json_match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()

        config = json.loads(text)
        log("HTTP", f"POST /chart-config → 200 OK  type={config.get('recommended_type')}")
        return JSONResponse(config)

    except Exception as e:
        log("HTTP", f"POST /chart-config → ERROR: {e}")
        # Fallback: auto-detect without LLM
        return JSONResponse(_auto_detect_chart(headers, rows))


def _auto_detect_chart(headers, rows):
    """Fallback chart config when LLM is unavailable."""
    numeric_cols = []
    label_col = 0

    for i, h in enumerate(headers):
        try:
            vals = [float(str(row[i]).replace(',', '').replace('$', '').replace('%', ''))
                    for row in rows if row[i]]
            if len(vals) == len(rows):
                numeric_cols.append(h)
            elif not numeric_cols:
                label_col = i
        except (ValueError, IndexError):
            if not numeric_cols:
                label_col = i

    return {
        "recommended_type": "bar",
        "reason": "Auto-detected (LLM unavailable)",
        "xAxis": headers[label_col] if headers else "",
        "yAxis": numeric_cols[:3] if numeric_cols else [headers[-1]] if headers else [],
        "title": "Data Visualization",
        "format": {"yPrefix": "", "ySuffix": "", "decimals": 0},
        "colors": ["#667eea", "#764ba2", "#f093fb", "#4facfe", "#43e97b"],
        "sortBy": "none",
        "showLegend": len(numeric_cols) > 1,
        "alternatives": ["pie", "line"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# LIFESPAN — Server starts IMMEDIATELY, schema loads in background
# ══════════════════════════════════════════════════════════════════════════════

from contextlib import asynccontextmanager

# Import approval workflow manager
from approval_workflow import ApprovalWorkflowManager, BackgroundMonitor

# Global background monitor
background_monitor = None

@asynccontextmanager
async def lifespan(app):
    global agent, background_monitor

    log("APP", LOG_SEP)
    log("APP", "Starting AI DB Agent (Vector Edition)...")
    log("APP", f"ChromaDB: {vector_store.CHROMA_PERSIST_DIR}")
    log("APP", f"Embedding: {vector_store.EMBED_MODEL}")

    # Print stats
    stats = vector_store.get_stats()
    log("APP", f"Vector store: {stats['total_messages']} messages, {stats['total_sessions']} sessions")

    # Load sessions count
    sessions = load_sessions()
    log("APP", f"Sessions (JSON): {len(sessions)} sessions")

    # Create the agent (fast — no DB calls)
    agent = create_agent()
    log("APP", "Agent created successfully")

    # Start schema loading in BACKGROUND (doesn't block server)
    start_schema_reload()
    log("APP", "Schema loading started in background...")

    # Start session cleanup task
    asyncio.create_task(session_manager.start_cleanup_task(interval_seconds=300))
    log("APP", "✅ Session cleanup started (runs every 5 minutes)")

    # Start background approval monitoring (checks every 2 minutes)
    try:
        workflow_manager = ApprovalWorkflowManager()
        background_monitor = BackgroundMonitor(
            workflow_manager,
            check_interval=120  # 2 minutes = 120 seconds
        )
        asyncio.create_task(background_monitor.start())
        log("APP", "✅ Approval monitoring started (checks every 2 minutes)")
    except Exception as e:
        log("APP", f"⚠️  Approval monitoring not available: {e}")

    # Start WatchFlow scheduler
    global watchflow_scheduler
    try:
        from integrations.watchflow.scheduler import WatchFlowScheduler
        from integrations.watchflow.executor import WorkflowExecutor
        wf_executor = WorkflowExecutor(agent, ask_agent)
        watchflow_scheduler = WatchFlowScheduler(wf_executor)
        asyncio.create_task(watchflow_scheduler.start())
        log("APP", "✅ WatchFlow scheduler started")
    except Exception as e:
        log("APP", f"⚠️  WatchFlow not available: {e}")

    log("APP", "")
    log("APP", "╔══════════════════════════════════════════════════╗")
    log("APP", "║  Server ready: http://localhost:3002             ║")
    log("APP", "║  Vector Edition (ChromaDB + nomic-embed-text)    ║")
    log("APP", "║  Schema loading in background...                 ║")
    log("APP", "║  Approval monitoring active (every 2 min)        ║")
    log("APP", "╚══════════════════════════════════════════════════╝")
    log("APP", "")

    yield

    log("APP", "Shutting down...")
    if background_monitor:
        background_monitor.stop()


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE WORKFLOW ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

async def cancel_task_endpoint(request: Request):
    """POST /task/cancel — Cancel active task"""
    try:
        body = await request.json()
        session_id = body.get("session_id")
        
        if not session_id:
            return JSONResponse({"error": "Missing session_id"}, status_code=400)
        
        success = session_manager.cancel_task(session_id)
        
        if success:
            log("HTTP", f"POST /task/cancel → Task cancelled for session {session_id}")
            return JSONResponse({"status": "cancelled", "session_id": session_id})
        else:
            return JSONResponse(
                {"error": "No active task to cancel"},
                status_code=404
            )
    
    except Exception as e:
        log("HTTP", f"POST /task/cancel → ERROR: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def respond_to_question_endpoint(request: Request):
    """POST /task/respond — Provide response to agent's question"""
    try:
        body = await request.json()
        session_id = body.get("session_id")
        response = body.get("response")
        
        if not session_id or not response:
            return JSONResponse(
                {"error": "Missing session_id or response"},
                status_code=400
            )
        
        success = session_manager.provide_response(session_id, response)
        
        if success:
            log("HTTP", f"POST /task/respond → Response provided: {response}")
            return JSONResponse({"status": "response_received", "response": response})
        else:
            return JSONResponse(
                {"error": "No pending question or invalid session"},
                status_code=404
            )
    
    except Exception as e:
        log("HTTP", f"POST /task/respond → ERROR: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def task_status_endpoint(request: Request):
    """GET /task/status?session_id=XXX — Get current task status. Without session_id, lists all active tasks."""
    session_id = request.query_params.get("session_id")
    
    if not session_id:
        # List all active/pending sessions
        active = []
        with session_manager._lock:
            for sid, session in session_manager.sessions.items():
                active.append({
                    "session_id": sid,
                    "status": session.status.value,
                    "has_question": session.pending_question is not None,
                })
        return JSONResponse({"sessions": active})
    
    session = session_manager.get_session(session_id)
    
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    
    return JSONResponse({
        "session_id": session_id,
        "status": session.status.value,
        "task_id": session.current_task_id,
        "has_question": session.pending_question is not None,
        "question": session.pending_question,
        "error": session.error_message
    })


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# WATCHFLOW ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

watchflow_scheduler = None

async def watchflow_list_endpoint(request: Request):
    """GET /watchflows/list — List all workflow definitions."""
    from integrations.watchflow.definitions import load_workflows
    return JSONResponse(load_workflows())

async def watchflow_create_endpoint(request: Request):
    """POST /watchflows/create — Create or update a workflow."""
    from integrations.watchflow.definitions import save_workflow
    body = await request.json()
    workflow = save_workflow(body)
    # Re-schedule if enabled
    if watchflow_scheduler and workflow.get("enabled"):
        watchflow_scheduler.schedule(workflow)
    return JSONResponse({"status": "saved", "workflow": workflow})

async def watchflow_delete_endpoint(request: Request):
    """POST /watchflows/delete — Delete a workflow."""
    from integrations.watchflow.definitions import delete_workflow
    body = await request.json()
    wf_id = body.get("id")
    if not wf_id:
        return JSONResponse({"error": "Missing 'id'"}, status_code=400)
    if watchflow_scheduler:
        watchflow_scheduler.unschedule(wf_id)
    deleted = delete_workflow(wf_id)
    return JSONResponse({"status": "deleted" if deleted else "not_found"})

async def watchflow_trigger_endpoint(request: Request):
    """POST /watchflows/trigger — Manually trigger a workflow now."""
    body = await request.json()
    wf_id = body.get("id")
    if not wf_id:
        return JSONResponse({"error": "Missing 'id'"}, status_code=400)
    if not watchflow_scheduler:
        return JSONResponse({"error": "Scheduler not initialized"}, status_code=500)
    result = await watchflow_scheduler.trigger_now(wf_id)
    return JSONResponse(result)

async def watchflow_toggle_endpoint(request: Request):
    """POST /watchflows/toggle — Enable/disable a workflow."""
    from integrations.watchflow.definitions import get_workflow, save_workflow
    body = await request.json()
    wf_id = body.get("id")
    workflow = get_workflow(wf_id)
    if not workflow:
        return JSONResponse({"error": "Workflow not found"}, status_code=404)
    workflow["enabled"] = not workflow.get("enabled", False)
    save_workflow(workflow)
    if watchflow_scheduler:
        if workflow["enabled"]:
            watchflow_scheduler.schedule(workflow)
        else:
            watchflow_scheduler.unschedule(wf_id)
    return JSONResponse({"status": "toggled", "enabled": workflow["enabled"]})

async def watchflow_status_endpoint(request: Request):
    """GET /watchflows/status — Scheduler status."""
    if not watchflow_scheduler:
        return JSONResponse({"running": False, "active_workflows": [], "count": 0})
    return JSONResponse(watchflow_scheduler.get_status())


# ══════════════════════════════════════════════════════════════════════════════
# APP SETUP
# ══════════════════════════════════════════════════════════════════════════════

routes = [
    Route("/", homepage),
    Route("/ask", ask_endpoint, methods=["POST"]),
    Route("/ask/stream", ask_stream_endpoint, methods=["POST"]),
    Route("/task/cancel", cancel_task_endpoint, methods=["POST"]),
    Route("/task/respond", respond_to_question_endpoint, methods=["POST"]),
    Route("/task/status", task_status_endpoint, methods=["GET"]),
    Route("/chart-config", chart_config_endpoint, methods=["POST"]),
    Route("/sessions/list", sessions_list_endpoint, methods=["GET"]),
    Route("/sessions/load", sessions_load_endpoint, methods=["POST"]),
    Route("/sessions/delete", sessions_delete_endpoint, methods=["POST"]),
    Route("/db/info", db_info_endpoint, methods=["GET"]),
    Route("/db/schema-status", schema_status_endpoint, methods=["GET"]),
    Route("/vector/stats", vector_stats_endpoint, methods=["GET"]),
    Route("/watchflows/list", watchflow_list_endpoint, methods=["GET"]),
    Route("/watchflows/create", watchflow_create_endpoint, methods=["POST"]),
    Route("/watchflows/delete", watchflow_delete_endpoint, methods=["POST"]),
    Route("/watchflows/trigger", watchflow_trigger_endpoint, methods=["POST"]),
    Route("/watchflows/toggle", watchflow_toggle_endpoint, methods=["POST"]),
    Route("/watchflows/status", watchflow_status_endpoint, methods=["GET"]),
    Mount("/static", StaticFiles(directory=STATIC_DIR), name="static"),
]

app = Starlette(routes=routes, lifespan=lifespan)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3002)
