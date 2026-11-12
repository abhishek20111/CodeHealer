"""
══════════════════════════════════════════════════════════════════════════════════
LANGGRAPH REACT AGENT — Vector Memory + GitHub Copilot LLM Edition
══════════════════════════════════════════════════════════════════════════════════

DIFFERENCE FROM ai_db_agent_langchain/agent.py:
─────────────────────────────────────────────────────────────────────────────────

OLD (JSON + sliding window):
    1. History stored in sessions.json as plain text
    2. build_history_context() → keeps last 6 messages + summarizes older ones
    3. Summary is EXTRACTIVE (just truncates text, no semantic understanding)
    4. If user asks about something from 20 messages ago → LOST forever
    5. Every message sent to LLM whether relevant or not

NEW (ChromaDB + semantic retrieval):
    1. History stored in ChromaDB as VECTOR EMBEDDINGS
    2. get_relevant_history() → SEMANTIC SEARCH finds relevant past messages
    3. Retrieval is SEMANTIC (finds by meaning, not position)
    4. If user asks about something from 20 messages ago → FOUND by similarity
    5. Only RELEVANT messages sent to LLM → better answers, fewer tokens

ARCHITECTURE:
─────────────────────────────────────────────────────────────────────────────────

    ┌──────────── Agent Invocation Flow ─────────────┐
    │                                                 │
    │  User: "how many employees are there?"          │
    │       │                                         │
    │       ▼                                         │
    │  vector_store.get_relevant_history(query)       │
    │       │                                         │
    │       ├─→ Embed query with nomic-embed-text     │
    │       ├─→ Cosine similarity search in ChromaDB  │
    │       ├─→ Top-K relevant + Last-N recent msgs   │
    │       │                                         │
    │       ▼                                         │
    │  Build message list:                            │
    │  [SystemMessage] + [relevant_history] + [query] │
    │       │                                         │
    │       ▼                                         │
    │  agent.ainvoke({"messages": [...]})             │
    │       │                                         │
    │       ▼                                         │
    │  LLM + Tool calls → Response                   │
    │       │                                         │
    │       ▼                                         │
    │  vector_store.add_message(response)             │
    │  (embedded + stored for future retrieval)       │
    └─────────────────────────────────────────────────┘

INTERVIEW QUESTIONS:
─────────────────────────────────────────────────────────────────────────────────

Q: "Why embed conversation history instead of just storing it?"
A: "Embedding enables RETRIEVAL BY MEANING. When the user asks about employees,
   the system finds past messages about employees regardless of when they occurred.
   Plain storage requires sequential scanning and loses old context."

Q: "What's the latency impact of vector retrieval vs. JSON?"
A: "JSON sliding window: ~0ms (just array slice).
   Vector retrieval: ~100-200ms (embed query + ANN search).
   But the LLM gets BETTER context, so it often needs fewer tool calls,
   which saves 3-10 seconds. Net result: faster overall."

Q: "Can this approach work across sessions?"
A: "Yes! Since all messages from all sessions are in one ChromaDB collection,
   we can search across sessions. If the user asked 'how many departments' last
   week, that answer is retrievable in today's new session."

DIFFERENCE FROM ai_db_agent_langchain_vector/agent.py:
─────────────────────────────────────────────────────────────────────────────────

OLD: ChatOllama (local qwen2.5-coder:7b, 7B params, often outputs tool calls as text)
NEW: ChatOpenAI via GitHub Models API (GPT-4o-mini, native tool calling, much smarter)

WHY GITHUB MODELS API:
    1. FREE tier with rate limits (good enough for dev/testing)
    2. OpenAI-compatible endpoint → just swap ChatOllama → ChatOpenAI
    3. GPT-4o-mini has native function calling → no more _looks_like_tool_call() hacks
    4. Better instruction following → fewer iterations, faster responses
    5. Embeddings stay LOCAL (nomic-embed-text via Ollama) → no API cost for storage

SETUP:
    1. Go to https://github.com/settings/tokens → Generate PAT (classic)
    2. Set env: export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"
    3. Run: python web_server.py

INTERVIEW:
    Q: "Why not use Copilot directly?"
    A: "VS Code Copilot's LLM is locked inside the extension — no external API.
       GitHub Models API provides the same GPT-4o/4o-mini models through an
       OpenAI-compatible endpoint at models.inference.ai.azure.com. Since
       LangChain's ChatOpenAI supports any OpenAI-compatible endpoint, it's
       a one-line config change — no architecture changes needed."

══════════════════════════════════════════════════════════════════════════════════
"""

import json
import os
from datetime import datetime
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent as _create_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from tools import ALL_TOOLS, load_databases, log, LOG_SEP
from integrations.github import ALL_GITHUB_TOOLS
from vector_store import get_relevant_history, C

# ── OBS Integration ──────────────────────────────────────────────────────────
try:
    from integrations.obs.tools import create_obs_tools
    OBS_TOOLS = create_obs_tools()
    log("OBS", "✅ OBS tools loaded successfully")
except Exception as e:
    OBS_TOOLS = []
    log("OBS", f"⚠️  OBS tools not available: {e}")
    log("OBS", "   Set OBS_APP_ID, OBS_USERNAME, OBS_PASSWORD in .env to enable OBS integration")

# ── Slack Integration ────────────────────────────────────────────────────────
try:
    from integrations.slack.tools import create_slack_tools
    from integrations.slack.workflow_tools import WORKFLOW_TOOLS
    from integrations.slack.enhanced_workflow_tools import ENHANCED_WORKFLOW_TOOLS
    SLACK_TOOLS = create_slack_tools() + WORKFLOW_TOOLS + ENHANCED_WORKFLOW_TOOLS
    log("SLACK", "✅ Slack tools loaded successfully")
except Exception as e:
    SLACK_TOOLS = []
    log("SLACK", f"⚠️  Slack tools not available: {e}")
    log("SLACK", "   Set SLACK_TOKEN in .env to enable Slack integration")

# ── VulnScan Integration ─────────────────────────────────────────────────
try:
    from integrations.vulnscan.tools import create_vulnscan_tools
    VulnScan_TOOLS = create_vulnscan_tools()
    log("VulnScan", "✅ VulnScan tools loaded successfully")
except Exception as e:
    VulnScan_TOOLS = []
    log("VulnScan", f"⚠️  VulnScan tools not available: {e}")

# ── QA Automation Integration ────────────────────────────────────────────────
try:
    from integrations.qa.tools import create_qa_tools
    QA_TOOLS = create_qa_tools()
    log("QA", "✅ QA automation tools loaded successfully")
except Exception as e:
    QA_TOOLS = []
    log("QA", f"⚠️  QA automation tools not available: {e}")

# ── Interactive UI Tools ──────────────────────────────────────────────────────
from integrations.ui.interactive_tools import INTERACTIVE_WORKFLOW_TOOLS, set_task_session

# ── Configuration ────────────────────────────────────────────────────────────
# LLM Backend: Copilot Proxy (default), GitHub Models, or Codex CLI Bridge
#
# COPILOT PROXY (DEFAULT — recommended):
#   1. Open copilot_proxy/ folder in a separate VS Code window
#   2. Press F5 to launch the extension (Extension Development Host)
#   3. Proxy starts on http://localhost:3100/v1
#   4. No token needed — uses your existing Copilot subscription
#
# GITHUB MODELS API (alternative — requires GitHub Enterprise to allow it):
#   1. Set LLM_BACKEND=github-models in .env
#   2. Set GITHUB_TOKEN=ghp_xxx in .env
#
# INTERVIEW: "I built a VS Code extension that exposes Copilot's LLM as an
# OpenAI-compatible local API. The Python agent connects to localhost:3100
# as if it were OpenAI — but requests go through Copilot instead. This
# bypasses corporate network blocks since everything runs locally."

LLM_BACKEND = os.environ.get("LLM_BACKEND", "copilot-proxy")

# Supported model strings (override via LLM_MODEL env var):
#   gpt-5           — flagship, best reasoning + tool use
#   gpt-5-mini      — faster / cheaper, good enough for most tasks
#   gpt-5.4         — latest patch release
#   GPT-5.3-Codex   — code-tuned variant on the Copilot proxy
if LLM_BACKEND == "github-models":
    MODEL = os.environ.get("LLM_MODEL", "gpt-5-mini")
    LLM_BASE_URL = "https://models.inference.ai.azure.com"
    LLM_API_KEY = os.environ.get("GITHUB_TOKEN", "")
elif LLM_BACKEND == "codex-bridge":
    MODEL = os.environ.get("CODEX_BRIDGE_MODEL", "codex-cli")
    LLM_BASE_URL = os.environ.get("CODEX_BRIDGE_URL", "http://127.0.0.1:3100/v1").rstrip("/")
    LLM_API_KEY = os.environ.get("CODEX_BRIDGE_TOKEN", "")
else:
    # Copilot Proxy — connects to the VS Code extension on localhost
    MODEL = os.environ.get("LLM_MODEL", "gpt-5")
    LLM_BASE_URL = "http://localhost:3100/v1"
    LLM_API_KEY = "copilot-proxy"  # Dummy key — proxy handles auth via VS Code

# ── Context Limits ───────────────────────────────────────────────────────────
# MAX_CONTEXT_CHARS: Hard cap on total context sent to LLM.
#   GPT-5 family supports a 128K+ token context window; we keep 60000 chars
#   as a reasonable cap to control API cost and keep responses focused.
#
# INTERVIEW: "Even with a 128K context window, we still cap context to control
# API costs and keep responses focused. More context != better answers."

MAX_CONTEXT_CHARS = 60000
MAX_SUPERVISOR_INTERNAL = 16000  # Char cap for supervisor's internal ReAct loop
MAX_STORED_RESPONSE = 4000   # Cap assistant responses stored in ChromaDB
MAX_CONTEXT_MSG_CHARS = 1500 # Per-message cap when loading history context


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
#
# Same as original — the prompt doesn't change between JSON and vector versions.
# The agent doesn't know or care HOW its context was assembled.
#
# INTERVIEW: "The system prompt is agnostic to the retrieval strategy. Whether
# context comes from a sliding window or semantic search, the LLM just sees
# messages. This separation of concerns is key to good architecture."
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert database assistant that queries real MySQL databases using tools.

CONVERSATION AWARENESS:
- Read the full conversation history above before responding.
- Remember information provided by the user (database names, table names, filters).
- When users provide missing information, use it with their previous request.

RULES:
- For data queries: ALWAYS use run_query tool. Never guess or explain theoretically.
- Only SELECT queries. Never modify data.
- If user mentions both database and table name, go DIRECTLY to run_query (skip list_databases, list_tables).
- Use exact table/column names from describe_table results.
- Present results in clear, human-readable format.
- For greetings/non-database questions, respond normally without tools.

FUZZY TABLE/COLUMN MATCHING:
- If the user's table or column name has typos or is approximate, do NOT fail.
- First call list_tables to see all available tables.
- Match the user's intent to the closest real table name (e.g., "employeee" → "employees", "dept" → "departments", "proj" → "projects").
- Same for column names: use describe_table to find the real column, then use it.
- NEVER say "table not found" without first checking all available tables.

QUERY DECOMPOSITION (for complex questions):
- Break complex questions into sub-questions internally:
  Example: "Which department has the most employees working on the highest-budget project?"
  → Sub-Q1: What is the highest-budget project? (run_query)
  → Sub-Q2: Which department does it belong to? (from result)
  → Sub-Q3: How many employees in that department? (run_query)
  → Combine results into final answer.
- For comparison questions, run separate queries and compare results.
- For "top N" or ranking questions, use ORDER BY + LIMIT in SQL.

SELF-VERIFICATION:
- After getting query results, verify they make sense:
  - Does the count match expectations?
  - Are there NULL values that might indicate a wrong JOIN?
  - Is the result empty? If so, try a broader query or check column values.
- If results seem wrong, explain what you found and try an alternative approach.
- Run a validation query if needed (e.g., SELECT COUNT(*) to verify).

CONFIDENCE & HONESTY:
- If a query returns 0 rows after 2 attempts: Say "I couldn't find data matching your criteria. Here's what I tried: [explain]. Would you like me to try a different approach?"
- If the table/column doesn't exist in any database: Say "I checked all available tables but couldn't find anything matching '[name]'. Available tables are: [list them]."
- If the question is ambiguous: Ask a clarifying question instead of guessing.
- NEVER make up data. If the query result is empty or the tool errors, be transparent about it.
- Rate your confidence: If you had to make assumptions, mention them briefly.

RESULT REUSE (from conversation history):
- If the user asks a follow-up that uses data from a previous answer, DO NOT re-query unless they ask for fresh data.
- Example: If you already showed "Marketing has 5 employees" and user asks "what percentage is that?", use the stored number (5) and total from context.
- For "drill-down" questions (user asks about a specific item from previous results), use the previous result to construct a targeted query.

OUTPUT FORMAT:
- For tabular data (3+ rows): ALWAYS format as a Markdown table:
  | Column 1 | Column 2 | Column 3 |
  |----------|----------|----------|
  | value    | value    | value    |
- For single values: respond in a clear sentence.
- For summaries: use bullet points or numbered lists.
- Use **bold** for key values and important findings.
- Add a brief summary sentence after tables (e.g., "Total: 5 departments, Marketing has the most employees.").

WORKFLOW:
1. User asks a question → Check if you already know the schema from context.
2. If table/column name unclear → Use list_tables + describe_table to find the right match.
3. Break complex questions into sub-queries if needed.
4. Write SQL with proper JOINs, GROUP BY, ORDER BY as needed.
5. Execute with run_query → Verify the result makes sense.
6. If result looks wrong → Try alternative approach.
7. Present final answer with proper Markdown formatting.

IMPORTANT: Use tools directly. Do NOT output JSON as text.
"""

CODEX_BRIDGE_PROMPTS = {
    "database": SYSTEM_PROMPT + """

CODEX BRIDGE MODE:
- Operate read-only. Do not change database data or workspace files.
- Use only read tools exposed through your configured MCP servers.
- If a required database tool is unavailable, state that clearly instead of guessing.
""",
    "code": """You are a read-only code and GitHub Enterprise assistant.
Inspect local files or approved enterprise-search MCP sources when useful.
Do not create branches, commits, pull requests, comments, or file changes.
Present code findings with paths and concise explanations. Never claim to have inspected data you could not access.""",
    "logs": """You are a read-only observability assistant.
Use approved enterprise-search tools to find actual log or incident evidence when available.
Never fabricate logs, timestamps, trace IDs, errors, or tool results.
If OBS-specific access is unavailable, state that limitation clearly.""",
    "slack": """You are a read-only enterprise communication assistant.
You may search approved Slack sources and summarize messages with useful channel and date context.
Do not post, edit, delete, or react to Slack messages. Never fabricate search results.""",
    "general": """You are a read-only engineering assistant.
Answer the user's question using the supplied conversation and approved local or MCP context.
Do not modify files or external systems. State access limitations rather than guessing.""",
}


# ══════════════════════════════════════════════════════════════════════════════
# AGENT SETUP
#
# Same create_agent() call as original. The agent itself doesn't change —
# only HOW we build the messages array before calling it changes.
#
# DIFFERENCE:
#   OLD: No checkpointer, manually managed sliding window
#   NEW: No checkpointer, manually managed VECTOR retrieval
#   (Both skip checkpointer because we want custom context assembly)
# ══════════════════════════════════════════════════════════════════════════════


def create_agent():
    """
    Create a LangGraph ReAct agent.
    
    Supports two backends:
    - copilot-proxy: VS Code extension on localhost:3100 (default)
    - github-models: GitHub Models API (needs GITHUB_TOKEN)
    - codex-bridge: Read-only Codex CLI bridge; Codex owns tool execution
    """
    if LLM_BACKEND == "github-models" and not LLM_API_KEY:
        print(f"\n{C.RED}{'='*60}{C.RESET}")
        print(f"{C.RED}{C.BOLD}  ❌ GITHUB_TOKEN not set!{C.RESET}")
        print(f"{C.RED}{'='*60}{C.RESET}")
        print(f"  Set GITHUB_TOKEN in .env or use LLM_BACKEND=copilot-proxy")
        print(f"{C.RED}{'='*60}{C.RESET}\n")
        raise ValueError("GITHUB_TOKEN required for github-models backend")

    if LLM_BACKEND == "codex-bridge":
        if not LLM_API_KEY:
            raise ValueError("CODEX_BRIDGE_TOKEN required for codex-bridge backend")
        print(f"\n{C.GREEN}{'='*60}{C.RESET}")
        print(f"{C.GREEN}{C.BOLD}  🤖 CREATING CODEX BRIDGE AGENT{C.RESET}")
        print(f"{C.GREEN}{'='*60}{C.RESET}")
        print(f"  Backend      : {C.CYAN}Codex CLI Bridge (read-only){C.RESET}")
        print(f"  Model        : {C.CYAN}{MODEL}{C.RESET}")
        print(f"  API Endpoint : {LLM_BASE_URL}")
        print(f"  Max Chars    : {MAX_CONTEXT_CHARS} (safety cap)")
        print(f"  Tool Calling : {C.CYAN}Owned by Codex/MCP{C.RESET}")
        print(f"{C.GREEN}  ✅ Codex bridge backend ready!{C.RESET}")
        print(f"{C.GREEN}{'='*60}{C.RESET}\n")
        return {"_backend": "codex-bridge"}

    backend_label = "Copilot Proxy (localhost)" if LLM_BACKEND == "copilot-proxy" else "GitHub Models API"
    
    print(f"\n{C.GREEN}{'='*60}{C.RESET}")
    print(f"{C.GREEN}{C.BOLD}  🤖 CREATING LANGGRAPH REACT AGENT{C.RESET}")
    print(f"{C.GREEN}{'='*60}{C.RESET}")
    print(f"  Backend      : {C.CYAN}{backend_label}{C.RESET}")
    print(f"  LLM Model    : {C.CYAN}{MODEL}{C.RESET}")
    print(f"  API Endpoint : {LLM_BASE_URL}")
    print(f"  Temperature  : 0 (deterministic)")
    print(f"  Max Chars    : {MAX_CONTEXT_CHARS} (safety cap)")
    print(f"  Tool Calling : {C.GREEN}NATIVE{C.RESET}")

    # Configure httpx to not use proxy - ignore environment variables
    import httpx
    
    # Create httpx client that ignores HTTP_PROXY/HTTPS_PROXY env vars
    http_client = httpx.Client(trust_env=False, transport=httpx.HTTPTransport(retries=3))
    
    llm = ChatOpenAI(
        model=MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0,
        http_client=http_client,
    )

    # ═══════════════════════════════════════════════════════════════
    # 1️⃣ DATABASE AGENT — SQL queries and data analysis
    # ═══════════════════════════════════════════════════════════════
    db_prompt = """You are a database assistant that helps query and analyze SQL databases.
You can list databases, list tables, describe schemas, and execute SELECT queries.
Always explain your query logic and present results in clear tables or charts.
For large result sets, show a preview and summary statistics."""

    db_agent = _create_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=db_prompt,
    )
    print(f"  1️⃣ DB Agent  ({len(ALL_TOOLS)} tools) : {C.CYAN}{[t.name for t in ALL_TOOLS]}{C.RESET}")

    # ═══════════════════════════════════════════════════════════════
    # 2️⃣ GITHUB AGENT — Code operations and repository management
    # ═══════════════════════════════════════════════════════════════
    github_prompt = """You are a code assistant that reads and manages GitHub Enterprise repositories.
You can read files, search code, list directories, create branches, commit fixes, and open pull requests.
When creating branches, use prefix 'ai-agent/' (e.g., 'ai-agent/fix-null-check').
When committing, prefix messages with '🤖'.
For LARGE files (>5K chars), use patch_file instead of commit_file — it reads the file, applies your find/replace, and commits without needing the full content.
Present code in markdown code blocks with language syntax highlighting.
For large files, summarize the key parts rather than showing everything.

BRANCH POLICY (very important — read before every tool call):
- If the user did NOT tell you a branch name, DO NOT invent one (no `self-serve`,
  no `main`, no guesses). Simply omit the `branch` argument. The tool will use
  the branch configured in `repos.json` (or the repo's default branch from the
  GitHub API) — currently `production-api` for the SampleApp fork.
- Only pass an explicit branch when: (a) the user named it, (b) you already
  confirmed it exists via `list_pull_requests` or a previous successful call.
- If a tool call fails with 404 or 422 mentioning a branch or ref, DO NOT retry
  with the same branch or a random guess. Retry ONCE with `branch` omitted
  (default resolution) — if that also fails, stop and report the error.

EFFICIENCY RULES (hard budget — the tools enforce this too):
- BUDGET PER TASK: at most 8 tool calls total, at most 2 reads per file,
  at most 3 code searches (search_github_code) before you MUST commit to
  an action (patch_file / commit_file / create_branch / create_pull_request)
  or return your findings as text.
- To find where a package/class is used: use search_github_code — never a
  recursive directory listing.
- Do NOT read a file just to "double-check". If the first read already
  showed the method or class you care about, PATCH it, don't re-read it.
- If you already know the exact literal (error code, function name,
  import string), use search_github_code once — do not fire the same
  query with different phrasings.
- STOP RULE: after 2 reads + 3 searches, either:
    (a) call patch_file / create_branch / create_pull_request, OR
    (b) return findings as text.
  Any additional read after the budget will be REJECTED by the tool
  with a warning message.
- If a repo or path gives 404/422 error, do NOT retry with variations —
  return what you have.

REPO-MAP FIRST (important):
- Before opening ANY file on a repo you have not touched recently, call list_repo_map
  once. It returns one line per source file with language + top symbols + a summary.
- Use that map to pick the smallest set of files worth reading. Then use search_github_code
  to jump to exact matches, and read_github_file with start_line/end_line for the slice.
- If list_repo_map is passed a path_filter (e.g. 'controller' or 'exception'), it narrows
  the listing — much cheaper than crawling directories.
- The sidecar is cached on disk and only rebuilt when the branch head SHA changes, so
  calling it repeatedly is free after the first hit.

SEMANTIC CODE SEARCH (best for concept-level queries):
- Use search_repo_semantic for questions like "where is the authentication middleware"
  or "code that handles the APP-1000 error" — it embeds your query and returns the
  top matching code chunks with paths and line ranges, so you jump straight to the
  right region.
- If a repo has not been indexed yet, search_repo_semantic returns an empty list.
  Call index_repo_semantic(repo, branch) once first (takes ~30-60 s the first time,
  a few seconds on refresh because file SHAs are checked against ChromaDB).
- Prefer search_repo_semantic over search_github_code when the question is fuzzy or
  concept-shaped. Use search_github_code when you already know the exact literal
  (an error code, a function name, an import string).
- After a match you can pass its (path, start_line, end_line) directly to
  read_github_file to see the surrounding code.

LINE-RANGE READING (important — keeps context small):
- For any file larger than ~400 lines, ALWAYS call read_github_file with start_line and end_line arguments.
- Typical workflow: first call search_github_code to locate the interesting symbol → note the line number → then
  read only that window (e.g. start_line=match-20, end_line=match+40).
- Default window size 40-60 lines is enough for reading a method or exception handler.
- Never request more than 800 lines in one call — the tool will emit a warning and the context will bloat.
- If you need to look at another region of the same file, call read_github_file again with the new range;
  the file body is cached so no extra network hop happens.

IMPORTANT: Return your findings as detailed text. The supervisor will handle user interaction."""

    def _trim_github_context(state):
        """Pre-model hook for GitHub agent: prevent loops by building a file registry from trimmed messages."""
        msgs = state["messages"]
        total = sum(len(str(getattr(m, 'content', '') or '')) for m in msgs)
        if total <= 24000 or len(msgs) <= 8:
            return {"messages": msgs}

        # Build file registry BEFORE trimming — scan all ToolMessages for file reads
        file_registry = []
        for m in msgs:
            if not isinstance(m, ToolMessage):
                continue
            content_str = str(getattr(m, 'content', '') or '')
            if '"file"' not in content_str and '"content"' not in content_str:
                continue
            try:
                data = json.loads(content_str)
                if "file" in data and "content" in data:
                    snippet = data["content"][:300].replace('\n', ' ')
                    file_registry.append(f"• {data['file']} ({data.get('size',0)}B): {snippet}")
            except (json.JSONDecodeError, TypeError):
                pass

        first = msgs[0]
        rest = msgs[1:]
        budget = 20000 - len(str(first.content or ''))
        kept = []
        chars = 0
        for m in reversed(rest):
            mc = len(str(getattr(m, 'content', '') or ''))
            if chars + mc > budget and len(kept) >= 4:
                break
            kept.insert(0, m)
            chars += mc
        while kept and isinstance(kept[0], ToolMessage):
            kept.pop(0)
        dropped = len(rest) - len(kept)
        if dropped <= 0:
            return {"messages": msgs}

        # Inject file registry into bridge message
        registry_text = "\n".join(file_registry[-6:]) if file_registry else "None"
        print(f"{C.YELLOW}  ✂️  github pre_model_hook: trimmed {dropped} msgs, registry: {len(file_registry)} files{C.RESET}")
        bridge = HumanMessage(content=f"""[{dropped} earlier steps trimmed to save context.

FILES ALREADY READ (do NOT re-read — use the summaries below):
{registry_text}

RULES: Do NOT call read_github_file on any file listed above. You already have the information. Proceed to your next action (create branch, patch, or return findings).]""")
        return {"messages": [first, bridge] + kept}

    github_agent = _create_agent(
        model=llm,
        tools=ALL_GITHUB_TOOLS,
        prompt=github_prompt,
        pre_model_hook=_trim_github_context,
    )
    print(f"  2️⃣ GitHub Agent ({len(ALL_GITHUB_TOOLS)} tools) : {C.CYAN}{[t.name for t in ALL_GITHUB_TOOLS]}{C.RESET}")

    # ═══════════════════════════════════════════════════════════════
    # 3️⃣ OBS AGENT — log observability queries
    # ═══════════════════════════════════════════════════════════════
    obs_prompt = """You are an observability assistant for OBS (log observability).

CRITICAL: You MUST use the OBS tools to query logs. NEVER make up results or claim you checked logs without actually calling the tools.

AVAILABLE TOOLS:
1. search_obs_logs - General log search with custom query
2. get_obs_errors - Get ERROR level logs for an application
3. count_obs_logs - Count logs matching criteria

WORKFLOW FOR ERROR ANALYSIS:
1. User asks about errors → IMMEDIATELY call get_obs_errors() or search_obs_logs()
2. Parse the actual results from the tool
3. Extract key details: application name, error message, stack trace, traceId, timestamp
4. Present findings with specific data points from the logs

TIME RANGES:
- Default: Use 'last 1 hour' if not specified
- For broader search: 'last 6 hours' or 'last 24 hours'
- Max allowed: 72 hours
- Format: 'last N hours' or 'last N days'

ERROR ANALYSIS:
- Show traceId for correlation
- Extract stack trace top line (the actual error)
- Identify affected application/service
- Note frequency if multiple occurrences
- Reference specific log entries, not summaries

OUTPUT FORMAT:
- Always include actual log data (timestamps, messages, traceIds)
- Use markdown tables for multiple entries
- Highlight ERROR level entries
- Link traceIds for drill-down: https://observe-dev.example.com/trace/{traceId}

IMPORTANT: 
- ALWAYS call a tool before answering log-related questions
- DO NOT fabricate log entries or error messages
- If tools return empty results, say "No logs found" - don't make up data
- Use exact data from tool responses, never approximate"""

    obs_agent = None
    if OBS_TOOLS:
        obs_agent = _create_agent(
            model=llm,
            tools=OBS_TOOLS,
            prompt=obs_prompt,
        )
        print(f"  3️⃣ OBS Agent ({len(OBS_TOOLS)} tools) : {C.CYAN}{[t.name for t in OBS_TOOLS]}{C.RESET}")
    else:
        print(f"  3️⃣ OBS Agent : {C.YELLOW}⚠️  Not configured{C.RESET}")

    # ═══════════════════════════════════════════════════════════════
    # 4️⃣ SLACK AGENT — Communication and channel search
    # ═══════════════════════════════════════════════════════════════
    slack_prompt = """You are a communication assistant that searches and interacts with Slack.
You can search messages, list channels, get channel history, and post messages.
When searching, use clear keywords and channel/user filters when specified.
Present results with channel names, user handles, message text, and permalinks.
For posting messages, be professional and clear."""

    slack_agent = None
    if SLACK_TOOLS:
        slack_agent = _create_agent(
            model=llm,
            tools=SLACK_TOOLS,
            prompt=slack_prompt,
        )
        print(f"  4️⃣ Slack Agent ({len(SLACK_TOOLS)} tools) : {C.CYAN}{[t.name for t in SLACK_TOOLS]}{C.RESET}")
    else:
        print(f"  4️⃣ Slack Agent : {C.YELLOW}⚠️  Not configured (set SLACK_TOKEN){C.RESET}")

    # ═══════════════════════════════════════════════════════════════
    # 5️⃣ VULNSCAN AGENT — Vulnerability and dependency insights
    # ═══════════════════════════════════════════════════════════════
    vulnscan_prompt = """You are a security and dependency insights assistant using VulnScan.
You can list repositories, find vulnerable packages (with severity and CVEs),
detect exposed secrets, and search dependencies across repos.
When reporting vulnerabilities, always include: package name, current version,
severity (critical/high/medium/low), CVE IDs, and the recommended safe version.
Group results by severity with critical first."""

    vulnscan_agent = None
    if VulnScan_TOOLS:
        vulnscan_agent = _create_agent(
            model=llm,
            tools=VulnScan_TOOLS,
            prompt=vulnscan_prompt,
        )
        print(f"  5️⃣ VulnScan Agent ({len(VulnScan_TOOLS)} tools) : {C.CYAN}{[t.name for t in VulnScan_TOOLS]}{C.RESET}")
    else:
        print(f"  5️⃣ VulnScan Agent : {C.YELLOW}⚠️  Not configured{C.RESET}")

    # ════════════════════════════════════════════════════════════════
    # 6️⃣ QA AUTOMATION AGENT — UI comparator + API test tool
    # ════════════════════════════════════════════════════════════════
    qa_prompt = """You are a QA automation assistant that orchestrates two local testing tools:

1. Frontend Automation (UI Comparator & Recorder) — records SPA user flows and replays them
   across environments with pixel-diff reports. Runs on http://localhost:3737.
2. Backend Automation (DataVeriTest) — discovers API endpoints from an OpenAPI spec, runs
   test suites (smoke / health / performance / status-code / content-type / all), executes
   individual requests, and verifies API responses against the connected DB. Runs on
   http://localhost:9090.

Workflow rules:
- Always call qa_frontend_health or qa_backend_health first if you are unsure the tool is running.
- For UI regression: list recorded sessions, then run qa_run_ui_comparison with both env URLs.
- For API validation: connect the backend tool via qa_connect_target_api, list endpoints with
  qa_list_api_endpoints, then pick the right suite (qa_run_smoke_tests, qa_run_health_tests, ...).
- Use qa_execute_single_endpoint for Swagger-style ad-hoc calls, and qa_db_query for read-only
  SQL verification against the connected database.
- Present a concise report summary with: session/suite name, pass/fail counts, and links or
  identifiers of the generated report when available. Never invent results — only report what
  the tools return.
"""

    qa_agent = None
    if QA_TOOLS:
        qa_agent = _create_agent(
            model=llm,
            tools=QA_TOOLS,
            prompt=qa_prompt,
        )
        print(f"  6️⃣ QA Agent ({len(QA_TOOLS)} tools) : {C.CYAN}{[t.name for t in QA_TOOLS]}{C.RESET}")
    else:
        print(f"  6️⃣ QA Agent : {C.YELLOW}⚠️  Not configured{C.RESET}")

    print(f"  Memory       : {C.PURPLE}ChromaDB (semantic retrieval){C.RESET}")
    print(f"  Embeddings   : {C.CYAN}nomic-embed-text (local via Ollama){C.RESET}")

    # ═══════════════════════════════════════════════════════════════
    # 6️⃣ SUPERVISOR — Orchestrates sub-agents and user interaction
    # ═══════════════════════════════════════════════════════════════

    # Wrap sub-agents as callable tools for the supervisor
    sub_agents = {
        "db": db_agent,
        "github": github_agent,
        "obs": obs_agent,
        "slack": slack_agent,
        "vulnscan": vulnscan_agent,
    }

    MAX_SUB_AGENT_RESPONSE = 4000  # Keep supervisor context manageable
    SUB_AGENT_RECURSION_LIMIT = 40  # Room for repo_map + semantic_search + a few line-range reads + action + retries

    def _truncate(text: str) -> str:
        if len(text) <= MAX_SUB_AGENT_RESPONSE:
            return text
        return text[:MAX_SUB_AGENT_RESPONSE] + f"\n\n... [truncated, {len(text)} total chars]"

    @tool
    async def query_database(task: str) -> str:
        """Delegate a database task to the DB sub-agent. Use for SQL queries, schema inspection, data analysis.
        
        Args:
            task: Clear description of what to query (e.g. "list all tables in company_db" or "count employees by department")
        """
        result = await db_agent.ainvoke({"messages": [HumanMessage(content=task)]}, config={"recursion_limit": SUB_AGENT_RECURSION_LIMIT})
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        return _truncate(ai_msgs[-1].content) if ai_msgs else "No response from DB agent"

    @tool
    async def query_github(task: str) -> str:
        """Delegate a GitHub/code task to the GitHub sub-agent. Use for reading files, searching code, creating branches, committing fixes, and opening PRs.

        BRANCH POLICY: If the user did NOT name a branch, do NOT invent one.
        Just describe the task without a branch; the tools will use the
        configured / repo default branch (see repos.json + API fallback).
        Only pass an explicit branch if the user asked for it or you already
        confirmed it exists.

        Args:
            task: Clear description of the code task (e.g. "search for APP-1000 in the sampleapp repo" or "create branch ai-agent/fix-x in the sampleapp repo").
        """
        result = await github_agent.ainvoke({"messages": [HumanMessage(content=task)]}, config={"recursion_limit": SUB_AGENT_RECURSION_LIMIT})
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        return _truncate(ai_msgs[-1].content) if ai_msgs else "No response from GitHub agent"

    @tool
    async def query_obs_logs(task: str) -> str:
        """Delegate a log/observability task to the OBS sub-agent. Use for checking application errors, searching traces, counting log entries.
        
        Args:
            task: Clear description of the log query (e.g. "get errors for sampleapi in the last 2 hours" or "search logs for traceId abc123")
        """
        if not obs_agent:
            return "OBS agent not configured. Set OBS_APP_ID, OBS_USERNAME, OBS_PASSWORD in .env"
        result = await obs_agent.ainvoke({"messages": [HumanMessage(content=task)]}, config={"recursion_limit": SUB_AGENT_RECURSION_LIMIT})
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        return _truncate(ai_msgs[-1].content) if ai_msgs else "No response from OBS agent"

    @tool
    async def query_slack(task: str) -> str:
        """Delegate a Slack/communication task to the Slack sub-agent. Use for searching messages, reading channels, posting messages.
        
        Args:
            task: Clear description of the Slack task (e.g. "search app-errors channel for recent errors" or "post summary to team-channel")
        """
        if not slack_agent:
            return "Slack agent not configured. Set SLACK_TOKEN in .env"
        result = await slack_agent.ainvoke({"messages": [HumanMessage(content=task)]}, config={"recursion_limit": SUB_AGENT_RECURSION_LIMIT})
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        return _truncate(ai_msgs[-1].content) if ai_msgs else "No response from Slack agent"

    @tool
    async def query_vulnscan(task: str) -> str:
        """Delegate a security/dependency task to the VulnScan sub-agent. Use for checking vulnerabilities, finding outdated packages, listing repos by APP_ID, and detecting secrets.
        
        Args:
            task: Clear description of the security task (e.g. "get critical vulnerabilities for APP_ID 100000001" or "search for axios package across all repos" or "show secrets in sampleapp")
        """
        if not vulnscan_agent:
            return "VulnScan agent not configured."
        result = await vulnscan_agent.ainvoke({"messages": [HumanMessage(content=task)]}, config={"recursion_limit": SUB_AGENT_RECURSION_LIMIT})
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        return _truncate(ai_msgs[-1].content) if ai_msgs else "No response from VulnScan agent"

    @tool
    async def query_qa(task: str) -> str:
        """Delegate a QA automation task to the QA sub-agent (UI Comparator + DataVeriTest).

        Use for running UI regression comparisons across environments, functional replays,
        connecting to a target API, listing endpoints, and running API vs DB validation.

        Args:
            task: Clear description of the QA task (e.g. "run UI comparison of the login flow
                between https://old.example.com and https://new.example.com" or "connect the
                API tool to http://localhost:8080 and list discovered endpoints").
        """
        if not qa_agent:
            return "QA automation agent not configured. Start the frontend tool (port 3737) and backend tool (port 9090), then restart the server."
        result = await qa_agent.ainvoke({"messages": [HumanMessage(content=task)]}, config={"recursion_limit": SUB_AGENT_RECURSION_LIMIT})
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        return _truncate(ai_msgs[-1].content) if ai_msgs else "No response from QA agent"

    supervisor_tools = [query_database, query_github, query_obs_logs, query_slack, query_vulnscan, query_qa] + INTERACTIVE_WORKFLOW_TOOLS

    supervisor_prompt = """You are the SUPERVISOR agent that orchestrates a team of specialized sub-agents to complete user tasks.

YOUR SUB-AGENTS:
1. query_database — SQL queries, schema inspection, data analysis
2. query_github — Read/search code, create branches, commit files, open PRs
3. query_obs_logs — Application error logs, traces, observability
4. query_slack — Search/post messages, channel history
5. query_vulnscan — Repository security insights, vulnerable dependencies (CVEs), outdated packages, detected secrets
6. query_qa — Frontend UI regression (UI Comparator) + Backend API validation (DataVeriTest) via local automation tools

YOUR INTERACTION TOOLS:
- ask_user_question — Ask the user a question with button options (use when user needs to choose)
- show_analysis_result — Present findings and ask for next action
- investigate_error_interactive — Ask user about investigating a specific error
- ask_create_pr — Ask user to confirm PR creation

WORKFLOW RULES:
1. DECOMPOSE: Break the user's request into clear sub-tasks.
2. DELEGATE: Call the right sub-agent for each sub-task. Pass SPECIFIC, COMPLETE instructions.
3. SYNTHESIZE: Combine sub-agent results into coherent findings.
4. INTERACT: When user input is needed (choices, confirmations), use ask_user_question.
5. ITERATE: After user responds, continue with the next sub-task.

TOOL-USE POLICY (highest priority — do not skip):
- For ANY question about logs, ERROR messages, code, files, PRs, workflow runs,
  vulnerabilities, tests, DB rows, or Slack messages, you MUST call the
  corresponding sub-agent tool. Never answer from memory.
- If unsure whether a tool call is needed, call the tool. A tool call that
  returns "no results" is better than an unverified guess.
- Do NOT summarise or paraphrase what you "think" the system would say —
  wait for the actual tool response and quote it.
- Only respond directly without tools for pure greetings ("hi", "thanks"),
  meta-questions about your own capabilities, or when the user explicitly
  says "no need to look it up".

STEP BUDGET (important):
- You have a hard cap of ~50 sub-agent calls per user turn. Plan efficiently.
- Prefer ONE action per sub-agent call. Do not chain "search AND read AND create PR" in one delegation — that pattern causes the sub-agent to loop.
- Never re-read a file you already read this turn (the GitHub cache will return CACHED and it wastes a step).
- If a single natural task needs > 8 sub-agent calls to finish, stop after step 7 and ask the user whether to continue.

WHEN TO ASK THE USER:
- Multiple options exist and user preference matters (e.g., which PR comment to address, which error to investigate)
- Before destructive/important actions (creating branches, PRs, posting to Slack)
- When sub-agent returns ambiguous results that need clarification

DELEGATION TIPS:
- Give sub-agents FULL context. Don't say "search code" — say "search for APP-1000 error handler in repo your-username/sample-app"
- Pass relevant data from previous steps (e.g., error details from OBS to GitHub agent)
- If a sub-agent returns a lot of data, summarize KEY points before showing user

EDGE CASES:
- If sub-agent returns multiple items (e.g., 4 PR comments), ask user which one to focus on
- If sub-agent fails, explain what went wrong and suggest alternatives
- For multi-step workflows, confirm the PLAN with user before executing destructive steps

ERROR-TO-FIX WORKFLOW (use when investigating errors and creating fixes):
1. DISCOVER: Call query_obs_logs to get actual error data (timestamps, messages, traceIds, counts)
2. LOCATE: Call query_github to search for the error code/message in the codebase
3. UNDERSTAND: Call query_github to read the relevant file(s) and understand the code structure
4. ANALYZE: Synthesize findings — what is the root cause? What files need changes?
5. PRESENT: Use show_analysis_result to show user your findings with options
6. CONFIRM: Use ask_user_question to confirm before creating branch/PR
7. FIX: Call query_github to create branch, commit fix, and open PR
Keep each query_github call focused on ONE task (don't ask it to search AND create PR in one call).

DEPENDENCY-UPGRADE WORKFLOW (use when fixing vulnerable/outdated packages):
1. SCAN: Call query_vulnscan to get vulnerable deps (filter by severity if specified)
2. IDENTIFY: Pick the package to upgrade — note current version, safe version, and affected file (pom.xml/package.json)
3. PRESENT: Show user the findings with severity and ask which to fix
4. READ DEP FILE: Call query_github to read the dependency file (pom.xml, package-lock.json, etc.)
5. CREATE BRANCH: Call query_github to create branch
6. PATCH VERSION: Call query_github to use patch_file to update the version string in the dep file
7. SEARCH ALL USAGES: Call query_github to search for ALL imports/usages of the package across the repo (e.g. "import com.fasterxml.jackson" or "require('axios')")
8. READ EACH USAGE FILE: For each file that imports/uses the package, call query_github to read it
9. CHECK API CHANGES: If the upgrade is a major version bump, check if any classes/methods/parameters changed. Look for deprecated APIs, renamed functions, or changed signatures.
10. PATCH CODE FILES: If any API changed, call query_github to use patch_file on each affected file (update imports, method names, parameters)
11. CREATE PR: Call query_github to create PR listing all changes made (dep version + code updates)

CRITICAL: After patching the version, ALWAYS search for usages. Don't skip step 7-10 even if it seems like a minor bump.

PR-COMMENT WORKFLOW (use when user asks to check/respond to PR comments):
1. READ: Call query_github to get_pr_comments on the specified PR
2. ANALYZE: Understand what each comment is asking for
3. FIX: If code changes needed, use query_github with patch_file to address each comment
4. REPLY: Call query_github to add_pr_comment with a summary of changes made

IMPORTANT:
- Never make up data. Only use actual results from sub-agents.
- Always call sub-agents rather than answering domain questions yourself.
- NEVER answer action requests from memory/context. If user asks to check, update, create, search, or fix anything — ALWAYS call the appropriate sub-agent with fresh tool calls, even if you saw similar results before.
- Conversation history is ONLY for understanding context, NOT for answering action tasks.
- For simple greetings or general questions, respond directly without calling sub-agents."""

    def _trim_supervisor_context(state):
        """Pre-model hook: trim old tool exchanges to prevent context overflow.
        
        LangGraph accumulates every AIMessage(tool_calls) + ToolMessage inside
        the ReAct loop. After 5-6 sub-agent calls this can exceed 30K chars,
        causing the Copilot API to return empty responses. This hook keeps only
        the initial question + the last few tool rounds within budget.
        """
        msgs = state["messages"]
        total = sum(len(str(getattr(m, 'content', '') or '')) for m in msgs)

        if total <= MAX_SUPERVISOR_INTERNAL or len(msgs) <= 6:
            return {"messages": msgs}

        # Always keep the first message (user's question / context start)
        first = msgs[0]
        rest = msgs[1:]
        budget = MAX_SUPERVISOR_INTERNAL - len(str(first.content or '')) - 200

        # Keep from the end within budget (preserves most recent tool context)
        kept = []
        chars = 0
        for m in reversed(rest):
            mc = len(str(getattr(m, 'content', '') or ''))
            if chars + mc > budget and len(kept) >= 4:
                break
            kept.insert(0, m)
            chars += mc

        # Don't start with orphan ToolMessage (needs preceding AIMessage)
        while kept and isinstance(kept[0], ToolMessage):
            kept.pop(0)

        dropped = len(rest) - len(kept)
        if dropped <= 0:
            return {"messages": msgs}

        print(f"{C.YELLOW}  ✂️  pre_model_hook: trimmed {dropped} old messages ({total} → ~{chars + len(str(first.content or ''))} chars){C.RESET}")
        bridge = HumanMessage(content=f"[{dropped} earlier tool steps trimmed. Continue with the task based on the most recent results.]")
        return {"messages": [first, bridge] + kept}

    supervisor_agent = _create_agent(
        model=llm,
        tools=supervisor_tools,
        prompt=supervisor_prompt,
        pre_model_hook=_trim_supervisor_context,
    )
    print(f"  5️⃣ Supervisor ({len(supervisor_tools)} tools) : {C.CYAN}{[t.name for t in supervisor_tools]}{C.RESET}")
    print(f"     Context trim : {C.CYAN}pre_model_hook @ {MAX_SUPERVISOR_INTERNAL} chars{C.RESET}")

    print(f"  Architecture : {C.YELLOW}Supervisor → Sub-agents (orchestrated){C.RESET}")
    print(f"{C.GREEN}  ✅ Supervisor multi-agent system ready!{C.RESET}")
    print(f"{C.GREEN}{'='*60}{C.RESET}\n")

    return {
        "supervisor": supervisor_agent,
        "db": db_agent,
        "github": github_agent,
        "obs": obs_agent,
        "slack": slack_agent,
        "vulnscan": vulnscan_agent,
        "llm": llm,
        "_sub_agents": sub_agents,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA LOADING — Same as original (no change needed)
# ══════════════════════════════════════════════════════════════════════════════

def load_schema_context() -> str:
    """Load compressed schema for sidebar display."""
    from tools import list_tables, describe_table

    log("SCHEMA", "Loading schema context for sidebar...")
    dbs = load_databases()
    active = [db for db in dbs if db.get("active")]

    if not active:
        return "No active databases."

    parts = []
    for db in active:
        db_name = db["name"]
        try:
            tables_json = list_tables.invoke({"db_name": db_name})
            tables = json.loads(tables_json)
            if isinstance(tables, dict) and "error" in tables:
                parts.append(f"\n--- {db_name}: UNREACHABLE ---")
                continue
            parts.append(f"\n--- {db_name} ({db['database']}@{db['host']}) ---")
            parts.append(f"Tables: {json.dumps(tables)}")

            for table in tables:
                try:
                    desc_json = describe_table.invoke({"db_name": db_name, "table_name": table})
                    raw = json.loads(desc_json)
                    if isinstance(raw, dict) and "columns" in raw:
                        cols = ", ".join(f"{c['column_name']}({c['data_type']})" for c in raw["columns"])
                        parts.append(f"  {table}: [{cols}]")
                except Exception as e:
                    parts.append(f"  {table}: ERROR: {e}")
        except Exception as e:
            parts.append(f"\n--- {db_name}: ERROR: {e} ---")

    schema = "\n".join(parts)
    log("SCHEMA", f"Schema loaded: {len(schema)} chars")
    return schema


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDING — The key difference from the original
#
# OLD: build_history_context(history_messages) → sliding window + summary
# NEW: build_vector_context(session_id, query) → semantic retrieval
#
# INTERVIEW: "The context building strategy is the architectural differentiator.
# Everything else (agent, tools, streaming) remains identical. This shows good
# separation of concerns — swapping the retrieval strategy doesn't require
# changing the agent or UI code."
# ══════════════════════════════════════════════════════════════════════════════


def build_vector_context(session_id: str, query: str) -> list:
    """
    Build LLM context using semantic retrieval from ChromaDB.
    Skips cross-session retrieval for action tasks to prevent stale-memory responses.
    """
    # Action keywords that require fresh tool execution, not memory recall
    action_words = ("update", "upgrade", "create", "fix", "patch", "commit", "check", "search", "list", "show", "get")
    is_action = any(query.lower().strip().startswith(w) for w in action_words)

    print(f"\n{C.PURPLE}  ┌─── 🧠 BUILD VECTOR CONTEXT ─────────────────────────────┐{C.RESET}")
    print(f"{C.PURPLE}  │{C.RESET}  Session: {C.CYAN}{session_id}{C.RESET}")
    print(f"{C.PURPLE}  │{C.RESET}  Query  : \"{query[:60]}\"")
    if is_action:
        print(f"{C.PURPLE}  │{C.RESET}  Mode   : {C.YELLOW}ACTION task — only same-session history{C.RESET}")
    else:
        print(f"{C.PURPLE}  │{C.RESET}  Strategy: Semantic (top-6) + Recency (last-4) hybrid")
    print(f"{C.PURPLE}  │{C.RESET}")
    print(f"{C.PURPLE}  │{C.RESET}  Calling get_relevant_history()...")
    print(f"{C.PURPLE}  └──────────────────────────────────────────────────────────┘{C.RESET}")
    
    # Get semantically relevant history from ChromaDB
    relevant_messages = get_relevant_history(session_id, query)
    
    if not relevant_messages:
        print(f"{C.DIM}  → No history found — this is a fresh conversation{C.RESET}")
        return []
    
    # Convert to LangChain message objects, capping per-message length
    context = []
    for msg in relevant_messages:
        content = msg["content"]
        if len(content) > MAX_CONTEXT_MSG_CHARS:
            content = content[:MAX_CONTEXT_MSG_CHARS] + "\n[...truncated]"
        if msg["role"] == "user":
            context.append(HumanMessage(content=content))
        elif msg["role"] == "assistant":
            context.append(AIMessage(content=content))
    
    # Apply hard character cap
    total_chars = sum(len(m.content) for m in context)
    removed_count = 0
    while total_chars > MAX_CONTEXT_CHARS and len(context) > 1:
        removed = context.pop(0)
        total_chars -= len(removed.content)
        removed_count += 1
    
    if removed_count > 0:
        print(f"{C.YELLOW}  ⚠️  Trimmed {removed_count} messages (exceeded {MAX_CONTEXT_CHARS} char cap){C.RESET}")
    
    print(f"{C.GREEN}  ✅ Context built: {len(context)} messages, {total_chars} chars{C.RESET}")
    return context


def _bridge_messages(route: str, context: list, question: str) -> list[dict]:
    """Convert retrieved LangChain history to bridge messages within the cap."""
    history = list(context)
    total_chars = sum(len(message.content) for message in history) + len(question)
    while total_chars > MAX_CONTEXT_CHARS and history:
        total_chars -= len(history.pop(0).content)

    messages = [{"role": "system", "content": CODEX_BRIDGE_PROMPTS[route]}]
    for message in history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        messages.append({"role": role, "content": message.content})
    messages.append({"role": "user", "content": question})
    return messages


def _simple_route(question: str) -> str:
    """Quick keyword route for codex bridge fallback only."""
    q = question.lower()
    if any(w in q for w in ("slack", "channel", "post to")):
        return "slack"
    if any(w in q for w in ("github", "repo", "code", "branch", "pr", "commit")):
        return "code"
    if any(w in q for w in ("database", "table", "sql", "employees", "departments")):
        return "database"
    if any(w in q for w in ("obs", "logs", "traceid", "error")):
        return "logs"
    return "general"


async def _ask_codex_bridge(question: str, session_id: str) -> str:
    """Call the read-only Codex bridge as a complete agent."""
    import httpx

    route = _simple_route(question)
    context = build_vector_context(session_id, question)
    messages = _bridge_messages(route, context, question)
    timeout_seconds = max(1, int(os.environ.get("CODEX_BRIDGE_TIMEOUT_MS", "300000"))) / 1000
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
    payload = {"model": MODEL, "messages": messages, "stream": False}

    log("CODEX", f"Bridge request route={route}, messages={len(messages)}, endpoint={LLM_BASE_URL}")
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(f"{LLM_BASE_URL}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("error", {}).get("message", "Codex bridge request failed")
        except Exception:
            detail = "Codex bridge request failed"
        raise RuntimeError(f"Codex bridge returned HTTP {exc.response.status_code}: {detail}") from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError(f"Codex bridge is unavailable: {exc}") from exc

    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Codex bridge returned an invalid completion response") from exc
    if not isinstance(answer, str) or not answer:
        raise RuntimeError("Codex bridge returned an empty answer")
    return answer


# ══════════════════════════════════════════════════════════════════════════════
# INVOKE + STREAM — Same as original, but using build_vector_context()
#
# DIFFERENCE:
#   OLD: ask_agent(agent, question, thread_id, history=[...from json...])
#   NEW: ask_agent(agent, question, session_id) — history comes from ChromaDB
# ══════════════════════════════════════════════════════════════════════════════


# ── Phase 3 critic (Pattern 4 in the plan) ──────────────────────────────────
# If the supervisor answers a data-related question WITHOUT calling any query_*
# sub-agent, we inject one directive and re-invoke ONCE. Zero cost on the
# happy path (no extra call), +1 call only when the model would otherwise
# hallucinate an answer with no evidence.

_DATA_TOOLS = {
    "query_database", "query_github", "query_obs_logs",
    "query_slack", "query_vulnscan", "query_qa",
}

# regex per data tool — first match wins
import re as _re
_DATA_TOOL_KEYWORDS = [
    ("query_obs_logs", [
        r"\berror(s)?\b", r"\blog(s)?\b", r"\btrace ?id\b", r"\belf\b",
        r"\bexception\b", r"\bAPP-\d+\b", r"\bstacktrace\b", r"\bsampleapi\b",
    ]),
    ("query_github", [
        r"\bcode\b", r"\bfile(s)?\b", r"\bclass(es)?\b", r"\bmethod(s)?\b",
        r"\bfunction(s)?\b", r"\bpatch\b", r"\bcommit\b", r"\bpr\b",
        r"\bbranch\b", r"\brepo(sitory)?\b", r"pom\.xml", r"package\.json",
        r"\bhandler\b",
    ]),
    ("query_vulnscan", [
        r"\bvuln(erabilit(y|ies))?\b", r"\bcve\b", r"\bdependenc(y|ies)\b",
        r"\bpackage\b", r"\bsecurit\w*\b", r"\boutdated\b", r"\bsecret(s)?\b",
    ]),
    ("query_slack", [
        r"\bslack\b", r"\bchannel(s)?\b", r"\bpost\b", r"\bmessage(s)?\b",
    ]),
    ("query_database", [
        r"\bdatabase(s)?\b", r"\btable(s)?\b", r"\bsql\b", r"\brow(s)?\b",
        r"\bemployee(s)?\b", r"\bdepartment(s)?\b", r"\bproject(s)?\b",
    ]),
    ("query_qa", [
        r"\btest(s|ing)?\b", r"\bqa\b", r"\bsmoke\b", r"\bplaywright\b",
        r"\bdataveritest\b",
    ]),
]


def _detect_needed_tool(question: str) -> Optional[str]:
    """Return the first sub-agent tool whose keywords match the question."""
    q = question or ""
    for tool_name, patterns in _DATA_TOOL_KEYWORDS:
        for pat in patterns:
            if _re.search(pat, q, _re.IGNORECASE):
                return tool_name
    return None


def _count_data_tool_calls(messages) -> int:
    """Count how many query_* tool responses appear in the message list."""
    count = 0
    for m in messages or []:
        if isinstance(m, ToolMessage):
            name = getattr(m, "name", "") or ""
            if name in _DATA_TOOLS:
                count += 1
    return count


def _specific_tool_was_called(messages, tool_name: str) -> bool:
    """Was the SPECIFIC named tool actually invoked this turn?"""
    for m in messages or []:
        if isinstance(m, ToolMessage):
            if (getattr(m, "name", "") or "") == tool_name:
                return True
    return False


def _critic_directive(needed_tool: str, question: str) -> str:
    """Build the injected message that forces the model to actually use a tool."""
    return (
        "CRITIC: Your previous response used no data tool, but the user's question "
        f"clearly requires it (matched keyword for `{needed_tool}`). "
        f"You MUST call `{needed_tool}` now with a concrete task derived from the "
        "original question, then re-answer based ONLY on the tool's output. "
        "Do NOT ask the user any question. Do NOT answer from memory."
    )


def _preflight_directive(needed_tool: str) -> str:
    """Prepend guidance so the first LLM call knows which data tool to start with."""
    return (
        f"[PLANNING HINT — auto-injected by keyword match]\n"
        f"This task requires `{needed_tool}`. You MUST call it BEFORE any other "
        f"data tool, and base your answer on its output — never on memory or "
        f"assumptions. If the task also needs other tools (e.g. GitHub for a fix "
        f"after logs), call `{needed_tool}` first, then chain the rest.\n\n"
        f"USER REQUEST:\n"
    )


async def ask_agent(agent, question: str, session_id: str) -> str:
    """
    Run the agent with vector-retrieved context (wait for full response).
    Supports both single agent and multi-agent (dict with router).
    """
    import time
    start_time = time.time()

    # Reset per-turn tool budget (file reads + code searches) so the LLM
    # cannot re-read the same file 5+ times across requests.
    from integrations.github.github_tools import _reset_file_cache
    _reset_file_cache()

    if isinstance(agent, dict) and agent.get("_backend") == "codex-bridge":
        return await _ask_codex_bridge(question, session_id)
    
    # Use supervisor if available (multi-agent dict), otherwise single agent
    if isinstance(agent, dict):
        active_agent = agent.get("supervisor") or agent["db"]
        print(f"\n{C.YELLOW}  🎯 SUPERVISOR MODE{C.RESET}")
    else:
        active_agent = agent
    
    print(f"\n{C.BOLD}{'='*60}{C.RESET}")
    print(f"{C.BOLD}  📨 AGENT INVOCATION (non-streaming){C.RESET}")
    print(f"{'='*60}")
    print(f"  Question : \"{question[:70]}{'...' if len(question) > 70 else ''}\"")
    print(f"  Session  : {C.CYAN}{session_id}{C.RESET}")
    print(f"{'='*60}")

    # Build context via vector retrieval (THE KEY DIFFERENCE)
    context_messages = build_vector_context(session_id, question)

    # Pre-flight: prepend planning hint when a specific data tool is required.
    preflight_tool = _detect_needed_tool(question)
    effective_question = question
    if preflight_tool:
        effective_question = _preflight_directive(preflight_tool) + question
        print(f"{C.CYAN}  🧭 PRE-FLIGHT: injected planning hint → `{preflight_tool}`{C.RESET}", flush=True)

    # Add the new question
    all_messages = context_messages + [HumanMessage(content=effective_question)]
    total_chars = sum(len(m.content) for m in all_messages)

    print(f"\n{C.CYAN}  📤 SENDING TO LLM:{C.RESET}")
    print(f"  ├─ Messages: {len(all_messages)} total")
    print(f"  ├─ Context : {len(context_messages)} from vector retrieval")
    print(f"  ├─ New     : 1 (current question)")
    print(f"  └─ Chars   : {total_chars} / {MAX_CONTEXT_CHARS} max")
    for i, msg in enumerate(all_messages):
        role = "USER" if isinstance(msg, HumanMessage) else "AI"
        icon = "🧑" if role == "USER" else "🤖"
        print(f"     {icon} [{i+1}] {msg.content[:60]}{'...' if len(msg.content) > 60 else ''}")
    
    print(f"\n  {C.YELLOW}⏳ Waiting for LLM response...{C.RESET}")

    # Loop to handle multiple tool calls (7B model limitation)
    conversation = all_messages.copy()
    max_iterations = 10  # Safety limit
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        result = await active_agent.ainvoke({"messages": conversation}, config={"recursion_limit": 100})
        
        # Get the last AI message
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        if not ai_messages:
            return "No response generated."
        
        answer = ai_messages[-1].content
        
        # Check if model output a tool call as text
        if answer and _looks_like_tool_call(answer):
            print(f"  {C.YELLOW}⚠️  Model output tool call as text — executing manually (iteration {iteration}){C.RESET}")
            tool_result = _execute_text_tool_call(answer)
            
            if tool_result:
                # Add the tool call and result to conversation
                conversation.append(AIMessage(content=answer))
                conversation.append(HumanMessage(content=f"Tool result: {tool_result}"))
                # Continue loop to get next response
                continue
            else:
                # Tool execution failed, break out
                print(f"  {C.RED}❌ Tool execution failed{C.RESET}")
                break
        else:
            # Got a real answer (not a tool call)
            needed = _detect_needed_tool(question)
            called_specific = _specific_tool_was_called(result.get("messages", []), needed) if needed else True
            tool_names_seen = [getattr(m, "name", "?") for m in result.get("messages", []) if isinstance(m, ToolMessage)]
            print(f"\n{C.CYAN}  🕵️  CRITIC CHECK: needed={needed!r}, called_specific={called_specific}, tools_seen={tool_names_seen}{C.RESET}", flush=True)
            if needed and not called_specific:
                # Critic fires: force one more invoke with a directive.
                print(f"\n{C.YELLOW}  🕵️  CRITIC: needed `{needed}` was NOT called (question keyword-matched to it). Re-invoking once.{C.RESET}")
                conversation.append(AIMessage(content=answer))
                conversation.append(HumanMessage(content=_critic_directive(needed, question)))
                result = await active_agent.ainvoke({"messages": conversation}, config={"recursion_limit": 100})
                ai_messages_2 = [m for m in result["messages"] if isinstance(m, AIMessage)]
                if ai_messages_2:
                    answer = ai_messages_2[-1].content
                called_specific_after = _specific_tool_was_called(result.get("messages", []), needed)
                if called_specific_after:
                    print(f"{C.GREEN}  ✅ CRITIC recovered: `{needed}` was called after directive.{C.RESET}")
                else:
                    print(f"{C.YELLOW}  ⚠️  CRITIC ran but `{needed}` still not called. Returning as-is.{C.RESET}")

            elapsed = time.time() - start_time
            print(f"\n{C.GREEN}  ✅ RESPONSE RECEIVED ({elapsed:.1f}s, {iteration} iteration(s)){C.RESET}")
            print(f"  ├─ Length: {len(answer)} chars")
            print(f"  └─ Preview: \"{answer[:100]}{'...' if len(answer) > 100 else ''}\"")
            print(f"{'='*60}\n")
            return answer
    
    # If we hit max iterations, return last answer
    elapsed = time.time() - start_time
    print(f"\n{C.YELLOW}  ⚠️  Max iterations reached ({max_iterations}){C.RESET}")
    print(f"  ├─ Length: {len(answer)} chars")
    print(f"  └─ Preview: \"{answer[:100]}{'...' if len(answer) > 100 else ''}\"")
    print(f"{'='*60}\n")
    return answer


async def stream_agent(agent, question: str, session_id: str) -> AsyncGenerator[str, None]:
    """
    Stream the agent's response token-by-token with vector-retrieved context.
    Supports both single agent and multi-agent (dict with router).
    """
    import time
    start_time = time.time()

    # Clear file read cache for fresh request
    from integrations.github.github_tools import _reset_file_cache
    _reset_file_cache()

    if isinstance(agent, dict) and agent.get("_backend") == "codex-bridge":
        # codex exec JSONL exposes the completed agent message rather than token
        # deltas. Yield it as one application-level streaming chunk.
        yield await _ask_codex_bridge(question, session_id)
        return
    
    # Use supervisor if available (multi-agent dict), otherwise single agent
    if isinstance(agent, dict):
        active_agent = agent.get("supervisor") or agent["db"]
        print(f"\n{C.YELLOW}  🎯 SUPERVISOR MODE{C.RESET}")
    else:
        active_agent = agent
    
    print(f"\n{C.BOLD}{'='*60}{C.RESET}")
    print(f"{C.BOLD}  ⚡ AGENT STREAMING INVOCATION{C.RESET}")
    print(f"{'='*60}")
    print(f"  Question : \"{question[:70]}{'...' if len(question) > 70 else ''}\"")
    print(f"  Session  : {C.CYAN}{session_id}{C.RESET}")
    print(f"{'='*60}")

    # Build context via semantic retrieval
    context_messages = build_vector_context(session_id, question)

    # Pre-flight: prepend planning hint when a specific data tool is required.
    preflight_tool = _detect_needed_tool(question)
    effective_question = question
    if preflight_tool:
        effective_question = _preflight_directive(preflight_tool) + question
        print(f"{C.CYAN}  🧭 PRE-FLIGHT: injected planning hint → `{preflight_tool}`{C.RESET}", flush=True)

    all_messages = context_messages + [HumanMessage(content=effective_question)]

    # Hard cap
    total_chars = sum(len(m.content) for m in all_messages)
    trimmed = 0
    while total_chars > MAX_CONTEXT_CHARS and len(all_messages) > 1:
        removed = all_messages.pop(0)
        total_chars -= len(removed.content)
        trimmed += 1

    print(f"\n{C.CYAN}  📤 SENDING TO LLM (streaming):{C.RESET}")
    print(f"  ├─ Messages: {len(all_messages)} total ({len(context_messages)} context + 1 new)")
    print(f"  ├─ Chars   : {total_chars} / {MAX_CONTEXT_CHARS} max")
    if trimmed:
        print(f"  ├─ Trimmed : {trimmed} messages (char cap exceeded)")
    print(f"  └─ Mode    : Token-by-token streaming")
    print(f"\n  {C.CYAN}📋 MESSAGE PREVIEW:{C.RESET}")
    for i, msg in enumerate(all_messages):
        role = "USER" if isinstance(msg, HumanMessage) else "AI"
        icon = "🧑" if role == "USER" else "🤖"
        preview = msg.content[:80].replace('\n', ' ')
        print(f"     {icon} [{i+1}/{len(all_messages)}] [{role:4s}] {preview}{'...' if len(msg.content) > 80 else ''}")
    
    print(f"\n  {C.YELLOW}⚡ Streaming started...{C.RESET}")

    tool_call_count = 0
    data_tool_count = 0
    token_count = 0
    accumulated_text = ""
    stream_failed = False
    _stream_tool_names_called: set = set()

    try:
        async for event in active_agent.astream_events(
            {"messages": all_messages},
            version="v2",
            config={"recursion_limit": 100},
        ):
            kind = event["event"]

            # Token from LLM
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content and isinstance(chunk.content, str):
                    token_count += 1
                    accumulated_text += chunk.content
                    yield chunk.content

            # Tool is being called
            elif kind == "on_tool_start":
                tool_name = event.get("name", "unknown")
                tool_input = event.get("data", {}).get("input", {})
                tool_call_count += 1
                _stream_tool_names_called.add(tool_name)
                if tool_name in _DATA_TOOLS:
                    data_tool_count += 1
                print(f"\n  {C.CYAN}🔧 TOOL CALL #{tool_call_count}{C.RESET}")
                print(f"     Name  : {C.BOLD}{tool_name}{C.RESET}")
                print(f"     Input : {tool_input}")
                yield f"\n\n🔧 *Calling {tool_name}...*\n\n"

            # Tool returned result
            elif kind == "on_tool_end":
                tool_name = event.get("name", "unknown")
                output = event.get("data", {}).get("output", "")
                output_str = str(output)
                output_len = len(output_str)
                preview = output_str[:150]
                print(f"     {C.GREEN}✅ Result ({output_len} chars): {preview}{'...' if output_len > 150 else ''}{C.RESET}")
                if 150 < output_len <= 1000:
                    print(f"     {C.GREEN}📄 Full output:{C.RESET}")
                    for line in output_str.split('\n')[:30]:
                        print(f"        {line}")
                elif output_len > 1000:
                    print(f"     {C.YELLOW}⚠️  Output too large ({output_len} chars) - showing preview only{C.RESET}")
    except Exception as stream_err:
        print(f"\n  {C.YELLOW}⚠️  Streaming failed: {stream_err}{C.RESET}")
        print(f"  {C.YELLOW}  Falling back to non-streaming invoke...{C.RESET}")
        stream_failed = True

    # Fallback: if streaming produced nothing, use non-streaming ainvoke
    if stream_failed or (token_count == 0 and tool_call_count == 0 and not accumulated_text):
        print(f"\n  {C.YELLOW}🔄 Streaming produced no output — falling back to ainvoke{C.RESET}")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await active_agent.ainvoke({"messages": all_messages}, config={"recursion_limit": 100})
                ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
                if ai_messages:
                    accumulated_text = ai_messages[-1].content
                    yield accumulated_text
                    token_count = len(accumulated_text)
                break
            except Exception as invoke_err:
                err_str = str(invoke_err)
                if attempt < max_retries - 1 and ("no choices" in err_str.lower() or "500" in err_str):
                    import asyncio
                    wait_secs = 2 * (attempt + 1)
                    print(f"  {C.YELLOW}⚠️  Attempt {attempt+1} failed: {err_str[:100]}. Retrying in {wait_secs}s...{C.RESET}")
                    await asyncio.sleep(wait_secs)
                else:
                    yield f"\n\n❌ Error: {invoke_err}"
                    return
            return

    # Post-stream: detect text tool calls and execute them in a loop
    if tool_call_count == 0 and _looks_like_tool_call(accumulated_text):
        print(f"\n  {C.YELLOW}⚠️  Model streamed a text tool call — executing iteratively{C.RESET}")
        
        conversation = all_messages.copy()
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # If this is first iteration, we already have accumulated_text as the tool call
            if iteration == 1:
                current_tool_call = accumulated_text
            else:
                # Get next response from agent
                result = await active_agent.ainvoke({"messages": conversation}, config={"recursion_limit": 100})
                ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
                if not ai_messages:
                    break
                current_tool_call = ai_messages[-1].content
            
            # Check if it's still a tool call
            if not _looks_like_tool_call(current_tool_call):
                # Got final answer, stream it
                yield f"\n\n{current_tool_call}"
                break
            
            # Execute the tool call
            tool_result = _execute_text_tool_call(current_tool_call)
            if not tool_result:
                yield f"\n\n❌ Tool execution failed"
                break
            
            # Don't show step numbers to user
            if iteration == 1:
                yield f"\n\n🔧 *Executing query...*\n\n"
            
            # Add to conversation
            conversation.append(AIMessage(content=current_tool_call))
            conversation.append(HumanMessage(content=f"Tool result: {tool_result}"))
        
        if iteration >= max_iterations:
            yield f"\n\n⚠️ Max iterations reached"

    # ── Phase 3 critic (streaming path) ────────────────────────────────
    needed = _detect_needed_tool(question)
    print(f"\n{C.CYAN}  🕵️  CRITIC CHECK (stream): needed={needed!r}, stream_tools_seen={sorted(_stream_tool_names_called)}{C.RESET}", flush=True)
    if needed and needed not in _stream_tool_names_called:
        print(f"\n{C.YELLOW}  🕵️  CRITIC: `{needed}` was NOT called during stream (question keyword-matched to it). Re-invoking once (non-stream).{C.RESET}")
        yield f"\n\n_🕵️ Verifying with `{needed}`..._\n\n"
        try:
            conversation2 = all_messages.copy()
            conversation2.append(AIMessage(content=accumulated_text or ""))
            conversation2.append(HumanMessage(content=_critic_directive(needed, question)))
            result = await active_agent.ainvoke({"messages": conversation2}, config={"recursion_limit": 100})
            ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
            if ai_messages:
                new_answer = ai_messages[-1].content or ""
                if new_answer:
                    accumulated_text += "\n\n" + new_answer
                    yield "\n\n" + new_answer
            called_after = _specific_tool_was_called(result.get("messages", []), needed)
            if called_after:
                print(f"{C.GREEN}  ✅ CRITIC recovered: `{needed}` was called after directive.{C.RESET}")
            else:
                print(f"{C.YELLOW}  ⚠️  CRITIC ran but `{needed}` still not called.{C.RESET}")
        except Exception as critic_err:
            print(f"{C.YELLOW}  ⚠️  CRITIC re-invoke failed: {critic_err}{C.RESET}")

    elapsed = time.time() - start_time
    print(f"\n{C.GREEN}  ✅ STREAM COMPLETE ({elapsed:.1f}s){C.RESET}")
    print(f"  ├─ Tokens streamed : {token_count}")
    print(f"  ├─ Tool calls      : {tool_call_count} (data tools: {data_tool_count})")
    print(f"  └─ Response length : {len(accumulated_text)} chars")
    print(f"\n  {C.CYAN}💬 LLM RESPONSE:{C.RESET}")
    if tool_call_count == 0:
        print(f"  {C.YELLOW}📋 NO TOOL CALLS - LLM answered from context/memory{C.RESET}")
    # Show response preview
    for i, line in enumerate(accumulated_text.split('\n')[:20]):  # Max 20 lines
        print(f"     {line}")
        if i >= 19 and len(accumulated_text.split('\n')) > 20:
            print(f"     ... ({len(accumulated_text.split('\n')) - 20} more lines)")
            break
    print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS — Same as original (text tool call detection)
# ══════════════════════════════════════════════════════════════════════════════


def _extract_json_tool_call(text: str) -> str | None:
    """
    Extract JSON tool call from text, handling:
    1. Pure JSON: {"name": "run_query", ...}
    2. Explanation + JSON: "I'll run a query...\n{"name": ...}"
    3. Markdown code blocks: ```json\n{"name": ...}\n```
    4. XML-style: <tool_call>{"name": ...}</tool_call>
    Returns the JSON string if found, None otherwise.
    """
    import re
    text = text.strip()
    
    # Case 0: <tool_call> tags (GPT models via Copilot proxy)
    tool_call_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
    if tool_call_match:
        json_str = tool_call_match.group(1).strip()
        try:
            data = json.loads(json_str)
            if "name" in data or "tool" in data:
                return json_str
        except (json.JSONDecodeError, TypeError):
            pass
    
    # Case 1: Pure JSON
    if text.startswith("{"):
        try:
            data = json.loads(text)
            if "name" in data or "tool" in data:
                return text
        except (json.JSONDecodeError, TypeError):
            pass
    
    # Case 2: Markdown code block (```json ... ``` or ``` ... ```)
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if code_block_match:
        json_str = code_block_match.group(1).strip()
        try:
            data = json.loads(json_str)
            if "name" in data or "tool" in data:
                return json_str
        except (json.JSONDecodeError, TypeError):
            pass
    
    # Case 3: Explanation text followed by bare JSON
    if '{"name":' in text or '{"tool":' in text:
        start_idx = text.find('{')
        if start_idx != -1:
            json_part = text[start_idx:]
            # Remove trailing markdown backticks if present
            backtick_idx = json_part.find('```')
            if backtick_idx != -1:
                json_part = json_part[:backtick_idx]
            try:
                data = json.loads(json_part.strip())
                if "name" in data or "tool" in data:
                    return json_part.strip()
            except (json.JSONDecodeError, TypeError):
                pass
    
    return None


def _looks_like_tool_call(text: str) -> bool:
    """Check if text looks like a JSON tool call from the model."""
    if not text:
        return False
    # Quick check for <tool_call> tags before expensive JSON parsing
    if "<tool_call>" in text:
        return True
    return _extract_json_tool_call(text) is not None


def _execute_text_tool_call(text: str) -> str | None:
    """Parse and execute a tool call that the model output as plain text."""
    try:
        # Use the shared extraction function
        json_str = _extract_json_tool_call(text)
        if not json_str:
            log("INVOKE", "❌ Could not extract JSON tool call from text")
            return None
        
        data = json.loads(json_str)
        tool_name = data.get("name") or data.get("tool")
        args = data.get("arguments") or data.get("args", {})

        # Combine all available tools for fallback
        all_available_tools = ALL_TOOLS + ALL_GITHUB_TOOLS
        if OBS_TOOLS:
            all_available_tools += OBS_TOOLS
        if SLACK_TOOLS:
            all_available_tools += SLACK_TOOLS
        tool_map = {t.name: t for t in all_available_tools}
        if tool_name not in tool_map:
            log("INVOKE", f"❌ Unknown tool in text: '{tool_name}'")
            return None

        log("INVOKE", f"🔧 Manually executing: {tool_name}({args})")
        result = tool_map[tool_name].invoke(args)
        log("INVOKE", f"✅ Manual tool result: {str(result)[:200]}")
        return result
    except Exception as e:
        log("INVOKE", f"❌ Failed to execute text tool call: {e}")
        return None
