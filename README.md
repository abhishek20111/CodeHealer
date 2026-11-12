# 🩺 CodeHealer — AI-Powered Autonomous Software Maintenance Platform

**Short description:** CodeHealer is an AI multi-agent assistant that heals aging enterprise applications. It investigates production errors, scans for vulnerable dependencies, and prepares validated fix Pull Requests with human approval — turning hours of manual maintenance into minutes.

**An AI-powered multi-agent system that continuously monitors, investigates, and fixes production issues autonomously — from detecting vulnerabilities and errors to creating Pull Requests with validated fixes.**

---

## 🔥 The Problem

Every enterprise application is expected to run for years. During that lifecycle, libraries get deprecated, security vulnerabilities are discovered, and APIs change. But engineering teams are always focused on shipping features — maintenance gets postponed. Technical debt accumulates silently until the dashboard lights up red: critical CVEs, unsupported libraries, compliance failures. Entire teams stop feature development to firefight problems that could have been prevented.

**Today's tools detect problems — but they cannot fix them.** Dependabot tells you "jackson-dataformat-cbor has 8 critical CVEs" — but it doesn't read your pom.xml, check if your code imports that package, create a branch, patch the version, verify no API broke, and open a Pull Request. Engineers still do all of that manually — **30 to 60 minutes per issue**, across hundreds of applications, **thousands of engineering hours every year**.

---

## 💡 The Solution

**CodeHealer doesn't just detect — it fixes.** 

Four capabilities no existing tool provides:

1. **Autonomous Remediation** — Reads the dependency file, applies a surgical patch, searches the codebase for affected imports, and creates a production-ready PR
2. **Scheduled Workflows (WatchFlow)** — Runs every night, scans for new vulnerabilities, creates fix PRs while engineers sleep
3. **Human-in-the-Loop Governance** — Every change goes through interactive confirmation before any code is committed
4. **Post-Change QA Automation** — After a fix ships, CodeHealer can trigger optional external QA tools (a Playwright UI comparator and an API ↔ DB validator) to verify the change didn't regress the app, then email a report to the requester (Outlook integration coming next)

> **Working prototype: 12,000+ lines of code • 69 tools • 7 agents • 7 enterprise / local systems • 6 real PRs created**

---

## 🎯 What is CodeHealer?

CodeHealer is an autonomous maintenance platform for engineering teams. It connects to your entire DevOps ecosystem — vulnerability scanners (VulnScan), log observability (OBS), code repositories (GitHub Enterprise), databases, and communication channels (Slack). Instead of engineers manually researching dependency upgrades, checking logs, and creating fixes, CodeHealer does it all through natural language conversation or fully automated scheduled workflows with human-in-the-loop governance.

### The Problem It Solves

```
Traditional workflow (30-60 min):
  1. Engineer notices alert
  2. Opens OBS/Splunk, searches for errors        → 5 min
  3. Identifies error code, searches codebase      → 10 min
  4. Reads exception handler, understands root cause → 10 min
  5. Creates branch, writes fix, opens PR          → 15 min
  6. Posts update to Slack                         → 5 min

CodeHealer workflow (5 min):
  1. Ask: "Check errors for my app and fix the top issue"
  2. CodeHealer investigates, asks for confirmation
  3. You click "Create PR"
  4. Done ✅

Autonomous workflow (0 min engineer time):
  1. WatchFlow runs every night at 1:00 AM
  2. Scans VulnScan for new critical vulnerabilities
  3. Auto-creates fix branches and PRs
  4. Engineer reviews in the morning ☕
```

---

## ✨ Key Benefits

| Benefit | Description |
|---------|-------------|
| **10x Faster Incident Response** | From error detection to PR in minutes, not hours |
| **Cross-Domain Intelligence** | Correlates logs + code + database + Slack in one query |
| **Human-in-the-Loop** | Never makes changes without your explicit confirmation |
| **Semantic Memory** | Remembers past conversations — "what was that error from yesterday?" |
| **Enterprise Ready** | Works behind corporate proxies, connects to GitHub Enterprise |

---

## 🏗️ Architecture

### High-Level Overview


```
┌──────────────────────────────────────────────────────────────────-───┐
│                     Web UI / Chat Interface                          │
│                     http://localhost:3002                            │
├──────────────────────────────────────────────────────────────-───────┤
│                  Starlette Server + SSE Streaming                    │
├───────────────────────────────────────────────────────────-──────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────-───┐ │
│  │               🧠 SUPERVISOR AGENT                               │ │
│  │                                                                 │ │
│  │  • Decomposes complex tasks into sub-tasks                      │ │
│  │  • Delegates to the right sub-agent                             │ │
│  │  • Synthesizes results from multiple agents                     │ │
│  │  • Asks user for confirmations when needed                      │ │
│  │  • pre_model_hook: auto-trims context to prevent overflow       │ │
│  │                                                                 │ │
│  └────────┬──────────┬──────────┬──────────┬───────────┬─────────-─┘ │
│           │          │          │          │           |             │
│  ┌────────▼───┐ ┌───▼────-─┐ ┌──▼─────┐ ┌──▼──────┐ ┌──▼──────┐      │
│  │  Database  │ │  GitHub  │ │  OBS   │ │  Slack  │ │   VulnScan   │      │
│  │   Agent    │ │  Agent   │ │  Agent │ │  Agent  │ │  Agent  │      │
│  │  4 tools   │ │ 13 tools │ │ 3 tools│ │ 9 tools │ │ 5 tools │      │
│  └────────────┘ └──────────┘ └────────┘ └─────────┘ └─────────┘      │
│                                                                      │
├──────────────────────────────┬───────────────────────────────────────┤
│   ChromaDB + nomic-embed-text │          LLM                         |
│   Semantic Vector Memory      │   localhost:3100 (OpenAI-compatible) │
└──────────────────────────────┴───────────────────────────────────────┘
```

### How a Request Flows — Vulnerability Fix (VulnScan)

```
┌─────────────────────────────────────────────────────────────┐
│                    USER QUESTION                              │
│  "Show critical vulns in sampleapp, fix the top one, make PR" │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  1️⃣ SUPERVISOR → VulnScan Agent                          │
│     query_vulnscan("get critical vulns for APP_ID 100000001")│
│     → GraphQL query to vulnscan.example.com                  │
│     → Returns: 21 critical (jackson-dataformat-cbor@2.17.0)  │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  2️⃣ SUPERVISOR → GitHub Agent                                │
│     query_github("read sampleapp-api/pom.xml")              │
│     → Confirms jackson-dataformat-cbor version 2.17.0        │
│     → Safe version: 2.22.1                                   │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  3️⃣ SUPERVISOR → GitHub Agent                                │
│     create_branch("ai-agent/fix-jackson-vuln")               │
│     patch_file(pom.xml, "2.17.0" → "2.22.1")                │
│     search_github_code("com.fasterxml.jackson.dataformat")   │
│     → No import usages found — safe to upgrade               │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  4️⃣ SUPERVISOR → GitHub Agent                                │
│     create_pull_request(title, description)                   │
│     → PR #6 created ✅                                        │
└──────────────────────┬────────────────────────────────────┘
                       │  (optional — asks user)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  5️⃣ (optional) SUPERVISOR → QA Agent                          │
│     qa_backend_health → DataVeriTest reachable                │
│     qa_run_smoke_tests → verifies /api endpoints still 200s   │
│     qa_db_query        → confirms row counts unchanged        │
│     → Smoke: 27 pass / 0 fail ✅                               │
└──────────────────────┬──────────────────────────────────--──┘
                       │  (roadmap — GitHub Actions feedback)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  6️⃣ (roadmap) CI VERIFICATION LOOP                            │
│     GitHub Agent watches Actions run for PR #6.               │
│     If FAIL: reads workflow logs → identifies broken import   │
│     → patch_file(fix) → commit_file → re-run action           │
│     Repeats up to N attempts before asking human.             │
└─────────────────────────────────────────────────────────────┘
```

### How a Request Flows — Error Investigation (OBS)

```
┌─────────────────────────────────────────────────────────────┐
│                    USER QUESTION                            │
│  "Check OBS errors for sampleapi, find root cause, create PR"  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────-─┐
│  1️⃣ SEMANTIC CONTEXT RETRIEVAL                               │
│     Query ChromaDB for relevant past conversations           │
│     nomic-embed-text → cosine similarity → top-K results     │
│     Each retrieved message capped at 1500 chars              │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────────────────-──┐
│  2️⃣ SUPERVISOR DECIDES                                       │
│     "This needs OBS logs + GitHub code search + user input"  │
│     → Calls query_obs_logs("get errors for sampleapi, 2h")      │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────-────┐
│  3️⃣ SUB-AGENT EXECUTES (OBS)                                 |
│     OBS Agent → get_obs_errors(app="sampleapi", time="2h")      │
│     Returns: "18 APP-1000 errors on /optimizations/*"        │
│     Response capped at 4000 chars                            │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────-───┐
│  4️⃣ SUPERVISOR CONTINUES                                      │
│     → Calls query_github("search APP-1000 in sampleapp")       │
│     → Reads GlobalExceptionHandler.java                      │
│     → Identifies: NoResourceFoundException not handled       │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────-───┐
│  5️⃣ ASKS USER (Interactive Q&A via SSE)                      │
│     "Root cause found. Proceed with fix?"                    │
│     [Proceed with fix]  [Cancel]                             │
│                                                              │
│     User clicks → POST /task/respond                         │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  6️⃣ EXECUTES FIX                                            │
│     → create_branch("ai-agent/fix-app1000-...")             │
│     → commit_file(modified GlobalExceptionHandler.java)     │
│     → create_pull_request(title, description)               │
│     → PR #4 created ✅                                      │
└──────────────────────┬──────────────────────────────────────┘
                       │  (optional — asks user before running)
                       ▼
┌────────────────────────────────────────────────────────────-─┐
│  7️⃣ (optional) QA VERIFICATION                                │
│     Supervisor → QA Agent                                     │
│     • qa_run_ui_functional("login-flow", NEW env)              │
│       replays recorded SPA flow against the fixed build.       │
│     • qa_run_smoke_tests → 27 endpoints → 27 pass ✅             │
│     • qa_get_last_test_results → fetches HTML report link      │
│     Result → report summary posted back to user + Slack.       │
└──────────────────────┬───────────────────────────────────-─┘
                       │  (roadmap — not yet implemented)
                       ▼
┌────────────────────────────────────────────────────────────-─┐
│  8️⃣ (roadmap) GITHUB ACTIONS FEEDBACK LOOP                    │
│     GitHub Agent polls the PR's Actions run.                  │
│     If build/CI fails:                                        │
│       • get_workflow_logs → finds failing step + stack trace  │
│       • identifies root cause (missing import, bad SQL, etc.) │
│       • patch_file → commit_file → re-triggers workflow       │
│     If Actions still fail after N attempts → asks the human.  │
└──────────────────────┬───────────────────────────────────-─┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────-────┐
│  9️⃣ STORE & RESPOND                                          │
│     Clean response → ChromaDB (semantic memory, 4K cap)      │
│     Full response → sessions.json (UI display)               │
│     Stream → SSE to Web UI                                   │
└────────────────────────────────────────────────────────-─────┘
```

### Context Management (Why it doesn't crash on long tasks)

```
┌──────────────────────────────────────────────────────-───────┐
│               CONTEXT OPTIMIZATION LAYERS                    │
├───────────────────────────────────────────────────────-──────┤
│                                                              │
│  Layer 1: pre_model_hook (internal ReAct loop)               │
│  ┌─────────────────────────────────────────────────────┐     │
│  │ Before each LLM call inside supervisor:             │     │
│  │ If total chars > 16K → keep first msg + last 6 msgs │     │
│  │ Drops old tool results that are no longer needed    │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                              │
│  Layer 2: Sub-agent response cap (4000 chars)                │
│  ┌─────────────────────────────────────────────────────┐     │
│  │ Each sub-agent returns max 4K chars to supervisor   │     │
│  │ Prevents single tool result from bloating context   │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                              │
│  Layer 3: Storage cleanup                                    │
│  ┌─────────────────────────────────────────────────────┐     │
│  │ Strip tool markers (🔧 Calling...) before storing   │     │
│  │ Cap stored response at 4000 chars                   │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                              │
│  Layer 4: Retrieval cap (1500 chars per message)             │
│  ┌─────────────────────────────────────────────────────┐     │
│  │ When loading history for next request:              │     │
│  │ Each past message capped at 1500 chars              │     │
│  │ Prevents giant past responses from consuming budget │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                              │
└────────────────────────────────────────────────────────────-─┘
```

---

## 🚀 Setup Guide

### Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| Python 3.12+ | Agent runtime | `brew install python` |
| Node.js 18+ | LLM proxy | `brew install node` |
| Ollama | Local embeddings | `brew install ollama` |
| MySQL | Database agent | `brew install mysql` |

### Step 1: Clone & Install

```bash
cd "/path/to/CodeHealer"
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Start Ollama (for embeddings)

```bash
ollama pull nomic-embed-text
ollama serve  # runs on localhost:11434
```

### Step 3: Configure Environment

```bash
cd CodeHealer
cp .env.example .env
# Edit .env with your credentials (see Configuration section below)
```

### Step 4: Start CodeHealer Server

```bash
cd CodeHealer
python web_server.py
# → Server ready: http://localhost:3002
```

### Step 5: (Optional) Connect the QA Automation Tools

The QA tools are separate services and are not included in this repository. If you run your own UI-comparison dashboard and API-validation service, point CodeHealer at them:

```bash
QA_FRONTEND_URL=http://localhost:3737
QA_BACKEND_URL=http://localhost:9090
```

The supervisor's `query_qa` tool calls these two services over HTTP. Without them, the QA agent simply reports them as unreachable.
Override ports with `QA_FRONTEND_URL` / `QA_BACKEND_URL` env vars.

### Step 6: Open Web UI

```
http://localhost:3002
```

---

## 💬 Example Questions & Answers

### 🗄️ Database Agent
| Question | Answer |
|----------|--------|
| "Show me top 3 highest paid employees" | Table: Karthik Rao ($115K), Suresh Menon ($105K), Ravi Prasad ($98K) |
| "How many projects are active?" | 5 active projects in the database |
| "Which department has the most employees?" | Engineering with 8 employees |
| "Show me employees hired after 2023" | Lists employees with hire dates |

### 📊 OBS Agent (Log Observability)
| Question | Answer |
|----------|--------|
| "Get errors for sampleapi in the last 1 hour" | 110 errors: APP-1000 (API 500s), Batch Submit failures |
| "Count logs for sampleapp in last 6 hours" | 2,847 log entries (1,203 ERROR, 1,644 WARN) |
| "Search traces for traceId abc123" | Full trace with timestamps, spans, and error details |

### 💻 GitHub Agent
| Question | Answer |
|----------|--------|
| "List files in the controller directory" | 24 Java controller classes with sizes |
| "Search for APP-1000 in sampleapp repo" | Found in ErrorCode.java — INTERNAL_SERVER_ERROR mapping |
| "Read GlobalExceptionHandler.java" | Full file content with exception handlers |
| "Create a branch and PR with the fix" | Branch created, file committed, PR #4 opened |

### 💬 Slack Agent
| Question | Answer |
|----------|--------|
| "Search app-errors channel for recent issues" | 12 messages about deployment failures |
| "Post error summary to team-channel" | ✅ Message posted to #team-channel |
| "What did the team discuss about the outage?" | Summary of channel conversation |

### 🧠 Multi-Agent (Full E2E Workflow)
```
User: "Check OBS errors for sampleapi in the last 2 hours, investigate 
       the top APP-1000 error in the code, then create a fix PR. 
       Ask me before creating the branch."

CodeHealer Flow:
  ┌─ Step 1: OBS Agent finds 18 APP-1000 errors (500 Internal Server Error)
  ├─ Step 2: GitHub Agent searches code, reads GlobalExceptionHandler.java
  ├─ Step 3: Identifies root cause: NoResourceFoundException not handled
  ├─ Step 4: Asks user → "Proceed with fix?" → User: "Yes"
  ├─ Step 5: Asks user → "Create PR on branch X?" → User: "Create PR"
  ├─ Step 6: Creates branch, commits fix, opens PR #4
  └─ Done ✅ (Total time: ~5 minutes)
```

### 🔒 VulnScan Agent (Vulnerability Scanning)
```
User: "Show me critical vulnerabilities in sampleapp"

CodeHealer:
  → VulnScan Agent queries GraphQL API (vulnscan.example.com)
  → Returns: 21 critical, 103 high, 87 medium, 15 low
  → Top critical: axios@1.13.6 (29 CVEs, safe: 1.18.1)
  → jackson-dataformat-cbor@2.17.0 (8 CVEs, safe: 2.22.1)
```

```
User: "Update jackson-dataformat-cbor to 2.22.1 in pom.xml, 
       search for code usages, and create a PR"

CodeHealer:
  ┌─ Step 1: Creates branch ai-agent/upgrade-jackson-cbor-v2
  ├─ Step 2: patch_file updates pom.xml (2.17.0 → 2.22.1)
  ├─ Step 3: search_github_code checks for import usages → none found
  └─ Step 4: Creates PR #6 to self-serve ✅
```

### ⏰ WatchFlow (Scheduled Autonomous Workflows)
```
Workflow: "SampleApp Vulnerability Monitor"
  Schedule: Every 60 minutes
  Steps:
    1. Scan VulnScan → get critical vulnerabilities
    2. Auto Fix → if CRITICAL found, create branch + patch + PR
  
  Runs autonomously — no human intervention needed for scanning.
  Human-in-the-loop for PR merge decisions.
```

### 🧪 QA Automation Agent (UI Comparator + DataVeriTest)
```
User: "The pom.xml upgrade PR is merged.
       Run the login flow comparison between https://old.sampleapp.example.com
       and https://new.sampleapp.example.com, and run the /users API validation."

CodeHealer (QA Agent):
  ┌─ Step 1: qa_frontend_health → dashboard reachable on :3737 ✅
  ├─ Step 2: qa_list_ui_sessions → finds recorded flow "login-happy-path"
  ├─ Step 3: qa_run_ui_comparison → replays flow on OLD vs NEW, produces pixel diff
  ├─ Step 4: qa_backend_health → DataVeriTest reachable on :9090 ✅
  ├─ Step 5: qa_connect_target_api → connects to new API, discovers endpoints
  ├─ Step 6: qa_run_smoke_tests → executes stored /users test cases, compares vs DB
  └─ Step 7: Summarizes pass/fail counts + report links back to the user
```

**External tools (not included in this repo):**

| Tool | Language | Runs on | Purpose |
|---|---|---|---|
| **UI Comparator & Recorder** | Node / Playwright | `localhost:3737` | Record SPA flows, replay across environments, generate pixel-diff reports |
| **DataVeriTest** | Java 21 / Spring Boot | `localhost:9090` | Auto-discover API endpoints, generate SQL, validate API responses against DB |

**Roadmap:** after a fix ships, WatchFlow will call the QA agent to verify the change and email a report to the requester via Outlook.

---

## 🔌 API Reference

### Main Chat Endpoint (SSE Streaming)
```bash
curl -N -X POST http://localhost:3002/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "show me all employees", "session_id": "my-session"}'
```

**Response (SSE events):**
```
data: {"type": "session_id", "content": "my-session"}
data: {"type": "task_session_id", "content": "uuid-for-interactive"}
data: {"type": "token", "content": "Here are the employees..."}
data: {"type": "question", "content": {"text": "...", "options": [...]}}
data: {"type": "done"}
```

### Respond to Interactive Question
```bash
curl -X POST http://localhost:3002/task/respond \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<task_session_id>", "response": "Proceed with fix"}'
```

### Other Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | Server health check |
| `/sessions/list` | GET | All conversation sessions |
| `/sessions/{id}` | GET | Single session with messages |
| `/db/info` | GET | Database connection status |
| `/db/schema-status` | GET | Schema loading progress |
| `/watchflows/list` | GET | All workflow definitions |
| `/watchflows/create` | POST | Create/update workflow |
| `/watchflows/trigger` | POST | Run workflow immediately |
| `/watchflows/toggle` | POST | Enable/disable workflow |
| `/watchflows/delete` | POST | Delete workflow |
| `/watchflows/status` | GET | Scheduler status |

---

## ⚙️ Configuration (.env)

```bash
# ═══════════════════════════════════════════════════════
# LLM Backend (
# ═══════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════
# GitHub Enterprise
# ═══════════════════════════════════════════════════════
GHE_TOKEN=ghp_xxxxxxxxxxxx
GHE_HOST=github.example.com

# ═══════════════════════════════════════════════════════
# OBS (Log Observability Platform)
# ═══════════════════════════════════════════════════════
OBS_APP_ID=100000001
OBS_USERNAME=your_username
OBS_PASSWORD=your_password
OBS_ENV=qa                    # qa | prod

# ═══════════════════════════════════════════════════════
# Slack
# ═══════════════════════════════════════════════════════
SLACK_TOKEN=xoxc-xxxxxxxxxxxx
SLACK_COOKIE=xoxd-xxxxxxxxxxxx

# ═══════════════════════════════════════════════════════
# Ollama (Local Embeddings)
# ═══════════════════════════════════════════════════════
OLLAMA_BASE_URL=http://localhost:11434

# ═══════════════════════════════════════════════════════
# MySQL Database
# ═══════════════════════════════════════════════════════
# Configured in json/databases.json
```

---

## 📁 Project Structure

```
CodeHealer/
├── agent.py                  # Supervisor + sub-agent creation + context management
├── web_server.py             # Starlette server, SSE streaming, dual storage
├── tools.py                  # Database tools (list_databases, run_query, etc.)
├── vector_store.py           # ChromaDB semantic memory (embed, store, retrieve)
├── session_manager.py        # Interactive Q&A with asyncio.Event
├── approval_workflow.py      # Background approval monitoring loop
├── requirements.txt          # Python dependencies
├── .env                      # Environment configuration (secrets)
├── .env.example              # Configuration template
│
├── integrations/
│   ├── github/               # GitHub Enterprise (13 tools)
│   │   ├── __init__.py
│   │   └── github_tools.py   # create_branch, commit_file, create_pr, etc.
│   ├── obs/                  # OBS Log Observability (3 tools)
│   │   ├── __init__.py
│   │   ├── query_client.py   # Production curl-based client (bypasses proxy)
│   │   └── tools.py          # search_obs_logs, get_obs_errors, count_obs_logs
│   ├── slack/                # Slack Workspace (9 tools)
│   │   ├── __init__.py
│   │   ├── slack_client.py
│   │   └── tools.py          # search, post, analyze errors
│   ├── vulnscan/         # VulnScan Vulnerability Scanner (5 tools)
│   │   ├── __init__.py
│   │   ├── query_client.py   # GraphQL over curl (vulnscan.example.com)
│   │   └── tools.py          # list_repos, get_vulnerabilities, get_secrets
│   ├── qa/                   # QA Automation MCP (25 tools)  ⭐ NEW
│   │   ├── __init__.py
│   │   ├── frontend_client.py    # HTTP client for UI Comparator (port 3737)
│   │   ├── backend_client.py     # HTTP client for DataVeriTest (port 9090)
│   │   └── tools.py              # 25 tools: UI recorder, comparator, API discovery, test suites, DB verification, ...
│   ├── watchflow/            # Scheduled Autonomous Workflows
│   │   ├── __init__.py
│   │   ├── scheduler.py      # Asyncio-based cron scheduler
│   │   ├── executor.py       # Runs workflow steps via ask_agent()
│   │   └── definitions.py    # JSON workflow CRUD
│   └── ui/                   # Interactive Workflow Tools
│       └── interactive_tools.py  # ask_user_question, show_analysis_result
│
├── copilot_proxy_simple/     # VS Code extension exposing Copilot as OpenAI API (port 3100)
│
├── static/                   # Web UI
│   ├── index.html            # Chat interface
│   ├── script.js             # SSE client, interactive buttons
│   └── style.css             # Styling
│
├── json/                     # Data configuration
│   ├── databases.json        # MySQL connection configs
│   ├── repos.json            # GitHub repositories to search
│   ├── sessions.json         # Session persistence
│   └── watchflows.json       # WatchFlow workflow definitions
│
└── chroma_db/                # Vector database (auto-created)
```

---

## 🛠️ Tools Summary (69 total)

| Agent | Tools | Count |
|-------|-------|-------|
| **Database** | list_databases, list_tables, describe_table, run_query | 4 |
| **GitHub** | list_repos, read_github_file, list_github_files, search_github_code, create_branch, commit_file, patch_file, create_pull_request, get_pr_comments, add_pr_comment, check_workflow_status, get_workflow_logs, list_pull_requests | 13 |
| **OBS** | search_obs_logs, get_obs_errors, count_obs_logs | 3 |
| **Slack** | search_slack_messages, list_slack_channels, get_slack_channel_history, post_slack_message, get_slack_user_info, analyze_last_slack_error, get_recent_errors_from_obs, post_error_analysis_to_slack, create_fix_proposal_for_approval | 9 |
| **VulnScan** | list_scan_repositories, get_scan_vulnerabilities, get_scan_secrets, search_scan_dependency, get_scan_repo_dependencies | 5 |
| **QA Automation — Frontend** | qa_frontend_health, qa_list_ui_sessions, qa_get_ui_session, qa_list_ui_reports, qa_run_ui_comparison, qa_run_ui_functional | 6 |
| **QA Automation — Backend** | qa_backend_health, qa_connect_target_api, qa_get_target_status, qa_get_target_db_config, qa_get_backend_status, qa_list_api_endpoints, qa_run_all_tests, qa_run_smoke_tests, qa_run_health_tests, qa_run_performance_tests, qa_run_status_code_tests, qa_run_content_type_tests, qa_get_last_test_results, qa_execute_single_endpoint, qa_list_db_connections, qa_get_db_relationships, qa_db_query, qa_list_backend_sessions, qa_get_backend_session | 19 |
| **Supervisor** | query_database, query_github, query_obs_logs, query_slack, query_vulnscan, query_qa, ask_user_question, investigate_error_interactive, ask_create_pr, show_analysis_result | 10 |

---

## 🧩 Key Design Decisions

| Decision | Why |
|----------|-----|
| **Supervisor > Router** | Multi-step workflows need orchestration, not single-shot routing |
| **`pre_model_hook` trimming** | Prevents "empty response" when internal context exceeds LLM token limit |
| **Sub-agent response cap (4K)** | Keeps supervisor's working memory focused |
| **Stored response cap (4K)** | Future context retrieval stays concise and relevant |
| **Per-message retrieval cap (1.5K)** | No single past message hogs the context budget |
| **ChromaDB + nomic-embed-text** | Fully local — no external API needed for memory |
| **Curl-based API clients** | Bypasses corporate HTTP proxies |
| **SSE streaming + asyncio.Event** | Real-time tokens + interactive Q&A without polling |

---

##  Performance Characteristics

| Metric | Value |
|--------|-------|
| Simple DB query | ~3-5 seconds |
| OBS log search | ~4-6 seconds |
| GitHub file read | ~3-4 seconds |
| Full E2E (error → PR) | ~5-8 minutes |
| Embedding (per message) | ~50-100ms |
| Context retrieval | ~100-200ms |
| Max internal context | 16,000 chars |
| Max stored response | 4,000 chars |
| Max retrieved message | 1,500 chars |
