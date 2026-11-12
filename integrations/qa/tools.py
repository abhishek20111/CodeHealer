"""LangChain tools exposing the QA automation stack to the supervisor.

Each tool is a thin wrapper around the frontend/backend HTTP clients. The
sub-agent decides which one to call and how to format the report for the user.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from . import backend_client, frontend_client


def _dump(payload: Any) -> str:
    return json.dumps(payload, indent=2, default=str)


# ══════════════════════════════════════════════════════════════════════════════
# Frontend (UI Comparator & Recorder) — Playwright dashboard on :3737
# ══════════════════════════════════════════════════════════════════════════════


@tool
def qa_frontend_health() -> str:
    """Check whether the UI Comparator dashboard (frontend automation tool) is reachable."""
    return _dump(frontend_client.health())


@tool
def qa_list_ui_sessions() -> str:
    """List all recorded UI flows available in the frontend automation tool."""
    return _dump(frontend_client.list_sessions())


@tool
def qa_get_ui_session(session_name: str) -> str:
    """Get details (routes, actions) for a specific recorded UI flow."""
    return _dump(frontend_client.get_session(session_name))


@tool
def qa_list_ui_reports(session_name: str) -> str:
    """List generated reports (compare or functional) for a recorded UI flow."""
    return _dump(frontend_client.list_reports(session_name))


@tool
def qa_run_ui_comparison(session_name: str, env1_url: str, env2_url: str) -> str:
    """Replay a recorded UI flow against two environments and produce a pixel-diff report."""
    return _dump(frontend_client.run_compare(session_name, env1_url, env2_url))


@tool
def qa_run_ui_functional(session_name: str, env_url: str) -> str:
    """Replay a recorded UI flow against one environment as a functional test."""
    return _dump(frontend_client.run_functional(session_name, env_url))


# ══════════════════════════════════════════════════════════════════════════════
# Backend (DataVeriTest) — Spring Boot on :9090
# ══════════════════════════════════════════════════════════════════════════════


# ── Target & status ─────────────────────────────────────────────────────────


@tool
def qa_backend_health() -> str:
    """Check whether the DataVeriTest backend automation tool is reachable."""
    return _dump(backend_client.health())


@tool
def qa_connect_target_api(target_url: str, openapi_path: str = "/v3/api-docs") -> str:
    """Point the backend automation tool at a Spring Boot API to discover its endpoints.

    Args:
        target_url: Base URL of the target API (e.g. http://localhost:8080).
        openapi_path: Path to the OpenAPI spec on the target (default /v3/api-docs).
    """
    return _dump(backend_client.connect_target(target_url, openapi_path))


@tool
def qa_get_target_status() -> str:
    """Get the current target API connection status and reachability."""
    return _dump(backend_client.target_status())


@tool
def qa_get_target_db_config() -> str:
    """Fetch the DB configuration reported by the currently connected target API."""
    return _dump(backend_client.target_db_config())


@tool
def qa_get_backend_status() -> str:
    """Full backend snapshot: target info, all DB connections, and whether a run is in progress."""
    return _dump(backend_client.full_status())


# ── Endpoints & runs ────────────────────────────────────────────────────────


@tool
def qa_list_api_endpoints() -> str:
    """List every API endpoint discovered from the connected target OpenAPI spec."""
    return _dump(backend_client.list_endpoints())


@tool
def qa_run_all_tests(excluded_endpoints_json: str = "[]") -> str:
    """Run the full test battery against every discovered endpoint.

    Args:
        excluded_endpoints_json: JSON array of endpoint paths to skip, e.g. '["/api/health"]'.
    """
    excluded = _parse_list(excluded_endpoints_json)
    return _dump(backend_client.run_all(excluded))


@tool
def qa_run_smoke_tests(excluded_endpoints_json: str = "[]") -> str:
    """Run smoke tests (basic GET/POST/PUT/DELETE reachability) against discovered endpoints."""
    excluded = _parse_list(excluded_endpoints_json)
    return _dump(backend_client.run_smoke(excluded))


@tool
def qa_run_health_tests(excluded_endpoints_json: str = "[]") -> str:
    """Run health check tests against the target API."""
    excluded = _parse_list(excluded_endpoints_json)
    return _dump(backend_client.run_health(excluded))


@tool
def qa_run_performance_tests(excluded_endpoints_json: str = "[]") -> str:
    """Run performance tests (response time thresholds) against the target API."""
    excluded = _parse_list(excluded_endpoints_json)
    return _dump(backend_client.run_performance(excluded))


@tool
def qa_run_status_code_tests(excluded_endpoints_json: str = "[]") -> str:
    """Run status-code validation tests against the target API."""
    excluded = _parse_list(excluded_endpoints_json)
    return _dump(backend_client.run_status_code(excluded))


@tool
def qa_run_content_type_tests(excluded_endpoints_json: str = "[]") -> str:
    """Run content-type validation tests against the target API."""
    excluded = _parse_list(excluded_endpoints_json)
    return _dump(backend_client.run_content_type(excluded))


@tool
def qa_get_last_test_results() -> str:
    """Return the results of the most recent test run without re-running."""
    return _dump(backend_client.last_results())


@tool
def qa_execute_single_endpoint(request_json: str) -> str:
    """Execute a single API request (Swagger-like 'Try it out').

    Args:
        request_json: JSON object with keys: endpoint (path), method, contentType,
            requestBody, headers, queryParams, pathParams. Example:
            '{"endpoint":"/api/employees/{id}","method":"GET","pathParams":{"id":5}}'
    """
    try:
        payload = json.loads(request_json)
    except json.JSONDecodeError as exc:
        return _dump({"error": f"Invalid JSON: {exc}"})
    return _dump(backend_client.execute_single(payload))


# ── Databases ───────────────────────────────────────────────────────────────


@tool
def qa_list_db_connections() -> str:
    """List all databases currently connected to the backend automation tool."""
    return _dump(backend_client.db_statuses())


@tool
def qa_get_db_relationships() -> str:
    """Return foreign-key relationships and columns of all tables in the connected DB."""
    return _dump(backend_client.db_relationships())


@tool
def qa_db_query(query: str) -> str:
    """Execute a read-only SQL query (SELECT / SHOW / DESCRIBE) against the connected DB.

    Args:
        query: SQL string; only SELECT, SHOW, and DESCRIBE are permitted.
    """
    return _dump(backend_client.db_query(query))


# ── Sessions ────────────────────────────────────────────────────────────────


@tool
def qa_list_backend_sessions() -> str:
    """List saved DataVeriTest sessions (test configurations)."""
    return _dump(backend_client.list_sessions())


@tool
def qa_get_backend_session(filename: str) -> str:
    """Load a saved DataVeriTest session by filename."""
    return _dump(backend_client.get_session(filename))


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _parse_list(text: str) -> list[str]:
    if not text:
        return []
    try:
        val = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(val, list):
        return [str(x) for x in val]
    return []


QA_TOOLS = [
    # Frontend
    qa_frontend_health,
    qa_list_ui_sessions,
    qa_get_ui_session,
    qa_list_ui_reports,
    qa_run_ui_comparison,
    qa_run_ui_functional,
    # Backend — target + status
    qa_backend_health,
    qa_connect_target_api,
    qa_get_target_status,
    qa_get_target_db_config,
    qa_get_backend_status,
    # Backend — endpoints & runs
    qa_list_api_endpoints,
    qa_run_all_tests,
    qa_run_smoke_tests,
    qa_run_health_tests,
    qa_run_performance_tests,
    qa_run_status_code_tests,
    qa_run_content_type_tests,
    qa_get_last_test_results,
    qa_execute_single_endpoint,
    # Backend — DB
    qa_list_db_connections,
    qa_get_db_relationships,
    qa_db_query,
    # Backend — sessions
    qa_list_backend_sessions,
    qa_get_backend_session,
]


def create_qa_tools() -> list:
    """Return the flat list of QA tools for use by the supervisor."""
    return QA_TOOLS
