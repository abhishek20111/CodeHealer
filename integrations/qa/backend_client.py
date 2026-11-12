"""HTTP client for the Backend Automation Tool (DataVeriTest / api-test-tool).

The tool runs a local Spring Boot server (default port 9090). It exposes the
following API surface used by the CodeHealer MCP integration:

Target connection
    POST   /api/target/connect
    GET    /api/target/status
    GET    /api/target/db-config
Databases
    POST   /api/db/connect
    DELETE /api/db/{id}
    GET    /api/db/status
    GET    /api/db/relationships
    POST   /api/db/keys
    POST   /api/db/query          (read-only: SELECT / SHOW / DESCRIBE)
Endpoints & test suites
    GET    /api/endpoints
    POST   /api/run-all
    POST   /api/run-smoke
    POST   /api/run-health
    POST   /api/run-performance
    POST   /api/run-status-code
    POST   /api/run-content-type
    GET    /api/results
    GET    /api/status
    POST   /api/execute            (Swagger-like single request)
Sessions
    POST   /api/sessions/save
    GET    /api/sessions
    GET    /api/sessions/{name}
    PUT    /api/sessions/{name}
    DELETE /api/sessions/{name}
"""

from __future__ import annotations

import os
from typing import Any

import httpx

BACKEND_BASE_URL = os.environ.get("QA_BACKEND_URL", "http://localhost:9090").rstrip("/")
DEFAULT_TIMEOUT = float(os.environ.get("QA_BACKEND_TIMEOUT", "60"))
# Long-running test runs may take a while; allow separate cap.
RUN_TIMEOUT = float(os.environ.get("QA_BACKEND_RUN_TIMEOUT", "300"))


def _client(timeout: float = DEFAULT_TIMEOUT) -> httpx.Client:
    return httpx.Client(base_url=BACKEND_BASE_URL, timeout=timeout, trust_env=False)


# ── Target ──────────────────────────────────────────────────────────────────

def health() -> dict[str, Any]:
    try:
        with _client(timeout=5) as c:
            r = c.get("/api/status")
            return {"ok": r.status_code == 200, "url": BACKEND_BASE_URL, "status_code": r.status_code}
    except httpx.HTTPError as exc:
        return {"ok": False, "url": BACKEND_BASE_URL, "error": str(exc)}


def connect_target(url: str, openapi_path: str = "/v3/api-docs") -> dict[str, Any]:
    with _client() as c:
        r = c.post("/api/target/connect", json={"url": url, "openApiPath": openapi_path})
        r.raise_for_status()
        return r.json()


def target_status() -> dict[str, Any]:
    with _client() as c:
        r = c.get("/api/target/status")
        r.raise_for_status()
        return r.json()


def target_db_config() -> dict[str, Any]:
    with _client() as c:
        r = c.get("/api/target/db-config")
        r.raise_for_status()
        return r.json()


def full_status() -> dict[str, Any]:
    with _client() as c:
        r = c.get("/api/status")
        r.raise_for_status()
        return r.json()


# ── Databases ───────────────────────────────────────────────────────────────

def connect_db(config: dict[str, Any]) -> dict[str, Any]:
    with _client() as c:
        r = c.post("/api/db/connect", json=config)
        r.raise_for_status()
        return r.json()


def db_statuses() -> list[dict[str, Any]]:
    with _client() as c:
        r = c.get("/api/db/status")
        r.raise_for_status()
        return r.json()


def db_relationships() -> dict[str, Any]:
    with _client() as c:
        r = c.get("/api/db/relationships")
        r.raise_for_status()
        return r.json()


def db_query(query: str) -> dict[str, Any]:
    with _client() as c:
        r = c.post("/api/db/query", json={"query": query})
        r.raise_for_status()
        return r.json()


# ── Endpoints & test suites ─────────────────────────────────────────────────

def list_endpoints() -> list[dict[str, Any]]:
    with _client() as c:
        r = c.get("/api/endpoints")
        r.raise_for_status()
        return r.json()


def _run_suite(path: str, excluded: list[str] | None = None) -> list[dict[str, Any]]:
    payload: dict[str, Any] = {}
    if excluded:
        payload["excludedEndpoints"] = excluded
    with _client(timeout=RUN_TIMEOUT) as c:
        r = c.post(path, json=payload)
        r.raise_for_status()
        return r.json()


def run_all(excluded: list[str] | None = None) -> list[dict[str, Any]]:
    return _run_suite("/api/run-all", excluded)


def run_smoke(excluded: list[str] | None = None) -> list[dict[str, Any]]:
    return _run_suite("/api/run-smoke", excluded)


def run_health(excluded: list[str] | None = None) -> list[dict[str, Any]]:
    return _run_suite("/api/run-health", excluded)


def run_performance(excluded: list[str] | None = None) -> list[dict[str, Any]]:
    return _run_suite("/api/run-performance", excluded)


def run_status_code(excluded: list[str] | None = None) -> list[dict[str, Any]]:
    return _run_suite("/api/run-status-code", excluded)


def run_content_type(excluded: list[str] | None = None) -> list[dict[str, Any]]:
    return _run_suite("/api/run-content-type", excluded)


def last_results() -> list[dict[str, Any]]:
    with _client() as c:
        r = c.get("/api/results")
        r.raise_for_status()
        return r.json()


def execute_single(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute a single Swagger-style request. Payload matches /api/execute contract."""
    with _client() as c:
        r = c.post("/api/execute", json=payload)
        r.raise_for_status()
        return r.json()


# ── Sessions ────────────────────────────────────────────────────────────────

def list_sessions() -> list[dict[str, Any]]:
    with _client() as c:
        r = c.get("/api/sessions")
        r.raise_for_status()
        return r.json()


def get_session(name: str) -> dict[str, Any]:
    with _client() as c:
        r = c.get(f"/api/sessions/{name}")
        r.raise_for_status()
        return r.json()
