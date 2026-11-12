"""HTTP client for the Frontend Automation Tool (UI Comparator & Recorder).

The tool runs a local Node/Playwright dashboard (default port 3737) exposing:
    GET  /api/sessions                              → list recorded flows
    GET  /api/sessions/{name}                       → session details + routes
    GET  /api/sessions/{name}/reports               → list generated reports
    POST /api/compare                               → run a two-env comparison
    POST /api/sessions/{name}/functional            → run functional test
    GET  /api/jobs/{job_id}                         → job status
    GET  /api/history                               → job history
"""

from __future__ import annotations

import os
from typing import Any

import httpx

FRONTEND_BASE_URL = os.environ.get("QA_FRONTEND_URL", "http://localhost:3737").rstrip("/")
DEFAULT_TIMEOUT = float(os.environ.get("QA_FRONTEND_TIMEOUT", "30"))


def _client() -> httpx.Client:
    return httpx.Client(base_url=FRONTEND_BASE_URL, timeout=DEFAULT_TIMEOUT, trust_env=False)


def health() -> dict[str, Any]:
    """Return whether the frontend dashboard is reachable."""
    try:
        with _client() as c:
            r = c.get("/api/env")
            return {"ok": r.status_code == 200, "url": FRONTEND_BASE_URL, "status_code": r.status_code}
    except httpx.HTTPError as exc:
        return {"ok": False, "url": FRONTEND_BASE_URL, "error": str(exc)}


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


def list_reports(name: str) -> list[dict[str, Any]]:
    with _client() as c:
        r = c.get(f"/api/sessions/{name}/reports")
        r.raise_for_status()
        return r.json()


def run_compare(session_name: str, env1_url: str, env2_url: str, extra: dict | None = None) -> dict[str, Any]:
    payload = {"session": session_name, "env1": env1_url, "env2": env2_url}
    if extra:
        payload.update(extra)
    with _client() as c:
        r = c.post("/api/compare", json=payload)
        r.raise_for_status()
        return r.json()


def run_functional(session_name: str, env_url: str, extra: dict | None = None) -> dict[str, Any]:
    payload = {"env": env_url}
    if extra:
        payload.update(extra)
    with _client() as c:
        r = c.post(f"/api/sessions/{session_name}/functional", json=payload)
        r.raise_for_status()
        return r.json()


def get_job(job_id: str) -> dict[str, Any]:
    with _client() as c:
        r = c.get(f"/api/jobs/{job_id}")
        r.raise_for_status()
        return r.json()
