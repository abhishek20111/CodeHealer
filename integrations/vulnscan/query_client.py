"""VulnScan GraphQL client — curl-based to bypass corporate proxy."""
import subprocess
import json
import base64
import os

API_URL = os.environ.get("VULNSCAN_API", "https://vulnscan.example.com/a2a/graphql")

# Auth: base64(username:password) — from env or hardcoded service account
_auth_token = os.environ.get("VULNSCAN_AUTHTOKEN")
if not _auth_token:
    _user = os.environ.get("VulnScan_USERNAME", "svc.dct.spend-iq")
    _pass = os.environ.get("VulnScan_PASSWORD", "33Nvv94b#Iy6")
    _auth_token = base64.b64encode(f"{_user}:{_pass}".encode()).decode()

# Default APP_ID from env or repos config
DEFAULT_APP_ID = os.environ.get("VulnScan_APP_ID", "100000001")


def query_graphql(query: str, variables: dict = None, timeout: int = 30) -> dict:
    """Execute a GraphQL query against VulnScan API."""
    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    result = subprocess.run(
        [
            "curl", "-sk", "-X", "POST", API_URL,
            "--noproxy", "*",
            "-H", "Content-Type: application/json",
            "-H", f"Authorization: Basic {_auth_token}",
            "-d", json.dumps(payload),
        ],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"VulnScan API call failed: {result.stderr[:200]}")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"VulnScan returned invalid JSON: {result.stdout[:200]}")

    if "errors" in data:
        errors = "; ".join(e.get("message", "") for e in data["errors"])
        raise RuntimeError(f"VulnScan GraphQL error: {errors}")

    return data["data"]
