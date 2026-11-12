"""WatchFlow Definitions — Load/save/manage workflow configurations."""
import json
import os
from datetime import datetime

WATCHFLOWS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "json", "watchflows.json")


def _ensure_file():
    if not os.path.exists(WATCHFLOWS_FILE):
        os.makedirs(os.path.dirname(WATCHFLOWS_FILE), exist_ok=True)
        with open(WATCHFLOWS_FILE, "w") as f:
            json.dump([], f)


def load_workflows() -> list[dict]:
    _ensure_file()
    with open(WATCHFLOWS_FILE, "r") as f:
        return json.load(f)


def get_workflow(workflow_id: str) -> dict | None:
    for w in load_workflows():
        if w["id"] == workflow_id:
            return w
    return None


def save_workflow(workflow: dict) -> dict:
    """Create or update a workflow definition."""
    workflows = load_workflows()
    workflow.setdefault("id", f"wf-{int(datetime.now().timestamp())}")
    workflow.setdefault("enabled", True)
    workflow.setdefault("created_at", datetime.now().isoformat())
    workflow.setdefault("history", [])

    # Update existing or append new
    found = False
    for i, w in enumerate(workflows):
        if w["id"] == workflow["id"]:
            workflows[i] = workflow
            found = True
            break
    if not found:
        workflows.append(workflow)

    with open(WATCHFLOWS_FILE, "w") as f:
        json.dump(workflows, f, indent=2)
    return workflow


def delete_workflow(workflow_id: str) -> bool:
    workflows = load_workflows()
    filtered = [w for w in workflows if w["id"] != workflow_id]
    if len(filtered) == len(workflows):
        return False
    with open(WATCHFLOWS_FILE, "w") as f:
        json.dump(filtered, f, indent=2)
    return True


def add_run_history(workflow_id: str, run_result: dict):
    """Append a run result to workflow history (keep last 20)."""
    workflows = load_workflows()
    for w in workflows:
        if w["id"] == workflow_id:
            w.setdefault("history", [])
            w["history"].append(run_result)
            w["history"] = w["history"][-20:]  # Keep last 20 runs
            w["last_run"] = run_result.get("started_at")
            break
    with open(WATCHFLOWS_FILE, "w") as f:
        json.dump(workflows, f, indent=2)
