"""WatchFlow Executor — Runs workflow steps by calling the existing agent."""
import asyncio
from datetime import datetime
from integrations.watchflow.definitions import add_run_history


class WorkflowExecutor:
    """Executes workflow steps sequentially using the existing ask_agent function."""

    def __init__(self, agent, ask_agent_fn):
        self.agent = agent
        self.ask_agent = ask_agent_fn
        self._running = {}  # {workflow_id: True/False}

    def is_running(self, workflow_id: str) -> bool:
        return self._running.get(workflow_id, False)

    async def execute(self, workflow: dict) -> dict:
        """Execute all steps of a workflow and return run result."""
        workflow_id = workflow["id"]
        if self.is_running(workflow_id):
            return {"error": "Workflow already running", "workflow_id": workflow_id}

        self._running[workflow_id] = True
        run_result = {
            "workflow_id": workflow_id,
            "started_at": datetime.now().isoformat(),
            "steps": [],
            "status": "running",
        }

        try:
            previous_results = {}
            for step in workflow.get("steps", []):
                step_result = await self._execute_step(step, workflow, previous_results)
                run_result["steps"].append(step_result)
                previous_results[step["id"]] = step_result

                # If step failed, stop workflow
                if step_result["status"] == "error":
                    run_result["status"] = "failed"
                    break

                # Check condition for next step
                if step_result["status"] == "skipped":
                    continue

            if run_result["status"] == "running":
                run_result["status"] = "completed"

        except Exception as e:
            run_result["status"] = "error"
            run_result["error"] = str(e)
        finally:
            run_result["finished_at"] = datetime.now().isoformat()
            self._running[workflow_id] = False
            add_run_history(workflow_id, run_result)

        return run_result

    async def _execute_step(self, step: dict, workflow: dict, previous_results: dict) -> dict:
        """Execute a single workflow step."""
        step_id = step["id"]
        step_result = {
            "id": step_id,
            "name": step.get("name", step_id),
            "started_at": datetime.now().isoformat(),
            "status": "running",
        }

        # Check condition (skip if not met)
        condition = step.get("condition")
        if condition and not self._check_condition(condition, previous_results):
            step_result["status"] = "skipped"
            step_result["reason"] = f"Condition not met: {condition}"
            return step_result

        try:
            question = step["question"]
            session_id = f"watchflow-{workflow['id']}-{step_id}-{int(datetime.now().timestamp())}"

            # Execute via existing agent
            answer = await self.ask_agent(self.agent, question, session_id=session_id)

            step_result["status"] = "completed"
            step_result["result"] = answer[:2000]  # Cap stored result
            step_result["session_id"] = session_id

        except Exception as e:
            step_result["status"] = "error"
            step_result["error"] = str(e)

        step_result["finished_at"] = datetime.now().isoformat()
        return step_result

    def _check_condition(self, condition: str, previous_results: dict) -> bool:
        """Evaluate a simple condition like 'scan.result contains CRITICAL'."""
        try:
            parts = condition.split(" contains ")
            if len(parts) == 2:
                ref, keyword = parts[0].strip(), parts[1].strip().strip("'\"")
                step_ref = ref.split(".")[0]
                if step_ref in previous_results:
                    result_text = previous_results[step_ref].get("result", "")
                    return keyword.lower() in result_text.lower()
            return True  # Default: condition passes
        except Exception:
            return True
