"""WatchFlow Scheduler — Manages cron/interval triggers for workflows."""
import asyncio
from datetime import datetime, timedelta
from integrations.watchflow.definitions import load_workflows, get_workflow


class WatchFlowScheduler:
    """Simple asyncio-based scheduler for workflows."""

    def __init__(self, executor):
        self.executor = executor
        self._tasks = {}  # {workflow_id: asyncio.Task}
        self._running = False

    async def start(self):
        """Load all enabled workflows and start their schedules."""
        self._running = True
        workflows = load_workflows()
        for wf in workflows:
            if wf.get("enabled", False):
                self.schedule(wf)
        print(f"[WATCHFLOW] Scheduler started with {len(self._tasks)} active workflows")

    def stop(self):
        """Cancel all scheduled tasks."""
        self._running = False
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()
        print("[WATCHFLOW] Scheduler stopped")

    def schedule(self, workflow: dict):
        """Schedule a workflow based on its schedule config."""
        wf_id = workflow["id"]
        if wf_id in self._tasks:
            self._tasks[wf_id].cancel()

        schedule = workflow.get("schedule", {})
        interval_minutes = schedule.get("minutes", 0)
        interval_hours = schedule.get("hours", 0)

        if not interval_minutes and not interval_hours:
            return  # No schedule = manual-only

        total_seconds = (interval_hours * 3600) + (interval_minutes * 60)
        self._tasks[wf_id] = asyncio.ensure_future(self._run_loop(wf_id, total_seconds))
        print(f"[WATCHFLOW] Scheduled '{workflow.get('name', wf_id)}' every {total_seconds}s")

    def unschedule(self, workflow_id: str):
        """Remove a workflow from the scheduler."""
        if workflow_id in self._tasks:
            self._tasks[workflow_id].cancel()
            del self._tasks[workflow_id]

    async def trigger_now(self, workflow_id: str) -> dict:
        """Manually trigger a workflow immediately."""
        workflow = get_workflow(workflow_id)
        if not workflow:
            return {"error": f"Workflow '{workflow_id}' not found"}
        return await self.executor.execute(workflow)

    async def _run_loop(self, workflow_id: str, interval_seconds: int):
        """Background loop: wait interval then execute."""
        await asyncio.sleep(10)  # Initial delay before first run
        while self._running:
            try:
                workflow = get_workflow(workflow_id)
                if not workflow or not workflow.get("enabled"):
                    break
                print(f"[WATCHFLOW] ⏰ Triggering scheduled run: {workflow.get('name', workflow_id)}")
                await self.executor.execute(workflow)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[WATCHFLOW] Error in scheduled run for {workflow_id}: {e}")
            await asyncio.sleep(interval_seconds)

    def get_status(self) -> dict:
        """Return scheduler status."""
        return {
            "running": self._running,
            "active_workflows": list(self._tasks.keys()),
            "count": len(self._tasks),
        }
