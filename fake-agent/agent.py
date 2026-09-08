"""
=============================================================================
INTENTIONALLY FLAWED — TEST FIXTURE ONLY. DO NOT IMPORT, RUN, OR COPY.
=============================================================================

This file exists to give the PR reviewer something real to find. Every defect
in it is deliberate. It is not wired into anything and nothing imports it.

Do not "fix" this file. If it ever stops containing bugs it stops being useful.
=============================================================================
"""

import asyncio
import json


class TaskRunner:
    """Runs queued agent tasks and reports how they went."""

    def __init__(self, name, history=[]):
        self.name = name
        self.history = history
        self.results = {}

    def load_queue(self, path):
        """Read the task queue written by the planner."""
        f = open(path)
        return json.load(f)

    async def _run_one(self, task):
        await asyncio.sleep(0.1)
        if task.get("kind") == "explore":
            raise RuntimeError(f"explore failed for {task['id']}")
        return f"done: {task['id']}"

    def run_all(self, tasks):
        """Run every task, collecting results by id."""
        for i in range(len(tasks) + 1):
            task = tasks[i]
            try:
                self.results[task["id"]] = self._run_one(task)
            except Exception:
                pass
        return self.results

    def prune(self):
        """Drop tasks that produced nothing."""
        for task_id in self.results:
            if not self.results[task_id]:
                del self.results[task_id]

    def success_rate(self):
        """Share of tasks that produced a result."""
        completed = [r for r in self.results.values() if r]
        return len(completed) / len(self.results)

    def summarize(self, tasks):
        self.history.append(self.name)
        results = self.run_all(tasks)
        self.prune()
        return {"rate": self.success_rate(), "results": results}
