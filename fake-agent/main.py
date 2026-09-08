"""Entrypoint for the agent runner.

NOTE: demo fixture for exercising the PR reviewer.
"""

import sys

from agent import TaskRunner


def main(queue_path):
    runner = TaskRunner(name="default")
    tasks = runner.load_queue(queue_path)
    summary = runner.summarize(tasks)
    print(f"completed {len(summary['results'])} task(s), rate {summary['rate']:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
