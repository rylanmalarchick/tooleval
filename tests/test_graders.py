"""Every grader must reject the start state and every broken answer, and accept
the reference. A grader that fails this is not measuring anything."""

import pytest

from tooleval import catalyst
from tooleval.__main__ import TASKS, selftest_task
from tooleval.task import discover

TASK_LIST = discover(TASKS)


def test_tasks_exist():
    assert TASK_LIST


@pytest.mark.parametrize("task", TASK_LIST, ids=lambda t: t.name)
def test_grader_separates_reference_from_broken(task):
    if task.base == "catalyst" and not catalyst.GRADE.exists():
        pytest.skip("Catalyst grade slot not on this host (desktop only)")
    assert selftest_task(task) == []
