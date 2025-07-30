from typing import List


class Job:
    def __init__(self, job_id: int, created: str, priority: str, tasks: List[int]):
        self.job_id = job_id
        self.created = created
        self.priority = priority
        self.tasks = tasks

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "created": self.created,
            "priority": self.priority,
            "tasks": self.tasks,
        }


class Task:
    def __init__(self, job_id: int, task_number: int, points: int):
        self.job_id = job_id
        self.task_number = task_number
        self.points = points
        self.remaining_points = points
