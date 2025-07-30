#!/usr/bin/env python3

import http.client
import json
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from utils import Job, Task


class JobWorker:
    def __init__(
        self,
        server_host: str = "localhost",
        server_port: int = 8000,
        max_timestamp="01:04:37",
        capacity: int = 15,
    ):
        self.server_host = server_host
        self.server_port = server_port

        self.capacity = capacity
        self.active_tasks: Dict[int, List[Task]] = {}  # job_id -> list of tasks
        self.job_priorities: Dict[int, str] = {}  # job_id -> priority
        self.execution_log: List[
            Tuple[str, List[Tuple[int, int, int]], int]
        ] = []  # [(timestamp, task_status, total_points)]
        self.current_time = datetime.strptime("00:00:00", "%H:%M:%S")
        self.max_timestamp = max_timestamp
        self._connection = None

    # late initialization of HTTP connection
    @property
    def connection(self):
        if self._connection is None:
            self._connection = http.client.HTTPConnection(
                self.server_host, self.server_port
            )
        return self._connection

    def fetch_job(self, timestamp: str) -> Optional[Job]:
        """Fetch job information from the server for a given timestamp."""
        print(f"Fetching job for timestamp: {timestamp}")
        self.connection.request("GET", f"/jobs?timestamp={timestamp}")
        response = self.connection.getresponse()
        data = json.loads(response.read().decode())
        return Job(**data.get("job")) if data.get("job") else None

    def add_job(self, job_data: Job):
        """Add a new job to the worker's task list."""
        if not job_data:
            return

        job_id = job_data.job_id
        tasks = []
        for i, points in enumerate(job_data.tasks, 1):
            tasks.append(Task(job_id, i, points))

        self.active_tasks[job_id] = tasks
        self.job_priorities[job_id] = job_data.priority  # Store the job's priority

    def calculate_priority_per_capacity(self, job_id: int, tasks: List[Task]) -> float:
        """Calculate Priority per Capacity indicator for a job."""
        # Sum up points for remaining tasks
        total_points = sum(task.remaining_points for task in tasks)
        # Convert priority to numeric value (High=2, Low=1)
        priority_value = 2 if self.job_priorities[job_id] == "High" else 1
        # Calculate indicator: (priority * total points) / capacity
        return (priority_value * total_points) / self.capacity

    def reschedule_jobs(self):
        """Reschedule jobs based on Priority per Capacity indicator."""
        return sorted(
            list(self.active_tasks.items()),
            key=lambda x: -self.calculate_priority_per_capacity(x[0], x[1])  # Higher values first
        )

    def process_tasks(self):
        """Process all active tasks and record their status."""
        # Record current state
        task_status = OrderedDict()
        executing_points = 0

        # Process each job's tasks, sorted by priority
        for job_id, tasks in self.reschedule_jobs():
            if not tasks:
                del self.active_tasks[job_id]
                del self.job_priorities[job_id]
                continue

            current_task = tasks[0]
            if current_task.remaining_points > 0:
                if (executing_points + current_task.remaining_points) <= self.capacity:
                    status = (
                        job_id,
                        current_task.task_number,
                        current_task.remaining_points,
                    )
                    task_status[job_id] = status
                    executing_points += current_task.remaining_points
                    current_task.remaining_points -= 1

                    # If task is complete, remove it and move to next
                    if current_task.remaining_points == 0:
                        tasks.pop(0)
                        if not tasks:  # If no more tasks for this job
                            del self.active_tasks[job_id]
                            del self.job_priorities[job_id]
                else:
                    print(f"Capacity exceeded for job {job_id}. Skipping processing.")
                    continue

        # Log the current state
        self.execution_log.append(
            (
                self.current_time.strftime("%H:%M:%S"),
                list(task_status.values()),
                executing_points,
            )
        )

    def run_simulation(self):
        """Run the job processing simulation."""
        while True:
            # Fetch any new jobs for the current timestamp
            job_data = self.fetch_job(self.current_time.strftime("%H:%M:%S"))
            if job_data:
                self.add_job(job_data)

            # Process active tasks
            self.process_tasks()

            # If no more active tasks and we're past the last example time, stop
            if not self.active_tasks and self.current_time >= datetime.strptime(
                self.max_timestamp, "%H:%M:%S"
            ):
                break

            # Move to next second
            self.current_time += timedelta(seconds=1)

    def save_execution_chart(self, filename: str = "execution_chart.md"):
        max_execution_column = len(max(self.execution_log, key=lambda x: len(x[1]))[1])
        headers = (
            ["timestamp", "JobID-Task No (Remain Point)"]
            + [" "] * (max_execution_column - 1)
            + ["Executing Point"]
        )
        table_str = "| " + " | ".join(headers) + " |\n"
        table_str += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for log_record in self.execution_log:
            row_cells = (
                [log_record[0]]
                + [f"{x[0]} - {x[1]}({x[2]})" for x in log_record[1]]
                + [" "] * (max_execution_column - len(log_record[1]))
                + [str(log_record[2])]
            )
            table_str += "| " + " | ".join(str(cell) for cell in row_cells) + " |\n"
        with open(filename, "w") as f:
            f.write("# Job Scheduler Execution Log\n\n")
            f.write(
                "This file contains the execution log of the job scheduler simulation.\n\n"
            )
            f.write("## Execution Chart\n\n")
            f.write(table_str)

    def __del__(self):
        """Close the HTTP connection when the worker is deleted."""
        if self._connection:
            self._connection.close()


def main():
    worker = JobWorker(capacity=15)
    worker.run_simulation()
    worker.save_execution_chart(filename="plot/task_2_3_execution_chart.md")


if __name__ == "__main__":
    main()
