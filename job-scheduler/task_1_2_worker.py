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
        max_timestamp="00:00:19",
    ):
        self.server_host = server_host
        self.server_port = server_port

        self.active_tasks: Dict[int, List[Task]] = {}  # job_id -> list of tasks
        self.execution_log: List[
            Tuple[str, List[Tuple[int, int, int]], int]
        ] = []  # [(timestamp, status_strings, total_points)]
        self.current_time = datetime.strptime("00:00:00", "%H:%M:%S")
        self.max_timestamp = max_timestamp
        self._connection = None

    # late init
    @property
    def connection(self):
        if self._connection is None:
            self._connection = http.client.HTTPConnection(
                self.server_host, self.server_port
            )
        return self._connection

    def fetch_job(self, timestamp: str) -> Optional[Job]:
        """Fetch job information from the server for a given timestamp."""
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

    def process_tasks(self):
        """Process all active tasks and record their status."""
        # Record current state
        task_status = OrderedDict()
        total_points = 0

        # Process each job's tasks
        for job_id, tasks in list(self.active_tasks.items()):
            if not tasks:
                del self.active_tasks[job_id]
                continue

            current_task = tasks[0]
            if current_task.remaining_points > 0:
                status = (
                    job_id,
                    current_task.task_number,
                    current_task.remaining_points,
                )
                task_status[job_id] = status
                total_points += current_task.remaining_points
                current_task.remaining_points -= 1

                # If task is complete, remove it and move to next
                if current_task.remaining_points == 0:
                    tasks.pop(0)

        # Log the current state
        self.execution_log.append(
            (
                self.current_time.strftime("%H:%M:%S"),
                list(task_status.values()),
                total_points,
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

    def print_execution_log(self):
        """Print the execution log in a table format."""
        print("timestamp | JobID-Task No (Remain Point) | Executing Point")
        print("-" * 60)
        for timestamp, status, points in self.execution_log:
            status_str = str(status) if status else ""
            print(f"{timestamp} | {status_str:<25} | {points}")

    def save_execution_chart(self):
        """Save the execution chart as an HTML table."""
        with open("execution_chart.html", "w") as f:
            # Start table
            f.write("<table>")

            # Write header row
            f.write(
                "<tr><td>timestamp</td><td>JobID-Task No (Remain Point)</td><td></td><td>Executing Point</td></tr>"
            )

            # Write data rows
            for timestamp, status, points in self.execution_log:
                f.write("<tr>")
                f.write(f"<td>{timestamp}</td>")

                # Handle task status columns
                if not status:
                    f.write("<td></td><td></td>")
                elif len(status) == 1:
                    f.write(f"<td>{status[0]}</td><td></td>")
                else:
                    f.write(f"<td>{status[0]}</td><td>{status[1]}</td>")

                # Write executing point
                f.write(f"<td>{points}</td>")
                f.write("</tr>")

            # Close table
            f.write("</table>")

    def __del__(self):
        """Close the HTTP connection when the worker is deleted."""
        if self._connection:
            self._connection.close()


def main():
    worker = JobWorker()
    worker.run_simulation()
    worker.print_execution_log()
    # worker.save_execution_chart()


if __name__ == "__main__":
    main()
