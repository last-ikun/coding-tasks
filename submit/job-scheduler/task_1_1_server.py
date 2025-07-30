#!/usr/bin/env python3

import http.server
import json
import os
import socketserver
from typing import Dict, Optional
from urllib.parse import parse_qs, urlparse
from utils import Job


class JobServer:
    def __init__(self):
        self.jobs: Dict[str, Job] = {}  # timestamp -> Job mapping
        self.load_jobs()

    def load_jobs(self):
        """Load jobs from job files in the data directory."""
        data_dir = "data"
        if not os.path.exists(data_dir):
            print(f"Warning: {data_dir} directory not found")
            return

        for filename in os.listdir(data_dir):
            if filename.endswith(".job"):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, "r") as f:
                    content = f.read().splitlines()
                    job_id = None
                    created = None
                    priority = None
                    tasks = []
                    section = None
                    for line in content:
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("[") and line.endswith("]"):
                            section = line[1:-1]
                            continue
                        if section == "JobID":
                            job_id = int(line)
                        elif section == "Created":
                            created = line
                        elif section == "Priority":
                            priority = line
                        elif section == "Tasks":
                            tasks.append(int(line))
                    if job_id is not None and created and priority and tasks:
                        self.jobs[created] = Job(job_id, created, priority, tasks)

    def get_jobs_by_timestamp(self, timestamp: str) -> Optional[Job]:
        """Return job information for the given timestamp."""
        return self.jobs.get(timestamp)


class JobRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.job_server = JobServer()
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests to retrieve job information."""
        parsed_url = urlparse(self.path)

        if parsed_url.path == "/jobs":
            # Parse query parameters
            query_params = parse_qs(parsed_url.query)
            timestamp = query_params.get("timestamp", [None])[0]

            if timestamp is None:
                self.send_error(400, "Missing timestamp parameter")
                return

            # Get job information for the timestamp
            job = self.job_server.get_jobs_by_timestamp(timestamp)

            # Send response
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            if job:
                response = {"job": job.to_dict()}
            else:
                response = {"job": None}

            self.wfile.write(json.dumps(response).encode())
            return

        self.send_error(404, "Not Found")


def run_server(port: int = 8000):
    with socketserver.TCPServer(("", port), JobRequestHandler) as httpd:
        print(f"Server running on port {port}")
        httpd.serve_forever()


if __name__ == "__main__":
    run_server()
