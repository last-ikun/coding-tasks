# Job Scheduler System

This directory contains a job scheduling system implemented in Python. The system consists of multiple server and worker implementations with different scheduling strategies and features.

## Source Code Files

### Core Components

- `utils.py`: Contains utility classes and functions used across the system, including the `Job` and `Task` classes.

### Basic Implementation (Task 1)

- `task_1_1_server.py`: Basic HTTP server implementation that serves job information based on timestamps.
- `task_1_2_worker.py`: Simple worker implementation that fetches and processes jobs sequentially.

### Enhanced Workers (Task 2)

- `task_2_1_worker.py`: Worker implementation with basic job processing capabilities.
- `task_2_2_worker.py`: Enhanced worker with improved scheduling logic.
- `task_2_3_worker.py`: Worker implementation with efficiency indicators and execution logging.

### Advanced Implementation (Task 3)

- `task_3_2_server.py`: Advanced server implementation with job queuing system, ensuring each job is processed only once.
- `task_3_2_worker.py`: Multi-worker implementation with concurrent job processing capabilities and priority-based scheduling.

## Additional Documentation

- `task_2_3_efficiency_indicator_report.md`: Report on efficiency indicators

## Project Structure
```
job-scheduler/
├── data/           # Job data files
├── plot/           # Execution charts and visualizations
```
