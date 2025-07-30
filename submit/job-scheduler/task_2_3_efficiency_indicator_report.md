# Efficiency Indicator Analysis Report of the JobWorker

To decide on an **efficiency indicator** for a task worker when you have **capacity** and **priority** functions, typically a metric is needed that links:

- **The worker's capacity**: How much work can the worker do in a certain time?
- **The priorities of tasks**: How important are the tasks the worker is actually completing?

## Efficiency Indicator Design

**Common indicators may include:**

- **Throughput-weighted with priority**: Are the highest-priority tasks being completed, given the worker's capacity?
- **Percentage capacity utilization**: How much of the worker's capacity is used?
- **Priority-weighted completion**: Sum of priorities of tasks completed per unit time divided by capacity.

For example, to formalize the last last indicator above:

### Efficiency Indicator Formula

```
efficiency = (sum of priorities of tasks completed during period) / (worker capacity * period length)
```

## How to Decide the Formula?

the efficiency indicator can be choosed based on which factor values more:

- if you want to penalize incomplete capacity use, use capacity as denominator.
- if you want to value high-priority tasks, use priority as numerator.
- you could use weighted averages or normalize values as needed.

## Conclusion

| Efficiency Indicator                 | Formula/Approach                                                                         | Use Case                                        |
|--------------------------------------|-----------------------------------------------------------------------------------------|-------------------------------------------------|
| Capacity Utilization                 | (Count of tasks completed) / (Worker capacity)                                          | General efficiency                              |
| Priority-Weighted Completion         | (Sum of priorities for completed tasks) / (Max possible, e.g., capacity*max_priority)   | Value high-priority execution                   |
| Priority per Capacity                | (Sum of priorities for completed tasks) / (Worker capacity)                             | Mix of both capacity and priority               |

you can decide based on what's most important to you: completing as much tasks as possible, completing top-priority, or a mix.
the JobWorker python class of task 2.3 submited choose the  **Priority per Capacity** as the indicator and will re-schedule the tasks accordingly.
