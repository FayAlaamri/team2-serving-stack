# W3D5 Capacity Note

- Locked model: Qwen/Qwen2.5-1.5B-Instruct-AWQ
- Target p95 SLO: 3.000 s
- Predicted knee: concurrency 4
- Knee concurrency: 16
- Tokens/s at knee: 1057.15
- p95 at knee: 0.688 s
- Max sustainable request rate: approximately 16.61 requests/s
- Limiting family: inspect GPU utilisation, clocks, memory-bandwidth symptoms, and host overhead before choosing compute / memory / overhead.
- Why knee, not peak: the knee is the highest tested concurrency that still meets the p95 SLO, so it is the capacity we can honestly promise.
