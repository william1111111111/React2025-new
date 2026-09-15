# Mode experiment runtime logging

This monitor is independent of the frozen model/protocol. It does not change training, restart jobs, or select checkpoints.

An independent session currently attaches to the existing detached queue. Every 30 seconds it writes `runtime/heartbeat.json` and appends `runtime/heartbeats.jsonl`: PID/start identity, process state and CPU/RSS, arm stages and steps, log freshness, GPU7 usage, host memory, free disk, and readable cgroup memory events. `runtime/events.jsonl` records startup, stage changes, watcher signals and queue disappearance. A vanished queue generates `runtime/exit.json`.

An attached non-child process has no retrievable wait status. Its disappearance is explicitly recorded as cause unknown, not inferred to be OOM or SIGKILL. For a future run, launch `watch.py --root <absolute-output-root> -- <absolute-python> -m mode_supervision.queue` in an independent background session; the monitor becomes the queue's parent and records its actual return code or terminating signal. SIGKILL cannot be handled by the killed process itself; the independent surviving monitor provides the observation. If both die (host restart or group-wide kill), the last persisted heartbeat remains, but the exact cause still requires host logs.

The current incident: both arms previously reached step 525, then queue and workers disappeared without a new Python traceback. Host uptime rules out a host reboot. Kernel journal access is insufficient to confirm/exclude host OOM. Original launch remained attached to an execution session; external session cleanup is plausible but unproven. Current queue is session leader, PPID 1, and resumed from step 500. No performance results existed at the incident.

Tests cover exit code 7, real SIGTERM termination, and `/proc` process identity. `watch.log` captures monitor failures. A heartbeat older than 90 seconds should prompt checking the monitor itself.
