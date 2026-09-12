# Analysis Worker

How web-submitted analyses are executed outside the API process. The API records a run; a
separate worker process claims it from Postgres, executes it, and writes the result back.
Implementation: `src/indication_scout/services/worker.py`, the claim and sweep methods in
`services/run_repository.py`, and the `scout worker` command in `cli/cli.py`.

## Roles

The API validates the drug, writes one pending run row, and returns its id. It never runs an
analysis. Polling, report download, and cancellation read and write the same rows, so any API
replica can answer for any run.

A worker is a long-lived process running one loop: sweep stale attempts, claim one pending run,
execute it, repeat. It handles one run at a time. Several workers can run against the same
database; the claim step guarantees each run is executed by exactly one of them.

The CLI (`scout find`, `scout investigate`) is unchanged. It creates its run and attempt in-process
and executes immediately. A worker never claims a CLI run.

```
browser ──POST──▶ API ──insert pending──▶ Postgres ◀──claim/heartbeat/result── worker
   │                                         ▲
   └──────────────GET status ────────────────┘
```

## Run, attempt, event

A run is the user's request: drug, kind (find or investigate), where it came from (api or cli),
optional holdout cutoff, status, and the validated result once done. Run status is one of
pending, running, done, error, cancelled.

An attempt is one execution of a run. It carries the worker id, deployment id and commit that ran
it, start and finish times, a heartbeat timestamp, duration, and the error if it failed. Attempt
status is running, done, error, cancelled, or interrupted. A run has one attempt per execution;
a requeued run gets attempt 2.

Events are the append-only trail: created, started, progress (the milestones the frontend renders),
cancellation requested, completed, failed, cancelled, interrupted, exhausted.

## Claim

The worker selects the oldest pending run whose source is `api`, locked with `FOR UPDATE SKIP
LOCKED`, opens the next attempt on it, and commits. A second worker running the same statement at
the same time skips the locked row and takes the next one, or gets nothing. The attempt records
the worker id (the Railway replica id, else hostname and pid) so the ledger shows who ran what.

## Execute

Execution is the code the API route used to run in its background task, moved verbatim: bind the
per-run log context, cost tracker and progress emitter; serve a fresh seed report if one exists
for the drug, otherwise run the supervisor pipeline; then record the cost snapshot and the terminal
state in one transaction. Failure persists the exception type and message on the attempt and marks
the run error.

Two tasks run side by side: the analysis, and a heartbeat loop that every
`WORKER_HEARTBEAT_INTERVAL_SECONDS` stamps the attempt's heartbeat and reads the run row.

## Cancel

`DELETE /api/analyses/{id}` on a pending run cancels it outright. On a running run it sets a
cancellation-requested timestamp on the row. The heartbeat loop sees the timestamp on its next
tick, cancels the analysis task, and the worker records the attempt as cancelled. Cancellation
therefore works from any API replica and does not need the process that started the run.

## Sweep and requeue

A worker that is killed cannot write a terminal state, so its attempt stays `running` with a
heartbeat that stops advancing. At the top of every loop iteration each worker sweeps: any running
attempt whose heartbeat is older than `WORKER_STALE_ATTEMPT_SECONDS` is marked interrupted with an
`analysis.interrupted` event, and its run is settled:

| Condition | Run status | Event |
|---|---|---|
| cancellation was requested | cancelled | analysis.cancelled |
| run came from the CLI | error | analysis.exhausted (no worker for source) |
| attempt number ≥ `WORKER_MAX_ATTEMPTS` | error | analysis.exhausted (max attempts) |
| otherwise | pending | — (next claim opens attempt 2) |

A requeued run restarts from the beginning. The disk cache makes the second attempt much cheaper
because every upstream call the first attempt completed is already cached.

If a sweep interrupts an attempt whose worker is in fact still alive (a heartbeat delayed past the
threshold), that worker's next heartbeat is rejected by the repository. The worker then cancels
its own analysis task and writes nothing, so the sweep's state stands and the run is not finished
twice. This is why the stale threshold must be much larger than the heartbeat interval.

## Shutdown

SIGTERM or SIGINT stops the worker claiming new runs; the run in progress finishes, then the
process exits. If the host kills the process before that, the attempt is left for the sweep.

## Constants

All four live in `.env.constants*` with no defaults, loaded through `config.py`.

| Name | Meaning |
|---|---|
| `WORKER_POLL_INTERVAL_SECONDS` | Sleep between claim attempts when nothing is pending |
| `WORKER_HEARTBEAT_INTERVAL_SECONDS` | Gap between heartbeat writes on a running attempt |
| `WORKER_STALE_ATTEMPT_SECONDS` | Heartbeat age at which a sweep treats the attempt as dead |
| `WORKER_MAX_ATTEMPTS` | Attempts per run before an interruption becomes a permanent error |

## Deployment

Docker Compose runs `worker` as a second service from the same image with `scout worker` as the
command, the same database URL, and the same cache volume. The dev override bind-mounts `src/`
and the local cache the same way it does for `api`.

On Railway the worker is a second service created from the same repo with the start command
overridden to `scout worker`, the same variables, and the same `/cache` volume. The health check in
`railway.toml` applies only to the API. Without a worker service, submitted analyses stay pending.

Runs that were `running` under the pre-worker API have no heartbeats, so the first sweep after
deploy treats every one of them as dead and re-executes it. Cancel them before the first worker
starts.

## Limits

One run per worker; throughput scales by adding worker replicas. Shared caps that are per-process
(the PubMed request semaphore, the embedding model lock) become per-worker, so the number of
workers is the knob that bounds total upstream and LLM concurrency. No agent-level checkpointing;
an interrupted run restarts from zero. Only interruption triggers a retry; an upstream failure
inside an attempt fails the run.
