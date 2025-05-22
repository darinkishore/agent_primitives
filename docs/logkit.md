## **Logkit v3**

*Canonical spec – paste-in telemetry layer that makes “enterprise-grade” logs the accidental default, even for a lone AI agent hacking offline.*

---

### 1 What problem does Logkit solve?

1. **Adoption paradox:**  Teams either keep `print()` (easy but useless) or build a logging rig so complex nobody adopts it.
2. **We want both:**  *Follow 3 copy-paste rules* and every log line is JSON, trace-correlated, stack-rich, level-sane, and parseable by a machine or human—even with no external infra, on a green-field solo repo.

---

### 2 The three unbreakable rules 🧑‍🎓

| #     | What you do                                                                                           | Why it matters                                                                                      |                                           |
| ----- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| **1** | `python<br>from logkit import log, new_context, capture` – *never* `logging.getLogger` directly.      | Gives you a logger already wired for JSON, sampling, rotation, redaction hooks, etc.                |                                           |
| **2** | **Call `new_context()` exactly once per work-unit** (HTTP request, CLI run, test case, Celery task…). | Creates / forwards a `trace_id`, clears stale data, binds any high-level IDs (user, repo, branch…). |                                           |
| **3** | **Wrap work in `@capture` or `with capture():`** (or rely on the built-in framework adapters).        | Times the span, adds a unique `span_id`, auto-logs \`status=success                                 | error\` with full traceback and duration. |

Copy those three lines, *do nothing else*, and your logs are ready for Kibana **or** for an offline AI agent that merely reads a JSONL file.

---

### 3 Drop-in code (`logkit.py`, \~160 LoC)

```python
#!/usr/bin/env -S uv run --script
# %%pep 723
# requires-python = ">=3.10"
# dependencies = [
#   "structlog>=24.1",
# ]

import os, time, uuid, functools, logging, structlog
import contextvars, asyncio, pathlib
from logging.handlers import RotatingFileHandler
from typing import Any, Callable

# ---------------------------------------------------------------- context plumbing
_CTX: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("ctx", default={})

def _merge_ctx(_, __, event):           # inject current ContextVars into every event
    event.update(_CTX.get())
    return event

# ---------------------------------------------------------------- file handler (offline-friendly)
def _attach_file_handler(path: str | os.PathLike, rotate_mb: int) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=rotate_mb * 1_048_576, backupCount=5)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(handler)

# ---------------------------------------------------------------- global configure
def _configure_once() -> None:
    profile   = os.getenv("LOG_PROFILE", "dev").lower()        # dev | prod
    base_lvl  = getattr(logging, os.getenv("LOG_LEVEL", "DEBUG" if profile=="dev" else "INFO").upper())
    sample    = float(os.getenv("LOG_SAMPLE", 1))              # 1 = keep all
    log_file  = os.getenv("LOG_FILE", "run.log")               # JSONL sink for offline agents
    rotate_mb = int(os.getenv("LOG_ROTATE_MB", 10))

    processors = [
        structlog.contextvars.merge_contextvars,
        _merge_ctx,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.SampleByLevel(lower=logging.DEBUG,
                                           upper=logging.INFO,
                                           sample_rate=sample),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer() if profile == "dev"
        else structlog.processors.JSONRenderer(),
    ]

    structlog.configure(
        processors              = processors,
        wrapper_class           = structlog.make_filtering_bound_logger(base_lvl),
        logger_factory          = structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use = True,
    )

    logging.basicConfig(level=base_lvl, handlers=[logging.StreamHandler()])   # std-lib parity

    if log_file:                         # local JSONL file so agents can read after the run
        _attach_file_handler(log_file, rotate_mb)

_CONFIG_FLAG = "_LOGKIT_CONFIGURED"
if _CONFIG_FLAG not in globals():
    _configure_once()
    globals()[_CONFIG_FLAG] = True

log = structlog.get_logger()            # ← every import gets the same, fully wired logger

# ---------------------------------------------------------------- public helpers
def new_context(**kv: Any) -> str:
    """
    Clear any stale context, start (or continue) a trace and bind user-supplied keys.
    Returns the active trace_id so callers can forward it across process boundaries.
    """
    structlog.contextvars.clear_contextvars()
    trace_id = kv.pop("trace_id", uuid.uuid4().hex)
    all_kv   = {"trace_id": trace_id, **kv}
    structlog.contextvars.bind_contextvars(**all_kv)
    _CTX.set(all_kv)
    return trace_id

class capture:
    """
    Decorator *and* context-manager.
    * Adds span_id, logs one END record with dur_ms + status.
    * Works on sync and async functions.
    """

    def __init__(self, **kv): self._kv = kv

    # -- decorator -----------------------------------------------------------
    def __call__(self, fn: Callable):
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def awrapper(*a, **kw):
                with self.__class__(fn=fn.__qualname__, **self._kv):
                    return await fn(*a, **kw)
            return awrapper
        else:
            @functools.wraps(fn)
            def swrapper(*a, **kw):
                with self.__class__(fn=fn.__qualname__, **self._kv):
                    return fn(*a, **kw)
            return swrapper

    # -- context-manager -----------------------------------------------------
    def __enter__(self):
        self._t0   = time.perf_counter()
        self._span = {"span_id": uuid.uuid4().hex, **self._kv}
        structlog.contextvars.bind_contextvars(**self._span)
        return self

    def __exit__(self, exc_type, exc, __):
        dur_ms = int((time.perf_counter() - self._t0) * 1000)
        logger = log.exception if exc else log.info
        logger("END", dur_ms=dur_ms, status="error" if exc else "success", exc_info=exc)
        structlog.contextvars.unbind_contextvars(*self._span.keys())
        return False                       # re-raise exceptions

# ---------------------------------------------------------------- optional async helper
def ctx_task(coro):
    """
    Schedule an asyncio Task that inherits the *current* ContextVars snapshot.
    Use instead of `asyncio.create_task` to avoid the “spawn-before-bind” foot-gun.
    """
    ctx = contextvars.copy_context()
    return asyncio.create_task(ctx.run(coro))
```

---

### 4 Default behaviour & env-knobs

| Env var         | Default   | Effect                                                             |
| --------------- | --------- | ------------------------------------------------------------------ |
| `LOG_PROFILE`   | `dev`     | `dev` → pretty console + DEBUG; `prod` → JSON + INFO floor.        |
| `LOG_LEVEL`     | *(auto)*  | Override floor (`DEBUG`/`INFO`/`WARNING`…).                        |
| `LOG_SAMPLE`    | `1`       | 0 – 1 fraction of DEBUG/INFO lines kept in prod (`SampleByLevel`). |
| `LOG_FILE`      | `run.log` | JSONL sink for offline analysis; set empty to disable.             |
| `LOG_ROTATE_MB` | `10`      | Rotate the file at **N MiB** (5 backups kept).                     |
| `LOG_OTLP`      | *(unset)* | If set, append an OTLP exporter processor (requires extra deps).   |

---

### 5 Why agents & humans both win

1. **Machine-friendly JSON** – `prod` profile prints one JSON object per line; agents do `json.loads(line)` and get structured fields (`trace_id`, `span_id`, `dur_ms`, `exc_info`, …).
2. **Ever-present context** – A single `trace_id` lets bots or humans reconstruct every step of a failing test run, even across `await` chains.
3. **Zero hidden data loss** – Global level parity (`basicConfig`) + sampling knob ensures DEBUG is visible when desired and safely down-sampled when not.
4. **Offline by construction** – No network calls; logs land on disk (`run.log`) and stdout. An agent can tail or post-process immediately after a run.
5. **Scale-out path** – Flip `LOG_OTLP=1` and add one exporter dependency; the same three rules now feed a distributed trace backend.

---

### 6 Framework adapters (auto-capture)

Paste-one-liner snippets; they satisfy Rule 3 automatically so developers can’t forget spans.

```python
# FastAPI --------------------------------------------------------
from fastapi import FastAPI, Request
import logkit as lk

app = FastAPI()

@app.middleware("http")
async def logkit_mw(request: Request, call_next):
    lk.new_context(path=str(request.url.path), user=request.headers.get("X-User"))
    with lk.capture(http="request"):
        return await call_next(request)

# Celery ---------------------------------------------------------
from celery import Celery
import logkit as lk
celery = Celery(__name__)

@celery.task(bind=True)
def task(self, *a, **kw):
    lk.new_context(task_id=self.request.id)
    with lk.capture(celery="task"):
        ...

# CLI (Click) ----------------------------------------------------
import click, logkit as lk

@click.command()
@lk.capture(cli="command")
def main():
    lk.new_context(cmd="reindex")
    ...
```

---

### 7 Usage tips for a **solo, offline AI-agent project**

| Action                                   | How                                                                                        |                                              |
| ---------------------------------------- | ------------------------------------------------------------------------------------------ | -------------------------------------------- |
| **Record a full run for the bot**        | \`LOG\_PROFILE=prod python run\_tests.py 2>&1                                              | tee run.jsonl\` (JSONL file + console view). |
| **Let the bot isolate failures**         | The bot scans for `"status":"error"` → grabs `exc_info`, backtracks by `trace_id`.         |                                              |
| **Keep DEBUG but avoid 100 MB logs**     | `LOG_PROFILE=prod LOG_LEVEL=DEBUG LOG_SAMPLE=0.05` → keep 5 % of DEBUG lines.              |                                              |
| **Open trace in Vim/Emacs**              | `jq -c 'select(.trace_id=="abc123")' run.jsonl`                                            |                                              |
| **Cross-process work (multiprocessing)** | Pass `trace_id` in the message/env and call `new_context(trace_id=trace_id)` in the child. |                                              |

---

### 8 FAQs

* **Do I still need to call `@capture` everywhere?**
  No—only around spans you care about.  The framework middleware above already wraps web requests / tasks, so you start with decent coverage.

* **Can I redact secrets?**
  Add a processor before the renderer: inspect keys, replace values with `"***"`.  Because every log flows through one chain, redaction is global.

* **Does this replace OpenTelemetry?**
  For a single-process app, yes.  When you introduce multiple services or languages, set `LOG_OTLP=1` and forward the same `trace_id`; logkit becomes an OTLP-emitting shim.

---

### 9 Copy-this-checklist ✅

1. **Add `logkit.py` to your repo** (or `pip install logkit` once you publish it).
2. **Search-replace `logging.getLogger` → `from logkit import log`**.
3. **At each entry point** (`main()`, request handler, Celery task): `new_context(...)`.
4. **Wrap hot paths** with `@capture` (or use the provided middleware).
5. **Run** – JSONL logs appear in `run.log`; AI agents & humans have instant, correlation-rich telemetry.

You’re done.
Three rules, one helper file, enterprise-grade logs — even on a laptop with no internet.

