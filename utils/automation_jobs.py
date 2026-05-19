from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List

FINAL_STATES = {"succeeded", "failed", "cancelled"}


@dataclass
class AutomationJob:
    job_id: str
    task: str
    status: str = "queued"
    warnings: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: Any = None
    cancel_requested: bool = False
    progress: float = 0.0
    stage: str = "queued"


def new_job(job_id: str, task: str) -> Dict[str, Any]:
    j = AutomationJob(job_id=job_id, task=task)
    j.logs.append(f"created task={task}")
    return j.__dict__.copy()


def append_log(job: Dict[str, Any], text: str, *, limit: int = 200) -> None:
    logs = job.setdefault("logs", [])
    logs.append(str(text))
    if limit > 0 and len(logs) > limit:
        del logs[0: len(logs) - limit]
    job["updated_at"] = time.time()


def add_warning(job: Dict[str, Any], text: str) -> None:
    warns = job.setdefault("warnings", [])
    msg = str(text)
    warns.append(msg)
    append_log(job, f"warning: {msg}")


def set_status(job: Dict[str, Any], status: str, *, stage: str | None = None, progress: float | None = None) -> None:
    job["status"] = str(status)
    if stage is not None:
        job["stage"] = str(stage)
    if progress is not None:
        job["progress"] = max(0.0, min(1.0, float(progress)))
    job["updated_at"] = time.time()


def checkpoint_or_cancel(job: Dict[str, Any], stage: str, progress: float) -> bool:
    set_status(job, job.get("status") or "running", stage=stage, progress=progress)
    if job.get("cancel_requested"):
        set_status(job, "cancelled", stage=f"cancelled:{stage}")
        append_log(job, f"cancelled at checkpoint: {stage}")
        return True
    return False


def status_payload(job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "job_id": job.get("job_id", ""),
        "task": job.get("task", ""),
        "status": job.get("status", ""),
        "stage": job.get("stage", ""),
        "progress": float(job.get("progress", 0.0) or 0.0),
        "warnings": list(job.get("warnings", [])),
        "cancel_requested": bool(job.get("cancel_requested", False)),
        "created_at": float(job.get("created_at", 0.0)),
        "updated_at": float(job.get("updated_at", 0.0)),
        "has_result": job.get("result") is not None,
    }


def normalize_task_result(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        return dict(result)
    return {"value": result}


def update_from_task_result(job: Dict[str, Any], result: Any) -> None:
    payload = normalize_task_result(result)
    warns = payload.get("warnings")
    if isinstance(warns, list):
        for w in warns:
            if str(w or "").strip():
                add_warning(job, str(w))
    stage = payload.get("stage")
    progress = payload.get("progress")
    if stage is not None or progress is not None:
        set_status(job, job.get("status") or "running", stage=str(stage or job.get("stage") or "running"), progress=float(progress if progress is not None else job.get("progress") or 0.0))


def mark_started(job: Dict[str, Any], task: str) -> None:
    set_status(job, "running", stage=f"starting:{task}", progress=0.02)
    append_log(job, f"started task={task}")


def mark_finished(job: Dict[str, Any], task: str, *, cancelled: bool = False) -> None:
    if cancelled:
        set_status(job, "cancelled", stage=f"cancelled:{task}", progress=1.0)
        append_log(job, f"cancelled task={task}")
    else:
        set_status(job, "succeeded", stage=f"completed:{task}", progress=1.0)
        append_log(job, f"finished task={task}")
