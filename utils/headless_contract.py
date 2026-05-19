from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List


class HeadlessExitCode(IntEnum):
    OK = 0
    CONFIG_ERROR = 2
    INPUT_ERROR = 3
    RUNTIME_ERROR = 4
    PARTIAL_FAILURE = 5


@dataclass
class HeadlessRunSummary:
    requested_dirs: List[str]
    processed_dirs: List[str]
    skipped_dirs: List[str]
    failed_dirs: List[str]
    warnings: List[str] = field(default_factory=list)

    def to_payload(self) -> Dict[str, object]:
        return {
            "requested": len(self.requested_dirs),
            "processed": len(self.processed_dirs),
            "skipped": len(self.skipped_dirs),
            "failed": len(self.failed_dirs),
            "processed_dirs": list(self.processed_dirs),
            "skipped_dirs": list(self.skipped_dirs),
            "failed_dirs": list(self.failed_dirs),
            "warnings": list(self.warnings),
        }

    def exit_code(self) -> int:
        if not self.requested_dirs:
            return int(HeadlessExitCode.INPUT_ERROR)
        if self.failed_dirs:
            return int(HeadlessExitCode.PARTIAL_FAILURE if self.processed_dirs else HeadlessExitCode.RUNTIME_ERROR)
        if not self.processed_dirs:
            return int(HeadlessExitCode.INPUT_ERROR)
        return int(HeadlessExitCode.OK)
