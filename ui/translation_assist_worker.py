from __future__ import annotations

import time
from typing import Dict, List

from qtpy.QtCore import QObject, Signal

from modules import GET_VALID_TRANSLATORS, GET_VALID_OCR, GET_VALID_TEXTDETECTORS, GET_VALID_INPAINTERS
from utils.translation_assist import normalize_provider_warning


class CompareWorker(QObject):
    """Background worker for Translation Assist provider comparison."""

    finished = Signal(dict)

    def __init__(
        self,
        scope: str,
        providers: List[str],
        source_text: str,
        lang_source: str,
        lang_target: str,
        current_module: str = "",
        project_cfg: dict = None,
    ):
        super().__init__()
        self.scope = str(scope or "translator").strip().lower()
        self.providers = [str(p) for p in (providers or []) if str(p).strip()]
        self.source_text = str(source_text or "").strip()
        self.lang_source = str(lang_source or "Auto").strip()
        self.lang_target = str(lang_target or "简体中文").strip()
        self.current_module = str(current_module or "").strip()
        self.project_cfg = dict(project_cfg or {})

    def run(self) -> None:
        candidates: List[dict] = []
        warnings: List[dict] = []
        telemetry: Dict[str, dict] = {}

        if self.scope == "translator":
            candidates, warnings, telemetry = self._run_translator_compare()
        else:
            candidates, warnings, telemetry = self._run_module_registry_compare()

        self.finished.emit({
            "ok": True,
            "scope": self.scope,
            "candidates": candidates,
            "warnings": warnings,
            "telemetry": telemetry,
            "count": len(candidates),
        })

    def _run_translator_compare(self):
        candidates: List[dict] = []
        warnings: List[dict] = []
        telemetry: Dict[str, dict] = {}
        valid = set(GET_VALID_TRANSLATORS())
        max_n = 12  # generous limit for compare

        for prov in self.providers:
            prov = prov.strip()
            if not prov:
                continue
            if prov in {"TM", "Glossary", "Concordance", "SFX"}:
                # These are project-local sources; handled by mainwindow's refresh path.
                # Skip here so the compare doesn't hang on missing project data.
                continue
            t0 = time.perf_counter()
            try:
                if prov not in valid:
                    warnings.append(normalize_provider_warning(prov, warning_text=f"translator '{prov}' is not registered/available"))
                    continue
                # Lazy import to avoid heavy init at module load time
                from modules.translators.base import TRANSLATORS
                cls = TRANSLATORS.module_dict.get(prov)
                if cls is None:
                    warnings.append(normalize_provider_warning(prov, warning_text=f"translator class for '{prov}' not found in registry"))
                    continue
                tr = cls(self.lang_source, self.lang_target, raise_unsupported_lang=False)
                result = tr.translate(self.source_text)
                text = str(result or "").strip()
                latency_ms = int((time.perf_counter() - t0) * 1000)
                tele = {"latency_ms": latency_ms, "source": "external_provider", "cost_usd": 0.0}
                candidates.append({
                    "candidate_id": f"ext_{prov}_{len(candidates) + 1}",
                    "provider": prov,
                    "text": text,
                    "telemetry": tele,
                })
                telemetry[prov] = tele
            except Exception as e:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                telemetry[prov] = {"latency_ms": latency_ms, "source": "external_provider", "error": str(e)}
                nw = normalize_provider_warning(prov, err=e)
                if nw:
                    warnings.append(nw)

        # De-duplicate real translator candidates by normalized text
        seen = set()
        out: List[dict] = []
        for row in candidates:
            key = " ".join(str(row.get("text", "")).split()).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(dict(row))
            if len(out) >= max_n:
                break
        # Re-label candidate IDs to match compare format
        for i, row in enumerate(out, start=1):
            row["candidate_id"] = f"compare_{i}"

        return out, warnings, telemetry

    def _run_module_registry_compare(self):
        candidates: List[dict] = []
        warnings: List[dict] = []
        telemetry: Dict[str, dict] = {}

        scope_map = {
            "ocr": GET_VALID_OCR,
            "detector": GET_VALID_TEXTDETECTORS,
            "inpainter": GET_VALID_INPAINTERS,
        }
        getter = scope_map.get(self.scope)
        if getter is None:
            return candidates, warnings, telemetry

        all_names = getter()
        for i, nm in enumerate(all_names, start=1):
            candidates.append({
                "candidate_id": f"module_{self.scope}_{i}",
                "provider": f"{self.scope}:{nm}",
                "text": nm,
                "telemetry": {
                    "source": f"{self.scope}_module_registry",
                    "is_current": (nm == self.current_module),
                },
            })

        return candidates, warnings, telemetry
