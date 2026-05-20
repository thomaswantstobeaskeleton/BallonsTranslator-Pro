from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Iterable
import hashlib


@dataclass
class TranslationAssistCandidate:
    candidate_id: str
    provider: str
    text: str
    latency_ms: int = 0


@dataclass
class TranslationAssistState:
    source_text: str = ""
    current_target_text: str = ""
    candidates: List[TranslationAssistCandidate] = field(default_factory=list)
    applied_candidate_id: str = ""


class TranslationAssistService:
    """Lightweight placeholder service for Translation Assist route wiring."""

    def __init__(self):
        self.state_by_key: Dict[str, TranslationAssistState] = {}
        self.candidate_cache: Dict[str, List[dict]] = {}

    def get_block(self, page: int, block: int) -> TranslationAssistState:
        key = f"{int(page)}:{int(block)}"
        return self.state_by_key.setdefault(key, TranslationAssistState())

    def set_candidates(self, page: int, block: int, source_text: str, current_target_text: str, candidates: List[dict]) -> TranslationAssistState:
        st = self.get_block(page, block)
        st.source_text = str(source_text or "")
        st.current_target_text = str(current_target_text or "")
        st.candidates = [
            TranslationAssistCandidate(
                candidate_id=str(c.get('candidate_id', f'c{i+1}')),
                provider=str(c.get('provider', 'unknown')),
                text=str(c.get('text', '')),
                latency_ms=int(c.get('latency_ms', 0) or 0),
            )
            for i, c in enumerate(candidates or [])
        ]
        return st

    def apply_candidate(self, page: int, block: int, candidate_id: str) -> TranslationAssistState:
        st = self.get_block(page, block)
        st.applied_candidate_id = str(candidate_id or "")
        return st

    def clear_block(self, page: int, block: int) -> None:
        key = f"{int(page)}:{int(block)}"
        self.state_by_key.pop(key, None)

    def cache_key(self, source_text: str, profile: str, providers: List[str]) -> str:
        src = " ".join(str(source_text or "").split()).lower()
        p = str(profile or "dialogue").strip().lower()
        prov = ",".join(str(x).strip().lower() for x in (providers or []))
        return f"{p}|{prov}|{src}"

    def get_cached_candidates(self, source_text: str, profile: str, providers: List[str]) -> List[dict]:
        return list(self.candidate_cache.get(self.cache_key(source_text, profile, providers), []) or [])

    def set_cached_candidates(self, source_text: str, profile: str, providers: List[str], candidates: List[dict]) -> None:
        self.candidate_cache[self.cache_key(source_text, profile, providers)] = list(candidates or [])

    def clear_cache(self) -> int:
        n = len(self.candidate_cache)
        self.candidate_cache.clear()
        return n


def build_candidates_from_sources(*, tm_hits: Iterable[dict], glossary_hits: Iterable[dict], sfx_hits: Iterable[dict], concordance_hits: Iterable[dict], max_candidates: int = 6, telemetry: Dict[str, dict] = None) -> List[dict]:
    """
    Build Translation Assist candidates from heterogeneous sources and de-duplicate by normalized text.
    """
    rows: List[dict] = []
    max_n = max(1, int(max_candidates or 6))

    telemetry = telemetry or {}

    def _push(provider: str, text: str):
        text = str(text or "").strip()
        if not text:
            return
        rows.append({"provider": provider, "text": text, "telemetry": dict(telemetry.get(provider, {}))})

    for h in list(tm_hits or [])[:max_n]:
        _push("TM", h.get("target", ""))
    for h in list(glossary_hits or [])[:max_n]:
        _push("Glossary", h.get("target", ""))
    for h in list(sfx_hits or [])[:max_n]:
        _push("SFX", h.get("common_en", "") or h.get("source", ""))
    for h in list(concordance_hits or [])[:max_n]:
        _push("Concordance", h.get("target", ""))

    seen = set()
    out: List[dict] = []
    for i, row in enumerate(rows, start=1):
        key = " ".join(str(row.get("text", "")).split()).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({"candidate_id": f"cand_{i}", "provider": row.get("provider", "unknown"), "text": row.get("text", ""), "telemetry": row.get("telemetry", {})})
        if len(out) >= max_n:
            break
    return out


def normalize_provider_warning(provider: str, err: Exception = None, warning_text: str = "") -> dict:
    text = str(warning_text or (str(err) if err is not None else "")).strip()
    if not text:
        return {}
    low = text.lower()
    code = "provider_warning"
    if "timeout" in low:
        code = "timeout"
    elif "unauthorized" in low or "api key" in low:
        code = "auth"
    elif "rate" in low and "limit" in low:
        code = "rate_limit"
    return {"provider": str(provider or "unknown"), "code": code, "message": text}


def estimate_candidate_cost_usd(text: str, provider: str) -> float:
    # lightweight placeholder estimate for compare telemetry. local sources are free.
    p = str(provider or "").lower()
    if p in {"tm", "glossary", "sfx", "concordance"}:
        return 0.0
    chars = len(str(text or ""))
    return round(chars * 0.000002, 6)


def make_synthetic_mt_candidate(source_text: str, provider: str, mode: str = "low_latency") -> dict:
    src = str(source_text or "").strip()
    digest = hashlib.sha1(f"{provider}:{mode}:{src}".encode("utf-8")).hexdigest()[:6]
    if mode == "high_quality":
        text = f"[{provider}] {src} (refined {digest})"
    else:
        text = f"[{provider}] {src} ({digest})"
    return {"provider": provider, "text": text}
