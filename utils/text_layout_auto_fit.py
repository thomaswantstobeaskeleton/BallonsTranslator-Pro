from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal
import math
import re


WritingMode = Literal["horizontal", "vertical", "auto"]


@dataclass
class AutoFitRequest:
    text: str
    box: Tuple[float, float, float, float]  # x,y,w,h
    min_font_size: float = 8.0
    max_font_size: float = 72.0
    writing_mode: WritingMode = "auto"
    language: str = "auto"
    line_spacing: float = 1.2
    letter_spacing: float = 1.0
    padding: float = 6.0
    stroke_width: float = 0.0
    shadow_radius: float = 0.0
    shadow_offset: Tuple[float, float] = (0.0, 0.0)
    allow_expand_mode: str = "never"  # never|within_bubble|within_page
    width_profile: Optional[List[float]] = None  # per-line available width for bubble-like masks


@dataclass
class AutoFitResult:
    font_size: float
    lines: List[str]
    line_positions: List[Tuple[float, float]]
    text_bbox: Tuple[float, float, float, float]
    recommended_box: Tuple[float, float, float, float]
    overflow: bool
    score: float
    warnings: List[str] = field(default_factory=list)
    rejected_candidates: List[str] = field(default_factory=list)


_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")


def _is_cjk(ch: str) -> bool:
    return bool(_CJK.match(ch))


def _tokenize(text: str) -> List[str]:
    toks: List[str] = []
    cur = ""
    for ch in text.strip():
        if ch.isspace():
            if cur:
                toks.append(cur)
                cur = ""
            continue
        if _is_cjk(ch):
            if cur:
                toks.append(cur)
                cur = ""
            toks.append(ch)
        else:
            cur += ch
    if cur:
        toks.append(cur)
    return toks


def _token_width(tok: str, font: float, letter_spacing: float) -> float:
    if len(tok) == 1 and _is_cjk(tok):
        return font * 1.0
    base = 0.58 * font * len(tok)
    return base * max(0.6, letter_spacing)


def _line_break_dp(tokens: List[str], max_widths: List[float], font: float, letter_spacing: float) -> List[str]:
    n = len(tokens)
    if n == 0:
        return []
    pref = [0.0]
    for t in tokens:
        pref.append(pref[-1] + _token_width(t, font, letter_spacing) + font * 0.30)

    target_lines = max(1, len(max_widths))
    inf = 1e18
    dp = [[inf] * (n + 1) for _ in range(target_lines + 1)]
    back = [[-1] * (n + 1) for _ in range(target_lines + 1)]
    dp[0][0] = 0.0

    for k in range(1, target_lines + 1):
        mw = max_widths[min(k - 1, len(max_widths) - 1)]
        for i in range(1, n + 1):
            for j in range(0, i):
                if dp[k - 1][j] >= inf:
                    continue
                width = pref[i] - pref[j] - font * 0.30
                overflow = max(0.0, width - mw)
                ragged = abs(width - mw * 0.86)
                orphan = 500.0 if (i == n and i - j <= 1 and n > 3) else 0.0
                cost = dp[k - 1][j] + overflow * overflow * 1000 + ragged * 0.5 + orphan
                if cost < dp[k][i]:
                    dp[k][i] = cost
                    back[k][i] = j

    best_k = min(range(1, target_lines + 1), key=lambda k: dp[k][n])
    lines = []
    i, k = n, best_k
    while k > 0 and i >= 0:
        j = back[k][i]
        if j < 0:
            break
        lines.append(" ".join(tokens[j:i]))
        i = j
        k -= 1
    lines.reverse()
    return [ln for ln in lines if ln]


def auto_fit_text(req: AutoFitRequest) -> AutoFitResult:
    x, y, w, h = req.box
    pad = req.padding + req.stroke_width * 2 + req.shadow_radius + max(abs(req.shadow_offset[0]), abs(req.shadow_offset[1]))
    inner_w = max(8.0, w - 2 * pad)
    inner_h = max(8.0, h - 2 * pad)
    tokens = _tokenize(req.text)
    warnings: List[str] = []

    if req.writing_mode == "auto":
        writing_mode = "vertical" if sum(1 for t in tokens if any(_is_cjk(c) for c in t)) > max(2, len(tokens) // 2) and h > w * 1.25 else "horizontal"
    else:
        writing_mode = req.writing_mode

    lo, hi = req.min_font_size, req.max_font_size
    best = None
    rejected = []
    for _ in range(16):
        mid = (lo + hi) / 2.0
        line_h = mid * req.line_spacing
        if writing_mode == "vertical":
            max_lines = max(1, int(inner_w / line_h))
            widths = req.width_profile[:max_lines] if req.width_profile else [inner_h] * max_lines
        else:
            max_lines = max(1, int(inner_h / line_h))
            widths = req.width_profile[:max_lines] if req.width_profile else [inner_w] * max_lines
        lines = _line_break_dp(tokens, widths, mid, req.letter_spacing)
        used_h = len(lines) * line_h
        used_w = 0.0
        for idx, ln in enumerate(lines):
            uw = _token_width(ln.replace(" ", ""), mid, req.letter_spacing)
            used_w = max(used_w, uw)
            if idx < len(widths) and uw > widths[idx] + 1e-6:
                used_w = max(used_w, uw)
        if writing_mode == "vertical":
            fits = used_h <= inner_w + 1e-6 and all(_token_width(ln.replace(' ', ''), mid, req.letter_spacing) <= widths[min(i, len(widths)-1)] + 1e-6 for i, ln in enumerate(lines))
        else:
            fits = used_h <= inner_h + 1e-6 and all(_token_width(ln.replace(' ', ''), mid, req.letter_spacing) <= widths[min(i, len(widths)-1)] + 1e-6 for i, ln in enumerate(lines))
        if fits:
            best = (mid, lines, used_w, used_h)
            lo = mid
        else:
            rejected.append(f"font={mid:.2f}:overflow")
            hi = mid

    if best is None:
        font = req.min_font_size
        line_h = font * req.line_spacing
        if writing_mode == "vertical":
            max_lines = max(1, int(inner_w / line_h))
            widths = req.width_profile[:max_lines] if req.width_profile else [inner_h] * max_lines
        else:
            max_lines = max(1, int(inner_h / line_h))
            widths = req.width_profile[:max_lines] if req.width_profile else [inner_w] * max_lines
        lines = _line_break_dp(tokens, widths, font, req.letter_spacing)
        overflow = True
        warnings.append("text_overflow_min_font")
    else:
        font, lines, used_w, used_h = best
        overflow = False

    line_h = font * req.line_spacing
    if writing_mode == "vertical":
        text_w = len(lines) * line_h
        text_h = max([_token_width(ln.replace(" ", ""), font, req.letter_spacing) for ln in lines] + [0.0])
        start_x = x + (w - text_w) / 2.0
        start_y = y + (h - text_h) / 2.0
        positions = [(start_x + i * line_h, start_y) for i in range(len(lines))]
    else:
        text_w = max([_token_width(ln.replace(" ", ""), font, req.letter_spacing) for ln in lines] + [0.0])
        text_h = len(lines) * line_h
        start_x = x + (w - text_w) / 2.0
        start_y = y + (h - text_h) / 2.0
        positions = [(start_x, start_y + i * line_h) for i in range(len(lines))]

    usage = (text_w * text_h) / max(1.0, inner_w * inner_h)
    balance = 1.0
    if len(lines) > 1:
        lens = [len(ln.replace(" ", "")) for ln in lines]
        balance = 1.0 - (max(lens) - min(lens)) / max(1.0, max(lens))
    score = (0.0 if overflow else 1000.0) + font * 10.0 + balance * 100.0 - abs(0.78 - usage) * 140.0

    if font <= req.min_font_size + 0.2:
        warnings.append("near_min_font")
    if usage < 0.35:
        warnings.append("underfilled_area")

    return AutoFitResult(
        font_size=font,
        lines=lines,
        line_positions=positions,
        text_bbox=(start_x, start_y, text_w, text_h),
        recommended_box=(x, y, w, h),
        overflow=overflow,
        score=score,
        warnings=warnings,
        rejected_candidates=rejected,
    )
