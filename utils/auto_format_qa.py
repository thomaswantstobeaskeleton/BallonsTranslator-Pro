from __future__ import annotations

from typing import Any, Dict, List

from utils.text_rendering import smart_fit_text_to_box, plan_atomic_bubble_fit


def _block_text(block: Any) -> str:
    return (getattr(block, 'translation', '') or getattr(block, 'get_text', lambda: '')() or '').strip()


def score_auto_format_candidates(blocks: List[Any], *, profile: str = 'balanced') -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, blk in enumerate(blocks or []):
        ff = getattr(blk, 'fontformat', None)
        if ff is None:
            continue
        xyxy = list(getattr(blk, 'xyxy', []) or [])
        if len(xyxy) < 4:
            continue
        w = max(1.0, float(xyxy[2]) - float(xyxy[0]))
        h = max(1.0, float(xyxy[3]) - float(xyxy[1]))
        text = _block_text(blk)
        if not text:
            continue
        before = smart_fit_text_to_box(
            text=text,
            font_size=float(getattr(ff, 'font_size', 24) or 24),
            box_size=(w, h),
            writing_mode=getattr(ff, 'writing_mode', 'auto'),
            fit_mode=getattr(ff, 'fit_mode', 'shrink'),
            line_spacing=float(getattr(ff, 'line_spacing', 1.15) or 1.15),
            letter_spacing=float(getattr(ff, 'letter_spacing', 1.0) or 1.0),
            padding=float(getattr(ff, 'text_padding', 0.0) or 0.0),
            stroke_width=float(getattr(ff, 'stroke_width', 0.0) or 0.0),
            secondary_stroke_width=float(getattr(ff, 'secondary_stroke_width', 0.0) or 0.0),
            line_break_strategy=getattr(ff, 'line_break_strategy', 'auto'),
        )
        after = plan_atomic_bubble_fit(
            text=text,
            font_size=float(getattr(ff, 'font_size', 24) or 24),
            box_size=(w, h),
            writing_mode=getattr(ff, 'writing_mode', 'auto'),
            fit_mode=getattr(ff, 'fit_mode', 'shrink'),
            line_break_strategy=getattr(ff, 'line_break_strategy', 'auto'),
            line_spacing=float(getattr(ff, 'line_spacing', 1.15) or 1.15),
            letter_spacing=float(getattr(ff, 'letter_spacing', 1.0) or 1.0),
            padding=float(getattr(ff, 'text_padding', 0.0) or 0.0),
            stroke_width=float(getattr(ff, 'stroke_width', 0.0) or 0.0),
            secondary_stroke_width=float(getattr(ff, 'secondary_stroke_width', 0.0) or 0.0),
            profile=profile,
        )
        rows.append({
            'index': idx,
            'text_preview': text[:80],
            'before_score': float(before.quality_score),
            'after_score': float(after.quality_score),
            'before_overflow': bool(before.overflow),
            'after_overflow': bool(after.overflow),
            'before_actions': list(before.actions or []),
            'after_actions': list(after.actions or []),
            'improvement': float(after.quality_score) - float(before.quality_score),
        })
    return rows


def summarize_auto_format_scores(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows or [])
    improved = [r for r in rows if float(r.get('improvement', 0.0) or 0.0) > 0.01]
    overflow_before = sum(1 for r in rows if bool(r.get('before_overflow')))
    overflow_after = sum(1 for r in rows if bool(r.get('after_overflow')))
    return {
        'count': len(rows),
        'improved_count': len(improved),
        'overflow_before': overflow_before,
        'overflow_after': overflow_after,
        'avg_delta': round(sum(float(r.get('improvement', 0.0) or 0.0) for r in rows) / max(1, len(rows)), 4),
    }
