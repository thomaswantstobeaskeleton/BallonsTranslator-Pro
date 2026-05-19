from types import SimpleNamespace

from utils.fontformat import FontFormat
from utils.auto_format_qa import score_auto_format_candidates, summarize_auto_format_scores


def test_auto_format_qa_scores_rows():
    blk = SimpleNamespace(
        xyxy=[0, 0, 120, 60],
        translation='This is a long line that usually needs balancing for comic bubbles',
        get_text=lambda: 'src',
        fontformat=FontFormat(font_size=24, writing_mode='horizontal_ltr', fit_mode='shrink', line_break_strategy='balanced'),
    )
    rows = score_auto_format_candidates([blk], profile='balanced')
    assert len(rows) == 1
    assert rows[0]['index'] == 0
    assert 'before_score' in rows[0]
    assert 'after_score' in rows[0]
    summary = summarize_auto_format_scores(rows)
    assert summary['count'] == 1
    assert 'avg_delta' in summary
