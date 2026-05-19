from utils.translation_qa_profiles import build_translation_qa_report, resolve_prompt_profile


class B:
    def __init__(self, src, tr):
        self.text = src
        self.translation = tr


def test_resolve_prompt_profile_defaults_to_dialogue():
    assert resolve_prompt_profile('unknown')['max_len_ratio'] == resolve_prompt_profile('dialogue')['max_len_ratio']


def test_qa_report_respects_sfx_profile_for_same_text():
    blocks = [B('BAM', 'BAM'), B('Hello', 'Hello')]
    report = build_translation_qa_report(blocks, glossary=[], profile='sfx', retry_issue_threshold=1)
    # sfx allows carry-over
    assert report['issue_blocks'] == 0


def test_qa_report_retry_candidates_by_issue_count():
    blocks = [B('alpha beta', 'alpha beta alpha beta alpha beta alpha beta')]
    report = build_translation_qa_report(blocks, glossary=[{'source': 'alpha', 'target': '阿尔法'}], profile='dialogue', retry_issue_threshold=2)
    assert report['retry_candidates'] == [0]
