from utils.translation_assist import TranslationAssistService
from utils.automation_api_contract import build_route_discovery


def test_translation_assist_placeholder_requires_explicit_apply():
    svc = TranslationAssistService()
    st = svc.set_candidates(0, 1, 'src', 'current', [
        {'candidate_id': 'c1', 'provider': 'mt', 'text': 'cand1'},
        {'candidate_id': 'c2', 'provider': 'llm', 'text': 'cand2'},
    ])
    assert st.current_target_text == 'current'
    assert st.applied_candidate_id == ''

    st2 = svc.apply_candidate(0, 1, 'c2')
    assert st2.applied_candidate_id == 'c2'
    assert st2.current_target_text == 'current'


def test_route_discovery_includes_translation_assist_namespace_entries():
    payload = build_route_discovery({
        'translation_assist_block': lambda body: {},
        'translation_assist_candidates': lambda body: {},
        'translation_assist_apply_candidate': lambda body: {},
        'translation_assist_tm': lambda body: {},
        'translation_assist_glossary': lambda body: {},
        'translation_assist_concordance': lambda body: {},
        'translation_assist_sfx': lambda body: {},
        'translation_assist_add_to_tm': lambda body: {},
        'translation_assist_add_to_glossary': lambda body: {},
        'translation_assist_qa': lambda body: {},
        'translation_assist_profiles': lambda body: {},
        'translation_assist_apply_text': lambda body: {},
        'translation_assist_cache': lambda body: {},
        'translation_assist_compare': lambda body: {},
    })
    assert 'translation_assist_block' in payload['routes']
    assert 'translation_assist_candidates' in payload['routes']
    assert 'translation_assist_apply_candidate' in payload['routes']
    assert 'translation_assist_tm' in payload['routes']
    assert 'translation_assist_glossary' in payload['routes']
    assert 'translation_assist_concordance' in payload['routes']
    assert 'translation_assist_sfx' in payload['routes']
    assert 'translation_assist_add_to_tm' in payload['routes']
    assert 'translation_assist_add_to_glossary' in payload['routes']
    assert 'translation_assist_qa' in payload['routes']
    assert 'translation_assist_profiles' in payload['routes']
    assert 'translation_assist_apply_text' in payload['routes']
    assert 'translation_assist_cache' in payload['routes']
    assert 'translation_assist_compare' in payload['routes']


def test_translation_assist_service_clear_block():
    svc = TranslationAssistService()
    svc.set_candidates(2, 3, 'src', 'current', [{'candidate_id': 'a', 'provider': 'mt', 'text': 'x'}])
    assert svc.get_block(2, 3).candidates
    svc.clear_block(2, 3)
    st = svc.get_block(2, 3)
    assert st.candidates == []
    assert st.source_text == ''
