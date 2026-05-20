from utils.translation_assist import (
    build_candidates_from_sources,
    TranslationAssistService,
    normalize_provider_warning,
    estimate_candidate_cost_usd,
    make_synthetic_mt_candidate,
)


def test_build_candidates_deduplicates_and_caps():
    out = build_candidates_from_sources(
        tm_hits=[{'target': 'Hello world'}, {'target': 'HELLO   world'}],
        glossary_hits=[{'target': 'Hero'}],
        sfx_hits=[{'common_en': 'Boom!'}],
        concordance_hits=[{'target': 'Narration line'}],
        max_candidates=3,
    )
    assert len(out) == 3
    texts = [r['text'] for r in out]
    assert 'Hello world' in texts
    assert 'Hero' in texts


def test_build_candidates_ignores_empty():
    out = build_candidates_from_sources(
        tm_hits=[{'target': ''}], glossary_hits=[], sfx_hits=[{'common_en': ''}], concordance_hits=[], max_candidates=5
    )
    assert out == []


def test_build_candidates_provider_order_prefers_tm_then_glossary():
    out = build_candidates_from_sources(
        tm_hits=[{'target': 'tm'}],
        glossary_hits=[{'target': 'glossary'}],
        sfx_hits=[{'common_en': 'sfx'}],
        concordance_hits=[{'target': 'cc'}],
        max_candidates=4,
    )
    assert [r['provider'] for r in out] == ['TM', 'Glossary', 'SFX', 'Concordance']


def test_build_candidates_includes_telemetry_when_provided():
    out = build_candidates_from_sources(
        tm_hits=[{'target': 'a'}], glossary_hits=[], sfx_hits=[], concordance_hits=[],
        max_candidates=3, telemetry={"TM": {"latency_ms": 12, "source": "project_tm"}}
    )
    assert out[0]["telemetry"]["latency_ms"] == 12


def test_translation_assist_service_candidate_cache_roundtrip():
    svc = TranslationAssistService()
    src = "Hello"
    prof = "dialogue"
    prov = ["current_translator"]
    assert svc.get_cached_candidates(src, prof, prov) == []
    svc.set_cached_candidates(src, prof, prov, [{"candidate_id": "c1", "provider": "TM", "text": "你好"}])
    got = svc.get_cached_candidates(src, prof, prov)
    assert len(got) == 1
    assert got[0]["text"] == "你好"
    assert svc.clear_cache() == 1


def test_provider_warning_and_cost_helpers():
    w = normalize_provider_warning("openai", warning_text="Timeout while waiting")
    assert w["code"] == "timeout"
    c = estimate_candidate_cost_usd("hello", "openai")
    assert c > 0
    row = make_synthetic_mt_candidate("source", "openai", "high_quality")
    assert "refined" in row["text"]
