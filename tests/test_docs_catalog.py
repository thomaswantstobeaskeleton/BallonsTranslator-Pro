from utils.docs_catalog import build_docs_catalog
from utils.automation_api_contract import build_route_discovery


def test_docs_catalog_has_expected_sections_and_items():
    catalog = build_docs_catalog()
    assert 'start_here' in catalog
    assert 'quality_translation_lettering' in catalog
    assert 'automation_realtime_plans' in catalog
    assert 'raw_sources_and_extensibility' in catalog
    for rows in catalog.values():
        assert rows
        for row in rows:
            assert str(row.get('path', '')).startswith('docs/')
            assert str(row.get('highlight', '')).strip()


def test_route_discovery_can_include_docs_catalog_endpoint():
    payload = build_route_discovery({
        'docs_catalog': lambda body: {},
        'open_project': lambda body: {},
    })
    assert 'docs_catalog' in payload['routes']
