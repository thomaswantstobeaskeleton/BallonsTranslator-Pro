from utils.automation_api_contract import build_route_discovery


def test_route_discovery_includes_realtime_mcp_routes():
    payload = build_route_discovery({
        'realtime_status': lambda body: {},
        'realtime_start': lambda body: {},
        'realtime_stop': lambda body: {},
        'realtime_translate_now': lambda body: {},
        'open_project': lambda body: {},
        'z_custom': lambda body: {},
    })
    assert 'realtime_status' in payload['mcp_routes']
    assert 'realtime_start' in payload['mcp_routes']
    assert 'realtime_stop' in payload['mcp_routes']
    assert 'realtime_translate_now' in payload['mcp_routes']
