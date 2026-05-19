import json
import urllib.request

from utils.local_automation_api import LocalAutomationApiServer


def test_routes_and_health_include_contract_metadata():
    handlers = {
        'open_project': lambda body: {'ok': True},
        'job_status': lambda body: {'status': 'idle'},
        'jobs_list': lambda body: {'jobs': []},
        'z_custom': lambda body: {},
    }
    server = LocalAutomationApiServer('127.0.0.1', 39605, handlers)
    server.start()
    try:
        with urllib.request.urlopen('http://127.0.0.1:39605/routes', timeout=2) as resp:
            routes = json.loads(resp.read().decode('utf-8'))
        with urllib.request.urlopen('http://127.0.0.1:39605/health', timeout=2) as resp:
            health = json.loads(resp.read().decode('utf-8'))
    finally:
        server.stop()

    assert routes['mcp_routes'] == ['open_project']
    assert routes['job_routes'] == ['job_status', 'jobs_list']
    assert routes['event_stream'] == '/events?job_id=<job_id>'
    assert health['status'] == 'ok'
    assert health['mcp_routes'] == routes['mcp_routes']
    assert health['job_routes'] == routes['job_routes']
