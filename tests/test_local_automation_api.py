import json
import urllib.request
import urllib.error

from utils.local_automation_api import LocalAutomationApiServer


def test_local_automation_api_basic_route():
    server = LocalAutomationApiServer('127.0.0.1', 39599, {'ping': lambda body: {'pong': body.get('x', 0)}})
    server.start()
    req = urllib.request.Request('http://127.0.0.1:39599/ping', data=json.dumps({'x': 7}).encode('utf-8'), method='POST', headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=2) as resp:
        out = json.loads(resp.read().decode('utf-8'))
    server.stop()
    assert out['ok'] is True
    assert out['result']['pong'] == 7


def test_local_automation_api_key_required():
    server = LocalAutomationApiServer('127.0.0.1', 39600, {'ping': lambda body: {'pong': 1}}, api_key='secret')
    server.start()
    req = urllib.request.Request('http://127.0.0.1:39600/ping', data=b'{}', method='POST', headers={'Content-Type': 'application/json'})
    try:
        urllib.request.urlopen(req, timeout=2)
        ok = False
    except urllib.error.HTTPError as e:
        ok = (e.code == 401)
    req2 = urllib.request.Request('http://127.0.0.1:39600/ping', data=b'{}', method='POST', headers={'Content-Type': 'application/json', 'X-API-Key': 'secret'})
    with urllib.request.urlopen(req2, timeout=2) as resp:
        out = json.loads(resp.read().decode('utf-8'))
    server.stop()
    assert ok
    assert out['ok'] is True


def test_local_automation_api_health_lists_routes():
    server = LocalAutomationApiServer('127.0.0.1', 39601, {'ping': lambda body: {'pong': 1}}, api_key='secret')
    server.start()
    req = urllib.request.Request('http://127.0.0.1:39601/health', method='GET', headers={'X-API-Key': 'secret'})
    with urllib.request.urlopen(req, timeout=2) as resp:
        out = json.loads(resp.read().decode('utf-8'))
    server.stop()
    assert out['ok'] is True
    assert out['status'] == 'ok'
    assert out['count'] == 1
    assert 'ping' in out['routes']
    assert out['methods']['GET'] == ['health', 'routes']
    assert out['methods']['POST'] == ['ping']


def test_local_automation_api_routes_endpoint_shape():
    handlers = {
        'open_project': lambda body: {'ok': True},
        'project_status': lambda body: {'pages': 3},
    }
    server = LocalAutomationApiServer('127.0.0.1', 39602, handlers)
    server.start()
    req = urllib.request.Request('http://127.0.0.1:39602/routes', method='GET')
    with urllib.request.urlopen(req, timeout=2) as resp:
        out = json.loads(resp.read().decode('utf-8'))
    server.stop()
    assert out['ok'] is True
    assert out['count'] == 2
    assert out['routes'] == ['open_project', 'project_status']
    assert out['methods']['GET'] == ['health', 'routes']
    assert out['methods']['POST'] == ['open_project', 'project_status']


def test_local_automation_api_routes_are_sorted_for_stability():
    handlers = {
        'zeta': lambda body: {},
        'alpha': lambda body: {},
        'mid': lambda body: {},
    }
    server = LocalAutomationApiServer('127.0.0.1', 39603, handlers)
    server.start()
    req = urllib.request.Request('http://127.0.0.1:39603/routes', method='GET')
    with urllib.request.urlopen(req, timeout=2) as resp:
        out = json.loads(resp.read().decode('utf-8'))
    server.stop()
    assert out['routes'] == ['alpha', 'mid', 'zeta']
    assert out['methods']['POST'] == ['alpha', 'mid', 'zeta']
