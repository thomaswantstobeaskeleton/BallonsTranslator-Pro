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
