from utils.server_mode_info import build_server_mode_info


def test_server_mode_info_has_urls_and_paths():
    d = build_server_mode_info(host='127.0.0.1', port=39542)
    assert d['health_url'].endswith('/health')
    assert d['routes_url'].endswith('/routes')
    assert 'docker_mount_hints' in d
