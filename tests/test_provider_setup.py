from utils.provider_setup import check_provider_connection, provider_endpoint_preset


def test_provider_endpoint_preset_for_ollama():
    assert provider_endpoint_preset('Ollama').startswith('http://localhost:11434')


def test_provider_connection_success_parses_model_count():
    def fake_fetch(url, headers, timeout):
        return 200, '{"data":[{"id":"a"},{"id":"b"}]}'
    rst = check_provider_connection('OpenAI', 'https://api.openai.com/v1', api_key='sk-test', fetcher=fake_fetch)
    assert rst.ok is True
    assert '2 model' in rst.detail


def test_provider_connection_failure_status():
    def fake_fetch(url, headers, timeout):
        return 401, '{}'
    rst = check_provider_connection('OpenRouter', 'https://openrouter.ai/api/v1', api_key='bad', fetcher=fake_fetch)
    assert rst.ok is False
    assert rst.status_code == 401
