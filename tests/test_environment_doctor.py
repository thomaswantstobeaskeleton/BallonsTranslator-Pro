from utils.environment_doctor import run_environment_doctor


def test_environment_doctor_returns_structured_checks():
    checks = run_environment_doctor()
    assert isinstance(checks, list)
    assert checks
    for item in checks:
        assert isinstance(item, tuple)
        assert len(item) == 3
    names = {c[0] for c in checks}
    assert 'python' in names
    assert 'auth:huggingface' in names
