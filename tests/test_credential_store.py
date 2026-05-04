from utils import credential_store as cs


def test_get_secret_without_keyring(monkeypatch):
    monkeypatch.setattr(cs, "_keyring_module", lambda: None)
    assert cs.get_secret("x") is None
    assert cs.set_secret("x", "y") is False
