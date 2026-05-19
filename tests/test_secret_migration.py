from types import SimpleNamespace
from utils.secret_migration import migrate_secrets_to_keyring_if_possible, scrub_plaintext_secrets_for_save


class _FakeKeyring:
    def __init__(self):
        self.store = {}
    def set_password(self, service, key, value):
        self.store[(service, key)] = value
    def get_password(self, service, key):
        return self.store.get((service, key))


def _pcfg():
    return SimpleNamespace(
        credential_migration_done=False,
        credential_use_plaintext_fallback=False,
        automation_api_key='autokey',
        module=SimpleNamespace(
            layout_review_api_key='sk-test',
            video_translator_flow_fixer_openrouter_apikey='',
            video_translator_flow_fixer_openai_apikey='sk-openai',
        )
    )


def test_secret_migration_moves_plaintext_and_clears_when_secure():
    fake = _FakeKeyring()
    pcfg = _pcfg()
    migrated = migrate_secrets_to_keyring_if_possible(
        pcfg,
        has_keyring=lambda: True,
        set_secret=lambda k, v: (fake.set_password('ballonstranslator-pro', k, v) or True),
    )
    assert migrated is True
    assert pcfg.credential_migration_done is True
    assert pcfg.module.layout_review_api_key == ''
    assert pcfg.module.video_translator_flow_fixer_openai_apikey == ''
    assert fake.get_password('ballonstranslator-pro', 'layout_review_api_key') == 'sk-test'


def test_scrub_plaintext_before_save_when_keyring_available():
    pcfg = _pcfg()
    scrub_plaintext_secrets_for_save(pcfg, has_keyring=lambda: True)
    assert pcfg.module.layout_review_api_key == ''
    assert pcfg.automation_api_key == ''
