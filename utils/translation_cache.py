import json
import os
import os.path as osp
import hashlib
from typing import Any, Dict, Optional

from . import shared


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return json.dumps(str(obj), ensure_ascii=False)


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


class TranslationCache:
    """
    File-per-key translation cache under `.btrans_cache/translation_cache/`.

    Stores small JSON blobs so we avoid one giant file and keep lookups simple.
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        base = cache_dir or osp.join(shared.cache_dir, "translation_cache")
        self.cache_dir = osp.normpath(osp.abspath(base))
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except OSError:
            pass

    def key_to_path(self, key: str) -> str:
        h = _sha256_hex(key)
        # shard to avoid too many files per directory
        sub = osp.join(self.cache_dir, h[:2], h[2:4])
        try:
            os.makedirs(sub, exist_ok=True)
        except OSError:
            pass
        return osp.join(sub, f"{h}.json")

    def get(self, key_obj: Any) -> Optional[Dict]:
        key = _safe_json_dumps(key_obj)
        p = self.key_to_path(key)
        try:
            if not osp.exists(p):
                return None
            with open(p, "r", encoding="utf8") as f:
                return json.load(f)
        except Exception:
            return None

    def set(self, key_obj: Any, value: Dict) -> None:
        key = _safe_json_dumps(key_obj)
        p = self.key_to_path(key)
        try:
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf8") as f:
                json.dump(value, f, ensure_ascii=False)
            os.replace(tmp, p)
        except Exception:
            return

