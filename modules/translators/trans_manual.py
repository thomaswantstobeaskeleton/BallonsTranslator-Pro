import json
from typing import Dict, List, Optional

from .base import BaseTranslator, register_translator


def _try_parse_response(raw: str) -> Optional[Dict]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    # Try JSON first
    try:
        return json.loads(s)
    except Exception:
        return None


@register_translator("manual")
class ManualTranslator(BaseTranslator):
    """
    Manual translation workflow:
    - Generates a JSON prompt with ids + sources.
    - User pastes a JSON response mapping ids to translations.

    This is useful for offline workflows or using any external tool.
    """

    concate_text = False

    params: Dict = {
        "response_json": {
            "type": "editor",
            "value": "",
            "description": "Paste response JSON here. Accepts: {\"translations\":[{\"id\":1,\"translation\":\"...\"}]} or {\"1\":\"...\"} or [\"t1\",\"t2\",...].",
        },
        "write_prompt_path": {
            "value": "",
            "description": "Optional: write the generated prompt JSON to this file path when response_json is empty.",
        },
        "description": "Manual translator: generates a prompt JSON and applies pasted response JSON.",
    }

    def _setup_translator(self):
        # Ensure all LANGMAP keys (except Auto) are "supported" so valid_lang_list is non-empty
        # and fallback in base works when source/target are invalid (e.g. from another translator).
        for k in list(self.lang_map.keys()):
            if k != 'Auto' and self.lang_map[k] == '':
                self.lang_map[k] = k

    def _build_prompt(self, src_list: List[str]) -> Dict:
        return {
            "instruction": (
                "Translate each 'source' to the target language. Return JSON only.\n"
                "Accepted response formats:\n"
                "1) {\"translations\":[{\"id\":1,\"translation\":\"...\"}, ...]}\n"
                "2) {\"1\":\"...\",\"2\":\"...\"}\n"
                "3) [\"...\", \"...\", ...] (same order)\n"
            ),
            "source_language": getattr(self, "lang_source", ""),
            "target_language": getattr(self, "lang_target", ""),
            "items": [{"id": i + 1, "source": s} for i, s in enumerate(src_list)],
        }

    def _apply_response(self, src_list: List[str], resp_obj: object) -> Optional[List[str]]:
        n = len(src_list)
        if isinstance(resp_obj, list):
            if len(resp_obj) != n:
                return None
            return [str(x) if x is not None else "" for x in resp_obj]

        if isinstance(resp_obj, dict):
            if "translations" in resp_obj and isinstance(resp_obj["translations"], list):
                out = [""] * n
                for el in resp_obj["translations"]:
                    if not isinstance(el, dict):
                        continue
                    try:
                        idx = int(el.get("id")) - 1
                    except Exception:
                        continue
                    if 0 <= idx < n:
                        out[idx] = str(el.get("translation") or "")
                return out
            # id->translation dict
            out = [""] * n
            ok = False
            for k, v in resp_obj.items():
                try:
                    idx = int(k) - 1
                except Exception:
                    continue
                if 0 <= idx < n:
                    out[idx] = str(v) if v is not None else ""
                    ok = True
            return out if ok else None
        return None

    def _translate(self, src_list: List[str]) -> List[str]:
        raw_resp = (self.get_param_value("response_json") or "").strip()
        prompt_obj = self._build_prompt(src_list)
        prompt_text = json.dumps(prompt_obj, ensure_ascii=False, indent=2)
        self.last_prompt = prompt_text

        if raw_resp:
            resp_obj = _try_parse_response(raw_resp)
            out = self._apply_response(src_list, resp_obj)
            if out is not None:
                return out
            # Bad response: fall back to returning source
            self.logger.warning("Manual translator: response_json could not be parsed/applied; leaving translation unchanged.")
            return src_list

        # No response: write prompt if configured, and return source so pipeline continues.
        path = (self.get_param_value("write_prompt_path") or "").strip()
        if path:
            try:
                with open(path, "w", encoding="utf8") as f:
                    f.write(prompt_text)
            except Exception as e:
                self.logger.warning(f"Manual translator: failed to write prompt to {path}: {e}")
        else:
            # Log once per call so headless users still see it.
            self.logger.info("Manual translator prompt:\n" + prompt_text)
        return src_list

