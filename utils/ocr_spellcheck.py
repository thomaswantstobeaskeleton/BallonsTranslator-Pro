"""Optional spell check for OCR results. Uses pyenchant if available."""
from utils.logger import logger as LOGGER

_spell_checker = None
_spell_available = None


def _init_enchant():
    global _spell_checker, _spell_available
    if _spell_available is not None:
        return _spell_available
    try:
        import enchant
        try:
            _spell_checker = enchant.DictWithPWL(None, None)
        except Exception:
            _spell_checker = enchant.Dict("en_US") if enchant.dict_exists("en_US") else enchant.Dict("en")
        _spell_available = True
        return True
    except Exception as e:
        LOGGER.debug("Spell check (enchant) not available: %s", e)
        _spell_available = False
        return False


def spell_check_line(line: str, lang_hint: str = None):
    """Return corrected line if a single suggestion exists for a misspelled word; else return line."""
    if not line or not line.strip():
        return line
    if not _init_enchant() or _spell_checker is None:
        return line
    try:
        import enchant
        words = line.split()
        out = []
        for w in words:
            if not any(c.isalpha() for c in w):
                out.append(w)
                continue
            clean = "".join(c for c in w if c.isalpha() or c in "'-")
            if not clean:
                out.append(w)
                continue
            if _spell_checker.check(clean):
                out.append(w)
                continue
            suggs = _spell_checker.suggest(clean)
            if len(suggs) == 1:
                res = w
                for i, c in enumerate(w):
                    if c.isalpha():
                        # Replace alphabetic segment with suggestion; keep surrounding punctuation
                        end = i
                        while end < len(w) and (w[end].isalpha() or w[end] in "'-"):
                            end += 1
                        res = w[:i] + suggs[0] + w[end:]
                        break
                out.append(res)
            else:
                out.append(w)
        return " ".join(out)
    except Exception as e:
        LOGGER.debug("Spell check error: %s", e)
        return line


def spell_check_textblocks(textblocks, **kwargs):
    """Postprocess hook: run spell check on each block's text if pcfg.ocr_spell_check is True."""
    from utils.config import pcfg
    if not getattr(pcfg, "ocr_spell_check", False):
        return
    if not _init_enchant():
        return
    for blk in textblocks:
        if not getattr(blk, "text", None) or not isinstance(blk.text, list):
            continue
        blk.text = [spell_check_line(line) for line in blk.text]


def get_spell_issues(text: str) -> list:
    """Return list of (word, start_idx, end_idx, suggestions) for misspelled words. For SpellCheck panel."""
    if not text or not _init_enchant() or _spell_checker is None:
        return []
    import re
    issues = []
    for m in re.finditer(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", text):
        word = m.group(0)
        clean = "".join(c for c in word if c.isalpha() or c in "'-")
        if not clean or _spell_checker.check(clean):
            continue
        suggs = _spell_checker.suggest(clean)
        if suggs:
            issues.append((word, m.start(), m.end(), suggs))
    return issues
