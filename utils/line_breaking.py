from typing import Callable, List, Optional, Tuple


def _badness(slack: float) -> float:
    # A common "pretty" heuristic: cubic penalty of leftover space
    if slack < 0:
        return 1e12
    return slack * slack * slack


# Penalty per line break to prefer fewer, longer lines (avoids 1–2 words per line)
# Make this strong but not so extreme that lines blow past bubble width.
_LINE_BREAK_PENALTY = 110.0


def find_optimal_breaks_dp(
    widths: List[int],
    max_width: int,
    delimiter_width: int = 0,
) -> List[int]:
    """
    Given token widths, return break indices (end-exclusive) for optimal line breaking.
    This is a simplified Knuth-Plass-style DP (no stretch/shrink), minimizing raggedness.

    Returns a list of indices where each line ends, e.g. [3, 7, 10] for 10 tokens.
    """
    n = len(widths)
    if n == 0:
        return []
    if max_width <= 0:
        return [n]

    # prefix sums
    ps = [0]
    for w in widths:
        ps.append(ps[-1] + int(w))

    # dp[i] = best cost for first i tokens
    dp = [1e18] * (n + 1)
    prev = [-1] * (n + 1)
    dp[0] = 0.0

    for i in range(1, n + 1):
        # consider break at j->i
        best = 1e18
        best_j = -1
        for j in range(0, i):
            tokens = i - j
            line_w = (ps[i] - ps[j]) + delimiter_width * max(0, tokens - 1)
            slack = float(max_width - line_w)
            # Penalize each line break to prefer fewer, longer lines (avoids 1-2 words per line)
            line_break_cost = _LINE_BREAK_PENALTY if j > 0 else 0.0
            cost = dp[j] + _badness(slack) + line_break_cost
            if cost < best:
                best = cost
                best_j = j
        dp[i] = best
        prev[i] = best_j

    # reconstruct
    ends = []
    i = n
    while i > 0:
        j = prev[i]
        if j < 0:
            break
        ends.append(i)
        i = j
    ends.reverse()
    return ends if ends else [n]


def hyphenate_word_pyphen(word: str, lang: str = "en_US") -> Optional[List[str]]:
    """
    Optional hyphenation using pyphen. Returns list of syllables (without hyphen).
    """
    try:
        import pyphen  # type: ignore

        dic = pyphen.Pyphen(lang=lang)
        h = dic.inserted(word)
        if not h or h == word:
            return None
        parts = [p for p in h.split("-") if p]
        return parts if len(parts) >= 2 else None
    except Exception:
        return None


def split_long_token_with_hyphenation(
    token: str,
    measure: Callable[[str], int],
    max_width: int,
    hyphenate: bool = True,
    hyphen_lang: str = "en_US",
) -> List[Tuple[str, int]]:
    """
    If token is wider than max_width, try to split it into smaller pieces with hyphens.
    Returns list of (piece, piece_width) where pieces may include trailing hyphen.
    """
    tok = token or ""
    w = int(measure(tok))
    if w <= max_width or max_width <= 0:
        return [(tok, w)]
    if not hyphenate:
        return [(tok, w)]

    parts = hyphenate_word_pyphen(tok, lang=hyphen_lang)
    if not parts:
        return [(tok, w)]

    out: List[Tuple[str, int]] = []
    cur = ""
    for i, p in enumerate(parts):
        nxt = (cur + p) if cur else p
        # add hyphen if not last syllable
        probe = nxt + ("-" if i < len(parts) - 1 else "")
        pw = int(measure(probe))
        if pw <= max_width or not cur:
            cur = nxt
            continue
        # flush current with hyphen
        piece = cur + "-"
        out.append((piece, int(measure(piece))))
        cur = p
    if cur:
        out.append((cur, int(measure(cur))))

    # If we failed to reduce width, return original
    if len(out) == 1 and out[0][0] == tok:
        return [(tok, w)]
    return out

