from __future__ import annotations

import re
import unicodedata


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", str(text)) if text else ""


def parse_money(text: str) -> int | None:
    if text is None:
        return None

    s = nfc(str(text)).replace(",", "")

    patterns = [
        (r"(\d+(?:\.\d+)?)\s*억\s*원?", 100_000_000),
        (r"(\d+(?:\.\d+)?)\s*만원", 10_000),
        (r"(\d+(?:\.\d+)?)\s*천원", 1_000),
        (r"(\d+(?:\.\d+)?)\s*원", 1),
    ]

    for pat, unit in patterns:
        m = re.search(pat, s)
        if m:
            return int(float(m.group(1)) * unit)

    only_num = re.sub(r"[^0-9.]", "", s)
    if only_num:
        try:
            return int(float(only_num))
        except ValueError:
            return None

    return None
