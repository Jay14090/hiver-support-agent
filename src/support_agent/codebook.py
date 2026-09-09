"""Load CODEBOOK.md and render it for prompts.

The codebook markdown is the SINGLE source of truth for label definitions: the classifier
prompt, both pre-label passes and the human adjudication CLI all read this file. If they
each carried their own copy of the definitions they would drift, and the golden labels
would end up measuring a different taxonomy from the one the classifier was told about.
"""
from __future__ import annotations

import re
from functools import lru_cache

from . import config

CODEBOOK_PATH = config.GOLDEN / "CODEBOOK.md"


@lru_cache(maxsize=4)
def parse_codebook() -> dict:
    """Return {intent: {definition, tie_breaks: [...], positives: [...]}}."""
    if not CODEBOOK_PATH.exists():
        return {}
    text = CODEBOOK_PATH.read_text(encoding="utf-8")
    out: dict = {}
    # Sections are "## `intent_name`"
    for m in re.finditer(r"^## `([a-z_]+)`\n(.*?)(?=^## `|\Z)", text, re.S | re.M):
        intent, body = m.group(1), m.group(2)
        d = re.search(r"\*\*Definition:\*\*\s*(.+?)(?=\n\n)", body, re.S)
        definition = re.sub(r"\s+", " ", d.group(1)).strip() if d else ""
        ties = [re.sub(r"\s+", " ", t).strip()
                for t in re.findall(r"\*\*Tie-break[^:]*:\*\*\s*(.+?)(?=\n\n|\n\*\*)", body, re.S)]
        pos = re.findall(r'^- "(.+?)"\s*$', body, re.M)
        out[intent] = {"definition": definition, "tie_breaks": ties, "positives": pos[:3]}
    return out


@lru_cache(maxsize=2)
def load_codebook_text(include_tiebreaks: bool = True) -> str:
    """Compact rendering for a system prompt -- definitions plus tie-break rules.

    Deliberately not the whole markdown file: the examples would triple the prompt cost
    on every classification call, and few-shot examples are supplied separately and
    dynamically by retrieval.
    """
    cb = parse_codebook()
    lines = []
    for intent in config.INTENTS:
        e = cb.get(intent)
        if not e:
            lines.append(f"- {intent}")
            continue
        lines.append(f"- {intent}: {e['definition']}")
        if include_tiebreaks:
            for t in e["tie_breaks"]:
                lines.append(f"    tie-break: {t}")
    return "\n".join(lines)


def missing_intents() -> list:
    """Intents in config with no codebook entry -- the P2.5 gate checks this is empty."""
    cb = parse_codebook()
    return [i for i in config.INTENTS if i not in cb]
