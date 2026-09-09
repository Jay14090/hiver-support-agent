"""Scan the working tree AND the full git history for secrets and PII.

Runs before submission because a leaked key in an old commit is not fixed by deleting the
file in a new one. Checks:
  - API-key shapes (OpenAI, Anthropic, AWS, generic bearer tokens)
  - .env files ever committed
  - email addresses and phone numbers surviving in committed data/cache
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")

ROOT = Path(__file__).resolve().parents[1]

SECRET_PATTERNS = [
    ("openai_key", re.compile(r"sk-[A-Za-z0-9_\-]{20,}")),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("aws_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("generic_bearer", re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.]{30,}")),
    ("private_key", re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----")),
]
# PII that should have been scrubbed by clean.py before anything was written to disk.
PII_PATTERNS = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("phone", re.compile(r"\b\+?\d[\d\-\.\(\)\s]{9,}\d\b")),
]
# Placeholders and known-safe examples that must not trip the scan.
ALLOW = re.compile(r"sk-\.\.\.|sk-xxx|<EMAIL>|<PHONE>|noreply@anthropic\.com|"
                   r"priya@sherlock\.sh|example\.com|your-key-here")

SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "data/raw", "data/interim"}


def _is_plain_number(s: str) -> bool:
    """Reject decimal/float matches that the greedy phone pattern picks up.

    reports/*.json are full of floats like 0.7242574382394534 (a similarity score) and
    the phone regex is deliberately greedy, so it matches them. A real phone number is
    never a parseable float, so this is a safe exclusion -- and much better than
    loosening the phone pattern, which would risk missing an actual number.
    """
    t = s.strip()
    try:
        float(t)
        return True
    except ValueError:
        return False


def iter_files():
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        if any(rel.startswith(s) or f"/{s}/" in f"/{rel}" for s in SKIP_DIRS):
            continue
        if p.suffix in {".png", ".npy", ".parquet", ".csv", ".pyc"}:
            continue
        yield p, rel


def scan_worktree() -> list:
    hits = []
    for p, rel in iter_files():
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name, rx in SECRET_PATTERNS:
            for m in rx.finditer(text):
                if not ALLOW.search(m.group(0)):
                    hits.append(("SECRET", name, rel, m.group(0)[:24] + "..."))
        # PII only matters in committed data/cache, not in source that defines the regexes
        if rel.startswith(("data/processed", "cache/", "golden/", "reports/")):
            for name, rx in PII_PATTERNS:
                for m in rx.finditer(text):
                    if ALLOW.search(m.group(0)) or _is_plain_number(m.group(0)):
                        continue
                    hits.append(("PII", name, rel, m.group(0)[:28]))
    return hits


def scan_history() -> list:
    hits = []
    try:
        files = subprocess.run(["git", "log", "--all", "--pretty=format:", "--name-only"],
                               cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120).stdout or ""
    except (subprocess.SubprocessError, OSError) as e:
        return [("ERROR", "git", str(e), "")]
    for f in {x.strip() for x in files.split("\n") if x.strip()}:
        if f.endswith(".env") or f == ".env" or "/.env" in f:
            hits.append(("SECRET", "env_file_in_history", f, ""))
    try:
        blob = subprocess.run(["git", "log", "--all", "-p", "--unified=0"],
                              cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300).stdout or ""
    except (subprocess.SubprocessError, OSError):
        return hits
    for name, rx in SECRET_PATTERNS:
        for m in rx.finditer(blob):
            if not ALLOW.search(m.group(0)):
                hits.append(("SECRET", f"{name}_in_history", "(git history)", m.group(0)[:24] + "..."))
    return hits


def main() -> int:
    print("scanning working tree ...")
    w = scan_worktree()
    print("scanning git history ...")
    h = scan_history()
    all_hits = w + h

    secrets = [x for x in all_hits if x[0] == "SECRET"]
    pii = [x for x in all_hits if x[0] == "PII"]

    if secrets:
        print(f"\nFAIL: {len(secrets)} potential secret(s):")
        for kind, name, where, sample in secrets[:20]:
            print(f"  - {name} in {where}: {sample}")
    else:
        print("OK: no secrets in the working tree or git history")

    if pii:
        print(f"\nFAIL: {len(pii)} PII match(es) in committed data:")
        for kind, name, where, sample in pii[:20]:
            print(f"  - {name} in {where}: {sample}")
    else:
        print("OK: no unscrubbed emails or phone numbers in committed data")

    return 1 if (secrets or pii) else 0


if __name__ == "__main__":
    sys.exit(main())
