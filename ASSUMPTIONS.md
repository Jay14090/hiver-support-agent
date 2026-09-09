# Assumptions

Decisions I made instead of stopping to ask. Format:
`[A-nn] <decision> — <why> — <what would change my mind>`

- **[A-01]** Project lives at `D:\study stuff\hiver-support-agent`, not under the home directory — the home directory is itself a git repo full of unrelated dotfiles, and committing a take-home inside it would produce a filthy history. — Nothing; this is purely local hygiene.
- **[A-02]** `autonomous-build-loop` skill is **not installed** on this machine (only `gsd-*` skills exist). The brief restates its rules in full in §0, so I follow those restated rules verbatim rather than blocking. Logged in `BLOCKERS.md`. — Would change if the skill file appears at `.claude/skills/autonomous-build-loop/SKILL.md`.
- **[A-03]** Python **3.13.7** instead of the specified 3.11 — that is the interpreter installed here, and nothing in the stack requires 3.11. — A dependency that lacks 3.13 wheels.
- **[A-04]** Dataset fetched from the HuggingFace mirror `SunidhiSriram/twcs` (byte-identical `twcs.csv`) instead of the Kaggle CLI — no Kaggle credentials exist on this machine and the brief's fallback chain permits any local copy of `twcs.csv`. `scripts/get_data.sh` tries Kaggle first, then the mirror, so a grader with Kaggle creds gets the canonical path. — Would change if the mirror's row count/schema diverges from the published 2.81M-row dataset (verified in Phase 1).
- **[A-05]** Both generator and judge come from **OpenAI**, because `OPENAI_API_KEY` is the only key present. Per D7 this violates the "different model family" preference, so: generator = `gpt-4o-mini`, judge = `gpt-4.1-mini` (different model, same family), and the self-preference probe in Phase 6 is **mandatory, not optional** — the measured bias is reported as a limitation rather than assumed away. — An `ANTHROPIC_API_KEY` appearing in the environment.
