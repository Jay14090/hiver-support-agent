# Blockers

Things that failed and how I worked around them.

## B-01 — `autonomous-build-loop` skill not installed
- **Attempts:** checked `.claude/skills/` in the project, `~/.claude/skills/`, and the session's registered skill list. Only `gsd-*` and vendor skills are present.
- **Fallback taken:** the brief's §0 restates the skill's rules in full ("restated so there is zero ambiguity"). I follow those. State is persisted to `STATE.json` / `PROGRESS.md` after every task exactly as the skill requires.
- **Status:** worked around, no impact on deliverables.

## B-02 — No Kaggle API credentials
- **Attempts:** `~/.kaggle/kaggle.json` absent; `KAGGLE_USERNAME`/`KAGGLE_KEY` unset; `kaggle datasets list` returned `You must authenticate before you can call the Kaggle API.`
- **Fallback taken:** downloaded the identical `twcs.csv` from the HuggingFace mirror `SunidhiSriram/twcs`. `scripts/get_data.sh` still attempts Kaggle first and prints exact credential-setup instructions before falling back.
- **Status:** worked around. Schema and row count are verified against the published dataset in Phase 1.
