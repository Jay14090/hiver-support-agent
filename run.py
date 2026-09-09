"""Cross-platform shim for the Makefile targets: `python run.py demo` == `make demo`.

Exists because this was developed on Windows, where `make` is not installed, and I
refuse to ship a Makefile whose targets I never actually executed. Both entry points
run the same command lists, defined once here and mirrored in the Makefile.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PY = ROOT / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
PY = str(VENV_PY) if VENV_PY.exists() else sys.executable

OFFLINE = {"LLM_OFFLINE": "1"}

TARGETS = {
    "data": [
        (["bash", "scripts/get_data.sh"], {}),
        ([PY, "-m", "src.support_agent.ingest", "--build"], {}),
        ([PY, "-m", "src.support_agent.ingest", "--verify"], {}),
    ],
    "demo": [
        ([PY, "-m", "eval.run_eval", "--system", "all", "--split", "golden", "--offline"], OFFLINE),
        ([PY, "-m", "eval.figures"], OFFLINE),
        ([PY, "-m", "eval.run_eval", "--headline"], OFFLINE),
    ],
    "full": [
        (["bash", "scripts/get_data.sh"], {}),
        ([PY, "-m", "src.support_agent.ingest", "--build"], {}),
        ([PY, "-m", "src.support_agent.taxonomy", "--induce"], {}),
        ([PY, "-m", "eval.run_eval", "--system", "all", "--split", "golden"], {}),
        ([PY, "-m", "eval.judge_validation", "--report", "--probes"], {}),
        ([PY, "-m", "eval.figures"], {}),
    ],
    "eval": [([PY, "-m", "eval.run_eval", "--system", "all", "--split", "golden"], {})],
    "test": [([PY, "-m", "pytest", "tests", "-q"], {})],
    "report": [([PY, "-m", "eval.figures"], {})],
    "gate": [([PY, "scripts/submission_gate.py"], {})],
    "check": [([PY, "-m", "pytest", "tests", "-q"], {}), ([PY, "scripts/submission_gate.py"], {})],
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in TARGETS:
        print("usage: python run.py {%s}" % "|".join(TARGETS))
        return 2
    for cmd, extra_env in TARGETS[sys.argv[1]]:
        env = {**os.environ, **extra_env}
        print(f"\n$ {' '.join(cmd)}")
        rc = subprocess.call(cmd, cwd=ROOT, env=env)
        if rc != 0:
            print(f"FAILED (exit {rc}): {' '.join(cmd)}")
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
