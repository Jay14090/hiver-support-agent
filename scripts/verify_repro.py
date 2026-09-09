"""Prove the 15-minute reproduction claim by actually doing it.

Clones the repo into a temp directory, creates a fresh venv, installs deps, and runs
`make demo` with **no API key in the environment** and `LLM_OFFLINE=1` (so any cache miss
is a hard error rather than a silent live call). Times the whole thing and writes
reports/repro_timing.txt.

Claiming "reproducible in under 15 minutes" without running it from a clean clone is
exactly the kind of unverified assertion this project is supposed to avoid.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, ".")
from src.support_agent import config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def run(cmd, cwd, env=None, timeout=1800):
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or ""), time.time() - t0


def main() -> int:
    skip_install = "--skip-install" in sys.argv
    tmp = Path(tempfile.mkdtemp(prefix="hiver_repro_"))
    clone = tmp / "hiver-support-agent"
    print(f"[repro] cloning into {clone}")

    total0 = time.time()
    rc, out, t_clone = run(["git", "clone", "--quiet", str(ROOT), str(clone)], cwd=tmp)
    if rc != 0:
        print(f"FAIL: clone failed\n{out[:2000]}")
        return 1
    print(f"[repro] clone: {t_clone:.0f}s")

    # A grader has no key. Strip it, and force offline so a cache miss is fatal.
    env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
    env["LLM_OFFLINE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    t_setup = 0.0
    py = sys.executable
    if not skip_install:
        print("[repro] creating venv + installing deps (this is the slow part)")
        rc, out, t1 = run([sys.executable, "-m", "venv", ".venv"], cwd=clone, env=env)
        vpy = clone / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
        rc2, out2, t2 = run([str(vpy), "-m", "pip", "install", "-q", "torch",
                             "--index-url", "https://download.pytorch.org/whl/cpu"],
                            cwd=clone, env=env)
        rc3, out3, t3 = run([str(vpy), "-m", "pip", "install", "-q", "-r", "requirements.txt"],
                            cwd=clone, env=env)
        t_setup = t1 + t2 + t3
        print(f"[repro] setup: {t_setup:.0f}s")
        if rc3 != 0:
            print(f"FAIL: pip install failed\n{out3[-2000:]}")
            return 1
        py = str(vpy)

    print("[repro] running `make demo` equivalent (python run.py demo), offline, no API key")
    rc, out, t_demo = run([py, "run.py", "demo"], cwd=clone, env=env)
    print(out[-3000:])
    total = time.time() - total0

    ok = rc == 0
    report = (
        f"reproduction check\n"
        f"------------------\n"
        f"clone      : {t_clone:.1f} s\n"
        f"setup      : {t_setup:.1f} s\n"
        f"make demo  : {t_demo:.1f} s\n"
        f"TOTAL      : {total:.1f} s  ({total/60:.1f} min)\n"
        f"budget     : 900 s (15 min)\n"
        f"api key    : REMOVED from the environment\n"
        f"LLM_OFFLINE: 1 (a cache miss is a hard error)\n"
        f"result     : {'PASS' if ok and total < 900 else 'FAIL'}\n"
        f"exit code  : {rc}\n"
    )
    (config.REPORTS / "repro_timing.txt").write_text(report, encoding="utf-8")
    print("\n" + report)

    if not ok:
        print("demo failed -- tail of output:")
        print(out[-4000:])
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if (ok and total < 900) else 1


if __name__ == "__main__":
    sys.exit(main())
