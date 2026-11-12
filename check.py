"""Run every check of the repository; one summary at the end.

    python check.py                 lint + the test suite (a few minutes on 2 CPU cores)
    python check.py --all           + the slow tests: every test over ~20 s on one core
    python check.py --only tests    one part (see --list)

Run from the repository root after `pip install -e ".[dev]"`. Everything runs on the CPU (GPU runs are
production runs). Exit status 0 only if every selected part passed. The checks are pinned to
twisted_bilayer_lattice/_shared/config.py as shipped.

Every run is kept in results/check/<date_time>/ (git-ignored): summary.json, one <part>.txt per part with its
output, and for the pytest parts <part>_junit.xml and <part>_tests.csv with every test (outcome, seconds,
message). results/check/latest.txt names the newest run, results/check/history.csv has one line per run.
`--no-record` writes nothing.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from twisted_bilayer_lattice._shared import records as REC  # noqa: E402  (standard library only)

PY = sys.executable
CPU_ENV = {"PYTHONUTF8": "1", "OMP_NUM_THREADS": "1", "TWISTED_BILAYER_FORCE_CPU": "1", "MPLBACKEND": "Agg"}

PARTS = {
    # name: (command, runs by default, what)
    "lint": ([PY, "-m", "ruff", "check", "."], True, "ruff: lint"),
    "format": ([PY, "-m", "ruff", "format", "--check", "."], True, "ruff: formatting"),
    "tests": ([PY, "-m", "pytest", "-q"], True, "pytest: the lattice model, Gutzwiller, clusters, phase maps"),
    "tests_slow": (
        [PY, "-m", "pytest", "-q", "-m", "slow"],
        False,
        "the slow tests: the tests over ~20 s on one core",
    ),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="include the slow parts")
    ap.add_argument("--only", default="", help="comma-separated part names")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-record", action="store_true", help="do not write results/check/<date_time>/")
    a = ap.parse_args(argv)
    if a.list:
        for k, (_, default, what) in PARTS.items():
            print(f"{k:12s} {'default' if default else 'with --all':11s} {what}")
        return 0
    if a.only:
        names = [n.strip() for n in a.only.split(",") if n.strip()]
        if unknown := [n for n in names if n not in PARTS]:
            ap.error(f"unknown part(s) {unknown}; see --list")
    else:
        names = [k for k, v in PARTS.items() if v[1] or a.all]
    env = dict(os.environ, **CPU_ENV)
    out = stamp = None
    if not a.no_record:
        out, stamp = REC.run_dir(ROOT / "results", "check")
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    results = []
    for n in names:
        cmd, _, what = PARTS[n]
        print(f"\n== {n}: {what}", flush=True)
        extra = []
        if out and cmd[2] == "pytest":
            extra = [f"--junitxml={os.path.join(out, n + '_junit.xml')}", "--durations=30", "-o", "junit_family=xunit2"]
        t0 = time.time()
        rc = _run(cmd + extra, env, os.path.join(out, f"{n}.txt") if out else None)
        row = dict(part=n, ok=rc == 0, exit=rc, seconds=round(time.time() - t0, 1))
        if extra:
            row["tests"] = REC.junit_to_csv(os.path.join(out, n + "_junit.xml"), os.path.join(out, f"{n}_tests.csv"))
        results.append(row)
    print("\n== summary")
    for r in results:
        s = "ok" if r["ok"] else f"FAILED (exit {r['exit']})"
        print(f"  {r['part']:12s} {s:18s} {r['seconds']:7.0f} s")
    failed = [r["part"] for r in results if not r["ok"]]
    print("all selected checks passed" if not failed else f"FAILED: {', '.join(failed)}")
    if out:
        tests = {r["part"]: r["tests"] for r in results if "tests" in r}
        REC.write_json(
            os.path.join(out, "summary.json"),
            dict(started=started, verdict="FAIL" if failed else "PASS", parts=results, git=_git()),
        )
        REC.mark_latest(ROOT / "results", "check", stamp)
        REC.append_history(
            ROOT / "results",
            "check",
            dict(
                run=stamp,
                verdict="FAIL" if failed else "PASS",
                parts=" ".join(r["part"] for r in results),
                failed_parts=" ".join(failed),
                passed_tests=sum(t["passed"] for t in tests.values()),
                failed_tests=sum(t["failed"] + t["error"] for t in tests.values()),
                minutes=round(sum(r["seconds"] for r in results) / 60, 1),
                git_commit=_git().get("commit"),
            ),
        )
        print(f"wrote {out}")
    return 1 if failed else 0


def _run(cmd, env, path) -> int:
    """Run a command from the repository root; its output goes to the console and, if path is given, to that file."""
    if path is None:
        return subprocess.call(cmd, cwd=ROOT, env=env)
    p = subprocess.Popen(
        cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8"
    )
    with open(path, "w", encoding="utf-8") as f:
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
    return p.wait()


def _git() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=5).stdout.strip()

    try:
        return dict(commit=git("rev-parse", "--short", "HEAD") or None, dirty=bool(git("status", "--porcelain")))
    except (OSError, subprocess.SubprocessError):
        return {}


if __name__ == "__main__":
    sys.exit(main())
