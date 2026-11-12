"""Records of check runs: every panel step and every `check.py` part writes its full numbers to files, so a
run can be read back later without rerunning it.

Layout of one run (`<root>/<kind>/<stamp>/`, stamp = local time YYYY-mm-dd_HHMMSS):

    summary.json        verdict, device, provenance (git commit, versions) and every step's result
    report.txt          the human-readable summary and each step's lines
    log.txt             everything the run printed
    <step>/step.json    the step's verdict, seconds, lines and numbers
    <step>/output.txt   what the step printed
    <step>/...          step-specific data (data.json, report.txt, junit.xml, tests.csv, fields .npz)

`<root>/<kind>/latest.txt` names the newest run; `<root>/<kind>/history.csv` has one line per run.
This module needs only the standard library (numpy and torch objects are converted if they are passed).
"""

from __future__ import annotations

import csv
import dataclasses
import datetime
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

MAX_ARRAY = 20000  # arrays with more elements are summarised (shape, dtype, min, max) instead of listed


def run_stamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")


def run_dir(root: str, kind: str, stamp: str | None = None) -> tuple[str, str]:
    """Create `<root>/<kind>/<stamp>/` (a suffix is added if it exists) and return (path, stamp)."""
    stamp = stamp or run_stamp()
    base = os.path.join(os.path.abspath(root), kind)
    path, k = os.path.join(base, stamp), 1
    while os.path.exists(path):
        k += 1
        path = os.path.join(base, f"{stamp}_{k}")
    os.makedirs(path)
    return path, os.path.basename(path)


def mark_latest(root: str, kind: str, stamp: str) -> None:
    with open(os.path.join(os.path.abspath(root), kind, "latest.txt"), "w", encoding="utf-8") as f:
        f.write(stamp + "\n")


def append_history(root: str, kind: str, row: dict) -> None:
    """One line per run in `<root>/<kind>/history.csv`. The header is that of the file; a row with a column
    the file does not have yet (a field added to the records later, e.g. git_dirty) extends the header: the
    file is rewritten with the new columns at the end and empty cells in the older rows (through a temporary
    file, so an interruption leaves the old file), instead of the column being dropped. Columns of the file
    that the row lacks stay empty in its line."""
    path = os.path.join(os.path.abspath(root), kind, "history.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    header, old_rows = list(row), []
    if not new:
        with open(path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None) or []
        missing = [k for k in row if k not in header]
        if missing or not header:
            with open(path, newline="", encoding="utf-8") as f:
                old_rows = list(csv.DictReader(f)) if header else []
            header = header + missing
            tmp = path + ".tmp"
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=header)
                w.writeheader()
                w.writerows(old_rows)
            os.replace(tmp, path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new:
            w.writeheader()
        w.writerow(row)


def to_jsonable(obj, max_array: int = MAX_ARRAY) -> object:
    """A JSON-serialisable copy: dataclasses and mappings become dicts, numpy and torch arrays lists (or a
    summary above max_array elements), non-finite floats strings, everything else str()."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name), max_array) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v, max_array) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v, max_array) for v in obj]
    mod = type(obj).__module__
    if mod.startswith("torch") and hasattr(obj, "detach"):
        obj = obj.detach().cpu()
        if obj.is_complex():
            obj = obj.abs()
        obj = obj.numpy()
        mod = "numpy"
    if mod.startswith("numpy"):
        import numpy as np

        a = np.asarray(obj)
        if np.iscomplexobj(a):
            a = np.abs(a)
        if a.ndim == 0:
            return to_jsonable(a.item(), max_array)
        if a.size <= max_array:
            return to_jsonable(a.tolist(), max_array)
        finite = a[np.isfinite(a)] if a.dtype.kind == "f" else a
        return dict(
            array_shape=list(a.shape),
            dtype=str(a.dtype),
            min=to_jsonable(float(finite.min()) if finite.size else None),
            max=to_jsonable(float(finite.max()) if finite.size else None),
        )
    if callable(getattr(obj, "item", None)):
        try:
            return to_jsonable(obj.item(), max_array)
        except (TypeError, ValueError):
            pass
    return str(obj)


def write_json(path: str, obj, max_array: int = MAX_ARRAY) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj, max_array), f, indent=1)


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text if text.endswith("\n") else text + "\n")


def junit_to_csv(xml_path: str, csv_path: str) -> dict:
    """One row per test of a pytest --junitxml file (test, outcome, seconds, message) and the counts."""
    counts = dict(passed=0, failed=0, error=0, skipped=0)
    rows = []
    if not os.path.exists(xml_path):
        return counts
    for case in ET.parse(xml_path).getroot().iter("testcase"):
        outcome, message = "passed", ""
        for tag in ("failure", "error", "skipped"):
            el = case.find(tag)
            if el is not None:
                outcome = {"failure": "failed"}.get(tag, tag)
                message = (el.get("message") or "").splitlines()[0][:300] if el.get("message") else ""
                break
        counts[outcome] += 1
        name = f"{case.get('classname', '')}::{case.get('name', '')}"
        rows.append(dict(test=name, outcome=outcome, seconds=case.get("time", ""), message=message))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["test", "outcome", "seconds", "message"])
        w.writeheader()
        w.writerows(rows)
    return counts


class Tee:
    """Copy everything written to sys.stdout and sys.stderr into a file while it still reaches the console."""

    def __init__(self, path: str):
        self.path = path

    def __enter__(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.f = open(self.path, "w", encoding="utf-8")
        self.saved = sys.stdout, sys.stderr
        sys.stdout = _Both(self.saved[0], self.f)
        sys.stderr = _Both(self.saved[1], self.f)
        return self

    def __exit__(self, *exc):
        sys.stdout, sys.stderr = self.saved
        self.f.close()
        return False


class _Both:
    def __init__(self, console, f):
        self.console, self.f = console, f

    def write(self, s):
        self.console.write(s)
        self.f.write(s)
        return len(s)

    def flush(self):
        self.console.flush()
        self.f.flush()

    def __getattr__(self, name):  # isatty, encoding, fileno ... from the console stream
        return getattr(self.console, name)
