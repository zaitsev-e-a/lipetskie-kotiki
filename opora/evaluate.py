"""Оценка неизменённым check.py организаторов в отдельном процессе.

Генератор ответов этот модуль не импортирует: эталоны появляются только здесь,
после того как answers.csv зафиксирован.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .config import ROOT

CHECK_PY = ROOT / "tools" / "check.py"
CHECK_SHA = (ROOT / "tools" / "check.py.sha256")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_integrity() -> dict:
    actual = sha256(CHECK_PY)
    expected = CHECK_SHA.read_text().split()[0] if CHECK_SHA.exists() else ""
    return {"sha256": actual, "expected": expected, "unchanged": actual == expected}


class EvalError(RuntimeError):
    pass


def load_gold(raw: bytes) -> list[dict]:
    try:
        gold = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise EvalError(f"файл эталонов не JSON: {e}")
    if not isinstance(gold, list) or not all(isinstance(g, dict) and "id" in g for g in gold):
        raise EvalError("ожидается список объектов с полями id, package, status, kind, expected, type")
    ids = [g["id"] for g in gold]
    if len(ids) != len(set(ids)):
        raise EvalError("в эталонах повторяются id")
    for g in gold:
        for k in ("package", "status", "kind", "expected", "type"):
            if k not in g:
                raise EvalError(f"у эталона {g['id']} нет поля {k}")
    return gold


def run_check(answers_csv: Path, gold_path: Path, packages_dir: Path, out_csv: Path) -> dict:
    proc = subprocess.run([sys.executable, str(CHECK_PY), str(answers_csv), "--gold", str(gold_path),
                           "--packages", str(packages_dir), "--out", str(out_csv)],
                          capture_output=True, timeout=300, env={**_env(), "PYTHONIOENCODING": "utf-8"})
    stdout = proc.stdout.decode("utf-8", "replace")
    stderr = proc.stderr.decode("utf-8", "replace")
    result = None
    for line in stdout.splitlines():
        if line.startswith("RESULT "):
            result = json.loads(line[7:])
    if proc.returncode != 0 or result is None:
        raise EvalError(f"check.py завершился с ошибкой: {stderr[-800:] or stdout[-800:]}")
    rows = []
    with open(out_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    return {"result": result, "rows": rows, "stdout": stdout, "stderr": stderr,
            "answers_sha256": sha256(answers_csv)}


def _env() -> dict:
    import os
    return dict(os.environ)


def expected_text(g: dict) -> str:
    kind, exp = g.get("kind"), g.get("expected") or []
    if g.get("status") == "нет в документах":
        return "—"
    if kind == "num":
        return "; ".join(f"{x:g}" if isinstance(x, (int, float)) else str(x) for x in exp)
    if kind == "date":
        return " ".join(exp)
    if kind == "yes":
        return "да"
    if kind == "no":
        return "нет / не установлено"
    if kind == "all_kw":
        return "ключевые слова: " + ", ".join(exp)
    return ", ".join(map(str, exp))


def by_type(rows: list[dict]) -> dict:
    out: dict = defaultdict(Counter)
    for r in rows:
        out[r["type"]][r["verdict"]] += 1
    return {t: {"всего": sum(c.values()), "верно": c["верно"], "осторожно": c["осторожно"], "обман": c["обман"]}
            for t, c in out.items()}
