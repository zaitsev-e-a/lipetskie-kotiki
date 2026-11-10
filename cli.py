"""Пакетный прогон без браузера — те же модули, что и у сайта.

python cli.py --packages путь/к/packages --questions questions.csv --baseline --gold answers.json
Результат: data/runs/<id>/answers.csv, baseline.csv, trace.json, evaluation/…
"""
import argparse
import os
import sys
import time
from pathlib import Path

if sys.version_info < (3, 10):
    sys.exit("Нужен Python 3.10 или новее")

from opora.config import settings
from opora.store import Registry, parse_questions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packages", required=True, help="папка с подпапками закупок (одна подпапка — одна закупка)")
    ap.add_argument("--questions", required=True, help="CSV id;package;question")
    ap.add_argument("--baseline", action="store_true", help="также прогнать «было» (один запрос)")
    ap.add_argument("--gold", help="файл эталонных ответов (JSON) — оценить через check.py после прогона")
    ap.add_argument("--no-passport", action="store_true", help="отвечать без паспорта (паспорт всё равно строится "
                    "для политики отсутствия, если не задан --skip-passport)")
    ap.add_argument("--mode", default=None, choices=["auto", "full_context", "retrieval", "exhaustive"],
                    help="режим чтения (по умолчанию READ_MODE из .env)")
    ap.add_argument("--resume-run", help="продолжить прерванный прогон по его id")
    ap.add_argument("--out", help="куда дополнительно скопировать answers.csv")
    a = ap.parse_args()

    if not settings.api_key:
        print("Не задан OPENROUTER_API_KEY (файл .env или переменная окружения)")
        return 2
    reg = Registry(settings.data_dir)
    if a.resume_run:
        run = reg.run(a.resume_run)
        print(f"Продолжаю прогон {run.id}: отвечено {len(run.final)} из {len(run.questions)}")
        run.start(resume=True)
        return _wait_and_report(run, a)
    c = reg.new_collection()
    root = Path(a.packages)
    n = 0
    for dirpath, _, files in os.walk(root):
        for f in files:
            full = Path(dirpath) / f
            rel = full.relative_to(root.parent).as_posix()
            if not c.add_file(rel, full.read_bytes()).get("skipped"):
                n += 1
    print(f"Файлов загружено: {n}")
    t0 = time.time()
    res = reg.start_ingest(c, background=False)
    if res.get("use"):
        c = reg.collection(res["use"])
        print(f"Этот снимок уже подготовлен: {c.id}")
    print(f"Подготовка: {time.time() - t0:.1f} с; снимок {c.meta.get('snapshot_id')}; закупок: {len(c.packages)}; "
          f"состояние: {c.meta['state']}")
    for w in c.meta.get("warnings") or []:
        print("  [!]", w)
    for pk in c.packages:
        s = c.package_summary(pk)
        conf = [x for x in s["conflicts"] if x["state"] == "confirmed"]
        print(f"  {pk}: файлов {len(s['docs'])}, ~{s['tokens']} токенов, окон {s['windows']}, "
              f"изменений {len(s['amendments'])}, расхождений {len(conf)}, индекс: "
              f"{'есть' if s['index'] else 'нет'}")
    questions = parse_questions(Path(a.questions).read_bytes(), c)
    opts = {"use_passport": not a.no_passport}
    if a.mode:
        opts["read_mode"] = a.mode
    run = reg.new_run(c, questions, baseline=a.baseline, options=opts)
    run.start()
    return _wait_and_report(run, a)


def _wait_and_report(run, a) -> int:
    questions = run.questions
    t1 = time.time()
    total = len(questions) * (2 if run.baseline else 1)
    while run.state == "running":
        time.sleep(1)
        done = len(run.final) + len(run.base)
        print(f"\r  ответов: {done}/{total}  ({time.time() - t1:.0f} с)", end="", flush=True)
    print()
    snap = run.snapshot()
    for v in ("final", "baseline"):
        m = snap["metrics"].get(v)
        if m:
            print(f"{'Стало' if v == 'final' else 'Было '}: вызовов {m['calls']}, токенов "
                  f"{m['prompt_tokens'] + m['completion_tokens']}, стоимость {m['cost_usd']} $, "
                  f"p50 {m['p50_ms'] / 1000:.1f} с, статусы {m['status']}")
    print(f"Время прогона: {time.time() - t1:.1f} с. Файлы: {run.dir}")
    if a.out:
        Path(a.out).write_bytes((run.dir / "answers.csv").read_bytes())
        print(f"answers.csv скопирован в {a.out}")
    if a.gold:
        ev = run.evaluate(Path(a.gold).read_bytes())
        for v, r in ev["variants"].items():
            res = r["result"]
            print(f"{'Стало' if v == 'final' else 'Было '}: верно {res['correct']}, осторожно {res['careful']}, "
                  f"обман {res['lies']}, индекс доверия {res['index']}")
        print(f"check.py не изменён: {ev['integrity']['unchanged']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
