"""Точки входа ответа на вопрос.

- answer_improved — «стало»: тонкая обёртка над графом LangGraph (opora/graph.py).
- answer_baseline — «было»: один логический запрос (вопрос + весь комплект + минимальная инструкция формата).
  Без паспорта, поиска, поправок, критика и поиска лучшего источника. Разрешён только общий прозрачный
  форматтер чисел/дат/статусов. Цитата и имя файла остаются такими, как их дала модель (только переносы
  строк схлопываются для CSV) — нечёткая замена цитаты форматной операцией не считается.
"""
from __future__ import annotations

import time

from . import prompts as P
from .formatter import format_answer, normalize_status
from .graph import QuestionCtx, answer_question
from .llm import LLMError, Tracker
from .sources import PackageSources, estimate_tokens, clean_model_quote
from .textnorm import norm, one_line


def answer_improved(ps: PackageSources, knowledge, retriever, question: str, llm, qid: str = "",
                    on_stage=None, history=None, journal=None, read_mode: str | None = None,
                    use_passport: bool | None = None, strict: bool | None = None) -> dict:
    from .config import settings
    ctx = QuestionCtx(ps=ps, knowledge=knowledge, retriever=retriever, llm=llm, history=list(history or []),
                      journal=journal, on_stage=on_stage,
                      read_mode=read_mode or settings.read_mode,
                      use_passport=settings.use_passport if use_passport is None else use_passport,
                      strict=settings.strict_coverage if strict is None else strict)
    return answer_question(ctx, question, qid)


def answer_baseline(ps: PackageSources, question: str, llm, qid: str = "", context_tokens: int | None = None) -> dict:
    tracker = Tracker()
    t0 = time.time()
    docs = "\n\n".join(f"Файл: {d.name}\n{d.text}" for d in ps.docs)
    user = P.BASELINE_USER.format(question=question, documents=docs)
    base = {"id": qid, "package": ps.package_id, "question": question, "variant": "baseline", "checks": {},
            "note": "", "mode": "full_context", "evidence": []}
    if context_tokens and estimate_tokens(user) + 2000 > context_tokens:
        # корпус не обрезается: фиксируем переполнение и отсутствие генерации
        return {**base, "status": "не знаю", "answer": "", "answer_text": "", "quote": "", "file": "",
                "context_overflow": True, "raw": None, "format_changes": ["context_overflow: генерации не было"],
                "calls": tracker.summary(), "call_log": [], "ms": 0}
    raw, error = {}, ""
    try:
        raw = llm.chat_json("baseline", P.BASELINE_SYSTEM, user, P.BASELINE_SCHEMA, tracker)
    except LLMError as e:
        error = f"{e.kind}: {e}"
    raw_fields = {k: str(raw.get(k) or "") for k in ("status", "answer", "quote", "file")} if raw else None
    changes = []
    status = normalize_status(raw_fields["status"]) if raw_fields else None
    if raw_fields is None:
        status = "не знаю"
        changes.append("технический сбой: ответа модели нет (учтён отдельно как protocol_error)")
    elif status is None:
        changes.append(f"статус «{raw_fields['status']}» не из четырёх допустимых → «не знаю»")
        status = "не знаю"
    elif status != one_line(raw_fields["status"]):
        changes.append(f"статус «{raw_fields['status']}» → «{status}»")
    answer = quote = file = ""
    if raw_fields and status in ("ответ", "расхождение"):
        answer = format_answer(question, status, raw_fields["answer"])
        if answer != one_line(raw_fields["answer"]):
            changes.append("ответ: числа/даты приведены к формату кейса")
        quote = one_line(clean_model_quote(raw_fields["quote"]))
        if quote != raw_fields["quote"]:
            changes.append("цитата: только переносы строк и служебные пометки")
        file = one_line(raw_fields["file"])
    found = bool(quote) and any(norm(quote) in norm(d.text) for d in ps.docs)
    return {**base, "status": status, "answer": answer, "answer_text": one_line(raw_fields["answer"]) if raw_fields else "",
            "quote": quote, "file": file, "quote_found": found, "raw": raw_fields, "format_changes": changes,
            "error": error, "protocol_error": bool(error), "calls": tracker.summary(),
            "call_log": tracker.to_list(), "ms": int((time.time() - t0) * 1000)}
