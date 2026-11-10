"""Детерминированные проверки утверждений и итога.

Код проверяет типы и привязки: источник из текущего снимка/закупки, точный срез, число и единица в
цитате, привязку числа к своему атрибуту внутри цитаты, действующую редакцию, полноту частей ответа.
Соответствие смыслу вопроса проверяет отдельный семантический валидатор (LLM). Ни одну из проверок
не выдаём за полное понимание естественного языка.
"""
from __future__ import annotations

import re

from .knowledge import value_matches_quote, typed_key
from .sources import PackageSources, SourceError, clean_model_quote
from .textnorm import numbers, norm

UNIT_MARKERS = {
    "percent": (r"%|процент",),
    "rub": (r"руб|₽",),
    "days": (r"дн|день|дня|сут",),
    "working_days": (r"рабоч",),
    "months": (r"месяц|мес\b",),
    "years": (r"\bгод|\bлет\b",),
}
_STOP = {"который", "которые", "которых", "также", "этого", "этом", "если", "иные", "иных", "более", "менее",
         "должен", "должна", "должны", "указанный", "размер", "размере", "значение"}


def stems(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[а-яa-z]{4,}", norm(text)) if w not in _STOP}


def _clauses(quote: str) -> list[tuple[int, int]]:
    """Части цитаты между ; , . — но не внутри чисел (1 800 000,00; 27.11.2026)."""
    cuts = [m.start() for m in re.finditer(r"[;,.](?!\d)", quote)]
    spans, prev = [], 0
    for c in cuts:
        spans.append((prev, c))
        prev = c + 1
    spans.append((prev, len(quote)))
    return [(s, e) for s, e in spans if quote[s:e].strip()]


def binding_problem(claim: dict, quote: str) -> str | None:
    """Число из claim должно относиться к его атрибуту: если в той же цитате другая часть с другим числом
    лучше совпадает с подписью атрибута, число взято не оттуда («5000 за заявку, 10000 за исполнение»)."""
    vals = set(numbers(claim.get("value") or "")) or set(numbers(claim.get("raw_value") or ""))
    label = stems(claim.get("attribute_label") or "")
    if not vals or not label:
        return None
    parts = _clauses(quote)
    scored, prev = [], 0
    for s, e in parts:
        seg = quote[s:e]
        nums = set(numbers(seg))
        own = len(label & stems(seg))
        # часть без собственной подписи продолжает предыдущую («…обеспечение заявки 1 % от НМЦК, что составляет
        # 18 000 руб.»): наследует её совпадение с подписью атрибута
        sc = own if own else prev
        scored.append((sc, nums, seg))
        prev = sc
    mine = [sc for sc, nums, _ in scored if vals & nums]
    others = [sc for sc, nums, _ in scored if nums and not (vals & nums)]
    if mine and others and max(others) > max(mine):
        best = max(scored, key=lambda x: x[0])
        return (f"число {', '.join(f'{v:g}' for v in vals)} в цитате относится к другой части "
                f"(«{best[2].strip()[:80]}» точнее соответствует «{claim.get('attribute_label')}»)")
    return None


def unit_problem(claim: dict, quote: str) -> str | None:
    unit = (claim.get("unit") or "").lower()
    q = norm(quote)
    pats = UNIT_MARKERS.get(unit)
    if pats and not any(re.search(p, q) for p in pats):
        return f"единица «{unit}» не подтверждена цитатой"
    if unit == "days" and re.search(r"рабоч\w* дн", q) and not re.search(r"календар", q):
        return "в цитате рабочие дни, а утверждение — о календарных"
    if unit == "working_days" and re.search(r"календар", q) and not re.search(r"рабоч", q):
        return "в цитате календарные дни, а утверждение — о рабочих"
    raw = norm(claim.get("comparator") or claim.get("raw_value") or "")
    if "не менее" in raw and "не более" in q and "не менее" not in q:
        return "в цитате «не более», а утверждение — «не менее»"
    if "не более" in raw and "не менее" in q and "не более" not in q:
        return "в цитате «не менее», а утверждение — «не более»"
    return None


_UNIT_SYNONYMS = [
    ("working_days", r"^(working_days|раб\w*\s*дн\w*|р\.?\s*д\.?)$"),
    ("days", r"^(days?|дн\w*|сут\w*|календарн\w*\s*дн\w*|к\.?\s*д\.?)$"),
    ("rub", r"^(rub|руб\w*\.?|₽|рубл\w*|р\.?)$"),
    ("percent", r"^(percent|%|проц\w*|процент\w*)$"),
    ("months", r"^(months?|мес\w*\.?)$"),
    ("years", r"^(years?|лет|год\w*)$"),
    ("count", r"^(count|шт\.?|штук\w*|ед\.?|человек\w*|чел\.?|договор\w*|контракт\w*)$"),
    ("datetime", r"^(datetime|дата\s*и\s*время)$"),
    ("date", r"^(date|дата)$"),
    ("bool", r"^(bool|boolean|да/нет)$"),
    ("text", r"^(text|текст|string)?$"),
]


def normalize_unit(unit: str, raw_value: str = "", quote: str = "") -> str:
    """Единица от модели → код схемы (rub, percent, days, working_days, …). Неизвестная остаётся как есть
    и потом проверяется по цитате; «дней» уточняется до рабочих/календарных по самой цитате."""
    u = re.sub(r"\s+", " ", (unit or "").strip().lower().replace("ё", "е"))
    code = next((c for c, rx in _UNIT_SYNONYMS if re.match(rx, u)), u)
    if code == "days" and re.search(r"рабоч\w*\s+дн", norm(raw_value + " " + quote)) and \
            not re.search(r"календар", norm(raw_value + " " + quote)):
        code = "working_days"
    return code


def verify_claim(ps: PackageSources, raw: dict, allowed: set[str] | None, window_id: str,
                 asks_original: bool) -> tuple[dict | None, list[str]]:
    """Строгое разрешение доказательств claim + проверки значения, единицы и привязки к атрибуту."""
    errors, evs = [], []
    for e in raw.get("evidence") or []:
        try:
            evs.append(ps.resolve(str(e.get("source_id", "")), str(e.get("quote", "")), allowed).to_dict())
        except SourceError as err:
            errors.append(f"[{err.source_id}] {err}: «{clean_model_quote(str(e.get('quote', '')))[:100]}»")
    if not evs:
        return None, errors or ["у утверждения нет доказательства"]
    # дословная цитата подозрительной команды не делает её фактом закупки: доказательство, пересекающее
    # находку защиты от инъекций, отбрасывается (остальные доказательства утверждения остаются)
    bad = [e for e in evs if ps.guard_findings(e["document_id"], e["start"], e["end"])]
    if bad:
        errors = errors + [f"[{e['source_id']}] цитата взята из фрагмента, похожего на инструкцию для ИИ, — он не "
                           "может быть источником факта" for e in bad]
        evs = [e for e in evs if e not in bad]
        if not evs:
            return None, errors
    claim = {"attribute": (raw.get("attribute") or "").strip().lower(),
             "attribute_label": raw.get("attribute_label") or "", "entity": raw.get("entity") or "",
             "raw_value": raw.get("raw_value") or "", "value": raw.get("value") or "",
             "unit": normalize_unit(raw.get("unit") or "", raw.get("raw_value") or "",
                                    " ".join(e["quote"] for e in evs)), "comparator": raw.get("comparator") or "",
             "conditions": raw.get("conditions") or [], "polarity": raw.get("polarity") or "positive",
             "answers_question": bool(raw.get("answers_question")), "evidence": evs, "window_id": window_id}
    quotes = " ".join(e["quote"] for e in evs)
    if claim["polarity"] != "negative" and claim["unit"] not in ("text", "bool", ""):
        why = value_matches_quote(claim["value"] or claim["raw_value"], claim["unit"], quotes)
        if why:
            return None, [f"{claim['attribute_label'] or claim['attribute']}: {why}"]
        holder = next((e["quote"] for e in evs if set(numbers(claim["value"])) & set(numbers(e["quote"]))),
                      evs[0]["quote"])
        for check in (binding_problem(claim, holder), unit_problem(claim, holder)):
            if check:
                return None, [f"{claim['attribute_label'] or claim['attribute']}: {check}"]
    claim["superseded"] = (not asks_original) and any(e.get("superseded_by") for e in evs)
    claim["typed"] = list(typed_key(claim))
    return claim, errors


def claim_key(c: dict) -> tuple:
    """Дубли перекрытий: тот же атрибут, значение и диапазон — одно утверждение, а не два подтверждения."""
    e = c["evidence"][0]
    return (c["attribute"], tuple(c["typed"]), e["document_id"], e["start"], e["end"])


def final_problems(ps: PackageSources, cand: dict, claims: dict) -> list[str]:
    """Финальная детерминированная проверка итога перед экспортом."""
    out = []
    st = cand.get("status")
    if st not in ("ответ", "расхождение", "нет в документах", "не знаю"):
        return [f"недопустимый статус {st!r}"]
    if st in ("ответ", "расхождение"):
        used = [claims[i] for i in cand.get("claim_ids") or [] if i in claims]
        if not used:
            return ["у ответа нет подтверждённых утверждений"]
        for c in used:
            for e in c["evidence"]:
                why = ps.verify_ref(e)
                if why:
                    out.append(f"{c['claim_id']}: {why}")
        if st == "ответ" and cand.get("calculation") is None:
            old = [c for c in used if c.get("superseded")]
            if old and len(old) == len(used):
                out.append("ответ опирается только на недействующую редакцию")
        if st == "расхождение":
            cur = [c for c in used if not c.get("superseded")]
            if len({tuple(c["typed"]) for c in cur}) < 2:
                out.append("расхождение: нет двух действующих сторон с разными значениями")
        if not str(cand.get("answer") or "").strip():
            out.append("пустой ответ")
    return out
