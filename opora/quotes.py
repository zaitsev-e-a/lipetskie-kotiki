"""Проверка цитат: границы считает код, в ответ идёт только текст оригинала."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

from .documents import Document
from .textnorm import norm, one_line, all_numbers

_QUOTES = set("«»„“”\"'")
_DASHES = set("–—−")


def norm_with_map(s: str) -> tuple[str, list[int]]:
    """norm() с картой «позиция в нормализованной строке → позиция в оригинале»."""
    out, idx = [], []
    prev_space = True  # схлопываем ведущие пробелы (strip)
    for i, ch in enumerate(s):
        c = ch.lower()[:1] if ch else ch
        if c == "ё":
            c = "е"
        if c in (" ", " ") or c.isspace():
            if prev_space:
                continue
            c = " "
            prev_space = True
        else:
            prev_space = False
            if c in _QUOTES:
                c = '"'
            elif c in _DASHES:
                c = "-"
        out.append(c)
        idx.append(i)
    while out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


@dataclass
class Evidence:
    file: str
    quote: str  # точный текст оригинала [start, end)
    start: int
    end: int
    line: int
    page: int | None = None
    value: str = ""
    role: str = "main"  # main | operand | side
    checker_ok: bool = False  # пройдёт ли официальный check.py
    superseded_by: dict | None = None  # изменение, отменившее эту редакцию
    repaired: bool = False  # цитату пришлось поправить кодом
    context: str = ""  # строка документа целиком (для панели доказательств)

    def to_dict(self) -> dict:
        return asdict(self)


def _find_in_doc(nq: str, doc: Document) -> tuple[int, int] | None:
    nt, mp = norm_with_map(doc.text)
    pos = nt.find(nq)
    if pos < 0:
        return None
    if nt.find(nq, pos + 1) >= 0:
        # несколько совпадений: берём первое, но это нормально для одинаковых фраз
        pass
    start = mp[pos]
    end = mp[pos + len(nq) - 1] + 1
    return start, end


def _make_evidence(doc: Document, start: int, end: int, value: str, role: str, repaired=False) -> Evidence:
    q = doc.text[start:end]
    li = doc.line_of(start)
    return Evidence(file=doc.name, quote=q, start=start, end=end, line=li, page=doc.page_of(start),
                    value=value, role=role, checker_ok=len(norm(q)) >= 8 and norm(q) in doc.checker_norm,
                    repaired=repaired, context=doc.line_text(li).strip()[:1200])


def _tokens(s: str) -> list[str]:
    return re.findall(r"[0-9a-zа-яе]+", norm(s))


def _sentences(line: str, base: int) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in re.finditer(r"(?<=[.;:!?])\s+(?=[А-ЯЁA-Z0-9«])", line):
        spans.append((base + start, base + m.start()))
        start = m.end()
    spans.append((base + start, base + len(line)))
    return [(s, e) for s, e in spans if e > s]


def locate(quote: str, docs: list[Document], file_hint: str = "", value: str = "",
           role: str = "main") -> Evidence | None:
    """Находит цитату модели в документах. Сначала указанный файл, потом остальные."""
    nq = norm(quote)
    if len(nq) < 4:
        return None
    ordered = sorted(docs, key=lambda d: 0 if d.name == (file_hint or "").strip() else 1)
    for d in ordered:
        hit = _find_in_doc(nq, d)
        if hit:
            return _make_evidence(d, hit[0], hit[1], value, role)
    # Без многоточий и обрывков: модель иногда обрезает цитату «…»
    trimmed = norm(re.sub(r"(\.\.\.|…)", " ", quote))
    parts = [p.strip() for p in re.split(r"\s{2,}|\" \"", trimmed) if len(p.strip()) >= 12]
    for p in sorted(parts, key=len, reverse=True):
        for d in ordered:
            hit = _find_in_doc(p, d)
            if hit:
                return _make_evidence(d, hit[0], hit[1], value, role, repaired=True)
    return fuzzy_locate(quote, ordered, value, role)


def fuzzy_locate(quote: str, docs: list[Document], value: str = "", role: str = "main",
                 accept=None) -> Evidence | None:
    """Кандидат по совпадению слов. В ответ идёт предложение оригинала, а не текст модели.

    accept(текст предложения) → bool — необязательный фильтр: предложения, меняющие смысл
    цитаты, не рассматриваются."""
    qt = _tokens(quote)
    if len(qt) < 3:
        return None
    qset = set(qt)
    vnums = all_numbers(value) if value else set()
    best, best_score = None, 0.0
    for d in docs:
        for i in range(len(d.lines)):
            s, e = d.lines[i]
            if e - s < 8:
                continue
            for ss, se in _sentences(d.text[s:e], s):
                st = set(_tokens(d.text[ss:se]))
                if not st or (accept is not None and not accept(d.text[ss:se])):
                    continue
                score = len(qset & st) / len(qset)
                if vnums and not (vnums & all_numbers(d.text[ss:se])):
                    score *= 0.5
                if score > best_score:
                    best, best_score = (d, ss, se), score
    if best and best_score >= 0.8:
        d, ss, se = best
        return _make_evidence(d, ss, se, value, role, repaired=True)
    return None


def expand_for_numbers(ev: Evidence, docs: list[Document], needed: set[float]) -> Evidence:
    """Если число ответа есть в той же строке, расширяем цитату до предложения с ним."""
    if not needed or needed <= all_numbers(ev.quote):
        return ev
    doc = next((d for d in docs if d.name == ev.file), None)
    if doc is None:
        return ev
    ls, le = doc.lines[ev.line]
    sents = _sentences(doc.text[ls:le], ls)
    chosen = [(s, e) for s, e in sents if (s < ev.end and e > ev.start)]
    for s, e in sents:
        if (needed - all_numbers(ev.quote)) & all_numbers(doc.text[s:e]):
            chosen.append((s, e))
    if not chosen:
        return ev
    s, e = min(c[0] for c in chosen), max(c[1] for c in chosen)
    if e - s > 400 or not (needed & all_numbers(doc.text[s:e])):
        return ev
    new = _make_evidence(doc, s, e, ev.value, ev.role, repaired=ev.repaired)
    new.superseded_by = ev.superseded_by
    return new


def expand_to_substring(ev: Evidence, docs: list[Document], needles: list[str]) -> Evidence:
    """Расширяет цитату до предложения(й) строки, где есть все needles (например, дата и время)."""
    if all(n in ev.quote for n in needles):
        return ev
    doc = next((d for d in docs if d.name == ev.file), None)
    if doc is None:
        return ev
    ls, le = doc.lines[ev.line]
    sents = _sentences(doc.text[ls:le], ls)
    chosen = [(s, e) for s, e in sents if s < ev.end and e > ev.start]
    chosen += [(s, e) for s, e in sents if any(n in doc.text[s:e] for n in needles)]
    if not chosen:
        return ev
    s, e = min(c[0] for c in chosen), max(c[1] for c in chosen)
    if e - s > 400:
        return ev
    new = _make_evidence(doc, s, e, ev.value, ev.role, repaired=ev.repaired)
    new.superseded_by = ev.superseded_by
    return new


@dataclass
class QuoteProjection:
    """Итог форматной проекции цитаты «было»: текст, файл и что поменялось."""
    quote: str
    file: str
    evidence: Evidence | None = None
    verbatim: bool = False  # итоговая цитата дословно есть в документах
    changes: list[str] = field(default_factory=list)


_SPLIT_RX = re.compile(r"\.\.\.|…|\s\|\s|\||\n|\s/\s")
_EDGE = " \t\"'«»„“”.,;:—–-()[]"


def _strip_markers(q: str) -> str:
    q = re.sub(r"\[\d+\]\s*", "", q or "")
    return re.sub(r"\{(⛔|⚠)[^}]*\}\s*", "", q)


def _verbatim_doc(nq: str, docs: list[Document], file_hint: str) -> Document | None:
    """Документ, в котором цитата есть дословно — так, как её проверит check.py."""
    if len(nq) < 8:
        return None
    ordered = sorted(docs, key=lambda d: 0 if d.name == (file_hint or "").strip() else 1)
    hit = next((d for d in ordered if d.checker_norm and nq in d.checker_norm), None)
    return hit or next((d for d in ordered if _find_in_doc(nq, d)), None)


_NEG_RX = re.compile(r"(?<![0-9a-zа-я])не |(?<![0-9a-zа-я])нет(?![0-9a-zа-я])|отсутств")
_ENUM_RX = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+")


def _has_negation(text: str) -> bool:
    return bool(_NEG_RX.search(norm(text) + " "))


def same_meaning(model_text: str, doc_text: str) -> bool:
    """Предложение документа не меняет смысл цитаты модели: все её числа на месте
    (номер пункта в начале цитаты — не значение) и та же полярность отрицания."""
    need = all_numbers(_ENUM_RX.sub("", one_line(model_text)))
    have = all_numbers(doc_text)
    if any(all(abs(n - h) > 0.005 for h in have) for n in need):
        return False
    return _has_negation(model_text) == _has_negation(doc_text)


def _fuzzy_same(text: str, docs: list[Document]) -> Evidence | None:
    return fuzzy_locate(text, docs, accept=lambda sent: same_meaning(text, sent))


def project_quote(raw_quote: str, pkg, file_hint: str = "", answer: str = "") -> QuoteProjection:
    """Форматная проекция цитаты ответа «было». Смысл не меняет, фрагмент выбирает модель.

    1. Убираем номера строк [N] и внешние кавычки.
    2. Цитата дословно есть в документах → оставляем (только в одну строку).
    3. Склейка через «...», «…», « | », перевод строки, « / »: каждый кусок ≥ 20 символов должен
       найтись дословно или нечётко (см. 4); иначе в склейке выдумка и цитата остаётся исходной.
       Из дословных кусков — тот, где больше чисел из ответа модели, затем самый длинный,
       при равенстве — первый.
    4. Иначе предложение оригинала по совпадению слов (fuzzy_locate, ≥ 0.8), только с теми же
       числами и той же полярностью отрицания.
    5. Иначе цитата модели остаётся как есть: это выдумка, и оценщик честно её засчитает.
    """
    docs = pkg.docs if hasattr(pkg, "docs") else list(pkg)
    hint = (file_hint or "").strip()
    original = raw_quote or ""
    changes: list[str] = []
    if not original.strip():
        return QuoteProjection("", "", changes=changes)

    stripped = _strip_markers(original)
    cleaned = one_line(stripped).strip().strip("«»„“”\"' ").strip()
    if one_line(stripped) != one_line(original):
        changes.append("цитата: убраны номера строк [N]")
    if cleaned != one_line(stripped).strip() and cleaned:
        changes.append("цитата: сняты внешние кавычки")

    def result(text: str, doc: Document, ev: Evidence | None, verbatim: bool) -> QuoteProjection:
        if hint and doc.name != hint:
            changes.append(f"файл: «{hint}» → «{doc.name}» (там найдена цитата)")
        elif not hint:
            changes.append(f"файл: указан «{doc.name}» (там найдена цитата)")
        return QuoteProjection(text, doc.name, ev, verbatim, changes)

    def evidence_of(text: str, doc: Document, repaired: bool) -> Evidence | None:
        hit = _find_in_doc(norm(text), doc)
        return _make_evidence(doc, hit[0], hit[1], "", "main", repaired) if hit else None

    def from_fuzzy(ev: Evidence) -> QuoteProjection:
        text = one_line(ev.quote)
        doc = next(d for d in docs if d.name == ev.file)
        changes.append("цитата: дословно не найдена — взято предложение документа с теми же словами и числами")
        return result(text, doc, ev, len(norm(text)) >= 8 and (not doc.checker_norm or norm(text) in doc.checker_norm))

    def fabricated() -> QuoteProjection:
        return QuoteProjection(one_line(original), one_line(hint), None, False, changes)

    ordered = sorted(docs, key=lambda d: 0 if d.name == hint else 1)
    ans_nums = all_numbers(answer) if answer else set()

    # 2. дословно целиком
    doc = _verbatim_doc(norm(cleaned), docs, hint)
    if doc is not None:
        if "\n" in original.strip():
            changes.append("цитата: переводы строк заменены пробелами")
        return result(cleaned, doc, evidence_of(cleaned, doc, False), True)

    # 3. склейка кусков
    pieces = []
    for p in _SPLIT_RX.split(stripped):
        p = one_line(p).strip()
        if len(norm(p.strip(_EDGE))) >= 8:
            pieces.append(p)
    found, fuzzy_found, lost = [], [], []
    for i, p in enumerate(pieces):
        # кусок как есть, затем без знаков на краях (обрывки «руб.,», «(90)…»)
        hit = None
        for variant in dict.fromkeys((p, p.strip(_EDGE))):
            d = _verbatim_doc(norm(variant), docs, hint)
            if d is not None:
                hit = (variant, d)
                break
        if hit is not None:
            variant, d = hit
            n_ans = sum(1 for n in ans_nums if any(abs(n - x) < 0.005 for x in all_numbers(variant)))
            found.append((n_ans, len(variant), -i, variant, d))
            continue
        if len(pieces) > 1:
            fev = _fuzzy_same(p, ordered) if len(norm(p.strip(_EDGE))) >= 8 else None
            if fev is not None:
                fuzzy_found.append((len(p), -i, fev))
            elif len(norm(p.strip(_EDGE))) >= 20:
                lost.append(p)
    if len(pieces) > 1 and lost:
        changes.append(f"цитата: в склейке есть фрагмент, которого нет в документах («{lost[0][:80]}») — "
                       "оставлена как у модели")
        return fabricated()
    if found:
        _, _, _, best, d = max(found)  # больше чисел ответа; затем самый длинный; при равенстве — первый
        changes.append(f"цитата: выбран дословный фрагмент из склейки ({len(pieces)} кусков, найдено {len(found)})"
                       if len(pieces) > 1 else "цитата: убраны лишние знаки по краям")
        return result(best, d, evidence_of(best, d, True), True)

    # 4. предложение оригинала по совпадению слов — без смены чисел и отрицания
    ev = _fuzzy_same(cleaned, ordered)
    if ev is None and fuzzy_found:
        ev = max(fuzzy_found, key=lambda x: x[:2])[2]
    if ev is not None:
        return from_fuzzy(ev)

    # 5. выдумка остаётся выдумкой
    changes.append("цитата: в документах не найдена — оставлена как у модели")
    return fabricated()


def csv_quote(ev: Evidence, limit: int = 300) -> str:
    """Цитата для answers.csv: одна строка, не длиннее limit, проходит check.py."""
    q = one_line(ev.quote)
    if len(q) <= limit:
        return q
    # Обрезаем по границе слова: обрезанная подстрока всё ещё дословна.
    cut = q[:limit]
    return cut[: cut.rfind(" ")] if " " in cut else cut
