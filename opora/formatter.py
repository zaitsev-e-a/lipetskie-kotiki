"""Форматтер ответа: формат выбирается по тексту вопроса, а не по эталону.

Правила формата ответа (совместимы с check.py кейса):
«да»/«нет» без лишних слов, даты ДД.ММ.ГГГГ ЧЧ:ММ, числа без пробелов
между разрядами, несколько чисел через «; », расхождение — оба значения.
"""
from __future__ import annotations

import re

from .textnorm import find_datetimes, one_line, to_decimal, fmt_decimal

STATUSES = ("ответ", "расхождение", "нет в документах", "не знаю")

NEGATION_RX = re.compile(
    r"не\s+(требуется|установлен\w*|предусмотрен\w*|нужн\w*|применя\w*|допуска\w*|обязательн\w*)|"
    r"отсутству\w*|исключ\w*|не\s+вправе|\bнет\b", re.IGNORECASE)


_NEG_WORD = re.compile(r"(?<![а-я])(не|нет)(?![а-я])|отсутств")


def normalize_status(s: str | None) -> str | None:
    """Строка → один из четырёх статусов или None.

    Статус распознаётся, только если строка (без регистра и пунктуации) — ровно статус или
    начинается с него, а в остатке нет отрицаний и другого статуса: «ответ отсутствует»,
    «расхождение не выявлено», «не знаю/нет в документах» — не статусы."""
    t = re.sub(r"[^а-яе ]", " ", (s or "").lower().replace("ё", "е"))
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None
    for st in STATUSES:
        if t == st:
            return st
        if t.startswith(st + " "):
            rest = t[len(st):].strip()
            if _NEG_WORD.search(rest) or any(x in rest for x in STATUSES):
                return None
            return st
    return None


def question_kind(q: str) -> str:
    ql = (q or "").lower().replace("ё", "е")
    if re.search(r"ответьте\s+да\s+или\s+нет|\(да/нет\)|да или нет\??\s*$", ql):
        return "yesno"
    if re.match(r"^\s*(установлен[оаы]?|требуется|нужн[аоы]?|должн[аоы]?|может|можно|включается|есть|"
                r"предусмотрен[аоы]?|допускается|является|вправе|обязан[аоы]?|применяется|разрешается)\s+ли\b", ql):
        return "yesno_detail" if re.search(r"если да|укажите|какой|сколько", ql) else "yesno"
    if re.search(r"(дат[аеуы]|числ[оа]|когда)\b.*\bвремен|до какой даты|дата и время|с какой даты", ql):
        return "datetime"
    if re.search(r"какой дат|какая дата|до какой|с какой даты|какого числа", ql):
        return "date"
    if re.search(r"сколько|размер|в рублях|в процентах|какая .*цена|какова .*цена|какой срок|"
                 r"в календарных днях|в месяцах|в днях|площадь|шаг аукциона|процент", ql):
        return "number"
    return "text"


_NUM_TOKEN = re.compile(r"\d+(?:[ \u00a0\u202f]\d{3})*(?:[.,]\d+)?")  # как check.numbers
_GROUPED = re.compile(r"(?<![\d.,])\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?(?![\d])")


def normalize_numbers(text: str) -> str:
    """«1 800 000,00 руб.» → «1800000 руб.»: оценщик не спутает разряды с соседними числами."""
    def repl(m: re.Match) -> str:
        d = to_decimal(m.group(0))
        return fmt_decimal(d) if d is not None else m.group(0)

    text = _GROUPED.sub(repl, text)
    return re.sub(r"(?<![\d.,])(\d+)[.,]00(?![\d])", r"\1", text)


def _value_str(v: str) -> str:
    v = one_line(v)
    d = to_decimal(v) if re.fullmatch(r"[\d   .,]+", v or "") else None
    return fmt_decimal(d) if d is not None else normalize_numbers(v)


def yes_no_polarity(answer: str) -> str:
    """«да» — ответ начинается с «да»; «нет» — начинается с «нет» или прямо отрицает требование;
    иначе "" (полярность не выражена)."""
    low = one_line(answer).lower()
    if re.match(r"^\s*да\b", low):
        return "да"
    if re.match(r"^\s*нет\b", low) or NEGATION_RX.search(low):
        return "нет"
    return ""


# номер пункта/раздела/статьи/приложения — не значение расхождения
_REF_RX = re.compile(r"(?<![а-я])п\.|пункт|раздел|(?<![а-я])ст\.|№|приложени")


def _is_reference(text: str, pos: int) -> bool:
    window = text[max(0, pos - 40):pos].lower().replace("ё", "е")
    # маркер не дальше 8 символов, и между ним и числом нет других чисел («№ 2 — 75»: 75 — значение)
    return any(len(window) - m.end() <= 8 and not re.search(r"\d", window[m.end():])
               for m in _REF_RX.finditer(window))


def format_answer(question: str, status: str, answer: str, values: list[str] | None = None,
                  computed: str | None = None) -> str:
    if status not in ("ответ", "расхождение"):
        return ""
    kind = question_kind(question)
    a = one_line(answer)
    vals = [_value_str(v) for v in (values or []) if one_line(v)]
    if computed is not None:
        return computed
    if status == "расхождение":
        uniq = list(dict.fromkeys(vals))
        if len(uniq) >= 2:
            return "; ".join(uniq)
        # значений списком нет («В извещении 60 дней, а в проекте контракта — 75»): берём
        # значения из текста по порядку появления, без дублей
        dts = list(dict.fromkeys(f"{d} {t}" if t else d for d, t in find_datetimes(a)))
        if len(dts) >= 2:
            return "; ".join(dts)
        nums = list(dict.fromkeys(_value_str(m.group(0)) for m in _NUM_TOKEN.finditer(a)
                                  if not _is_reference(a, m.start())))
        if len(nums) >= 2:
            return "; ".join(nums)
        return normalize_numbers(a)
    if kind in ("yesno", "yesno_detail"):
        pol = yes_no_polarity(a)
        if pol == "нет":
            # «нет» засчитывается, а слово «да» в ответе «нет» запрещено
            return "нет"
        if pol == "да":
            if kind == "yesno":
                return "да"
            rest = re.sub(r"^\s*да\b[\s,.:;—-]*", "", a, flags=re.IGNORECASE)
            rest = re.sub(r"\bнет\b", "", normalize_numbers(rest), flags=re.IGNORECASE).strip(" ,;")
            return f"да, {rest}" if rest else "да"
        # явной полярности нет: ответ остаётся текстом, «да» не подставляется
        return normalize_numbers(a)
    if NEGATION_RX.search(a.lower()) and not re.search(r"\d", a):
        # прямое отрицание требования: «не установлено», «не требуется»
        return re.sub(r"\bда\b", "", a, flags=re.IGNORECASE).strip() or "не установлено"
    if kind in ("datetime", "date"):
        found = find_datetimes(a) or [(v, None) for v in vals if re.match(r"\d{2}\.\d{2}\.\d{4}", v)]
        if found:
            d, t = found[0]
            if t is None:
                # время могло прийти отдельным значением
                tm = re.search(r"\b(\d{1,2}):(\d{2})\b", a)
                t = f"{int(tm.group(1)):02d}:{tm.group(2)}" if tm else None
            return f"{d} {t}" if t else d
    if kind == "number":
        nums = [v for v in vals if re.search(r"\d", v)]
        if len(nums) >= 2:
            return "; ".join(dict.fromkeys(nums))
        if len(nums) == 1 and not re.search(r"\d", a):
            return nums[0]
    return normalize_numbers(a)
