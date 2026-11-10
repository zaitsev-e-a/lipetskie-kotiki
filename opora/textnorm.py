"""Нормализация текста ровно как в check.py организаторов + разбор чисел и дат.

Финальная проверка цитаты должна совпадать с тем, как её проверит оценщик,
поэтому norm() и numbers() повторяют check.py символ в символ.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

NBSP = " "
NNBSP = " "


def norm(s: str | None) -> str:
    """Копия check.norm: регистр, ё→е, кавычки, тире, пробелы."""
    s = (s or "").lower().replace("ё", "е").replace(NBSP, " ").replace(NNBSP, " ")
    s = re.sub("[«»„“”\"']", '"', s)
    s = re.sub("[–—−]", "-", s)
    return re.sub(r"\s+", " ", s).strip()


_NUM_RE = re.compile(r"\d+(?:[   ]\d{3})*(?:[.,]\d+)?")


def numbers(s: str | None) -> list[float]:
    """Копия check.numbers: так оценщик читает числа из ответа."""
    out = []
    for m in _NUM_RE.findall(s or ""):
        try:
            out.append(float(re.sub(r"[   ]", "", m).replace(",", ".")))
        except ValueError:
            pass
    return out


def one_line(s: str | None) -> str:
    """Схлопывает переносы и пробелы: check.py читает CSV построчно."""
    return re.sub(r"\s+", " ", (s or "")).strip()


def to_decimal(token: str) -> Decimal | None:
    t = re.sub(r"[   ]", "", token or "").replace(",", ".")
    t = re.sub(r"[^\d.]", "", t)
    if not t or t.count(".") > 1:
        return None
    try:
        return Decimal(t)
    except InvalidOperation:
        return None


def fmt_decimal(d: Decimal) -> str:
    """1800000.00 → 1800000; 5.50 → 5.5. Без разделителей разрядов."""
    q = d.normalize()
    s = format(q, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


# --- Числительные прописью (для проверки «тридцати дней», «семьдесят пять») ---
_UNITS = {
    "ноль": 0, "один": 1, "одна": 1, "одного": 1, "одной": 1, "одному": 1, "два": 2, "две": 2, "двух": 2,
    "три": 3, "трех": 3, "трёх": 3, "четыре": 4, "четырех": 4, "четырёх": 4, "пять": 5, "пяти": 5,
    "шесть": 6, "шести": 6, "семь": 7, "семи": 7, "восемь": 8, "восьми": 8, "девять": 9, "девяти": 9,
    "десять": 10, "десяти": 10, "одиннадцать": 11, "одиннадцати": 11, "двенадцать": 12, "двенадцати": 12,
    "тринадцать": 13, "тринадцати": 13, "четырнадцать": 14, "четырнадцати": 14, "пятнадцать": 15,
    "пятнадцати": 15, "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19,
}
_TENS = {
    "двадцать": 20, "двадцати": 20, "тридцать": 30, "тридцати": 30, "сорок": 40, "сорока": 40,
    "пятьдесят": 50, "пятидесяти": 50, "шестьдесят": 60, "шестидесяти": 60, "семьдесят": 70,
    "семидесяти": 70, "восемьдесят": 80, "восьмидесяти": 80, "девяносто": 90, "девяноста": 90,
}
_HUNDREDS = {
    "сто": 100, "ста": 100, "двести": 200, "двухсот": 200, "триста": 300, "трехсот": 300,
    "четыреста": 400, "пятьсот": 500, "пятисот": 500, "шестьсот": 600, "семьсот": 700,
    "восемьсот": 800, "девятьсот": 900,
}


def word_numbers(s: str) -> list[int]:
    """Находит числа, записанные словами: «тридцать шесть» → 36."""
    words = re.findall(r"[а-яё]+", (s or "").lower())
    out, cur, active = [], 0, False
    for w in words:
        if w in _HUNDREDS:
            if active and cur % 1000 >= 100:
                out.append(cur)
                cur = 0
            cur += _HUNDREDS[w]
            active = True
        elif w in _TENS:
            if active and cur % 100 >= 10:
                out.append(cur)
                cur = 0
            cur += _TENS[w]
            active = True
        elif w in _UNITS:
            if active and cur % 10 != 0:
                out.append(cur)
                cur = 0
            cur += _UNITS[w]
            active = True
        else:
            if active:
                out.append(cur)
            cur, active = 0, False
    if active:
        out.append(cur)
    return out


def all_numbers(s: str) -> set[float]:
    """Числа цифрами и словами, для проверки «число есть в цитате»."""
    return set(numbers(s)) | {float(x) for x in word_numbers(s)}


MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6, "июл": 7,
    "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}


def find_datetimes(s: str) -> list[tuple[str, str | None]]:
    """Возвращает пары (ДД.ММ.ГГГГ, ЧЧ:ММ|None), найденные в тексте."""
    text = s or ""
    res: list[tuple[int, str, str | None]] = []
    for m in re.finditer(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", text):
        res.append((m.start(), f"{int(m.group(1)):02d}.{int(m.group(2)):02d}.{m.group(3)}", None))
    for m in re.finditer(r"\b(\d{1,2})\s+([а-яё]+)\s+(\d{4})", text.lower()):
        mon = None
        for stem, num in MONTHS.items():
            if m.group(2).startswith(stem) and (stem != "ма" or m.group(2) in ("мая", "май")):
                mon = num
                break
        if mon:
            res.append((m.start(), f"{int(m.group(1)):02d}.{mon:02d}.{m.group(3)}", None))
    times = [(m.start(), f"{int(m.group(1)):02d}:{m.group(2)}")
             for m in re.finditer(r"(?<![\d.])(\d{1,2})[:.](\d{2})(?![\d.])", text)
             if int(m.group(1)) < 24 and int(m.group(2)) < 60]
    out = []
    for pos, d, _ in sorted(res):
        # ближайшее к дате время в пределах 60 символов
        near = [(abs(tpos - pos), t) for tpos, t in times if abs(tpos - pos) <= 60]
        out.append((d, min(near)[1] if near else None))
    return out
