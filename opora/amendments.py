"""Разбор изменений извещения кодом, без LLM.

Шаблон из документов: Пункт «начало старого текста…» (файл) изложить в
следующей редакции: «новый текст». Поддержаны также «Пункт 2.4 … изложить»
и «слова «X» заменить словами «Y»». Неразобранные строки с признаками
изменения помечаются unresolved и показываются модели как есть.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from .documents import Document
from .textnorm import norm

RX_QUOTED = re.compile(
    r"(?:Пункт|Подпункт|Абзац|Положение|Строку|Строка)\s+«(?P<old>.+?)(?:…|\.\.\.)?»\s*"
    r"(?:\((?P<file>[^)]+)\))?\s*[,]?\s*(?:изложить|читать)\s+в\s+(?:следующей|новой)\s+редакции\s*:?\s*"
    r"«(?P<new>.*)»\s*[.;]?\s*$", re.IGNORECASE)
RX_NUMBERED = re.compile(
    r"(?:Пункт|Подпункт)\s+(?P<num>\d+(?:\.\d+)*\.?)\s*(?:\((?P<file>[^)]+)\))?[^«]{0,80}?"
    r"(?:изложить|читать)\s+в\s+(?:следующей|новой)\s+редакции\s*:?\s*«(?P<new>.*)»\s*[.;]?\s*$",
    re.IGNORECASE)
RX_WORDS = re.compile(
    r"(?:\((?P<file>[^)]+)\)\s*)?слова\s+«(?P<old>[^»]+)»\s+заменить\s+словами\s+«(?P<new>[^»]+)»",
    re.IGNORECASE)
RX_HINT = re.compile(r"изложить в (следующей|новой) редакции|заменить словами|читать в (следующей|новой) редакции",
                     re.IGNORECASE)


@dataclass
class Amendment:
    source_file: str
    source_line: int
    source_text: str
    locator: str
    new_text: str
    target_file: str | None = None
    target_line: int | None = None
    old_text: str | None = None
    kind: str = "replace_paragraph"  # replace_paragraph | replace_words
    state: str = "resolved"  # resolved | unresolved

    def to_dict(self) -> dict:
        return asdict(self)


def _match_file(name: str | None, docs: list[Document]) -> list[Document]:
    if not name:
        return docs
    n = name.strip().lower()
    exact = [d for d in docs if d.name.lower() == n]
    if exact:
        return exact
    stem = re.sub(r"\.\w+$", "", n)
    return [d for d in docs if stem and stem in d.name.lower()] or docs


def _find_line(prefix: str, docs: list[Document]) -> tuple[Document, int] | None:
    p = norm(prefix).rstrip(" .…")
    if len(p) < 6:
        return None
    hits = []
    for d in docs:
        for i in range(len(d.lines)):
            if p in norm(d.line_text(i)):
                hits.append((d, i))
    return hits[0] if len(hits) == 1 else (hits[0] if hits and len({(h[0].name) for h in hits}) == 1 else None)


def _find_numbered(num: str, docs: list[Document]) -> tuple[Document, int] | None:
    n = num.rstrip(".")
    rx = re.compile(rf"^\s*{re.escape(n)}\.?\s")
    hits = [(d, i) for d in docs for i in range(len(d.lines)) if rx.match(d.line_text(i))]
    return hits[0] if len(hits) == 1 else None


def parse_amendments(docs: list[Document]) -> list[Amendment]:
    out: list[Amendment] = []
    for src in docs:
        others = [d for d in docs if d is not src]
        for i in range(len(src.lines)):
            line = src.line_text(i).strip()
            if not line or not RX_HINT.search(line):
                continue
            body = re.sub(r"^\s*\d+[.)]\s*", "", line)
            m = RX_QUOTED.search(body)
            if m:
                cands = _match_file(m.group("file"), others)
                hit = _find_line(m.group("old"), cands) or _find_line(m.group("old"), others)
                a = Amendment(src.name, i, line, m.group("old"), m.group("new").strip())
                if hit:
                    a.target_file, a.target_line = hit[0].name, hit[1]
                    a.old_text = hit[0].line_text(hit[1]).strip()
                else:
                    a.state = "unresolved"
                out.append(a)
                continue
            m = RX_NUMBERED.search(body)
            if m:
                cands = _match_file(m.group("file"), others)
                hit = _find_numbered(m.group("num"), cands)
                a = Amendment(src.name, i, line, "пункт " + m.group("num"), m.group("new").strip())
                if hit:
                    a.target_file, a.target_line = hit[0].name, hit[1]
                    a.old_text = hit[0].line_text(hit[1]).strip()
                else:
                    a.state = "unresolved"
                out.append(a)
                continue
            m = RX_WORDS.search(body)
            if m:
                cands = _match_file(m.group("file"), others)
                hit = _find_line(m.group("old"), cands)
                a = Amendment(src.name, i, line, m.group("old"), m.group("new").strip(), kind="replace_words")
                if hit:
                    a.target_file, a.target_line = hit[0].name, hit[1]
                    a.old_text = hit[0].line_text(hit[1]).strip()
                else:
                    a.state = "unresolved"
                out.append(a)
                continue
            out.append(Amendment(src.name, i, line, "", "", state="unresolved"))
    return out


def superseded_index(amendments: list[Amendment]) -> dict[tuple[str, int], Amendment]:
    """(файл, строка) → изменение, отменившее эту редакцию."""
    return {(a.target_file, a.target_line): a for a in amendments
            if a.state == "resolved" and a.target_file is not None and a.kind == "replace_paragraph"}


def amendment_sources(amendments: list[Amendment]) -> set[tuple[str, int]]:
    return {(a.source_file, a.source_line) for a in amendments}
