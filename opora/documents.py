"""Загрузка документов закупки: кодировки, PDF/DOCX, пустые файлы, инъекции.

Канонический текст документа хранится без изменений. Для проверки цитат
используется «вид оценщика»: файл читается так же, как его прочитает check.py.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

from .textnorm import norm

OCR_CACHE: Path | None = None  # задаётся сервером: data/ocr_cache

SUPPORTED = {".txt", ".pdf", ".docx", ".md"}
SKIP_DIRS = {"node_modules", ".git", ".next", "__pycache__", ".venv", "venv", "dist", "build", ".openai"}

# Признаки инструкции, спрятанной в документе (косвенная промпт-инъекция).
INJECTION_PATTERNS = [
    r"игнорируй|проигнорируй|забудь (все|предыдущ)|не обращай внимани[ея] на (инструкц|правил)",
    r"ignore (all|any|the|previous)|disregard (all|previous)|system prompt|you are (now )?an? ",
    r"(языков(ая|ой) модел|нейросет|ИИ[- ]ассистент|искусственн\w+ интеллект|assistant|chatgpt|llm)\W.{0,60}"
    r"(долж|обязан|ответ|напиш|укаж|сообщ)",
    r"(при ответе|отвечая) на вопрос.{0,40}(указ|пиш|ответ|сообщ)",
    r"системн\w+ (промпт|инструкц|сообщени)",
    r"\b(ответь|отвечай|напиши|сообщи)\b[^.]{0,40}(что|только|всегда)",
]
_INJ = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def read_like_checker(raw: bytes) -> str:
    """Ровно как check.read_file: сначала UTF-8, потом CP1251."""
    for enc in ("utf-8", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", "replace")


def decode_text(raw: bytes) -> tuple[str, str]:
    """Определяет кодировку TXT: BOM UTF-8/UTF-16, UTF-8, CP1251."""
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", "replace"), "utf-8-bom"
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", "replace").lstrip("﻿"), "utf-16"
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    # UTF-16 без BOM: много нулевых байтов
    if raw[:2000].count(b"\x00") > len(raw[:2000]) // 4:
        try:
            return raw.decode("utf-16-le"), "utf-16-le"
        except UnicodeDecodeError:
            pass
    return raw.decode("cp1251", "replace"), "cp1251"


def _ocr_engine():
    """OCR сканов: pytesseract + Tesseract (rus+eng). Необязательно: без него страница — unreadable."""
    try:
        import pytesseract
        from PIL import Image  # noqa: F401
    except ImportError:
        return None
    cmd = os.environ.get("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return None
    return pytesseract


def _ocr_page(page, engine) -> str:
    from PIL import Image
    texts = []
    for img in page.images:
        try:
            im = Image.open(io.BytesIO(img.data))
            if im.width < 300 or im.height < 300:
                continue  # логотипы и печати не распознаём
            im = im.convert("L")
            texts.append(engine.image_to_string(im, lang=os.environ.get("OCR_LANG", "rus+eng")))
        except Exception:
            continue
    return "\n".join(t.strip() for t in texts if t.strip())


def _pdf_text(raw: bytes) -> tuple[str, list[dict], list[str]]:
    try:
        from pypdf import PdfReader  # необязательная зависимость
    except ImportError:
        return "", [], ["Для PDF установите pypdf: pip install pypdf"]
    sha = hashlib.sha256(raw).hexdigest()
    cache = OCR_CACHE / f"{sha}.json" if OCR_CACHE else None
    cached = json.loads(cache.read_text(encoding="utf-8")) if cache and cache.exists() else {}
    engine = None
    warnings, pages, parts, pos, ocr_pages = [], [], [], 0, []
    reader = PdfReader(io.BytesIO(raw))
    for i, page in enumerate(reader.pages, 1):
        try:
            t = page.extract_text() or ""
        except Exception as e:  # повреждённая страница
            t = ""
            warnings.append(f"стр. {i}: ошибка извлечения ({type(e).__name__})")
        # \x00 — вырезанные в оригинале поля (номер, даты): удаляем, сведений там нет
        t = re.sub(r"[ \t]+\n", "\n", t.replace("\x00", "")).strip()
        method = "native"
        if len(t) < 30 and page.images:
            if str(i) in cached:
                t, method = cached[str(i)], "ocr"
            else:
                engine = engine or _ocr_engine()
                if engine is not None:
                    o = _ocr_page(page, engine)
                    if len(o) >= 30:
                        t, method = o, "ocr"
                        cached[str(i)] = o
        if method == "ocr":
            ocr_pages.append(i)
        state = "readable" if len(t) >= 30 else ("blank" if not page.images else "unreadable")
        if state == "unreadable":
            warnings.append(f"стр. {i}: скан без текстового слоя, OCR недоступен")
        pages.append({"page": i, "start": pos, "end": pos + len(t), "state": state, "method": method})
        parts.append(t)
        pos += len(t) + 2
    if ocr_pages:
        warnings.append(f"распознано OCR (Tesseract) страниц: {len(ocr_pages)} — возможны ошибки распознавания")
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
    return "\n\n".join(parts), pages, warnings


def _docx_text(raw: bytes) -> str:
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    body = root.find("w:body", ns)
    out: list[str] = []

    def para_text(p) -> str:
        return "".join(t.text or "" for t in p.iter(f"{{{ns['w']}}}t"))

    for el in list(body) if body is not None else []:
        tag = el.tag.split("}")[1]
        if tag == "p":
            out.append(para_text(el))
        elif tag == "tbl":
            for row in el.iter(f"{{{ns['w']}}}tr"):
                cells = [" ".join(para_text(p) for p in c.iter(f"{{{ns['w']}}}p")).strip()
                         for c in row.iter(f"{{{ns['w']}}}tc")]
                out.append(" | ".join(cells))
    return "\n".join(out)


@dataclass
class Document:
    package: str
    name: str
    text: str
    kind: str = "txt"
    encoding: str = "utf-8"
    state: str = "readable"  # readable | blank | partial | unreadable | failed
    sha256: str = ""
    warnings: list[str] = field(default_factory=list)
    pages: list[dict] = field(default_factory=list)
    suspicious: list[dict] = field(default_factory=list)
    checker_norm: str = ""  # как этот файл увидит check.py
    lines: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self):
        pos = 0
        self.lines = []
        for ln in self.text.split("\n"):
            self.lines.append((pos, pos + len(ln)))
            pos += len(ln) + 1

    def line_of(self, offset: int) -> int:
        lo, hi = 0, len(self.lines) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.lines[mid][0] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def line_text(self, i: int) -> str:
        s, e = self.lines[i]
        return self.text[s:e]

    def page_of(self, offset: int) -> int | None:
        for p in self.pages:
            if p["start"] <= offset <= p["end"]:
                return p["page"]
        return None

    def summary(self) -> dict:
        return {
            "package": self.package, "name": self.name, "kind": self.kind, "encoding": self.encoding,
            "state": self.state, "chars": len(self.text), "lines": len(self.lines), "sha256": self.sha256,
            "warnings": self.warnings, "suspicious": self.suspicious,
            "pages": len(self.pages) or None,
        }


def detect_injections(text: str) -> list[dict]:
    """Находки компонента защиты (opora/guard.py): предложение/абзац, вид, серьёзность, границы в тексте.
    Старые правила INJECTION_PATTERNS дополняют guard (medium), если guard их не нашёл."""
    from .guard import scan_text, _sentence
    found = [f.to_dict() for f in scan_text(text)]
    for i, line in enumerate(text.split("\n")):
        for rx in _INJ:
            m = rx.search(line)
            if m:
                ls = sum(len(x) + 1 for x in text.split("\n")[:i])
                a, b = _sentence(text, ls + m.start(), ls + m.end())
                if not any(f["start"] < b and a < f["end"] for f in found):
                    found.append({"kind": "legacy_pattern", "severity": "medium", "line": i, "start": a, "end": b,
                                  "match": m.group(0)[:80], "text": text[a:b][:300], "verdict": ""})
                break
    return sorted(found, key=lambda f: f["start"])


def load_document(package: str, name: str, raw: bytes) -> tuple[Document, bytes]:
    """Возвращает документ и байты «текстовой проекции» для оценщика."""
    ext = Path(name).suffix.lower()
    sha = hashlib.sha256(raw).hexdigest()
    warnings: list[str] = []
    pages: list[dict] = []
    kind, enc, state = ext.lstrip("."), "", "readable"
    try:
        if ext in (".txt", ".md"):
            text, enc = decode_text(raw)
            projection = raw  # оценщик читает исходные байты
            if enc.startswith("utf-16"):
                warnings.append("Файл в UTF-16: официальный check.py его не прочитает, "
                                "цитаты из него в answers.csv не засчитаются")
            elif enc == "cp1251":
                warnings.append("Кодировка CP1251 (распознана автоматически)")
        elif ext == ".pdf":
            text, pages, w = _pdf_text(raw)
            warnings += w
            enc = "pdf-text"
            projection = text.encode("utf-8")
            if any(p["state"] == "unreadable" for p in pages):
                state = "partial" if text.strip() else "unreadable"
        elif ext == ".docx":
            text = _docx_text(raw)
            enc = "docx-xml"
            projection = text.encode("utf-8")
        else:
            raise ValueError("неподдерживаемый формат")
    except Exception as e:  # повреждённый файл
        text, projection, state = "", b"", "failed"
        warnings.append(f"Не удалось прочитать: {type(e).__name__}: {e}"[:300])
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    if state == "readable" and not text.strip():
        state = "blank"
        warnings.append("Пустой файл: прочитан полностью, сведений нет")
    doc = Document(package=package, name=name, text=text, kind=kind, encoding=enc, state=state,
                   sha256=sha, warnings=warnings, pages=pages)
    doc.suspicious = detect_injections(text)
    if doc.suspicious:
        doc.warnings.append(f"Подозрительные инструкции в тексте: {len(doc.suspicious)} — "
                            "переданы модели как данные с пометкой")
    doc.checker_norm = norm(read_like_checker(projection)) if projection else ""
    return doc, projection


def is_supported(rel_path: str) -> bool:
    parts = Path(rel_path.replace("\\", "/")).parts
    if any(p in SKIP_DIRS or p.startswith(".") for p in parts[:-1]):
        return False
    name = parts[-1] if parts else ""
    if name.lower() in ("readme.md", "questions.csv") or name.startswith("."):
        return False
    return Path(name).suffix.lower() in SUPPORTED and Path(name).suffix.lower() != ".md"


def package_of(rel_path: str) -> str:
    """Закупка = папка, в которой лежит файл (P01, MP01_FILO_lot634, …)."""
    parts = Path(rel_path.replace("\\", "/")).parts
    return parts[-2] if len(parts) >= 2 else "PACKAGE"
