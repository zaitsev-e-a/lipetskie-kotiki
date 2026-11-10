"""Реестр источников закупки внутри снимка: блоки, чанки, окна и строгое разрешение цитат.

Правила:
- диапазон [start, end) в Unicode code points канонического текста; canonical_text[start:end] == quote;
- source_id выдаёт сервер; модель возвращает source_id и цитату, глобальные границы считает код;
- нормализация при поиске — как у оценщика (пробелы, регистр, кавычки, тире, ё/е), срез — из оригинала;
- никаких совпадений «по 80 % слов»: не нашли дословно — это ошибка источника, повод для исправления;
- несколько совпадений внутри блока — неоднозначность, первое совпадение не выбирается.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from .documents import Document
from .quotes import norm_with_map
from .schemas import Block, Chunk, Window, EvidenceRef
from .textnorm import norm

# Консервативная оценка: в русском тексте у BPE-токенайзеров обычно 3–4 символа на токен.
# Берём 2.8, чтобы скорее переоценить размер и не переполнить окно модели.
CHARS_PER_TOKEN = 2.8


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


class SourceError(Exception):
    """Ошибка источника: kind ∈ unknown_source | not_allowed | too_short | not_found | ambiguous."""

    def __init__(self, kind: str, source_id: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.source_id = source_id

    def to_dict(self) -> dict:
        return {"kind": self.kind, "source_id": self.source_id, "message": str(self)}


_SENT = re.compile(r"(?<=[.;:!?])\s+(?=[«\"(]?[А-ЯЁA-Z0-9])")
_HEADING = re.compile(r"^\s*((\d+(\.\d+)*\.?)\s+)?([А-ЯЁA-Z][А-ЯЁA-Z0-9 ,.«»\"()\-–]{3,})$")
_MARKS = re.compile(r"\{(⛔|⚠)[^}]*\}\s*")
_SID_PREFIX = re.compile(r"^\s*\[?S\d{3,5}\]?\s*[:\-–]?\s*")


def clean_model_quote(q: str) -> str:
    """Убирает то, что модель могла скопировать из разметки контекста: [S0012], пометки, внешние кавычки."""
    q = _SID_PREFIX.sub("", q or "")
    q = _MARKS.sub("", q)
    q = q.strip()
    if len(q) >= 2 and q[0] in "«\"'“" and q[-1] in "»\"'”":
        q = q[1:-1]
    return q.strip()


def _split_long(text: str, base: int, max_chars: int) -> list[tuple[int, int]]:
    """Режет длинный абзац по предложениям, а слишком длинное предложение — по пробелам."""
    spans, pos = [], 0
    for m in _SENT.finditer(text):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(text)))
    out: list[tuple[int, int]] = []
    cur_s, cur_e = None, None
    for s, e in spans:
        if e <= s:
            continue
        if cur_s is None:
            cur_s, cur_e = s, e
        elif e - cur_s <= max_chars:
            cur_e = e
        else:
            out.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    if cur_s is not None:
        out.append((cur_s, cur_e))
    final: list[tuple[int, int]] = []
    for s, e in out:
        while e - s > max_chars:
            cut = text.rfind(" ", s, s + max_chars)
            cut = cut if cut > s + max_chars // 2 else s + max_chars
            final.append((s, cut))
            s = cut + 1 if text[cut:cut + 1] == " " else cut
        if e > s:
            final.append((s, e))
    return [(base + s, base + e) for s, e in final]


@dataclass
class SourceConfig:
    block_max_chars: int = 1500
    chunk_tokens: int = 800
    chunk_overlap_tokens: int = 120
    window_tokens: int = 6000


class PackageSources:
    """Источники одной закупки в одном снимке. Текст документов после создания не меняется."""

    def __init__(self, snapshot_id: str, package_id: str, docs: list[Document], cfg: SourceConfig | None = None):
        self.snapshot_id = snapshot_id
        self.package_id = package_id
        self.cfg = cfg or SourceConfig()
        self.docs: list[Document] = sorted(docs, key=lambda d: d.name)
        self.doc_by_id: dict[str, Document] = {}
        self.doc_id_by_name: dict[str, str] = {}
        self.text_sha: dict[str, str] = {}
        self.blocks: dict[str, Block] = {}
        self.order: list[str] = []
        self.chunks: list[Chunk] = []
        self.windows: list[Window] = []
        # Отменённые поправками диапазоны: document_id → [(start, end, info)]
        self.superseded: dict[str, list[tuple[int, int, dict]]] = {}
        self._build()

    # ------------------------------------------------------------------ построение
    def _build(self) -> None:
        n = 0
        for d in self.docs:
            sha = hashlib.sha256(d.text.encode("utf-8")).hexdigest()
            did = "D" + hashlib.sha256(f"{self.package_id}/{d.name}:{sha}".encode()).hexdigest()[:10]
            self.doc_by_id[did] = d
            self.doc_id_by_name[d.name] = did
            self.text_sha[did] = sha
            section = ""
            for li, (ls, le) in enumerate(d.lines):
                line = d.text[ls:le]
                if not line.strip():
                    continue
                # пробелы по краям не входят в блок: срез начинается с первого значимого символа
                lead = len(line) - len(line.lstrip())
                trail = len(line.rstrip())
                s0, e0 = ls + lead, ls + trail
                if len(line.strip()) < 140 and _HEADING.match(line.strip()):
                    section = line.strip()[:120]
                for s, e in _split_long(d.text[s0:e0], s0, self.cfg.block_max_chars):
                    n += 1
                    sid = f"S{n:04d}"
                    self.blocks[sid] = Block(source_id=sid, document_id=did, file=d.name, start=s, end=e,
                                             line=li, tokens=estimate_tokens(d.text[s:e]),
                                             page=d.page_of(s), section=section)
                    self.order.append(sid)
        self._build_chunks()
        self._build_windows()

    def _doc_blocks(self, did: str) -> list[Block]:
        return [self.blocks[s] for s in self.order if self.blocks[s].document_id == did]

    def _build_chunks(self) -> None:
        cfg = self.cfg
        for did in self.doc_by_id:
            bl = self._doc_blocks(did)
            i = 0
            while i < len(bl):
                j, tok = i, 0
                while j < len(bl) and (tok == 0 or tok + bl[j].tokens <= cfg.chunk_tokens):
                    tok += bl[j].tokens
                    j += 1
                part = bl[i:j]
                cid = f"C{len(self.chunks) + 1:04d}"
                self.chunks.append(Chunk(chunk_id=cid, document_id=did, file=part[0].file, start=part[0].start,
                                         end=part[-1].end, block_ids=[b.source_id for b in part], tokens=tok,
                                         section=part[0].section, page=part[0].page))
                if j >= len(bl):
                    break
                # перекрытие: следующий чанк начинается с последних блоков текущего (не больше overlap)
                back, ov = j, 0
                while back - 1 > i and ov + bl[back - 1].tokens <= cfg.chunk_overlap_tokens:
                    back -= 1
                    ov += bl[back].tokens
                i = back if back > i else j

    def _build_windows(self) -> None:
        cap = self.cfg.window_tokens
        pending: list[Block] = []
        pend_tok = 0

        def flush():
            nonlocal pending, pend_tok
            if pending:
                self.windows.append(Window(window_id=f"W{len(self.windows) + 1:03d}",
                                           block_ids=[b.source_id for b in pending],
                                           files=sorted({b.file for b in pending}),
                                           tokens=pend_tok))
            pending, pend_tok = [], 0

        for did in self.doc_by_id:
            bl = self._doc_blocks(did)
            doc_tok = sum(b.tokens for b in bl)
            if doc_tok <= cap // 2 and pend_tok + doc_tok <= cap:
                pending += bl  # маленькие документы пакуются в одно окно, source_id остаются раздельными
                pend_tok += doc_tok
                continue
            flush()
            for b in bl:
                if pend_tok and pend_tok + b.tokens > cap:
                    flush()
                pending.append(b)
                pend_tok += b.tokens
            flush()
        flush()

    # ------------------------------------------------------------------ доступ
    @property
    def total_tokens(self) -> int:
        return sum(b.tokens for b in self.blocks.values())

    # ------------------------------------------------------------------ защита от инъекций
    def guard_findings(self, did: str, start: int, end: int, severity: str | None = None) -> list[dict]:
        """Действующие находки guard, пересекающие диапазон (снятые LLM-классификатором — нет)."""
        d = self.doc_by_id[did]
        out = []
        for f in getattr(d, "suspicious", []) or []:
            if f.get("verdict") == "document_text" or "start" not in f:
                continue
            if severity and f.get("severity") != severity:
                continue
            if f["start"] < end and start < f["end"]:
                out.append(f)
        return out

    @property
    def guard_active(self) -> bool:
        return any(f.get("verdict") != "document_text" for d in self.docs for f in getattr(d, "suspicious", []) or [])

    def guard_nonce(self) -> str:
        from .guard import nonce_of
        return nonce_of(self.snapshot_id, self.package_id)

    def quarantined_texts(self) -> list[str]:
        return [d.text[f["start"]:f["end"]] for d in self.docs for f in getattr(d, "suspicious", []) or []
                if f.get("severity") == "high" and f.get("verdict") != "document_text" and "start" in f]

    def is_suspicious(self, sid: str) -> bool:
        """Блок пересекает фрагмент, который защита отметила как возможную инструкцию для ИИ."""
        b = self.blocks.get(sid)
        if b is None:
            return False
        return bool(self.guard_findings(b.document_id, b.start, b.end))

    def guarded_block_text(self, sid: str) -> str:
        """Текст блока для модели: фрагменты high в карантине (заменены пометкой), остальное — как в документе."""
        from .guard import quarantine_note
        b = self.blocks[sid]
        d = self.doc_by_id[b.document_id]
        hi = sorted(self.guard_findings(b.document_id, b.start, b.end, "high"), key=lambda f: f["start"])
        if not hi:
            return d.text[b.start:b.end]
        out, pos = [], b.start
        for f in hi:
            a, e = max(f["start"], b.start), min(f["end"], b.end)
            if a > pos:
                out.append(d.text[pos:a])
            if a >= pos:
                out.append(quarantine_note(f))
            pos = max(pos, e)
        out.append(d.text[pos:b.end])
        return "".join(out)

    def block_text(self, sid: str) -> str:
        b = self.blocks[sid]
        return self.doc_by_id[b.document_id].text[b.start:b.end]

    def window(self, wid: str) -> Window:
        for w in self.windows:
            if w.window_id == wid:
                return w
        raise KeyError(wid)

    def windows_of_blocks(self, sids) -> list[str]:
        want = set(sids)
        return [w.window_id for w in self.windows if want & set(w.block_ids)]

    def doc(self, name: str) -> Document | None:
        did = self.doc_id_by_name.get(name)
        return self.doc_by_id.get(did) if did else None

    def blocks_of_line(self, file: str, line: int) -> list[str]:
        did = self.doc_id_by_name.get(file)
        return [s for s in self.order if self.blocks[s].document_id == did and self.blocks[s].line == line]

    # ------------------------------------------------------------------ строгое разрешение цитаты
    def resolve(self, source_id: str, quote: str, allowed: set[str] | None = None) -> EvidenceRef:
        sid = (source_id or "").strip().strip("[]")
        b = self.blocks.get(sid)
        if b is None:
            raise SourceError("unknown_source", sid, f"источник {sid!r} не зарегистрирован в этой закупке и снимке")
        if allowed is not None and sid not in allowed:
            raise SourceError("not_allowed", sid, f"источник {sid} не передавался на этом шаге")
        q = clean_model_quote(quote)
        nq = norm(q)
        if len(nq) < 8:
            raise SourceError("too_short", sid, "цитата короче 8 символов")
        doc = self.doc_by_id[b.document_id]
        # Цитата может продолжиться в следующий блок того же документа (длинный абзац, разрезанный кодом),
        # но начинаться обязана внутри указанного блока.
        idx = self.order.index(sid)
        region_end = b.end
        if idx + 1 < len(self.order):
            nb = self.blocks[self.order[idx + 1]]
            if nb.document_id == b.document_id:
                region_end = nb.end
        region = doc.text[b.start:region_end]
        nt, mp = norm_with_map(region)
        hits, pos = [], nt.find(nq)
        while pos >= 0:
            start = b.start + mp[pos]
            if start < b.end:
                hits.append((start, b.start + mp[pos + len(nq) - 1] + 1))
            pos = nt.find(nq, pos + 1)
        if not hits:
            raise SourceError("not_found", sid, f"цитаты нет дословно в источнике {sid}")
        if len({h[0] for h in hits}) > 1:
            raise SourceError("ambiguous", sid, f"цитата встречается в {sid} {len(hits)} раза — нужна более длинная")
        return self.evidence(b.document_id, hits[0][0], hits[0][1], sid)

    def evidence(self, did: str, start: int, end: int, sid: str | None = None) -> EvidenceRef:
        doc = self.doc_by_id[did]
        q = doc.text[start:end]
        li = doc.line_of(start)
        ev = EvidenceRef(source_id=sid or self._sid_at(did, start), snapshot_id=self.snapshot_id,
                         package_id=self.package_id, document_id=did, file=doc.name,
                         text_sha256=self.text_sha[did], start=start, end=end, quote=q, page=doc.page_of(start),
                         line=li, checker_ok=len(norm(q)) >= 8 and norm(q) in doc.checker_norm,
                         context=doc.line_text(li).strip()[:1500])
        ev.superseded_by = self.superseded_info(did, start, end)
        return ev

    def _sid_at(self, did: str, pos: int) -> str:
        for s in self.order:
            b = self.blocks[s]
            if b.document_id == did and b.start <= pos < b.end:
                return s
        return ""

    def superseded_info(self, did: str, start: int, end: int) -> dict | None:
        for s, e, info in self.superseded.get(did, []):
            if start < e and end > s:
                return info
        return None

    def verify_ref(self, ev: dict) -> str | None:
        """Повторная проверка EvidenceRef (final_validate): снимок, закупка, хеш, диапазон, срез."""
        if ev.get("snapshot_id") != self.snapshot_id or ev.get("package_id") != self.package_id:
            return "источник из другого снимка или закупки"
        did = ev.get("document_id")
        doc = self.doc_by_id.get(did)
        if doc is None:
            return "документ не зарегистрирован"
        if ev.get("text_sha256") != self.text_sha[did]:
            return "хеш текста документа не совпадает"
        s, e = ev.get("start"), ev.get("end")
        if not (isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(doc.text)):
            return "диапазон вне текста"
        if doc.text[s:e] != ev.get("quote"):
            return "срез текста не совпадает с цитатой"
        return None

    # ------------------------------------------------------------------ отрисовка для модели
    def render(self, block_ids, marks: dict[str, list[str]] | None = None) -> str:
        """Блоки с источниками [Sxxxx], сгруппированные по файлам в порядке документа."""
        want = set(block_ids)
        out, cur = [], None
        # spotlighting: при находках защиты границы данных получают метку снимка — подделать их нельзя
        tag = f' id="{self.guard_nonce()}"' if self.guard_active else ""
        for sid in self.order:
            if sid not in want:
                continue
            b = self.blocks[sid]
            if b.file != cur:
                if cur is not None:
                    out.append(f"<<<КОНЕЦ ДОКУМЕНТА{tag}>>>")
                doc = self.doc_by_id[b.document_id]
                out.append(f'<<<ДОКУМЕНТ{tag} file="{b.file}" состояние="{doc.state}">>>')
                cur = b.file
            m = (marks or {}).get(sid) or []
            prefix = " ".join(f"{{{x}}}" for x in m)
            out.append(f"[{sid}] {prefix + ' ' if prefix else ''}{self.guarded_block_text(sid)}")
        if cur is not None:
            out.append(f"<<<КОНЕЦ ДОКУМЕНТА{tag}>>>")
        return "\n".join(out)

    def manifest(self) -> list[dict]:
        return [{"document_id": did, "file": d.name, "text_sha256": self.text_sha[did], "sha256": d.sha256,
                 "encoding": d.encoding, "state": d.state, "kind": d.kind, "chars": len(d.text),
                 "pages": len(d.pages) or None,
                 "blocks": sum(1 for b in self.blocks.values() if b.document_id == did)}
                for did, d in self.doc_by_id.items()]
