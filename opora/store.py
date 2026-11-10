"""Снимки закупок, подготовка (паспорт + индекс Qdrant), прогоны, диалоги. Всё хранится на диске в data/.

Снимок неизменяем: после запуска подготовки файлы не добавляются (409). snapshot_id — хеш манифеста файлов,
версий извлечения/нарезки/схем/промптов и моделей. Подготовка возобновляется после перезапуска: готовые
окна паспорта не читаются повторно. Прогоны и диалоги пишутся на диск по мере работы и восстанавливаются;
прогон, прерванный перезапуском, получает состояние interrupted и продолжается с неотвеченных вопросов.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import evaluate as E
from . import prompts as P
from .config import settings
from .documents import load_document, package_of, is_supported
from .knowledge import Knowledge, build_knowledge, restore_superseded
from .llm import Tracker, LLMError, client, model_info_of
from .pipeline import answer_improved, answer_baseline
from .retrieval import Retriever, QdrantError, QdrantUnavailable, get_qdrant, get_local_index, LocalIndex
from .schemas import EXTRACTION_VERSION, CHUNKING_VERSION, KNOWLEDGE_SCHEMA, INDEX_SCHEMA
from .sources import PackageSources, SourceConfig
from .textnorm import one_line

SAFE = re.compile(r"[^0-9A-Za-zА-Яа-яЁё._ ()\-]+")
READ_MODES = ("auto", "full_context", "retrieval", "exhaustive")


class Conflict409(ValueError):
    """Операция противоречит состоянию (снимок уже запечатан, прогон уже идёт)."""


def _safe(name: str) -> str:
    n = SAFE.sub("_", name).strip(" .")
    return n[:150] or "file"


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)  # атомарная запись


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def versions() -> dict:
    return {"extraction": EXTRACTION_VERSION, "chunking": CHUNKING_VERSION, "knowledge_schema": KNOWLEDGE_SCHEMA,
            "index_schema": INDEX_SCHEMA, "prompts": P.PROMPTS_VERSION}


def snapshot_id_of(manifest: str) -> tuple[str, dict]:
    key = {"manifest": manifest, **versions(), "model": settings.model, "passport_models": settings.passport_models(),
           "embedding_model": settings.embedding_model,
           "block_max_chars": settings.block_max_chars, "chunk_tokens": settings.chunk_tokens,
           "chunk_overlap_tokens": settings.chunk_overlap_tokens, "window_tokens": settings.window_tokens}
    sid = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return sid, key


def source_config() -> SourceConfig:
    return SourceConfig(block_max_chars=settings.block_max_chars, chunk_tokens=settings.chunk_tokens,
                        chunk_overlap_tokens=settings.chunk_overlap_tokens, window_tokens=settings.window_tokens)


def run_options(raw: dict | None) -> dict:
    raw = raw or {}
    mode = raw.get("read_mode") or settings.read_mode
    if mode not in READ_MODES:
        raise ValueError(f"режим чтения {mode!r}: допустимы {', '.join(READ_MODES)}")
    return {"read_mode": mode,
            "use_passport": bool(raw.get("use_passport", settings.use_passport)),
            "strict_coverage": bool(raw.get("strict_coverage", settings.strict_coverage))}


# =================================================================== коллекция = неизменяемый снимок
class Collection:
    STAGES = (("read", "Чтение файлов и кодировок"), ("blocks", "Блоки, чанки и окна"),
              ("amend", "Изменения извещения"), ("passport", "Паспорт: факты по всем окнам"),
              ("conflicts", "Расхождения между документами"), ("index", "Индекс Qdrant (dense + BM25)"))

    def __init__(self, cid: str, root: Path):
        self.id = cid
        self.dir = root / cid
        self.packages: dict[str, PackageSources] = {}
        self.knowledge: dict[str, Knowledge] = {}
        self.retrievers: dict[str, Retriever] = {}
        self.meta: dict = {"id": cid, "created": time.time(), "state": "uploading", "stages": [],
                           "files": [], "packages": [], "errors": [], "warnings": [], "calls": {},
                           "snapshot_id": None, "index": {}}
        self.lock = threading.RLock()
        self.tracker = Tracker()

    # ---------------------------------------------------------- загрузка файлов (только до запечатывания)
    def add_file(self, rel_path: str, raw: bytes) -> dict:
        with self.lock:
            if self.meta.get("state") != "uploading":
                raise Conflict409("снимок уже запечатан: подготовка запущена, файлы не добавляются. "
                                  "Для другого содержимого создайте новый снимок")
        rel = rel_path.replace("\\", "/").lstrip("/")
        if ".." in rel.split("/") or "\x00" in rel:
            raise ValueError("недопустимый путь")
        if not is_supported(rel):
            return {"skipped": True, "path": rel}
        if len(raw) > 128 * 1024 * 1024:
            raise ValueError("файл больше 128 МиБ")
        pkg = _safe(package_of(rel))
        name = _safe(Path(rel).name)
        dst = self.dir / "files" / pkg / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.read_bytes() != raw:
            raise ValueError(f"в закупке {pkg} уже есть другой файл {name}")
        dst.write_bytes(raw)
        with self.lock:
            self.meta["files"].append({"path": rel, "package": pkg, "name": name, "bytes": len(raw)})
            self.meta.pop("manifest", None)
        return {"skipped": False, "package": pkg, "name": name}

    @staticmethod
    def manifest_of(files_dir: Path) -> tuple[str, int]:
        """SHA-256 от отсортированных строк «закупка/файл:sha256(байты)»: не зависит от порядка загрузки."""
        lines = []
        if files_dir.is_dir():
            for pdir in files_dir.iterdir():
                if not pdir.is_dir():
                    continue
                for f in pdir.iterdir():
                    if f.is_file():
                        lines.append(f"{pdir.name}/{f.name}:{hashlib.sha256(f.read_bytes()).hexdigest()}")
        lines.sort()
        if not lines:
            return "", 0
        return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest(), len(lines)

    def manifest(self) -> str:
        m, _ = self.manifest_of(self.dir / "files")
        with self.lock:
            self.meta["manifest"] = m
        return m

    def seal(self) -> str:
        """Запечатывает снимок: дальше файлы не меняются, snapshot_id фиксирован."""
        with self.lock:
            if self.meta.get("snapshot_id"):
                return self.meta["snapshot_id"]
            m = self.meta.get("manifest") or self.manifest()
            if not m:
                raise ValueError("в снимке нет поддерживаемых файлов")
            sid, key = snapshot_id_of(m)
            self.meta.update(snapshot_id=sid, snapshot_key=key, sealed=time.time())
            self.save()
            return sid

    def save(self) -> None:
        with self.lock:
            self.meta["packages"] = [self.package_summary(p) for p in self.packages]
            self.meta["calls"] = self.tracker.summary()
            _write_json(self.dir / "meta.json", self.meta)

    def stage(self, key: str, state: str, done: int | None = None, total: int | None = None,
              detail: str = "", items: dict | None = None) -> None:
        title = dict(self.STAGES).get(key, key)
        with self.lock:
            st = next((s for s in self.meta["stages"] if s["key"] == key), None)
            if st is None:
                st = {"key": key, "title": title}
                self.meta["stages"].append(st)
            st.update({"state": state, "done": done, "total": total, "detail": detail, "t": time.time()})
            if items is not None:
                st["items"] = items

    # ---------------------------------------------------------- источники (без LLM)
    def load_sources(self) -> None:
        sid = self.meta["snapshot_id"]
        files_dir, text_dir = self.dir / "files", self.dir / "text"
        pk_dirs = sorted(p for p in files_dir.iterdir() if p.is_dir()) if files_dir.exists() else []
        total = sum(1 for p in pk_dirs for f in p.iterdir() if f.is_file())
        done, per_doc = 0, {}
        self.stage("read", "running", 0, total)
        packages = {}
        for pdir in pk_dirs:
            docs = []
            for f in sorted(x for x in pdir.iterdir() if x.is_file()):
                doc, projection = load_document(pdir.name, f.name, f.read_bytes())
                docs.append(doc)
                # текстовая проекция для check.py: TXT — исходные байты, PDF/DOCX — извлечённый текст
                proj = text_dir / pdir.name / (f.name if f.suffix.lower() == ".txt" else f.name + ".txt")
                if not proj.exists():
                    proj.parent.mkdir(parents=True, exist_ok=True)
                    proj.write_bytes(projection)
                done += 1
                per_doc[f"{pdir.name}/{f.name}"] = doc.state
                self.stage("read", "running", done, total, f"{pdir.name}/{f.name}", items=dict(per_doc))
            packages[pdir.name] = PackageSources(sid, pdir.name, docs, source_config())
        self.packages = packages
        bad = [k for k, v in per_doc.items() if v not in ("readable", "blank")]
        self.stage("read", "done" if not bad else "warning", total, total,
                   f"нечитаемые: {', '.join(bad)}" if bad else "", items=per_doc)
        nb = sum(len(p.blocks) for p in packages.values())
        nw = sum(len(p.windows) for p in packages.values())
        nc = sum(len(p.chunks) for p in packages.values())
        self.stage("blocks", "done", nb, nb, f"блоков {nb}, чанков {nc}, окон {nw}")

    # ---------------------------------------------------------- подготовка (возобновляемая)
    def _kpath(self, pkg: str) -> Path:
        return self.dir / "knowledge" / f"{_safe(pkg)}.json"

    def ingest(self, llm=None) -> None:
        try:
            with self.lock:
                self.meta["state"] = "processing"
                self.meta["errors"], self.meta["warnings"] = [], []
            self.seal()
            if not self.packages:
                self.load_sources()
            self.save()
            pk = list(self.packages.values())
            llm = llm or (client() if settings.api_key else None)
            use_llm = llm is not None
            if not use_llm:
                for key in ("amend", "passport", "conflicts", "index"):
                    self.stage(key, "skipped", detail="нет OPENROUTER_API_KEY")
                self.meta["errors"].append("Не задан OPENROUTER_API_KEY — паспорт, индекс и ответы недоступны")
            else:
                self._build_passports(pk, llm)
                self._build_index(pk, llm)
            failed = [p.package_id for p in pk if self.knowledge.get(p.package_id)
                      and self.knowledge[p.package_id].coverage()["windows_failed"]]
            if failed:
                self.meta["warnings"].append("паспорт построен не по всем окнам: " + ", ".join(failed)
                                             + " — «нет в документах» для них будет запрещено")
            self.meta["state"] = "ready_with_warnings" if self.meta["errors"] or self.meta["warnings"] else "ready"
        except Exception as e:  # техническая ошибка видна в интерфейсе
            self.meta["state"] = "failed"
            self.meta["errors"].append(f"{type(e).__name__}: {e}")
        finally:
            self.meta["prepared"] = time.time()
            self.save()

    def _build_passports(self, pk: list[PackageSources], llm) -> None:
        total_w = sum(len(p.windows) for p in pk)
        prog = {"done": 0}
        per_pkg: dict[str, dict] = {}
        self.stage("amend", "running", 0, len(pk))
        self.stage("passport", "running", 0, total_w)
        save_lock = threading.Lock()

        def one(ps: PackageSources):
            partial = None
            saved = _read_json(self._kpath(ps.package_id))
            if saved and saved.get("snapshot_id") == ps.snapshot_id and saved.get("schema") == KNOWLEDGE_SCHEMA:
                partial = Knowledge.from_json(saved)
                if partial.provenance.get("complete"):
                    restore_superseded(ps, partial)
                    with self.lock:
                        self.knowledge[ps.package_id] = partial
                        prog["done"] += len(ps.windows)
                        per_pkg[ps.package_id] = partial.coverage()
                        self.stage("passport", "running", prog["done"], total_w, f"{ps.package_id}: из сохранённого",
                                   items=dict(per_pkg))
                    return
            base = sum(1 for w in (partial.windows.values() if partial else []) if w.get("state") == "done")
            last = [base]

            def checkpoint(k: Knowledge):
                with save_lock:
                    _write_json(self._kpath(ps.package_id), k.to_json())

            def progress(done, total, facts):
                with self.lock:
                    prog["done"] += done - last[0]
                    last[0] = done
                    per_pkg[ps.package_id] = {"windows_done": done, "windows_total": total, "facts": facts}
                    self.stage("passport", "running", prog["done"], total_w, f"{ps.package_id}: окно {done}/{total}",
                               items=dict(per_pkg))

            k = build_knowledge(ps, llm, self.tracker, workers=max(1, settings.question_concurrency),
                                progress=progress, partial=partial, checkpoint=checkpoint)
            k.provenance["complete"] = True
            checkpoint(k)
            with self.lock:
                self.knowledge[ps.package_id] = k
                per_pkg[ps.package_id] = k.coverage()
                am = sum(1 for kk in self.knowledge.values() for a in kk.amendments)
                self.stage("amend", "running", len(self.knowledge), len(pk), f"изменений: {am}")
            self.save()

        with ThreadPoolExecutor(max_workers=max(1, min(len(pk), settings.question_concurrency))) as ex:
            list(ex.map(one, pk))
        am = [a for k in self.knowledge.values() for a in k.amendments]
        unresolved = [a for a in am if a.get("state") != "resolved"]
        self.stage("amend", "done" if not unresolved else "warning", len(pk), len(pk),
                   f"применено {len(am) - len(unresolved)}, не разобрано {len(unresolved)}")
        self.stage("passport", "done", total_w, total_w, f"фактов: {sum(len(k.facts) for k in self.knowledge.values())}",
                   items={p: k.coverage() for p, k in self.knowledge.items()})
        conf = [c for k in self.knowledge.values() for c in k.conflicts]
        by_state: dict[str, int] = {}
        for c in conf:
            by_state[c["state"]] = by_state.get(c["state"], 0) + 1
        self.stage("conflicts", "done", len(conf), len(conf),
                   ", ".join(f"{k}: {v}" for k, v in sorted(by_state.items())) or "нет кандидатов")

    def _build_index(self, pk: list[PackageSources], llm) -> None:
        q = get_qdrant()
        diag = q.diagnostics() if hasattr(q, "diagnostics") else {"ok": True}
        if not diag.get("ok"):
            if not settings.local_index:
                self.stage("index", "unavailable", detail=diag.get("error", "Qdrant недоступен"))
                self.meta["warnings"].append("Qdrant недоступен — режим retrieval переключится на exhaustive "
                                             "(переключение пишется в журнал вопроса)")
                return
            # самовосстановление: тот же гибридный поиск (dense + BM25 + RRF) в памяти процесса
            q = get_local_index()
            self.meta["warnings"].append("Qdrant недоступен — поиск идёт по локальному индексу в памяти "
                                         "(dense + BM25 + RRF); запустите docker compose up -d qdrant для сервера")
        self.stage("index", "running", 0, len(pk))
        for i, ps in enumerate(pk, 1):
            rt = Retriever(q, llm)
            prev = (self.meta.get("index") or {}).get(ps.package_id) or {}
            try:
                if prev.get("snapshot_id") == ps.snapshot_id and prev.get("collection") and \
                        q.count(prev["collection"], {"must": [
                            {"key": "snapshot_id", "match": {"value": ps.snapshot_id}},
                            {"key": "package_id", "match": {"value": ps.package_id}}]}) >= prev.get("points", 1):
                    rt.attach(prev["collection"], prev["dim"])
                    info = prev
                else:
                    info = {**rt.index_package(ps, self.knowledge.get(ps.package_id), self.tracker),
                            "snapshot_id": ps.snapshot_id, "index_schema": INDEX_SCHEMA,
                            "backend": "local" if isinstance(q, LocalIndex) else "qdrant"}
                self.retrievers[ps.package_id] = rt
                with self.lock:
                    self.meta.setdefault("index", {})[ps.package_id] = info
            except (QdrantError, QdrantUnavailable, LLMError) as e:
                self.meta["warnings"].append(f"{ps.package_id}: индекс не построен ({e})"[:400])
            self.stage("index", "running", i, len(pk), ps.package_id)
            self.save()
        ok = len(self.retrievers)
        self.stage("index", "done" if ok == len(pk) else "warning", ok, len(pk),
                   f"bm25: {getattr(q, 'bm25_mode', None)}")

    # ---------------------------------------------------------- восстановление после перезапуска
    @classmethod
    def restore(cls, cid: str, root: Path) -> "Collection":
        c = cls(cid, root)
        c.meta = json.loads((c.dir / "meta.json").read_text(encoding="utf-8"))
        c.meta.setdefault("warnings", [])
        c.meta.setdefault("index", {})
        if not c.meta.get("snapshot_id"):
            return c  # ещё загружается
        stages = c.meta.get("stages", [])
        c.load_sources()
        c.meta["stages"] = stages
        for pkg, ps in c.packages.items():
            saved = _read_json(c._kpath(pkg))
            if saved and saved.get("snapshot_id") == ps.snapshot_id:
                k = Knowledge.from_json(saved)
                restore_superseded(ps, k)
                if k.provenance.get("complete"):
                    c.knowledge[pkg] = k
        for pkg, info in (c.meta.get("index") or {}).items():
            if pkg in c.packages and info.get("collection"):
                try:
                    if info.get("backend") == "local":
                        # локальный индекс живёт в памяти: пересобирается из кеша эмбеддингов (без затрат)
                        if settings.api_key:
                            rt = Retriever(get_local_index(), client())
                            rt.index_package(c.packages[pkg], c.knowledge.get(pkg), c.tracker)
                            c.retrievers[pkg] = rt
                        continue
                    rt = Retriever(get_qdrant(), None)
                    rt.attach(info["collection"], info["dim"])
                    c.retrievers[pkg] = rt
                except (QdrantError, QdrantUnavailable, LLMError):
                    pass
        if c.meta.get("state") == "processing":
            c.meta["state"] = "interrupted"  # «Продолжить подготовку» — готовые окна не повторяются
        return c

    def retriever(self, pkg: str, llm) -> Retriever | None:
        rt = self.retrievers.get(pkg)
        if rt is not None and rt.llm is None:
            rt.llm = llm
        return rt

    # ---------------------------------------------------------- представления для интерфейса
    def package_summary(self, pkg: str) -> dict:
        ps = self.packages[pkg]
        k = self.knowledge.get(pkg)
        docs = []
        for m in ps.manifest():
            d = ps.doc_by_id[m["document_id"]]
            docs.append({**m, "name": d.name, "summary": d.summary(),
                         "superseded": [{"start": s, "end": e, **info}
                                        for s, e, info in ps.superseded.get(m["document_id"], [])]})
        return {"name": pkg, "snapshot_id": ps.snapshot_id, "docs": docs, "tokens": ps.total_tokens,
                "blocks": len(ps.blocks), "chunks": len(ps.chunks), "windows": len(ps.windows),
                "passport": k.coverage() if k else None,
                "amendments": [{"state": a.get("state"), "kind": a.get("kind"), "origin": a.get("origin"),
                                "source_file": a.get("source_file"), "source_line": a.get("source_line"),
                                "target_file": a.get("target_file"), "target_line": a.get("target_line"),
                                "old_text": (a.get("old_text") or "")[:400], "new_text": (a.get("new_text") or "")[:400],
                                "source_text": (a.get("source_text") or "")[:400],
                                "chain": [a.get("chain_position"), a.get("chain_length")]}
                               for a in (k.amendments if k else [])],
                "conflicts": [{"id": c["conflict_id"], "state": c["state"], "attribute": c.get("attribute"),
                               "label": c.get("label"), "unit": c.get("unit"), "conditions": c.get("conditions"),
                               "reason": c.get("reason", ""), "sides": c.get("sides")}
                              for c in (k.conflicts if k else [])],
                "index": (self.meta.get("index") or {}).get(pkg)}

    def source(self, package: str, name: str) -> dict:
        ps = self.packages.get(package)
        if ps is None or name not in ps.doc_id_by_name:
            raise KeyError("документ не найден")
        did = ps.doc_id_by_name[name]
        d = ps.doc_by_id[did]
        return {"package": package, "name": name, "document_id": did, "text": d.text, "summary": d.summary(),
                "text_sha256": ps.text_sha[did], "snapshot_id": ps.snapshot_id,
                "superseded": [{"start": s, "end": e, **info} for s, e, info in ps.superseded.get(did, [])]}

    def knowledge_view(self, package: str) -> dict:
        k = self.knowledge.get(package)
        if k is None:
            saved = _read_json(self._kpath(package))
            if not saved:
                raise KeyError("паспорт закупки не построен")
            return {**saved, "complete": False}
        return {**k.to_json(), "complete": True}


# =================================================================== вопросы CSV
def parse_questions(raw: bytes, collection: Collection) -> list[dict]:
    text = None
    for enc in ("utf-8-sig", "cp1251"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("не удалось определить кодировку CSV")
    first = text.split("\n", 1)[0]
    delim = ";" if first.count(";") >= first.count(",") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delim)
    fields = [(f or "").strip().lower() for f in (reader.fieldnames or [])]
    if not {"id", "package", "question"} <= set(fields):
        raise ValueError("нужны колонки id;package;question")
    out, seen, errors = [], set(), []
    for n, r in enumerate(reader, start=2):
        r = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
        if not any(r.values()):
            continue
        qid, pkg, q = r.get("id", ""), r.get("package", ""), r.get("question", "")
        if not qid or not pkg or not q:
            errors.append(f"строка {n}: пустое поле")
        elif qid in seen:
            errors.append(f"строка {n}: повтор id {qid}")
        elif pkg not in collection.packages:
            errors.append(f"строка {n}: закупка {pkg} не загружена")
        else:
            seen.add(qid)
            out.append({"id": qid, "package": pkg, "question": q[:12000]})
    if errors:
        raise ValueError("; ".join(errors[:20]) + (f" … и ещё {len(errors) - 20}" if len(errors) > 20 else ""))
    if not out:
        raise ValueError("в CSV нет вопросов")
    return out


# =================================================================== прогоны
class Run:
    """Каждый ответ пишется в results/<variant>/<id>.json сразу после получения."""

    def __init__(self, rid: str, root: Path, collection: Collection, questions: list[dict], baseline: bool,
                 options: dict | None = None):
        self.id = rid
        self.dir = root / rid
        self.collection = collection
        self.questions = questions
        self.baseline = baseline
        self.options = run_options(options)
        self.final: dict[str, dict] = {}
        self.base: dict[str, dict] = {}
        self.running: dict[str, str] = {}
        self.state = "queued"
        self.started = self.finished = None
        self.evaluation: dict | None = None
        self.lock = threading.RLock()
        self.cancelled = False
        self.llm_factory = client

    # ---------------------------------------------------------- состояние на диске
    def _save_state(self) -> None:
        _write_json(self.dir / "state.json", {
            "id": self.id, "state": self.state, "collection": self.collection.id,
            "snapshot_id": self.collection.meta.get("snapshot_id"), "baseline": self.baseline,
            "options": self.options, "started": self.started, "finished": self.finished})

    def persist_new(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.dir / "questions.json", self.questions)
        self._save_state()

    @classmethod
    def restore(cls, rid: str, root: Path, registry: "Registry") -> "Run":
        d = root / rid
        st = _read_json(d / "state.json")
        qs = _read_json(d / "questions.json")
        if not st or qs is None:
            raise KeyError("прогон не найден")
        c = registry.collection(st["collection"])
        r = cls(rid, root, c, qs, st.get("baseline", True), st.get("options"))
        r.started, r.finished = st.get("started"), st.get("finished")
        for variant, store in (("final", r.final), ("baseline", r.base)):
            vd = d / "results" / variant
            if vd.is_dir():
                for f in vd.glob("*.json"):
                    res = _read_json(f)
                    if res and res.get("id"):
                        store[res["id"]] = res
        r.state = st.get("state", "queued")
        if r.state in ("running",):
            r.state = "interrupted"  # процесс перезапущен; ответы, что уже на диске, сохранены
            r._save_state()
        r.evaluation = _read_json(d / "evaluation" / "metrics.json")
        return r

    # ---------------------------------------------------------- запуск / продолжение
    def start(self, resume: bool = False) -> None:
        with self.lock:
            if self.state == "running":
                raise Conflict409("прогон уже идёт")
            if self.state == "completed" and not resume:
                raise Conflict409("прогон уже завершён")
            self.state = "running"
            self.cancelled = False
            self.started = self.started or time.time()
            self.finished = None
        llm = self.llm_factory()
        cfg_path = self.dir / "run_config.json"
        cfg = _read_json(cfg_path) or {}
        info = model_info_of(llm, "reader")
        if not cfg:
            cfg = {"run_id": self.id, "collection": self.collection.id,
                   "snapshot_id": self.collection.meta.get("snapshot_id"),
                   "snapshot_key": self.collection.meta.get("snapshot_key"), "versions": versions(),
                   "questions": len(self.questions), "baseline": self.baseline, "options": self.options,
                   "settings": settings.run_snapshot(), "model_info": info, "started": self.started,
                   "langgraph": _lib_version("langgraph"), "resumes": []}
        if resume:
            cfg.setdefault("resumes", []).append({"t": time.time(), "settings": settings.run_snapshot(),
                                                  "model_info": info, "answered_before": len(self.final)})
        _write_json(cfg_path, cfg)
        _write_json(self.dir / "prompts.json", P.all_prompts())
        _write_json(self.dir / "source_manifest.json",
                    {p: ps.manifest() for p, ps in self.collection.packages.items()})
        _write_json(self.dir / "knowledge_versions.json",
                    {p: {"snapshot_id": k.snapshot_id, "schema": KNOWLEDGE_SCHEMA, "provenance": k.provenance,
                         "coverage": k.coverage(), "sha256": hashlib.sha256(json.dumps(k.to_json(), sort_keys=True,
                         ensure_ascii=False).encode()).hexdigest()}
                     for p, k in self.collection.knowledge.items()})
        self._save_state()
        threading.Thread(target=self._work, daemon=True).start()

    def _one(self, q: dict, variant: str) -> None:
        if self.cancelled:
            return
        if (self.base if variant == "baseline" else self.final).get(q["id"]):
            return  # уже отвечен до перезапуска
        c = self.collection
        ps = c.packages[q["package"]]
        llm = self.llm_factory()
        key = f"{variant}:{q['id']}"
        with self.lock:
            self.running[key] = "чтение"
        journal_path = self.dir / "journal" / f"{_safe(q['id'])}.jsonl"
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        jlock = threading.Lock()

        def journal(ev: dict):  # журнал узлов графа пишется построчно — виден и после сбоя
            with jlock, journal_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.time(), **ev}, ensure_ascii=False, default=str) + "\n")

        try:
            if variant == "baseline":
                info = model_info_of(llm, "baseline")
                r = answer_baseline(ps, q["question"], llm, q["id"], context_tokens=info.get("context_tokens"))
            else:
                def st(s):
                    with self.lock:
                        self.running[key] = s
                r = answer_improved(ps, c.knowledge.get(q["package"]), c.retriever(q["package"], llm),
                                    q["question"], llm, q["id"], on_stage=st, journal=journal,
                                    read_mode=self.options["read_mode"], use_passport=self.options["use_passport"],
                                    strict=self.options["strict_coverage"])
        except Exception as e:  # строка не теряется: техническая ошибка задачи отдельно от «не знаю»
            r = {"id": q["id"], "package": q["package"], "question": q["question"], "variant": variant,
                 "status": "не знаю", "answer": "", "quote": "", "file": "", "evidence": [], "checks": {},
                 "task_error": {"stage": variant, "error": f"{type(e).__name__}: {e}"[:500], "retryable": True},
                 "uncertainty_reason": "", "calls": {}, "ms": 0}
        r["t_done"] = time.time()
        _write_json(self.dir / "results" / variant / f"{_safe(q['id'])}.json", r)
        with self.lock:
            (self.base if variant == "baseline" else self.final)[q["id"]] = r
            self.running.pop(key, None)

    def _work(self) -> None:
        jobs = [(q, "final") for q in self.questions]
        if self.baseline:
            jobs += [(q, "baseline") for q in self.questions]
        try:
            with ThreadPoolExecutor(max_workers=max(1, settings.question_concurrency)) as ex:
                list(ex.map(lambda j: self._one(*j), jobs))
        finally:
            self.finished = time.time()
            self.state = "cancelled" if self.cancelled else "completed"
            self.write_artifacts()
            self._save_state()

    # ---------------------------------------------------------- артефакты
    def csv_bytes(self, variant: str = "final") -> bytes:
        res = self.final if variant == "final" else self.base
        buf = io.StringIO()
        w = csv.writer(buf, delimiter=";", lineterminator="\n")
        w.writerow(["id", "status", "answer", "quote", "file"])
        for q in self.questions:
            r = res.get(q["id"])
            if r is None:
                w.writerow([q["id"], "не знаю", "", "", ""])
                continue
            w.writerow([q["id"], one_line(r["status"]), one_line(r.get("answer")),
                        one_line(r.get("quote")), one_line(r.get("file"))])
        return buf.getvalue().encode("utf-8")

    def write_artifacts(self) -> None:
        order = [q["id"] for q in self.questions]
        (self.dir / "answers.csv").write_bytes(self.csv_bytes("final"))
        _write_json(self.dir / "answers.json", [self.final.get(i) for i in order])
        _write_json(self.dir / "coverage.json", {i: {"mode": (self.final.get(i) or {}).get("mode"),
                                                     "coverage": (self.final.get(i) or {}).get("coverage"),
                                                     "absence_blocked": (self.final.get(i) or {}).get("absence_blocked")}
                                                 for i in order})
        _write_json(self.dir / "calculations.json", {i: self.final[i]["calculation"] for i in order
                                                     if self.final.get(i) and self.final[i].get("calculation")})
        if self.baseline:
            (self.dir / "baseline.csv").write_bytes(self.csv_bytes("baseline"))
            _write_json(self.dir / "baseline.json", [self.base.get(i) for i in order])
            _write_json(self.dir / "baseline_raw.json", {i: {"raw": (self.base.get(i) or {}).get("raw"),
                                                             "format_changes": (self.base.get(i) or {}).get("format_changes"),
                                                             "context_overflow": (self.base.get(i) or {}).get("context_overflow", False),
                                                             "error": (self.base.get(i) or {}).get("error", "")}
                                                         for i in order})
        _write_json(self.dir / "trace.json", {"run": self.snapshot(full=True)})
        if self.state == "completed" or all(self.final.get(i) for i in order):
            (self.dir / "completed").write_text(str(self.finished or ""), encoding="utf-8")

    def metrics(self, res: dict) -> dict:
        rows = [r for r in res.values() if r]
        calls = [r.get("calls") or {} for r in rows]
        ms = sorted(r.get("ms", 0) for r in rows)
        status, reasons, modes, checks = {}, {}, {}, {}
        for r in rows:
            status[r["status"]] = status.get(r["status"], 0) + 1
            if r.get("uncertainty_reason"):
                reasons[r["uncertainty_reason"]] = reasons.get(r["uncertainty_reason"], 0) + 1
            if r.get("mode"):
                modes[r["mode"]] = modes.get(r["mode"], 0) + 1
            for k, v in (r.get("checks") or {}).items():
                c = checks.setdefault(k, {"ok": 0, "fail": 0, "na": 0})
                ok = (v or {}).get("ok")
                c["ok" if ok is True else "fail" if ok is False else "na"] += 1
        costs = [c.get("cost_usd") for c in calls if c.get("cost_usd") is not None]
        return {
            "answered": len(rows), "status": status, "uncertainty_reasons": reasons, "modes": modes,
            "checks": checks,
            "task_errors": sum(1 for r in rows if r.get("task_error")),
            "source_errors": sum(len(r.get("source_errors") or []) for r in rows),
            "absence_blocked": sum(1 for r in rows if r.get("absence_blocked")),
            "judge": sum(1 for r in rows if r.get("judge")),
            "context_overflow": sum(1 for r in rows if r.get("context_overflow")),
            "calls": sum(c.get("calls", 0) for c in calls),
            "cached": sum(c.get("cached", 0) for c in calls),
            "prompt_tokens": sum(c.get("prompt_tokens", 0) for c in calls),
            "completion_tokens": sum(c.get("completion_tokens", 0) for c in calls),
            "cost_usd": round(sum(costs), 4) if costs else None,
            "p50_ms": ms[len(ms) // 2] if ms else 0, "p95_ms": ms[max(0, int(len(ms) * 0.95) - 1)] if ms else 0,
        }

    def snapshot(self, full: bool = False) -> dict:
        with self.lock:
            def slim(r):
                if r is None or full:
                    return r
                return {k: v for k, v in r.items() if k not in ("call_log",)}
            return {
                "id": self.id, "state": self.state, "baseline": self.baseline, "options": self.options,
                "collection": self.collection.id, "snapshot_id": self.collection.meta.get("snapshot_id"),
                "questions": self.questions,
                "final": {k: slim(v) for k, v in self.final.items()},
                "base": {k: slim(v) for k, v in self.base.items()},
                "running": dict(self.running),
                "elapsed": round((self.finished or time.time()) - self.started, 1) if self.started else 0,
                "metrics": {"final": self.metrics(self.final),
                            "baseline": self.metrics(self.base) if self.baseline else None},
                "evaluation": self.evaluation,
            }

    def summary(self) -> dict:
        return {"id": self.id, "state": self.state, "collection": self.collection.id, "questions": len(self.questions),
                "answered": len(self.final), "baseline": self.baseline, "options": self.options,
                "started": self.started, "finished": self.finished}

    # ---------------------------------------------------------- оценка (check.py без изменений)
    def evaluate(self, gold_raw: bytes) -> dict:
        if self.state != "completed":
            raise E.EvalError("оценка доступна после завершения прогона")
        gold = E.load_gold(gold_raw)
        ids = {q["id"] for q in self.questions}
        sub = [g for g in gold if g["id"] in ids]
        missing = ids - {g["id"] for g in sub}
        if not sub:
            raise E.EvalError("в эталонах нет ни одного id из этого прогона")
        for g in sub:
            q = next(x for x in self.questions if x["id"] == g["id"])
            if g["package"] != q["package"]:
                raise E.EvalError(f"{g['id']}: закупка в эталоне {g['package']}, в вопросах {q['package']}")
        ev_dir = self.dir / "evaluation"
        ev_dir.mkdir(exist_ok=True)
        (ev_dir / "gold_original.json").write_bytes(gold_raw)
        gpath = ev_dir / "gold_for_run.json"
        gpath.write_text(json.dumps(sub, ensure_ascii=False, indent=1), encoding="utf-8")
        packages_dir = self.collection.dir / "text"
        out = {"integrity": E.check_integrity(), "missing_in_gold": sorted(missing),
               "gold_sha256": E.sha256(gpath), "variants": {}}
        for variant in (["final", "baseline"] if self.baseline else ["final"]):
            csv_path = self.dir / ("answers.csv" if variant == "final" else "baseline.csv")
            res = E.run_check(csv_path, gpath, packages_dir, ev_dir / f"result_{variant}.csv")
            (ev_dir / f"check_{variant}_stdout.txt").write_text(res["stdout"], encoding="utf-8")
            (ev_dir / f"check_{variant}_stderr.txt").write_text(res["stderr"], encoding="utf-8")
            out["variants"][variant] = {"result": res["result"], "by_type": E.by_type(res["rows"]),
                                        "answers_sha256": res["answers_sha256"],
                                        "stdout_sha256": hashlib.sha256(res["stdout"].encode()).hexdigest(),
                                        "verdicts": {r["id"]: {"verdict": r["verdict"], "why": r["why"]}
                                                     for r in res["rows"]}}
        out["gold"] = {g["id"]: {"status": g["status"], "expected": E.expected_text(g), "type": g["type"],
                                 "quote": g.get("quote", ""), "file": g.get("file", "")} for g in sub}
        _write_json(ev_dir / "metrics.json", out)
        self.evaluation = out
        return out


def _lib_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


# =================================================================== реестр
class Registry:
    READY = ("ready", "ready_with_warnings")

    def __init__(self, root: Path):
        self.root = root
        import opora.documents as D
        D.OCR_CACHE = root / "ocr_cache"
        for sub in ("collections", "runs", "chats"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        self.collections: dict[str, Collection] = {}
        self.runs: dict[str, Run] = {}
        self.lock = threading.RLock()
        self.llm_factory = client

    # ---------------------------------------------------------- снимки
    def new_collection(self) -> Collection:
        cid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        c = Collection(cid, self.root / "collections")
        c.dir.mkdir(parents=True)
        c.save()
        with self.lock:
            self.collections[cid] = c
        return c

    def collection(self, cid: str) -> Collection:
        with self.lock:
            if cid in self.collections:
                return self.collections[cid]
            if not re.fullmatch(r"[\w-]+", cid) or not (self.root / "collections" / cid / "meta.json").exists():
                raise KeyError("коллекция не найдена")
            c = Collection.restore(cid, self.root / "collections")
            self.collections[cid] = c
            return c

    def _metas(self) -> list[dict]:
        out = []
        for d in (self.root / "collections").iterdir():
            mpath = d / "meta.json"
            if not mpath.is_file():
                continue
            with self.lock:
                live = self.collections.get(d.name)
            if live is not None:
                with live.lock:
                    meta = dict(live.meta)
            else:
                meta = _read_json(mpath)
                if not isinstance(meta, dict) or "id" not in meta:
                    continue
                if meta.get("state") == "processing":
                    meta["state"] = "interrupted"
            out.append(meta)
        out.sort(key=lambda x: (x.get("created") or 0, x["id"]), reverse=True)
        return out

    def duplicate_of(self, c: Collection) -> str | None:
        """id готового снимка с тем же snapshot_id (те же файлы, версии и модели) или None."""
        m = c.manifest()
        if not m:
            return None
        sid, _ = snapshot_id_of(m)
        for meta in self._metas():
            if meta["id"] != c.id and meta.get("state") in self.READY and meta.get("snapshot_id") == sid:
                return meta["id"]
        return None

    def delete_collection(self, cid: str) -> None:
        import shutil
        d = (self.root / "collections" / cid).resolve()
        if not re.fullmatch(r"[\w-]+", cid) or d.parent != (self.root / "collections").resolve():
            raise KeyError("коллекция не найдена")
        with self.lock:
            self.collections.pop(cid, None)
        shutil.rmtree(d, ignore_errors=True)

    def start_ingest(self, c: Collection, background: bool = True) -> dict:
        """Запечатать и подготовить. Тот же снимок второй раз не готовится. Прерванная подготовка продолжается."""
        with c.lock:
            if c.meta.get("state") == "processing":
                return {"ok": True, "already": True}
            resuming = c.meta.get("state") in ("interrupted", "failed", "ready_with_warnings")
        if not resuming:
            dup = self.duplicate_of(c)
            if dup:
                self.delete_collection(c.id)
                return {"ok": True, "use": dup}
        with c.lock:
            c.meta["state"] = "processing"
        llm = self.llm_factory() if settings.api_key else None
        if background:
            threading.Thread(target=c.ingest, args=(llm,), daemon=True).start()
        else:
            c.ingest(llm)
        return {"ok": True, "resumed": resuming}

    def list_collections(self) -> list[dict]:
        chosen: dict[str, dict] = {}
        for meta in self._metas():
            key = meta.get("snapshot_id") or "id:" + meta["id"]
            cur = chosen.get(key)
            if cur is None or (cur.get("state") not in self.READY and meta.get("state") in self.READY):
                chosen[key] = meta
        out = []
        for meta in sorted(chosen.values(), key=lambda x: (x.get("created") or 0, x["id"]), reverse=True):
            names = [p["name"] for p in meta.get("packages") or [] if isinstance(p, dict) and "name" in p]
            if not names:
                names = sorted({f.get("package", "") for f in meta.get("files") or []} - {""})
            files = {(f.get("package"), f.get("name")) for f in meta.get("files") or []}
            out.append({"id": meta["id"], "created": meta.get("created"), "state": meta.get("state"),
                        "snapshot_id": meta.get("snapshot_id"), "packages": names, "files": len(files),
                        "label": ", ".join(names)})
        return out[:20]

    # ---------------------------------------------------------- прогоны
    def new_run(self, c: Collection, questions: list[dict], baseline: bool, options: dict | None = None) -> Run:
        if c.meta.get("state") not in self.READY:
            raise Conflict409("снимок ещё не подготовлен")
        rid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        r = Run(rid, self.root / "runs", c, questions, baseline, options)
        r.llm_factory = self.llm_factory
        r.persist_new()
        with self.lock:
            self.runs[rid] = r
        return r

    def run(self, rid: str) -> Run:
        with self.lock:
            if rid in self.runs:
                return self.runs[rid]
            if not re.fullmatch(r"[\w-]+", rid) or not (self.root / "runs" / rid / "state.json").exists():
                raise KeyError("прогон не найден")
            r = Run.restore(rid, self.root / "runs", self)
            r.llm_factory = self.llm_factory
            self.runs[rid] = r
            return r

    def list_runs(self, collection: str | None = None) -> list[dict]:
        out = []
        for d in sorted((self.root / "runs").iterdir(), reverse=True):
            if not (d / "state.json").exists():
                continue
            with self.lock:
                live = self.runs.get(d.name)
            if live is not None:
                s = live.summary()
            else:
                st = _read_json(d / "state.json") or {}
                n = len(_read_json(d / "questions.json") or [])
                done = len(list((d / "results" / "final").glob("*.json"))) if (d / "results" / "final").is_dir() else 0
                state = st.get("state")
                s = {"id": d.name, "state": "interrupted" if state == "running" else state,
                     "collection": st.get("collection"), "questions": n, "answered": done,
                     "baseline": st.get("baseline"), "options": st.get("options"),
                     "started": st.get("started"), "finished": st.get("finished")}
            if collection and s.get("collection") != collection:
                continue
            out.append(s)
        return out[:30]

    # ---------------------------------------------------------- диалоги (JSONL на диске)
    def _chat_path(self, c: Collection, package: str, session: str) -> Path:
        return self.root / "chats" / c.id / _safe(package) / f"{_safe(session)}.jsonl"

    def chat_history(self, c: Collection, package: str, session: str) -> list[dict]:
        p = self._chat_path(c, package, session)
        if not p.exists():
            return []
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def chat(self, c: Collection, package: str, question: str, session: str, options: dict | None = None) -> dict:
        """Вопрос из диалога: уточнения переписываются в самостоятельный вопрос (узел plan_question)."""
        if package not in c.packages:
            raise KeyError("закупка не найдена")
        opts = run_options(options)
        hist = self.chat_history(c, package, session)
        history = [{"role": m["role"], "text": m["text"]} for m in hist]
        llm = self.llm_factory()
        n = sum(1 for m in hist if m["role"] == "user") + 1
        r = answer_improved(c.packages[package], c.knowledge.get(package), c.retriever(package, llm), question, llm,
                            qid=f"chat-{n}", history=history, read_mode=opts["read_mode"],
                            use_passport=opts["use_passport"], strict=opts["strict_coverage"])
        r["asked"] = question
        p = self._chat_path(c, package, session)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"role": "user", "text": question, "t": time.time()}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"role": "assistant", "text": f"[{r['status']}] {r.get('answer_text') or r['answer']}",
                                "t": time.time(), "result": {k: v for k, v in r.items() if k != "call_log"}},
                               ensure_ascii=False, default=str) + "\n")
        return r


def diagnostics() -> dict:
    """Состояние инфраструктуры для стартового экрана: модель, эмбеддинги, Qdrant, LangGraph."""
    q = get_qdrant()
    try:
        qd = q.diagnostics()
    except Exception as e:  # noqa: BLE001
        qd = {"ok": False, "error": str(e)}
    info = {}
    if settings.api_key:
        try:
            info = client().model_info()
        except Exception as e:  # noqa: BLE001
            info = {"error": str(e)[:200]}
    return {"model": settings.model, "embedding_model": settings.embedding_model, "model_info": info,
            "api_key_configured": bool(settings.api_key), "qdrant": qd, "langgraph": _lib_version("langgraph"),
            "versions": versions(), "read_mode": settings.read_mode, "use_passport": settings.use_passport,
            "strict_coverage": settings.strict_coverage}
