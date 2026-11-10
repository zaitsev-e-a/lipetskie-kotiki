"""Гибридный поиск в Qdrant Server: dense (OpenRouter embeddings) + BM25 + RRF.

- Работа через REST API Qdrant (стандартная библиотека). Проверено по документации Qdrant 1.20:
  BM25 считается по полнотекстовому индексу payload-поля `text` с `scoring` (Qdrant ≥ 1.20),
  запрос — `{"text": "..."}` с `using` = имя поля. Для Qdrant 1.15.2–1.19 — запасной вариант:
  sparse-вектор BM25 с серверной инференцией `qdrant/bm25` и модификатором IDF.
- Русский текст: токенизатор word, нижний регистр, стеммер Snowball russian. Стоп-слова НЕ удаляются:
  в стандартном русском списке есть «не»/«нет», а отрицания для нас существенны.
- Обе ветви получают кандидатов независимо, объединение — RRF по рангам (k из настроек), без сложения
  сырых BM25 и косинуса.
- Во всех ветвях фильтр snapshot_id + package_id задаёт сервер приложения.
- Коллекция версионируется по embedding-модели, размерности и схеме индекса: другая модель — другая коллекция.
- Недоступный Qdrant не маскируется: при LOCAL_INDEX=1 тот же гибридный поиск идёт по LocalIndex в памяти
  процесса, иначе граф переключает режим чтения; переключение пишется в предупреждения и журнал.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
import uuid

from .config import settings
from .llm import LLMError, Tracker
from .schemas import INDEX_SCHEMA
from .sources import PackageSources

TEXT_FIELD = "text"


class QdrantUnavailable(RuntimeError):
    pass


class QdrantError(RuntimeError):
    pass


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40]


class Qdrant:
    def __init__(self, url: str | None = None, api_key: str | None = None, timeout: int = 30):
        self.url = (url or settings.qdrant_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.qdrant_api_key
        self.timeout = timeout
        self.bm25_mode: str | None = None  # fulltext | sparse_inference

    # ------------------------------------------------------------------ HTTP
    def _req(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json",
                                              **({"api-key": self.api_key} if self.api_key else {})})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:600] if e.fp else str(e)
            raise QdrantError(f"Qdrant {method} {path}: HTTP {e.code}: {detail}")
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise QdrantUnavailable(f"Qdrant недоступен по адресу {self.url}: {e}")

    # ------------------------------------------------------------------ диагностика
    def health(self) -> dict:
        info = self._req("GET", "/")
        return {"ok": True, "url": self.url, "version": info.get("version"), "title": info.get("title")}

    def diagnostics(self, collection: str | None = None) -> dict:
        try:
            out = self.health()
        except (QdrantUnavailable, QdrantError) as e:
            return {"ok": False, "url": self.url, "error": str(e)}
        if collection:
            try:
                c = self._req("GET", f"/collections/{collection}").get("result") or {}
                params = (c.get("config") or {}).get("params") or {}
                out.update(collection=collection, points=c.get("points_count"), status=c.get("status"),
                           vectors=params.get("vectors"), sparse_vectors=params.get("sparse_vectors"),
                           payload_schema=c.get("payload_schema"), bm25_mode=self.bm25_mode)
            except QdrantError as e:
                out.update(collection=collection, collection_error=str(e))
        return out

    # ------------------------------------------------------------------ коллекция
    @staticmethod
    def collection_name(embedding_model: str, dim: int) -> str:
        return f"opora_{INDEX_SCHEMA.replace('.', '_').replace('-', '_')}_{_slug(embedding_model)}_{dim}"

    def ensure_collection(self, name: str, dim: int) -> None:
        try:
            self._req("GET", f"/collections/{name}")
            self._detect_bm25_mode(name)
            return
        except QdrantError:
            pass
        self._req("PUT", f"/collections/{name}", {
            "vectors": {"dense": {"size": dim, "distance": "Cosine"}},
            "sparse_vectors": {"bm25": {"modifier": "idf"}},
        })
        for field, schema in (("snapshot_id", "keyword"), ("package_id", "keyword"), ("kind", "keyword"),
                              ("document_id", "keyword")):
            self._req("PUT", f"/collections/{name}/index?wait=true", {"field_name": field, "field_schema": schema})
        text_index = {"type": "text", "tokenizer": "word", "lowercase": True,
                      "stemmer": {"type": "snowball", "language": "russian"}}
        # BM25 на полнотекстовом индексе (Qdrant ≥ 1.20). В документации встречаются две записи
        # параметра scoring — пробуем обе; если сервер старее, переходим на sparse BM25.
        for scoring in (True, {"type": "bm25"}):
            try:
                self._req("PUT", f"/collections/{name}/index?wait=true",
                          {"field_name": TEXT_FIELD, "field_schema": {**text_index, "scoring": scoring}})
                self.bm25_mode = "fulltext"
                break
            except QdrantError:
                continue
        if self.bm25_mode is None:
            self._req("PUT", f"/collections/{name}/index?wait=true",
                      {"field_name": TEXT_FIELD, "field_schema": text_index})
            self.bm25_mode = "sparse_inference"

    def _detect_bm25_mode(self, name: str) -> None:
        c = self._req("GET", f"/collections/{name}").get("result") or {}
        ps = c.get("payload_schema") or {}
        params = ((ps.get(TEXT_FIELD) or {}).get("params") or {})
        self.bm25_mode = "fulltext" if params.get("scoring") else "sparse_inference"

    # ------------------------------------------------------------------ запись
    def upsert(self, name: str, points: list[dict]) -> None:
        for i in range(0, len(points), 64):
            self._req("PUT", f"/collections/{name}/points?wait=true", {"points": points[i:i + 64]})

    def count(self, name: str, flt: dict) -> int:
        return int((self._req("POST", f"/collections/{name}/points/count", {"filter": flt, "exact": True})
                    .get("result") or {}).get("count") or 0)

    # ------------------------------------------------------------------ поиск
    def _bm25_query(self, text: str) -> dict:
        if self.bm25_mode == "fulltext":
            return {"query": {"text": text}, "using": TEXT_FIELD}
        return {"query": {"text": text, "model": "qdrant/bm25", "options": {"language": "russian"}},
                "using": "bm25"}

    def dense_search(self, name: str, vector: list[float], flt: dict, limit: int) -> list[dict]:
        r = self._req("POST", f"/collections/{name}/points/query",
                      {"query": vector, "using": "dense", "filter": flt, "limit": limit, "with_payload": True})
        return (r.get("result") or {}).get("points") or []

    def bm25_search(self, name: str, text: str, flt: dict, limit: int) -> list[dict]:
        r = self._req("POST", f"/collections/{name}/points/query",
                      {**self._bm25_query(text), "filter": flt, "limit": limit, "with_payload": True})
        return (r.get("result") or {}).get("points") or []

    def hybrid_search(self, name: str, vector: list[float], text: str, flt: dict, limit: int,
                      prefetch_limit: int, rrf_k: int) -> list[dict]:
        body = {"prefetch": [{"query": vector, "using": "dense", "filter": flt, "limit": prefetch_limit},
                             {**self._bm25_query(text), "filter": flt, "limit": prefetch_limit}],
                "query": {"rrf": {"k": rrf_k}}, "filter": flt, "limit": limit, "with_payload": True}
        r = self._req("POST", f"/collections/{name}/points/query", body)
        return (r.get("result") or {}).get("points") or []


def scope_filter(snapshot_id: str, package_id: str, kind: str | None = None) -> dict:
    must = [{"key": "snapshot_id", "match": {"value": snapshot_id}},
            {"key": "package_id", "match": {"value": package_id}}]
    if kind:
        must.append({"key": "kind", "match": {"value": kind}})
    return {"must": must}


def _point_id(*parts: str) -> str:
    return str(uuid.UUID(hashlib.md5("/".join(parts).encode()).hexdigest()))


class Retriever:
    """Индекс закупки в Qdrant: чанки и факты паспорта. Используется узлом select_context."""

    def __init__(self, qdrant: Qdrant | None = None, llm=None):
        self.q = qdrant or Qdrant()
        self.llm = llm
        self.collection: str | None = None
        self.dim: int | None = None

    def index_package(self, ps: PackageSources, knowledge, tracker: Tracker, progress=None) -> dict:
        """Чанки (kind=chunk) и проверенные факты (kind=fact). Текст перед embedding не обрезается:
        чанки ограничены токенным бюджетом при построении."""
        texts, payloads, ids = [], [], []
        for c in ps.chunks:
            doc = ps.doc_by_id[c.document_id]
            t = doc.text[c.start:c.end]
            texts.append(t)
            payloads.append({"kind": "chunk", "snapshot_id": ps.snapshot_id, "package_id": ps.package_id,
                             "document_id": c.document_id, "file": c.file, "chunk_id": c.chunk_id,
                             "start": c.start, "end": c.end, "block_ids": c.block_ids,
                             "text_sha256": ps.text_sha[c.document_id], "page": c.page, "section": c.section,
                             TEXT_FIELD: t})
            ids.append(_point_id(ps.snapshot_id, ps.package_id, c.chunk_id))
        for f in (knowledge.facts if knowledge else []):
            t = f"{f['attribute_label'] or f['attribute']}: {f['raw_value']}. {f['evidence']['quote']}"
            texts.append(t)
            payloads.append({"kind": "fact", "snapshot_id": ps.snapshot_id, "package_id": ps.package_id,
                             "fact_id": f["fact_id"], "document_id": f["evidence"]["document_id"],
                             "file": f["file"], "block_ids": [f["evidence"]["source_id"]], TEXT_FIELD: t})
            ids.append(_point_id(ps.snapshot_id, ps.package_id, f["fact_id"]))
        if not texts:
            return {"points": 0}
        vectors = self.llm.embed(texts, tracker)
        dim = len(vectors[0])
        name = self.q.collection_name(settings.embedding_model, dim)
        self.q.ensure_collection(name, dim)
        self.collection, self.dim = name, dim
        points = []
        for pid, v, pl in zip(ids, vectors, payloads):
            vec = {"dense": v}
            if self.q.bm25_mode == "sparse_inference":
                vec["bm25"] = {"text": pl[TEXT_FIELD], "model": "qdrant/bm25", "options": {"language": "russian"}}
            points.append({"id": pid, "vector": vec, "payload": pl})
        self.q.upsert(name, points)
        n = self.q.count(name, scope_filter(ps.snapshot_id, ps.package_id))
        if n < len(points):
            raise QdrantError(f"в индексе {n} точек из {len(points)}")
        return {"collection": name, "dim": dim, "points": n, "bm25_mode": self.q.bm25_mode,
                "embedding_model": settings.embedding_model}

    def attach(self, collection: str, dim: int) -> None:
        self.collection, self.dim = collection, dim
        self.q._detect_bm25_mode(collection)

    def search(self, ps: PackageSources, question: str, tracker: Tracker, kind: str = "chunk",
               limit: int | None = None) -> dict:
        if not self.collection:
            raise QdrantUnavailable("закупка не проиндексирована в Qdrant")
        try:
            qv = self.llm.embed([question], tracker)[0]
        except LLMError as e:
            raise QdrantUnavailable(f"эмбеддинг вопроса не получен: {e}")
        flt = scope_filter(ps.snapshot_id, ps.package_id, kind)
        lim = limit or settings.retrieval_top_k
        hits = self.q.hybrid_search(self.collection, qv, question, flt, lim, settings.prefetch_limit, settings.rrf_k)
        # ветви отдельно — для журнала и проверки независимости
        dense = self.q.dense_search(self.collection, qv, flt, lim)
        bm25 = self.q.bm25_search(self.collection, question, flt, lim)
        for h in hits:  # защита от подмешивания: сервер обязан вернуть только наш scope
            pl = h.get("payload") or {}
            if pl.get("snapshot_id") != ps.snapshot_id or pl.get("package_id") != ps.package_id:
                raise QdrantError("Qdrant вернул точку другой закупки/снимка — запрос отклонён")
        return {"hits": [{"id": h.get("id"), "score": h.get("score"), **(h.get("payload") or {})} for h in hits],
                "dense_ids": [h.get("id") for h in dense], "bm25_ids": [h.get("id") for h in bm25],
                "collection": self.collection, "bm25_mode": self.q.bm25_mode, "rrf_k": settings.rrf_k}


# ====================================================================== локальный индекс (самовосстановление)
_RU_SUFFIX = re.compile(r"(иями|ями|ами|иях|ого|его|ому|ему|ыми|ими|ией|ей|ой|ий|ый|ая|яя|ое|ее|ые|ие|ую|юю|"
                        r"ов|ев|ах|ях|ам|ям|ом|ем|ия|ья|а|я|о|е|ы|и|у|ю|ь)$")


def _lex_terms(t: str) -> list[str]:
    """Лёгкий русский стеммер для локального BM25: окончания срезаются, корень не короче 4 букв."""
    out = []
    for w in re.findall(r"[а-яёa-z0-9]+", (t or "").lower().replace("ё", "е")):
        if len(w) > 5:
            w = _RU_SUFFIX.sub("", w) if len(_RU_SUFFIX.sub("", w)) >= 4 else w
        out.append(w[:8])
    return out


class LocalIndex:
    """Гибридный индекс в памяти процесса: dense (те же эмбеддинги OpenRouter, косинус) + BM25 + RRF.

    Включается автоматически, если Qdrant Server недоступен: вместо «прочитать все окна» (дорого и упирается
    в бюджет вызовов) вопрос по большой закупке по-прежнему идёт через поиск. Переключение пишется в
    предупреждения снимка. Тот же интерфейс, что у Qdrant: фильтры payload, ветви по отдельности, RRF по рангам.
    Эмбеддинги кешируются на диске, поэтому после перезапуска индекс восстанавливается без затрат.
    """
    url = "local://memory"

    def __init__(self):
        self.cols: dict[str, dict] = {}
        self._lex: dict[str, dict] = {}
        self.bm25_mode = "local_bm25"

    def health(self) -> dict:
        return {"ok": True, "version": "local", "url": self.url}

    def diagnostics(self, collection: str | None = None) -> dict:
        return {"ok": True, "version": "local", "url": self.url, "bm25_mode": self.bm25_mode,
                "collections": {k: len(v) for k, v in self.cols.items()}}

    def collection_name(self, model: str, dim: int) -> str:
        return f"local_{INDEX_SCHEMA.replace('.', '_').replace('-', '_')}_{_slug(model)}_{dim}"

    def ensure_collection(self, name: str, dim: int) -> None:
        self.cols.setdefault(name, {})

    def _detect_bm25_mode(self, name: str) -> None:
        return None

    def upsert(self, name: str, points: list[dict]) -> None:
        import math
        col = self.cols.setdefault(name, {})
        for pt in points:
            v = pt["vector"]["dense"]
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            col[pt["id"]] = {**pt, "_unit": [x / n for x in v], "_terms": _lex_terms(pt["payload"].get(TEXT_FIELD, ""))}
        self._lex.pop(name, None)

    @staticmethod
    def _match(pl: dict, flt: dict | None) -> bool:
        return all(pl.get(c["key"]) == c["match"]["value"] for c in (flt or {}).get("must", []))

    def count(self, name: str, flt: dict | None = None) -> int:
        return sum(1 for p in self.cols.get(name, {}).values() if self._match(p["payload"], flt))

    def _pts(self, name, flt):
        return [p for p in self.cols.get(name, {}).values() if self._match(p["payload"], flt)]

    def dense_search(self, name, vector, flt, limit):
        import math
        n = math.sqrt(sum(x * x for x in vector)) or 1.0
        q = [x / n for x in vector]
        res = [{"id": p["id"], "score": sum(a * b for a, b in zip(q, p["_unit"])), "payload": p["payload"]}
               for p in self._pts(name, flt)]
        return sorted(res, key=lambda h: -h["score"])[:limit]

    def bm25_search(self, name, text, flt, limit):
        import math
        from collections import Counter
        pts = self._pts(name, flt)
        docs = [Counter(p["_terms"]) for p in pts]
        n = len(docs) or 1
        avg = (sum(sum(d.values()) for d in docs) / n) or 1
        df = Counter(t for d in docs for t in d)
        res = []
        for p, d in zip(pts, docs):
            ln, sc = sum(d.values()), 0.0
            for t in set(_lex_terms(text)):
                if d[t]:
                    idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    sc += idf * d[t] * 2.2 / (d[t] + 1.2 * (0.25 + 0.75 * ln / avg))
            if sc > 0:
                res.append({"id": p["id"], "score": sc, "payload": p["payload"]})
        return sorted(res, key=lambda h: -h["score"])[:limit]

    def hybrid_search(self, name, vector, text, flt, limit, prefetch, rrf_k):
        score: dict = {}
        pl: dict = {}
        for branch in (self.dense_search(name, vector, flt, prefetch), self.bm25_search(name, text, flt, prefetch)):
            for rank, h in enumerate(branch):
                score[h["id"]] = score.get(h["id"], 0.0) + 1.0 / (rrf_k + rank + 1)
                pl[h["id"]] = h["payload"]
        best = sorted(score, key=lambda i: -score[i])[:limit]
        return [{"id": i, "score": score[i], "payload": pl[i]} for i in best]


_local: "LocalIndex | None" = None


def get_local_index() -> LocalIndex:
    global _local
    if _local is None:
        _local = LocalIndex()
    return _local


# ====================================================================== фабрика
_qdrant: "Qdrant | None" = None


def get_qdrant():
    global _qdrant
    if _qdrant is None:
        _qdrant = Qdrant()
    return _qdrant


def set_qdrant(q) -> None:
    global _qdrant
    _qdrant = q
