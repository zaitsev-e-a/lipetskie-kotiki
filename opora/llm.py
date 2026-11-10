"""Клиент OpenRouter на стандартной библиотеке: чат, эмбеддинги, учёт вызовов.

- Размышления задаются по ролям (REASONING_<РОЛЬ>): по умолчанию выключены у быстрых ролей и минимальны у проверяющих.
- Структурированный вывод: json_schema → json_object → разбор текста.
- Повторы только для транспортных ошибок (429, 5xx, таймаут) с Retry-After.
- Каждый вызов записывается: роль, модель, время, токены, стоимость, ошибка.
- У каждой роли своя модель (MODEL_<РОЛЬ> в .env) и, при желании, запасная (FALLBACK_MODEL_<РОЛЬ> или
  OPENROUTER_FALLBACK_MODEL): если основная после повторов и исправления формата так и не дала ответ,
  тот же запрос уходит запасной — компонент сам восстанавливается, вместо ошибки в ответе.
"""
from __future__ import annotations

import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict

from .config import settings


class LLMError(RuntimeError):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind  # auth | balance | model | rate | transport | protocol | config


@dataclass
class Call:
    role: str
    model: str
    ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float | None = None
    retries: int = 0
    ok: bool = True
    error: str = ""
    provider: str = ""
    finish_reason: str = ""
    cached: bool = False
    schema_repair: bool = False
    format_mode: str = ""
    fallback: bool = False


@dataclass
class Tracker:
    """Счётчик вызовов одного вопроса или прогона (потокобезопасный)."""
    calls: list[Call] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, c: Call) -> None:
        with self._lock:
            self.calls.append(c)

    def extend(self, other: "Tracker") -> None:
        with self._lock:
            self.calls.extend(other.calls)

    def summary(self) -> dict:
        with self._lock:
            calls = list(self.calls)
        by_role: dict[str, int] = {}
        for c in calls:
            by_role[c.role] = by_role.get(c.role, 0) + 1
        costs = [c.cost for c in calls if c.cost is not None]
        return {
            "calls": len(calls),
            "by_role": by_role,
            "prompt_tokens": sum(c.prompt_tokens for c in calls),
            "completion_tokens": sum(c.completion_tokens for c in calls),
            "cost_usd": round(sum(costs), 6) if costs else None,
            "llm_ms": sum(c.ms for c in calls),
            "errors": sum(1 for c in calls if not c.ok),
            "retries": sum(c.retries for c in calls),
            "cached": sum(1 for c in calls if c.cached),
            "schema_repairs": sum(1 for c in calls if c.schema_repair),
            "fallbacks": sum(1 for c in calls if c.fallback),
            "by_model": _count_by(calls, "model"),
        }

    def to_list(self) -> list[dict]:
        with self._lock:
            return [asdict(c) for c in self.calls]


def _count_by(calls: list, attr: str) -> dict:
    out: dict[str, int] = {}
    for c in calls:
        if c.role != "embed" and not c.cached:
            k = getattr(c, attr) or "?"
            out[k] = out.get(k, 0) + 1
    return out


# ошибки, после которых имеет смысл попробовать запасную модель (ключ и баланс — нет)
FALLBACK_KINDS = {"transport", "rate", "protocol", "model", "length"}


def parse_json_loose(text: str) -> dict:
    """Достаёт JSON из ответа модели: снимает ```json, <think>, лишний текст."""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t.strip())
    try:
        v = json.loads(t)
        if isinstance(v, dict):
            return v
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        v = json.loads(t[start:i + 1])
                        if isinstance(v, dict):
                            return v
                    except json.JSONDecodeError:
                        break
        start = t.find("{", start + 1)
    raise LLMError("protocol", "модель вернула не JSON")


class OpenRouter:
    def __init__(self, cfg=settings):
        self.cfg = cfg
        self._sem = threading.BoundedSemaphore(max(1, cfg.global_concurrency))
        self._format_mode: dict[str, str] = {}  # model → json_schema | json_object | text
        self._model_infos: dict[str, dict] = {}

    # ---------------------------------------------------------------- HTTP
    def _post(self, path: str, payload: dict) -> dict:
        if not self.cfg.api_key:
            raise LLMError("config", "Не задан OPENROUTER_API_KEY в файле .env")
        req = urllib.request.Request(
            self.cfg.base_url.rstrip("/") + path,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.cfg.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost/opora",
                "X-Title": "Opora - Lipetskie kotiki",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.cfg.timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
        data = json.loads(body) if body.strip() else {}
        if isinstance(data, dict) and data.get("error"):
            err = data["error"]
            code = err.get("code") if isinstance(err, dict) else None
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise urllib.error.HTTPError(req.full_url, int(code or 502), str(msg), None, None)
        return data

    def _with_retries(self, fn, call: Call):
        attempt = 0
        while True:
            try:
                with self._sem:
                    return fn()
            except urllib.error.HTTPError as e:
                status = e.code
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")[:500] if e.fp else str(e.msg)
                except Exception:
                    detail = str(e.msg)
                if status == 401:
                    raise LLMError("auth", "Неверный ключ OpenRouter (401)")
                if status == 402:
                    raise LLMError("balance", "Недостаточно средств на OpenRouter (402)")
                if status in (400, 404, 422) and attempt == 0:
                    raise LLMError("model", f"Запрос отклонён ({status}): {detail or e.msg}")
                if status in (408, 429, 500, 502, 503, 504) and attempt < self.cfg.transport_retries:
                    attempt += 1
                    call.retries = attempt
                    ra = e.headers.get("Retry-After") if e.headers else None
                    delay = float(ra) if ra and ra.replace(".", "").isdigit() else 1.5 * 2 ** (attempt - 1)
                    time.sleep(min(delay, 20) + random.random() * 0.5)
                    continue
                raise LLMError("rate" if status == 429 else "transport", f"HTTP {status}: {detail or e.msg}")
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                if attempt < self.cfg.transport_retries:
                    attempt += 1
                    call.retries = attempt
                    time.sleep(1.5 * 2 ** (attempt - 1) + random.random() * 0.5)
                    continue
                raise LLMError("transport", f"Нет связи с OpenRouter: {e}")

    # ---------------------------------------------------------------- chat
    def _cache_path(self, key: str):
        from pathlib import Path
        d = Path(self.cfg.data_dir) / "cache" / "llm" / key[:2]
        return d / f"{key}.json"

    def chat_json(self, role: str, system: str, user: str, schema: dict | None, tracker: Tracker | None,
                  max_tokens: int | None = None, cache: bool = True) -> dict:
        """Один логический вызов роли: транспортные повторы, проверка схемы, одно исправление формата,
        при неудаче — запасная модель роли (если задана).

        finish_reason=length — незавершённый протокол (LLMError kind="length"), а не успешный JSON:
        вызывающий узел уменьшает порцию входа или повторяет с большим бюджетом.
        """
        primary = self.cfg.role_model(role)
        try:
            return self._chat_json_model(role, primary, self.cfg.role_provider(role), system, user, schema,
                                         tracker, max_tokens, cache)
        except LLMError as e:
            fb = self.cfg.role_fallback(role)
            if not fb or e.kind not in FALLBACK_KINDS:
                raise
            return self._chat_json_model(role, fb, "", system, user, schema, tracker, max_tokens, cache,
                                         fallback=True)

    def _chat_json_model(self, role: str, model: str, provider: str, system: str, user: str, schema: dict | None,
                         tracker: Tracker | None, max_tokens: int | None, cache: bool,
                         fallback: bool = False) -> dict:
        import hashlib
        from .prompts import PROMPTS_VERSION
        from .schemas import validate_schema, SchemaError
        limit = max_tokens or self.cfg.role_max_tokens(role)
        reasoning = self.cfg.role_reasoning(role)
        key = hashlib.sha256(json.dumps([model, provider, role, reasoning, limit, PROMPTS_VERSION,
                                         system, user, schema], ensure_ascii=False).encode()).hexdigest()
        use_cache = cache and self.cfg.llm_cache and role != "diagnostic"
        if use_cache:
            path = self._cache_path(key)
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if tracker is not None:
                        tracker.add(Call(role=role, model=model, cached=True, fallback=fallback))
                    return data
                except (OSError, json.JSONDecodeError):
                    pass
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        result = None
        for attempt in range(2):  # 0 — обычный ответ; 1 — одно исправление формата по ошибке схемы
            kw = dict(model=model, provider=provider, fallback=fallback, schema_repair=attempt > 0)
            try:
                raw = self._chat_once(role, messages, schema, limit, reasoning, tracker, **kw)
            except LLMError as e:
                # finish_reason=length: один повтор с удвоенным бюджетом ответа (не принимаем обрезанный JSON);
                # если и этого мало — ошибка length уходит в узел, который уменьшит порцию входа
                if e.kind != "length" or limit >= 16000:
                    raise
                limit = min(16000, limit * 2)
                raw = self._chat_once(role, messages, schema, limit, reasoning, tracker, **kw)
            try:
                result = parse_json_loose(raw)
                if schema:
                    validate_schema(result, schema)
                break
            except (LLMError, SchemaError) as e:
                if attempt == 1:
                    raise LLMError("protocol", f"ответ роли {role} не соответствует схеме: {e}")
                messages = messages + [{"role": "assistant", "content": raw[:4000]},
                                       {"role": "user", "content": f"Ответ не соответствует схеме ({e}). "
                                                                   "Верни исправленный JSON-объект строго по схеме."}]
        if use_cache and result is not None:
            path = self._cache_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        return result

    def _chat_once(self, role: str, messages: list, schema: dict | None, limit: int, reasoning: str,
                   tracker: Tracker | None, schema_repair: bool = False, model: str | None = None,
                   provider: str = "", fallback: bool = False) -> str:
        model = model or self.cfg.role_model(role)
        call = Call(role=role, model=model, schema_repair=schema_repair, fallback=fallback)
        t0 = time.time()
        payload = {"model": model, "messages": messages, "temperature": self.cfg.temperature,
                   "max_tokens": limit, "usage": {"include": True}}
        payload["reasoning"] = {"enabled": False} if reasoning == "off" else {"effort": reasoning, "exclude": True}
        if provider:
            payload["provider"] = {"order": [provider], "allow_fallbacks": False}
        mode = self._format_mode.get(model, "json_schema" if schema else "json_object")
        try:
            while True:
                p = dict(payload)
                if mode == "json_schema" and schema:
                    p["response_format"] = {"type": "json_schema",
                                            "json_schema": {"name": role, "strict": False, "schema": schema}}
                elif mode == "json_object":
                    p["response_format"] = {"type": "json_object"}
                try:
                    data = self._with_retries(lambda: self._post("/chat/completions", p), call)
                except LLMError as e:
                    # маршрут не поддерживает формат — понижаем и запоминаем; схема всё равно проверяется кодом
                    if e.kind == "model" and mode != "text":
                        mode = "json_object" if mode == "json_schema" else "text"
                        self._format_mode[model] = mode
                        continue
                    raise
                break
            choice = (data.get("choices") or [{}])[0]
            call.finish_reason = choice.get("finish_reason") or ""
            call.provider = data.get("provider", "")
            call.format_mode = mode
            usage = data.get("usage") or {}
            call.prompt_tokens = int(usage.get("prompt_tokens") or 0)
            call.completion_tokens = int(usage.get("completion_tokens") or 0)
            if usage.get("cost") is not None:
                call.cost = float(usage["cost"])
            if call.finish_reason == "length":
                raise LLMError("length", f"ответ роли {role} обрезан по длине (finish_reason=length, лимит {limit})")
            return (choice.get("message") or {}).get("content") or ""
        except LLMError as e:
            call.ok, call.error = False, f"{e.kind}: {e}"[:300]
            raise
        finally:
            call.ms = int((time.time() - t0) * 1000)
            if tracker is not None:
                tracker.add(call)

    def model_info(self, role: str = "reader") -> dict:
        """Лимит окна модели роли: из настроек или из каталога OpenRouter (кешируется по модели)."""
        model = self.cfg.role_model(role)
        if model in self._model_infos:
            return self._model_infos[model]
        info = {"model": model, "context_tokens": self.cfg.model_context_tokens or 0, "source": "env"}
        if not info["context_tokens"] and self.cfg.api_key:
            try:
                req = urllib.request.Request(self.cfg.base_url.rstrip("/") + "/models",
                                             headers={"Authorization": f"Bearer {self.cfg.api_key}"})
                with urllib.request.urlopen(req, timeout=20) as resp:
                    rows = json.loads(resp.read().decode("utf-8")).get("data") or []
                row = next((r for r in rows if r.get("id") == model), None)
                if row and row.get("context_length"):
                    info.update(context_tokens=int(row["context_length"]), source="openrouter")
            except Exception as e:  # каталог недоступен — консервативный лимит ниже
                info["error"] = str(e)[:200]
        if not info["context_tokens"]:
            info.update(context_tokens=32000, source="conservative_default")
        self._model_infos[model] = info
        return info

    # ---------------------------------------------------------------- embeddings
    def _emb_path(self, text: str):
        import hashlib
        from pathlib import Path
        key = hashlib.sha256(json.dumps([self.cfg.embedding_model, text], ensure_ascii=False).encode()).hexdigest()
        return Path(self.cfg.data_dir) / "cache" / "emb" / key[:2] / f"{key}.json"

    def embed(self, texts: list[str], tracker: Tracker | None = None, batch: int = 32) -> list[list[float]]:
        """Эмбеддинги с дисковым кешем: повторная подготовка того же текста не платит заново."""
        out: list = [None] * len(texts)
        todo = []
        for i, t in enumerate(texts):
            p = self._emb_path(t) if self.cfg.llm_cache else None
            if p is not None and p.exists():
                try:
                    out[i] = json.loads(p.read_text(encoding="utf-8"))
                    continue
                except (OSError, json.JSONDecodeError):
                    pass
            todo.append(i)
        # пачки эмбеддингов параллельно (большой комплект — тысячи фрагментов), порядок сохраняется
        from concurrent.futures import ThreadPoolExecutor
        parts = [todo[j:j + batch] for j in range(0, len(todo), batch)]
        with ThreadPoolExecutor(max_workers=max(1, min(4, len(parts)))) as ex:
            got = list(ex.map(lambda idx: self._embed_api([texts[i] for i in idx], tracker, batch), parts))
        fresh = [v for part in got for v in part]
        for i, v in zip(todo, fresh):
            out[i] = v
            if self.cfg.llm_cache:
                p = self._emb_path(texts[i])
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix(f".{threading.get_ident()}.tmp")
                tmp.write_text(json.dumps(v), encoding="utf-8")
                tmp.replace(p)
        return out

    def _embed_api(self, texts: list[str], tracker: Tracker | None = None, batch: int = 32) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch):
            part = texts[i:i + batch]
            call = Call(role="embed", model=self.cfg.embedding_model)
            t0 = time.time()
            try:
                data = self._with_retries(lambda: self._post(
                    "/embeddings", {"model": self.cfg.embedding_model, "input": part}), call)
                rows = sorted(data.get("data") or [], key=lambda r: r.get("index", 0))
                if len(rows) != len(part):
                    raise LLMError("protocol", "эмбеддинги вернулись не для всех фрагментов")
                out.extend(r["embedding"] for r in rows)
                usage = data.get("usage") or {}
                call.prompt_tokens = int(usage.get("prompt_tokens") or 0)
                if usage.get("cost") is not None:
                    call.cost = float(usage["cost"])
            except LLMError as e:
                call.ok, call.error = False, str(e)[:300]
                raise
            finally:
                call.ms = int((time.time() - t0) * 1000)
                if tracker is not None:
                    tracker.add(call)
        return out

    def check(self) -> dict:
        """Короткий диагностический запрос по кнопке «Проверить модель»."""
        t = Tracker()
        t0 = time.time()
        res = self.chat_json("diagnostic", "Ответь JSON.", 'Верни {"ok": true}', None, t, max_tokens=20)
        return {"ok": bool(res.get("ok")), "ms": int((time.time() - t0) * 1000), "model": self.cfg.model,
                "models_by_role": self.cfg.models_by_role()}


def model_info_of(llm, role: str = "reader") -> dict:
    """model_info(role); у клиентов без параметра роли — общий лимит."""
    fn = getattr(llm, "model_info", None)
    if fn is None:
        return {"context_tokens": 32000, "source": "conservative_default"}
    try:
        return fn(role)
    except TypeError:
        return fn()


_client: OpenRouter | None = None


def client() -> OpenRouter:
    global _client
    if _client is None:
        _client = OpenRouter()
    return _client


def set_client(c) -> None:
    """Подмена клиента (например, другим провайдером с тем же интерфейсом)."""
    global _client
    _client = c
