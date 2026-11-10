"""HTTP API и раздача интерфейса. Только стандартная библиотека, слушает localhost."""
from __future__ import annotations

import json
import mimetypes
import os
import re
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

from . import voice
from .config import settings, ROOT
from .evaluate import check_integrity, EvalError
from .llm import client, LLMError
from .store import Registry, parse_questions, Conflict409, diagnostics

WEB = ROOT / "web"
registry = Registry(settings.data_dir)
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
MAX_BODY = 130 * 1024 * 1024


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _start_job(fn) -> str:
    jid = uuid.uuid4().hex[:12]
    job = {"id": jid, "state": "running", "stage": "в очереди", "result": None, "error": "", "t0": time.time()}
    with jobs_lock:
        jobs[jid] = job
        # старые задачи чистим, чтобы память не росла
        for k in [k for k, v in jobs.items() if time.time() - v["t0"] > 3600]:
            jobs.pop(k, None)

    def run():
        try:
            job["result"] = fn(lambda s: job.__setitem__("stage", s))
            job["state"] = "done"
        except (LLMError, KeyError, ValueError) as e:
            job["state"], job["error"] = "error", str(e)
        except Exception as e:
            traceback.print_exc()
            job["state"], job["error"] = "error", f"{type(e).__name__}: {e}"

    threading.Thread(target=run, daemon=True).start()
    return jid


class Handler(BaseHTTPRequestHandler):
    server_version = "Opora/1.0"

    def log_message(self, fmt, *args):  # тише в консоли: только ошибки
        if args and str(args[1]).startswith(("4", "5")):
            super().log_message(fmt, *args)

    # ------------------------------------------------------------ helpers
    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, status: int = 200) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8",
                   {"Cache-Control": "no-store"})

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ApiError(413, "слишком большой файл")
        return self.rfile.read(n) if n else b""

    def _body_json(self) -> dict:
        try:
            return json.loads(self._body().decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ApiError(400, "ожидается JSON")

    def _same_origin(self) -> None:
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc not in (self.headers.get("Host"), f"localhost:{settings.port}",
                                                       f"127.0.0.1:{settings.port}"):
            raise ApiError(403, "запрос с чужого сайта отклонён")

    def _route(self, method: str):
        url = urlparse(self.path)
        path, qs = url.path, {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if method != "GET":
                self._same_origin()
            if path.startswith("/api/"):
                return self._api(method, path, qs)
            if method != "GET":
                raise ApiError(405, "метод не поддерживается")
            return self._static(path)
        except ApiError as e:
            self._json({"error": str(e)}, e.status)
        except Conflict409 as e:
            self._json({"error": str(e)}, 409)
        except KeyError as e:
            self._json({"error": str(e).strip("'")}, 404)
        except (ValueError, EvalError, voice.VoiceError) as e:
            self._json({"error": str(e)}, 400)
        except LLMError as e:
            self._json({"error": str(e), "kind": e.kind}, 502)
        except Exception as e:
            traceback.print_exc()
            self._json({"error": f"внутренняя ошибка: {type(e).__name__}"}, 500)

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def do_PUT(self):
        self._route("PUT")

    # ------------------------------------------------------------ static
    def _static(self, path: str) -> None:
        rel = "index.html" if path in ("/", "") else unquote(path.lstrip("/"))
        f = (WEB / rel).resolve()
        if WEB.resolve() not in f.parents and f != WEB.resolve() or not f.is_file():
            raise ApiError(404, "не найдено")
        ctype = mimetypes.guess_type(str(f))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",):
            ctype += "; charset=utf-8"
        cache = "no-cache" if f.suffix in (".html", ".js", ".css") else "max-age=86400"
        self._send(200, f.read_bytes(), ctype, {"Cache-Control": cache})

    # ------------------------------------------------------------ API
    def _api(self, method: str, path: str, qs: dict) -> None:
        p = path.rstrip("/").split("/")[2:]  # после /api/
        m = (method, p)

        if m == ("GET", ["config"]):
            return self._json({**settings.public(), "check_py": check_integrity(), "voice": voice.status()})
        if m == ("GET", ["diagnostics"]):
            return self._json(diagnostics())
        if m == ("POST", ["models", "check"]):
            return self._json(client().check())
        if m == ("POST", ["voice"]):
            return self._json(voice.transcribe(self._body()))
        if p[:1] == ["jobs"] and len(p) == 2 and method == "GET":
            job = jobs.get(p[1])
            if job is None:
                raise ApiError(404, "задача не найдена")
            return self._json({k: v for k, v in job.items() if k != "t0"})

        # ---- снимки закупок
        if m == ("GET", ["collections"]):
            return self._json(registry.list_collections())
        if m == ("POST", ["collections"]):
            return self._json({"id": registry.new_collection().id})
        if p[:1] == ["collections"] and len(p) >= 2:
            c = registry.collection(p[1])
            rest = p[2:]
            if method == "GET" and rest == []:
                with c.lock:
                    c.meta["packages"] = [c.package_summary(pk) for pk in c.packages]
                    return self._json(c.meta)
            if method == "PUT" and rest == ["files"]:
                rel = qs.get("path") or ""
                if not rel:
                    raise ApiError(400, "нет параметра path")
                return self._json(c.add_file(rel, self._body()))
            if method == "POST" and rest == ["ingest"]:
                # тот же снимок уже подготовлен → {"use": id}; прерванная подготовка продолжается
                return self._json(registry.start_ingest(c))
            if method == "GET" and rest == ["source"]:
                return self._json(c.source(qs.get("package", ""), qs.get("file", "")))
            if method == "GET" and rest == ["knowledge"]:
                return self._json(c.knowledge_view(qs.get("package", "")))
            if method == "GET" and rest == ["runs"]:
                return self._json(registry.list_runs(c.id))
            if method == "GET" and rest == ["chat"]:
                return self._json(registry.chat_history(c, qs.get("package", ""), qs.get("session") or "default"))
            if method == "POST" and rest == ["chat"]:
                b = self._body_json()
                q = (b.get("question") or "").strip()
                if not q:
                    raise ApiError(400, "пустой вопрос")
                if not settings.api_key:
                    raise ApiError(400, "Не задан OPENROUTER_API_KEY в файле .env")
                pkg, sess = b.get("package", ""), str(b.get("session") or "default")[:64]
                if pkg not in c.packages:
                    raise ApiError(404, "закупка не найдена")
                opts = b.get("options") or {}
                jid = _start_job(lambda stage: registry.chat(c, pkg, q, sess, opts))
                return self._json({"job": jid})
            if method == "POST" and rest == ["runs"]:
                questions = parse_questions(self._body(), c)
                opts = {k: qs[k] for k in ("read_mode",) if k in qs}
                for k in ("use_passport", "strict_coverage"):
                    if k in qs:
                        opts[k] = qs[k] == "1"
                r = registry.new_run(c, questions, baseline=qs.get("baseline", "1") == "1", options=opts)
                pkgs = sorted({q["package"] for q in questions})
                return self._json({"id": r.id, "count": len(questions), "packages": pkgs,
                                   "questions": questions, "options": r.options})
            raise ApiError(404, "нет такого метода")

        # ---- прогоны
        if m == ("GET", ["runs"]):
            return self._json(registry.list_runs(qs.get("collection")))
        if p[:1] == ["runs"] and len(p) >= 2:
            r = registry.run(p[1])
            rest = p[2:]
            if method == "POST" and rest in (["start"], ["resume"]):
                if not settings.api_key:
                    raise ApiError(400, "Не задан OPENROUTER_API_KEY в файле .env")
                b = self._body_json()
                if r.state == "queued" and "baseline" in b:
                    r.baseline = bool(b["baseline"])
                if rest == ["resume"] and r.state not in ("interrupted", "cancelled"):
                    raise ApiError(409, f"прогон в состоянии {r.state}: продолжать нечего")
                r.start(resume=rest == ["resume"])
                return self._json({"ok": True, "state": r.state})
            if method == "GET" and rest == []:
                return self._json(r.snapshot())
            if method == "POST" and rest == ["cancel"]:
                r.cancelled = True
                return self._json({"ok": True})
            if method == "POST" and rest == ["evaluate"]:
                return self._json(r.evaluate(self._body()))
            if method == "GET" and len(rest) == 2 and rest[0] == "download":
                name = rest[1]
                files = {"answers.csv", "baseline.csv", "trace.json", "answers.json", "baseline.json",
                         "baseline_raw.json", "prompts.json", "run_config.json", "metrics.json", "coverage.json",
                         "calculations.json", "source_manifest.json", "knowledge_versions.json"}
                if name not in files:
                    raise ApiError(404, "нет такого файла")
                f = r.dir / ("evaluation/metrics.json" if name == "metrics.json" else name)
                if name.endswith(".csv") and r.state != "completed":
                    raise ApiError(409, "прогон ещё не завершён")
                if not f.exists():
                    raise ApiError(404, "файл ещё не создан")
                ctype = "text/csv; charset=utf-8" if name.endswith(".csv") else "application/json; charset=utf-8"
                return self._send(200, f.read_bytes(), ctype,
                                  {"Content-Disposition": f'attachment; filename="{name}"'})
            raise ApiError(404, "нет такого метода")
        raise ApiError(404, "нет такого метода")


def serve() -> None:
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")
    mimetypes.add_type("image/svg+xml", ".svg")
    try:
        httpd = ThreadingHTTPServer((settings.host, settings.port), Handler)
    except OSError as e:
        print(f"\n  [!] Порт {settings.port} занят ({e}).\n      Возможно, сервер уже запущен — откройте "
              f"http://localhost:{settings.port}\n      или укажите другой PORT в файле .env\n")
        return
    httpd.daemon_threads = True
    url = f"http://{'localhost' if settings.host in ('127.0.0.1', '0.0.0.0') else settings.host}:{settings.port}"
    if os.environ.get("OPEN_BROWSER") == "1":
        import webbrowser
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()  # сервер уже слушает порт
    print(f"\n  Липецкие котики — ИИ, который не обманывает\n  Откройте в браузере: {url}\n"
          f"  Модель: {settings.model} · эмбеддинги: {settings.embedding_model}\n"
          + "".join(f"    {r}: {m}\n" for r, m in settings.models_by_role().items() if m != settings.model) +
          f"  Критики: {', '.join(settings.enabled_critics()) or 'выключены'} · защита от инъекций: код"
          f"{' + модель' if settings.guard_llm else ''}\n"
          f"  Qdrant: {settings.qdrant_url} · режим чтения: {settings.read_mode}\n"
          f"  Ключ OpenRouter: {'задан' if settings.api_key else 'НЕ ЗАДАН — заполните .env'}\n"
          f"  Остановить: Ctrl+C\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
