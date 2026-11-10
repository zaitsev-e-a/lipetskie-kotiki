"""Оркестрация вопроса на LangGraph.

Узлы: plan_question → select_context → read_windows → resolve_sources → accumulate_evidence →
(select_context …) → resolve_relations → merge_candidates → [judge] → [calculate] → semantic_validate →
critic_panel → final_validate → format_export. Возвраты: к read_windows при ошибке цитаты, к select_context
при нехватке сведений, возражении валидатора или критика с проверенной цитатой. Вопрос, состоящий из команды
модели, сразу уходит в final_validate. Все циклы ограничены счётчиками и бюджетом.

Состояние — сериализуемый словарь (снимок настроек, план, охват, claims, кандидаты, проверки, счётчики).
После каждого узла запись в журнал задачи (journal.jsonl). Тяжёлые объекты (реестр источников, паспорт,
клиент модели) передаются через контекст вопроса, а не через состояние.
"""
from __future__ import annotations

import copy
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Callable, TypedDict

from . import prompts as P
from .calculator import compute, CalcError
from .config import settings
from .coverage import new_coverage, record_read, unvisited, summary as cov_summary, absence_policy
from .formatter import format_answer, normalize_status, question_kind
from .guard import check_question, check_output
from .knowledge import Knowledge, knowledge_context, typed_key
from .llm import LLMError, Tracker, model_info_of
from .retrieval import Retriever, QdrantUnavailable, QdrantError
from .schemas import Window
from .sources import PackageSources, SourceError, estimate_tokens
from .textnorm import numbers, one_line, to_decimal, fmt_decimal
from .validators import verify_claim, claim_key, final_problems, stems

try:  # LangGraph обязателен в рабочем приложении (requirements.txt)
    from langgraph.graph import StateGraph, START, END
except ImportError as _e:  # pragma: no cover - понятная ошибка вместо тихой подмены
    StateGraph = None
    START, END = "__start__", "__end__"
    _LANGGRAPH_ERROR = _e
else:
    _LANGGRAPH_ERROR = None

FULL = "FULL"  # окно полного контекста: все блоки закупки


class QState(TypedDict, total=False):
    question: str
    standalone: str
    qid: str
    plan: dict
    mode: str
    round: int
    batch: list
    feedback: dict
    raw_reads: dict
    retry: list
    coverage: dict
    claims: dict
    source_errors: list
    uncertainties: list
    next_windows: list
    repairs: int
    read_repairs: dict
    amendments: list
    conflict: dict
    ambiguity: list
    groups: list
    candidate: dict
    judge: dict
    merge_attempts: int
    validation: dict
    reason: str
    task_error: dict
    trace: list
    result: dict
    critics: dict


@dataclass
class QuestionCtx:
    ps: PackageSources
    knowledge: Knowledge | None
    retriever: Retriever | None
    llm: Any
    tracker: Tracker = field(default_factory=Tracker)
    history: list = field(default_factory=list)
    journal: Callable[[dict], None] | None = None
    on_stage: Callable[[str], None] | None = None
    t0: float = field(default_factory=time.time)
    use_passport: bool = True
    read_mode: str = "auto"
    strict: bool = True
    top_windows: int = 3          # сколько окон поиска читать в первом раунде (вторая попытка — шире)

    def calls_used(self) -> int:
        s = self.tracker.summary()
        return s["calls"] - s.get("cached", 0) - s.get("by_role", {}).get("embed", 0)

    def calls_left(self) -> int:
        return settings.max_calls_per_question - self.calls_used()

    def time_left(self) -> float:
        return settings.question_deadline - (time.time() - self.t0)

    def stage(self, s: str) -> None:
        if self.on_stage:
            self.on_stage(s)


# ====================================================================== вспомогательное
# служебные слова формулировки ответа, допустимые без подтверждения цитатой
_ANSWER_WORDS = {"требу", "нужна", "нужно", "нужен", "необх", "устан", "указа", "преду", "отсут", "имеет", "требо",
                 "участ", "закуп", "докум", "огран", "налич", "лицен", "сведе", "также", "кроме", "согла", "извещ",
                 "проек", "контр", "договор"[:5], "пункт", "срока", "месяц", "рубле", "проце", "кален", "рабоч"}


def _window(ctx: QuestionCtx, wid: str) -> Window:
    if wid == FULL:
        return Window(window_id=FULL, block_ids=list(ctx.ps.order), files=[d.name for d in ctx.ps.docs],
                      tokens=ctx.ps.total_tokens)
    if wid.startswith(GROUP):
        ws = [ctx.ps.window(x) for x in _members(wid)]
        return Window(window_id=wid, block_ids=[b for w in ws for b in w.block_ids],
                      files=sorted({f for w in ws for f in w.files}), tokens=sum(w.tokens for w in ws))
    return ctx.ps.window(wid)


GROUP = "G:"  # порция из нескольких соседних окон (сплошной обход большого комплекта)


def _members(wid: str) -> list[str]:
    return wid[len(GROUP):].split(",") if wid.startswith(GROUP) else [wid]


def _group_windows(ctx: QuestionCtx, wids: list[str]) -> list[str]:
    """Сплошной обход большого комплекта (доказательство отсутствия, поиск всех сторон): соседние окна
    объединяются в порции до SWEEP_TOKENS — тот же полный охват, но в разы меньше вызовов, и бюджет вызовов
    вопроса не заканчивается на середине комплекта."""
    if len(wids) <= 4:
        return wids
    order = {w.window_id: i for i, w in enumerate(ctx.ps.windows)}
    real = sorted((w for w in wids if w in order), key=lambda w: order[w])
    out, cur, tok = [], [], 0
    for w in real:
        t = ctx.ps.window(w).tokens
        if cur and tok + t > settings.sweep_tokens:
            out.append(cur)
            cur, tok = [], 0
        cur.append(w)
        tok += t
    if cur:
        out.append(cur)
    return [GROUP + ",".join(g) if len(g) > 1 else g[0] for g in out] + [w for w in wids if w not in order]


def _asks_original(q: str) -> bool:
    import re
    return bool(re.search(r"первоначальн|исходн\w+ редакц|до изменени|изначальн|было до", (q or "").lower()))


def _current_answering(claims: dict) -> list[dict]:
    return [c for c in claims.values() if c["answers_question"] and not c.get("superseded")]


def _relevant_facts(ctx: QuestionCtx, question: str, hits: list[dict] | None = None) -> list[dict]:
    """Факты паспорта по вопросу: из Qdrant (kind=fact) в режиме поиска; иначе — подсказка по совпадению
    основ слов подписи атрибута с вопросом (это не поиск-доказательство, а выбор подсказок)."""
    k = ctx.knowledge
    if not (k and ctx.use_passport):
        return []
    if hits:
        ids = [h.get("fact_id") for h in hits if h.get("kind") == "fact"]
        return [f for f in (k.fact(i) for i in ids) if f]
    q = stems(question)
    scored = [(len(q & stems(f["attribute_label"] + " " + f["attribute"].replace("_", " "))), f) for f in k.facts]
    return [f for sc, f in sorted(scored, key=lambda x: -x[0]) if sc > 0][:20]


def _marks(ctx: QuestionCtx, block_ids) -> dict:
    from .knowledge import _marks as km
    return km(ctx.ps, block_ids)


def _claims_text(claims: list[dict]) -> str:
    rows = []
    for c in claims:
        ev = "; ".join(f"[{e['source_id']}] {e['file']}: «{e['quote']}»" for e in c["evidence"])
        cond = f"; условия: {', '.join(c['conditions'])}" if c.get("conditions") else ""
        old = " [НЕДЕЙСТВУЮЩАЯ РЕДАКЦИЯ]" if c.get("superseded") else ""
        rows.append(f"- {c['claim_id']}: {c['attribute_label'] or c['attribute']} = {c['raw_value']} "
                    f"(значение {c['value']}, единица {c['unit']}, полярность {c['polarity']}, "
                    f"{'отвечает на вопрос' if c['answers_question'] else 'вспомогательное'}){cond}{old}\n  {ev}")
    return "\n".join(rows)


# ====================================================================== узлы
def build_graph(ctx: QuestionCtx):
    if StateGraph is None:
        raise RuntimeError(f"LangGraph не установлен: pip install -r requirements.txt ({_LANGGRAPH_ERROR})")

    def node(name: str, fn):
        def run(state: QState) -> dict:
            t = time.time()
            ctx.stage(name)
            upd = fn(state) or {}
            entry = {"node": name, "ms": int((time.time() - t) * 1000), "calls_used": ctx.calls_used(),
                     **(upd.pop("_log", {}) or {})}
            upd["trace"] = list(state.get("trace") or []) + [entry]
            if ctx.journal:
                ctx.journal(entry)
            return upd
        return run

    # ------------------------------------------------------------------ plan_question
    def plan_question(state: QState) -> dict:
        q = state["question"]
        # защита от прямой инъекции: команды модели из вопроса удаляются до любого вызова модели
        qg = check_question(q)
        q = qg["question"]
        standalone = q
        if not q:
            plan = {"kind": "text", "asks_original": False, "blocked": "question_injection", "guard": qg}
            return {"standalone": state["question"], "plan": plan, "mode": ctx.read_mode,
                    "coverage": new_coverage(ctx.ps), "claims": {}, "round": 0, "feedback": {}, "repairs": 0,
                    "read_repairs": {}, "source_errors": [], "uncertainties": [], "next_windows": [],
                    "merge_attempts": 0, "candidate": {"status": "не знаю"}, "reason": "question_injection",
                    "_log": {"guard": "вопрос состоит из команды модели — не отвечаем"}}
        if ctx.history:
            try:
                res = ctx.llm.chat_json("contextualizer", P.CONTEXT_SYSTEM,
                                        "История:\n" + "\n".join(f"{m['role']}: {m['text']}" for m in ctx.history[-6:])
                                        + f"\n\nПоследний вопрос: {q}", P.CONTEXT_SCHEMA, ctx.tracker)
                standalone = one_line(str(res.get("question") or "")) or q
            except LLMError:
                standalone = q
        info = model_info_of(ctx.llm, "reader")
        budget = int(info["context_tokens"] * settings.context_budget_ratio) - settings.role_max_tokens("reader") - 2500
        fits = ctx.ps.total_tokens + estimate_tokens(P.READER_SYSTEM) + 600 <= budget
        mode, switches = ctx.read_mode, []
        if mode == "auto":
            small = ctx.ps.total_tokens <= settings.full_context_max_tokens
            mode = "full_context" if fits and small else "retrieval"
            if fits and not small:
                switches.append(f"auto → retrieval: комплект ~{ctx.ps.total_tokens} токенов больше "
                                f"FULL_CONTEXT_MAX_TOKENS={settings.full_context_max_tokens}")
        if mode == "full_context" and not fits:
            switches.append("full_context → retrieval: комплект не помещается в окно модели")
            mode = "retrieval"
        if mode == "retrieval" and ctx.retriever is None:
            switches.append("retrieval → exhaustive: Qdrant недоступен или закупка не проиндексирована; "
                            "гибридный поиск НЕ выполнялся")
            mode = "exhaustive"
        plan = {"kind": question_kind(standalone), "asks_original": _asks_original(standalone),
                "context_budget_tokens": budget, "package_tokens": ctx.ps.total_tokens, "fits": fits,
                "model_context": info, "strict": ctx.strict, "use_passport": ctx.use_passport,
                "guard": qg if qg["injected"] else None}
        cov = new_coverage(ctx.ps)
        cov["switches"] = switches
        return {"standalone": standalone, "plan": plan, "mode": mode, "coverage": cov, "claims": {}, "round": 0,
                "feedback": {}, "repairs": 0, "read_repairs": {}, "source_errors": [], "uncertainties": [],
                "next_windows": [], "merge_attempts": 0,
                "_log": {"mode": mode, "fits": fits, "switches": switches,
                         **({"guard": f"из вопроса удалены команды модели: {[f['kind'] for f in qg['findings']]}"}
                            if qg["injected"] else {})}}

    # ------------------------------------------------------------------ select_context
    def select_context(state: QState) -> dict:
        mode, cov = state["mode"], copy.deepcopy(state["coverage"])
        log: dict = {}
        if state.get("round", 0) == 0:
            if mode == "full_context":
                batch = [FULL]
            elif mode == "exhaustive":
                batch = [w.window_id for w in ctx.ps.windows]
            else:
                try:
                    res = ctx.retriever.search(ctx.ps, state["standalone"], ctx.tracker, kind="chunk")
                    fres = ctx.retriever.search(ctx.ps, state["standalone"], ctx.tracker, kind="fact", limit=8) \
                        if ctx.use_passport and ctx.knowledge else {"hits": []}
                except (QdrantUnavailable, QdrantError) as e:
                    cov["switches"].append(f"retrieval → exhaustive: {e}")
                    return {"mode": "exhaustive", "coverage": cov, "batch": [w.window_id for w in ctx.ps.windows],
                            "round": 1, "_log": {"switch": str(e)}}
                ranked = []
                for h in res["hits"] + fres["hits"]:
                    for wid in ctx.ps.windows_of_blocks(h.get("block_ids") or []):
                        if wid not in ranked:
                            ranked.append(wid)
                batch = ranked[:ctx.top_windows] or [ctx.ps.windows[0].window_id]
                log = {"hybrid": {"collection": res["collection"], "bm25_mode": res["bm25_mode"],
                                  "rrf_k": res["rrf_k"], "hits": [h.get("chunk_id") or h.get("fact_id")
                                                                  for h in res["hits"]],
                                  "dense_ids": res["dense_ids"][:5], "bm25_ids": res["bm25_ids"][:5],
                                  "fact_hits": [h.get("fact_id") for h in fres["hits"]]},
                       "retrieval_hits": res["hits"][:5]}
        else:
            batch = [w for w in state.get("next_windows") or [] if w in unvisited(cov) or w in state.get("feedback", {})]
        if mode != "full_context":
            batch = _group_windows(ctx, batch)
        # бюджет: окна + резерв на объединение, проверку смысла, проверку отсутствия и судью
        reserve = 4
        left = ctx.calls_left()
        if len(batch) + reserve > left or ctx.time_left() < 20:
            fit = max(0, left - reserve)
            if fit < len(batch):
                cov["switches"].append(f"бюджет: прочитано будет {fit} из {len(batch)} окон")
                batch = batch[:fit]
                return {"batch": batch, "coverage": cov, "round": state.get("round", 0) + 1,
                        "uncertainties": state.get("uncertainties", []) + ["content_budget_exhausted"],
                        "_log": {**log, "batch": batch, "budget_left": left}}
        return {"batch": batch, "coverage": cov, "round": state.get("round", 0) + 1,
                "_log": {**log, "batch": batch}}

    # ------------------------------------------------------------------ read_windows
    def _read_blocks(state: QState, wid: str, block_ids: list[str], depth: int = 0) -> dict:
        facts = _relevant_facts(ctx, state["standalone"], None) if ctx.use_passport else []
        kn_text, kn_blocks = knowledge_context(ctx.ps, ctx.knowledge, facts) if facts else ("", [])
        extra = [b for b in kn_blocks if b not in set(block_ids)]
        # если в окне есть отменённая редакция — действующий текст изменения прикладывается к окну
        for sid in block_ids:
            b = ctx.ps.blocks[sid]
            info = ctx.ps.superseded_info(b.document_id, b.start, b.end)
            for x in (info or {}).get("source_blocks") or []:
                if x not in set(block_ids) and x not in extra and x in ctx.ps.blocks:
                    extra.append(x)
        text = ctx.ps.render(block_ids, _marks(ctx, block_ids))
        if extra:
            kn_text += "Дополнительные фрагменты (факты паспорта и действующие изменения):\n" + ctx.ps.render(extra, _marks(ctx, extra)) + "\n\n"
        fbs = state.get("feedback", {})
        fb = fbs.get(wid) or [x for m in _members(wid) for x in fbs.get(m) or []][:8] or None
        plan = state["plan"]
        plan_txt = ("Вопрос о первоначальной редакции (до изменений)." if plan["asks_original"] else "")
        user = P.READER_USER.format(question=state["standalone"], plan=plan_txt, knowledge=kn_text,
                                    window_id=wid, text=text,
                                    feedback=P.READER_FEEDBACK.format(problems="\n".join(f"- {p}" for p in fb))
                                    if fb else "")
        try:
            raw = ctx.llm.chat_json("reader", P.READER_SYSTEM, user, P.READER_SCHEMA, ctx.tracker)
            if not isinstance(raw, dict):
                raise LLMError("protocol", "ответ reader не объект")
            return {"raw": raw, "allowed": sorted(set(block_ids) | set(extra)), "blocks": block_ids}
        except LLMError as e:
            # finish_reason=length — незавершённый протокол: та же часть текста читается меньшими порциями,
            # а не принимается обрезанный JSON
            if e.kind == "length" and len(block_ids) > 1 and depth < 2 and ctx.calls_left() > 6:
                h = len(block_ids) // 2
                parts = [_read_blocks(state, wid, block_ids[:h], depth + 1),
                         _read_blocks(state, wid, block_ids[h:], depth + 1)]
                bad = [x["error"] for x in parts if "error" in x]
                if bad:
                    return {"error": f"length → разбиение не помогло: {bad[0]}"[:300], "blocks": block_ids}
                outs = [x["raw"].get("outcome") for x in parts]
                claims = [c for x in parts for c in x["raw"].get("claims") or []]
                order = ("supported", "contradiction_candidate", "partial")
                outcome = next((o for o in order if o in outs), None) if claims else None
                if outcome is None:
                    outcome = "uncertain" if "uncertain" in outs or any(o not in P.READER_SCHEMA["properties"][
                        "outcome"]["enum"] for o in outs) else "no_relevant_evidence"
                raw = {"outcome": outcome, "claims": claims,
                       "missing": [m for x in parts for m in x["raw"].get("missing") or []],
                       "uncertainties": [u for x in parts for u in x["raw"].get("uncertainties") or []]}
                return {"raw": raw, "allowed": sorted({a for x in parts for a in x["allowed"]}), "blocks": block_ids,
                        "split": depth + 1}
            return {"error": f"{e.kind}: {e}"[:300], "blocks": block_ids}

    def read_one(state: QState, wid: str) -> tuple[str, dict]:
        return wid, _read_blocks(state, wid, list(_window(ctx, wid).block_ids))

    def read_windows(state: QState) -> dict:
        batch = state.get("batch") or []
        with ThreadPoolExecutor(max_workers=max(1, min(len(batch), settings.global_concurrency))) as ex:
            new = dict(ex.map(lambda wid: read_one(state, wid), batch))
        results = {**(state.get("raw_reads") or {}), **new}  # при перечитывании прочие окна не теряются
        return {"raw_reads": results, "_log": {"read": batch,
                                               "errors": {k: v["error"] for k, v in results.items() if "error" in v}}}

    # ------------------------------------------------------------------ resolve_sources
    def resolve_sources(state: QState) -> dict:
        reads = state.get("raw_reads") or {}
        plan = state["plan"]
        verified: dict[str, list] = {}
        problems: dict[str, list] = {}
        src_errors = list(state.get("source_errors") or [])
        for wid, r in reads.items():
            if "error" in r:
                continue
            raw = r["raw"]
            out = []
            for rc in raw.get("claims") or []:
                if not isinstance(rc, dict):
                    continue
                c, errs = verify_claim(ctx.ps, rc, set(r["allowed"]), wid, plan["asks_original"])
                if errs:
                    src_errors += [{"window": wid, "error": e} for e in errs]
                    if rc.get("answers_question") or c is None:
                        problems.setdefault(wid, []).extend(errs)
                if c is not None:
                    out.append(c)
            verified[wid] = out
        repairs = dict(state.get("read_repairs") or {})
        retry = [w for w in problems if repairs.get(w, 0) < settings.repair_rounds and ctx.calls_left() > 4]
        upd = {"source_errors": src_errors, "retry": retry,
               "_log": {"verified": {k: len(v) for k, v in verified.items()},
                        "problems": {k: v[:3] for k, v in problems.items()}, "retry": retry}}
        if retry:
            fb = dict(state.get("feedback") or {})
            for w in retry:
                fb[w] = problems[w][:6]
                repairs[w] = repairs.get(w, 0) + 1
            # подтверждённые claims других окон сохраняются в raw_reads; окно с ошибкой перечитывается
            upd.update(feedback=fb, read_repairs=repairs, batch=retry)
        return upd

    # ------------------------------------------------------------------ accumulate_evidence
    def accumulate_evidence(state: QState) -> dict:
        claims = copy.deepcopy(state.get("claims") or {})
        cov = copy.deepcopy(state["coverage"])
        reads = state.get("raw_reads") or {}
        seen = {claim_key(c) for c in claims.values()}
        added = 0
        uncert = list(state.get("uncertainties") or [])
        fb = dict(state.get("feedback") or {})
        for wid, r in reads.items():
            if "error" in r:
                if wid == FULL:
                    record_read(cov, FULL, [], None, 0, [], error=r["error"])
                else:
                    for m in _members(wid):
                        record_read(cov, m, ctx.ps.window(m).block_ids if wid.startswith(GROUP) else r["blocks"],
                                    None, 0, [], error=r["error"])
                continue
            raw = r["raw"]
            ok = []
            for rc in raw.get("claims") or []:
                if not isinstance(rc, dict):
                    continue
                c, _ = verify_claim(ctx.ps, rc, set(r["allowed"]), wid, state["plan"]["asks_original"])
                if c is not None:
                    ok.append(c)
            for c in ok:
                k = claim_key(c)
                if k in seen:  # перекрытие или повтор — не новое подтверждение
                    continue
                seen.add(k)
                c["claim_id"] = f"C{len(claims) + 1:03d}"
                claims[c["claim_id"]] = c
                added += 1
            outcome = raw.get("outcome") if raw.get("outcome") in P.READER_SCHEMA["properties"]["outcome"]["enum"] \
                else "uncertain"
            if wid == FULL:
                for w in ctx.ps.windows:
                    record_read(cov, w.window_id, w.block_ids, outcome, len(ok), raw.get("uncertainties") or [])
            elif wid.startswith(GROUP):
                for m in _members(wid):
                    record_read(cov, m, ctx.ps.window(m).block_ids, outcome, len(ok), raw.get("uncertainties") or [])
            else:
                record_read(cov, wid, r["blocks"], outcome, len(ok), raw.get("uncertainties") or [])
            if outcome == "uncertain":
                uncert.append(f"{wid}: " + "; ".join(raw.get("uncertainties") or ["окно не понято уверенно"]))
            for m in _members(wid):
                fb.pop(m, None)
            fb.pop(wid, None)
        nxt = _next_windows(state, claims, cov)
        return {"claims": claims, "coverage": cov, "uncertainties": uncert, "next_windows": nxt, "feedback": fb,
                "raw_reads": {}, "_log": {"added_claims": added, "next": nxt}}

    def _next_windows(state: QState, claims: dict, cov: dict) -> list[str]:
        """Строгий охват: положительный ответ не завершает анализ — проверяем альтернативы, поправки и
        конфликты; если полноту проверки нельзя обосновать паспортом — полный обход."""
        mode = state["mode"]
        left = unvisited(cov)
        if not left or mode == "full_context":
            return []
        answering = _current_answering(claims)
        if not answering:
            return left  # для отсутствия нужно прочитать всё
        if mode == "exhaustive":
            return left
        k = ctx.knowledge
        if not ctx.strict:
            return []
        if not (k and ctx.use_passport) or any(v.get("state") != "done" for v in k.windows.values()):
            return left  # без полного паспорта полноту альтернатив не обосновать
        attrs = {c["attribute"] for c in answering}
        blocks = set()
        for f in k.related_facts(attrs):
            blocks.add(f["evidence"]["source_id"])
        for c in k.conflicts:
            if c["attribute"] in attrs:
                blocks |= {s["source_id"] for s in c["sides"]}
        ev_blocks = {e["source_id"] for c in answering for e in c["evidence"]}
        for a in k.amendments:
            tb, sb = set(a.get("target_blocks") or []), set(a.get("source_blocks") or [])
            if (tb | sb) & ev_blocks or a.get("state") != "resolved":
                blocks |= tb | sb
        return [w for w in ctx.ps.windows_of_blocks(blocks) if w in left]

    # ------------------------------------------------------------------ resolve_relations
    def resolve_relations(state: QState) -> dict:
        claims = state.get("claims") or {}
        answering = _current_answering(claims)
        unresolved = []
        if ctx.knowledge:
            for a in ctx.knowledge.amendments:
                if a.get("state") == "resolved":
                    continue
                text = a.get("source_text") or ""
                if not answering or any(stems(text) & stems(c["attribute_label"]) for c in answering):
                    unresolved.append({"source_file": a.get("source_file"), "text": text[:200],
                                       "state": a.get("state")})
        groups: dict[tuple, list[str]] = {}
        for c in answering:
            groups.setdefault(tuple(c["typed"]), []).append(c["claim_id"])
        glist = [{"key": list(k), "claim_ids": v,
                  "values": sorted({claims[i]["value"] for i in v}),
                  "attributes": sorted({claims[i]["attribute"] for i in v})} for k, v in groups.items()]
        conflict, ambiguity = {}, []
        # атрибуты от модели — свободный текст («гарантийный срок» / «гарантийный срок на результат работ»):
        # одна группа = та же единица и тот же атрибут ИЛИ подписи, совпадающие по основам слов не меньше чем наполовину
        def _same_attr(a: dict, b: dict) -> bool:
            if a["attribute"] == b["attribute"]:
                return True
            sa = stems(a["attribute_label"] + " " + a["attribute"].replace("_", " "))
            sb = stems(b["attribute_label"] + " " + b["attribute"].replace("_", " "))
            return bool(sa and sb) and len(sa & sb) * 2 >= min(len(sa), len(sb))
        clusters: list[tuple[tuple, list]] = []
        for g in glist:
            c0 = claims[g["claim_ids"][0]]
            for (attr, unit), gs in clusters:
                if unit == c0["unit"] and unit not in ("text", "bool", "") and \
                        _same_attr(claims[gs[0]["claim_ids"][0]], c0):
                    gs.append(g)
                    break
            else:
                clusters.append(((c0["attribute"], c0["unit"]), [g]))
        for (attr, unit), gs in clusters:
            if len(gs) < 2:
                continue  # разные атрибуты — части составного ответа, а не неоднозначность
            conds = {tuple(sorted(claims[g["claim_ids"][0]].get("conditions") or [])) for g in gs}
            attrs0 = {claims[i]["attribute"] for g in gs for i in g["claim_ids"]}
            # «разные условия» — только если это установил паспорт (сопоставимость проверена отдельно);
            # формулировки условий от читателя («с даты приёмки») сами по себе не снимают противоречие
            diff_cond = any(c["attribute"] in attrs0 and c["state"] == "different_conditions"
                            for c in (ctx.knowledge.conflicts if ctx.knowledge else []))
            if (len(conds) == 1 or not diff_cond) and not conflict:
                blocks = {claims[i]["evidence"][0]["source_id"] for g in gs for i in g["claim_ids"]}
                attrs = {claims[i]["attribute"] for g in gs for i in g["claim_ids"]}
                pk = next((c for c in (ctx.knowledge.conflicts if ctx.knowledge else [])
                           if c["attribute"] in attrs and c["state"] == "confirmed"), None)
                pk = pk or next((c for c in (ctx.knowledge.conflicts if ctx.knowledge else [])
                                 if c["attribute"] in attrs), None)
                conflict = {"attribute": attr, "unit": unit, "sides": gs, "distinct_places": len(blocks),
                            "passport_state": pk["state"] if pk else None,
                            "state": "confirmed" if pk and pk["state"] == "confirmed" else "candidate"}
            else:
                ambiguity += gs  # одинаковый атрибут, разные условия — нужна явная развязка
        applied = []
        for c in claims.values():
            for e in c["evidence"]:
                if e.get("superseded_by"):
                    applied.append({"claim_id": c["claim_id"], "file": e["file"], **e["superseded_by"]})
        for i, g in enumerate(ambiguity):
            g["gid"] = f"G{i + 1}"
        return {"amendments": applied, "groups": glist, "conflict": conflict, "ambiguity": ambiguity,
                "uncertainties": list(state.get("uncertainties") or []) +
                                 [f"unresolved_amendment: {u['text']}" for u in unresolved],
                "_log": {"groups": len(glist), "conflict": bool(conflict), "unresolved_amendments": len(unresolved)}}

    # ------------------------------------------------------------------ merge_candidates
    def merge_candidates(state: QState) -> dict:
        claims = state.get("claims") or {}
        glist = list(state.get("groups") or [])
        conflict = state.get("conflict") or {}
        judge = state.get("judge") or {}
        amb = state.get("ambiguity") or []
        if judge.get("choice") is not None:
            drop = {tuple(g["key"]) for g in amb if g.get("gid") != judge["choice"]}
            glist = [g for g in glist if tuple(g["key"]) not in drop]
        answering = _current_answering(claims)
        if not claims:
            return _absence_or_unknown(state, claims, ignore_partials=False)
        # неоднозначность после детерминированного объединения: разные атрибуты/условия — к судье
        if amb and judge.get("choice") is None and not judge.get("done"):
            return {"candidate": {"route": "judge"}, "_log": {"to_judge": len(amb)}}
        if judge.get("done") and judge.get("choice") is None:
            return {"candidate": {"status": "не знаю"}, "reason": "unresolved_conflict",
                    "_log": {"judge": "insufficient"}}
        analysis = []
        if conflict:
            analysis.append(f"конфликт значений атрибута {conflict['attribute']} "
                            f"({conflict['state']}, мест: {conflict['distinct_places']}): "
                            + " | ".join(", ".join(g["values"]) for g in conflict["sides"]))
        elif glist:
            analysis.append("значения, отвечающие на вопрос: " + " | ".join(", ".join(g["values"]) for g in glist))
        else:
            analysis.append("нет утверждений, прямо отвечающих на вопрос; возможен расчёт из вспомогательных")
        if state["plan"]["asks_original"]:
            analysis.append("вопрос о первоначальной редакции")
        use = [claims[i] for g in glist for i in g["claim_ids"]] if glist else []
        use += [c for c in claims.values() if c not in use]
        fb = state.get("candidate", {}).get("problems") if state.get("merge_attempts") else None
        user = P.MERGE_USER.format(question=state["standalone"], analysis="; ".join(analysis),
                                   claims=_claims_text(use)) + (f"\nЗамечания к прошлому варианту: {fb}" if fb else "")
        try:
            m = ctx.llm.chat_json("merge", P.MERGE_SYSTEM, user, P.MERGE_SCHEMA, ctx.tracker)
        except LLMError as e:
            return {"candidate": {"status": "не знаю"}, "reason": "validation_incomplete",
                    "task_error": {"stage": "merge", "error": f"{e.kind}: {e}", "retryable": True},
                    "_log": {"merge_error": str(e)}}
        cand, probs = _check_merge(m, claims, conflict, state)
        attempts = state.get("merge_attempts", 0) + 1
        if probs:
            if attempts <= 1 and ctx.calls_left() > 3:
                return {"candidate": {"status": None, "problems": "; ".join(probs), "route": "merge"},
                        "merge_attempts": attempts, "_log": {"merge_problems": probs}}
            reason = "unresolved_conflict" if any("расхождение" in p for p in probs) else "insufficient_evidence"
            return {"candidate": {"status": "не знаю"}, "reason": reason, "merge_attempts": attempts,
                    "_log": {"merge_problems": probs}}
        if cand["status"] in ("нет в документах", "не знаю") and not answering:
            # claims есть, но ни одно не отвечает на вопрос: решение об отсутствии принимает политика охвата
            return _absence_or_unknown({**state, "merge_attempts": attempts}, claims, ignore_partials=True)
        return {"candidate": cand, "merge_attempts": attempts, "_log": {"merge": cand.get("status")}}

    def _absence_or_unknown(state: QState, claims: dict, ignore_partials: bool) -> dict:
        unresolved = [u for u in state.get("uncertainties") or [] if u.startswith("unresolved_amendment")]
        reasons = absence_policy(ctx.ps, state["coverage"], [] if ignore_partials else list(claims.values()),
                                 ctx.knowledge if ctx.use_passport else None, unresolved)
        unc = [u for u in state.get("uncertainties") or [] if not u.startswith("unresolved_amendment")]
        if state.get("source_errors"):
            reasons.append(f"insufficient_evidence: модель указала сведения, но {len(state['source_errors'])} "
                           "цитат(ы) не подтвердились по источнику — отсутствие не доказано")
        if any(u.startswith("content_budget_exhausted") for u in unc):
            reasons.append("content_budget_exhausted: бюджет вызовов исчерпан до полного чтения")
        if reasons:
            code = reasons[0].split(":")[0]
            return {"candidate": {"status": "не знаю", "absence_blocked": reasons}, "reason": code,
                    "_log": {"absence_blocked": reasons}}
        return {"candidate": {"status": "нет в документах", "claim_ids": []}, "_log": {"absence": "policy_ok"}}

    def _check_merge(m: dict, claims: dict, conflict: dict, state: QState) -> tuple[dict, list[str]]:
        probs = []
        st = normalize_status(m.get("status"))
        if st is None:
            probs.append(f"статус {m.get('status')!r} недопустим")
        ids = [i for i in m.get("claim_ids") or [] if i in claims]
        if st in ("ответ", "расхождение") and not ids:
            probs.append("ответ без claim_ids")
        calc = m.get("calculation") or {"operation": "none"}
        op = calc.get("operation") or "none"
        if st == "расхождение" and not conflict:
            probs.append("расхождение без подтверждённого конфликта значений")
        if st == "ответ" and op == "none" and ids:
            # текстовый ответ — формулировками документа: каждое значимое слово ответа должно быть в цитатах
            # (или в вопросе); сокращения и пересказ своими словами отклоняются
            src = set()
            for i in ids:
                src |= stems(" ".join(e["quote"] for e in claims[i]["evidence"]))
            extra = stems(str(m.get("answer") or "")) - src - stems(state["standalone"]) - _ANSWER_WORDS
            if extra:
                probs.append("в ответе слова, которых нет в цитатах: " + ", ".join(sorted(extra)[:6])
                             + " — используй формулировки документа дословно, без сокращений и пересказа")
        if st == "ответ" and op == "none":
            allowed = set()
            for i in ids:
                allowed |= set(numbers(claims[i]["value"])) | set(numbers(claims[i]["raw_value"]))
            vals = set()
            for v in m.get("values") or []:
                vals |= set(numbers(v))
            miss = [v for v in vals if all(abs(v - a) > 1e-6 for a in allowed)]
            if miss and not any(claims[i]["polarity"] == "negative" for i in ids):
                probs.append(f"числа {', '.join(f'{x:g}' for x in miss)} не взяты из claims")
        if op != "none":
            oids = [i for i in calc.get("operand_claim_ids") or [] if i in claims]
            if len(oids) < 2:
                probs.append("расчёт: нужны два операнда из claims")
            calc = {**calc, "operand_claim_ids": oids}
        if st == "нет в документах" and _current_answering(claims):
            probs.append("есть утверждения, отвечающие на вопрос — отсутствие недопустимо")
        if st == "ответ" and op == "none" and conflict and conflict.get("state") == "confirmed":
            # подтверждённое расхождение действующих сторон: выбор одной стороны ничем не обоснован
            side_of = {cid: i for i, sd in enumerate(conflict.get("sides") or []) for cid in sd["claim_ids"]}
            if len({side_of[i] for i in ids if i in side_of}) < len(conflict.get("sides") or []):
                probs.append("по атрибуту есть подтверждённое расхождение действующих редакций — нужно "
                             "«расхождение» со всеми сторонами, а не одно значение")
        cand = {"status": st, "answer": one_line(str(m.get("answer") or "")), "values": m.get("values") or [],
                "claim_ids": ids, "calculation": calc if op != "none" else None,
                "reason": one_line(str(m.get("reason") or ""))[:300]}
        return cand, probs

    # ------------------------------------------------------------------ judge
    def judge(state: QState) -> dict:
        claims = state.get("claims") or {}
        glist = state.get("ambiguity") or []
        desc = []
        for g in glist:
            cs = [claims[i] for i in g["claim_ids"]]
            desc.append(f"{g['gid']}: значения {', '.join(g['values'])}\n" + _claims_text(cs))
        try:
            res = ctx.llm.chat_json("judge", P.JUDGE_SYSTEM,
                                    P.JUDGE_USER.format(question=state["standalone"], candidates="\n".join(desc)),
                                    P.JUDGE_SCHEMA, ctx.tracker)
            choice = str(res.get("choice") or "").strip()
            ok = {g["gid"] for g in glist}
            ch = next((c for c in ok if c.lower() == choice.lower()), None)
            return {"judge": {"done": True, "choice": ch, "reason": one_line(str(res.get("reason") or ""))[:300],
                              "raw": choice}, "_log": {"judge": ch or "insufficient"}}
        except LLMError as e:
            return {"judge": {"done": True, "choice": None, "reason": f"судья недоступен: {e.kind}"},
                    "task_error": {"stage": "judge", "error": str(e), "retryable": True},
                    "_log": {"judge_error": str(e)}}

    # ------------------------------------------------------------------ calculate
    def calculate(state: QState) -> dict:
        cand = dict(state["candidate"])
        claims = state.get("claims") or {}
        calc = cand.get("calculation") or {}
        ops = [claims[i] for i in calc.get("operand_claim_ids") or []]
        probs = []
        for c in ops:
            if c.get("superseded"):
                probs.append(f"операнд {c['claim_id']} из недействующей редакции")
        if calc.get("operation") == "percent_of" and len(ops) >= 2:
            if ops[1]["unit"] != "percent":
                probs.append("второй операнд percent_of должен быть процентом")
            if ops[0]["unit"] == "percent":
                probs.append("база процента не может быть процентом")
        vals = [to_decimal(c["value"]) for c in ops]
        if any(v is None for v in vals):
            probs.append("операнд не число")
        if probs:
            return {"candidate": {"status": "не знаю"}, "reason": "insufficient_evidence", "_log": {"calc": probs}}
        try:
            res, expr = compute(calc["operation"], [fmt_decimal(v) for v in vals])
        except CalcError as e:
            return {"candidate": {"status": "не знаю"}, "reason": "insufficient_evidence", "_log": {"calc": str(e)}}
        unit = "rub" if any(c["unit"] == "rub" for c in ops) else ops[0]["unit"]
        cand.update(values=[res], answer=res, claim_ids=sorted(set(cand.get("claim_ids") or []) |
                                                               {c["claim_id"] for c in ops}),
                    calculation={"operation": calc["operation"], "expression": expr, "result": res, "unit": unit,
                                 "rounding": "ROUND_HALF_UP 0.01",
                                 "operands": [{"claim_id": c["claim_id"], "value": c["value"], "unit": c["unit"],
                                               "evidence": c["evidence"][0]} for c in ops],
                                 "base_from_question": calc.get("base_from_question") or ""})
        return {"candidate": cand, "_log": {"calc": expr}}

    # ------------------------------------------------------------------ semantic_validate
    def semantic_validate(state: QState) -> dict:
        cand = dict(state["candidate"])
        claims = state.get("claims") or {}
        if cand["status"] == "нет в документах":
            return _validate_absence(state, cand)
        used = [claims[i] for i in cand.get("claim_ids") or [] if i in claims]
        ctx_blocks = sorted({e["source_id"] for c in used for e in c["evidence"]})
        calc = cand.get("calculation")
        user = P.VALIDATOR_USER.format(
            question=state["standalone"], status=cand["status"], answer=cand.get("answer", ""),
            calculation=(f"Расчёт кода: {calc['expression']}\n" if calc else ""),
            amendments="; ".join(f"{a['file']} → {a['new_text'][:120]}" for a in state.get("amendments") or [])
            or "нет", claims=_claims_text(used), context=ctx.ps.render(ctx_blocks, _marks(ctx, ctx_blocks)))
        res = None
        for _ in range(2):  # ограниченный технический повтор
            try:
                res = ctx.llm.chat_json("validator", P.VALIDATOR_SYSTEM, user, P.VALIDATOR_SCHEMA, ctx.tracker)
                if isinstance(res, dict):
                    break
            except LLMError as e:
                res = {"_error": f"{e.kind}: {e}"}
        if not isinstance(res, dict) or "_error" in res or "answer_verdict" not in res:
            err = (res or {}).get("_error", "ответ валидатора не разобран") if isinstance(res, dict) else "ответ не объект"
            return {"validation": {"ok": None, "status": "unavailable", "detail": err},
                    "candidate": {"status": "не знаю"}, "reason": "validation_incomplete",
                    "task_error": {"stage": "validator", "error": err, "retryable": True},
                    "_log": {"validator": "unavailable"}}
        per = {str(x.get("claim_id")): x for x in res.get("claims") or [] if isinstance(x, dict)}
        bad = [i for i in cand.get("claim_ids") or [] if (per.get(i) or {}).get("verdict") != "supported"]
        if res.get("answer_verdict") == "supported" and bad:
            # самоисправление вместо перечитывания: ответ в целом подтверждён, а «недостаточными» названы лишние
            # утверждения (дубль, отменённая редакция, придуманное условие) — убираем их из основания ответа, если
            # оставшиеся подтверждённые покрывают все значения ответа и все стороны расхождения
            pruned = _prune_unsupported(cand, claims, per, state.get("conflict") or {})
            if pruned is not None:
                return {"validation": {"ok": True, "status": "supported", "detail": res.get("answer_reason", ""),
                                       "claims": per, "pruned": [i for i in cand["claim_ids"]
                                                                 if i not in pruned["claim_ids"]]},
                        "candidate": pruned, "_log": {"validator": "supported", "pruned": bad}}
        if res.get("answer_verdict") == "supported" and not bad:
            return {"validation": {"ok": True, "status": "supported", "detail": res.get("answer_reason", ""),
                                   "claims": per}, "_log": {"validator": "supported"}}
        # возражение: контрдоказательства проверяются так же строго; затем повторное чтение окон
        counter_windows = set()
        for x in per.values():
            for ev in x.get("counter_evidence") or []:
                try:
                    ref = ctx.ps.resolve(str(ev.get("source_id", "")), str(ev.get("quote", "")))
                    counter_windows |= set(ctx.ps.windows_of_blocks([ref.source_id]))
                except SourceError:
                    continue
        reasons = [f"{i}: {(per.get(i) or {}).get('verdict', 'нет оценки')} — {(per.get(i) or {}).get('reason', '')}"
                   for i in bad] or [res.get("answer_reason", "")]
        repairs = state.get("repairs", 0)
        if repairs < settings.repair_rounds and ctx.calls_left() > 4:
            own = set(ctx.ps.windows_of_blocks(ctx_blocks)) if state["mode"] != "full_context" else {FULL}
            wins = sorted(own | counter_windows) if state["mode"] != "full_context" else [FULL]
            fb = dict(state.get("feedback") or {})
            for w in wins:
                fb[w] = [f"валидатор: {r}" for r in reasons][:6]
            # отклонённые утверждения убираются, окна перечитываются с замечаниями
            # удаляются только утверждения, которые валидатор признал ПРОТИВОРЕЧАЩИМИ источнику; «недостаточно» —
            # повод перечитать с замечаниями, но не основание терять сторону расхождения или операнд
            contra = {i for i in bad if (per.get(i) or {}).get("verdict") == "contradicted"}
            keep = {k: v for k, v in claims.items() if k not in contra}
            cov = copy.deepcopy(state["coverage"])
            return {"validation": {"ok": False, "status": res.get("answer_verdict"), "detail": "; ".join(reasons)},
                    "claims": keep, "feedback": fb, "next_windows": wins, "repairs": repairs + 1,
                    "candidate": {"route": "reread"}, "coverage": cov, "judge": {}, "merge_attempts": 0,
                    "_log": {"validator": res.get("answer_verdict"), "reread": wins}}
        return {"validation": {"ok": False, "status": res.get("answer_verdict"), "detail": "; ".join(reasons)},
                "candidate": {"status": "не знаю"}, "reason": "insufficient_evidence",
                "_log": {"validator": res.get("answer_verdict"), "final": "не знаю"}}

    def _prune_unsupported(cand: dict, claims: dict, per: dict, conflict: dict) -> dict | None:
        if any((per.get(i) or {}).get("verdict") == "contradicted" for i in cand.get("claim_ids") or []):
            return None
        keep = [i for i in cand.get("claim_ids") or [] if (per.get(i) or {}).get("verdict") == "supported"]
        if not keep:
            return None
        calc = cand.get("calculation")
        if calc:
            ops = {o["claim_id"] for o in calc.get("operands") or []}
            if not ops <= set(keep):
                return None
        else:
            have = set()
            for i in keep:
                have |= set(numbers(claims[i]["value"])) | set(numbers(claims[i]["raw_value"]))
            need = set()
            for v in cand.get("values") or []:
                need |= set(numbers(str(v)))
            if any(all(abs(n - h) > 1e-6 for h in have) for n in need):
                return None
            if not any(claims[i]["answers_question"] and not claims[i].get("superseded") for i in keep):
                return None
        if cand["status"] == "расхождение":
            sides = conflict.get("sides") or []
            if any(not (set(sd["claim_ids"]) & set(keep)) for sd in sides):
                return None
        return {**cand, "claim_ids": keep}

    # ------------------------------------------------------------------ critic_panel
    def critic_panel(state: QState) -> dict:
        from .critics import code_critics, run_llm_critics, decide
        cand = dict(state["candidate"])
        claims = state.get("claims") or {}
        if cand.get("status") not in ("ответ", "расхождение"):
            return {"_log": {"critics": "не применимо"}}
        used = [claims[i] for i in cand.get("claim_ids") or [] if i in claims]
        kind = state["plan"]["kind"]
        code = code_critics(ctx.ps, state["standalone"], kind, cand, used,
                            ctx.knowledge if ctx.use_passport else None, state["plan"]["asks_original"])
        llm_res = run_llm_critics(ctx, state, cand, used, _claims_text(used), kind) if settings.use_critic else {}
        dec = decide(code, llm_res)
        panel = {"code": code, "llm": llm_res, "hard": dec["hard"], "soft": dec["soft"]}
        log = {"critics_code_failed": [k for k, v in code.items() if v.get("ok") is False],
               "critics_llm": {k: v.get("verdict") for k, v in llm_res.items()}}
        if not dec["reread"]:
            return {"critics": panel, "_log": {**log, "critics": "ok"}}
        repairs = state.get("repairs", 0)
        items = dec["hard"] or dec["soft"]
        sig = [cand.get("answer"), dec["signature"]]
        stalled = sig == (state.get("critics") or {}).get("signature")  # то же возражение к тому же ответу
        panel["signature"] = sig
        if not stalled and repairs < settings.repair_rounds and ctx.calls_left() > 4:
            ev_blocks = [e["source_id"] for c in used for e in c["evidence"]]
            extra = [ev["source_id"] for it in items for ev in it.get("evidence") or []] + \
                    [b for it in items for b in it.get("blocks") or []]
            if state["mode"] == "full_context":
                wins = [FULL]
            else:
                wins = sorted(set(ctx.ps.windows_of_blocks(ev_blocks + extra)) |
                              {w for it in items for w in it.get("windows") or []})
            notes = []
            for it in items:
                ev = "; ".join(f"[{e['source_id']}] «{e['quote'][:160]}»" for e in it.get("evidence") or [])
                notes.append(f"критик {it['critic']}: {it['reason'][:200]}" + (f" — см. {ev}" if ev else ""))
            fb = dict(state.get("feedback") or {})
            for w in wins:
                fb[w] = notes[:6]
            return {"critics": panel, "feedback": fb, "next_windows": wins, "repairs": repairs + 1,
                    "candidate": {"route": "reread"}, "judge": {}, "merge_attempts": 0,
                    "_log": {**log, "critics": "reread", "reread": wins}}
        if dec["hard"] and dec["veto"]:
            return {"critics": panel, "candidate": {"status": "не знаю"}, "reason": "critic_objection",
                    "_log": {**log, "critics": "objection_final"}}
        if dec["hard"]:
            panel["overruled"] = [h["critic"] for h in dec["hard"]]
            return {"critics": panel, "_log": {**log, "critics": "overruled"}}
        return {"critics": panel, "_log": {**log, "critics": "soft_only"}}

    def _near_facts(state: QState) -> list[dict]:
        """Факты паспорта, подпись которых целиком входит в вопрос: проверка отсутствия обязана их рассмотреть."""
        if not (ctx.use_passport and ctx.knowledge):
            return []
        qs = stems(state["standalone"])
        out = []
        for f in ctx.knowledge.facts:
            ls = stems(f["attribute_label"] or f["attribute"].replace("_", " "))
            if not f.get("superseded") and len(ls) >= 2 and ls <= qs:
                out.append(f)
        return out[:12]

    def _validate_absence(state: QState, cand: dict) -> dict:
        """Отдельная смысловая проверка отсутствия: весь текст пакетами по окнам + близкие факты паспорта."""
        near = _near_facts(state)
        hint = ("\nФакты паспорта, похожие на ответ (проверь каждый: если он отвечает на вопрос — verdict found "
                "с дословной цитатой из текста):\n" + "\n".join(
                    f"- [{f['evidence']['source_id']}] {f['attribute_label']}: {f['raw_value']} — «{f['evidence']['quote'][:200]}»"
                    for f in near) + "\n") if near else ""
        if state["mode"] == "full_context":
            parts = [list(ctx.ps.order)]
        else:
            parts = [_window(ctx, g).block_ids for g in _group_windows(ctx, [w.window_id for w in ctx.ps.windows])]
        cs = cov_summary(ctx.ps, state["coverage"])
        found = []
        for i, blocks in enumerate(parts, 1):
            user = P.ABSENCE_USER.format(question=state["standalone"],
                                         coverage=f"прочитано окон {cs['windows_read']} из {cs['windows_total']}, "
                                                  f"блоков {cs['blocks_read']} из {cs['blocks_total']}",
                                         part=i, parts=len(parts), text=ctx.ps.render(blocks, _marks(ctx, blocks))) + hint
            res = None
            for _ in range(2):
                try:
                    res = ctx.llm.chat_json("absence", P.ABSENCE_SYSTEM, user, P.ABSENCE_SCHEMA, ctx.tracker)
                    break
                except LLMError as e:
                    res = {"_error": str(e)}
            if not isinstance(res, dict) or "_error" in res:
                return {"validation": {"ok": None, "status": "unavailable", "detail": "проверка отсутствия не выполнена"},
                        "candidate": {"status": "не знаю"}, "reason": "validation_incomplete",
                        "task_error": {"stage": "absence", "error": str((res or {}).get("_error")), "retryable": True},
                        "_log": {"absence": "unavailable"}}
            if res.get("verdict") == "uncertain":
                return {"validation": {"ok": False, "status": "uncertain", "detail": res.get("reason", "")},
                        "candidate": {"status": "не знаю"}, "reason": "validation_incomplete",
                        "_log": {"absence": "uncertain"}}
            if res.get("verdict") == "found":
                for ev in res.get("found") or []:
                    try:
                        ref = ctx.ps.resolve(str(ev.get("source_id", "")), str(ev.get("quote", "")), set(blocks))
                        found.append(ref)
                    except SourceError:
                        continue
                if not found:  # «нашёл», но без проверяемой цитаты — согласия с отсутствием нет
                    return {"validation": {"ok": False, "status": "found_unverified", "detail": res.get("reason", "")},
                            "candidate": {"status": "не знаю"}, "reason": "validation_incomplete",
                            "_log": {"absence": "found_unverified"}}
        if found:
            repairs = state.get("repairs", 0)
            wins = [FULL] if state["mode"] == "full_context" else sorted(
                set(ctx.ps.windows_of_blocks([f.source_id for f in found])))
            if repairs < settings.repair_rounds and ctx.calls_left() > 4:
                fb = dict(state.get("feedback") or {})
                for w in wins:
                    fb[w] = [f"проверка отсутствия нашла сведения: [{f.source_id}] «{f.quote[:150]}»" for f in found][:4]
                return {"validation": {"ok": False, "status": "found", "detail": "найдены сведения"},
                        "feedback": fb, "next_windows": wins, "repairs": repairs + 1, "candidate": {"route": "reread"},
                        "merge_attempts": 0, "_log": {"absence": "found", "reread": wins}}
            return {"candidate": {"status": "не знаю"}, "reason": "insufficient_evidence",
                    "validation": {"ok": False, "status": "found", "detail": "сведения найдены, но не разобраны"},
                    "_log": {"absence": "found_no_budget"}}
        return {"validation": {"ok": True, "status": "absent", "detail": f"проверено частей: {len(parts)}"},
                "_log": {"absence": "confirmed"}}

    # ------------------------------------------------------------------ final_validate
    def final_validate(state: QState) -> dict:
        cand = dict(state.get("candidate") or {"status": "не знаю"})
        if cand.get("status") is None:
            cand["status"] = "не знаю"
        probs = final_problems(ctx.ps, cand, state.get("claims") or {})
        if cand["status"] in ("ответ", "расхождение"):
            kind = state["plan"]["kind"]
            ans = (cand.get("answer") or "").strip().lower()
            if kind == "yesno" and not (ans.startswith("да") or ans.startswith("нет")):
                probs.append("на вопрос «да или нет» ответ не начинается с «да»/«нет»")
            if (state.get("validation") or {}).get("ok") is not True:
                probs.append("смысловая проверка не подтвердила ответ")
        if cand["status"] == "нет в документах" and (state.get("validation") or {}).get("ok") is not True:
            probs.append("отсутствие не подтверждено отдельной проверкой")
        if cand["status"] in ("ответ", "расхождение"):
            # защита от инъекций, последний рубеж: ни ответ, ни цитаты не несут команд и текста из карантина
            qs = [e["quote"] for i in cand.get("claim_ids") or [] for e in (state.get("claims") or {}).get(i, {}).get(
                "evidence", [])]
            quarantined = ctx.ps.quarantined_texts()
            for qt in qs or [""]:
                probs += check_output(cand.get("answer") or "", qt, quarantined)
        if probs:
            return {"candidate": {"status": "не знаю"}, "reason": state.get("reason") or "validation_incomplete",
                    "_log": {"final_problems": probs}}
        return {"candidate": cand, "_log": {"final": cand["status"]}}

    # ------------------------------------------------------------------ format_export
    def format_export(state: QState) -> dict:
        cand = state.get("candidate") or {"status": "не знаю"}
        claims = state.get("claims") or {}
        st = cand.get("status") or "не знаю"
        used = [claims[i] for i in cand.get("claim_ids") or [] if i in claims] if st in ("ответ", "расхождение") else []
        evidence = []
        for c in used:
            for e in c["evidence"]:
                evidence.append({**e, "claim_id": c["claim_id"], "value": c["raw_value"] or c["value"],
                                 "role": "operand" if cand.get("calculation") and c["claim_id"] in
                                 {o["claim_id"] for o in cand["calculation"]["operands"]} else "main"})
        evidence.sort(key=lambda e: (1 if e.get("superseded_by") else 0, 0 if e["role"] == "main" else 1))
        if cand.get("calculation"):  # главная цитата расчёта — ставка (процент)
            rate = next((o for o in cand["calculation"]["operands"] if o["unit"] == "percent"), None)
            if rate:
                evidence.sort(key=lambda e: 0 if e["start"] == rate["evidence"]["start"] and
                              e["file"] == rate["evidence"]["file"] else 1)
        answer = format_answer(state["standalone"], st, cand.get("answer") or "", cand.get("values") or [],
                               cand.get("calculation", {}).get("result") if cand.get("calculation") else None)
        main = next((e for e in evidence if e["checker_ok"] and not e.get("superseded_by")), evidence[0] if evidence else None)
        from .quotes import csv_quote, Evidence  # noqa: F401
        quote = one_line(main["quote"])[:300] if main else ""
        cs = cov_summary(ctx.ps, state["coverage"])
        v = state.get("validation") or {}
        checks = {
            "status": {"ok": True, "detail": st},
            "sources": {"ok": not state.get("source_errors") or bool(used) if st in ("ответ", "расхождение") else None,
                        "detail": f"ошибок источников при чтении: {len(state.get('source_errors') or [])}"},
            "binding": {"ok": True if used else None, "detail": "число, единица и атрибут привязаны к цитате"
                        if used else ""},
            "amendment": {"ok": True if state.get("amendments") or (ctx.knowledge and ctx.knowledge.amendments) else None,
                          "detail": f"применено к ответу: {len(state.get('amendments') or [])}"},
            "discrepancy": {"ok": True if st == "расхождение" else None,
                            "detail": "две действующие стороны" if st == "расхождение" else ""},
            "calculation": {"ok": True if cand.get("calculation") else None,
                            "detail": cand.get("calculation", {}).get("expression", "") if cand.get("calculation") else ""},
            "coverage": {"ok": cs["blocks_read"] >= cs["blocks_total"],
                         "detail": f"окон {cs['windows_read']}/{cs['windows_total']}, блоков "
                                   f"{cs['blocks_read']}/{cs['blocks_total']}"},
            "validator": {"ok": v.get("ok"), "detail": f"{v.get('status', 'не выполнялась')}: {v.get('detail', '')}"[:300]},
        }
        panel = state.get("critics") or {}
        if panel:
            hard = [h["critic"] for h in panel.get("hard") or []]
            checks["critics"] = {"ok": not hard, "detail": (
                f"код: {sum(1 for x in panel['code'].values() if x.get('ok') is not None)} проверок, "
                f"модели: " + ", ".join(f"{k}={x.get('verdict')}" for k, x in panel.get("llm", {}).items())
                + (f"; возражения: {', '.join(hard)}" if hard else ""))[:300]}
        result = {
            "id": state.get("qid", ""), "package": ctx.ps.package_id, "question": state["question"],
            "standalone": state["standalone"] if state["standalone"] != state["question"] else "",
            "variant": "final", "status": st, "answer": answer, "answer_text": cand.get("answer") or "",
            "quote": quote, "file": main["file"] if main else "", "quote_checker_ok": bool(main and main["checker_ok"]),
            "evidence": evidence, "calculation": cand.get("calculation"), "checks": checks,
            "note": cand.get("reason") or "",
            "uncertainty_reason": (state.get("reason") or "insufficient_evidence") if st == "не знаю" else "",
            "absence_blocked": cand.get("absence_blocked") or [],
            "task_error": state.get("task_error"), "mode": state["mode"], "coverage": cs,
            "conflict": state.get("conflict") or None, "amendments_applied": state.get("amendments") or [],
            "uncertainties": state.get("uncertainties") or [], "judge": state.get("judge") or None,
            "critics": state.get("critics") or None,
            "claims": list(claims.values()), "source_errors": (state.get("source_errors") or [])[:30],
            "journal": state.get("trace") or [], "snapshot_id": ctx.ps.snapshot_id,
            "guard": {"question": (state.get("plan") or {}).get("guard"),
                      "package_findings": sum(1 for d in ctx.ps.docs for f in getattr(d, "suspicious", []) or []
                                              if f.get("verdict") != "document_text")},
        }
        return {"result": result, "_log": {"status": st}}

    # ------------------------------------------------------------------ маршруты
    def after_select(state: QState) -> str:
        return "read_windows" if state.get("batch") else "resolve_relations"

    def after_resolve(state: QState) -> str:
        return "read_windows" if state.get("retry") else "accumulate_evidence"

    def after_accumulate(state: QState) -> str:
        return "select_context" if state.get("next_windows") else "resolve_relations"

    def after_merge(state: QState) -> str:
        c = state.get("candidate") or {}
        if c.get("route") == "judge":
            return "judge"
        if c.get("route") == "merge":
            return "merge_candidates"
        if c.get("status") in ("не знаю", None):
            return "final_validate"
        if c.get("calculation"):
            return "calculate"
        return "semantic_validate"

    def after_calc(state: QState) -> str:
        return "final_validate" if (state.get("candidate") or {}).get("status") == "не знаю" else "semantic_validate"

    def after_validate(state: QState) -> str:
        if (state.get("candidate") or {}).get("route") == "reread":
            return "select_context"
        if (state.get("validation") or {}).get("ok") is True and \
                (state.get("candidate") or {}).get("status") in ("ответ", "расхождение"):
            return "critic_panel"
        return "final_validate"

    def after_critics(state: QState) -> str:
        return "select_context" if (state.get("candidate") or {}).get("route") == "reread" else "final_validate"

    g = StateGraph(QState)
    for name, fn in (("plan_question", plan_question), ("select_context", select_context),
                     ("read_windows", read_windows), ("resolve_sources", resolve_sources),
                     ("accumulate_evidence", accumulate_evidence), ("resolve_relations", resolve_relations),
                     ("merge_candidates", merge_candidates), ("judge", judge), ("calculate", calculate),
                     ("semantic_validate", semantic_validate), ("critic_panel", critic_panel),
                     ("final_validate", final_validate),
                     ("format_export", format_export)):
        g.add_node(name, node(name, fn))
    g.add_edge(START, "plan_question")
    g.add_conditional_edges("plan_question", lambda st: "final_validate" if st["plan"].get("blocked")
                            else "select_context", {"final_validate": "final_validate",
                                                    "select_context": "select_context"})
    g.add_conditional_edges("select_context", after_select, {"read_windows": "read_windows",
                                                             "resolve_relations": "resolve_relations"})
    g.add_edge("read_windows", "resolve_sources")
    g.add_conditional_edges("resolve_sources", after_resolve, {"read_windows": "read_windows",
                                                               "accumulate_evidence": "accumulate_evidence"})
    g.add_conditional_edges("accumulate_evidence", after_accumulate, {"select_context": "select_context",
                                                                      "resolve_relations": "resolve_relations"})
    g.add_edge("resolve_relations", "merge_candidates")
    g.add_conditional_edges("merge_candidates", after_merge, {"judge": "judge", "merge_candidates": "merge_candidates",
                                                              "final_validate": "final_validate",
                                                              "calculate": "calculate",
                                                              "semantic_validate": "semantic_validate"})
    g.add_edge("judge", "merge_candidates")
    g.add_conditional_edges("calculate", after_calc, {"final_validate": "final_validate",
                                                      "semantic_validate": "semantic_validate"})
    g.add_conditional_edges("semantic_validate", after_validate, {"select_context": "select_context",
                                                                  "critic_panel": "critic_panel",
                                                                  "final_validate": "final_validate"})
    g.add_conditional_edges("critic_panel", after_critics, {"select_context": "select_context",
                                                            "final_validate": "final_validate"})
    g.add_edge("final_validate", "format_export")
    g.add_edge("format_export", END)
    return g.compile()


def _run_graph(ctx: QuestionCtx, question: str, qid: str) -> dict:
    graph = build_graph(ctx)
    try:
        state = graph.invoke({"question": question, "qid": qid, "trace": []}, {"recursion_limit": 120})
        return state["result"]
    except Exception as e:  # техническая ошибка задачи не маскируется под содержательный ответ
        return {"id": qid, "package": ctx.ps.package_id, "question": question, "variant": "final",
                "status": "не знаю", "answer": "", "answer_text": "", "quote": "", "file": "", "evidence": [],
                "checks": {}, "uncertainty_reason": "", "mode": ctx.read_mode,
                "task_error": {"stage": "graph", "error": f"{type(e).__name__}: {e}"[:500], "retryable": True,
                               "traceback": traceback.format_exc()[-2000:]},
                "journal": [], "coverage": None}


# причины «не знаю», которые стоит перепроверить другой стратегией чтения (не «вопрос — инъекция»)
ESCALATE_REASONS = {"insufficient_evidence", "validation_incomplete", "content_budget_exhausted",
                    "unresolved_conflict", "critic_objection", "incomplete_coverage"}


def _alternative(ctx: QuestionCtx, res: dict) -> QuestionCtx | None:
    """Вторая попытка другим способом чтения: целиком → по окнам, поиск → шире поиск, по окнам → целиком."""
    mode = res.get("mode") or ctx.read_mode
    alt = dict(read_mode=mode, top_windows=ctx.top_windows)
    if mode == "full_context":
        if len(ctx.ps.windows) > 12:
            return None
        alt["read_mode"] = "exhaustive"
    elif mode == "retrieval":
        alt["top_windows"] = max(8, ctx.top_windows * 3)
    elif mode == "exhaustive":
        if ctx.ps.total_tokens > settings.full_context_max_tokens:
            return None
        alt["read_mode"] = "full_context"
    else:
        return None
    return QuestionCtx(ps=ctx.ps, knowledge=ctx.knowledge, retriever=ctx.retriever, llm=ctx.llm,
                       history=ctx.history, journal=ctx.journal, on_stage=ctx.on_stage,
                       use_passport=ctx.use_passport, strict=ctx.strict, **alt)


def answer_question(ctx: QuestionCtx, question: str, qid: str = "") -> dict:
    """Граф вопроса + самовосстановление:
    1) техническая ошибка задачи (сбой модели/сети после всех повторов) — вопрос повторяется заново
       (успешные вызовы берутся из кеша, бюджет новый), до QUESTION_RETRIES раз;
    2) честное «не знаю» из-за неполной проверки — вторая попытка другим способом чтения (ESCALATE_UNKNOWN):
       её ответ проходит те же проверки, критиков и валидаторы, поэтому принимается только подтверждённым."""
    t0 = time.time()
    res = _run_graph(ctx, question, qid)
    attempts = []
    tries = 0
    while res.get("status") == "не знаю" and (res.get("task_error") or {}).get("retryable") \
            and tries < settings.question_retries:
        tries += 1
        attempts.append({"kind": "retry", "why": (res.get("task_error") or {}).get("error", "")[:200]})
        sub = copy.copy(ctx)
        sub.tracker, sub.t0 = Tracker(), time.time()
        res = _run_graph(sub, question, qid)
        ctx.tracker.extend(sub.tracker)
    if res.get("status") == "не знаю" and settings.escalate_unknown and \
            res.get("uncertainty_reason") in ESCALATE_REASONS and not res.get("task_error"):
        alt = _alternative(ctx, res)
        if alt is not None:
            first = res
            res2 = _run_graph(alt, question, qid)
            ctx.tracker.extend(alt.tracker)
            attempts.append({"kind": "escalation", "from": first.get("mode"), "to": alt.read_mode,
                             "top_windows": alt.top_windows, "first_reason": first.get("uncertainty_reason"),
                             "status": res2.get("status")})
            if res2.get("status") != "не знаю":
                res = res2
                res["first_attempt"] = {k: first.get(k) for k in ("status", "uncertainty_reason", "mode", "journal")}
    res["attempts"] = attempts
    res["calls"] = ctx.tracker.summary()
    res["call_log"] = ctx.tracker.to_list()
    res["ms"] = int((time.time() - t0) * 1000)
    return res
