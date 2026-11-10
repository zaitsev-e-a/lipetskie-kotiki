"""Журнал охвата чтения и политика «нет в документах».

Отсутствие нельзя выводить из пустой выдачи поиска, одной отрицательной попытки, успешного
извлечения файлов или готового паспорта. Статус разрешён только когда весь необходимый текст покрыт
успешными чтениями по текущему вопросу и все исходы окон известны и не противоречат отсутствию.
"""
from __future__ import annotations

from .sources import PackageSources


def new_coverage(ps: PackageSources) -> dict:
    return {"windows": {w.window_id: {"state": "planned", "outcome": None, "claims": 0, "uncertainties": [],
                                      "blocks": len(w.block_ids)} for w in ps.windows},
            "blocks_read": [], "switches": [], "reads": 0}


def record_read(cov: dict, wid: str, block_ids, outcome: str | None, claims: int, uncertainties: list,
                error: str = "") -> None:
    w = cov["windows"].setdefault(wid, {"blocks": len(block_ids)})
    w.update(state="failed" if error else "read", outcome=outcome, claims=claims,
             uncertainties=list(uncertainties or []), error=error)
    cov["reads"] = cov.get("reads", 0) + 1
    if not error:
        cov["blocks_read"] = sorted(set(cov["blocks_read"]) | set(block_ids))


def unvisited(cov: dict) -> list[str]:
    return [wid for wid, w in cov["windows"].items() if w.get("state") != "read"]


def summary(ps: PackageSources, cov: dict) -> dict:
    readable = [s for s in ps.order if ps.doc_by_id[ps.blocks[s].document_id].state in ("readable", "partial")]
    read = set(cov["blocks_read"])
    outcomes: dict[str, int] = {}
    for w in cov["windows"].values():
        if w.get("outcome"):
            outcomes[w["outcome"]] = outcomes.get(w["outcome"], 0) + 1
    return {"windows_total": len(ps.windows), "windows_read": sum(1 for w in cov["windows"].values()
                                                                  if w.get("state") == "read"),
            "windows_failed": sum(1 for w in cov["windows"].values() if w.get("state") == "failed"),
            "blocks_total": len(readable), "blocks_read": len(read & set(readable)), "outcomes": outcomes,
            "switches": cov.get("switches", []),
            "unreadable_files": [d.name for d in ps.docs if d.state not in ("readable", "blank")],
            "blank_files": [d.name for d in ps.docs if d.state == "blank"]}


def absence_policy(ps: PackageSources, cov: dict, claims: list[dict], knowledge, unresolved_amendments: list) -> list[str]:
    """Пустой список — политика отсутствия выполнена (кроме финальной смысловой проверки).
    Иначе — причины, по которым «нет в документах» запрещено."""
    reasons = []
    bad = [d.name for d in ps.docs if d.state not in ("readable", "blank")]
    if bad:
        reasons.append("relevant_unreadable_source: " + ", ".join(bad))
    s = summary(ps, cov)
    if s["blocks_read"] < s["blocks_total"]:
        reasons.append(f"incomplete_coverage: прочитано {s['blocks_read']} из {s['blocks_total']} блоков")
    unc = [wid for wid, w in cov["windows"].items() if w.get("outcome") == "uncertain"]
    if unc:
        reasons.append("incomplete_coverage: окна с неопределённостью " + ", ".join(unc))
    if s["windows_failed"]:
        reasons.append("incomplete_coverage: окна не прочитаны из-за ошибок")
    if claims:
        reasons.append("insufficient_evidence: есть относящиеся частичные сведения, они не разрешены")
    if unresolved_amendments:
        reasons.append("unresolved_amendment: есть неразобранные поправки")
    if knowledge is not None:
        failed = [w for w, v in knowledge.windows.items() if v.get("state") != "done"]
        if failed:
            reasons.append("incomplete_coverage: паспорт построен не по всем окнам (" + ", ".join(failed) + ")")
    return reasons
