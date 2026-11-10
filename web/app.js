/* «Липецкие котики» — интерфейс. Без сборки и фреймворков. */
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const STATUS_CLASS = { "ответ": "st-ok", "расхождение": "st-disc", "нет в документах": "st-none", "не знаю": "st-unk" };
const CHECK_NAMES = {
  status: "Статус — одно из четырёх значений",
  sources: "Цитаты разрешены строго: source_id, снимок, диапазон, хеш",
  binding: "Число, единица и атрибут привязаны к своей части цитаты",
  amendment: "Учтены изменения извещения (действующая редакция)",
  discrepancy: "Расхождение: две действующие стороны с разными значениями",
  calculation: "Расчёт выполнен кодом (Decimal) по подтверждённым операндам",
  coverage: "Охват: сколько окон и блоков прочитано",
  validator: "Смысловой валидатор проверил каждое утверждение",
};
const REASONS = {
  insufficient_evidence: "Доказательств недостаточно — система не стала угадывать",
  incomplete_coverage: "Прочитано не всё — отсутствие ответа не доказано",
  unresolved_amendment: "Есть изменение извещения, которое не удалось однозначно применить",
  unresolved_conflict: "Документы противоречат друг другу, и выбрать сторону нельзя",
  relevant_unreadable_source: "Часть документов не прочитана (скан/битый файл) — отсутствие не доказано",
  validation_incomplete: "Смысловая проверка не выполнена (валидатор недоступен) — ответ не подтверждён",
  content_budget_exhausted: "Исчерпан бюджет запросов/времени на вопрос до полного чтения",
};
const MODE_NAMES = { auto: "auto", full_context: "весь комплект", retrieval: "поиск Qdrant", exhaustive: "все окна" };
// роли запросов к ИИ — человеческими словами (нулевые не показываем)
const ROLE_NAMES = {
  extract: "паспорт: извлечение фактов", amend: "разбор изменений", conflict: "сопоставимость расхождений",
  embed: "эмбеддинги", reader: "чтение окон", merge: "объединение", validator: "смысловой валидатор",
  absence: "проверка отсутствия", judge: "судья", contextualizer: "уточнение вопроса по диалогу",
  baseline: "один запрос «было»", diagnostic: "проверка доступа",
};
const NODE_NAMES = {
  plan_question: "план вопроса", select_context: "выбор контекста", read_windows: "чтение окон",
  resolve_sources: "строгая проверка источников", accumulate_evidence: "накопление доказательств",
  resolve_relations: "поправки и конфликты", merge_candidates: "объединение", judge: "судья", calculate: "калькулятор",
  semantic_validate: "смысловая проверка", final_validate: "финальная проверка кода", format_export: "оформление",
};
const roleList = (byRole) => Object.entries(byRole || {}).filter(([, v]) => +v > 0)
  .map(([k, v]) => `${ROLE_NAMES[k] || k}: ${v}`);
const VERDICT_COLORS = { "верно": "var(--ok)", "осторожно": "var(--care)", "обман": "var(--bad)" };
// порядок типов при равном числе вопросов (как в эталоне кейса); незнакомые — в конце по алфавиту
const TYPE_ORDER = ["факт", "ловушка: нет в документах", "расчёт", "после изменений", "ловушка: требования нет", "расхождение"];

const state = { config: null, diag: null, cid: null, meta: null, packages: [], run: null, runSnap: null, evaluation: null,
  chatSessions: {}, opts: { read_mode: "auto", use_passport: true, strict_coverage: true } };
const store = {  // localStorage — только удобство одного браузера; может быть недоступен
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* приватный режим */ } },
};

/* ------------------------------------------------------------------ API */
async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch { data = null; }
  if (!res.ok) throw new Error((data && data.error) || `Ошибка ${res.status}`);
  return data;
}
function fmtTokens(n) { n = +n || 0; return n >= 10000 ? Math.round(n / 1000) + "k" : n >= 1000 ? (n / 1000).toFixed(1) + "k" : String(n); }
function toast(msg, ms = 2600) {
  const t = $("#toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), ms);
}
function statusChip(s) {
  if (!s) return `<span class="status st-run">в работе</span>`;
  return `<span class="status ${STATUS_CLASS[s] || "st-unk"}">${esc(s)}</span>`;
}

/* ------------------------------------------------------------------ навигация */
function go(page) {
  $("#pageDocs").hidden = page !== "docs";
  $("#pageAsk").hidden = page !== "ask";
  $("#backToDocs").hidden = page !== "ask";
  closeCloud();
  window.scrollTo({ top: 0, behavior: "smooth" });
}
$$("[data-go]").forEach((el) => el.addEventListener("click", (e) => {
  e.preventDefault();
  if (el.dataset.go === "docs") resetDocs();
}));
function resetDocs() {
  // всё сохранено на диске: прежнюю загрузку можно открыть из «Ранее загруженных»
  go("docs");
  state.cid = null; state.meta = null; state.run = null;
  $("#docsHero").hidden = false; $("#ingest").hidden = true; $("#ingestResult").hidden = true;
  $("#folderInput").value = "";
  loadRecent();
}

/* ------------------------------------------------------------------ старт */
async function init() {
  try {
    state.config = await api("/api/config");
    state.opts = store.get("opora.opts", { read_mode: state.config.read_mode || "auto",
      use_passport: state.config.use_passport !== false, strict_coverage: state.config.strict_coverage !== false });
    $("#optMode").value = state.opts.read_mode; $("#optPassport").checked = state.opts.use_passport;
    $("#optStrict").checked = state.opts.strict_coverage;
    renderSettings();
  } catch (e) { toast("Сервер недоступен: " + e.message); }
  loadDiagnostics();
  loadRecent();
}
async function loadDiagnostics() {
  try {
    const d = state.diag = await api("/api/diagnostics");
    const mi = d.model_info || {};
    $("#modelName").textContent = d.api_key_configured ? `${d.model}${mi.context_tokens ? " · " + fmtTokens(mi.context_tokens) + " ток." : ""}` : "нет ключа OpenRouter (.env)";
    $("#modelChip").classList.toggle("err", !d.api_key_configured);
    if (mi.source === "mock") { $("#modelName").textContent = "ИМИТАЦИЯ модели — не живой прогон"; $("#modelChip").classList.add("err"); }
    $("#modelChip").title = `Модель OpenRouter${mi.source ? " · окно: " + mi.source : ""} — нажмите, чтобы проверить доступ`;
    $("#embedName").textContent = d.embedding_model;
    const q = d.qdrant || {};
    $("#qdrantName").textContent = q.ok ? `Qdrant ${q.version || ""}`.trim() : "Qdrant недоступен";
    $("#qdrantChip").classList.toggle("err", !q.ok);
    $("#qdrantChip").title = q.ok ? `Qdrant ${q.url} — гибридный поиск dense + BM25 + RRF` : `Qdrant: ${q.error || "нет ответа"} — режим retrieval будет честно переключён на exhaustive`;
    renderSettings();
  } catch (e) { $("#qdrantName").textContent = "диагностика недоступна"; }
}
function renderSettings() {
  const o = state.opts;
  $("#settingsSummary").textContent = `${MODE_NAMES[o.read_mode] || o.read_mode} · паспорт ${o.use_passport ? "вкл" : "выкл"} · строгий охват ${o.strict_coverage ? "вкл" : "выкл"}`;
  const q = state.diag?.qdrant;
  $("#settingsNote").textContent = (o.read_mode === "retrieval" && q && !q.ok)
    ? "Qdrant недоступен: вопросы будут прочитаны всеми окнами (exhaustive), переключение запишется в журнал."
    : (o.use_passport ? "" : "Без паспорта ответы всё равно проверяются по источникам; диагностический режим.");
}
["#optMode", "#optPassport", "#optStrict"].forEach((id) => $(id).addEventListener("change", () => {
  state.opts = { read_mode: $("#optMode").value, use_passport: $("#optPassport").checked, strict_coverage: $("#optStrict").checked };
  store.set("opora.opts", state.opts); renderSettings();
}));
$("#modelChip").style.cursor = "pointer";
$("#modelChip").addEventListener("click", async () => {
  toast("Проверяю модель…", 2000);
  try { const r = await api("/api/models/check", { method: "POST" }); toast(`Модель ${r.model} отвечает · ${r.ms} мс`); $("#modelChip").classList.remove("err"); }
  catch (e) { toast("Модель недоступна: " + e.message, 7000); $("#modelChip").classList.add("err"); }
});
async function loadRecent() {
  try {
    const list = (await api("/api/collections")).filter((c) => c.state && (c.state.startsWith("ready") || c.state === "interrupted"));
    $("#recent").hidden = !list.length;
    // без S4 сервер может вернуть одну и ту же папку несколько раз — показываем самую новую
    const serverDedup = list.some((c) => c.files != null || c.label != null);
    const seenLbl = new Set();
    const uniq = serverDedup ? list : list.filter((c) => {
      const key = (c.packages || []).join(",") || c.id;
      if (seenLbl.has(key)) return false;
      seenLbl.add(key); return true;
    });
    const list6 = uniq.slice(0, 6);
    $("#recentList").innerHTML = list6.map((c) => {
      const label = c.label || (c.packages || []).join(", ") || c.id;
      const nf = Number.isFinite(+c.files) && c.files != null ? +c.files : null;
      const dt = c.created ? new Date(typeof c.created === "number" ? c.created * 1000 : c.created) : null;
      const when = dt && !isNaN(dt) ? dt.toLocaleString("ru-RU", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) : "";
      const title = [label, nf != null ? `${nf} ${plural(nf, "документ", "документа", "документов")}` : "", when].filter(Boolean).join(" · ");
      return `<button type="button" class="recent-item" data-cid="${esc(c.id)}" title="${esc(title)}">
        <span class="recent-label">${esc(label)}</span>${nf != null ? `<span class="recent-count">${nf} ${plural(nf, "файл", "файла", "файлов")}</span>` : ""}${c.state === "interrupted" ? `<span class="recent-count warn">прервано</span>` : ""}</button>`;
    }).join("");
    $$(".recent-item").forEach((b) => b.addEventListener("click", () => openCollection(b.dataset.cid)));
  } catch { /* не критично */ }
}

/* ------------------------------------------------------------------ загрузка папки */
const SUPPORTED = /\.(txt|pdf|docx)$/i;
const SKIP = /(^|\/)(node_modules|\.git|\.next|__pycache__|\.venv|venv|dist|build|\.openai)\//;
$("#pickFolder").addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#folderInput").click(); } });
$("#folderInput").addEventListener("change", async (e) => {
  const all = [...e.target.files];
  let files = all.filter((f) => SUPPORTED.test(f.name) && !SKIP.test(f.webkitRelativePath || f.name) && !/^readme/i.test(f.name));
  // если выбрана папка с подпапками закупок, файлы в её корне (кейс, положение) — не закупка
  const depth = (f) => (f.webkitRelativePath || f.name).split("/").length;
  if (files.some((f) => depth(f) >= 3)) files = files.filter((f) => depth(f) >= 3);
  if (!files.length) { toast("В папке нет файлов TXT, PDF или DOCX"); return; }
  $("#docsHero").hidden = true;
  $("#ingest").hidden = false; $("#ingestResult").hidden = true;
  $("#ingestTitle").textContent = "Котик изучает документы…";
  $("#ingestErrors").innerHTML = "";
  try {
    const { id } = await api("/api/collections", { method: "POST" });
    state.cid = id;
    let done = 0;
    const total = files.length;
    renderStages([{ key: "upload", title: "Загрузка файлов на сервер", state: "running", done, total }]);
    const queue = [...files];
    const errors = [];
    async function worker() {
      while (queue.length) {
        const f = queue.shift();
        const rel = f.webkitRelativePath || f.name;
        try {
          await api(`/api/collections/${id}/files?path=${encodeURIComponent(rel)}`, { method: "PUT", body: f });
        } catch (err) { errors.push(`${rel}: ${err.message}`); }
        done++;
        renderStages([{ key: "upload", title: "Загрузка файлов на сервер", state: "running", done, total, detail: rel }]);
        setBar(done / total * 0.2);
      }
    }
    await Promise.all([worker(), worker(), worker(), worker()]);
    if (errors.length) $("#ingestErrors").innerHTML = errors.map((x) => `<div class="err">${esc(x)}</div>`).join("");
    const ing = await api(`/api/collections/${id}/ingest`, { method: "POST" });
    if (ing && ing.use && ing.use !== id) {
      // тот же снимок (файлы + версии + модели) уже подготовлен: не дублируем
      toast("Этот снимок уже подготовлен — открываю");
      openCollection(String(ing.use));
      return;
    }
    pollCollection(id, total);
  } catch (err) {
    $("#ingestErrors").innerHTML = `<div class="err">${esc(err.message)}</div>`;
  }
});
$("#resumeIngest").addEventListener("click", async () => {
  try {
    $("#resumeRow").hidden = true;
    await api(`/api/collections/${state.cid}/ingest`, { method: "POST" });
    pollCollection(state.cid, (state.meta?.files || []).length);
  } catch (e) { toast(e.message); }
});
function setBar(x, done = false) {
  const b = $("#ingestBar");
  b.style.width = Math.max(3, Math.min(100, x * 100)) + "%";
  b.classList.toggle("done", done);
}
function renderStages(stages) {
  const icon = { done: "✓", running: "", skipped: "–", pending: "", warning: "!", unavailable: "✗" };
  $("#stages").innerHTML = stages.map((s) => {
    const cnt = s.total != null ? ` <small>${s.done ?? 0} / ${s.total}</small>` : "";
    return `<li class="stage ${esc(s.state)}"><span class="stage-icon">${icon[s.state] ?? ""}</span>
      <div><b>${esc(s.title)}${cnt}</b><small>${esc(s.detail || "")}</small></div></li>`;
  }).join("");
}
async function pollCollection(id, nFiles) {
  const upload = { key: "upload", title: "Загрузка файлов на сервер", state: "done", done: nFiles, total: nFiles };
  for (;;) {
    if (state.cid !== id) return;
    let meta;
    try { meta = await api(`/api/collections/${id}`); } catch (e) { toast(e.message); await sleep(1500); continue; }
    state.meta = meta;
    renderStages([upload, ...meta.stages]);
    const weights = { read: 0.1, blocks: 0.05, amend: 0.05, passport: 0.4, conflicts: 0.05, index: 0.15 };
    let p = 0.2;
    for (const s of meta.stages) {
      const w = weights[s.key] || 0;
      p += s.state === "running" ? (s.total ? w * (s.done || 0) / s.total : 0) : w;  // без имитации: только завершённое
    }
    const fin = !["processing", "uploading"].includes(meta.state);
    $("#resumeRow").hidden = meta.state !== "interrupted";
    setBar(fin ? 1 : p, fin);
    if (meta.errors?.length) $("#ingestErrors").innerHTML = meta.errors.map((x) => `<div class="err">${esc(x)}</div>`).join("");
    if (fin) { showCollection(meta); return; }
    await sleep(700);
  }
}
async function openCollection(cid) {
  state.cid = cid;
  $("#docsHero").hidden = true; $("#ingest").hidden = false;
  try {
    const meta = await api(`/api/collections/${cid}`);
    state.meta = meta;
    renderStages(meta.stages);
    if (meta.state === "processing") { pollCollection(cid, (meta.files || []).length); return; }
    setBar(meta.state === "interrupted" ? 0.5 : 1, meta.state !== "interrupted");
    $("#resumeRow").hidden = meta.state !== "interrupted";
    if (meta.state === "interrupted") { $("#ingestTitle").textContent = "Подготовка прервана"; $("#ingestSub").textContent = "Нажмите «Продолжить подготовку»"; return; }
    showCollection(meta);
  } catch (e) { toast(e.message); resetDocs(); }
}

/* ------------------------------------------------------------------ итоги загрузки */
function showCollection(meta) {
  state.packages = meta.packages || [];
  const failed = meta.state === "failed";
  $("#ingestTitle").textContent = failed ? "Не получилось подготовить документы" :
    `Готово: ${state.packages.length} ${plural(state.packages.length, "закупка", "закупки", "закупок")}`;
  $("#ingestSub").textContent = failed ? "Подробности ниже" : ingestCallsText(meta.calls || {}) + (meta.snapshot_id ? ` · снимок ${meta.snapshot_id}` : "");
  const notes = [...(meta.errors || []).map((x) => `<div class="err">${esc(x)}</div>`), ...(meta.warnings || []).map((x) => `<div class="warn-line">⚠ ${esc(x)}</div>`)];
  $("#ingestErrors").innerHTML = notes.join("");
  $("#ingestResult").hidden = failed;
  if (failed) return;
  $("#pkgGrid").innerHTML = state.packages.map(pkgCard).join("");
  $$(".file-chip").forEach((c) => c.addEventListener("click", () => openDoc(c.dataset.pkg, c.dataset.file)));
  renderAmendments(); renderDiscrepancies();
}
function ingestCallsText(calls) {
  const by = calls.by_role || {};
  const n = calls.calls ?? Object.values(by).reduce((a, v) => a + (+v || 0), 0);
  const cost = calls.cost_usd != null ? ` · $${calls.cost_usd}` : "";
  if (!n) return "Запросов к ИИ при подготовке не было" + cost;
  const parts = Object.entries(by).filter(([, v]) => +v > 0).map(([k, v]) => `${ROLE_NAMES[k] || k}: ${v}`);
  return `Запросов к ИИ при подготовке: ${n}` + (parts.length ? ` — ${parts.join(", ")}` : "") + cost;
}
function pkgCard(p) {
  const docs = p.docs || [];
  const pc = p.passport;
  const conf = (p.conflicts || []).filter((d) => d.state === "confirmed").length;
  const warn = docs.flatMap((d) => (d.summary?.warnings || []).map((w) => `${d.name}: ${w}`));
  if (pc && pc.windows_failed) warn.push(`Паспорт: не прочитано окон ${pc.windows_failed} из ${pc.windows_total} — «нет в документах» запрещено`);
  const badges = [
    `<span class="badge">${docs.length} ${plural(docs.length, "файл", "файла", "файлов")}</span>`,
    `<span class="badge" title="Блоков ${p.blocks}, чанков ${p.chunks}">~${fmtTokens(p.tokens)} ток. · окон ${p.windows}</span>`,
    pc ? `<span class="badge ${pc.windows_failed ? "bad" : "blue"}" title="Окна паспорта: прочитано/всего">паспорт ${pc.windows_done}/${pc.windows_total} · фактов ${pc.facts}</span>` : `<span class="badge warn">паспорта нет</span>`,
    p.index ? `<span class="badge blue" title="${esc(p.index.collection || "")} · BM25: ${esc(p.index.bm25_mode || "")}">Qdrant: ${p.index.points} точек</span>` : `<span class="badge warn" title="Индекс не построен — retrieval переключится на exhaustive">без индекса</span>`,
    (p.amendments || []).length ? `<span class="badge blue">изменений: ${p.amendments.length}</span>` : "",
    conf ? `<span class="badge warn">расхождений: ${conf}</span>` : "",
  ].join("");
  return `<article class="pkg">
    <div class="pkg-head"><span class="pkg-name">${esc(p.name)}</span><span class="pkg-badges">${badges}</span></div>
    <div class="pkg-files">${docs.map((d) => {
      const st = d.state !== "readable" ? ` · ${d.state}` : "";
      const w = d.summary?.warnings || [];
      return `<span class="file-chip ${w.length || d.state !== "readable" ? "warn" : ""}" data-pkg="${esc(p.name)}" data-file="${esc(d.name)}" title="${esc(d.encoding)} · ${d.chars} символов · блоков ${d.blocks}${st}${w.length ? " · " + esc(w.join("; ")) : ""}">${esc(d.name)}</span>`;
    }).join("")}</div>
    ${warn.length ? `<div class="pkg-warn">⚠ ${warn.map(esc).join("<br>⚠ ")}</div>` : ""}
  </article>`;
}
const AM_STATE = { resolved: "", unresolved: "не разобрано", ambiguous: "неоднозначно" };
function renderAmendments() {
  const rows = state.packages.flatMap((p) => (p.amendments || []).map((a) => ({ p: p.name, a })));
  if (!rows.length) { $("#amendments").innerHTML = `<p class="muted">Изменений извещения в закупках нет.</p>`; return; }
  $("#amendments").innerHTML = `<div class="table-wrap"><table class="amend-table"><colgroup><col style="width:11%"><col style="width:37%"><col style="width:30%"><col style="width:22%"></colgroup><thead><tr><th>Закупка</th><th>Было</th><th>Стало</th><th>Источник</th></tr></thead><tbody>${
    rows.map(({ p, a }) => `<tr><td><b>${esc(p)}</b>${a.origin === "llm" ? `<div class="muted" style="font-size:.72rem">разобрано ИИ, ссылки проверены кодом</div>` : ""}</td>
      <td>${a.state === "resolved" ? `<div class="src">${docLink(p, a.target_file, a.target_line)}<br><q>${esc(a.old_text)}</q></div>` : `<span class="badge bad">${esc(AM_STATE[a.state] || a.state)}</span>`}</td>
      <td>${a.new_text ? `<div class="src"><q>${esc(a.new_text)}</q></div>` : esc(a.source_text)}${a.chain && a.chain[1] > 1 ? `<div class="muted" style="font-size:.72rem">цепочка: ${a.chain[0]} из ${a.chain[1]}</div>` : ""}</td>
      <td><div class="src">${docLink(p, a.source_file, a.source_line)}</div></td></tr>`).join("")}</tbody></table></div>`;
  bindDocLinks($("#amendments"));
}
// имя файла со строкой — ссылка, открывает документ с подсветкой строки
function docLink(pkg, file, line) {
  if (!file) return "";
  const ok = Number.isInteger(line) && line >= 0;
  return `<button type="button" class="fname doc-link" data-pkg="${esc(pkg)}" data-file="${esc(file)}" data-line="${ok ? line : ""}"
    title="Открыть документ"><span class="nw">${esc(file)}${ok ? "," : ""}</span>${ok ? ` <span class="nw">строка ${line + 1}</span>` : ""}</button>`;
}
function bindDocLinks(root) {
  $$(".doc-link", root).forEach((b) => b.addEventListener("click", () =>
    openDoc(b.dataset.pkg, b.dataset.file, b.dataset.line === "" ? [] : [+b.dataset.line])));
}
const CONFLICT_STATE = { confirmed: ["подтверждено", "st-disc"], different_conditions: ["разные условия", "st-none"],
  resolved_by_amendment: ["снято изменением", "st-ok"], unresolved: ["не разрешено", "st-unk"], candidate: ["кандидат", "st-unk"] };
function renderDiscrepancies() {
  const all = state.packages.flatMap((p) => (p.conflicts || []).map((d) => ({ p: p.name, d })));
  if (!all.length) {
    $("#discrepancies").innerHTML = `<div class="empty-note"><img src="/assets/cat-bot.webp" alt="">Кандидатов в противоречия паспорт не нашёл.</div>`;
    return;
  }
  all.sort((a, b) => (a.d.state === "confirmed" ? 0 : 1) - (b.d.state === "confirmed" ? 0 : 1));
  $("#discrepancies").innerHTML = `<div class="table-wrap"><table><thead><tr><th>id закупки</th><th>Объект</th><th>Значения</th><th>Источник</th></tr></thead><tbody>${
    all.map(({ p, d }) => { const [lbl, cls] = CONFLICT_STATE[d.state] || [d.state, "st-unk"];
      return `<tr><td><b>${esc(p)}</b><div><span class="status ${cls}">${esc(lbl)}</span></div></td><td>${esc(d.label)}${d.reason ? `<div class="muted" style="font-size:.74rem">${esc(d.reason)}</div>` : ""}</td>
      <td>${(d.sides || []).map((s) => `<div><span class="status ${s.superseded ? "st-unk" : "st-disc"}">${esc(s.value)}</span>${s.superseded ? ` <small class="muted">недействующая</small>` : ""}</div>`).join("")}</td>
      <td>${(d.sides || []).map((s) => `<div class="src"><button class="link" data-pkg="${esc(p)}" data-file="${esc(s.file)}" data-line="${s.line}">${esc(s.file)}</button><br><q>${esc(s.quote)}</q></div>`).join("")}</td></tr>`; }).join("")}</tbody></table></div>`;
  $$("#discrepancies .link").forEach((b) => b.addEventListener("click", () => openDoc(b.dataset.pkg, b.dataset.file, [+b.dataset.line])));
}
function plural(n, a, b, c) { const m = n % 10, h = n % 100; return m === 1 && h !== 11 ? a : m >= 2 && m <= 4 && (h < 10 || h >= 20) ? b : c; }

$("#goQuestions").addEventListener("click", () => {
  const sel = $("#pkgSelect");
  sel.innerHTML = state.packages.map((p) => `<option value="${esc(p.name)}">${esc(p.name)}</option>`).join("");
  const ex = ["Какая начальная (максимальная) цена контракта?", "До какой даты и времени принимаются заявки?",
    "Какой срок выполнения работ?", "Нужна ли лицензия ФСБ России? Ответьте да или нет."];
  $("#examples").innerHTML = ex.map((q) => `<button class="example" type="button">${esc(q)}</button>`).join("");
  $$(".example").forEach((b) => b.addEventListener("click", () => { $("#question").value = b.textContent; $("#question").focus(); }));
  go("ask");
  loadChatHistory();
  loadRuns();
  $("#question").focus();
});
function chatSession(pkg) {
  const key = `opora.chat.${state.cid}.${pkg}`;
  let s = state.chatSessions[pkg] || store.get(key, null);
  if (!s) { s = Math.random().toString(36).slice(2); }
  state.chatSessions[pkg] = s; store.set(key, s);
  return s;
}
async function loadChatHistory() {
  const pkg = $("#pkgSelect").value;
  if (!pkg) return;
  $$("#chatList .msg").forEach((m) => m.remove());
  try {
    const hist = await api(`/api/collections/${state.cid}/chat?package=${encodeURIComponent(pkg)}&session=${chatSession(pkg)}`);
    if (hist.length) $("#chatEmpty")?.setAttribute("hidden", "");
    else $("#chatEmpty")?.removeAttribute("hidden");
    for (const m of hist) {
      if (m.role === "user") addMsg("user", `<span class="pkg-tag">${esc(pkg)}</span>${esc(m.text)}`);
      else if (m.result) renderBotAnswer(addMsg("bot", ""), m.result);
    }
  } catch { /* история — удобство */ }
}
$("#pkgSelect").addEventListener("change", loadChatHistory);

/* ------------------------------------------------------------------ вкладки */
$$(".tab").forEach((t) => t.addEventListener("click", () => {
  $$(".tab").forEach((x) => { x.classList.toggle("active", x === t); x.setAttribute("aria-selected", x === t); });
  $("#tabChat").hidden = t.dataset.tab !== "chat";
  $("#tabBatch").hidden = t.dataset.tab !== "batch";
  $("#tabEval").hidden = t.dataset.tab !== "eval";
  closeCloud();
}));
function openTab(name) { const t = $(`.tab[data-tab="${name}"]`); if (t && !t.disabled) t.click(); }

/* ------------------------------------------------------------------ диалог */
$("#composer").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("#question").value.trim();
  const pkg = $("#pkgSelect").value;
  if (!q) { $("#question").focus(); return; }
  $("#question").value = "";
  $("#chatEmpty")?.setAttribute("hidden", "");
  addMsg("user", `<span class="pkg-tag">${esc(pkg)}</span>${esc(q)}`);
  const bot = addMsg("bot", `<span class="typing"><i></i><i></i><i></i> <span class="stg">котик читает документы…</span></span>`);
  $("#sendBtn").disabled = true;
  const session = chatSession(pkg);
  try {
    const { job } = await api(`/api/collections/${state.cid}/chat`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ package: pkg, question: q, session, options: state.opts }),
    });
    for (;;) {
      await sleep(450);
      const j = await api(`/api/jobs/${job}`);
      if (j.state === "running") { $(".stg", bot).textContent = (j.stage || "думаю") + "…"; continue; }
      if (j.state === "error") throw new Error(j.error);
      renderBotAnswer(bot, j.result);
      break;
    }
  } catch (err) {
    $(".bubble", bot).innerHTML = `<span class="status st-unk">ошибка сервиса</span><div class="note">${esc(err.message)}</div>`;
  } finally { $("#sendBtn").disabled = false; $("#question").focus(); }
});
function addMsg(who, html) {
  const el = document.createElement("div");
  el.className = `msg ${who}`;
  el.innerHTML = `<img class="avatar" src="/assets/${who === "user" ? "cat-user" : "cat-bot"}.webp" alt=""><div class="bubble">${html}</div>`;
  $("#chatList").appendChild(el);
  $("#chatList").scrollTop = $("#chatList").scrollHeight;
  return el;
}
function renderBotAnswer(el, r) {
  const ans = r.answer || r.answer_text || "";
  const cov = r.coverage;
  const note = r.task_error ? `Техническая ошибка: ${r.task_error.error}` : r.status === "не знаю" ? (REASONS[r.uncertainty_reason] || "") :
    (r.status === "нет в документах" ? `Прочитано окон ${cov?.windows_read ?? "?"}/${cov?.windows_total ?? "?"}, блоков ${cov?.blocks_read ?? "?"}/${cov?.blocks_total ?? "?"} — сведений нет` : "");
  const sec = r.ms ? ` · ${(r.ms / 1000).toFixed(1)} с · ${r.calls?.calls ?? 0} ${plural(r.calls?.calls ?? 0, "запрос", "запроса", "запросов")}` : "";
  $(".bubble", el).innerHTML = `${statusChip(r.status)}${ans ? `<div class="answer">${esc(ans)}</div>` : ""}
    ${r.standalone ? `<div class="note">Понял вопрос как: «${esc(r.standalone)}»</div>` : ""}
    ${note ? `<div class="note">${esc(note)}</div>` : ""}<div class="more">Обоснование →${esc(sec)}</div>`;
  $(".bubble", el).addEventListener("click", () => openCloud(r));
  $("#chatList").scrollTop = $("#chatList").scrollHeight;
}

/* ------------------------------------------------------------------ голос: WAV 16 кГц прямо в браузере */
const rec = { on: false, ctx: null, stream: null, node: null, chunks: [], rate: 48000, t0: 0, timer: null };
$("#micBtn").addEventListener("click", () => (rec.on ? stopRec() : startRec()));
async function startRec() {
  if (!navigator.mediaDevices?.getUserMedia) { toast("Браузер не даёт доступ к микрофону (нужен localhost или HTTPS)"); return; }
  try {
    rec.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  } catch { toast("Нет доступа к микрофону — разрешите его в браузере"); return; }
  rec.ctx = new (window.AudioContext || window.webkitAudioContext)();
  rec.rate = rec.ctx.sampleRate;
  const src = rec.ctx.createMediaStreamSource(rec.stream);
  rec.node = rec.ctx.createScriptProcessor(4096, 1, 1);
  rec.chunks = [];
  rec.node.onaudioprocess = (e) => rec.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  src.connect(rec.node); rec.node.connect(rec.ctx.destination);
  rec.on = true; rec.t0 = Date.now();
  $("#micBtn").classList.add("recording"); $("#recTimer").hidden = false;
  rec.timer = setInterval(() => {
    const s = Math.floor((Date.now() - rec.t0) / 1000);
    $("#recTimer").textContent = s;
    if (s >= 22) stopRec(); // автоматический предел записи
  }, 250);
  toast("Говорите… нажмите микрофон ещё раз, чтобы закончить", 2500);
}
async function stopRec() {
  rec.on = false; clearInterval(rec.timer);
  $("#micBtn").classList.remove("recording"); $("#recTimer").hidden = true;
  rec.node.disconnect(); rec.stream.getTracks().forEach((t) => t.stop()); await rec.ctx.close();
  const wav = encodeWav(downsample(rec.chunks, rec.rate, 16000), 16000);
  $("#micBtn").disabled = true;
  toast("GigaAM распознаёт речь…", 8000);
  try {
    const r = await api("/api/voice", { method: "POST", headers: { "Content-Type": "audio/wav" }, body: wav });
    const input = $("#question");
    input.value = (input.value ? input.value + " " : "") + r.text;
    input.focus();
    toast(`Распознано за ${r.seconds} с записи — проверьте текст и отправьте`);
  } catch (err) { toast("Голосовой ввод: " + err.message, 6000); }
  finally { $("#micBtn").disabled = false; }
}
function downsample(chunks, from, to) {
  const len = chunks.reduce((a, c) => a + c.length, 0);
  const data = new Float32Array(len);
  let o = 0; for (const c of chunks) { data.set(c, o); o += c.length; }
  if (from === to) return data;
  const ratio = from / to, out = new Float32Array(Math.floor(len / ratio));
  for (let i = 0; i < out.length; i++) {
    const s = Math.floor(i * ratio), e = Math.min(len, Math.floor((i + 1) * ratio));
    let sum = 0; for (let j = s; j < e; j++) sum += data[j];
    out[i] = sum / Math.max(1, e - s);
  }
  return out;
}
function encodeWav(samples, rate) {
  const buf = new ArrayBuffer(44 + samples.length * 2), v = new DataView(buf);
  const w = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, "data");
  v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) { const s = Math.max(-1, Math.min(1, samples[i])); v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true); }
  return new Blob([buf], { type: "audio/wav" });
}

/* ------------------------------------------------------------------ облачко с обоснованием */
function evidenceHtml(e, label = "") {
  const ctx = e.context || e.quote;
  const pos = ctx.indexOf(e.quote);
  const text = pos >= 0 ? esc(ctx.slice(0, pos)) + `<mark>${esc(e.quote)}</mark>` + esc(ctx.slice(pos + e.quote.length)) : `<mark>${esc(e.quote)}</mark>`;
  const old = e.superseded_by ? `<span class="badge bad" title="${esc(e.superseded_by.new_text || "")}">недействующая редакция → ${esc(e.superseded_by.file)}</span>` : "";
  return `<div class="evidence ${e.superseded_by ? "old" : ""}">
    <div class="ev-head"><span class="ev-file">${esc(e.file)} · строка ${(e.line ?? 0) + 1}${e.page ? " · стр. " + e.page : ""} · [${esc(e.source_id)}] символы ${e.start}–${e.end}</span>
      <span>${label} ${old} ${e.checker_ok === false ? `<span class="badge warn" title="Официальный check.py не найдёт эту цитату (файл не TXT в UTF-8/CP1251)">вне check.py</span>` : ""}
      <button class="link" data-file="${esc(e.file)}" data-line="${e.line}">открыть документ</button></span></div>
    <div class="ev-text">${text}</div>
    <div class="muted ev-meta">снимок ${esc(e.snapshot_id)} · документ ${esc(e.document_id)} · sha256 ${esc(String(e.text_sha256 || "").slice(0, 12))}…</div>
  </div>`;
}
function openCloud(r, opts = {}) {
  const isBase = r.variant === "baseline";
  const ev = r.evidence || [];
  const checks = Object.entries(r.checks || {});
  const ci = (ok) => ok === true ? `<span class="ci ok">✓</span>` : ok === false ? `<span class="ci bad">✗</span>` : `<span class="ci na">–</span>`;
  const claims = Object.fromEntries((r.claims || []).map((c) => [c.claim_id, c]));
  // доказательства: в расхождении — по сторонам, в расчёте — по операндам, иначе — список
  let evHtml = "";
  if (r.calculation && r.calculation.operands) {
    evHtml = r.calculation.operands.map((o) => evidenceHtml(o.evidence, `<span class="badge">операнд: ${esc(o.value)} ${esc(o.unit)}</span>`)).join("");
  } else if (r.status === "расхождение" && r.conflict?.sides) {
    evHtml = r.conflict.sides.map((sd, i) => `<div class="side"><b>Сторона ${i + 1}: ${esc(sd.values.join(", "))}</b>${
      sd.claim_ids.flatMap((cid) => (claims[cid]?.evidence || []).map((e) => evidenceHtml(e))).join("")}</div>`).join("");
  } else {
    evHtml = ev.map((e, i) => evidenceHtml(e, i === 0 ? `<span class="badge">в CSV</span>` : "")).join("");
  }
  const cov = r.coverage;
  const covHtml = cov && !isBase ? `<h5>Охват чтения</h5><div class="cov">
      <div>режим: <b>${esc(MODE_NAMES[r.mode] || r.mode)}</b> · окон ${cov.windows_read}/${cov.windows_total} · блоков ${cov.blocks_read}/${cov.blocks_total}${cov.windows_failed ? ` · <span class="bad">ошибок чтения: ${cov.windows_failed}</span>` : ""}</div>
      ${Object.keys(cov.outcomes || {}).length ? `<div class="muted">итоги окон: ${Object.entries(cov.outcomes).map(([k, v]) => `${esc(k)} ${v}`).join(", ")}</div>` : ""}
      ${(cov.switches || []).map((x) => `<div class="warn-line">↪ ${esc(x)}</div>`).join("")}
      ${(cov.unreadable_files || []).length ? `<div class="bad">нечитаемые: ${cov.unreadable_files.map(esc).join(", ")}</div>` : ""}
      ${(r.absence_blocked || []).map((x) => `<div class="warn-line">⛔ «нет в документах» запрещено: ${esc(x)}</div>`).join("")}</div>` : "";
  const calcHtml = r.calculation ? `<h5>Расчёт (код, Decimal)</h5><div class="calc">${esc(r.calculation.expression)}</div>
    <div class="muted" style="font-size:.76rem">округление ${esc(r.calculation.rounding || "")}</div>` : "";
  const journal = Array.isArray(r.journal) ? r.journal : [];
  const jHtml = journal.length ? `<details class="attempts" ${opts.openAttempts ? "open" : ""}><summary>Ход проверки (LangGraph): ${journal.length} ${plural(journal.length, "шаг", "шага", "шагов")}</summary>
    <ol class="journal">${journal.map((j) => { const { node, ms, calls_used, ...rest } = j;
      const det = Object.entries(rest).filter(([, v]) => v != null && !(Array.isArray(v) && !v.length) && !(typeof v === "object" && !Array.isArray(v) && !Object.keys(v).length))
        .map(([k, v]) => `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`).join(" · ");
      return `<li><b>${esc(NODE_NAMES[node] || node)}</b> <span class="muted">${ms} мс · запросов ${calls_used}</span>${det ? `<div class="muted jd">${esc(det.slice(0, 600))}</div>` : ""}</li>`; }).join("")}</ol></details>` : "";
  const srcErr = (r.source_errors || []).length ? `<details class="attempts"><summary>Отклонённые утверждения читателя: ${r.source_errors.length}</summary>
    ${r.source_errors.map((x) => `<div class="bad">✗ [${esc(x.window)}] ${esc(x.error)}</div>`).join("")}</details>` : "";
  const fmt = Array.isArray(r.format_changes) ? r.format_changes.filter(Boolean) : [];
  const raw = r.raw && typeof r.raw === "object" ? r.raw : null;
  const rawHtml = isBase && raw ? `<h5>Исходный ответ модели</h5><div class="raw-answer">
      <div><span class="raw-k">статус</span><span>${esc(raw.status ?? "") || "<i class=\"muted\">пусто</i>"}</span></div>
      <div><span class="raw-k">ответ</span><span>${esc(raw.answer ?? "") || "<i class=\"muted\">пусто</i>"}</span></div>
      <div><span class="raw-k">цитата</span><span>${raw.quote ? "«" + esc(String(raw.quote)) + "»" : "<i class=\"muted\">пусто</i>"}</span></div>
      ${raw.file ? `<div><span class="raw-k">файл</span><span class="ev-file">${esc(raw.file)}</span></div>` : ""}</div>` : "";
  const fmtHtml = isBase && fmt.length ? `<h5>Что поправило оформление</h5><ul class="fmt-changes">${fmt.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : "";
  const baseQuote = isBase && r.quote ? `<h5>Цитата модели</h5><div class="evidence"><div class="ev-text">«${esc(r.quote)}»</div>
    <div class="why" style="margin-top:6px">${r.quote_found ? "✓ фрагмент есть в документах закупки (файл и смысл не проверялись)" : "✗ такого фрагмента нет в документах"}</div></div>` : "";
  $("#cloudBody").innerHTML = `
    <h3>${esc(r.id && !String(r.id).startsWith("chat-") ? r.id + " · " : "")}${esc(r.question)}</h3>
    ${isBase ? `<span class="badge warn">Было · один запрос (тот же форматтер, без проверок)</span> ` : ""}${statusChip(r.status)}
    ${r.answer ? `<div class="ans-big">${esc(r.answer)}</div>` : ""}
    ${r.standalone ? `<div class="muted" style="font-size:.84rem">Понял вопрос как: «${esc(r.standalone)}»</div>` : ""}
    ${r.answer_text && r.answer_text !== r.answer ? `<div class="muted" style="font-size:.84rem">Формулировка модели: «${esc(r.answer_text)}»</div>` : ""}
    ${r.task_error ? `<div class="reason bad">Техническая ошибка (${esc(r.task_error.stage)}): ${esc(r.task_error.error)}</div>` : ""}
    ${r.context_overflow ? `<div class="reason">Комплект не помещается в окно модели — генерации не было (корпус не обрезается)</div>` : ""}
    ${r.status === "не знаю" && !isBase ? `<div class="reason">${esc(REASONS[r.uncertainty_reason] || "Система не уверена")}</div>` : ""}
    ${calcHtml}
    ${evHtml ? `<h5>Обоснование из документов</h5>${evHtml}` : baseQuote}
    ${covHtml}${fmtHtml}${rawHtml}
    ${checks.length ? `<h5>Выполненные проверки</h5><ul class="checks">${checks.map(([k, v]) => `<li>${ci(v.ok)}<span>${esc(CHECK_NAMES[k] || k)}${v.detail ? `<small>${esc(v.detail)}</small>` : ""}</span></li>`).join("")}</ul>` : ""}
    ${r.judge ? `<div class="attempt judge"><b>Судья</b>: ${r.judge.choice ? "выбрал " + esc(r.judge.choice) : "не выбрал — честное «не знаю»"} <span class="muted">${esc(r.judge.reason || "")}</span></div>` : ""}
    ${srcErr}${jHtml}
    ${r.calls ? `<p class="muted" style="font-size:.76rem;margin-top:14px">Запросов к ИИ: ${r.calls.calls ?? 0}${r.calls.cached ? ` (из кеша ${r.calls.cached})` : ""}${roleList(r.calls.by_role).length ? ` — ${esc(roleList(r.calls.by_role).join(", "))}` : ""} · ${((r.ms || 0) / 1000).toFixed(1)} с${r.calls.cost_usd != null ? " · $" + esc(r.calls.cost_usd) : ""}</p>` : ""}`;
  const allEv = [...ev, ...(r.claims || []).flatMap((c) => c.evidence || [])];
  $$("#cloudBody .link").forEach((b) => b.addEventListener("click", () => openDoc(r.package, b.dataset.file,
    allEv.filter((e) => e.file === b.dataset.file).map((e) => [e.start, e.end]))));
  $("#cloud").hidden = false;
  document.body.classList.add("cloud-open");
}
function closeCloud() { $("#cloud").hidden = true; document.body.classList.remove("cloud-open"); }
$("#cloudClose").addEventListener("click", closeCloud);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closeCloud(); $("#docModal").hidden = true; } });

// ranges: [[start, end], …] — диапазоны символов (EvidenceRef) или номера строк (старые ссылки)
async function openDoc(pkg, file, ranges = []) {
  try {
    const d = await api(`/api/collections/${state.cid}/source?package=${encodeURIComponent(pkg)}&file=${encodeURIComponent(file)}`);
    $("#docTitle").textContent = `${pkg} / ${file} · ${d.summary.encoding} · ${d.summary.state} · снимок ${d.snapshot_id}`;
    const text = d.text || "";
    const lines = []; let off = 0;
    for (const t of text.split("\n")) { lines.push([off, off + t.length, t]); off += t.length + 1; }
    const spans = ranges.map((r) => Array.isArray(r) ? r : (lines[r] ? [lines[r][0], lines[r][1]] : null)).filter(Boolean);
    const old = (d.superseded || []).map((x) => [x.start, x.end, x]);
    const mark = (ls, le, t) => {  // разметка строки: <mark> для доказательств, <s> для недействующих диапазонов
      const cuts = new Set([0, t.length]);
      for (const [a, b] of [...spans, ...old]) { if (a < le && b > ls) { cuts.add(Math.max(0, a - ls)); cuts.add(Math.min(t.length, b - ls)); } }
      const pts = [...cuts].sort((x, y) => x - y);
      let out = "";
      for (let i = 0; i < pts.length - 1; i++) {
        const a = ls + pts[i], b = ls + pts[i + 1], piece = esc(t.slice(pts[i], pts[i + 1]));
        const hl = spans.some(([x, y]) => x < b && y > a), o = old.find(([x, y]) => x < b && y > a);
        let h = piece;
        if (o) h = `<s class="old-range" title="Недействующая редакция → ${esc(o[2].file || "")}: ${esc(o[2].new_text || "")}">${h}</s>`;
        if (hl) h = `<mark>${h}</mark>`;
        out += h;
      }
      return out;
    };
    $("#docView").innerHTML = text ? lines.map(([ls, le, t], i) => {
      const hit = spans.some(([a, b]) => a < le + 1 && b > ls);
      return `<div class="doc-line ${hit ? "hl" : ""}"><span class="ln">${i + 1}</span><span>${mark(ls, le, t) || "&nbsp;"}</span></div>`;
    }).join("") : `<p class="muted" style="padding:20px">Файл пустой — прочитан полностью, текста нет.</p>`;
    $("#docModal").hidden = false;
    const first = $(".doc-line.hl", $("#docView"));
    if (first) first.scrollIntoView({ block: "center" });
  } catch (e) { toast(e.message); }
}
$("#docClose").addEventListener("click", () => ($("#docModal").hidden = true));
$("#docModal").addEventListener("click", (e) => { if (e.target.id === "docModal") $("#docModal").hidden = true; });

/* ------------------------------------------------------------------ пакет вопросов */
function runQuery() {
  const o = state.opts;
  return `baseline=${$("#withBaseline").checked ? 1 : 0}&read_mode=${encodeURIComponent(o.read_mode)}&use_passport=${o.use_passport ? 1 : 0}&strict_coverage=${o.strict_coverage ? 1 : 0}`;
}
$("#csvInput").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  $("#csvInfo").className = "csv-info"; $("#csvInfo").textContent = "Проверяю CSV…";
  try {
    const r = await api(`/api/collections/${state.cid}/runs?${runQuery()}`, { method: "POST", body: f });
    state.run = r;
    $("#csvInfo").innerHTML = `✓ <b>${r.count}</b> ${plural(r.count, "вопрос", "вопроса", "вопросов")} · закупки: ${r.packages.map(esc).join(", ")} · файл <code>${esc(f.name)}</code>
      <div class="muted">режим: ${esc(MODE_NAMES[r.options.read_mode])} · паспорт ${r.options.use_passport ? "вкл" : "выкл"} · строгий охват ${r.options.strict_coverage ? "вкл" : "выкл"} (меняется в «Режим работы» на первой странице)</div>`;
    $("#startRun").hidden = false;
  } catch (err) {
    $("#csvInfo").className = "csv-info err"; $("#csvInfo").textContent = err.message; $("#startRun").hidden = true;
  }
  e.target.value = "";
});
$("#startRun").addEventListener("click", async () => {
  if (!state.run) return;
  $("#startRun").disabled = true;
  try {
    await api(`/api/runs/${state.run.id}/start`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ baseline: $("#withBaseline").checked }) });
    showRun(state.run.id, state.run.questions, $("#withBaseline").checked);
  } catch (err) { toast(err.message); }
  finally { $("#startRun").disabled = false; }
});
function showRun(rid, questions, withBaseline) {
  $("#batchStart").hidden = true; $("#batchRun").hidden = false;
  $("#evaluation").hidden = true; $("#evaluation").innerHTML = ""; state.evaluation = null;
  $("#evalTab").disabled = true; $("#evalEmpty").hidden = false;
  setRunActions(false, withBaseline);
  $("#runKpis").innerHTML = ""; $("#runBar").style.width = "0"; $("#runBar").classList.remove("done");
  buildTable(questions);
  pollRun(rid);
}
async function loadRuns() {
  try {
    const runs = await api(`/api/collections/${state.cid}/runs`);
    const label = { completed: "завершён", running: "идёт", interrupted: "прерван", cancelled: "остановлен", queued: "не запущен" };
    $("#runsList").innerHTML = runs.length ? `<h4>Прогоны по этому снимку</h4>${runs.slice(0, 8).map((r) => {
      const dt = r.started ? new Date(r.started * 1000).toLocaleString("ru-RU", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) : "";
      return `<button type="button" class="run-item" data-rid="${esc(r.id)}"><b>${esc(dt || r.id)}</b>
        <span class="status ${r.state === "completed" ? "st-ok" : r.state === "interrupted" ? "st-disc" : "st-unk"}">${esc(label[r.state] || r.state)}</span>
        <span class="muted">${r.answered}/${r.questions} · ${esc(MODE_NAMES[r.options?.read_mode] || "")}${r.baseline ? " · с «было»" : ""}</span></button>`;
    }).join("")}` : "";
    $$("#runsList .run-item").forEach((b) => b.addEventListener("click", () => openRun(b.dataset.rid)));
  } catch { /* список прогонов — удобство */ }
}
async function openRun(rid) {
  try {
    const s = await api(`/api/runs/${rid}`);
    state.run = { id: rid, questions: s.questions };
    showRun(rid, s.questions, s.baseline);
    if (s.evaluation) { state.evaluation = s.evaluation; state.runSnap = s; renderEvaluation(false); }
  } catch (e) { toast(e.message); }
}
$("#resumeRun").addEventListener("click", async () => {
  if (!state.run) return;
  try {
    await api(`/api/runs/${state.run.id}/resume`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    $("#resumeRun").hidden = true;
    pollRun(state.run.id);
  } catch (e) { toast(e.message); }
});
function buildTable(qs) {
  $("#qtable tbody").innerHTML = qs.map((q, i) => `<tr data-id="${esc(q.id)}"><td>${i + 1}</td><td>${esc(q.id)}</td><td>${esc(q.package)}</td>
    <td>${esc(q.question)}</td><td class="c-status">${statusChip(null)}</td><td class="c-answer"><span class="muted">…</span></td></tr>`).join("");
}
// кнопки скачивания включаются только после атомарной записи всех артефактов (completed)
const RUN_FILES = { "#dlAnswers": "answers.csv", "#dlBaseline": "baseline.csv", "#dlTrace": "trace.json" };
const ARTIFACTS = ["answers.json", "baseline.json", "baseline_raw.json", "coverage.json", "calculations.json", "run_config.json",
  "source_manifest.json", "prompts.json", "knowledge_versions.json", "metrics.json"];
function setRunActions(enabled, withBaseline, rid = null) {
  $("#dlBaseline").hidden = !withBaseline;
  for (const [id, file] of Object.entries(RUN_FILES)) {
    const a = $(id);
    if (enabled && rid) a.setAttribute("href", `/api/runs/${encodeURIComponent(rid)}/download/${file}`);
    else a.removeAttribute("href");
    a.toggleAttribute("disabled", !enabled);
    a.setAttribute("aria-disabled", String(!enabled));
    if (enabled) a.removeAttribute("tabindex"); else a.setAttribute("tabindex", "-1");
  }
  const ev = $("#evalBtn");
  ev.toggleAttribute("disabled", !enabled);
  ev.setAttribute("aria-disabled", String(!enabled));
  ev.setAttribute("tabindex", enabled ? "0" : "-1");
  $("#goldInput").disabled = !enabled;
  $("#artifacts").hidden = !enabled;
  if (enabled && rid) $("#artifactLinks").innerHTML = ARTIFACTS.filter((f) => withBaseline || !f.startsWith("baseline"))
    .map((f) => `<a class="btn btn-ghost" href="/api/runs/${encodeURIComponent(rid)}/download/${f}" download>${f}</a>`).join("");
}
$("#evalBtn").addEventListener("keydown", (e) => {
  if ((e.key === "Enter" || e.key === " ") && !$("#goldInput").disabled) { e.preventDefault(); $("#goldInput").click(); }
});
function answerCell(r, qid, quoteLen = 90) {
  const main = r.answer ? `<span class="ans-main">${esc(r.answer)}</span>` : `<span class="ans-main ans-status">${esc(r.status || "нет ответа")}</span>`;
  let sub = "";
  if (r.task_error) sub = "техническая ошибка: " + esc(String(r.task_error.error).slice(0, 80));
  else if (r.status === "не знаю" && REASONS[r.uncertainty_reason]) sub = esc(REASONS[r.uncertainty_reason]);
  else if (r.quote) { const qq = String(r.quote).replace(/^[«"]+|[»"]+$/g, ""); sub = "«" + esc(qq.slice(0, quoteLen)) + (qq.length > quoteLen ? "…" : "") + "»"; }
  return `<div class="ans-cell ${r.task_error ? "task-err" : ""}" role="button" tabindex="0" data-q="${esc(qid)}" title="Открыть обоснование">${main}${sub ? `<small>${sub}</small>` : ""}</div>`;
}
function bindAnswerCell(cell, getResult, opts) {
  const open = () => { const r = getResult(); if (r) openCloud(r, opts); };
  cell.addEventListener("click", open);
  cell.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
}
async function pollRun(rid) {
  const seen = new Set();
  for (;;) {
    if (!state.run || state.run.id !== rid) return;
    let s;
    try { s = await api(`/api/runs/${rid}`); } catch (e) { toast(e.message); await sleep(2000); continue; }
    state.runSnap = s;
    const total = s.questions.length * (s.baseline ? 2 : 1);
    const done = Object.keys(s.final).length + Object.keys(s.base).length;
    $("#runCount").textContent = `${Object.keys(s.final).length} / ${s.questions.length}`;
    const stTxt = { completed: `готово за ${s.elapsed} с`, interrupted: "прерван перезапуском — ответы, что уже на диске, сохранены",
      cancelled: "остановлен" }[s.state] || `отвечаю… ${s.elapsed} с` + (s.baseline ? ` · «было»: ${Object.keys(s.base).length}/${s.questions.length}` : "");
    $("#runState").textContent = `${stTxt} · ${MODE_NAMES[s.options?.read_mode] || ""}`;
    $("#runBar").style.width = (done / total * 100) + "%";
    $("#runBar").classList.toggle("done", s.state === "completed");
    for (const q of s.questions) {
      const r = s.final[q.id];
      const tr = $(`#qtable tr[data-id="${CSS.escape(q.id)}"]`);
      if (!tr) continue;
      if (r && !seen.has(q.id)) {
        seen.add(q.id);
        tr.classList.add("fresh");
        $(".c-status", tr).innerHTML = statusChip(r.status);
        $(".c-answer", tr).innerHTML = answerCell(r, q.id);
        bindAnswerCell($(".ans-cell", tr), () => state.runSnap?.final[q.id]);
      } else if (!r) {
        const st = s.running[`final:${q.id}`];
        if (st) $(".c-status", tr).innerHTML = `<span class="status st-run">${esc(NODE_NAMES[st] || st)}</span>`;
      }
    }
    renderKpis(s);
    $("#resumeRun").hidden = !["interrupted", "cancelled"].includes(s.state);
    const fin = s.state === "completed";
    setRunActions(fin, s.baseline, rid);
    if (fin || ["interrupted", "cancelled"].includes(s.state)) { loadRuns(); return; }
    await sleep(1000);
  }
}
function renderKpis(s) {
  const m = s.metrics.final, b = s.metrics.baseline;
  const k = (v, l, cls = "") => `<div class="kpi ${cls}"><b>${v}</b><span>${l}</span></div>`;
  $("#runKpis").innerHTML = k(m.calls, "запросов") + k(fmtTokens(m.prompt_tokens + m.completion_tokens), "токенов") +
    k(m.cost_usd != null ? "$" + m.cost_usd : "нет данных", "стоимость") + k((m.p50_ms / 1000).toFixed(1) + " с", "медиана/вопрос") +
    (m.task_errors ? k(m.task_errors, "техн. ошибок", "bad") : "") + (b ? k(b.calls, "запросов «было»") : "");
}
$("#newQuestions").addEventListener("click", () => {
  state.run = null; state.runSnap = null;
  $("#batchRun").hidden = true; $("#batchStart").hidden = false; $("#startRun").hidden = true; $("#csvInfo").textContent = "";
  $("#evaluation").hidden = true; $("#evaluation").innerHTML = ""; state.evaluation = null;
  $("#evalTab").disabled = true; $("#evalEmpty").hidden = false;
  closeCloud(); loadRuns();
});

/* ------------------------------------------------------------------ проверка через check.py (вкладка «Проверка») */
$("#goldInput").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f || !state.run) return;
  toast("Запускаю официальный check.py…", 2500);
  try {
    state.evaluation = await api(`/api/runs/${state.run.id}/evaluate`, { method: "POST", body: f });
    state.runSnap = await api(`/api/runs/${state.run.id}`);
    $("#toast").hidden = true;
    renderEvaluation(true);
  } catch (err) { toast("Проверка: " + err.message, 6000); }
  e.target.value = "";
});
function verdictCard(title, res, cls) {
  const n = res.n || 1;
  return `<div class="eval-card ${cls}"><h4>${title}</h4>
    <div class="big-index">${res.index}<small> индекс доверия</small></div>
    <div class="verdicts"><span class="verdict v-ok">верно ${res.correct}</span><span class="verdict v-care">осторожно ${res.careful}</span><span class="verdict v-lie">обман ${res.lies}</span></div>
    <div class="stack"><span style="width:${res.correct / n * 100}%;background:var(--ok)"></span><span style="width:${res.careful / n * 100}%;background:var(--care)"></span><span style="width:${res.lies / n * 100}%;background:var(--bad)"></span></div>
    <div class="muted" style="font-size:.8rem;margin-top:8px">Верных ${Math.round(res.correct / n * 100)}% из ${res.n} вопросов</div></div>`;
}
// По каждому типу вопросов — полосы «было» и «стало»: стопка верно / осторожно / обман в абсолютных числах.
// Полоса целиком = все вопросы этого типа, поэтому «было» и «стало» сравниваются напрямую.
const VERDICTS = [["верно", "ok"], ["осторожно", "care"], ["обман", "lie"]];
function typeOrder(fin) {
  const ord = (t) => { const i = TYPE_ORDER.indexOf(t); return i < 0 ? 99 : i; };
  return Object.keys(fin).sort((a, b) => ((fin[b]?.["всего"] || 0) - (fin[a]?.["всего"] || 0)) || (ord(a) - ord(b)) || a.localeCompare(b, "ru"));
}
function typeChart(ev) {
  const fin = ev.variants.final?.by_type || {}, base = ev.variants.baseline?.by_type || null;
  const types = typeOrder(fin);
  if (!types.length) return "";
  const cnt = (t, v) => +(t?.[v] || 0);
  const bar = (variant, name, t, type) => {
    const total = cnt(t, "всего");
    const segs = VERDICTS.filter(([v]) => cnt(t, v) > 0).map(([v, c]) =>
      `<span class="seg seg-${c}" style="flex-grow:${cnt(t, v)}" title="${esc(type)} · ${name}: ${v} — ${cnt(t, v)} из ${total}">${cnt(t, v)}</span>`).join("");
    return `<div class="tc-bar ${variant}"><span class="tc-variant">${name}</span><div class="tc-track" role="img"
      aria-label="${esc(type)}, ${name}: ${VERDICTS.map(([v]) => `${v} ${cnt(t, v)}`).join(", ")}">${segs || `<span class="tc-empty">нет данных</span>`}</div></div>`;
  };
  const rows = types.map((t) => `<div class="tc-cat">
      <div class="tc-label">${esc(t)} <span class="tc-n">${cnt(fin[t], "всего")}</span></div>
      <div class="tc-bars">${base ? bar("base", "было", base[t], t) : ""}${bar("final", "стало", fin[t], t)}</div></div>`).join("");
  const sum = (bt, v) => types.reduce((a, t) => a + cnt(bt?.[t], v), 0);
  const cells = (bt, t) => VERDICTS.map(([v, c]) => `<td class="num n-${c}">${cnt(bt?.[t], v)}</td>`).join("");
  const sumCells = (bt) => VERDICTS.map(([v, c]) => `<td class="num n-${c}">${sum(bt, v)}</td>`).join("");
  const vh = VERDICTS.map(([v]) => `<th class="num">${v}</th>`).join("");
  const table = `<div class="table-wrap tc-table"><table>
    <thead><tr><th rowspan="2">Тип вопроса</th><th rowspan="2" class="num">всего</th>${base ? `<th colspan="3" class="grp">было</th>` : ""}<th colspan="3" class="grp">стало</th></tr>
      <tr>${base ? vh : ""}${vh}</tr></thead>
    <tbody>${types.map((t) => `<tr><td>${esc(t)}</td><td class="num">${cnt(fin[t], "всего")}</td>${base ? cells(base, t) : ""}${cells(fin, t)}</tr>`).join("")}</tbody>
    <tfoot><tr><td>Итого</td><td class="num">${sum(fin, "всего")}</td>${base ? sumCells(base) : ""}${sumCells(fin)}</tr></tfoot></table></div>`;
  return `<div class="chart-box"><h4>Результат по типам вопросов${base ? ": было → стало" : ""}</h4>
    <div class="legend">${VERDICTS.map(([v, c]) => `<span><i class="sw-${c}"></i>${v}</span>`).join("")}
      <span class="legend-note">цифры — число вопросов; полоса — все вопросы этого типа</span></div>
    <div class="type-chart">${rows}</div>${table}</div>`;
}
const UNC_ORDER = Object.keys(REASONS);
function metricsTable(m, b) {
  const row = (l, f, bb) => `<tr><td>${l}</td>${b ? `<td class="num">${bb ?? "—"}</td>` : ""}<td class="num">${f}</td></tr>`;
  const sec = (x) => ((x || 0) / 1000).toFixed(1) + " с";
  return `<div class="table-wrap"><table class="metrics-table"><thead><tr><th>Показатель</th>${b ? "<th class=num>было</th>" : ""}<th class="num">стало</th></tr></thead><tbody>
    ${row("запросов к ИИ", m.calls, b?.calls)}${row("из кеша", m.cached || 0, b?.cached || 0)}
    ${row("токенов", fmtTokens(m.prompt_tokens + m.completion_tokens), b ? fmtTokens(b.prompt_tokens + b.completion_tokens) : null)}
    ${row("стоимость", m.cost_usd != null ? "$" + m.cost_usd : "нет данных", b ? (b.cost_usd != null ? "$" + b.cost_usd : "нет данных") : null)}
    ${row("медиана / p95 на вопрос", sec(m.p50_ms) + " / " + sec(m.p95_ms), b ? sec(b.p50_ms) + " / " + sec(b.p95_ms) : null)}
    ${row("технические ошибки (отдельно от «не знаю»)", m.task_errors, b?.task_errors)}
    ${row("переполнение окна (без генерации)", m.context_overflow || 0, b?.context_overflow || 0)}
    ${row("отклонённых утверждений читателя (строгие источники)", m.source_errors, null)}
    ${row("«нет в документах» запрещено политикой охвата", m.absence_blocked, null)}
    ${row("вызовов судьи", m.judge, null)}
  </tbody></table></div>`;
}
function checksCards(m) {
  const c = m.checks || {};
  const card = (k, v) => `<div class="critic"><b>${v.ok}<small> / ${v.ok + v.fail + v.na}</small></b>${esc(CHECK_NAMES[k] || k)}
    <div class="muted" style="font-size:.74rem">пройдено · не пройдено ${v.fail} · не применялось ${v.na}</div></div>`;
  const unc = Object.entries(m.uncertainty_reasons || {}).sort((a, b) => UNC_ORDER.indexOf(a[0]) - UNC_ORDER.indexOf(b[0]))
    .map(([k, v]) => `<div class="critic alt"><b>${v}</b>«не знаю»: ${esc(REASONS[k] || k)}</div>`).join("");
  return Object.entries(c).map(([k, v]) => card(k, v)).join("") + unc;
}
function renderEvaluation(switchTab) {
  const ev = state.evaluation, s = state.runSnap;
  const fin = ev.variants.final, base = ev.variants.baseline;
  // пойманный обман — по настоящему ответу «было» (check.py вынес «обман»), а не по наличию ловушки в эталоне
  const caught = base ? s.questions.filter((q) => base.verdicts[q.id]?.verdict === "обман" && fin.verdicts[q.id]?.verdict !== "обман") : [];
  const caughtHtml = caught.map((q, i) => {
    const b = s.base[q.id] || {}, f = s.final[q.id] || {};
    const how = [...(f.source_errors || []).slice(0, 2).map((x) => x.error), ...(f.absence_blocked || []).slice(0, 1),
      ...Object.entries(f.checks || {}).filter(([, v]) => v.ok === false).map(([k]) => "не пройдено: " + (CHECK_NAMES[k] || k))];
    return `<div class="caught-item" ${i >= 3 ? 'data-more hidden' : ""}><b>${esc(q.id)}</b> · ${esc(q.question)}
      <div class="row"><span class="badge bad">было</span> ${statusChip(b.status)} <b>${b.answer ? esc(b.answer) : `<span class="ans-status">${esc(b.status || "нет ответа")}</span>`}</b> <span class="why">check.py: ${esc(base.verdicts[q.id].why)}</span> <button class="link" data-open-base="${esc(q.id)}">ответ модели →</button></div>
      <div class="row"><span class="badge">стало</span> ${statusChip(f.status)} <b>${f.answer ? esc(f.answer) : `<span class="ans-status">${esc(f.status || "нет ответа")}</span>`}</b> <button class="link" data-open="${esc(q.id)}">обоснование →</button></div>
      <div class="muted" style="font-size:.8rem;margin-top:4px">Что сработало: ${esc(how.length ? how.join(" · ") : "строгие источники, привязка значения и смысловая проверка")}</div></div>`;
  }).join("");
  $("#evaluation").innerHTML = `
    ${state.diag?.model_info?.source === "mock" ? `<div class="reason bad">Это прогон на сценарной имитации модели (проверка кода и интерфейса). Цифры ниже — НЕ результат живой модели.</div>` : ""}
    <h3 class="section-title">Проверка официальным check.py ${ev.integrity.unchanged ? `<span class="badge">check.py не изменён · sha256 ${esc(ev.integrity.sha256.slice(0, 12))}…</span>` : `<span class="badge bad">check.py изменён!</span>`}</h3>
    <p class="muted" style="font-size:.82rem">Режим: ${esc(MODE_NAMES[s.options?.read_mode] || "")} · паспорт ${s.options?.use_passport ? "вкл" : "выкл"} · снимок ${esc(s.snapshot_id || "")} · эталон sha256 ${esc(String(ev.gold_sha256 || "").slice(0, 12))}…
      Оценщик проверяет значение и наличие цитаты в файлах закупки; имя файла и смысловую связь цитаты он не проверяет.</p>
    ${ev.missing_in_gold.length ? `<p class="muted">Нет в эталоне: ${ev.missing_in_gold.map(esc).join(", ")}</p>` : ""}
    <div class="eval-cards">${base ? verdictCard("Было · один запрос (тот же форматтер, без проверок)", base.result, "") : ""}${verdictCard("Стало · LangGraph: источники, охват, валидатор", fin.result, "final")}</div>
    ${typeChart(ev)}
    <h3 class="section-title">Стоимость, время и технические ошибки</h3>${metricsTable(s.metrics.final, s.metrics.baseline)}
    ${caught.length ? `<h3 class="section-title">Обман «было», которого нет в «стало» <span class="hint">${caught.length} ${plural(caught.length, "случай", "случая", "случаев")}</span></h3><div class="caught">${caughtHtml}</div>${caught.length > 3 ? `<button class="btn btn-ghost" id="moreCaught" style="margin-top:8px">Показать все ${caught.length}</button>` : ""}` : ""}
    <h3 class="section-title">Фактически выполненные проверки <span class="hint">счётчики срабатываний, не «вклад в точность»</span></h3><div class="critics">${checksCards(s.metrics.final)}</div>
    <h3 class="section-title">Несовпавшие вопросы
      <span class="filter" id="evFilter"><button class="active" data-f="all">все</button><button data-f="обман">обман</button><button data-f="осторожно">осторожно</button></span>
      ${base ? `<span class="filter" id="evVariant"><button class="active" data-v="final">стало</button><button data-v="baseline">было</button></span>` : ""}</h3>
    <div id="wrongTable"></div>`;
  $("#evaluation").hidden = false; $("#evalEmpty").hidden = true;
  $("#evalTab").disabled = false;
  $$("#evaluation [data-open]").forEach((b) => b.addEventListener("click", () => openCloud(s.final[b.dataset.open], { openAttempts: true })));
  $$("#evaluation [data-open-base]").forEach((b) => b.addEventListener("click", () => openCloud(s.base[b.dataset.openBase])));
  $("#moreCaught")?.addEventListener("click", (e) => { $$("#evaluation [data-more]").forEach((x) => (x.hidden = false)); e.target.remove(); });
  let filt = "all", variant = "final";
  const draw = () => {
    const vv = ev.variants[variant];
    const res = variant === "final" ? s.final : s.base;
    const rows = s.questions.filter((q) => vv.verdicts[q.id] && vv.verdicts[q.id].verdict !== "верно" && (filt === "all" || vv.verdicts[q.id].verdict === filt));
    $("#wrongTable").innerHTML = rows.length ? `<div class="table-wrap"><table class="wrong-table"><colgroup><col style="width:4%"><col style="width:7%"><col style="width:19%"><col style="width:14%"><col style="width:17%"><col style="width:13%"><col style="width:13%"><col style="width:13%"></colgroup><thead><tr><th>№</th><th>id</th><th>Вопрос</th><th>Правильный тип</th><th>Правильный ответ</th><th>Наш тип</th><th>Наш ответ</th><th>Итог check.py</th></tr></thead><tbody>${
      rows.map((q, i) => { const g = ev.gold[q.id], r = res[q.id] || {}; const v = vv.verdicts[q.id];
        return `<tr><td>${i + 1}</td><td>${esc(q.id)}</td><td>${esc(q.question)}</td><td>${statusChip(g.status)}</td>
          <td>${esc(g.expected)}${g.quote ? `<div class="src"><span class="fname">${esc(g.file)}</span> <q>${esc(g.quote)}</q></div>` : ""}</td>
          <td>${statusChip(r.status)}</td><td>${answerCell(r, q.id, 80)}</td>
          <td><span class="verdict ${v.verdict === "обман" ? "v-lie" : "v-care"}">${esc(v.verdict)}</span><div class="why">${esc(v.why)}</div></td></tr>`; }).join("")}</tbody></table></div>`
      : `<div class="empty-note"><img src="/assets/cat-bot.webp" alt="">Таких вопросов нет — все ответы засчитаны как верные.</div>`;
    $$("#wrongTable .ans-cell").forEach((c) => bindAnswerCell(c, () => res[c.dataset.q]));
  };
  $$("#evFilter button").forEach((b) => b.addEventListener("click", () => { filt = b.dataset.f; $$("#evFilter button").forEach((x) => x.classList.toggle("active", x === b)); draw(); }));
  $$("#evVariant button").forEach((b) => b.addEventListener("click", () => { variant = b.dataset.v; $$("#evVariant button").forEach((x) => x.classList.toggle("active", x === b)); draw(); }));
  draw();
  if (switchTab) openTab("eval");
}

init();
