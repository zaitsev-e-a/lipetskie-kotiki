"""Настройки из окружения и файла .env (без внешних библиотек)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        # Переменные окружения важнее файла.
        os.environ.setdefault(key, value)


_load_dotenv(ROOT / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on", "да")


def _str(name: str, default: str = "") -> str:
    """Пустое значение в .env (`DATA_DIR=`) означает «по умолчанию», а не пустую строку."""
    v = os.environ.get(name)
    return v.strip() if v and v.strip() else default


# Роли компонентов. Для каждой можно задать свою модель: MODEL_<РОЛЬ>=..., а также
# PROVIDER_<РОЛЬ>, FALLBACK_MODEL_<РОЛЬ>, MAX_TOKENS_<РОЛЬ>, REASONING_<РОЛЬ>.
# Критики наследуют от группы: MODEL_CRITIC_UNITS → MODEL_CRITIC → OPENROUTER_MODEL.
ROLE_TITLES = {
    "extract": "паспорт: извлечение фактов по окнам",
    "passport_repair": "паспорт: исправление отклонённых фактов",
    "passport_gap": "паспорт: добор пропущенных атрибутов чек-листа",
    "amend": "паспорт: разбор поправок",
    "conflict": "паспорт: сверка расхождений",
    "guard": "защита от инъекций: классификация подозрительных фрагментов",
    "contextualizer": "диалог: самостоятельная формулировка вопроса",
    "reader": "чтение окон по вопросу",
    "merge": "сборка ответа из проверенных утверждений",
    "judge": "судья неоднозначностей",
    "validator": "общий критик (смысловая проверка)",
    "absence": "проверка отсутствия сведений",
    "critic_attribute": "критик: тот ли атрибут и объект",
    "critic_units": "критик: единицы, периоды, «не менее/не более»",
    "critic_amendment": "критик: действующая редакция после изменений",
    "critic_discrepancy": "критик: поиск другого значения (адвокат дьявола)",
    "critic_calculation": "критик: смысл операндов расчёта",
    "critic_polarity": "критик: да/нет и отрицания",
    "baseline": "«было»: один запрос",
    "diagnostic": "кнопка «Проверить модель»",
}
ROLE_GROUPS = {r: "critic" for r in ROLE_TITLES if r.startswith("critic_")}
ROLE_GROUPS.update({"passport_repair": "extract", "passport_gap": "extract"})
CRITIC_NAMES = [r[len("critic_"):] for r in ROLE_TITLES if r.startswith("critic_")]
# роли, от которых зависит паспорт снимка (смена их модели — другой снимок)
PASSPORT_ROLES = ("extract", "passport_repair", "passport_gap", "amend", "conflict", "guard")

# Лимиты ответа и размышления по ролям (переопределяются MAX_TOKENS_<РОЛЬ>, REASONING_<РОЛЬ>)
ROLE_MAX_TOKENS = {"extract": 6000, "passport_repair": 3000, "passport_gap": 2500, "reader": 3500, "merge": 1500,
                   "validator": 6000, "absence": 4000, "judge": 1500, "contextualizer": 300, "baseline": 1500,
                   "conflict": 2000, "amend": 1500, "guard": 1200, "diagnostic": 20, "critic": 2500}
ROLE_REASONING = {"extract": "off", "passport_repair": "off", "passport_gap": "off", "reader": "off",
                  "merge": "off", "baseline": "off", "contextualizer": "off", "amend": "off", "guard": "off",
                  "conflict": "low", "validator": "low", "absence": "low", "judge": "low", "critic": "off"}


@dataclass
class Settings:
    api_key: str = field(default_factory=lambda: _str("OPENROUTER_API_KEY"))
    model: str = field(default_factory=lambda: _str("OPENROUTER_MODEL", "qwen/qwen3.8-flash"))
    # запасная модель для всех ролей (если у роли нет своей FALLBACK_MODEL_<РОЛЬ>); пусто — без замены
    fallback_model: str = field(default_factory=lambda: _str("OPENROUTER_FALLBACK_MODEL"))
    embedding_model: str = field(
        default_factory=lambda: _str("OPENROUTER_EMBEDDING_MODEL", "qwen/qwen3-embedding-8b"))
    base_url: str = field(default_factory=lambda: _str("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    provider: str = field(default_factory=lambda: _str("OPENROUTER_PROVIDER"))
    # Размышления отключены: быстрее и дешевле, качество держат проверки кодом.
    reasoning: bool = field(default_factory=lambda: _bool("OPENROUTER_REASONING", False))
    temperature: float = 0.0
    timeout: int = field(default_factory=lambda: _int("LLM_TIMEOUT_SECONDS", 90))
    transport_retries: int = field(default_factory=lambda: _int("TRANSPORT_RETRIES", 3))
    max_output_tokens: int = field(default_factory=lambda: _int("MAX_OUTPUT_TOKENS", 1500))
    global_concurrency: int = field(default_factory=lambda: _int("GLOBAL_LLM_CONCURRENCY", 16))
    question_concurrency: int = field(default_factory=lambda: _int("QUESTION_CONCURRENCY", 6))
    max_calls_per_question: int = field(default_factory=lambda: _int("MAX_LLM_CALLS_PER_QUESTION", 40))
    question_deadline: int = field(default_factory=lambda: _int("QUESTION_DEADLINE_SECONDS", 300))
    repair_rounds: int = field(default_factory=lambda: _int("REPAIR_ROUNDS", 2))
    # Чтение: auto | full_context | retrieval | exhaustive; строгий охват включён по умолчанию
    read_mode: str = field(default_factory=lambda: _str("READ_MODE", "auto"))
    # auto читает комплект целиком, только если он не больше этого числа токенов (и помещается в окно модели);
    # большие комплекты — через поиск: быстрее и дешевле, а полноту держит строгий охват по паспорту
    full_context_max_tokens: int = field(default_factory=lambda: _int("FULL_CONTEXT_MAX_TOKENS", 120000))
    strict_coverage: bool = field(default_factory=lambda: _bool("STRICT_COVERAGE", True))
    use_passport: bool = field(default_factory=lambda: _bool("USE_PASSPORT", True))
    # 0 — спросить у OpenRouter (/models → context_length); иначе явный лимит окна модели в токенах
    model_context_tokens: int = field(default_factory=lambda: _int("MODEL_CONTEXT_TOKENS", 0))
    context_budget_ratio: float = field(default_factory=lambda: float(os.environ.get("CONTEXT_BUDGET_RATIO", "0.8")))
    block_max_chars: int = field(default_factory=lambda: _int("BLOCK_MAX_CHARS", 1500))
    chunk_tokens: int = field(default_factory=lambda: _int("CHUNK_TOKENS", 800))
    chunk_overlap_tokens: int = field(default_factory=lambda: _int("CHUNK_OVERLAP_TOKENS", 120))
    window_tokens: int = field(default_factory=lambda: _int("WINDOW_TOKENS", 6000))
    # сплошной обход большого комплекта: соседние окна объединяются в порции до этого размера (токены)
    sweep_tokens: int = field(default_factory=lambda: _int("SWEEP_TOKENS", 60000))
    # Qdrant: dense + BM25 (полнотекстовый индекс) + RRF
    qdrant_url: str = field(default_factory=lambda: _str("QDRANT_URL", "http://localhost:6333"))
    qdrant_api_key: str = field(default_factory=lambda: os.environ.get("QDRANT_API_KEY", ""))
    retrieval_top_k: int = field(default_factory=lambda: _int("RETRIEVAL_TOP_K", 12))
    prefetch_limit: int = field(default_factory=lambda: _int("PREFETCH_LIMIT", 40))
    rrf_k: int = field(default_factory=lambda: _int("RRF_K", 60))
    # Qdrant недоступен → локальный гибридный индекс в памяти (1) или полный обход окон (0)
    local_index: bool = field(default_factory=lambda: _bool("LOCAL_INDEX", True))
    llm_cache: bool = field(default_factory=lambda: _bool("LLM_CACHE", True))
    use_critic: bool = field(default_factory=lambda: _bool("USE_CRITIC", True))
    # Панель критиков после общего критика: список через запятую или all / none
    critics: str = field(default_factory=lambda: _str("CRITICS", "all"))
    # Защита от инъекций: код всегда; LLM-классификатор подозрительных фрагментов — GUARD_LLM=1
    guard_llm: bool = field(default_factory=lambda: _bool("GUARD_LLM", True))
    # Самоисправление: повтор вопроса при технической ошибке и вторая попытка «не знаю» другим режимом чтения
    question_retries: int = field(default_factory=lambda: _int("QUESTION_RETRIES", 1))
    escalate_unknown: bool = field(default_factory=lambda: _bool("ESCALATE_UNKNOWN", True))
    # Паспорт: повтор отклонённых фактов окна и добор атрибутов чек-листа
    passport_repair: bool = field(default_factory=lambda: _bool("PASSPORT_REPAIR", True))
    passport_checklist: bool = field(default_factory=lambda: _bool("PASSPORT_CHECKLIST", True))
    data_dir: Path = field(default_factory=lambda: Path(_str("DATA_DIR", str(ROOT / "data"))))
    host: str = field(default_factory=lambda: _str("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _int("PORT", 8000))
    asr_model: str = field(default_factory=lambda: _str("ASR_MODEL", "v3_e2e_rnnt"))

    # ------------------------------------------------------------------ настройки по ролям
    def _role_env(self, prefix: str, role: str) -> str:
        """PREFIX_<РОЛЬ>, затем PREFIX_<ГРУППА> (critic_units → CRITIC)."""
        for r in (role, ROLE_GROUPS.get(role)):
            if r:
                v = _str(f"{prefix}_{r.upper()}")
                if v:
                    return v
        return ""

    def role_model(self, role: str) -> str:
        """Модель компонента: MODEL_<РОЛЬ> → MODEL_<ГРУППА> → OPENROUTER_MODEL."""
        return self._role_env("MODEL", role) or self.model

    def role_provider(self, role: str) -> str:
        return self._role_env("PROVIDER", role) or self.provider

    def role_fallback(self, role: str) -> str:
        """Запасная модель роли: FALLBACK_MODEL_<РОЛЬ> → OPENROUTER_FALLBACK_MODEL; совпадающая с основной — нет."""
        fb = self._role_env("FALLBACK_MODEL", role) or self.fallback_model
        return fb if fb and fb != self.role_model(role) else ""

    def role_max_tokens(self, role: str) -> int:
        default = ROLE_MAX_TOKENS.get(role) or ROLE_MAX_TOKENS.get(ROLE_GROUPS.get(role, ""), self.max_output_tokens)
        return _int(f"MAX_TOKENS_{role.upper()}", _int(f"MAX_TOKENS_{ROLE_GROUPS.get(role, role).upper()}", default))

    def role_reasoning(self, role: str) -> str:
        """off | low | medium | high. Быстрый режим разрешён для паспортов; проверяющим — минимальное размышление."""
        v = self._role_env("REASONING", role)
        if v:
            return v.lower()
        if self.reasoning:
            return "medium"
        return ROLE_REASONING.get(role) or ROLE_REASONING.get(ROLE_GROUPS.get(role, ""), "off")

    def enabled_critics(self) -> list[str]:
        v = self.critics.strip().lower()
        if v in ("", "none", "0", "off", "нет"):
            return []
        if v in ("all", "1", "on", "все"):
            return list(CRITIC_NAMES)
        return [c for c in (x.strip() for x in v.split(",")) if c in CRITIC_NAMES]

    def models_by_role(self) -> dict:
        return {r: self.role_model(r) for r in ROLE_TITLES}

    def passport_models(self) -> dict:
        return {r: self.role_model(r) for r in PASSPORT_ROLES}

    def run_snapshot(self) -> dict:
        """Всё, что влияет на ответы, — фиксируется в run_config.json до запуска."""
        roles = sorted(ROLE_TITLES)
        return {**self.public(), "provider": self.provider, "temperature": self.temperature,
                "roles": {r: {"model": self.role_model(r), "provider": self.role_provider(r),
                              "fallback_model": self.role_fallback(r), "max_tokens": self.role_max_tokens(r),
                              "reasoning": self.role_reasoning(r)} for r in roles},
                "critics": self.enabled_critics(), "guard_llm": self.guard_llm,
                "question_retries": self.question_retries, "escalate_unknown": self.escalate_unknown,
                "passport_repair": self.passport_repair, "passport_checklist": self.passport_checklist,
                "full_context_max_tokens": self.full_context_max_tokens,
                "read_mode": self.read_mode, "strict_coverage": self.strict_coverage,
                "use_passport": self.use_passport, "model_context_tokens": self.model_context_tokens,
                "context_budget_ratio": self.context_budget_ratio, "block_max_chars": self.block_max_chars,
                "chunk_tokens": self.chunk_tokens, "chunk_overlap_tokens": self.chunk_overlap_tokens,
                "window_tokens": self.window_tokens, "retrieval_top_k": self.retrieval_top_k,
                "prefetch_limit": self.prefetch_limit, "rrf_k": self.rrf_k, "qdrant_url": self.qdrant_url,
                "max_calls_per_question": self.max_calls_per_question,
                "question_deadline": self.question_deadline, "repair_rounds": self.repair_rounds}

    def public(self) -> dict:
        """То, что можно показать в интерфейсе: без ключа."""
        return {
            "model": self.model,
            "models_by_role": {r: m for r, m in self.models_by_role().items() if m != self.model},
            "embedding_model": self.embedding_model,
            "api_key_configured": bool(self.api_key),
            "reasoning": self.reasoning,
            "question_concurrency": self.question_concurrency,
            "max_calls_per_question": self.max_calls_per_question,
            "read_mode": self.read_mode,
            "use_passport": self.use_passport,
            "strict_coverage": self.strict_coverage,
            "qdrant_url": self.qdrant_url,
        }


settings = Settings()
