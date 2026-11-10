"""Типы данных и проверка JSON-ответов модели по схеме во время выполнения.

Все объекты — простые словари/датаклассы, сериализуемые в JSON: они лежат в состоянии
LangGraph, в trace.json и в паспортах. Версии схем фиксируются в снимке и run_config.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

EXTRACTION_VERSION = "ext-4.0"      # извлечение текста и канонизация
CHUNKING_VERSION = "chunk-4.0"      # блоки, чанки, окна
KNOWLEDGE_SCHEMA = "kn-4.0"         # паспорта, факты, связи, конфликты
INDEX_SCHEMA = "idx-4.0"            # точки Qdrant

READER_OUTCOMES = ("supported", "partial", "contradiction_candidate", "no_relevant_evidence", "uncertain")
FINAL_STATUSES = ("ответ", "расхождение", "нет в документах", "не знаю")
UNCERTAINTY_REASONS = ("insufficient_evidence", "incomplete_coverage", "unresolved_amendment",
                       "unresolved_conflict", "relevant_unreadable_source", "validation_incomplete",
                       "content_budget_exhausted")
CONFLICT_STATES = ("candidate", "confirmed", "different_conditions", "resolved_by_amendment", "unresolved")


@dataclass
class EvidenceRef:
    """Ссылка на точный срез канонического текста: canonical_text[start:end] == quote."""
    source_id: str
    snapshot_id: str
    package_id: str
    document_id: str
    file: str
    text_sha256: str
    start: int
    end: int
    quote: str
    page: int | None = None
    line: int = 0
    checker_ok: bool = False          # пройдёт ли официальный check.py (видимость файла оценщику)
    superseded_by: dict | None = None  # поправка, отменившая эту редакцию
    context: str = ""                 # строка документа целиком — для панели доказательств

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Block:
    """Структурный блок: непрерывный срез документа (абзац или часть длинного абзаца)."""
    source_id: str
    document_id: str
    file: str
    start: int
    end: int
    line: int
    tokens: int
    page: int | None = None
    section: str = ""


@dataclass
class Chunk:
    """Единица индекса Qdrant: подряд идущие блоки одного документа с перекрытием."""
    chunk_id: str
    document_id: str
    file: str
    start: int
    end: int
    block_ids: list[str]
    tokens: int
    section: str = ""
    page: int | None = None


@dataclass
class Window:
    """Единица чтения reader: непересекающиеся блоки (один документ или пачка маленьких)."""
    window_id: str
    block_ids: list[str]
    files: list[str]
    tokens: int


@dataclass
class SchemaError(Exception):
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "integer": int,
          "number": (int, float), "null": type(None)}


def validate_schema(data, schema: dict, path: str = "$") -> None:
    """Подмножество JSON Schema: type, required, properties, items, enum. Без внешних библиотек.

    Используется после ответа модели всегда — даже если маршрут OpenRouter не поддержал
    json_schema и пришлось откатиться на json_object или текст.
    """
    t = schema.get("type")
    if t:
        types = t if isinstance(t, list) else [t]
        ok = any(isinstance(data, _TYPES[x]) and not (x in ("integer", "number") and isinstance(data, bool))
                 for x in types if x in _TYPES)
        if not ok:
            raise SchemaError(path, f"ожидался тип {t}, получен {type(data).__name__}")
    if "enum" in schema and data not in schema["enum"]:
        raise SchemaError(path, f"значение {data!r} не из {schema['enum']}")
    if isinstance(data, dict):
        for k in schema.get("required", []):
            if k not in data:
                raise SchemaError(path, f"нет обязательного поля {k!r}")
        for k, sub in (schema.get("properties") or {}).items():
            if k in data and data[k] is not None:
                validate_schema(data[k], sub, f"{path}.{k}")
    if isinstance(data, list) and "items" in schema:
        for i, x in enumerate(data):
            validate_schema(x, schema["items"], f"{path}[{i}]")
