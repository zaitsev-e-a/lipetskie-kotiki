"""Голосовой ввод: GigaAM v3 e2e_rnnt локально на CPU. Аудио не уходит в OpenRouter.

Браузер сам записывает WAV 16 кГц моно, поэтому сервер ничего не перекодирует.
Модель загружается при первом обращении и переиспользуется (кеш весов: ~/.cache/gigaam).
Установка: install_voice.bat (Windows) или pip install --upgrade -r requirements-voice.txt.
В пакете gigaam с PyPI (0.1.0) моделей v3 нет; в репозитории GitHub — есть (v3_e2e_rnnt и др.).
По README GigaAM: нужен ffmpeg в PATH, Python >= 3.10, .transcribe — только для записей до 25 секунд.
"""
from __future__ import annotations

import os
import shutil
import struct
import tempfile
import threading
import time

from .config import settings

MAX_SECONDS = 25
_model = None
_lock = threading.Lock()
_error = ""
_load_seconds: float | None = None


class VoiceError(RuntimeError):
    pass


def status() -> dict:
    try:
        import importlib.util
        installed = importlib.util.find_spec("gigaam") is not None
    except Exception:
        installed = False
    return {"installed": installed, "version": gigaam_version() if installed else "",
            "ffmpeg": shutil.which("ffmpeg") is not None, "loaded": _model is not None,
            "load_seconds": _load_seconds, "model": settings.asr_model, "error": _error}


def _wav_seconds(data: bytes) -> float:
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise VoiceError("ожидается WAV (браузер записывает его сам)")
    pos, rate, block = 12, 16000, 2
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        if cid == b"fmt ":
            _, ch, rate, _, block, _ = struct.unpack("<HHIIHH", data[pos + 8:pos + 24])
        elif cid == b"data":
            return size / max(1, rate * block)
        pos += 8 + size + (size & 1)
    raise VoiceError("в WAV нет аудиоданных")


INSTALL_HINT = ("Запустите install_voice.bat в папке программы: он ставит torch (CPU) и GigaAM с GitHub "
                "(в пакете gigaam с PyPI моделей v3 нет) и проверяет ffmpeg. Вручную: "
                "pip install --upgrade -r requirements-voice.txt")
FFMPEG_HINT = ("GigaAM читает аудио через ffmpeg, а его нет в PATH. Windows: winget install Gyan.FFmpeg, "
               "затем откройте новое окно и снова запустите run.bat")


def gigaam_version() -> str:
    try:
        from importlib.metadata import version
        return version("gigaam")
    except Exception:
        return ""


def _gigaam_models(gigaam, exc: Exception | None = None) -> str:
    """Модели, которые знает установленный пакет gigaam (имя атрибута в разных версиях может отличаться)."""
    for attr in ("_MODEL_NAMES", "MODEL_NAMES", "_MODELS", "MODELS"):
        names = getattr(gigaam, attr, None)
        if isinstance(names, (list, tuple, set, dict)) and names:
            return ", ".join(sorted(map(str, names)))
    fn = getattr(gigaam, "list_models", None)
    if callable(fn):
        try:
            return ", ".join(map(str, fn()))
        except Exception:
            pass
    return str(exc or "").strip()[:300]  # сообщение load_model обычно само перечисляет модели


def _load_hf():
    """Запасной путь: официальная интеграция GigaAM-v3 в transformers (ai-sage/GigaAM-v3)."""
    from transformers import AutoModel  # необязательная зависимость
    revision = settings.asr_model.removeprefix("v3_")
    return AutoModel.from_pretrained("ai-sage/GigaAM-v3", revision=revision, trust_remote_code=True)


def _load():
    global _model, _error, _load_seconds
    with _lock:
        if _model is not None:
            return _model
        t0 = time.time()
        errors = []
        reason = ""
        try:
            import gigaam  # тяжёлая зависимость, импортируем лениво
            try:
                _model = gigaam.load_model(settings.asr_model)
            except Exception as e:
                ver = gigaam_version() or "неизвестной версии"
                errors.append(f"gigaam {ver}: {type(e).__name__}: {e}")
                # список моделей: из атрибута пакета или из текста ошибки (без имени, которое мы просили)
                known = (_gigaam_models(gigaam) or str(e)).replace(settings.asr_model, "")
                no_v3 = "v3" not in known and ("v1_" in known or "v2_" in known)
                if settings.asr_model.startswith("v3_") and no_v3:
                    reason = (f"установлен gigaam {ver} без моделей v3 (это версия с PyPI); "
                              f"доступные модели: {_gigaam_models(gigaam, e)}")
                else:
                    reason = f"gigaam {ver} не загрузил {settings.asr_model}: {e}"
        except ImportError as e:
            errors.append(f"пакет gigaam не установлен ({e})")
            reason = "пакет gigaam не установлен"
        if _model is None and settings.asr_model.startswith("v3_"):
            try:
                _model = _load_hf()
            except ImportError:
                errors.append("transformers не установлен")
            except Exception as e:
                errors.append(f"Hugging Face: {type(e).__name__}: {e}")
        if _model is None:
            _error = f"не удалось загрузить {settings.asr_model}: {reason or errors[-1]}. {INSTALL_HINT}"
            print("[voice] " + " | ".join(errors))
            raise VoiceError(_error)
        _error = ""
        _load_seconds = round(time.time() - t0, 1)
        return _model


def transcribe(wav: bytes) -> dict:
    sec = _wav_seconds(wav)
    if sec > MAX_SECONDS:
        raise VoiceError(f"запись {sec:.0f} с длиннее {MAX_SECONDS} с — запишите вопрос короче")
    if sec < 0.3:
        raise VoiceError("запись слишком короткая")
    model = _load()
    fd, path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(wav)
        res = model.transcribe(path)
        text = getattr(res, "text", None) or (res.get("transcription") if isinstance(res, dict) else None) or res
        return {"text": str(text).strip(), "seconds": round(sec, 1), "model": settings.asr_model}
    except VoiceError:
        raise
    except FileNotFoundError as e:  # subprocess не нашёл ffmpeg
        if shutil.which("ffmpeg") is None:
            raise VoiceError(f"ошибка распознавания: {e}. {FFMPEG_HINT}") from e
        raise VoiceError(f"ошибка распознавания: {e}") from e
    except Exception as e:
        hint = f" {FFMPEG_HINT}" if "ffmpeg" in str(e).lower() and shutil.which("ffmpeg") is None else ""
        raise VoiceError(f"ошибка распознавания: {e}.{hint}") from e
    finally:
        try:
            os.remove(path)  # временная запись удаляется сразу
        except OSError:
            pass
