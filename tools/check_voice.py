"""Проверка голосового ввода: загружает GigaAM и распознаёт 2 секунды тона (WAV 16 кГц моно).

python tools/check_voice.py      → код 0: GigaAM готов; код 1: понятная ошибка и что сделать.
Первый запуск скачивает веса модели (кеш ~/.cache/gigaam), поэтому может идти несколько минут.
"""
from __future__ import annotations

import io
import math
import shutil
import struct
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def tone_wav(seconds: float = 2.0, rate: int = 16000, hz: float = 440.0) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        n = int(seconds * rate)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * hz * i / rate)))
                               for i in range(n)))
    return buf.getvalue()


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    print(f"Python {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        print("[!] GigaAM требует Python 3.10 или новее — установите его с https://www.python.org/downloads/")
        return 1
    from opora import voice
    from opora.config import settings

    if shutil.which("ffmpeg") is None:
        print("[!] ffmpeg не найден в PATH. Windows: winget install Gyan.FFmpeg — затем откройте НОВОЕ окно.")
    st = voice.status()
    if st["installed"]:
        print(f"Пакет gigaam: {st['version'] or 'версия неизвестна'}")
    print(f"Модель: {settings.asr_model} (первый запуск скачивает веса, подождите)…", flush=True)

    t0 = time.time()
    try:
        voice._load()
        t_load = time.time() - t0
        t1 = time.time()
        res = voice.transcribe(tone_wav())
        t_asr = time.time() - t1
    except voice.VoiceError as e:
        print(f"\n[!] Голосовой ввод не готов: {e}")
        return 1
    except Exception as e:  # неожиданная ошибка torch/gigaam — без трассировки, но с типом
        print(f"\n[!] Голосовой ввод не готов: {type(e).__name__}: {e}\n    {voice.INSTALL_HINT}")
        return 1
    print(f"\nGigaAM готов: модель {res['model']}, время загрузки {t_load:.1f} с, "
          f"распознавание 2 с тона за {t_asr:.1f} с.")
    print(f"Текст на тестовом тоне: «{res['text']}» (для тона без речи пустой текст — норма).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
