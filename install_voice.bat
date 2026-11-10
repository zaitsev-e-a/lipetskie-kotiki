@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Липецкие котики — установка голосового ввода GigaAM v3

rem --- ищем настоящий Python так же, как run.bat (заглушка из Microsoft Store не подходит)
set PY=
py -3 --version >nul 2>nul && set PY=py -3
if not defined PY (
  python --version >nul 2>nul && set PY=python
)
if not defined PY (
  echo.
  echo  [!] Python не найден.
  echo      Установите Python 3.10+ с https://www.python.org/downloads/
  echo      и при установке отметьте галочку "Add python.exe to PATH".
  echo      Затем снова запустите install_voice.bat.
  echo.
  pause
  exit /b 1
)
echo Найден:
%PY% --version

rem --- GigaAM требует Python 3.10 или новее (requires-python в pyproject.toml)
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
  echo.
  echo  [!] Для GigaAM нужен Python 3.10 или новее. Установите его с https://www.python.org/downloads/
  echo.
  pause
  exit /b 1
)

echo.
echo  Шаг 1/3. Пакеты: torch CPU, torchaudio, GigaAM с GitHub.
echo  Это несколько сотен мегабайт, установка займёт несколько минут.
echo.
%PY% -m pip install --upgrade pip
rem gigaam с PyPI 0.1.0 без моделей v3 — удаляем его, чтобы точно встал пакет с GitHub
%PY% -m pip uninstall -y gigaam >nul 2>nul
%PY% -m pip install --upgrade -r requirements-voice.txt
if errorlevel 1 (
  echo.
  echo  [!] Пакеты не установились. Проверьте интернет и пришлите текст ошибки выше.
  echo.
  pause
  exit /b 1
)

echo.
echo  Шаг 2/3. ffmpeg — GigaAM читает аудио через него.
ffmpeg -version >nul 2>nul
if errorlevel 1 (
  echo.
  echo  [!] ffmpeg не найден в PATH. Установите его командой:
  echo.
  echo        winget install Gyan.FFmpeg
  echo.
  echo      Затем закройте это окно и откройте НОВОЕ: PATH обновляется только в новых окнах.
  echo      После этого снова запустите install_voice.bat - torch повторно не скачивается.
  echo.
) else (
  echo  ffmpeg найден.
)

echo.
echo  Шаг 3/3. Проверка: загрузка модели и распознавание тестового звука.
echo  В первый раз скачиваются веса модели, подождите.
echo.
%PY% tools\check_voice.py
set RC=%errorlevel%
echo.
if "%RC%"=="0" (
  echo  Готово. Запустите run.bat - кнопка микрофона в диалоге заработает.
) else (
  echo  Голосовой ввод пока не готов - причина и подсказка выше.
)
echo.
pause
exit /b %RC%
