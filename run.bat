@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Липецкие котики — ИИ, который не обманывает

rem --- ищем настоящий Python (заглушка из Microsoft Store не подходит)
set PY=
py -3 --version >nul 2>nul && set PY=py -3
if not defined PY (
  python --version >nul 2>nul && set PY=python
)
if not defined PY (
  echo.
  echo  [!] Python не найден.
  echo      Установите Python 3.12 с https://www.python.org/downloads/
  echo      и при установке отметьте галочку "Add python.exe to PATH".
  echo      Затем снова запустите run.bat.
  echo.
  pause
  exit /b 1
)
echo Найден:
%PY% --version

rem --- файл настроек
for %%A in (.env) do if %%~zA==0 del .env
if not exist .env (
  copy .env.example .env >nul
  echo.
  echo  Создан файл .env. Впишите ключ после OPENROUTER_API_KEY= , сохраните,
  echo  закройте блокнот — сервер запустится сам.
  notepad .env
)

rem --- зависимости (LangGraph и др.) — один раз
%PY% -c "import langgraph" >nul 2>nul || (
  echo Устанавливаю зависимости из requirements.txt ...
  %PY% -m pip install -r requirements.txt
)

rem --- Qdrant в Docker (если установлен Docker Desktop)
docker --version >nul 2>nul && (
  docker compose up -d qdrant || echo  [!] Qdrant не запущен: поиск пойдёт по локальному индексу в памяти
) || echo  [!] Docker не найден: поиск пойдёт по локальному индексу в памяти, см. README

rem --- сервер сам откроет браузер, когда будет готов
set OPEN_BROWSER=1
%PY% app.py
echo.
echo  Сервер остановлен. Если выше есть ошибка — пришлите её текст.
pause
