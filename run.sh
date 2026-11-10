#!/usr/bin/env sh
# Ubuntu 24.04 / Linux / macOS. Первый запуск создаёт .venv и ставит зависимости.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
[ -d .venv ] || $PY -m venv .venv
. .venv/bin/activate
python -c "import langgraph" 2>/dev/null || python -m pip install -r requirements.txt
[ -f .env ] || { cp .env.example .env; echo "Создан .env — впишите OPENROUTER_API_KEY и запустите снова"; exit 1; }
if command -v docker >/dev/null 2>&1; then
  docker compose up -d qdrant || echo "[!] Qdrant не запущен — поиск пойдёт по локальному индексу в памяти"
else
  echo "[!] Docker не найден — поиск пойдёт по локальному индексу в памяти (см. README)"
fi
exec python app.py
