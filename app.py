"""Запуск сайта: python app.py → http://localhost:8000"""
import sys

if sys.version_info < (3, 10):
    sys.exit(f"Нужен Python 3.10 или новее (рекомендуется 3.12), сейчас {sys.version.split()[0]}")

try:
    import langgraph  # noqa: F401
    if getattr(langgraph, "__stub__", False):
        raise ImportError("подключена заглушка вместо LangGraph")
except ImportError as e:
    sys.exit("Не установлен LangGraph — граф вопроса без него не работает.\n"
             f"  ({e})\n  Установите зависимости:  python -m pip install -r requirements.txt")

from opora.server import serve

if __name__ == "__main__":
    serve()
