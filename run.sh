#!/bin/bash
set -e
cd "$(dirname "$0")"

# ── Проверка .env ──────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "Файл .env создан из шаблона. Заполни токены и запусти снова:"
    echo ""
    echo "  nano .env          # или любой другой редактор"
    echo "  ./run.sh"
    echo ""
    exit 1
fi

if grep -qE "^TELEGRAM_TOKEN=your_telegram|^OPENAI_API_KEY=sk-\.\.\." .env; then
    echo ""
    echo "Открой .env и замени плейсхолдеры на настоящие токены:"
    echo ""
    echo "  TELEGRAM_TOKEN=<токен от @BotFather>"
    echo "  OPENAI_API_KEY=<ключ с platform.openai.com>"
    echo ""
    exit 1
fi

# ── Виртуальное окружение ──────────────────────────────────────────────────────
if [ ! -d "venv" ]; then
    echo "Создаю виртуальное окружение…"
    python3 -m venv venv
fi

# ── Зависимости ───────────────────────────────────────────────────────────────
venv/bin/pip install -q -r requirements.txt

# ── Запуск ────────────────────────────────────────────────────────────────────
set -a && source .env && set +a
exec venv/bin/python bot.py
