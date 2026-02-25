#!/bin/bash
set -e
cd "$(dirname "$0")"

# Создать venv если не существует
if [ ! -d "venv" ]; then
    echo "Создаю виртуальное окружение…"
    python3 -m venv venv
fi

# Установить/обновить зависимости
venv/bin/pip install -q -r requirements.txt

# Запустить бота
set -a && source .env && set +a
exec venv/bin/python bot.py
