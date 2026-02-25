#!/usr/bin/env python3
"""
Telegram бот: аудио → Mermaid-диаграмма (только OpenAI).

Поток:
  аудио → Whisper (транскрипция) → GPT-4o (генерация Mermaid) →
  → рендер (mmdc / mermaid.ink) → если ошибка → GPT-4o (исправление) → ...
"""

import base64
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

import httpx
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

# ── Логирование ───────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Конфигурация ──────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
OPENAI_API_KEY   = os.environ["OPENAI_API_KEY"]
GPT_MODEL        = os.environ.get("GPT_MODEL", "gpt-4o")
MAX_FIX_ATTEMPTS = int(os.environ.get("MAX_FIX_ATTEMPTS", "5"))
PUPPETEER_CONFIG = os.environ.get("PUPPETEER_CONFIG_PATH", "puppeteer-config.json")

client = OpenAI(api_key=OPENAI_API_KEY)


# ════════════════════════════════════════════════════════════════════════════════
# ТРАНСКРИПЦИЯ (Whisper)
# ════════════════════════════════════════════════════════════════════════════════

def transcribe_audio(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
        )
    return result.text.strip()


# ════════════════════════════════════════════════════════════════════════════════
# ГЕНЕРАЦИЯ И ИСПРАВЛЕНИЕ MERMAID (GPT-4o)
# ════════════════════════════════════════════════════════════════════════════════

_GENERATE_SYSTEM = """\
Ты эксперт по Mermaid-диаграммам. Получив текст, создаёшь наиболее подходящую \
Mermaid-диаграмму. Возвращаешь ТОЛЬКО блок кода — ничего лишнего.

Правила синтаксиса:
- ID узлов: только буквы/цифры/подчёркивания, без пробелов.
- Метки со спецсимволами (скобки, двоеточие, апостроф и т.д.) — в двойных кавычках: A["Метка (пример)"].
- Стрелки строго по документации: -->, ---, -.-> и т.д.
- Типы диаграмм: graph/flowchart, sequenceDiagram, classDiagram, stateDiagram-v2, mindmap, timeline, erDiagram."""

_FIX_SYSTEM = """\
Ты эксперт по исправлению Mermaid-диаграмм. Получаешь код и ошибку рендеринга — \
возвращаешь ТОЛЬКО исправленный блок кода, без пояснений.

Чеклист:
1. Верное ключевое слово типа диаграммы.
2. ID узлов — только буквенно-цифровые + подчёркивание.
3. Метки со спецсимволами — в двойных кавычках.
4. Корректный синтаксис стрелок.
5. Все упоминаемые узлы определены.
6. Нет незакрытых скобок или кавычек."""


def _extract_mermaid(text: str) -> str:
    m = re.search(r"```mermaid\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _gpt(system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=2048,
        temperature=0.2,
    )
    return resp.choices[0].message.content


def generate_mermaid(transcript: str) -> str:
    user_msg = (
        f"Создай Mermaid-диаграмму по следующей транскрипции:\n\n{transcript}"
    )
    return _extract_mermaid(_gpt(_GENERATE_SYSTEM, user_msg))


def fix_mermaid(code: str, error: str) -> str:
    user_msg = (
        f"Диаграмма не рендерится. Ошибка:\n{error}\n\n"
        f"Код:\n```mermaid\n{code}\n```"
    )
    return _extract_mermaid(_gpt(_FIX_SYSTEM, user_msg))


# ════════════════════════════════════════════════════════════════════════════════
# РЕНДЕРИНГ MERMAID
# ════════════════════════════════════════════════════════════════════════════════

def _render_mmdc(code: str, out: str) -> tuple[bool, str]:
    """Рендер через локальный mermaid-cli (mmdc)."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".mmd", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        inp = f.name

    cmd = ["mmdc", "-i", inp, "-o", out, "-b", "white", "--quiet"]
    if Path(PUPPETEER_CONFIG).exists():
        cmd.extend(["-p", PUPPETEER_CONFIG])

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
        if r.returncode == 0 and Path(out).exists() and Path(out).stat().st_size > 100:
            return True, ""
        err = (r.stderr or r.stdout or f"exit {r.returncode}").strip()
        return False, err
    except FileNotFoundError:
        return False, "mmdc_not_found"
    except subprocess.TimeoutExpired:
        return False, "mmdc timeout (40 с)"
    finally:
        Path(inp).unlink(missing_ok=True)


def _render_ink(code: str, out: str) -> tuple[bool, str]:
    """Рендер через mermaid.ink API (fallback)."""
    payload = base64.urlsafe_b64encode(code.encode()).decode()
    url = f"https://mermaid.ink/img/{payload}?bgColor=white"
    try:
        r = httpx.get(url, timeout=25, follow_redirects=True)
        ct = r.headers.get("content-type", "")
        if r.status_code == 200 and "image" in ct and len(r.content) > 100:
            Path(out).write_bytes(r.content)
            return True, ""
        return False, f"mermaid.ink HTTP {r.status_code}: {r.text[:300]}"
    except httpx.TimeoutException:
        return False, "mermaid.ink timeout"
    except Exception as e:
        return False, f"mermaid.ink: {e}"


def render_mermaid(code: str, out: str) -> tuple[bool, str]:
    """Пробует mmdc, при неудаче — mermaid.ink."""
    ok, err = _render_mmdc(code, out)
    if ok:
        return True, ""
    if err == "mmdc_not_found":
        logger.info("mmdc не найден → mermaid.ink")
    else:
        logger.warning("mmdc: %s → mermaid.ink", err)
    return _render_ink(code, out)


# ════════════════════════════════════════════════════════════════════════════════
# TELEGRAM ОБРАБОТЧИКИ
# ════════════════════════════════════════════════════════════════════════════════

async def on_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message

    if msg.voice:
        tg_file = await msg.voice.get_file()
        suffix = ".ogg"
    elif msg.audio:
        tg_file = await msg.audio.get_file()
        suffix = Path(msg.audio.file_name or "audio.mp3").suffix or ".mp3"
    elif msg.document and "audio" in (msg.document.mime_type or ""):
        tg_file = await msg.document.get_file()
        suffix = Path(msg.document.file_name or "audio.mp3").suffix or ".mp3"
    else:
        await msg.reply_text("Пожалуйста, отправьте аудио-файл или голосовое сообщение.")
        return

    status = await msg.reply_text("⏳ Загружаю аудио…")

    async def upd(text: str) -> None:
        try:
            await status.edit_text(text)
        except Exception:
            pass

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = os.path.join(tmp, f"audio{suffix}")
        await tg_file.download_to_drive(audio_path)

        # 1. Транскрипция
        await upd("🎙️ Транскрибирую аудио (Whisper)…")
        try:
            transcript = transcribe_audio(audio_path)
        except Exception as e:
            logger.exception("Transcription error")
            await upd(f"❌ Ошибка транскрипции: {e}")
            return

        logger.info("Transcript (%d chars): %s…", len(transcript), transcript[:80])
        preview = transcript[:300] + ("…" if len(transcript) > 300 else "")
        await upd(f"✅ Транскрипция:\n{preview}\n\n⏳ Генерирую диаграмму (GPT-4o)…")

        # 2. Генерация Mermaid
        try:
            mermaid_code = generate_mermaid(transcript)
        except Exception as e:
            logger.exception("Generation error")
            await upd(f"❌ Ошибка генерации: {e}")
            return

        logger.info("Generated:\n%s", mermaid_code)

        # 3. Цикл рендеринга / исправления
        out_path = os.path.join(tmp, "diagram.png")

        for attempt in range(1, MAX_FIX_ATTEMPTS + 1):
            await upd(f"🔄 Рендеринг (попытка {attempt}/{MAX_FIX_ATTEMPTS})…")

            ok, error = render_mermaid(mermaid_code, out_path)

            if ok:
                await upd(
                    f"✅ Готово! (попытка {attempt})\n\n"
                    f"Mermaid-код:\n```\n{mermaid_code}\n```"
                )
                with open(out_path, "rb") as img:
                    await msg.reply_photo(img, caption="📊 Mermaid-диаграмма")
                return

            logger.warning("Attempt %d failed: %s", attempt, error)

            if attempt < MAX_FIX_ATTEMPTS:
                await upd(
                    f"⚠️ Попытка {attempt} не удалась:\n{error[:200]}\n\n"
                    f"🔧 GPT-4o исправляет синтаксис…"
                )
                try:
                    mermaid_code = fix_mermaid(mermaid_code, error)
                    logger.info("Fixed:\n%s", mermaid_code)
                except Exception as e:
                    await upd(f"❌ Ошибка исправления: {e}")
                    return

        await upd(
            f"❌ Не удалось отрендерить за {MAX_FIX_ATTEMPTS} попыток.\n\n"
            f"Последний код:\n```\n{mermaid_code}\n```"
        )


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Привет! Отправь голосовое сообщение или аудио-файл.\n\n"
        "Я транскрибирую его через Whisper, сгенерирую Mermaid-диаграмму "
        "через GPT-4o и автоматически исправлю ошибки, если диаграмма "
        "не отрендерится с первого раза."
    )


def main() -> None:
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(
        MessageHandler(
            filters.VOICE | filters.AUDIO | filters.Document.AUDIO,
            on_audio,
        )
    )
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    logger.info("Bot running…")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
