#!/usr/bin/env python3
"""
HuggingFace Daily Papers — Telegram Bot

Команды:
  /start               — справка
  /papers              — статьи за вчера
  /papers YYYY-MM-DD   — статьи за указанную дату

Навигация кнопками ◀️ / ▶️.
Если задан CHAT_ID, каждый день в DAILY_HOUR UTC бот присылает статьи автоматически.
"""

import logging
import os
from datetime import datetime, timedelta, timezone, time as dt_time

import html

import httpx
from openai import AsyncOpenAI
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GPT_MODEL = os.environ.get("GPT_MODEL", "gpt-4o-mini")
CHAT_ID = os.environ.get("CHAT_ID")          # optional: for daily scheduled push
DAILY_HOUR = int(os.environ.get("DAILY_HOUR", "9"))  # UTC hour for daily push

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

HF_API = "https://huggingface.co/api/daily_papers"


# ── HuggingFace API ────────────────────────────────────────────────────────────

async def fetch_papers(date: str) -> list[dict]:
    """Fetch papers list from HuggingFace API for given date (YYYY-MM-DD)."""
    async with httpx.AsyncClient(timeout=30) as http:
        r = await http.get(HF_API, params={"date": date})
        r.raise_for_status()
        data = r.json()
    return data if isinstance(data, list) else []


# ── OpenAI Summarization ───────────────────────────────────────────────────────

async def summarize(title: str, abstract: str) -> str:
    """Generate 2-3 sentence plain-language summary in Russian."""
    if not abstract:
        return "Аннотация недоступна."
    resp = await openai_client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{
            "role": "user",
            "content": (
                "Напиши саммари научной статьи в 2-3 предложения простым языком на русском:\n"
                "— какую проблему решали\n"
                "— метод или подход\n"
                "— главный результат\n\n"
                f"Название: {title}\n\n"
                f"Аннотация: {abstract}"
            ),
        }],
        max_tokens=200,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


# ── Card UI ───────────────────────────────────────────────────────────────────

def card_text(paper: dict, idx: int, total: int) -> str:
    p = paper["paper"]
    title = html.escape(p["title"])
    summary = html.escape(paper.get("_summary", ""))
    authors = p.get("authors", [])
    names = html.escape(", ".join(a["name"] for a in authors[:3]))
    if len(authors) > 3:
        names += " et al."
    date_raw = p.get("publishedAt", "")[:10]
    return (
        f"<b>{idx + 1} / {total}</b>\n\n"
        f"<b>{title}</b>\n\n"
        f"{summary}\n\n"
        f"👥 <i>{names}</i>\n"
        f"📅 {date_raw}"
    )


def card_keyboard(idx: int, total: int, paper_id: str) -> InlineKeyboardMarkup:
    nav_row = []
    if idx > 0:
        nav_row.append(InlineKeyboardButton("◀️", callback_data=f"nav:{idx - 1}"))
    nav_row.append(InlineKeyboardButton(f"{idx + 1}/{total}", callback_data="noop"))
    if idx < total - 1:
        nav_row.append(InlineKeyboardButton("▶️", callback_data=f"nav:{idx + 1}"))
    open_btn = InlineKeyboardButton(
        "📖 Открыть статью",
        url=f"https://huggingface.co/papers/{paper_id}",
    )
    return InlineKeyboardMarkup([nav_row, [open_btn]])


# ── Core ──────────────────────────────────────────────────────────────────────

async def load_and_send(
    chat_id: int,
    date_str: str,
    context: ContextTypes.DEFAULT_TYPE,
    status_msg=None,
) -> None:
    """Fetch, summarize, and send the first paper card to the chat."""
    try:
        papers = await fetch_papers(date_str)
    except httpx.HTTPError as exc:
        text = f"❌ Ошибка загрузки: {exc}"
        if status_msg:
            await status_msg.edit_text(text)
        else:
            await context.bot.send_message(chat_id=chat_id, text=text)
        return

    if not papers:
        text = f"За {date_str} статей не найдено."
        if status_msg:
            await status_msg.edit_text(text)
        else:
            await context.bot.send_message(chat_id=chat_id, text=text)
        return

    if status_msg:
        await status_msg.edit_text(
            f"⏳ Генерирую саммари для {len(papers)} статей…"
        )

    for paper in papers:
        if "_summary" not in paper:
            p = paper["paper"]
            paper["_summary"] = await summarize(p["title"], p.get("abstract", ""))

    # Persist state per chat
    context.application.chat_data.setdefault(chat_id, {}).update(
        {"papers": papers, "index": 0, "date": date_str}
    )

    if status_msg:
        await status_msg.delete()

    paper = papers[0]
    await context.bot.send_message(
        chat_id=chat_id,
        text=card_text(paper, 0, len(papers)),
        parse_mode="HTML",
        reply_markup=card_keyboard(0, len(papers), paper["paper"]["id"]),
    )


# ── Handlers ──────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "<b>HuggingFace Daily Papers Bot</b> 🤖\n\n"
        "Команды:\n"
        "• /papers — статьи за вчера\n"
        "• /papers 2024-01-15 — статьи за конкретную дату\n\n"
        "Листай карточки кнопками ◀️ ▶️",
        parse_mode="HTML",
    )


async def cmd_papers(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.args:
        date_str = context.args[0]
    else:
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        date_str = yesterday.strftime("%Y-%m-%d")

    status = await update.message.reply_text(f"⏳ Загружаю статьи за {date_str}…")
    await load_and_send(update.effective_chat.id, date_str, context, status)


async def on_nav(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    if query.data == "noop":
        return

    chat_id = update.effective_chat.id
    state = context.application.chat_data.get(chat_id, {})
    papers = state.get("papers")

    if not papers:
        await query.edit_message_text("Сессия устарела — введите /papers заново.")
        return

    new_idx = int(query.data.split(":")[1])
    if not (0 <= new_idx < len(papers)):
        return

    state["index"] = new_idx
    paper = papers[new_idx]
    await query.edit_message_text(
        text=card_text(paper, new_idx, len(papers)),
        parse_mode="HTML",
        reply_markup=card_keyboard(new_idx, len(papers), paper["paper"]["id"]),
    )


# ── Scheduled job ─────────────────────────────────────────────────────────────

async def daily_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    if not CHAT_ID:
        return
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    date_str = yesterday.strftime("%Y-%m-%d")
    logger.info("Daily push: %s → chat %s", date_str, CHAT_ID)
    await load_and_send(int(CHAT_ID), date_str, context)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("papers", cmd_papers))
    app.add_handler(CallbackQueryHandler(on_nav, pattern=r"^(nav:\d+|noop)$"))

    if CHAT_ID:
        app.job_queue.run_daily(
            daily_job,
            time=dt_time(hour=DAILY_HOUR, minute=0, tzinfo=timezone.utc),
        )
        logger.info("Daily job: %02d:00 UTC → chat %s", DAILY_HOUR, CHAT_ID)

    logger.info("Bot started")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
