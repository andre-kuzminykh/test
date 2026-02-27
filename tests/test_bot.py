"""
Тесты для bot.py (HuggingFace Daily Papers Bot).

Все внешние зависимости (OpenAI, httpx, Telegram) мокируются.
"""
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Фейковые env vars ДО импорта bot
os.environ.setdefault("TELEGRAM_TOKEN", "fake_token")
os.environ.setdefault("OPENAI_API_KEY", "fake_api_key")

import bot  # noqa: E402


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_paper(
    paper_id: str = "2401.00001",
    title: str = "Test Paper",
    abstract: str = "This is an abstract.",
    authors: list | None = None,
    published: str = "2024-01-15T00:00:00",
    summary: str | None = None,
) -> dict:
    p = {
        "paper": {
            "id": paper_id,
            "title": title,
            "abstract": abstract,
            "authors": authors or [{"name": "Alice"}, {"name": "Bob"}],
            "publishedAt": published,
        }
    }
    if summary is not None:
        p["_summary"] = summary
    return p


# ── fetch_papers ──────────────────────────────────────────────────────────────

class TestFetchPapers:
    @pytest.mark.asyncio
    async def test_returns_list_on_success(self):
        fake_data = [make_paper()]
        mock_resp = MagicMock(status_code=200)
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = fake_data

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("bot.httpx.AsyncClient", return_value=mock_client):
            result = await bot.fetch_papers("2024-01-15")

        assert result == fake_data

    @pytest.mark.asyncio
    async def test_returns_empty_list_for_non_list_response(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"error": "not found"}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("bot.httpx.AsyncClient", return_value=mock_client):
            result = await bot.fetch_papers("2024-01-15")

        assert result == []

    @pytest.mark.asyncio
    async def test_passes_date_as_query_param(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = []

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("bot.httpx.AsyncClient", return_value=mock_client):
            await bot.fetch_papers("2024-06-01")

        mock_client.get.assert_called_once_with(
            bot.HF_API, params={"date": "2024-06-01"}
        )


# ── summarize ─────────────────────────────────────────────────────────────────

class TestSummarize:
    @pytest.mark.asyncio
    async def test_returns_summary_from_openai(self):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "  Summary text.  "

        with patch.object(
            bot.openai_client.chat.completions, "create", new=AsyncMock(return_value=mock_resp)
        ):
            result = await bot.summarize("Title", "Abstract text")

        assert result == "Summary text."

    @pytest.mark.asyncio
    async def test_returns_fallback_for_empty_abstract(self):
        result = await bot.summarize("Title", "")
        assert result == "Аннотация недоступна."

    @pytest.mark.asyncio
    async def test_prompt_contains_title_and_abstract(self):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "Summary"
        captured = {}

        async def fake_create(**kwargs):
            captured["messages"] = kwargs["messages"]
            return mock_resp

        with patch.object(bot.openai_client.chat.completions, "create", side_effect=fake_create):
            await bot.summarize("My Title", "My Abstract")

        content = captured["messages"][0]["content"]
        assert "My Title" in content
        assert "My Abstract" in content


# ── card_text ─────────────────────────────────────────────────────────────────

class TestCardText:
    def test_contains_counter(self):
        paper = make_paper(summary="Summary here")
        text = bot.card_text(paper, 0, 5)
        assert "1 / 5" in text

    def test_contains_title(self):
        paper = make_paper(title="Amazing Research", summary="Summary")
        text = bot.card_text(paper, 0, 3)
        assert "Amazing Research" in text

    def test_contains_summary(self):
        paper = make_paper(summary="Key finding here")
        text = bot.card_text(paper, 0, 1)
        assert "Key finding here" in text

    def test_shows_up_to_three_authors(self):
        authors = [{"name": f"Author{i}"} for i in range(5)]
        paper = make_paper(authors=authors, summary="S")
        text = bot.card_text(paper, 0, 1)
        assert "et al." in text
        assert "Author0" in text
        assert "Author3" not in text

    def test_escapes_html_in_title(self):
        paper = make_paper(title="<b>bold</b>", summary="S")
        text = bot.card_text(paper, 0, 1)
        # The raw <b> tag should be escaped, not passed through as HTML
        # card_text wraps title in its own <b>...</b>, so check inner content
        assert "&lt;b&gt;" in text

    def test_shows_date(self):
        paper = make_paper(published="2024-03-22T10:00:00Z", summary="S")
        text = bot.card_text(paper, 0, 1)
        assert "2024-03-22" in text


# ── card_keyboard ─────────────────────────────────────────────────────────────

class TestCardKeyboard:
    def test_no_prev_button_on_first_card(self):
        kb = bot.card_keyboard(0, 5, "2401.00001")
        nav_row = kb.inline_keyboard[0]
        labels = [btn.text for btn in nav_row]
        assert "◀️" not in labels
        assert "▶️" in labels

    def test_no_next_button_on_last_card(self):
        kb = bot.card_keyboard(4, 5, "2401.00001")
        nav_row = kb.inline_keyboard[0]
        labels = [btn.text for btn in nav_row]
        assert "▶️" not in labels
        assert "◀️" in labels

    def test_both_buttons_in_middle(self):
        kb = bot.card_keyboard(2, 5, "2401.00001")
        nav_row = kb.inline_keyboard[0]
        labels = [btn.text for btn in nav_row]
        assert "◀️" in labels
        assert "▶️" in labels

    def test_counter_button_shows_position(self):
        kb = bot.card_keyboard(1, 5, "2401.00001")
        nav_row = kb.inline_keyboard[0]
        counter = next(b for b in nav_row if "/" in b.text)
        assert counter.text == "2/5"

    def test_open_button_has_correct_url(self):
        kb = bot.card_keyboard(0, 1, "2401.12345")
        open_row = kb.inline_keyboard[1]
        assert len(open_row) == 1
        assert "2401.12345" in open_row[0].url

    def test_prev_callback_data(self):
        kb = bot.card_keyboard(3, 5, "2401.00001")
        nav_row = kb.inline_keyboard[0]
        prev = next(b for b in nav_row if b.text == "◀️")
        assert prev.callback_data == "nav:2"

    def test_next_callback_data(self):
        kb = bot.card_keyboard(3, 5, "2401.00001")
        nav_row = kb.inline_keyboard[0]
        nxt = next(b for b in nav_row if b.text == "▶️")
        assert nxt.callback_data == "nav:4"


# ── cmd_start ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestCmdStart:
    async def test_replies_with_help(self):
        update = MagicMock()
        update.message.reply_text = AsyncMock()
        await bot.cmd_start(update, MagicMock())
        update.message.reply_text.assert_called_once()

    async def test_mentions_papers_command(self):
        update = MagicMock()
        update.message.reply_text = AsyncMock()
        await bot.cmd_start(update, MagicMock())
        text = update.message.reply_text.call_args[0][0]
        assert "/papers" in text


# ── cmd_papers ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestCmdPapers:
    def _make_ctx(self, chat_id: int = 123):
        ctx = MagicMock()
        ctx.args = []
        ctx.application.chat_data = {}
        ctx.bot.send_message = AsyncMock()
        return ctx

    async def test_uses_yesterday_when_no_args(self):
        update = MagicMock()
        update.message.reply_text = AsyncMock(
            return_value=MagicMock(edit_text=AsyncMock(), delete=AsyncMock())
        )
        update.effective_chat.id = 42
        ctx = self._make_ctx(42)
        ctx.args = []

        papers = [make_paper(summary="S")]
        with patch("bot.fetch_papers", new=AsyncMock(return_value=papers)), \
             patch("bot.summarize", new=AsyncMock(return_value="S")):
            await bot.cmd_papers(update, ctx)

    async def test_uses_provided_date_arg(self):
        update = MagicMock()
        update.message.reply_text = AsyncMock(
            return_value=MagicMock(edit_text=AsyncMock(), delete=AsyncMock())
        )
        update.effective_chat.id = 99
        ctx = self._make_ctx(99)
        ctx.args = ["2024-05-01"]

        papers = [make_paper(summary="S")]
        captured_date = []

        async def fake_fetch(date):
            captured_date.append(date)
            return papers

        with patch("bot.fetch_papers", side_effect=fake_fetch):
            await bot.cmd_papers(update, ctx)

        assert captured_date[0] == "2024-05-01"


# ── on_nav ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestOnNav:
    def _make_nav_update(self, data: str, chat_id: int = 1):
        query = MagicMock()
        query.data = data
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()

        update = MagicMock()
        update.callback_query = query
        update.effective_chat.id = chat_id
        return update, query

    async def test_noop_does_nothing(self):
        update, query = self._make_nav_update("noop")
        ctx = MagicMock()
        await bot.on_nav(update, ctx)
        query.edit_message_text.assert_not_called()

    async def test_nav_updates_message(self):
        papers = [make_paper(summary=f"S{i}") for i in range(3)]
        update, query = self._make_nav_update("nav:2", chat_id=7)
        ctx = MagicMock()
        ctx.application.chat_data = {7: {"papers": papers, "index": 0}}

        await bot.on_nav(update, ctx)

        query.edit_message_text.assert_called_once()
        kwargs = query.edit_message_text.call_args[1]
        assert "3 / 3" in kwargs["text"]

    async def test_stale_session_sends_error(self):
        update, query = self._make_nav_update("nav:1", chat_id=8)
        ctx = MagicMock()
        ctx.application.chat_data = {}

        await bot.on_nav(update, ctx)

        query.edit_message_text.assert_called_once()
        assert "устарела" in query.edit_message_text.call_args[0][0]

    async def test_out_of_bounds_index_ignored(self):
        papers = [make_paper(summary="S")]
        update, query = self._make_nav_update("nav:99", chat_id=9)
        ctx = MagicMock()
        ctx.application.chat_data = {9: {"papers": papers, "index": 0}}

        await bot.on_nav(update, ctx)

        query.edit_message_text.assert_not_called()
