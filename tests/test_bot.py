"""
Тесты для bot.py.

Все внешние зависимости (OpenAI, httpx, subprocess, Telegram) мокируются.
"""
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Выставляем фейковые переменные окружения ДО импорта bot,
# чтобы он не упал с KeyError и не обратился к реальному OpenAI.
os.environ.setdefault("TELEGRAM_TOKEN", "fake_token")
os.environ.setdefault("OPENAI_API_KEY", "fake_api_key")

import bot  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════════
# _extract_mermaid
# ════════════════════════════════════════════════════════════════════════════════

class TestExtractMermaid:
    def test_mermaid_code_block(self):
        text = "```mermaid\ngraph TD\n  A --> B\n```"
        assert bot._extract_mermaid(text) == "graph TD\n  A --> B"

    def test_generic_code_block(self):
        text = "```\ngraph TD\n  A --> B\n```"
        assert bot._extract_mermaid(text) == "graph TD\n  A --> B"

    def test_plain_text_returned_as_is(self):
        text = "graph TD\n  A --> B"
        assert bot._extract_mermaid(text) == "graph TD\n  A --> B"

    def test_prefers_mermaid_block_over_generic(self):
        text = "```\nignored\n```\n```mermaid\ngraph TD\n  A --> B\n```"
        assert bot._extract_mermaid(text) == "graph TD\n  A --> B"

    def test_strips_surrounding_whitespace(self):
        text = "```mermaid\n\n  graph TD\n  A --> B\n\n```"
        assert bot._extract_mermaid(text) == "graph TD\n  A --> B"

    def test_multiline_diagram(self):
        code = "sequenceDiagram\n  Alice->>Bob: Hi\n  Bob-->>Alice: Hello"
        text = f"```mermaid\n{code}\n```"
        assert bot._extract_mermaid(text) == code


# ════════════════════════════════════════════════════════════════════════════════
# generate_mermaid / fix_mermaid
# ════════════════════════════════════════════════════════════════════════════════

class TestGenerateMermaid:
    def test_calls_gpt_and_extracts_code(self):
        fake = "```mermaid\ngraph TD\n  A --> B\n```"
        with patch("bot._gpt", return_value=fake) as mock_gpt:
            result = bot.generate_mermaid("Meeting notes about project flow")

        assert result == "graph TD\n  A --> B"
        mock_gpt.assert_called_once()
        system, user = mock_gpt.call_args[0]
        assert "транскрипц" in user.lower()

    def test_returns_plain_text_if_no_code_block(self):
        fake = "graph TD\n  A --> B"
        with patch("bot._gpt", return_value=fake):
            result = bot.generate_mermaid("some text")
        assert result == "graph TD\n  A --> B"


class TestFixMermaid:
    def test_calls_gpt_with_error_and_code(self):
        fake = "```mermaid\ngraph TD\n  A --> B\n```"
        with patch("bot._gpt", return_value=fake) as mock_gpt:
            result = bot.fix_mermaid("graph TD\n  A ->-> B", "Parse error at line 2")

        assert result == "graph TD\n  A --> B"
        system, user = mock_gpt.call_args[0]
        assert "Parse error" in user
        assert "A ->-> B" in user

    def test_uses_fix_system_prompt(self):
        with patch("bot._gpt", return_value="graph TD\n  A --> B") as mock_gpt:
            bot.fix_mermaid("bad code", "error")

        system, _ = mock_gpt.call_args[0]
        assert system is bot._FIX_SYSTEM


# ════════════════════════════════════════════════════════════════════════════════
# _render_mmdc
# ════════════════════════════════════════════════════════════════════════════════

class TestRenderMmdc:
    def _make_png(self) -> str:
        """Создаёт временный файл, имитирующий PNG (>100 байт)."""
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        f.write(b"\x89PNG" + b"x" * 200)
        f.close()
        return f.name

    def test_success(self):
        out = self._make_png()
        mock_run = MagicMock(returncode=0)
        with patch("bot.subprocess.run", return_value=mock_run):
            ok, err = bot._render_mmdc("graph TD\n  A --> B", out)
        assert ok is True
        assert err == ""
        Path(out).unlink(missing_ok=True)

    def test_nonzero_exit_returns_error(self):
        mock_run = MagicMock(returncode=1, stderr="Parse error", stdout="")
        with patch("bot.subprocess.run", return_value=mock_run):
            ok, err = bot._render_mmdc("bad code", "/tmp/nonexistent.png")
        assert ok is False
        assert "Parse error" in err

    def test_mmdc_not_installed(self):
        with patch("bot.subprocess.run", side_effect=FileNotFoundError):
            ok, err = bot._render_mmdc("graph TD\n  A --> B", "/tmp/out.png")
        assert ok is False
        assert err == "mmdc_not_found"

    def test_timeout(self):
        with patch(
            "bot.subprocess.run",
            side_effect=subprocess.TimeoutExpired("mmdc", 40),
        ):
            ok, err = bot._render_mmdc("graph TD\n  A --> B", "/tmp/out.png")
        assert ok is False
        assert "timeout" in err.lower()

    def test_empty_output_file_treated_as_failure(self):
        out = tempfile.mktemp(suffix=".png")
        Path(out).write_bytes(b"")            # файл есть, но пустой
        mock_run = MagicMock(returncode=0, stderr="", stdout="")
        with patch("bot.subprocess.run", return_value=mock_run):
            ok, err = bot._render_mmdc("graph TD\n  A --> B", out)
        assert ok is False
        Path(out).unlink(missing_ok=True)


# ════════════════════════════════════════════════════════════════════════════════
# _render_ink
# ════════════════════════════════════════════════════════════════════════════════

class TestRenderInk:
    def test_success_writes_file(self):
        out = tempfile.mktemp(suffix=".png")
        fake_bytes = b"\x89PNG" + b"x" * 200
        mock_resp = MagicMock(
            status_code=200,
            headers={"content-type": "image/png"},
            content=fake_bytes,
        )
        with patch("bot.httpx.get", return_value=mock_resp):
            ok, err = bot._render_ink("graph TD\n  A --> B", out)

        assert ok is True
        assert err == ""
        assert Path(out).read_bytes() == fake_bytes
        Path(out).unlink(missing_ok=True)

    def test_http_error_status(self):
        mock_resp = MagicMock(
            status_code=400,
            headers={"content-type": "text/plain"},
            text="Bad Request",
            content=b"",
        )
        with patch("bot.httpx.get", return_value=mock_resp):
            ok, err = bot._render_ink("bad code", "/tmp/out.png")
        assert ok is False
        assert "400" in err

    def test_non_image_content_type(self):
        mock_resp = MagicMock(
            status_code=200,
            headers={"content-type": "text/html"},
            text="Error page",
            content=b"x" * 200,
        )
        with patch("bot.httpx.get", return_value=mock_resp):
            ok, err = bot._render_ink("bad code", "/tmp/out.png")
        assert ok is False

    def test_timeout(self):
        import httpx as httpx_lib
        with patch("bot.httpx.get", side_effect=httpx_lib.TimeoutException("timeout")):
            ok, err = bot._render_ink("graph TD\n  A --> B", "/tmp/out.png")
        assert ok is False
        assert "timeout" in err.lower()

    def test_url_contains_base64_encoded_code(self):
        import base64
        code = "graph TD\n  A --> B"
        expected_payload = base64.urlsafe_b64encode(code.encode()).decode()
        captured_url = []

        def fake_get(url, **kwargs):
            captured_url.append(url)
            return MagicMock(
                status_code=400,
                headers={"content-type": "text/plain"},
                text="err",
                content=b"",
            )

        with patch("bot.httpx.get", side_effect=fake_get):
            bot._render_ink(code, "/tmp/out.png")

        assert expected_payload in captured_url[0]


# ════════════════════════════════════════════════════════════════════════════════
# render_mermaid (оркестратор)
# ════════════════════════════════════════════════════════════════════════════════

class TestRenderMermaid:
    def test_mmdc_success_skips_ink(self):
        with patch("bot._render_mmdc", return_value=(True, "")) as mmdc, \
             patch("bot._render_ink") as ink:
            ok, err = bot.render_mermaid("graph TD\n  A --> B", "/tmp/out.png")
        assert ok is True
        ink.assert_not_called()

    def test_mmdc_error_falls_back_to_ink(self):
        with patch("bot._render_mmdc", return_value=(False, "render error")), \
             patch("bot._render_ink", return_value=(True, "")) as ink:
            ok, err = bot.render_mermaid("graph TD\n  A --> B", "/tmp/out.png")
        assert ok is True
        ink.assert_called_once()

    def test_mmdc_not_found_falls_back_to_ink(self):
        with patch("bot._render_mmdc", return_value=(False, "mmdc_not_found")), \
             patch("bot._render_ink", return_value=(True, "")) as ink:
            ok, err = bot.render_mermaid("graph TD\n  A --> B", "/tmp/out.png")
        assert ok is True
        ink.assert_called_once()

    def test_both_backends_fail(self):
        with patch("bot._render_mmdc", return_value=(False, "mmdc error")), \
             patch("bot._render_ink", return_value=(False, "ink error")):
            ok, err = bot.render_mermaid("bad code", "/tmp/out.png")
        assert ok is False
        assert err == "ink error"


# ════════════════════════════════════════════════════════════════════════════════
# Telegram-обработчики (async)
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
class TestOnText:
    async def test_replies_with_greeting(self):
        update = MagicMock()
        update.message.reply_text = AsyncMock()
        await bot.on_text(update, MagicMock())
        update.message.reply_text.assert_called_once()

    async def test_reply_mentions_audio(self):
        update = MagicMock()
        update.message.reply_text = AsyncMock()
        await bot.on_text(update, MagicMock())
        text = update.message.reply_text.call_args[0][0]
        assert any(word in text.lower() for word in ("аудио", "голосов"))


@pytest.mark.asyncio
class TestOnAudio:
    def _make_update(self, *, voice=False, audio=False, doc_mime=None):
        msg = MagicMock()
        msg.reply_text = AsyncMock(return_value=MagicMock(edit_text=AsyncMock()))
        msg.reply_photo = AsyncMock()

        tg_file = AsyncMock()
        tg_file.download_to_drive = AsyncMock()

        if voice:
            msg.voice = MagicMock()
            msg.voice.get_file = AsyncMock(return_value=tg_file)
            msg.audio = None
            msg.document = None
        elif audio:
            msg.audio = MagicMock(file_name="song.mp3")
            msg.audio.get_file = AsyncMock(return_value=tg_file)
            msg.voice = None
            msg.document = None
        elif doc_mime:
            msg.document = MagicMock(
                mime_type=doc_mime, file_name="file.mp3"
            )
            msg.document.get_file = AsyncMock(return_value=tg_file)
            msg.voice = None
            msg.audio = None
        else:
            msg.voice = None
            msg.audio = None
            msg.document = None

        update = MagicMock()
        update.message = msg
        return update, msg

    async def test_non_audio_sends_error(self):
        update, msg = self._make_update()
        await bot.on_audio(update, MagicMock())
        msg.reply_text.assert_called_once()
        assert "аудио" in msg.reply_text.call_args[0][0].lower()

    async def test_non_audio_document_sends_error(self):
        update, msg = self._make_update(doc_mime="application/pdf")
        await bot.on_audio(update, MagicMock())
        msg.reply_text.assert_called_once()

    async def test_voice_message_sends_photo_on_success(self):
        update, msg = self._make_update(voice=True)

        with patch("bot.transcribe_audio", return_value="Meeting notes"), \
             patch("bot.generate_mermaid", return_value="graph TD\n  A --> B"), \
             patch("bot.render_mermaid", return_value=(True, "")), \
             patch("builtins.open", MagicMock(
                 __enter__=MagicMock(return_value=MagicMock()),
                 __exit__=MagicMock(return_value=False),
             )):
            await bot.on_audio(update, MagicMock())

        msg.reply_photo.assert_called_once()

    async def test_audio_file_sends_photo_on_success(self):
        update, msg = self._make_update(audio=True)

        with patch("bot.transcribe_audio", return_value="Notes"), \
             patch("bot.generate_mermaid", return_value="graph TD\n  A --> B"), \
             patch("bot.render_mermaid", return_value=(True, "")), \
             patch("builtins.open", MagicMock(
                 __enter__=MagicMock(return_value=MagicMock()),
                 __exit__=MagicMock(return_value=False),
             )):
            await bot.on_audio(update, MagicMock())

        msg.reply_photo.assert_called_once()

    async def test_transcription_error_reports_to_user(self):
        update, msg = self._make_update(voice=True)

        with patch("bot.transcribe_audio", side_effect=Exception("API error")):
            await bot.on_audio(update, MagicMock())

        # Статус-сообщение должно было обновиться с текстом ошибки
        status = msg.reply_text.return_value
        calls = [c[0][0] for c in status.edit_text.call_args_list]
        assert any("❌" in t or "ошибк" in t.lower() for t in calls)

    async def test_all_render_attempts_exhausted(self):
        update, msg = self._make_update(voice=True)

        with patch("bot.transcribe_audio", return_value="Notes"), \
             patch("bot.generate_mermaid", return_value="graph TD\n  A --> B"), \
             patch("bot.render_mermaid", return_value=(False, "render failed")), \
             patch("bot.fix_mermaid", return_value="graph TD\n  A --> B"):
            await bot.on_audio(update, MagicMock())

        msg.reply_photo.assert_not_called()
        status = msg.reply_text.return_value
        last_call = status.edit_text.call_args_list[-1][0][0]
        assert "❌" in last_call
