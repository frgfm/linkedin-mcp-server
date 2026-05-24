"""Tests for LinkedInExtractor messaging helpers.

Focused on `_type_message_with_newlines`, which protects against issue #441
(multi-paragraph `send_message` bodies arriving as multiple separate sends
because `page.keyboard.type("\\n")` is treated as an Enter key press by
Patchright, and LinkedIn's composer submits on Enter).
"""

from unittest.mock import AsyncMock, MagicMock, call

import pytest

from linkedin_mcp_server.scraping.extractor import LinkedInExtractor


@pytest.fixture
def keyboard_mock() -> MagicMock:
    """Mock keyboard with `type` and `press` as AsyncMocks sharing a parent.

    Sharing a parent (`MagicMock()`) lets us assert call ordering via
    `mock_calls` on the parent — important here because the bug is about the
    interleaving of `type(...)` and `press("Shift+Enter")`.
    """
    keyboard = MagicMock()
    keyboard.type = AsyncMock()
    keyboard.press = AsyncMock()
    return keyboard


@pytest.fixture
def extractor(keyboard_mock: MagicMock) -> LinkedInExtractor:
    page = MagicMock()
    page.keyboard = keyboard_mock
    return LinkedInExtractor(page)


class TestTypeMessageWithNewlines:
    async def test_single_line_uses_type_only(
        self, extractor: LinkedInExtractor, keyboard_mock: MagicMock
    ) -> None:
        await extractor._type_message_with_newlines("Hello there")

        keyboard_mock.type.assert_awaited_once_with("Hello there", delay=15)
        keyboard_mock.press.assert_not_awaited()

    async def test_double_newline_splits_with_shift_enter(
        self, extractor: LinkedInExtractor, keyboard_mock: MagicMock
    ) -> None:
        await extractor._type_message_with_newlines("First\n\nSecond")

        # Empty middle segment is skipped, but the two newlines still emit
        # two Shift+Enter presses so the paragraph break is preserved.
        assert keyboard_mock.mock_calls == [
            call.type("First", delay=15),
            call.press("Shift+Enter"),
            call.press("Shift+Enter"),
            call.type("Second", delay=15),
        ]

    async def test_crlf_normalized_to_lf(
        self, extractor: LinkedInExtractor, keyboard_mock: MagicMock
    ) -> None:
        await extractor._type_message_with_newlines("a\r\nb")

        assert keyboard_mock.mock_calls == [
            call.type("a", delay=15),
            call.press("Shift+Enter"),
            call.type("b", delay=15),
        ]

    async def test_trailing_newline_emits_shift_enter_only(
        self, extractor: LinkedInExtractor, keyboard_mock: MagicMock
    ) -> None:
        await extractor._type_message_with_newlines("a\n")

        assert keyboard_mock.mock_calls == [
            call.type("a", delay=15),
            call.press("Shift+Enter"),
        ]
