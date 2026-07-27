from typing import Any, Callable, Coroutine, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp import FastMCP
from fastmcp.tools import FunctionTool

from linkedin_mcp_server.callbacks import MCPContextProgressCallback
from linkedin_mcp_server.scraping.extractor import (
    ExtractedSection,
    _RATE_LIMITED_MSG,
    _normalize_feed_post,
)
from linkedin_mcp_server.scraping.link_metadata import FeedPost


async def get_tool_fn(
    mcp: FastMCP, name: str
) -> Callable[..., Coroutine[Any, Any, dict[str, Any]]]:
    """Extract tool function from FastMCP by name using public API."""
    tool = await mcp.get_tool(name)
    if tool is None:
        raise ValueError(f"Tool '{name}' not found")
    return cast(FunctionTool, tool).fn


def _make_mock_extractor(scrape_result: dict) -> MagicMock:
    """Create a mock LinkedInExtractor that returns the given result."""
    mock = MagicMock()
    mock.scrape_person = AsyncMock(return_value=scrape_result)
    mock.connect_with_person = AsyncMock(return_value=scrape_result)
    mock.scrape_company = AsyncMock(return_value=scrape_result)
    mock.scrape_job = AsyncMock(return_value=scrape_result)
    mock.search_jobs = AsyncMock(return_value=scrape_result)
    mock.get_saved_jobs = AsyncMock(return_value=scrape_result)
    mock.search_people = AsyncMock(return_value=scrape_result)
    mock.get_sidebar_profiles = AsyncMock(return_value=scrape_result)
    mock.get_inbox = AsyncMock(return_value=scrape_result)
    mock.get_conversation = AsyncMock(return_value=scrape_result)
    mock.search_conversations = AsyncMock(return_value=scrape_result)
    mock.send_message = AsyncMock(return_value=scrape_result)
    mock.get_pending_invitations = AsyncMock(return_value=scrape_result)
    mock.act_on_invitation = AsyncMock(return_value=scrape_result)
    mock.get_my_profile = AsyncMock(return_value=scrape_result)
    mock.search_companies = AsyncMock(return_value=scrape_result)
    mock.search_posts = AsyncMock(return_value=scrape_result)
    mock.get_company_employees = AsyncMock(return_value=scrape_result)
    mock.extract_page = AsyncMock(
        return_value=ExtractedSection(text="some text", references=[])
    )
    mock.extract_feed = AsyncMock(return_value=ExtractedSection(text="", references=[]))
    return mock


class TestPersonTool:
    async def test_get_person_profile_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/in/test-user/",
            "sections": {"main_profile": "John Doe\nSoftware Engineer"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_person_profile")
        result = await tool_fn("test-user", mock_context, extractor=mock_extractor)
        assert result["url"] == "https://www.linkedin.com/in/test-user/"
        assert "main_profile" in result["sections"]
        assert "pages_visited" not in result
        assert "sections_requested" not in result

    async def test_get_person_profile_with_sections(self, mock_context):
        """Verify sections parameter is passed through."""
        expected = {
            "url": "https://www.linkedin.com/in/test-user/",
            "sections": {
                "main_profile": "John Doe",
                "experience": "Work history",
                "contact_info": "Email: test@test.com",
            },
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_person_profile")
        result = await tool_fn(
            "test-user",
            mock_context,
            sections="experience,contact_info",
            extractor=mock_extractor,
        )
        assert "main_profile" in result["sections"]
        assert "experience" in result["sections"]
        assert "contact_info" in result["sections"]
        # Verify scrape_person was called exactly once with a set[str]
        mock_extractor.scrape_person.assert_awaited_once()
        call_args = mock_extractor.scrape_person.call_args
        assert isinstance(call_args[0][1], set)
        assert "experience" in call_args[0][1]
        assert "contact_info" in call_args[0][1]

    async def test_get_person_profile_passes_callbacks(self, mock_context):
        """Verify tool wires MCPContextProgressCallback to the extractor."""
        expected = {
            "url": "https://www.linkedin.com/in/test-user/",
            "sections": {"main_profile": "John Doe"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_person_profile")
        await tool_fn("test-user", mock_context, extractor=mock_extractor)

        call_kwargs = mock_extractor.scrape_person.call_args.kwargs
        assert "callbacks" in call_kwargs
        assert isinstance(call_kwargs["callbacks"], MCPContextProgressCallback)

    async def test_get_person_profile_passes_max_scrolls(self, mock_context):
        """Verify max_scrolls parameter is forwarded to scrape_person."""
        expected = {
            "url": "https://www.linkedin.com/in/test-user/",
            "sections": {"main_profile": "John Doe"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_person_profile")
        await tool_fn(
            "test-user",
            mock_context,
            max_scrolls=15,
            extractor=mock_extractor,
        )

        call_kwargs = mock_extractor.scrape_person.call_args.kwargs
        assert call_kwargs["max_scrolls"] == 15

    async def test_get_person_profile_rejects_invalid_max_scrolls(self, mock_context):
        """Verify max_scrolls=0 is rejected by Field(ge=1) validation."""
        # FastMCP wraps the pydantic error raised by Field() constraints in
        # its own ValidationError, which does not subclass pydantic's.
        from fastmcp.exceptions import ValidationError

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        with pytest.raises(ValidationError, match="max_scrolls"):
            await mcp.call_tool(
                "get_person_profile",
                {"linkedin_username": "test-user", "max_scrolls": 0},
            )

    async def test_get_person_profile_unknown_section(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/in/test-user/",
            "sections": {"main_profile": "John Doe"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_person_profile")
        result = await tool_fn(
            "test-user",
            mock_context,
            sections="bogus_section",
            extractor=mock_extractor,
        )
        assert result["unknown_sections"] == ["bogus_section"]

    async def test_get_person_profile_error(self, mock_context):
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.exceptions import SessionExpiredError

        mock_extractor = MagicMock()
        mock_extractor.scrape_person = AsyncMock(side_effect=SessionExpiredError())

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_person_profile")
        with pytest.raises(ToolError, match="Session expired"):
            await tool_fn("test-user", mock_context, extractor=mock_extractor)

    async def test_get_person_profile_auth_error(self, monkeypatch):
        """Auth failures in the DI layer trigger auto-relogin and report the login browser."""
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.core.exceptions import AuthenticationError
        from linkedin_mcp_server.exceptions import AuthenticationStartedError

        mock_browser = MagicMock()
        mock_browser.page = MagicMock()
        monkeypatch.setattr(
            "linkedin_mcp_server.dependencies.ensure_tool_ready_or_raise",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            "linkedin_mcp_server.dependencies.get_or_create_browser",
            AsyncMock(return_value=mock_browser),
        )
        monkeypatch.setattr(
            "linkedin_mcp_server.dependencies.ensure_authenticated",
            AsyncMock(side_effect=AuthenticationError("Session expired or invalid.")),
        )
        monkeypatch.setattr(
            "linkedin_mcp_server.dependencies.get_runtime_policy",
            lambda: "managed",
        )
        monkeypatch.setattr(
            "linkedin_mcp_server.dependencies.close_browser",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            "linkedin_mcp_server.dependencies.invalidate_auth_and_trigger_relogin",
            AsyncMock(
                side_effect=AuthenticationStartedError(
                    "Session expired. A login browser window has been opened."
                )
            ),
        )

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        with pytest.raises(ToolError, match="Session expired"):
            await mcp.call_tool("get_person_profile", {"linkedin_username": "test"})

    async def test_search_people(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/search/results/people/?keywords=AI+engineer&location=New+York",
            "sections": {"search_results": "Jane Doe\nAI Engineer at Acme\nNew York"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "search_people")
        result = await tool_fn(
            "AI engineer", mock_context, location="New York", extractor=mock_extractor
        )
        assert "search_results" in result["sections"]
        assert "pages_visited" not in result
        mock_extractor.search_people.assert_awaited_once_with(
            "AI engineer",
            "New York",
            network=None,
            current_company=None,
        )

    async def test_search_people_with_network_and_company_filters(self, mock_context):
        expected = {
            "url": (
                "https://www.linkedin.com/search/results/people/"
                "?keywords=engineer&network=%5B%22F%22%5D"
                "&currentCompany=%5B%221115%22%5D"
            ),
            "sections": {
                "search_results": "Jennifer Bonuso\nPresident Americas at SAP"
            },
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "search_people")
        result = await tool_fn(
            "engineer",
            mock_context,
            network=["F"],
            current_company="1115",
            extractor=mock_extractor,
        )
        assert "search_results" in result["sections"]
        mock_extractor.search_people.assert_awaited_once_with(
            "engineer",
            None,
            network=["F"],
            current_company="1115",
        )

    async def test_search_people_validation_error_surfaced_as_tool_error(
        self, mock_context
    ):
        """A FilterValidationError raised by the extractor should surface to
        the MCP client as a ToolError carrying the same message, rather than
        being collapsed to the generic "Error calling tool" mask."""
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.scraping.extractor import FilterValidationError
        from linkedin_mcp_server.tools.person import register_person_tools

        mock_extractor = MagicMock()
        mock_extractor.search_people = AsyncMock(
            side_effect=FilterValidationError("must be a numeric URN")
        )

        mcp = FastMCP("test")
        register_person_tools(mcp)
        tool_fn = await get_tool_fn(mcp, "search_people")

        with pytest.raises(ToolError, match="must be a numeric URN"):
            await tool_fn(
                "engineer",
                mock_context,
                current_company="SAP",
                extractor=mock_extractor,
            )

    async def test_connect_with_person(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/in/test-user/",
            "status": "connected",
            "message": "Connection request sent.",
            "note_sent": True,
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "connect_with_person")
        result = await tool_fn(
            "test-user",
            mock_context,
            note="Let us connect.",
            extractor=mock_extractor,
        )

        assert result["status"] == "connected"
        assert result["note_sent"] is True
        mock_extractor.connect_with_person.assert_awaited_once_with(
            "test-user",
            note="Let us connect.",
        )

    async def test_connect_with_person_no_note(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/in/test-user/",
            "status": "connected",
            "message": "Connection request sent.",
            "note_sent": False,
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "connect_with_person")
        result = await tool_fn(
            "test-user",
            mock_context,
            extractor=mock_extractor,
        )

        assert result["status"] == "connected"
        mock_extractor.connect_with_person.assert_awaited_once_with(
            "test-user",
            note=None,
        )

    async def test_connect_with_person_custom_note_limit_reached(self, mock_context):
        """The custom_note_limit_reached status returns LinkedIn's message."""
        expected = {
            "url": "https://www.linkedin.com/in/test-user/",
            "status": "custom_note_limit_reached",
            "message": "Wysyłaj nieograniczoną liczbę spersonalizowanych zaproszeń dzięki Premium",
            "note_sent": False,
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "connect_with_person")
        result = await tool_fn(
            "test-user",
            mock_context,
            note="Hello!",
            extractor=mock_extractor,
        )

        assert result["status"] == "custom_note_limit_reached"
        assert (
            result["message"]
            == "Wysyłaj nieograniczoną liczbę spersonalizowanych zaproszeń dzięki Premium"
        )
        assert result["note_sent"] is False
        mock_extractor.connect_with_person.assert_awaited_once_with(
            "test-user",
            note="Hello!",
        )

    async def test_connect_with_person_auth_error(self, monkeypatch):
        """Auth failures in the DI layer trigger auto-relogin and report the login browser."""
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.core.exceptions import AuthenticationError
        from linkedin_mcp_server.exceptions import AuthenticationStartedError

        mock_browser = MagicMock()
        mock_browser.page = MagicMock()
        monkeypatch.setattr(
            "linkedin_mcp_server.dependencies.ensure_tool_ready_or_raise",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            "linkedin_mcp_server.dependencies.get_or_create_browser",
            AsyncMock(return_value=mock_browser),
        )
        monkeypatch.setattr(
            "linkedin_mcp_server.dependencies.ensure_authenticated",
            AsyncMock(side_effect=AuthenticationError("Session expired or invalid.")),
        )
        monkeypatch.setattr(
            "linkedin_mcp_server.dependencies.get_runtime_policy",
            lambda: "managed",
        )
        monkeypatch.setattr(
            "linkedin_mcp_server.dependencies.close_browser",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            "linkedin_mcp_server.dependencies.invalidate_auth_and_trigger_relogin",
            AsyncMock(
                side_effect=AuthenticationStartedError(
                    "Session expired. A login browser window has been opened."
                )
            ),
        )

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        with pytest.raises(ToolError, match="Session expired"):
            await mcp.call_tool(
                "connect_with_person",
                {"linkedin_username": "test"},
            )


class TestCompanyTools:
    async def test_get_company_profile(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/company/testcorp/",
            "sections": {"about": "TestCorp\nWe build things"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.company import register_company_tools

        mcp = FastMCP("test")
        register_company_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_company_profile")
        result = await tool_fn("testcorp", mock_context, extractor=mock_extractor)
        assert "about" in result["sections"]
        assert "pages_visited" not in result

    async def test_get_company_profile_passes_callbacks(self, mock_context):
        """Verify tool wires MCPContextProgressCallback to the extractor."""
        expected = {
            "url": "https://www.linkedin.com/company/testcorp/",
            "sections": {"about": "TestCorp\nWe build things"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.company import register_company_tools

        mcp = FastMCP("test")
        register_company_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_company_profile")
        await tool_fn("testcorp", mock_context, extractor=mock_extractor)

        call_kwargs = mock_extractor.scrape_company.call_args.kwargs
        assert "callbacks" in call_kwargs
        assert isinstance(call_kwargs["callbacks"], MCPContextProgressCallback)

    async def test_get_company_profile_unknown_section(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/company/testcorp/",
            "sections": {"about": "TestCorp\nWe build things"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.company import register_company_tools

        mcp = FastMCP("test")
        register_company_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_company_profile")
        result = await tool_fn(
            "testcorp", mock_context, sections="bogus", extractor=mock_extractor
        )
        assert result["unknown_sections"] == ["bogus"]

    async def test_get_company_posts(self, mock_context):
        mock_extractor = MagicMock()
        mock_extractor.extract_page = AsyncMock(
            return_value=ExtractedSection(text="Post 1\nPost 2", references=[])
        )

        from linkedin_mcp_server.tools.company import register_company_tools

        mcp = FastMCP("test")
        register_company_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_company_posts")
        result = await tool_fn("testcorp", mock_context, extractor=mock_extractor)
        assert "posts" in result["sections"]
        assert result["sections"]["posts"] == "Post 1\nPost 2"
        assert "pages_visited" not in result
        assert "sections_requested" not in result

    async def test_get_company_posts_omits_rate_limited_sentinel(self, mock_context):
        mock_extractor = MagicMock()
        mock_extractor.extract_page = AsyncMock(
            return_value=ExtractedSection(text=_RATE_LIMITED_MSG, references=[])
        )

        from linkedin_mcp_server.tools.company import register_company_tools

        mcp = FastMCP("test")
        register_company_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_company_posts")
        result = await tool_fn("testcorp", mock_context, extractor=mock_extractor)
        assert result["sections"] == {}

    async def test_get_company_posts_returns_section_errors(self, mock_context):
        mock_extractor = MagicMock()
        mock_extractor.extract_page = AsyncMock(
            return_value=ExtractedSection(
                text="",
                references=[],
                error={"issue_template_path": "/tmp/company-posts-issue.md"},
            )
        )

        from linkedin_mcp_server.tools.company import register_company_tools

        mcp = FastMCP("test")
        register_company_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_company_posts")
        result = await tool_fn("testcorp", mock_context, extractor=mock_extractor)
        assert result["sections"] == {}
        assert result["section_errors"]["posts"]["issue_template_path"] == (
            "/tmp/company-posts-issue.md"
        )

    async def test_get_company_posts_omits_orphaned_references(self, mock_context):
        mock_extractor = MagicMock()
        mock_extractor.extract_page = AsyncMock(
            return_value=ExtractedSection(
                text="",
                references=[
                    {
                        "kind": "company",
                        "url": "/company/testcorp/",
                        "text": "TestCorp",
                    }
                ],
            )
        )

        from linkedin_mcp_server.tools.company import register_company_tools

        mcp = FastMCP("test")
        register_company_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_company_posts")
        result = await tool_fn("testcorp", mock_context, extractor=mock_extractor)
        assert result["sections"] == {}
        assert "references" not in result


class TestJobTools:
    async def test_get_job_details(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/jobs/view/12345/",
            "sections": {"job_posting": "Software Engineer\nGreat opportunity"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.job import register_job_tools

        mcp = FastMCP("test")
        register_job_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_job_details")
        result = await tool_fn("12345", mock_context, extractor=mock_extractor)
        assert "job_posting" in result["sections"]
        assert "pages_visited" not in result

    async def test_search_jobs(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/jobs/search/?keywords=python",
            "sections": {"search_results": "Job 1\nJob 2"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.job import register_job_tools

        mcp = FastMCP("test")
        register_job_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "search_jobs")
        result = await tool_fn(
            "python", mock_context, location="Remote", extractor=mock_extractor
        )
        assert "search_results" in result["sections"]
        assert "pages_visited" not in result

    async def test_get_saved_jobs(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/my-items/saved-jobs/",
            "sections": {"saved_jobs": "Saved Job 1\nSaved Job 2"},
            "job_ids": ["111", "222"],
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.job import register_job_tools

        mcp = FastMCP("test")
        register_job_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_saved_jobs")
        result = await tool_fn(mock_context, max_pages=2, extractor=mock_extractor)
        assert "saved_jobs" in result["sections"]
        assert result["job_ids"] == ["111", "222"]
        mock_extractor.get_saved_jobs.assert_awaited_once_with(max_pages=2)


class TestGetSidebarProfilesTool:
    async def test_get_sidebar_profiles_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/in/test-user/",
            "sidebar_profiles": {
                "more_profiles_for_you": ["/in/alice/", "/in/bob/"],
            },
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_sidebar_profiles")
        result = await tool_fn("test-user", mock_context, extractor=mock_extractor)

        assert result["url"] == "https://www.linkedin.com/in/test-user/"
        assert "more_profiles_for_you" in result["sidebar_profiles"]
        mock_extractor.get_sidebar_profiles.assert_awaited_once_with("test-user")

    async def test_get_sidebar_profiles_empty_result(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/in/test-user/",
            "sidebar_profiles": {},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_sidebar_profiles")
        result = await tool_fn("test-user", mock_context, extractor=mock_extractor)

        assert result["sidebar_profiles"] == {}

    async def test_get_sidebar_profiles_error(self, mock_context):
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.exceptions import SessionExpiredError

        mock_extractor = MagicMock()
        mock_extractor.get_sidebar_profiles = AsyncMock(
            side_effect=SessionExpiredError()
        )

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_sidebar_profiles")
        with pytest.raises(ToolError, match="Session expired"):
            await tool_fn("test-user", mock_context, extractor=mock_extractor)


class TestMessagingTools:
    async def test_get_inbox_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/messaging/",
            "sections": {"inbox": "Conversation 1\nConversation 2"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.messaging import register_messaging_tools

        mcp = FastMCP("test")
        register_messaging_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_inbox")
        result = await tool_fn(mock_context, extractor=mock_extractor)

        assert result["sections"]["inbox"] == "Conversation 1\nConversation 2"
        mock_extractor.get_inbox.assert_awaited_once_with(limit=20)

    async def test_get_conversation_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/messaging/thread/abc123/",
            "sections": {
                "messages": [
                    {
                        "timestamp": "2026-02-10T15:17:00",
                        "status": "sent",
                        "sender": 0,
                        "content": "Hello!",
                    }
                ],
                "members": [
                    {
                        "kind": "person",
                        "url": "/in/alice/",
                        "name": "Alice",
                        "is_self": True,
                    },
                ],
            },
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.messaging import register_messaging_tools

        mcp = FastMCP("test")
        register_messaging_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_conversation")
        result = await tool_fn(
            mock_context, linkedin_username="testuser", extractor=mock_extractor
        )

        assert result["sections"]["messages"][0]["content"] == "Hello!"
        assert result["sections"]["members"][0]["url"] == "/in/alice/"
        mock_extractor.get_conversation.assert_awaited_once_with(
            linkedin_username="testuser",
            thread_id=None,
            index=0,
            max_scrolls=3,
        )

    async def test_search_conversations_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/messaging/",
            "sections": {"search_results": "Result 1\nResult 2"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.messaging import register_messaging_tools

        mcp = FastMCP("test")
        register_messaging_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "search_conversations")
        result = await tool_fn("hello", mock_context, extractor=mock_extractor)

        assert result["sections"]["search_results"] == "Result 1\nResult 2"
        mock_extractor.search_conversations.assert_awaited_once_with("hello", limit=20)

    async def test_send_message_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/messaging/thread/abc123/",
            "status": "sent",
            "message": "Message sent.",
            "recipient_selected": True,
            "sent": True,
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.messaging import register_messaging_tools

        mcp = FastMCP("test")
        register_messaging_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "send_message")
        result = await tool_fn(
            "testuser",
            "Hello!",
            True,
            mock_context,
            extractor=mock_extractor,
        )

        assert result["status"] == "sent"
        assert result["sent"] is True
        mock_extractor.send_message.assert_awaited_once_with(
            "testuser", "Hello!", confirm_send=True, profile_urn=None
        )

    async def test_send_message_with_profile_urn(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/messaging/thread/abc123/",
            "status": "sent",
            "message": "Message sent.",
            "recipient_selected": True,
            "sent": True,
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.messaging import register_messaging_tools

        mcp = FastMCP("test")
        register_messaging_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "send_message")
        result = await tool_fn(
            "testuser",
            "Hello!",
            True,
            mock_context,
            profile_urn="ACoAAB1IelEB",
            extractor=mock_extractor,
        )

        assert result["status"] == "sent"
        mock_extractor.send_message.assert_awaited_once_with(
            "testuser", "Hello!", confirm_send=True, profile_urn="ACoAAB1IelEB"
        )

    async def test_message_invitation_sender_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/messaging/thread/abc123/",
            "status": "confirmation_required",
            "message": "Set confirm_send=true to send the message.",
            "recipient_selected": True,
            "sent": False,
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.messaging import register_messaging_tools

        mcp = FastMCP("test")
        register_messaging_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "message_invitation_sender")
        result = await tool_fn(
            "testuser",
            "/messaging/compose/?recipient=ACoAAB&invitation=urn",
            "Hello!",
            False,
            mock_context,
            extractor=mock_extractor,
        )

        assert result["status"] == "confirmation_required"
        assert result["sent"] is False
        mock_extractor.send_message.assert_awaited_once_with(
            "testuser",
            "Hello!",
            confirm_send=False,
            compose_url="/messaging/compose/?recipient=ACoAAB&invitation=urn",
        )

    async def test_send_message_error(self, mock_context):
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.exceptions import SessionExpiredError

        mock_extractor = MagicMock()
        mock_extractor.send_message = AsyncMock(side_effect=SessionExpiredError())

        from linkedin_mcp_server.tools.messaging import register_messaging_tools

        mcp = FastMCP("test")
        register_messaging_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "send_message")
        with pytest.raises(ToolError, match="Session expired"):
            await tool_fn(
                "testuser",
                "Hello!",
                True,
                mock_context,
                extractor=mock_extractor,
            )


class TestGetMyProfileTool:
    async def test_get_my_profile_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/in/johndoe/",
            "sections": {"main_profile": "John Doe\nSoftware Engineer"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_my_profile")
        result = await tool_fn(mock_context, extractor=mock_extractor)
        assert result["url"] == "https://www.linkedin.com/in/johndoe/"
        assert "main_profile" in result["sections"]
        mock_extractor.get_my_profile.assert_awaited_once()

    async def test_get_my_profile_with_sections(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/in/johndoe/",
            "sections": {"main_profile": "John Doe", "experience": "Work history"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_my_profile")
        result = await tool_fn(
            mock_context, sections="experience", extractor=mock_extractor
        )
        assert "main_profile" in result["sections"]
        assert "experience" in result["sections"]
        call_kwargs = mock_extractor.get_my_profile.call_args.kwargs
        assert "experience" in call_kwargs["sections"]

    async def test_get_my_profile_passes_callbacks(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/in/johndoe/",
            "sections": {"main_profile": "John Doe"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_my_profile")
        await tool_fn(mock_context, extractor=mock_extractor)

        call_kwargs = mock_extractor.get_my_profile.call_args.kwargs
        assert "callbacks" in call_kwargs
        assert isinstance(call_kwargs["callbacks"], MCPContextProgressCallback)

    async def test_get_my_profile_unknown_section(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/in/johndoe/",
            "sections": {"main_profile": "John Doe"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_my_profile")
        result = await tool_fn(
            mock_context, sections="bogus_section", extractor=mock_extractor
        )
        assert result["unknown_sections"] == ["bogus_section"]

    async def test_get_my_profile_error(self, mock_context):
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.exceptions import SessionExpiredError

        mock_extractor = MagicMock()
        mock_extractor.get_my_profile = AsyncMock(side_effect=SessionExpiredError())

        from linkedin_mcp_server.tools.person import register_person_tools

        mcp = FastMCP("test")
        register_person_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_my_profile")
        with pytest.raises(ToolError, match="Session expired"):
            await tool_fn(mock_context, extractor=mock_extractor)


class TestSearchCompaniesTool:
    async def test_search_companies_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/search/results/companies/?keywords=fintech",
            "sections": {"search_results": "Stripe\nFintech company\nSan Francisco"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.company import register_company_tools

        mcp = FastMCP("test")
        register_company_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "search_companies")
        result = await tool_fn("fintech", mock_context, extractor=mock_extractor)
        assert "search_results" in result["sections"]
        mock_extractor.search_companies.assert_awaited_once_with("fintech")

    async def test_search_companies_error(self, mock_context):
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.exceptions import SessionExpiredError

        mock_extractor = MagicMock()
        mock_extractor.search_companies = AsyncMock(side_effect=SessionExpiredError())

        from linkedin_mcp_server.tools.company import register_company_tools

        mcp = FastMCP("test")
        register_company_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "search_companies")
        with pytest.raises(ToolError, match="Session expired"):
            await tool_fn("fintech", mock_context, extractor=mock_extractor)


class TestGetCompanyEmployeesTool:
    async def test_get_company_employees_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/company/anthropic/people/",
            "sections": {"employees": "Jane Doe\nResearch Engineer\nSan Francisco"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.company import register_company_tools

        mcp = FastMCP("test")
        register_company_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_company_employees")
        result = await tool_fn("anthropic", mock_context, extractor=mock_extractor)
        assert "employees" in result["sections"]
        mock_extractor.get_company_employees.assert_awaited_once_with(
            "anthropic", keywords=None
        )

    async def test_get_company_employees_with_keywords(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/company/anthropic/people/?keywords=engineer",
            "sections": {"employees": "Jane Doe\nResearch Engineer"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.company import register_company_tools

        mcp = FastMCP("test")
        register_company_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_company_employees")
        result = await tool_fn(
            "anthropic", mock_context, keywords="engineer", extractor=mock_extractor
        )
        assert "employees" in result["sections"]
        mock_extractor.get_company_employees.assert_awaited_once_with(
            "anthropic", keywords="engineer"
        )

    async def test_get_company_employees_error(self, mock_context):
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.exceptions import SessionExpiredError

        mock_extractor = MagicMock()
        mock_extractor.get_company_employees = AsyncMock(
            side_effect=SessionExpiredError()
        )

        from linkedin_mcp_server.tools.company import register_company_tools

        mcp = FastMCP("test")
        register_company_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_company_employees")
        with pytest.raises(ToolError, match="Session expired"):
            await tool_fn("anthropic", mock_context, extractor=mock_extractor)


class TestFeedTools:
    async def test_get_feed_success(self, mock_context):
        """sections["feed"] is the structured FeedPost list, not raw text."""
        posts: list[FeedPost] = [
            {
                "url": "/feed/update/urn:li:activity:1/",
                "post_age": "21min",
                "author": {
                    "name": "Sami B",
                    "profile_url": "/in/sami/",
                    "headline": "Data scientist",
                    "degree": "1st",
                },
                "content": "Hello world",
                "is_promoted": False,
                "media": None,
                "reactions_count": 1,
                "comment_count": None,
                "repost_count": None,
            }
        ]
        mock_extractor = MagicMock()
        mock_extractor.extract_feed = AsyncMock(
            return_value=ExtractedSection(text="", references=[], posts=posts)
        )

        from linkedin_mcp_server.tools.feed import register_feed_tools

        mcp = FastMCP("test")
        register_feed_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_feed")
        result = await tool_fn(mock_context, extractor=mock_extractor)
        assert result["url"] == "https://www.linkedin.com/feed/"
        # the tool must populate the section key (guards against the else
        # branch being dropped) ...
        assert "feed" in result["sections"]
        # ... and surface the structured list verbatim.
        assert result["sections"]["feed"] == posts
        assert result["sections"]["feed"][0]["author"]["degree"] == "1st"
        assert "posts" not in result

    async def test_get_feed_surfaces_references(self, mock_context):
        """References from the extractor flow through to the tool result."""
        mock_extractor = MagicMock()
        mock_extractor.extract_feed = AsyncMock(
            return_value=ExtractedSection(
                text="",
                references=[
                    {
                        "kind": "feed_post",
                        "url": "/posts/alice_hello-ugcPost-1-xx",
                        "context": "feed",
                    },
                    {
                        "kind": "feed_post",
                        "url": "/feed/update/urn:li:activity:1234567890/",
                    },
                ],
            )
        )

        from linkedin_mcp_server.tools.feed import register_feed_tools

        mcp = FastMCP("test")
        register_feed_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_feed")
        result = await tool_fn(mock_context, extractor=mock_extractor)
        assert "posts" not in result
        assert "feed" in result["references"]
        urls = [r["url"] for r in result["references"]["feed"]]
        assert "/posts/alice_hello-ugcPost-1-xx" in urls
        assert "/feed/update/urn:li:activity:1234567890/" in urls

    async def test_get_feed_rate_limited_surfaces_section_error(self, mock_context):
        """Rate-limit sentinel becomes a typed section_errors entry."""
        mock_extractor = MagicMock()
        mock_extractor.extract_feed = AsyncMock(
            return_value=ExtractedSection(text=_RATE_LIMITED_MSG, references=[])
        )

        from linkedin_mcp_server.tools.feed import register_feed_tools

        mcp = FastMCP("test")
        register_feed_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_feed")
        result = await tool_fn(mock_context, extractor=mock_extractor)
        assert "feed" not in result["sections"]
        assert result["section_errors"]["feed"]["error_type"] == "rate_limit"
        assert result["section_errors"]["feed"]["error_message"] == _RATE_LIMITED_MSG

    async def test_get_feed_returns_section_errors(self, mock_context):
        mock_extractor = MagicMock()
        mock_extractor.extract_feed = AsyncMock(
            return_value=ExtractedSection(
                text="",
                references=[],
                error={"issue_template_path": "/tmp/feed-issue.md"},
            )
        )

        from linkedin_mcp_server.tools.feed import register_feed_tools

        mcp = FastMCP("test")
        register_feed_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_feed")
        result = await tool_fn(mock_context, extractor=mock_extractor)
        assert result["sections"] == {}
        assert "feed" in result["section_errors"]

    async def test_get_feed_rejects_zero_num_posts(self, mock_context):
        """Verify num_posts=0 is rejected by Field(ge=1) validation."""
        # FastMCP wraps the pydantic error raised by Field() constraints in
        # its own ValidationError, which does not subclass pydantic's.
        from fastmcp.exceptions import ValidationError

        from linkedin_mcp_server.tools.feed import register_feed_tools

        mcp = FastMCP("test")
        register_feed_tools(mcp)

        with pytest.raises(ValidationError, match="num_posts"):
            await mcp.call_tool("get_feed", {"num_posts": 0})

    async def test_get_feed_rejects_excessive_num_posts(self, mock_context):
        """Verify num_posts=51 is rejected by Field(le=50) validation."""
        # FastMCP wraps the pydantic error raised by Field() constraints in
        # its own ValidationError, which does not subclass pydantic's.
        from fastmcp.exceptions import ValidationError

        from linkedin_mcp_server.tools.feed import register_feed_tools

        mcp = FastMCP("test")
        register_feed_tools(mcp)

        with pytest.raises(ValidationError, match="num_posts"):
            await mcp.call_tool("get_feed", {"num_posts": 51})


class TestNormalizeFeedPost:
    """Python-side normalization of raw cards from _FEED_POSTS_JS."""

    def _raw(self, **overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "url": "/feed/update/urn:li:activity:7/",
            "post_age": "15h",
            "author": {
                "name": "Corentin Hugot",
                "profile_url": "/in/corentin/",
                "headline": "Co-founder",
                "degree": "2nd",
            },
            "content": "I applied to YC 5 times",
            "is_promoted": False,
            "media": None,
            "reactions_count": 141,
            "comment_count": 34,
            "repost_count": 1,
        }
        base.update(overrides)
        return base

    def test_native_post_round_trips(self):
        assert _normalize_feed_post(self._raw()) == self._raw()

    def test_sponsored_post_author_shape(self):
        raw = self._raw(
            url=None,
            post_age=None,
            is_promoted=True,
            author={
                "name": "Vanta",
                "profile_url": "/company/vanta/",
                "headline": "137,470 followers",
                "degree": None,
            },
            content="Achieve SOC 2 compliance",
            reactions_count=3,
            comment_count=None,
            repost_count=None,
        )
        post = _normalize_feed_post(raw)
        assert post is not None
        assert post["is_promoted"] is True
        assert post["author"]["profile_url"] == "/company/vanta/"
        assert post["author"]["headline"] == "137,470 followers"
        assert post["author"]["degree"] is None
        assert post["url"] is None
        assert post["post_age"] is None

    def test_silent_post_counts_are_none(self):
        post = _normalize_feed_post(
            self._raw(reactions_count=None, comment_count=None, repost_count=None)
        )
        assert post is not None
        assert post["reactions_count"] is None
        assert post["comment_count"] is None
        assert post["repost_count"] is None

    @pytest.mark.parametrize(
        "media,expected",
        [
            (
                {"type": "image", "url": "https://media.licdn.com/x.jpg"},
                {"type": "image", "url": "https://media.licdn.com/x.jpg"},
            ),
            (
                {"type": "video", "url": "https://media.licdn.com/v.mp4"},
                {"type": "video", "url": "https://media.licdn.com/v.mp4"},
            ),
            (
                {"type": "link", "url": "https://lnkd.in/abc"},
                {"type": "link", "url": "https://lnkd.in/abc"},
            ),
            ({"type": "video", "url": None}, None),  # no src -> dropped
            ({"type": "bogus", "url": "https://x"}, None),  # bad type
            (None, None),
            ("notadict", None),
        ],
    )
    def test_media_normalization(self, media, expected):
        post = _normalize_feed_post(self._raw(media=media))
        assert post is not None
        assert post["media"] == expected

    @pytest.mark.parametrize(
        "raw_degree,expected",
        [
            ("1st", "1st"),
            ("2nd", "2nd"),
            ("3rd+", "3rd+"),
            ("• 1st", "1st"),
            ("1ST", "1st"),
            ("", None),
            ("first", None),
            (None, None),
        ],
    )
    def test_degree_normalization(self, raw_degree, expected):
        author = dict(self._raw()["author"], degree=raw_degree)
        post = _normalize_feed_post(self._raw(author=author))
        assert post is not None
        assert post["author"]["degree"] == expected

    @pytest.mark.parametrize(
        "raw_age,expected",
        [
            ("21min", "21min"),
            ("15h", "15h"),
            ("1d", "1d"),
            ("2mo", "2mo"),
            ("3w", "3w"),
            ("1y", "1y"),
            ("21m", None),  # ambiguous bare form; JS emits the long token
            ("yesterday", None),
            ("", None),
            (None, None),
        ],
    )
    def test_age_validation(self, raw_age, expected):
        post = _normalize_feed_post(self._raw(post_age=raw_age))
        assert post is not None
        assert post["post_age"] == expected

    def test_count_coercion(self):
        post = _normalize_feed_post(
            self._raw(reactions_count="141", comment_count=-3, repost_count=True)
        )
        assert post is not None
        assert post["reactions_count"] == 141  # numeric string coerced
        assert post["comment_count"] is None  # negative -> None
        assert post["repost_count"] is None  # bool rejected

    def test_truncated_content_preserved(self):
        post = _normalize_feed_post(self._raw(content="line one\nline two"))
        assert post is not None
        assert post["content"] == "line one\nline two"

    def test_chrome_with_no_signal_dropped(self):
        raw = self._raw(
            url=None,
            content=None,
            author={
                "name": None,
                "profile_url": None,
                "headline": None,
                "degree": None,
            },
        )
        assert _normalize_feed_post(raw) is None

    def test_non_dict_dropped(self):
        assert _normalize_feed_post("nope") is None
        assert _normalize_feed_post(None) is None


class TestNetworkTools:
    async def test_get_pending_invitations_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/mynetwork/invitation-manager/received/",
            "invitations": [
                {
                    "type": "page_follow",
                    "invitation_age": "1h",
                    "sender": {
                        "name": "Juan Manuel M. Pérez",
                        "url": "/in/juanmanuelperez/",
                        "headline": None,
                        "mutual_connections": None,
                    },
                    "note": None,
                    "target": {
                        "page": {
                            "name": "Magical Potion Consulting",
                            "url": "/company/magical-potion-consulting/",
                        },
                        "newsletter": None,
                    },
                    "message_url": None,
                }
            ],
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.network import register_network_tools

        mcp = FastMCP("test")
        register_network_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_pending_invitations")
        result = await tool_fn(mock_context, extractor=mock_extractor)

        assert list(result) == ["url", "invitations"]
        assert result["invitations"][0]["type"] == "page_follow"
        assert result["invitations"][0]["sender"]["url"] == "/in/juanmanuelperez/"
        mock_extractor.get_pending_invitations.assert_awaited_once_with(
            limit=20,
            kind="received",
        )

    async def test_get_pending_invitations_sent_kind(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/mynetwork/invitation-manager/sent/",
            "invitations": [
                {
                    "type": "connection_request",
                    "invitation_age": "1w",
                    "recipient": {
                        "name": "Laurent SORBIER",
                        "url": "/in/laurent-sorbier/",
                        "headline": "chargé d’affaires chez belectric",
                    },
                }
            ],
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.network import register_network_tools

        mcp = FastMCP("test")
        register_network_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_pending_invitations")
        result = await tool_fn(
            mock_context,
            limit=5,
            kind="sent",
            extractor=mock_extractor,
        )

        assert result["url"].endswith("/sent/")
        invitation = result["invitations"][0]
        assert invitation["recipient"]["headline"] == "chargé d’affaires chez belectric"
        assert "sender" not in invitation
        mock_extractor.get_pending_invitations.assert_awaited_once_with(
            limit=5,
            kind="sent",
        )

    async def test_get_pending_invitations_structured_content_shape(self):
        expected = {
            "url": "https://www.linkedin.com/mynetwork/invitation-manager/received/",
            "invitations": [],
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.network import register_network_tools

        mcp = FastMCP("test")
        register_network_tools(mcp)

        with patch(
            "linkedin_mcp_server.tools.network.get_ready_extractor",
            new_callable=AsyncMock,
            return_value=mock_extractor,
        ):
            result = await mcp.call_tool(
                "get_pending_invitations",
                {"kind": "received", "limit": 2},
            )

        assert result.structured_content == expected
        assert list(result.structured_content or {}) == ["url", "invitations"]

    async def test_get_pending_invitations_rejects_invalid_kind(self):
        from fastmcp.exceptions import ValidationError

        from linkedin_mcp_server.tools.network import register_network_tools

        mcp = FastMCP("test")
        register_network_tools(mcp)

        with pytest.raises(ValidationError, match="kind"):
            await mcp.call_tool("get_pending_invitations", {"kind": "archived"})

    async def test_get_pending_invitations_rejects_excessive_limit(self):
        from fastmcp.exceptions import ValidationError

        from linkedin_mcp_server.tools.network import register_network_tools

        mcp = FastMCP("test")
        register_network_tools(mcp)

        with pytest.raises(ValidationError, match="limit"):
            await mcp.call_tool("get_pending_invitations", {"limit": 101})

    async def test_invitation_action_tools_are_registered(self):
        from linkedin_mcp_server.tools.network import register_network_tools

        mcp = FastMCP("test")
        register_network_tools(mcp)

        assert await mcp.get_tool("ignore_connection_request") is not None
        assert await mcp.get_tool("withdraw_invitation") is not None
        # The public surface intentionally does not expose accept — the
        # tool only ignores. (Accept remains available internally via
        # extractor.act_on_invitation for connect_with_person's
        # auto-accept path.)
        assert await mcp.get_tool("respond_to_invitation") is None

    async def test_ignore_connection_request_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/in/alice/",
            "status": "ignored",
            "message": "Invitation ignored.",
            "action": "ignore",
            "linkedin_username": "alice",
            "performed": True,
            "profile_url": "/in/alice/",
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.network import register_network_tools

        mcp = FastMCP("test")
        register_network_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "ignore_connection_request")
        result = await tool_fn(
            linkedin_username="alice",
            ctx=mock_context,
            extractor=mock_extractor,
        )

        assert result["action"] == "ignore"
        assert result["status"] == "ignored"
        mock_extractor.act_on_invitation.assert_awaited_once_with("alice", "ignore")

    async def test_withdraw_invitation_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/mynetwork/invitation-manager/sent/",
            "status": "withdrawn",
            "message": "Invitation withdrawn.",
            "action": "withdraw",
            "linkedin_username": "bob",
            "performed": True,
            "profile_url": "/in/bob/",
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.network import register_network_tools

        mcp = FastMCP("test")
        register_network_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "withdraw_invitation")
        result = await tool_fn(
            linkedin_username="bob",
            ctx=mock_context,
            extractor=mock_extractor,
        )

        assert result["status"] == "withdrawn"
        assert result["action"] == "withdraw"
        mock_extractor.act_on_invitation.assert_awaited_once_with("bob", "withdraw")

    async def test_get_connections_success(self, mock_context):
        expected = {
            "url": "https://www.linkedin.com/mynetwork/invite-connect/connections/",
            "connections": [
                {
                    "name": "Rob Choy",
                    "url": "/in/robchoy/",
                    "headline": "Founder & investor",
                    "connected_on": "2026-05-25",
                },
                {
                    "name": "Santiago Moreno",
                    "url": "/in/santiago-moreno-7098138b/",
                    "headline": "Regional Operations Manager - RWE Renewables France",
                    "connected_on": "2026-05-24",
                },
            ],
        }
        mock_extractor = _make_mock_extractor(expected)
        mock_extractor.get_connections = AsyncMock(return_value=expected)

        from linkedin_mcp_server.tools.network import register_network_tools

        mcp = FastMCP("test")
        register_network_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "get_connections")
        result = await tool_fn(mock_context, extractor=mock_extractor)

        assert list(result) == ["url", "connections"]
        assert result["connections"][0]["url"] == "/in/robchoy/"
        assert result["connections"][0]["connected_on"] == "2026-05-25"
        mock_extractor.get_connections.assert_awaited_once_with(limit=20)

    async def test_get_connections_structured_content_shape(self):
        expected = {
            "url": "https://www.linkedin.com/mynetwork/invite-connect/connections/",
            "connections": [],
        }
        mock_extractor = _make_mock_extractor(expected)
        mock_extractor.get_connections = AsyncMock(return_value=expected)

        from linkedin_mcp_server.tools.network import register_network_tools

        mcp = FastMCP("test")
        register_network_tools(mcp)

        with patch(
            "linkedin_mcp_server.tools.network.get_ready_extractor",
            new_callable=AsyncMock,
            return_value=mock_extractor,
        ):
            result = await mcp.call_tool("get_connections", {"limit": 5})

        assert result.structured_content == expected
        assert list(result.structured_content or {}) == ["url", "connections"]
        mock_extractor.get_connections.assert_awaited_once_with(limit=5)

    async def test_get_connections_rejects_excessive_limit(self):
        from fastmcp.exceptions import ValidationError

        from linkedin_mcp_server.tools.network import register_network_tools

        mcp = FastMCP("test")
        register_network_tools(mcp)

        with pytest.raises(ValidationError, match="limit"):
            await mcp.call_tool("get_connections", {"limit": 101})


class TestPostTools:
    async def test_search_posts_success(self, mock_context):
        expected = {
            "url": (
                "https://www.linkedin.com/search/results/content/"
                "?keywords=Buscamos+Unity&origin=FACETED_SEARCH"
            ),
            "sections": {"search_results": "Acme is hiring a Unity dev!"},
        }
        mock_extractor = _make_mock_extractor(expected)

        from linkedin_mcp_server.tools.post import register_post_tools

        mcp = FastMCP("test")
        register_post_tools(mcp)

        tool_fn = await get_tool_fn(mcp, "search_posts")
        result = await tool_fn(
            "Buscamos Unity",
            mock_context,
            date_posted="past-week",
            extractor=mock_extractor,
        )
        assert "search_results" in result["sections"]
        mock_extractor.search_posts.assert_awaited_once_with(
            "Buscamos Unity",
            date_posted="past-week",
            max_pages=3,
        )

    async def test_search_posts_validation_error_surfaced_as_tool_error(
        self, mock_context
    ):
        """A FilterValidationError from the extractor surfaces to the client as
        a ToolError carrying the same message, not the generic mask."""
        from fastmcp.exceptions import ToolError

        from linkedin_mcp_server.scraping.extractor import FilterValidationError
        from linkedin_mcp_server.tools.post import register_post_tools

        mock_extractor = MagicMock()
        mock_extractor.search_posts = AsyncMock(
            side_effect=FilterValidationError("Invalid date_posted 'last-year'")
        )

        mcp = FastMCP("test")
        register_post_tools(mcp)
        tool_fn = await get_tool_fn(mcp, "search_posts")

        with pytest.raises(ToolError, match="Invalid date_posted"):
            await tool_fn(
                "python",
                mock_context,
                date_posted="last-year",
                extractor=mock_extractor,
            )

    async def test_search_posts_rejects_zero_max_pages(self, mock_context):
        """Verify max_pages=0 is rejected by Field(ge=1) validation."""
        # FastMCP wraps the pydantic error raised by Field() constraints in
        # its own ValidationError, which does not subclass pydantic's.
        from fastmcp.exceptions import ValidationError

        from linkedin_mcp_server.tools.post import register_post_tools

        mcp = FastMCP("test")
        register_post_tools(mcp)

        with pytest.raises(ValidationError, match="max_pages"):
            await mcp.call_tool("search_posts", {"keywords": "python", "max_pages": 0})


class TestToolTimeouts:
    async def test_all_tools_have_global_timeout(self):
        from linkedin_mcp_server.server import create_mcp_server

        custom_timeout = 7.5
        mcp = create_mcp_server(tool_timeout=custom_timeout)

        tool_names = (
            "get_person_profile",
            "connect_with_person",
            "get_sidebar_profiles",
            "search_people",
            "get_company_profile",
            "get_company_posts",
            "get_job_details",
            "search_jobs",
            "get_saved_jobs",
            "get_inbox",
            "get_conversation",
            "search_conversations",
            "send_message",
            "message_invitation_sender",
            "get_pending_invitations",
            "get_connections",
            "ignore_connection_request",
            "withdraw_invitation",
            "get_feed",
            "search_posts",
            "close_session",
        )

        for name in tool_names:
            tool = await mcp.get_tool(name)
            assert tool is not None
            assert tool.timeout == custom_timeout

    async def test_all_tools_have_default_timeout(self):
        from linkedin_mcp_server.config.schema import DEFAULT_TOOL_TIMEOUT_SECONDS
        from linkedin_mcp_server.server import create_mcp_server

        mcp = create_mcp_server()

        tool_names = (
            "get_person_profile",
            "get_my_profile",
            "connect_with_person",
            "get_sidebar_profiles",
            "search_people",
            "get_company_profile",
            "get_company_posts",
            "search_companies",
            "get_company_employees",
            "get_job_details",
            "search_jobs",
            "get_saved_jobs",
            "get_inbox",
            "get_conversation",
            "search_conversations",
            "send_message",
            "message_invitation_sender",
            "get_pending_invitations",
            "get_connections",
            "ignore_connection_request",
            "withdraw_invitation",
            "get_feed",
            "search_posts",
            "close_session",
        )

        for name in tool_names:
            tool = await mcp.get_tool(name)
            assert tool is not None
            assert tool.timeout == DEFAULT_TOOL_TIMEOUT_SECONDS
