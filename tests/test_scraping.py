"""Tests for the LinkedInExtractor scraping engine."""

from contextlib import ExitStack, contextmanager
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from linkedin_mcp_server.callbacks import ProgressCallback
from linkedin_mcp_server.core.exceptions import (
    AuthenticationError,
    LinkedInScraperException,
    ProxyConnectionError,
)
from linkedin_mcp_server.scraping.connection import (
    ActionSignals,
    detect_connection_state,
)
from linkedin_mcp_server.scraping.extractor import (
    ExtractedSection,
    LinkedInExtractor,
    _ARCHIVE_CONVERSATION_JS,
    _CONNECTION_CARDS_JS,
    _FEED_POSTS_JS,
    _INVITATION_CARDS_JS,
    _CONTENT_DATE_POSTED_MAP,
    _RATE_LIMITED_MSG,
    _build_feed_references,
    _connection_identity_key,
    _normalize_connection,
    _normalize_structured_invitation,
    _parse_connected_on,
    _truncate_linkedin_noise,
    strip_conversation_chrome,
    strip_linkedin_noise,
)
from linkedin_mcp_server.scraping import conversation as conversation_parser
from linkedin_mcp_server.scraping.link_metadata import Reference


def extracted(
    text: str,
    references: list[Reference] | None = None,
    error: dict | None = None,
) -> ExtractedSection:
    """Create an ExtractedSection for tests."""
    return ExtractedSection(text=text, references=references or [], error=error)


class TestBuildJobSearchUrl:
    """Tests for _build_job_search_url URL construction."""

    def test_keywords_only(self):
        url = LinkedInExtractor._build_job_search_url("python developer")
        assert url == "https://www.linkedin.com/jobs/search/?keywords=python+developer"

    def test_with_location(self):
        url = LinkedInExtractor._build_job_search_url("python", location="Remote")
        assert "keywords=python" in url
        assert "location=Remote" in url

    def test_date_posted_normalization(self):
        url = LinkedInExtractor._build_job_search_url("python", date_posted="past_week")
        assert "f_TPR=r604800" in url

    def test_date_posted_passthrough(self):
        url = LinkedInExtractor._build_job_search_url("python", date_posted="r3600")
        assert "f_TPR=r3600" in url

    def test_experience_level_normalization(self):
        url = LinkedInExtractor._build_job_search_url(
            "python", experience_level="entry"
        )
        assert "f_E=2" in url

    def test_experience_level_csv(self):
        url = LinkedInExtractor._build_job_search_url(
            "python", experience_level="entry,director"
        )
        assert "f_E=2,5" in url

    def test_work_type_normalization(self):
        url = LinkedInExtractor._build_job_search_url("python", work_type="remote")
        assert "f_WT=2" in url

    def test_work_type_csv(self):
        url = LinkedInExtractor._build_job_search_url(
            "python", work_type="on_site,hybrid"
        )
        assert "f_WT=1,3" in url

    def test_easy_apply(self):
        url = LinkedInExtractor._build_job_search_url("python", easy_apply=True)
        assert "f_EA=true" in url

    def test_easy_apply_false_omitted(self):
        url = LinkedInExtractor._build_job_search_url("python", easy_apply=False)
        assert "f_EA" not in url

    def test_sort_by_normalization(self):
        url = LinkedInExtractor._build_job_search_url("python", sort_by="date")
        assert "sortBy=DD" in url

    def test_job_type_normalization(self):
        url = LinkedInExtractor._build_job_search_url("python", job_type="full_time")
        assert "f_JT=F" in url

    def test_job_type_csv(self):
        url = LinkedInExtractor._build_job_search_url(
            "python", job_type="full_time,contract"
        )
        assert "f_JT=F,C" in url

    def test_job_type_passthrough(self):
        url = LinkedInExtractor._build_job_search_url("python", job_type="F")
        assert "f_JT=F" in url

    def test_all_filters_combined(self):
        url = LinkedInExtractor._build_job_search_url(
            "python",
            location="Berlin",
            date_posted="past_week",
            experience_level="entry,mid_senior",
            work_type="remote",
            easy_apply=True,
            sort_by="date",
        )
        assert "keywords=python" in url
        assert "location=Berlin" in url
        assert "f_TPR=r604800" in url
        assert "f_E=2,4" in url
        assert "f_WT=2" in url
        assert "f_EA=true" in url
        assert "sortBy=DD" in url


@pytest.fixture
def mock_page():
    """Create a mock Patchright page."""
    page = MagicMock()
    page.goto = AsyncMock()
    page.title = AsyncMock(return_value="LinkedIn")
    page.wait_for_selector = AsyncMock()
    page.wait_for_function = AsyncMock()
    page.evaluate = AsyncMock(
        return_value={"source": "root", "text": "Sample page text", "references": []}
    )
    page.url = "https://www.linkedin.com/in/testuser/"
    page.locator = MagicMock()
    # Default: no modals, no CAPTCHA
    mock_locator = MagicMock()
    mock_locator.count = AsyncMock(return_value=0)
    mock_locator.is_visible = AsyncMock(return_value=False)
    mock_locator.first = mock_locator
    mock_locator.inner_text = AsyncMock(return_value="normal page content")
    mock_locator.filter = MagicMock(return_value=mock_locator)
    page.locator.return_value = mock_locator
    page.main_frame = object()
    page.on = MagicMock()
    page.remove_listener = MagicMock()
    return page


class TestExtractPage:
    async def test_extract_page_returns_text(self, mock_page):
        mock_page.evaluate = AsyncMock(
            return_value={
                "source": "root",
                "text": "Sample profile text",
                "references": [],
            }
        )
        extractor = LinkedInExtractor(mock_page)
        # Patch scroll_to_bottom and detect_rate_limit to avoid complex mock chains
        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.scroll_to_bottom",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            result = await extractor.extract_page(
                "https://www.linkedin.com/in/testuser/",
                section_name="main_profile",
            )

        assert result.text == "Sample profile text"
        assert result.references == []
        mock_page.goto.assert_awaited_once()

    async def test_root_content_filters_empty_href_before_resolution(self, mock_page):
        mock_page.evaluate = AsyncMock(
            return_value={
                "source": "root",
                "text": "Sample profile text",
                "references": [],
            }
        )
        extractor = LinkedInExtractor(mock_page)

        await extractor._extract_root_content(["main"])

        await_args = mock_page.evaluate.await_args
        assert await_args is not None
        script = await_args.args[0]
        assert "MAX_HEADING_CONTAINERS = 300" in script
        assert "MAX_REFERENCE_ANCHORS = 500" in script
        assert "const getPreviousHeading = node =>" in script
        assert "index < 3" in script
        assert "if (!rawHref || rawHref === '#')" in script
        assert ".slice(0, MAX_REFERENCE_ANCHORS)" in script
        assert "in_list" not in script
        assert ".filter(Boolean);" in script

    async def test_extract_page_returns_empty_on_failure(self, mock_page):
        mock_page.goto = AsyncMock(side_effect=Exception("Network error"))
        extractor = LinkedInExtractor(mock_page)

        with patch(
            "linkedin_mcp_server.scraping.extractor.build_issue_diagnostics",
            return_value={"issue_template_path": "/tmp/issue.md"},
        ):
            result = await extractor.extract_page(
                "https://www.linkedin.com/in/bad/",
                section_name="main_profile",
            )
        assert result.text == ""
        assert result.references == []
        assert result.error == {"issue_template_path": "/tmp/issue.md"}

    async def test_extract_page_raises_auth_error_for_account_picker(self, mock_page):
        mock_page.goto = AsyncMock(side_effect=Exception("net::ERR_TOO_MANY_REDIRECTS"))
        extractor = LinkedInExtractor(mock_page)

        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_auth_barrier",
                new_callable=AsyncMock,
                return_value="auth barrier text: welcome back + sign in using another account",
            ),
            pytest.raises(AuthenticationError, match="--login"),
        ):
            await extractor.extract_page(
                "https://www.linkedin.com/in/testuser/",
                section_name="main_profile",
            )

    async def test_rate_limit_detected(self, mock_page):
        from linkedin_mcp_server.core.exceptions import RateLimitError

        extractor = LinkedInExtractor(mock_page)
        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
                side_effect=RateLimitError("Rate limited", suggested_wait_time=3600),
            ),
            pytest.raises(RateLimitError),
        ):
            await extractor.extract_page(
                "https://www.linkedin.com/in/testuser/",
                section_name="main_profile",
            )

    async def test_returns_rate_limited_msg_after_retry(self, mock_page):
        """When both attempts return only noise, surface rate limit message."""
        noise_only = (
            "More profiles for you\n\n"
            "You've approached your profile search limit\n\n"
            "About\nAccessibility\nTalent Solutions"
        )
        mock_page.evaluate = AsyncMock(
            return_value={"source": "root", "text": noise_only, "references": []}
        )
        extractor = LinkedInExtractor(mock_page)
        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.scroll_to_bottom",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.extract_page(
                "https://www.linkedin.com/in/testuser/details/experience/",
                section_name="experience",
            )

        assert result.text == _RATE_LIMITED_MSG
        # goto called twice (initial + retry)
        assert mock_page.goto.await_count == 2

    async def test_retry_succeeds_after_rate_limit(self, mock_page):
        """When first attempt is rate-limited but retry succeeds, return content."""
        noise_only = "More profiles for you\n\nAbout\nAccessibility\nTalent Solutions"
        call_count = 0

        async def evaluate_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return noise_only
            return "Education\nHarvard University\n1973 – 1975"

        async def root_content_side_effect(*args, **kwargs):
            return {
                "source": "root",
                "text": await evaluate_side_effect(),
                "references": [],
            }

        mock_page.evaluate = AsyncMock(side_effect=root_content_side_effect)
        extractor = LinkedInExtractor(mock_page)
        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.scroll_to_bottom",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.extract_page(
                "https://www.linkedin.com/in/testuser/details/education/",
                section_name="education",
            )

        assert result.text == "Education\nHarvard University\n1973 – 1975"

    async def test_media_only_controls_are_not_misclassified_as_rate_limited(
        self, mock_page
    ):
        mock_page.evaluate = AsyncMock(
            return_value={
                "source": "root",
                "text": "Play\nLoaded: 100.00%\nRemaining time 0:07\nShow captions",
                "references": [],
            }
        )
        extractor = LinkedInExtractor(mock_page)
        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.scroll_to_bottom",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            result = await extractor._extract_page_once(
                "https://www.linkedin.com/in/testuser/recent-activity/all/",
                section_name="posts",
            )

        assert result.text == ""
        assert result.references == []

    async def test_extract_search_page_raises_auth_error_for_login_barrier(
        self, mock_page
    ):
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "_navigate_to_page",
                new_callable=AsyncMock,
                side_effect=AuthenticationError("Run with --login"),
            ),
            pytest.raises(AuthenticationError, match="--login"),
        ):
            await extractor._extract_search_page_once(
                "https://www.linkedin.com/jobs/search/?keywords=test",
                section_name="search_results",
            )


class TestNavigationDiagnostics:
    async def test_goto_with_auth_checks_clicks_remember_me_and_retries(
        self, mock_page
    ):
        extractor = LinkedInExtractor(mock_page)

        async def goto_side_effect(*args, **kwargs):
            if mock_page.goto.await_count == 1:
                raise Exception("net::ERR_TOO_MANY_REDIRECTS")
            return None

        mock_page.goto = AsyncMock(side_effect=goto_side_effect)

        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.resolve_remember_me_prompt",
                new_callable=AsyncMock,
                side_effect=[True],
            ) as mock_resolve,
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_auth_barrier_quick",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            await extractor._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        assert mock_page.goto.await_count == 2
        mock_resolve.assert_awaited_once()

    async def test_goto_with_auth_checks_unhooks_outer_listener_before_retry(
        self, mock_page
    ):
        extractor = LinkedInExtractor(mock_page)
        listener_events: list[str] = []

        def record_on(event_name, callback):
            listener_events.append(f"on:{event_name}")

        def record_remove(event_name, callback):
            listener_events.append(f"off:{event_name}")

        mock_page.on.side_effect = record_on
        mock_page.remove_listener.side_effect = record_remove

        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.resolve_remember_me_prompt",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_auth_barrier_quick",
                new_callable=AsyncMock,
                side_effect=["account picker", None],
            ),
        ):
            await extractor._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        assert listener_events == [
            "on:framenavigated",
            "off:framenavigated",
            "on:framenavigated",
            "off:framenavigated",
        ]

    async def test_goto_with_auth_checks_records_original_failure_before_retry(
        self, mock_page
    ):
        extractor = LinkedInExtractor(mock_page)
        mock_page.goto = AsyncMock(
            side_effect=[
                Exception("net::ERR_TOO_MANY_REDIRECTS"),
                Exception("retry failed"),
            ]
        )

        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.resolve_remember_me_prompt",
                new_callable=AsyncMock,
                side_effect=[True, False],
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.record_page_trace",
                new_callable=AsyncMock,
            ) as mock_trace,
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_auth_barrier",
                new_callable=AsyncMock,
                return_value=None,
            ),
            pytest.raises(Exception, match="retry failed"),
        ):
            await extractor._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        trace_steps = [call.args[1] for call in mock_trace.await_args_list]
        assert "extractor-navigation-error-before-remember-me-retry" in trace_steps

        trace_call = next(
            call
            for call in mock_trace.await_args_list
            if call.args[1] == "extractor-navigation-error-before-remember-me-retry"
        )
        assert (
            trace_call.kwargs["extra"]["error"]
            == "Exception: net::ERR_TOO_MANY_REDIRECTS"
        )

    async def test_goto_with_auth_checks_logs_failure_context(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        mock_page.goto = AsyncMock(side_effect=Exception("net::ERR_TOO_MANY_REDIRECTS"))

        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.resolve_remember_me_prompt",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_auth_barrier",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.object(
                extractor,
                "_log_navigation_failure",
                new_callable=AsyncMock,
            ) as mock_log_failure,
            pytest.raises(Exception, match="ERR_TOO_MANY_REDIRECTS"),
        ):
            await extractor._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        mock_log_failure.assert_awaited_once()
        mock_page.on.assert_called_once()
        mock_page.remove_listener.assert_called_once()


class TestScrapePersonUrls:
    """Test that scrape_person visits the correct URLs per section set."""

    async def test_baseline_always_included(self, mock_page):
        """Passing only experience still visits main profile."""
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "extract_page",
                new_callable=AsyncMock,
                return_value=extracted("text"),
            ) as mock_extract,
            patch.object(
                extractor,
                "_extract_overlay",
                new_callable=AsyncMock,
                return_value=extracted(""),
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.scrape_person("testuser", {"experience"})

        urls = [call.args[0] for call in mock_extract.call_args_list]
        assert "main_profile" in result["sections"]
        assert any(u.endswith("/in/testuser/") for u in urls)
        assert any("/details/experience/" in u for u in urls)

    async def test_basic_info_only_visits_main_profile(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "extract_page",
                new_callable=AsyncMock,
                return_value=extracted("profile text"),
            ) as mock_extract,
            patch.object(
                extractor,
                "_extract_overlay",
                new_callable=AsyncMock,
                return_value=extracted(""),
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.scrape_person("testuser", {"main_profile"})

        urls = [call.args[0] for call in mock_extract.call_args_list]
        assert len(urls) == 1
        assert urls[0].endswith("/in/testuser/")
        assert set(result["sections"]) == {"main_profile"}

    async def test_scrape_person_returns_section_errors(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        # main_profile now returns a structured dict via extract_main_profile
        # (see scraping/main_profile.py). Patch it out so we don't depend on
        # the parser's internals — this test is about section orchestration.
        sentinel_profile = {"name": "profile text"}
        with (
            patch.object(
                extractor,
                "extract_page",
                new_callable=AsyncMock,
                side_effect=[
                    extracted("profile text"),
                    extracted("", error={"issue_template_path": "/tmp/issue.md"}),
                ],
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.extract_main_profile",
                new_callable=AsyncMock,
                return_value=sentinel_profile,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.scrape_person("testuser", {"posts"})

        assert result["sections"]["main_profile"] == sentinel_profile
        assert (
            result["section_errors"]["posts"]["issue_template_path"] == "/tmp/issue.md"
        )

    async def test_experience_education_visits_correct_urls(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "extract_page",
                new_callable=AsyncMock,
                return_value=extracted("text"),
            ) as mock_extract,
            patch.object(
                extractor,
                "_extract_overlay",
                new_callable=AsyncMock,
                return_value=extracted(""),
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.scrape_person(
                "testuser", {"main_profile", "experience", "education"}
            )

        urls = [call.args[0] for call in mock_extract.call_args_list]
        assert len(urls) == 3
        assert any(u.endswith("/in/testuser/") for u in urls)
        assert any("/details/experience/" in u for u in urls)
        assert any("/details/education/" in u for u in urls)
        assert set(result["sections"]) == {"main_profile", "experience", "education"}

    async def test_all_sections_visit_all_urls(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        all_sections = {
            "main_profile",
            "experience",
            "education",
            "interests",
            "honors",
            "languages",
            "certifications",
            "skills",
            "projects",
            "contact_info",
            "posts",
        }
        with (
            patch.object(
                extractor,
                "extract_page",
                new_callable=AsyncMock,
                return_value=extracted("text"),
            ) as mock_extract,
            patch.object(
                extractor,
                "_extract_overlay",
                new_callable=AsyncMock,
                return_value=extracted("contact text"),
            ) as mock_overlay,
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.scrape_person("testuser", all_sections)

        page_urls = [call.args[0] for call in mock_extract.call_args_list]
        overlay_urls = [call.args[0] for call in mock_overlay.call_args_list]
        all_urls = page_urls + overlay_urls
        # 10 full-page sections + 1 overlay (contact_info)
        assert len(page_urls) == 10
        assert len(overlay_urls) == 1
        # Verify each expected suffix was navigated
        assert any(u.endswith("/in/testuser/") for u in all_urls)
        assert any("/details/experience/" in u for u in all_urls)
        assert any("/details/education/" in u for u in all_urls)
        assert any("/details/interests/" in u for u in all_urls)
        assert any("/details/honors/" in u for u in all_urls)
        assert any("/details/languages/" in u for u in all_urls)
        assert any("/details/certifications/" in u for u in all_urls)
        assert any("/details/skills/" in u for u in all_urls)
        assert any("/details/projects/" in u for u in all_urls)
        assert any("/overlay/contact-info/" in u for u in overlay_urls)
        assert any("/recent-activity/all/" in u for u in all_urls)
        assert set(result["sections"]) == all_sections

    async def test_posts_visits_recent_activity(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "extract_page",
                new_callable=AsyncMock,
                return_value=extracted("Post 1\nPost 2"),
            ) as mock_extract,
            patch.object(
                extractor,
                "_extract_overlay",
                new_callable=AsyncMock,
                return_value=extracted(""),
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.scrape_person("test-user", {"posts"})

        urls = [call.args[0] for call in mock_extract.call_args_list]
        assert any("/recent-activity/all/" in url for url in urls)
        assert "posts" in result["sections"]

    async def test_certifications_visits_details_page(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "extract_page",
                new_callable=AsyncMock,
                return_value=extracted("Python for Data Science\nIBM"),
            ) as mock_extract,
            patch.object(
                extractor,
                "_extract_overlay",
                new_callable=AsyncMock,
                return_value=extracted(""),
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.scrape_person("test-user", {"certifications"})

        urls = [call.args[0] for call in mock_extract.call_args_list]
        assert any("/details/certifications/" in url for url in urls)
        assert "certifications" in result["sections"]

    async def test_skills_visits_details_page(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "extract_page",
                new_callable=AsyncMock,
                return_value=extracted("Python\nData Analysis"),
            ) as mock_extract,
            patch.object(
                extractor,
                "_extract_overlay",
                new_callable=AsyncMock,
                return_value=extracted(""),
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.scrape_person("test-user", {"skills"})

        urls = [call.args[0] for call in mock_extract.call_args_list]
        assert any("/details/skills/" in url for url in urls)
        assert "skills" in result["sections"]

    async def test_projects_visits_details_page(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "extract_page",
                new_callable=AsyncMock,
                return_value=extracted("Portfolio Website\nBuilt with React"),
            ) as mock_extract,
            patch.object(
                extractor,
                "_extract_overlay",
                new_callable=AsyncMock,
                return_value=extracted(""),
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.scrape_person("test-user", {"projects"})

        urls = [call.args[0] for call in mock_extract.call_args_list]
        assert any("/details/projects/" in url for url in urls)
        assert "projects" in result["sections"]

    async def test_scrape_person_passes_max_scrolls(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "extract_page",
                new_callable=AsyncMock,
                return_value=extracted("text"),
            ) as mock_extract,
            patch.object(
                extractor,
                "_extract_overlay",
                new_callable=AsyncMock,
                return_value=extracted(""),
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            await extractor.scrape_person(
                "test-user", {"certifications"}, max_scrolls=15
            )

        for call in mock_extract.call_args_list:
            assert call.kwargs.get("max_scrolls") == 15


class TestDetectConnectionState:
    """Tests for locale-independent connection-state detection.

    Every state is decided purely from the structural ActionSignals; no
    profile text is read for any state, including incoming_request (whose
    Accept/Ignore action row is fingerprinted by ``has_incoming_action_row``).
    """

    @staticmethod
    def _signals(
        invite: bool = False,
        compose_in_root: bool = False,
        edit: bool = False,
        labeled_action: bool = False,
        labeled_anchor: bool = False,
        incoming_row: bool = False,
    ) -> ActionSignals:
        return ActionSignals(
            has_invite_anchor=invite,
            has_compose_anchor_in_action_root=compose_in_root,
            has_edit_intro_anchor=edit,
            has_labeled_action_button=labeled_action,
            has_labeled_action_anchor=labeled_anchor,
            has_incoming_action_row=incoming_row,
        )

    def test_self_profile(self):
        assert detect_connection_state(self._signals(edit=True)) == "self_profile"

    def test_connectable(self):
        assert detect_connection_state(self._signals(invite=True)) == "connectable"

    def test_already_connected(self):
        # 1st-degree: Message anchor in action root, but no Follow/Connect/Pending
        # button (no aria-label on any action-root button).
        assert (
            detect_connection_state(
                self._signals(compose_in_root=True, labeled_action=False)
            )
            == "already_connected"
        )

    def test_follow_only(self):
        # No invite anchor anywhere, but a primary action <button> (Follow
        # / Save in Sales Navigator) is present alongside the Message
        # anchor.
        assert (
            detect_connection_state(
                self._signals(compose_in_root=True, labeled_action=True)
            )
            == "follow_only"
        )

    def test_pending_via_labeled_anchor(self):
        # Pending is rendered as <a aria-label="Pending, click to ..."> in
        # the action root — distinct from Follow's <button aria-label=...>.
        assert (
            detect_connection_state(
                self._signals(compose_in_root=True, labeled_anchor=True)
            )
            == "pending"
        )

    def test_pending_takes_priority_over_already_connected(self):
        # If the labeled anchor is present alongside compose-in-root with
        # no labeled button, pending wins over the already_connected
        # fallthrough that would otherwise apply.
        assert (
            detect_connection_state(
                self._signals(compose_in_root=True, labeled_anchor=True)
            )
            == "pending"
        )

    def test_incoming_request_via_structural_row(self):
        assert (
            detect_connection_state(self._signals(incoming_row=True))
            == "incoming_request"
        )

    def test_incoming_structural_beats_pending_misclassification(self):
        # Regression for the sidebar mis-anchor: on incoming profiles the
        # compose-anchor action-root walk lands on sidebar cards and
        # produces garbage signals (compose, labeled button, labeled
        # anchor all True). The structural incoming signal must win over
        # the pending check those garbage signals would trigger.
        assert (
            detect_connection_state(
                self._signals(
                    incoming_row=True,
                    compose_in_root=True,
                    labeled_action=True,
                    labeled_anchor=True,
                )
            )
            == "incoming_request"
        )

    def test_connectable_takes_priority_over_incoming_row(self):
        assert (
            detect_connection_state(self._signals(invite=True, incoming_row=True))
            == "connectable"
        )

    def test_self_profile_takes_priority_over_incoming_row(self):
        assert (
            detect_connection_state(self._signals(edit=True, incoming_row=True))
            == "self_profile"
        )

    def test_unavailable_when_no_signals(self):
        assert detect_connection_state(self._signals()) == "unavailable"

    def test_unavailable_when_compose_missing(self):
        # Restricted profile: no compose anchor, no labels, no invite.
        assert (
            detect_connection_state(self._signals(labeled_action=True)) == "unavailable"
        )


class TestConnectWithPerson:
    @contextmanager
    def _mock_scrape(
        self,
        extractor: LinkedInExtractor,
        profile_text: str,
        *,
        follow_up_text: str | None = None,
    ):
        """Patch both ``scrape_person`` and ``_read_main_innertext``.

        ``scrape_person`` now returns a structured dict for
        ``main_profile`` (see ``scraping/main_profile.py``); the
        connect flow re-reads the raw ``<main>`` innerText via
        ``_read_main_innertext`` for
        :func:`detect_connection_state`'s locale fallback. This
        helper installs both patches at once so connect-flow tests
        only need to swap their old ``patch.object(extractor,
        "scrape_person", ...)`` line for ``self._mock_scrape(
        extractor, text, ...)``.

        When ``follow_up_text`` is given, both mocks are configured
        for two sequential calls (initial state + post-action
        verification).
        """
        sentinel_profile = {"name": "stub"}
        first_section = {
            "url": "https://www.linkedin.com/in/testuser/",
            "sections": {"main_profile": sentinel_profile},
        }
        if follow_up_text is None:
            scrape_mock = AsyncMock(return_value=first_section)
            text_mock = AsyncMock(return_value=profile_text)
        else:
            second_section = {
                "url": "https://www.linkedin.com/in/testuser/",
                "sections": {"main_profile": sentinel_profile},
            }
            scrape_mock = AsyncMock(side_effect=[first_section, second_section])
            text_mock = AsyncMock(side_effect=[profile_text, follow_up_text])

        with (
            patch.object(extractor, "scrape_person", scrape_mock),
            patch.object(extractor, "_read_main_innertext", text_mock),
        ):
            yield

    @staticmethod
    def _signals(
        invite: bool = False,
        compose: bool = False,
        edit: bool = False,
        labeled_action: bool = False,
        labeled_anchor: bool = False,
        incoming_row: bool = False,
    ) -> ActionSignals:
        return ActionSignals(
            has_invite_anchor=invite,
            has_compose_anchor_in_action_root=compose,
            has_edit_intro_anchor=edit,
            has_labeled_action_button=labeled_action,
            has_labeled_action_anchor=labeled_anchor,
            has_incoming_action_row=incoming_row,
        )

    async def test_connectable_navigates_deeplink_and_verifies(self, mock_page):
        """Connect via deeplink: dialog opens, submit succeeds, anchor disappears."""
        extractor = LinkedInExtractor(mock_page)
        text = "Jane\n\n· 3rd\n\nEngineer\n\nConnect\nMore\nAbout\n"
        post_text = "Jane\n\n· 3rd\n\nEngineer\n\nMessage\nPending\nMore\nAbout\n"

        with (
            self._mock_scrape(extractor, text, follow_up_text=post_text),
            patch.object(
                extractor,
                "_read_action_signals",
                new_callable=AsyncMock,
                side_effect=[self._signals(invite=True), self._signals()],
            ),
            patch.object(
                extractor,
                "_navigate_to_page",
                new_callable=AsyncMock,
            ) as mock_nav,
            patch.object(
                extractor,
                "_dialog_is_open",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                extractor,
                "_click_dialog_primary_button",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = await extractor.connect_with_person("testuser")

        assert result["status"] == "connected"
        mock_nav.assert_awaited_once()
        await_args = mock_nav.await_args
        assert await_args is not None
        assert "preload/custom-invite" in await_args.args[0]

    async def test_connectable_send_failed_when_anchor_persists(self, mock_page):
        """Dialog submitted but profile still exposes Connect → send_failed."""
        extractor = LinkedInExtractor(mock_page)
        text = "Jane\n\n· 3rd\n\nEngineer\n\nConnect\nMore\nAbout\n"

        with (
            self._mock_scrape(extractor, text, follow_up_text=text),
            patch.object(
                extractor,
                "_read_action_signals",
                new_callable=AsyncMock,
                side_effect=[self._signals(invite=True), self._signals(invite=True)],
            ),
            patch.object(extractor, "_navigate_to_page", new_callable=AsyncMock),
            patch.object(
                extractor, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                extractor,
                "_click_dialog_primary_button",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = await extractor.connect_with_person("testuser")

        assert result["status"] == "send_failed"

    async def test_premium_upsell_message_reads_linkedin_dialog_text(self, mock_page):
        """Premium upsell detection returns LinkedIn's raw dialog text."""
        extractor = LinkedInExtractor(mock_page)
        premium_link = MagicMock()
        premium_link.wait_for = AsyncMock(return_value=None)
        premium_link.is_visible = AsyncMock(return_value=True)
        premium_link.inner_text = AsyncMock(return_value="fallback")
        premium_link.first = premium_link
        mock_page.locator.return_value = premium_link
        mock_page.evaluate = AsyncMock(
            return_value="Wysyłaj nieograniczoną liczbę spersonalizowanych zaproszeń dzięki Premium"
        )

        result = await extractor._get_premium_upsell_message(timeout=1234)

        assert (
            result
            == "Wysyłaj nieograniczoną liczbę spersonalizowanych zaproszeń dzięki Premium"
        )
        mock_page.locator.assert_called_once_with(
            'dialog[open] a[href*="/premium/"], [role="dialog"] a[href*="/premium/"]'
        )
        premium_link.wait_for.assert_awaited_once_with(state="visible", timeout=1234)

    async def test_submit_invite_dialog_reports_premium_after_add_note(self, mock_page):
        """Add-note Premium upsell is a note-limit block, not no-dialog."""
        from patchright.async_api import TimeoutError as PlaywrightTimeoutError

        extractor = LinkedInExtractor(mock_page)
        textarea = MagicMock()
        textarea.count = AsyncMock(return_value=0)
        add_note_button = MagicMock()
        add_note_button.click = AsyncMock(return_value=None)
        buttons = MagicMock()
        buttons.count = AsyncMock(return_value=3)
        buttons.nth.return_value = add_note_button

        def locator_for(selector: str):
            return textarea if "textarea" in selector else buttons

        mock_page.locator.side_effect = locator_for
        mock_page.wait_for_selector = AsyncMock(
            side_effect=PlaywrightTimeoutError("textarea timeout")
        )

        with (
            patch.object(
                extractor, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                extractor,
                "_get_premium_upsell_message",
                new_callable=AsyncMock,
                return_value="Wysyłaj nieograniczoną liczbę spersonalizowanych zaproszeń dzięki Premium",
            ) as mock_message,
            patch.object(
                extractor, "_dismiss_dialog", new_callable=AsyncMock
            ) as mock_dismiss,
        ):
            result = await extractor._submit_invite_dialog("Hello")

        assert result == (
            False,
            False,
            "Wysyłaj nieograniczoną liczbę spersonalizowanych zaproszeń dzięki Premium",
        )
        add_note_button.click.assert_awaited_once()
        mock_message.assert_awaited_once()
        mock_dismiss.assert_awaited_once()

    async def test_submit_invite_dialog_reports_premium_after_send_click_failure(
        self, mock_page
    ):
        """Premium upsell intercepting the Send click is a note-limit block.

        When LinkedIn swaps the invite dialog for the Premium upsell at the
        moment of submit, the original primary button is detached or pointer-
        event covered, so ``_click_dialog_primary_button`` and the keyboard
        fallback both fail. Without the post-click upsell probe the caller
        would dismiss the dialog and report ``connect_unavailable`` even
        though LinkedIn's raw quota message is sitting in the visible modal.
        """
        extractor = LinkedInExtractor(mock_page)

        # Textarea already exposed so the reveal/fill branch succeeds and the
        # test focuses on the post-submit failure path.
        textarea = MagicMock()
        textarea.count = AsyncMock(return_value=1)
        textarea.first = textarea
        textarea.fill = AsyncMock()

        buttons = MagicMock()
        buttons.count = AsyncMock(return_value=2)
        primary_button = MagicMock()
        primary_button.focus = AsyncMock()
        buttons.nth.return_value = primary_button

        def locator_for(selector: str):
            return textarea if "textarea" in selector else buttons

        mock_page.locator.side_effect = locator_for
        mock_page.keyboard = MagicMock()
        mock_page.keyboard.press = AsyncMock()

        message = "You're out of free custom notes. Bypass the limit with Premium..."

        with (
            patch.object(
                extractor,
                "_dialog_is_open",
                new_callable=AsyncMock,
                # First call: dialog open at entry. Second call: still open
                # after the keyboard fallback, so sent remains False.
                side_effect=[True, True],
            ),
            patch.object(
                extractor,
                "_fill_dialog_textarea",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                extractor,
                "_click_dialog_primary_button",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                extractor,
                "_get_premium_upsell_message",
                new_callable=AsyncMock,
                return_value=message,
            ) as mock_message,
            patch.object(
                extractor, "_dismiss_dialog", new_callable=AsyncMock
            ) as mock_dismiss,
        ):
            result = await extractor._submit_invite_dialog("Hello")

        assert result == (False, False, message)
        mock_message.assert_awaited_once()
        mock_dismiss.assert_awaited_once()

    async def test_connectable_no_dialog_returns_connect_unavailable(self, mock_page):
        """Deeplink opened but no dialog appeared → connect_unavailable."""
        extractor = LinkedInExtractor(mock_page)
        text = "Jane\n\n· 3rd\n\nEngineer\n\nConnect\nMore\nAbout\n"

        with (
            self._mock_scrape(extractor, text),
            patch.object(
                extractor,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=self._signals(invite=True),
            ),
            patch.object(extractor, "_navigate_to_page", new_callable=AsyncMock),
            patch.object(
                extractor, "_dialog_is_open", new_callable=AsyncMock, return_value=False
            ),
            patch.object(extractor, "_dismiss_dialog", new_callable=AsyncMock),
        ):
            result = await extractor.connect_with_person("testuser")

        assert result["status"] == "connect_unavailable"

    async def test_returns_already_connected_via_anchor(self, mock_page):
        """1st-degree detected via /messaging/compose anchor."""
        extractor = LinkedInExtractor(mock_page)
        text = "Collin\n\n· 1st\n\nEngineer\n\nMessage\nMore\nAbout\n"

        with (
            self._mock_scrape(extractor, text),
            patch.object(
                extractor,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=self._signals(compose=True),
            ),
        ):
            result = await extractor.connect_with_person("testuser")

        assert result["status"] == "already_connected"

    async def test_returns_self_profile_via_edit_intro_anchor(self, mock_page):
        """Editing-your-own-profile anchor blocks connect attempts."""
        extractor = LinkedInExtractor(mock_page)
        text = "Daniel\n\nEdit profile\n"

        with (
            self._mock_scrape(extractor, text),
            patch.object(
                extractor,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=self._signals(edit=True),
            ),
        ):
            result = await extractor.connect_with_person("testuser")

        assert result["status"] == "connect_unavailable"
        assert "own profile" in result["message"]

    async def test_connect_via_more_menu(self, mock_page):
        """Follow-primary profile with Connect under More: detection sees
        no invite anchor initially, _open_more_menu surfaces it, deeplink
        fires."""
        extractor = LinkedInExtractor(mock_page)
        # Pre-More: Follow primary, Connect hidden under the More dropdown.
        pre = "Christian\n\n· 2nd\n\nFounder\n\nFollow\nMessage\nMore\n"
        post = "Christian\n\n· 2nd\n\nFounder\n\nMessage\nPending\nMore\n"

        with (
            self._mock_scrape(extractor, pre, follow_up_text=post),
            patch.object(
                extractor,
                "_read_action_signals",
                new_callable=AsyncMock,
                # 1st: follow_only (compose+labeled, no invite).
                # 2nd: post-More reread reveals invite anchor.
                # 3rd: post-deeplink verification — invite anchor gone.
                side_effect=[
                    self._signals(compose=True, labeled_action=True),
                    self._signals(invite=True, compose=True, labeled_action=True),
                    self._signals(),
                ],
            ),
            patch.object(
                extractor,
                "_open_more_menu",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_open_more,
            patch.object(
                extractor, "_navigate_to_page", new_callable=AsyncMock
            ) as mock_nav,
            patch.object(
                extractor,
                "_dialog_is_open",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                extractor,
                "_click_dialog_primary_button",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = await extractor.connect_with_person("testuser")

        assert result["status"] == "connected"
        mock_open_more.assert_awaited_once()
        # Deeplink fired exactly once.
        assert mock_nav.await_count == 1
        await_args = mock_nav.await_args
        assert await_args is not None
        assert "preload/custom-invite" in await_args.args[0]

    async def test_follow_only_after_more_does_not_send(self, mock_page):
        """Pending or genuinely follow-only profile: invite anchor never
        appears even after More-menu open. Critical write-gate guardrail —
        no deeplink fires, no connection request goes out."""
        extractor = LinkedInExtractor(mock_page)
        text = "Public Figure\n\n· 3rd+\n\nCEO\n\nFollow\nMessage\nMore\n"

        with (
            self._mock_scrape(extractor, text),
            patch.object(
                extractor,
                "_read_action_signals",
                new_callable=AsyncMock,
                # Both reads (initial + post-More) show no invite anchor.
                side_effect=[
                    self._signals(compose=True, labeled_action=True),
                    self._signals(compose=True, labeled_action=True),
                ],
            ),
            patch.object(
                extractor,
                "_open_more_menu",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_open_more,
            patch.object(
                extractor, "_navigate_to_page", new_callable=AsyncMock
            ) as mock_nav,
            patch.object(
                extractor, "_submit_invite_dialog", new_callable=AsyncMock
            ) as mock_submit,
        ):
            result = await extractor.connect_with_person("testuser")

        assert result["status"] == "connect_unavailable"
        assert result.get("note_sent") is False or "note_sent" not in result
        mock_open_more.assert_awaited_once()
        # Critical: deeplink must NOT fire and dialog must NOT be submitted.
        mock_nav.assert_not_awaited()
        mock_submit.assert_not_awaited()

    async def test_follow_only_with_note_reports_note_limit_from_deeplink_probe(
        self, mock_page
    ):
        """A requested note may reveal Premium quota without submitting."""
        extractor = LinkedInExtractor(mock_page)
        text = "Public Figure\n\n· 3rd+\n\nCEO\n\nFollow\nMessage\nMore\n"

        with (
            self._mock_scrape(extractor, text),
            patch.object(
                extractor,
                "_read_action_signals",
                new_callable=AsyncMock,
                side_effect=[
                    self._signals(compose=True, labeled_action=True),
                    self._signals(compose=True, labeled_action=True),
                ],
            ),
            patch.object(
                extractor,
                "_open_more_menu",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                extractor, "_navigate_to_page", new_callable=AsyncMock
            ) as mock_nav,
            patch.object(
                extractor,
                "_probe_invite_note_limit",
                new_callable=AsyncMock,
                return_value="Wysyłaj nieograniczoną liczbę spersonalizowanych zaproszeń dzięki Premium",
            ) as mock_probe,
            patch.object(
                extractor, "_submit_invite_dialog", new_callable=AsyncMock
            ) as mock_submit,
        ):
            result = await extractor.connect_with_person("testuser", note="Hello")

        assert result["status"] == "custom_note_limit_reached"
        assert (
            result["message"]
            == "Wysyłaj nieograniczoną liczbę spersonalizowanych zaproszeń dzięki Premium"
        )
        assert result["note_sent"] is False
        mock_nav.assert_awaited_once()
        mock_probe.assert_awaited_once()
        mock_submit.assert_not_awaited()

    async def test_more_menu_unavailable_does_not_send(self, mock_page):
        """Action root present but no More button (unusual but possible):
        _open_more_menu returns False, no retry, no deeplink fires."""
        extractor = LinkedInExtractor(mock_page)
        text = "Public Figure\n\n· 3rd+\n\nCEO\n\nFollow\nMessage\n"

        with (
            self._mock_scrape(extractor, text),
            patch.object(
                extractor,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=self._signals(compose=True, labeled_action=True),
            ),
            patch.object(
                extractor,
                "_open_more_menu",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch.object(
                extractor, "_navigate_to_page", new_callable=AsyncMock
            ) as mock_nav,
            patch.object(
                extractor, "_submit_invite_dialog", new_callable=AsyncMock
            ) as mock_submit,
        ):
            result = await extractor.connect_with_person("testuser")

        assert result["status"] == "connect_unavailable"
        mock_nav.assert_not_awaited()
        mock_submit.assert_not_awaited()

    async def test_returns_pending(self, mock_page):
        """Profile with a pending invitation: detected via labeled <a> in
        the action root. Returns status='pending' without firing the
        deeplink (LinkedIn would only show 'already invited' anyway)."""
        extractor = LinkedInExtractor(mock_page)
        text = "Frank\n\n· 3rd\n\nFounder\n\nMessage\nPending\nMore\n"

        with (
            self._mock_scrape(extractor, text),
            patch.object(
                extractor,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=self._signals(compose=True, labeled_anchor=True),
            ),
            patch.object(
                extractor, "_navigate_to_page", new_callable=AsyncMock
            ) as mock_nav,
            patch.object(
                extractor, "_submit_invite_dialog", new_callable=AsyncMock
            ) as mock_submit,
            patch.object(
                extractor,
                "_respond_via_received_invitations",
                new_callable=AsyncMock,
                return_value={"status": "not_found"},
            ) as mock_accept,
        ):
            result = await extractor.connect_with_person("testuser")

        assert result["status"] == "pending"
        mock_accept.assert_awaited_once_with("testuser", "accept")
        # No write-path side effects.
        mock_nav.assert_not_awaited()
        mock_submit.assert_not_awaited()

    @pytest.mark.parametrize(
        ("action_status", "expected_status"),
        [("accepted", "accepted"), ("verification_failed", "send_failed")],
    )
    async def test_pending_received_invitation_uses_connection_result(
        self, mock_page, action_status, expected_status
    ):
        """A received invitation can share the outgoing Pending fingerprint."""
        extractor = LinkedInExtractor(mock_page)
        text = "Alice\n\nAccept\nIgnore\n"

        with (
            self._mock_scrape(extractor, text),
            patch.object(
                extractor,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=self._signals(labeled_anchor=True),
            ),
            patch.object(
                extractor,
                "_respond_via_received_invitations",
                new_callable=AsyncMock,
                return_value={
                    "status": action_status,
                    "message": "Invitation action result.",
                },
            ) as mock_accept,
        ):
            result = await extractor.connect_with_person("alice")

        assert result == {
            "url": "https://www.linkedin.com/in/alice/",
            "status": expected_status,
            "message": "Invitation action result.",
            "note_sent": False,
            "profile": text,
        }
        mock_accept.assert_awaited_once_with("alice", "accept")

    async def test_returns_incoming_request_accepted(self, mock_page):
        """Structural detection + structural accept click, German locale."""
        extractor = LinkedInExtractor(mock_page)
        pre = "Eric\n\n· 2.\n\nAachen\n\nAnnehmen\nIgnorieren\nMehr\nInfo\n"
        post = "Eric\n\n· 1.\n\nAachen\n\nNachricht\nMehr\nInfo\n"

        with (
            self._mock_scrape(extractor, pre, follow_up_text=post),
            patch.object(
                extractor,
                "_read_action_signals",
                new_callable=AsyncMock,
                side_effect=[
                    self._signals(incoming_row=True),
                    self._signals(compose=True),
                ],
            ),
            patch.object(
                extractor,
                "_click_incoming_accept",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_accept,
            patch.object(
                extractor,
                "_navigate_to_page",
                new_callable=AsyncMock,
            ) as mock_nav,
            patch.object(
                extractor,
                "_submit_invite_dialog",
                new_callable=AsyncMock,
            ) as mock_submit,
        ):
            result = await extractor.connect_with_person("testuser")

        assert result["status"] == "accepted"
        mock_accept.assert_awaited_once()
        mock_nav.assert_not_awaited()
        mock_submit.assert_not_awaited()

    async def test_incoming_request_send_failed_when_click_fails(self, mock_page):
        """Structural accept click did not land; no locale-text guessing —
        report send_failed without navigating or clicking by text."""
        extractor = LinkedInExtractor(mock_page)
        pre = "Eric\n\n· 2.\n\nAachen\n\nAnnehmen\nIgnorieren\nMehr\nInfo\n"

        with (
            self._mock_scrape(extractor, pre),
            patch.object(
                extractor,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=self._signals(incoming_row=True),
            ),
            patch.object(
                extractor,
                "_click_incoming_accept",
                new_callable=AsyncMock,
                return_value=False,
            ) as mock_text_click,
            patch.object(
                extractor,
                "_navigate_to_page",
                new_callable=AsyncMock,
            ) as mock_nav,
        ):
            result = await extractor.connect_with_person("testuser")

        assert result["status"] == "send_failed"
        mock_nav.assert_not_awaited()
        mock_text_click.assert_awaited_once()

    async def test_incoming_request_send_failed_when_no_first_degree(self, mock_page):
        """Accept clicked but profile never transitions to 1st-degree."""
        extractor = LinkedInExtractor(mock_page)
        pre = "Eric\n\n· 2.\n\nAachen\n\nAnnehmen\nIgnorieren\nMehr\nInfo\n"

        with (
            patch.object(
                extractor,
                "scrape_person",
                AsyncMock(
                    return_value={
                        "url": "https://www.linkedin.com/in/testuser/",
                        "sections": {"main_profile": pre},
                    }
                ),
            ),
            patch.object(
                extractor,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=self._signals(incoming_row=True),
            ),
            patch.object(
                extractor,
                "_click_incoming_accept",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.connect_with_person("testuser")

        assert result["status"] == "send_failed"

    async def test_incoming_request_accepted_on_settle_retry(self, mock_page):
        """The first post-click read still renders the old top card;
        the settle retry sees the 1st-degree state and reports accepted."""
        extractor = LinkedInExtractor(mock_page)
        pre = "Eric\n\n· 2.\n\nAachen\n\nAnnehmen\nIgnorieren\nMehr\nInfo\n"
        post = "Eric\n\n· 1.\n\nAachen\n\nNachricht\nMehr\nInfo\n"
        page = {
            "url": "https://www.linkedin.com/in/testuser/",
            "sections": {"main_profile": pre},
        }
        page_post = {
            "url": "https://www.linkedin.com/in/testuser/",
            "sections": {"main_profile": post},
        }

        with (
            patch.object(
                extractor,
                "scrape_person",
                AsyncMock(side_effect=[page, page, page_post]),
            ),
            patch.object(
                extractor,
                "_read_action_signals",
                new_callable=AsyncMock,
                side_effect=[
                    self._signals(incoming_row=True),
                    self._signals(incoming_row=True),
                    self._signals(compose=True),
                ],
            ),
            patch.object(
                extractor,
                "_click_incoming_accept",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ) as mock_sleep,
        ):
            result = await extractor.connect_with_person("testuser")

        assert result["status"] == "accepted"
        mock_sleep.assert_awaited_once()

    async def test_returns_unavailable_when_no_signals_and_text(self, mock_page):
        """No structural signals, no actionable text → connect_unavailable."""
        extractor = LinkedInExtractor(mock_page)
        text = "Public Figure\n\n· 3rd+\n\nCEO\n\nFollow\nMore\nAbout\n"

        with (
            self._mock_scrape(extractor, text),
            patch.object(
                extractor,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=self._signals(),
            ),
            patch.object(extractor, "_navigate_to_page", new_callable=AsyncMock),
            patch.object(
                extractor, "_dialog_is_open", new_callable=AsyncMock, return_value=False
            ),
            patch.object(extractor, "_dismiss_dialog", new_callable=AsyncMock),
        ):
            result = await extractor.connect_with_person("testuser")

        # follow_only path goes through deeplink; no dialog opens → unavailable
        assert result["status"] == "connect_unavailable"

    async def test_returns_unavailable_on_empty_page(self, mock_page):
        extractor = LinkedInExtractor(mock_page)

        with patch.object(
            extractor,
            "scrape_person",
            AsyncMock(
                return_value={
                    "url": "https://www.linkedin.com/in/testuser/",
                    "sections": {},
                }
            ),
        ):
            result = await extractor.connect_with_person("testuser")

        assert result["status"] == "unavailable"

    async def test_submit_invite_dialog_handles_two_button_gating_dialog(
        self, mock_page
    ):
        """Two-button "Add a note to your invitation?" gating dialog (issue
        #455): nth(0) is "Add a note", nth(1) is "Send without a note".

        Asserts the secondary-button click that reveals the textarea fires
        even with btn_count == 2 (legacy guard required >= 3 and skipped
        the click, leaving the textarea unmounted)."""
        extractor = LinkedInExtractor(mock_page)

        # Track each button click so we can assert the "Add a note" path
        # was taken to reveal the textarea.
        clicks: list[int] = []

        textarea_visible = {"value": False}

        # Two button locators inside the gating dialog: nth(0) "Add a
        # note" reveals the textarea, nth(1) "Send without a note".
        button_locators = [MagicMock(), MagicMock()]
        for idx, btn in enumerate(button_locators):

            def make_click(i: int):
                async def _click(*args, **kwargs):
                    clicks.append(i)
                    if i == 0:
                        textarea_visible["value"] = True
                    return None

                return _click

            btn.click = AsyncMock(side_effect=make_click(idx))
            btn.focus = AsyncMock()

        button_collection = MagicMock()
        button_collection.count = AsyncMock(return_value=2)
        button_collection.nth = MagicMock(side_effect=lambda i: button_locators[i])

        textarea_locator = MagicMock()
        textarea_locator.count = AsyncMock(
            side_effect=lambda: 1 if textarea_visible["value"] else 0
        )
        textarea_locator.first = textarea_locator
        textarea_locator.fill = AsyncMock()

        # Route page.locator() calls by selector — buttons vs textarea —
        # so the gating dialog's button collection is distinguishable
        # from the textarea probe.
        def locator_router(selector: str):
            if "textarea" in selector:
                return textarea_locator
            return button_collection

        mock_page.locator = MagicMock(side_effect=locator_router)
        mock_page.wait_for_selector = AsyncMock()
        mock_page.keyboard = MagicMock()
        mock_page.keyboard.press = AsyncMock()

        with (
            patch.object(
                extractor, "_dialog_is_open", new_callable=AsyncMock, return_value=True
            ),
            patch.object(
                extractor,
                "_get_premium_upsell_message",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            (
                submitted,
                note_sent,
                note_limit_message,
            ) = await extractor._submit_invite_dialog("Hi from a test")

        assert submitted is True
        assert note_sent is True
        assert note_limit_message is None
        # Clicked "Add a note" (index 0) to reveal the textarea, then the
        # primary button (index 1) to send.
        assert clicks == [0, 1]
        textarea_locator.fill.assert_awaited_once()

    async def test_references_are_grouped_by_section(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "extract_page",
                new_callable=AsyncMock,
                side_effect=[
                    extracted(
                        "profile text",
                        [
                            {
                                "kind": "person",
                                "url": "/in/testuser/",
                                "text": "Test User",
                            }
                        ],
                    ),
                    extracted(
                        "post text",
                        [
                            {
                                "kind": "article",
                                "url": "/pulse/test-post/",
                                "text": "Test post",
                            }
                        ],
                    ),
                ],
            ),
            patch.object(
                extractor,
                "_extract_overlay",
                new_callable=AsyncMock,
                return_value=extracted(""),
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.scrape_person("testuser", {"posts"})

        assert result["references"] == {
            "main_profile": [
                {"kind": "person", "url": "/in/testuser/", "text": "Test User"}
            ],
            "posts": [
                {"kind": "article", "url": "/pulse/test-post/", "text": "Test post"}
            ],
        }

    async def test_error_isolation(self, mock_page):
        """One section failing doesn't block others."""

        async def extract_with_failure(url, *args, **kwargs):
            if "experience" in url:
                raise Exception("Simulated failure")
            return extracted(f"text for {url}")

        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "extract_page",
                side_effect=extract_with_failure,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.build_issue_diagnostics",
                return_value={"issue_template_path": "/tmp/issue.md"},
            ),
            patch.object(
                extractor,
                "_extract_overlay",
                new_callable=AsyncMock,
                return_value=extracted(""),
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.scrape_person(
                "testuser", {"main_profile", "experience", "education"}
            )

        # main_profile and education should have sections, experience should not
        assert "main_profile" in result["sections"]
        assert "education" in result["sections"]
        assert "experience" not in result["sections"]
        assert result["section_errors"]["experience"]["issue_template_path"] == (
            "/tmp/issue.md"
        )

    async def test_rate_limited_sections_are_omitted(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "extract_page",
                new_callable=AsyncMock,
                side_effect=[
                    extracted(_RATE_LIMITED_MSG),
                    extracted("Post text"),
                ],
            ),
            patch.object(
                extractor,
                "_extract_overlay",
                new_callable=AsyncMock,
                return_value=extracted(""),
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.scrape_person("testuser", {"posts"})

        assert "main_profile" not in result["sections"]
        assert result["sections"]["posts"] == "Post text"


class TestScrapeCompany:
    async def test_company_baseline_always_included(self, mock_page):
        """Passing only posts still visits about page."""
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "extract_page",
                new_callable=AsyncMock,
                return_value=extracted("text"),
            ) as mock_extract,
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.scrape_company("testcorp", {"posts"})

        urls = [call.args[0] for call in mock_extract.call_args_list]
        assert any("/about/" in u for u in urls)
        assert any("/posts/" in u for u in urls)
        assert "about" in result["sections"]
        assert "posts" in result["sections"]

    async def test_about_only_visits_about(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "extract_page",
                new_callable=AsyncMock,
                return_value=extracted("about text"),
            ) as mock_extract,
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.scrape_company("testcorp", {"about"})

        urls = [call.args[0] for call in mock_extract.call_args_list]
        assert len(urls) == 1
        assert "/about/" in urls[0]
        assert set(result["sections"]) == {"about"}

    async def test_all_sections_visit_correct_urls(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "extract_page",
                new_callable=AsyncMock,
                return_value=extracted("text"),
            ) as mock_extract,
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.scrape_company(
                "testcorp", {"about", "posts", "jobs"}
            )

        urls = [call.args[0] for call in mock_extract.call_args_list]
        assert len(urls) == 3
        assert any("/about/" in u for u in urls)
        assert any("/posts/" in u for u in urls)
        assert any("/jobs/" in u for u in urls)
        assert set(result["sections"]) == {"about", "posts", "jobs"}

    async def test_rate_limited_company_sections_are_omitted(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "extract_page",
                new_callable=AsyncMock,
                side_effect=[
                    extracted(_RATE_LIMITED_MSG),
                    extracted("Posts text"),
                ],
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.scrape_company("testcorp", {"posts"})

        assert "about" not in result["sections"]
        assert result["sections"]["posts"] == "Posts text"

    async def test_scrape_company_extracts_company_urn(self, mock_page):
        """End-to-end: a canned-search anchor on the company about page
        produces a ``company_urn`` reference with the parent-company id.

        Stubs ``_extract_root_content`` (rather than ``extract_page``) so
        the real ``build_references`` pipeline runs against raw anchor
        data, mirroring what the JS crawler emits live.
        """
        extractor = LinkedInExtractor(mock_page)
        raw_root = {
            "source": "root",
            "text": "About SAP\nCompany overview",
            "references": [
                {
                    "href": "https://www.linkedin.com/search/results/people/"
                    "?currentCompany=%5B%221115%22%5D"
                    "&origin=COMPANY_PAGE_CANNED_SEARCH",
                    "text": "10K+ employees",
                    "aria_label": "",
                    "title": "",
                    "heading": "",
                    "in_article": False,
                    "in_nav": False,
                    "in_footer": False,
                }
            ],
        }
        with (
            patch.object(
                extractor,
                "_extract_root_content",
                new_callable=AsyncMock,
                return_value=raw_root,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.scroll_to_bottom",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.scrape_company("sap", {"about"})

        urns = [
            ref for ref in result["references"]["about"] if ref["kind"] == "company_urn"
        ]
        assert len(urns) == 1
        assert urns[0]["value"] == "1115"
        assert urns[0]["url"] == (
            "/search/results/people/?currentCompany=%5B%221115%22%5D"
        )
        assert "text" not in urns[0]


class TestScrapeJob:
    async def test_scrape_job(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with patch.object(
            extractor,
            "extract_page",
            new_callable=AsyncMock,
            return_value=extracted("Job: Software Engineer"),
        ):
            result = await extractor.scrape_job("12345")

        assert result["url"] == "https://www.linkedin.com/jobs/view/12345/"
        assert "job_posting" in result["sections"]
        assert "pages_visited" not in result
        assert "sections_requested" not in result

    async def test_scrape_job_omits_rate_limited_sentinel(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with patch.object(
            extractor,
            "extract_page",
            new_callable=AsyncMock,
            return_value=extracted(_RATE_LIMITED_MSG),
        ):
            result = await extractor.scrape_job("12345")

        assert result["sections"] == {}

    async def test_scrape_job_omits_orphaned_references_when_text_empty(
        self, mock_page
    ):
        extractor = LinkedInExtractor(mock_page)
        with patch.object(
            extractor,
            "extract_page",
            new_callable=AsyncMock,
            return_value=extracted(
                "",
                [{"kind": "job", "url": "/jobs/view/12345/", "text": "Engineer"}],
            ),
        ):
            result = await extractor.scrape_job("12345")

        assert result["sections"] == {}
        assert "references" not in result


class TestSearchJobs:
    """Tests for search_jobs with job ID extraction and pagination."""

    @pytest.fixture(autouse=True)
    def _set_search_url(self, mock_page):
        mock_page.url = "https://www.linkedin.com/jobs/search/?keywords=python"

    async def test_returns_job_ids(self, mock_page):
        """search_jobs should return a job_ids list extracted from hrefs."""
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "_extract_search_page",
                new_callable=AsyncMock,
                return_value=extracted("Job 1\nJob 2\nJob 3"),
            ),
            patch.object(
                extractor,
                "_extract_job_ids",
                new_callable=AsyncMock,
                return_value=["111", "222", "333"],
            ),
            patch.object(
                extractor,
                "_get_total_search_pages",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.search_jobs("python", max_pages=1)

        assert result["job_ids"] == ["111", "222", "333"]
        assert "search_results" in result["sections"]

    async def test_returns_references(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "_extract_search_page",
                new_callable=AsyncMock,
                return_value=extracted(
                    "Job 1",
                    [{"kind": "job", "url": "/jobs/view/111/", "text": "Job 1"}],
                ),
            ),
            patch.object(
                extractor,
                "_extract_job_ids",
                new_callable=AsyncMock,
                return_value=["111"],
            ),
            patch.object(
                extractor,
                "_get_total_search_pages",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.search_jobs("python", max_pages=1)

        assert result["references"] == {
            "search_results": [
                {"kind": "job", "url": "/jobs/view/111/", "text": "Job 1"}
            ]
        }

    async def test_pagination_uses_fixed_page_size(self, mock_page):
        """Pages use &start= with fixed 25-per-page offset."""
        extractor = LinkedInExtractor(mock_page)
        page1_ids = ["100", "200", "300"]
        page2_ids = ["400", "500"]
        id_pages = iter([page1_ids, page2_ids])
        text_pages = iter(["Page 1 text", "Page 2 text"])
        urls_visited: list[str] = []

        async def mock_extract(url, *args, **kwargs):
            urls_visited.append(url)
            return extracted(next(text_pages))

        with (
            patch.object(extractor, "_extract_search_page", side_effect=mock_extract),
            patch.object(
                extractor,
                "_extract_job_ids",
                new_callable=AsyncMock,
                side_effect=lambda: next(id_pages),
            ),
            patch.object(
                extractor,
                "_get_total_search_pages",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.search_jobs("python", max_pages=2)

        assert result["job_ids"] == ["100", "200", "300", "400", "500"]
        assert len(urls_visited) == 2
        assert "&start=25" in urls_visited[1]

    async def test_deduplication_across_pages(self, mock_page):
        """Duplicate job IDs across pages should be deduplicated."""
        extractor = LinkedInExtractor(mock_page)
        id_pages = iter([["100", "200"], ["200", "300"]])
        with (
            patch.object(
                extractor,
                "_extract_search_page",
                new_callable=AsyncMock,
                return_value=extracted("text"),
            ) as mock_extract,
            patch.object(
                extractor,
                "_extract_job_ids",
                new_callable=AsyncMock,
                side_effect=lambda: next(id_pages),
            ),
            patch.object(
                extractor,
                "_get_total_search_pages",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.search_jobs("python", max_pages=2)

        assert result["job_ids"] == ["100", "200", "300"]
        assert mock_extract.await_count == 2

    async def test_early_stop_no_new_ids(self, mock_page):
        """Should stop early when a page yields no new job IDs."""
        extractor = LinkedInExtractor(mock_page)
        # Page 2 returns same IDs as page 1
        id_pages = iter([["100", "200"], ["100", "200"]])
        extract_call_count = 0

        async def mock_extract(url, *args, **kwargs):
            nonlocal extract_call_count
            extract_call_count += 1
            if extract_call_count == 1:
                return extracted(
                    "text",
                    [{"kind": "job", "url": "/jobs/view/100/", "text": "Job 100"}],
                )
            return extracted(
                "text",
                [{"kind": "job", "url": "/jobs/view/200/", "text": "Job 200"}],
            )

        with (
            patch.object(extractor, "_extract_search_page", side_effect=mock_extract),
            patch.object(
                extractor,
                "_extract_job_ids",
                new_callable=AsyncMock,
                side_effect=lambda: next(id_pages),
            ),
            patch.object(
                extractor,
                "_get_total_search_pages",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.search_jobs("python", max_pages=5)

        assert result["job_ids"] == ["100", "200"]
        assert extract_call_count == 2
        assert result["references"] == {
            "search_results": [
                {"kind": "job", "url": "/jobs/view/100/", "text": "Job 100"},
                {"kind": "job", "url": "/jobs/view/200/", "text": "Job 200"},
            ]
        }

    async def test_stops_at_total_pages(self, mock_page):
        """Should stop when total_pages from pagination state is reached."""
        extractor = LinkedInExtractor(mock_page)
        # Distinct IDs per page so the no-new-IDs guard never fires
        id_pages = iter([["100"], ["200"]])
        with (
            patch.object(
                extractor,
                "_extract_search_page",
                new_callable=AsyncMock,
                return_value=extracted("text"),
            ) as mock_extract,
            patch.object(
                extractor,
                "_extract_job_ids",
                new_callable=AsyncMock,
                side_effect=lambda: next(id_pages),
            ),
            patch.object(
                extractor,
                "_get_total_search_pages",
                new_callable=AsyncMock,
                return_value=2,
            ) as mock_total_pages,
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.search_jobs("python", max_pages=10)

        # Should only visit 2 pages despite max_pages=10
        assert mock_extract.await_count == 2
        assert mock_total_pages.await_count == 1
        assert result["job_ids"] == ["100", "200"]

    async def test_zero_max_pages_fetches_nothing(self, mock_page):
        """max_pages=0 should fetch zero pages (validation at tool boundary)."""
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "_extract_search_page",
                new_callable=AsyncMock,
                return_value=extracted("text"),
            ) as mock_extract,
            patch.object(
                extractor,
                "_extract_job_ids",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch.object(
                extractor,
                "_get_total_search_pages",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.search_jobs("python", max_pages=0)

        assert result["job_ids"] == []
        assert mock_extract.await_count == 0

    async def test_single_page(self, mock_page):
        """max_pages=1 should only visit one page; filters appear in URL."""
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "_extract_search_page",
                new_callable=AsyncMock,
                return_value=extracted("Job posting text"),
            ) as mock_extract,
            patch.object(
                extractor,
                "_extract_job_ids",
                new_callable=AsyncMock,
                return_value=["42"],
            ),
            patch.object(
                extractor,
                "_get_total_search_pages",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.search_jobs(
                "python",
                "Remote",
                max_pages=1,
                date_posted="past_week",
                work_type="remote",
                easy_apply=True,
            )

        assert result["job_ids"] == ["42"]
        assert "keywords=python" in result["url"]
        assert "location=Remote" in result["url"]
        assert "f_TPR=r604800" in result["url"]
        assert "f_WT=2" in result["url"]
        assert "f_EA=true" in result["url"]
        assert mock_extract.await_count == 1

    async def test_page_texts_joined_with_separator(self, mock_page):
        """Multiple pages should join text with --- separator."""
        extractor = LinkedInExtractor(mock_page)
        text_pages = iter(["Page 1 content", "Page 2 content"])
        id_pages = iter([["100"], ["200"]])
        with (
            patch.object(
                extractor,
                "_extract_search_page",
                new_callable=AsyncMock,
                side_effect=lambda url, *args, **kwargs: extracted(next(text_pages)),
            ) as mock_extract,
            patch.object(
                extractor,
                "_extract_job_ids",
                new_callable=AsyncMock,
                side_effect=lambda: next(id_pages),
            ),
            patch.object(
                extractor,
                "_get_total_search_pages",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.search_jobs("python", max_pages=2)

        assert "\n---\n" in result["sections"]["search_results"]
        assert "Page 1 content" in result["sections"]["search_results"]
        assert "Page 2 content" in result["sections"]["search_results"]
        assert mock_extract.await_count == 2

    async def test_empty_results(self, mock_page):
        """Should handle empty results gracefully and skip ID extraction."""
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "_extract_search_page",
                new_callable=AsyncMock,
                return_value=extracted(""),
            ),
            patch.object(
                extractor,
                "_extract_job_ids",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_ids,
            patch.object(
                extractor,
                "_get_total_search_pages",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.search_jobs("nonexistent_xyz")

        assert result["job_ids"] == []
        assert result["sections"] == {}
        # Empty text should skip ID extraction to avoid stale DOM
        mock_ids.assert_not_awaited()

    async def test_no_ids_on_first_page_captures_text(self, mock_page):
        """Non-empty text with zero job IDs should be returned in sections."""
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "_extract_search_page",
                new_callable=AsyncMock,
                return_value=extracted("No matching jobs found"),
            ),
            patch.object(
                extractor,
                "_extract_job_ids",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch.object(
                extractor,
                "_get_total_search_pages",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.search_jobs("xyzzy123", max_pages=1)

        assert result["job_ids"] == []
        assert result["sections"]["search_results"] == "No matching jobs found"

    async def test_url_redirect_skips_id_extraction(self, mock_page):
        """Unexpected page URL should skip ID extraction but capture text."""
        extractor = LinkedInExtractor(mock_page)
        mock_page.url = "https://www.linkedin.com/uas/login"
        with (
            patch.object(
                extractor,
                "_extract_search_page",
                new_callable=AsyncMock,
                return_value=extracted(
                    "Login page content",
                    [{"kind": "person", "url": "/in/testuser/", "text": "Test User"}],
                ),
            ),
            patch.object(
                extractor,
                "_extract_job_ids",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_ids,
            patch.object(
                extractor,
                "_get_total_search_pages",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.search_jobs("python", max_pages=2)

        mock_ids.assert_not_awaited()
        assert result["job_ids"] == []
        assert result["sections"]["search_results"] == "Login page content"
        assert result["references"] == {
            "search_results": [
                {"kind": "person", "url": "/in/testuser/", "text": "Test User"}
            ]
        }

    async def test_rate_limited_skips_ids_and_text(self, mock_page):
        """Rate-limited pages should yield no IDs or text."""
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "_extract_search_page",
                new_callable=AsyncMock,
                return_value=extracted(_RATE_LIMITED_MSG),
            ),
            patch.object(
                extractor,
                "_extract_job_ids",
                new_callable=AsyncMock,
                return_value=["100"],
            ) as mock_ids,
            patch.object(
                extractor,
                "_get_total_search_pages",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.search_jobs("python", max_pages=1)

        assert result["job_ids"] == []
        assert result["sections"] == {}
        mock_ids.assert_not_awaited()


class TestGetSavedJobs:
    """Tests for get_saved_jobs with job ID extraction and pagination."""

    @pytest.fixture(autouse=True)
    def _set_saved_jobs_url(self, mock_page):
        mock_page.url = "https://www.linkedin.com/my-items/saved-jobs/"

    async def test_returns_job_ids(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "_extract_saved_jobs_page",
                new_callable=AsyncMock,
                return_value=extracted("Saved Job 1\nSaved Job 2"),
            ),
            patch.object(
                extractor,
                "_extract_job_ids",
                new_callable=AsyncMock,
                return_value=["111", "222"],
            ),
            patch.object(
                extractor,
                "_get_total_list_pages",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.get_saved_jobs(max_pages=1)

        assert result["job_ids"] == ["111", "222"]
        assert "saved_jobs" in result["sections"]
        assert result["url"] == "https://www.linkedin.com/my-items/saved-jobs/"

    async def test_returns_references(self, mock_page):
        """References are keyed by the section name, per the return contract."""
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "_extract_saved_jobs_page",
                new_callable=AsyncMock,
                return_value=extracted(
                    "Job 1",
                    [{"kind": "job", "url": "/jobs/view/111/", "text": "Job 1"}],
                ),
            ),
            patch.object(
                extractor,
                "_extract_job_ids",
                new_callable=AsyncMock,
                return_value=["111"],
            ),
            patch.object(
                extractor,
                "_get_total_list_pages",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.get_saved_jobs(max_pages=1)

        assert result["references"] == {
            "saved_jobs": [{"kind": "job", "url": "/jobs/view/111/", "text": "Job 1"}]
        }

    async def test_page_texts_joined_with_separator(self, mock_page):
        """Multi-page text is joined so the caller can tell pages apart."""
        extractor = LinkedInExtractor(mock_page)
        texts = iter([extracted("page one"), extracted("page two")])
        id_pages = iter([["100"], ["200"]])
        with (
            patch.object(
                extractor,
                "_extract_saved_jobs_page",
                new_callable=AsyncMock,
                side_effect=lambda *a, **kw: next(texts),
            ),
            patch.object(
                extractor,
                "_extract_job_ids",
                new_callable=AsyncMock,
                side_effect=lambda: next(id_pages),
            ),
            patch.object(
                extractor,
                "_get_total_list_pages",
                new_callable=AsyncMock,
                return_value=2,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.get_saved_jobs(max_pages=2)

        assert result["sections"]["saved_jobs"] == "page one\n---\npage two"

    async def test_pagination_uses_start_offset(self, mock_page):
        """The my-items list pages in 10s, not the 25 used by job search."""
        extractor = LinkedInExtractor(mock_page)
        id_pages = iter([["100", "200"], ["300"], ["400"]])
        urls_visited: list[str] = []

        async def mock_extract(url, *args, **kwargs):
            urls_visited.append(url)
            return extracted("page text")

        with (
            patch.object(
                extractor, "_extract_saved_jobs_page", side_effect=mock_extract
            ),
            patch.object(
                extractor,
                "_extract_job_ids",
                new_callable=AsyncMock,
                side_effect=lambda: next(id_pages),
            ),
            patch.object(
                extractor,
                "_get_total_list_pages",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.get_saved_jobs(max_pages=3)

        assert result["job_ids"] == ["100", "200", "300", "400"]
        assert urls_visited == [
            "https://www.linkedin.com/my-items/saved-jobs/",
            "https://www.linkedin.com/my-items/saved-jobs/?start=10",
            "https://www.linkedin.com/my-items/saved-jobs/?start=20",
        ]

    async def test_early_stop_no_new_ids(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        id_pages = iter([["100"], ["100"]])
        with (
            patch.object(
                extractor,
                "_extract_saved_jobs_page",
                new_callable=AsyncMock,
                return_value=extracted("text"),
            ) as mock_extract,
            patch.object(
                extractor,
                "_extract_job_ids",
                new_callable=AsyncMock,
                side_effect=lambda: next(id_pages),
            ),
            patch.object(
                extractor,
                "_get_total_list_pages",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.get_saved_jobs(max_pages=5)

        assert result["job_ids"] == ["100"]
        # Stops on the repeat page rather than exhausting max_pages
        assert mock_extract.await_count == 2

    async def test_stops_at_total_pages(self, mock_page):
        """The pager's page count caps pagination below max_pages."""
        extractor = LinkedInExtractor(mock_page)
        id_pages = iter([["100"], ["200"], ["300"]])
        with (
            patch.object(
                extractor,
                "_extract_saved_jobs_page",
                new_callable=AsyncMock,
                return_value=extracted("text"),
            ) as mock_extract,
            patch.object(
                extractor,
                "_extract_job_ids",
                new_callable=AsyncMock,
                side_effect=lambda: next(id_pages),
            ),
            patch.object(
                extractor,
                "_get_total_list_pages",
                new_callable=AsyncMock,
                return_value=2,
            ) as mock_total_pages,
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.get_saved_jobs(max_pages=10)

        # Both pages the pager reports, and no more.
        assert mock_extract.await_count == 2
        assert mock_total_pages.await_count == 1
        assert result["job_ids"] == ["100", "200"]

    async def test_rate_limited_page_keeps_earlier_pages(self, mock_page):
        """A rate-limited later page stops pagination without losing page 1.

        Matches the sibling behaviour of ``search_jobs``: the sentinel page
        contributes no text, and a soft rate limit is not reported as a
        section error (it is not a scraper bug).
        """
        extractor = LinkedInExtractor(mock_page)
        pages = iter([extracted("first page"), extracted(_RATE_LIMITED_MSG)])
        with (
            patch.object(
                extractor,
                "_extract_saved_jobs_page",
                new_callable=AsyncMock,
                side_effect=lambda *a, **kw: next(pages),
            ),
            patch.object(
                extractor,
                "_extract_job_ids",
                new_callable=AsyncMock,
                return_value=["100"],
            ),
            patch.object(
                extractor,
                "_get_total_list_pages",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.get_saved_jobs(max_pages=3)

        assert result["job_ids"] == ["100"]
        # The blocked page contributes nothing; page 1 survives intact.
        assert result["sections"]["saved_jobs"] == "first page"
        assert "section_errors" not in result

    async def test_url_redirect_skips_id_extraction(self, mock_page):
        """A redirect away from the list captures the landing page, no IDs.

        Mirrors ``search_jobs``: the text is kept deliberately so the caller
        can see what LinkedIn served instead (e.g. a login wall).
        """
        extractor = LinkedInExtractor(mock_page)
        mock_page.url = "https://www.linkedin.com/uas/login"
        with (
            patch.object(
                extractor,
                "_extract_saved_jobs_page",
                new_callable=AsyncMock,
                return_value=extracted("Login page content"),
            ),
            patch.object(
                extractor,
                "_extract_job_ids",
                new_callable=AsyncMock,
                return_value=["999"],
            ) as mock_ids,
            patch.object(
                extractor,
                "_get_total_list_pages",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.get_saved_jobs(max_pages=2)

        # Never mine IDs off a page that is not the saved-jobs list.
        mock_ids.assert_not_awaited()
        assert result["job_ids"] == []
        assert result["sections"]["saved_jobs"] == "Login page content"


class TestSearchPeople:
    async def test_search_people_omits_orphaned_references(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with patch.object(
            extractor,
            "extract_page",
            new_callable=AsyncMock,
            return_value=extracted(
                "",
                [
                    {
                        "kind": "person",
                        "url": "/in/testuser/",
                        "text": "Test User",
                    }
                ],
            ),
        ):
            result = await extractor.search_people("python")

        assert result["sections"] == {}
        assert "references" not in result

    async def test_search_people_network_filter_first_degree(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with patch.object(
            extractor,
            "extract_page",
            new_callable=AsyncMock,
            return_value=extracted("Jane Doe"),
        ):
            result = await extractor.search_people("engineer", network=["F"])

        assert "network=%5B%22F%22%5D" in result["url"]

    async def test_search_people_network_filter_multi_degree(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with patch.object(
            extractor,
            "extract_page",
            new_callable=AsyncMock,
            return_value=extracted("Jane Doe"),
        ):
            result = await extractor.search_people("engineer", network=["F", "S"])

        assert "network=%5B%22F%22%2C%22S%22%5D" in result["url"]

    async def test_search_people_current_company_filter(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with patch.object(
            extractor,
            "extract_page",
            new_callable=AsyncMock,
            return_value=extracted("Jane Doe"),
        ):
            result = await extractor.search_people("engineer", current_company="1115")

        assert "currentCompany=%5B%221115%22%5D" in result["url"]

    async def test_search_people_invalid_network_token_raises(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with pytest.raises(ValueError, match="Invalid network token"):
            await extractor.search_people("engineer", network=["X"])

    async def test_search_people_rejects_plain_company_name(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with pytest.raises(ValueError, match="must be a numeric"):
            await extractor.search_people("engineer", current_company="SAP")

    async def test_search_people_rejects_unicode_digit_company(self, mock_page):
        """LinkedIn URN ids are ASCII decimal; reject Unicode digits even
        though ``str.isdigit()`` would accept them."""
        extractor = LinkedInExtractor(mock_page)
        with pytest.raises(ValueError, match="must be a numeric"):
            await extractor.search_people("engineer", current_company="١١١٥")

    async def test_search_people_empty_current_company_is_noop(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with patch.object(
            extractor,
            "extract_page",
            new_callable=AsyncMock,
            return_value=extracted("Jane Doe"),
        ):
            result = await extractor.search_people("engineer", current_company="")

        assert "currentCompany" not in result["url"]

    async def test_search_people_combines_all_filters(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with patch.object(
            extractor,
            "extract_page",
            new_callable=AsyncMock,
            return_value=extracted("Jane Doe"),
        ):
            result = await extractor.search_people(
                "engineer",
                location="Seattle",
                network=["F"],
                current_company="1115",
            )

        assert "keywords=engineer" in result["url"]
        assert "location=Seattle" in result["url"]
        assert "network=%5B%22F%22%5D" in result["url"]
        assert "currentCompany=%5B%221115%22%5D" in result["url"]


class TestBuildContentSearchUrl:
    """Tests for _build_content_search_url URL construction."""

    def test_basic_keywords(self):
        url = LinkedInExtractor._build_content_search_url("Buscamos Unity")
        assert url == (
            "https://www.linkedin.com/search/results/content/"
            "?keywords=Buscamos+Unity&origin=FACETED_SEARCH"
        )

    def test_date_posted_past_week(self):
        url = LinkedInExtractor._build_content_search_url(
            "Buscamos Unity", date_posted="past-week"
        )
        assert "datePosted=%5B%22past-week%22%5D" in url

    def test_date_posted_alias_normalized(self):
        url = LinkedInExtractor._build_content_search_url(
            "python", date_posted="past_24_hours"
        )
        assert "datePosted=%5B%22past-24h%22%5D" in url

    def test_every_accepted_date_posted_reaches_linkedin_as_a_real_token(self):
        """LinkedIn ignores an unrecognized token instead of rejecting it, so
        an accepted value that never maps to one of its three would return
        unfiltered results while looking filtered."""
        for accepted, expected in _CONTENT_DATE_POSTED_MAP.items():
            url = LinkedInExtractor._build_content_search_url(
                "python", date_posted=accepted
            )
            assert expected in ("past-24h", "past-week", "past-month")
            assert f"%22{expected}%22" in url

    def test_no_date_posted_omits_facet(self):
        url = LinkedInExtractor._build_content_search_url("python")
        assert "datePosted" not in url

    def test_whitespace_date_posted_omits_facet(self):
        # Whitespace-only date_posted must be ignored, not appended as an
        # invalid facet token (regression guard).
        url = LinkedInExtractor._build_content_search_url("python", date_posted="   ")
        assert "datePosted" not in url


@pytest.mark.asyncio
class TestSearchPosts:
    async def test_returns_results_and_url(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with patch.object(
            extractor,
            "extract_page",
            new_callable=AsyncMock,
            return_value=extracted("We're hiring a Unity dev"),
        ) as mock_extract:
            result = await extractor.search_posts("Buscamos Unity")

        assert "/search/results/content/" in result["url"]
        assert "origin=FACETED_SEARCH" in result["url"]
        assert result["sections"]["search_results"] == "We're hiring a Unity dev"
        # max_pages default (3) -> 15 scrolls
        mock_extract.assert_awaited_once_with(
            ANY, section_name="search_results", max_scrolls=15
        )

    async def test_date_posted_in_url(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with patch.object(
            extractor,
            "extract_page",
            new_callable=AsyncMock,
            return_value=extracted("post"),
        ):
            result = await extractor.search_posts(
                "Buscamos Unity", date_posted="past-week"
            )

        assert "datePosted=%5B%22past-week%22%5D" in result["url"]

    async def test_max_pages_controls_scroll_depth(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with patch.object(
            extractor,
            "extract_page",
            new_callable=AsyncMock,
            return_value=extracted("post"),
        ) as mock_extract:
            await extractor.search_posts("python", max_pages=2)

        mock_extract.assert_awaited_once_with(
            ANY, section_name="search_results", max_scrolls=10
        )

    async def test_invalid_date_posted_raises(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with pytest.raises(ValueError, match="Invalid date_posted"):
            await extractor.search_posts("python", date_posted="last-year")

    async def test_empty_results_omit_optional_keys(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with patch.object(
            extractor,
            "extract_page",
            new_callable=AsyncMock,
            return_value=extracted(""),
        ):
            result = await extractor.search_posts("nothing matches this query")

        assert result["sections"] == {}
        assert "references" not in result
        assert "section_errors" not in result

    async def test_rate_limited_surfaces_section_error(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with patch.object(
            extractor,
            "extract_page",
            new_callable=AsyncMock,
            return_value=extracted(_RATE_LIMITED_MSG),
        ):
            result = await extractor.search_posts("python")

        assert result["sections"] == {}
        assert result["section_errors"]["search_results"]["error_type"] == "rate_limit"

    async def test_navigation_error_surfaces_section_error(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with patch.object(
            extractor,
            "extract_page",
            new_callable=AsyncMock,
            return_value=extracted(
                "", error={"error_type": "navigation_error", "error_message": "timeout"}
            ),
        ):
            result = await extractor.search_posts("python")

        assert result["sections"] == {}
        assert result["section_errors"]["search_results"] == {
            "error_type": "navigation_error",
            "error_message": "timeout",
        }


class TestStripLinkedInNoise:
    def test_strips_footer(self):
        text = "Bill Gates\nChair, Gates Foundation\n\nAbout\nAccessibility\nTalent Solutions\nCareers"
        assert strip_linkedin_noise(text) == "Bill Gates\nChair, Gates Foundation"

    def test_strips_footer_with_talent_solutions_variant(self):
        text = "Profile content here\n\nAbout\nTalent Solutions\nMore footer"
        assert strip_linkedin_noise(text) == "Profile content here"

    def test_strips_sidebar_recommendations(self):
        text = "Experience\nCo-chair\nGates Foundation\n\nMore profiles for you\nSundar Pichai\nCEO at Google"
        assert strip_linkedin_noise(text) == "Experience\nCo-chair\nGates Foundation"

    def test_strips_premium_upsell(self):
        text = "Education\nHarvard University\n\nExplore premium profiles\nRandom Person\nSoftware Engineer"
        assert strip_linkedin_noise(text) == "Education\nHarvard University"

    def test_picks_earliest_marker(self):
        text = "Content\n\nExplore premium profiles\nStuff\n\nMore profiles for you\nMore stuff\n\nAbout\nAccessibility"
        assert strip_linkedin_noise(text) == "Content"

    def test_no_noise_returns_unchanged(self):
        text = "Clean content with no LinkedIn chrome"
        assert strip_linkedin_noise(text) == "Clean content with no LinkedIn chrome"

    def test_empty_string(self):
        assert strip_linkedin_noise("") == ""

    def test_truncate_noise_preserves_media_controls_for_rate_limit_detection(self):
        text = "Play\nLoaded: 100.00%\nRemaining time 0:07\nShow captions"
        assert _truncate_linkedin_noise(text) == text
        assert strip_linkedin_noise(text) == ""

    def test_about_in_profile_content_not_stripped(self):
        """'About' followed by actual content (not 'Accessibility') should be preserved."""
        text = "About\nChair of the Gates Foundation.\n\nFeatured\nPost"
        assert (
            strip_linkedin_noise(text)
            == "About\nChair of the Gates Foundation.\n\nFeatured\nPost"
        )

    def test_real_footer_with_languages(self):
        text = (
            "Company info\n\n"
            "About\nAccessibility\nTalent Solutions\nCareers\n"
            "Select language\nEnglish (English)\nDeutsch (German)"
        )
        assert strip_linkedin_noise(text) == "Company info"

    def test_preserves_real_careers_content(self):
        text = "Careers\nWe're hiring globally.\nOpen roles in engineering and design."
        assert strip_linkedin_noise(text) == text

    def test_preserves_real_questions_content(self):
        text = "Questions?\nReach out to our recruiting team for details."
        assert strip_linkedin_noise(text) == text

    def test_strips_media_controls_lines(self):
        text = (
            "Feed post number 1\n"
            "Play\n"
            "Loaded: 100.00%\n"
            "Remaining time 0:07\n"
            "Playback speed\n"
            "Actual post content\n"
            "Show captions\n"
            "Close modal window"
        )
        assert strip_linkedin_noise(text) == "Feed post number 1\nActual post content"


class TestStripConversationChrome:
    THREAD = (
        "MAY 25\n"
        "Grace Hopper sent the following message at 5:27 PM\n"
        "Grace Hopper  5:27 PM\n"
        "\n"
        "Hello!"
    )
    PAGE = (
        "Messaging\n"
        "Search messages\n"
        "Compose a new message\n"
        "Inbox\n"
        "Attention screen reader users, messaging items continuously update.\n"
        "Ada Lovelace\n"
        "Jun 8\n"
        "Ada: Preview belonging to a different conversation\n"
        ". Press return to go to conversation details\n"
        "Open the options list in your conversation with Ada Lovelace and Grace Hopper\n"
        "Status is reachable\n"
        "Load more conversations\n"
        "Grace Hopper\n"
        "Status is online\n"
        "Open the options list in your conversation with Grace Hopper and Ada Lovelace\n"
        + THREAD
        + "\n"
        "Maximize compose field\n"
        "Attach an image to your conversation with Grace Hopper\n"
        "Open GIF Keyboard\n"
        "Send\n"
        "Open send options"
    )

    def test_strips_sidebar_and_composer(self):
        assert strip_conversation_chrome(self.PAGE) == self.THREAD

    def test_other_conversation_previews_removed(self):
        assert "different conversation" not in strip_conversation_chrome(self.PAGE)
        assert "Ada Lovelace" not in strip_conversation_chrome(self.PAGE)

    def test_missing_composer_strips_only_leading_chrome(self):
        text = (
            "Open the options list in your conversation with Grace Hopper and Ada Lovelace\n"
            + self.THREAD
        )
        assert strip_conversation_chrome(text) == self.THREAD

    def test_missing_thread_header_strips_only_composer(self):
        text = self.THREAD + "\nMaximize compose field\nOpen send options"
        assert strip_conversation_chrome(text) == self.THREAD

    def test_quoted_composer_string_in_message_survives(self):
        text = (
            "Open the options list in your conversation with Grace Hopper and Ada Lovelace\n"
            "Maximize compose field\n"
            "is the label I keep seeing\n"
            "Maximize compose field\n"
            "Open send options"
        )
        assert (
            strip_conversation_chrome(text)
            == "Maximize compose field\nis the label I keep seeing"
        )

    def test_quoted_companion_with_suffix_does_not_confirm_composer(self):
        text = "Hello!\nMaximize compose field\nOpen send options is what I clicked"
        assert strip_conversation_chrome(text) == text

    def test_quoted_attach_text_does_not_confirm_composer(self):
        text = (
            "Hello!\n"
            "Maximize compose field\n"
            "Attach an image to your conversation with Grace is the label I clicked"
        )
        assert strip_conversation_chrome(text) == text

    def test_distant_companion_text_does_not_confirm_composer(self):
        filler = "\n".join(f"message {n}" for n in range(10))
        text = (
            "Maximize compose field\n"
            + filler
            + "\nOpen send options is what I clicked"
        )
        assert strip_conversation_chrome(text) == text

    def test_quoted_composer_without_companions_does_not_truncate(self):
        text = (
            "Open the options list in your conversation with Grace Hopper and Ada Lovelace\n"
            "Hello!\n"
            "Maximize compose field\n"
            "is what the button says"
        )
        assert (
            strip_conversation_chrome(text)
            == "Hello!\nMaximize compose field\nis what the button says"
        )

    def test_quoted_thread_header_in_message_keeps_earlier_messages(self):
        text = (
            "Load more conversations\n"
            "Grace Hopper\n"
            "Open the options list in your conversation with Grace Hopper and Ada Lovelace\n"
            "Hello!\n"
            "Open the options list in your conversation with is a label I quoted\n"
            "Bye!\n"
            "Maximize compose field\n"
            "Open send options"
        )
        assert strip_conversation_chrome(text) == (
            "Hello!\n"
            "Open the options list in your conversation with is a label I quoted\n"
            "Bye!"
        )

    def test_sidebar_end_without_thread_header_still_strips_sidebar(self):
        text = (
            "Ada: Preview belonging to a different conversation\n"
            "Load more conversations\n" + self.THREAD
        )
        assert strip_conversation_chrome(text) == self.THREAD

    def test_unknown_locale_returns_unchanged(self):
        assert strip_conversation_chrome(self.PAGE, locale="de") == self.PAGE

    def test_no_markers_returns_stripped_text(self):
        assert strip_conversation_chrome("Hello!\nHi there!") == "Hello!\nHi there!"

    def test_empty_string(self):
        assert strip_conversation_chrome("") == ""


class TestActivityFeedExtraction:
    """Tests for activity page detection and wait behavior in _extract_page_once."""

    async def test_activity_page_waits_for_content_and_uses_slow_scroll(
        self, mock_page
    ):
        """Activity URLs should call wait_for_function and use slower scroll params."""
        mock_page.evaluate = AsyncMock(
            return_value={
                "source": "root",
                "text": "Post content " * 50,
                "references": [],
            }
        )
        mock_page.wait_for_function = AsyncMock()
        extractor = LinkedInExtractor(mock_page)
        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.scroll_to_bottom",
                new_callable=AsyncMock,
            ) as mock_scroll,
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            result = await extractor._extract_page_once(
                "https://www.linkedin.com/in/billgates/recent-activity/all/",
                section_name="posts",
            )

        mock_page.wait_for_function.assert_awaited_once()
        mock_scroll.assert_awaited_once()
        _, kwargs = mock_scroll.call_args
        assert kwargs["pause_time"] == 1.0
        assert kwargs["max_scrolls"] == 10
        assert len(result.text) > 200

    async def test_company_posts_page_waits_for_content_and_uses_slow_scroll(
        self, mock_page
    ):
        """Company posts URLs get the same lazy-load wait and scroll budget
        as person activity pages, even though they lack /recent-activity/."""
        mock_page.evaluate = AsyncMock(
            return_value={
                "source": "root",
                "text": "Post content " * 50,
                "references": [],
            }
        )
        mock_page.wait_for_function = AsyncMock()
        extractor = LinkedInExtractor(mock_page)
        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.scroll_to_bottom",
                new_callable=AsyncMock,
            ) as mock_scroll,
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            result = await extractor._extract_page_once(
                "https://www.linkedin.com/company/microsoft/posts/",
                section_name="posts",
            )

        mock_page.wait_for_function.assert_awaited_once()
        mock_scroll.assert_awaited_once()
        _, kwargs = mock_scroll.call_args
        assert kwargs["pause_time"] == 1.0
        assert kwargs["max_scrolls"] == 10
        assert len(result.text) > 200

    async def test_company_posts_page_with_query_string_still_waits(self, mock_page):
        """The lazy-load branch keys off the parsed path, so a company posts
        url carrying a query string is not mistaken for a static page."""
        mock_page.evaluate = AsyncMock(
            return_value={
                "source": "root",
                "text": "Post content " * 50,
                "references": [],
            }
        )
        mock_page.wait_for_function = AsyncMock()
        extractor = LinkedInExtractor(mock_page)
        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.scroll_to_bottom",
                new_callable=AsyncMock,
            ) as mock_scroll,
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            await extractor._extract_page_once(
                "https://www.linkedin.com/company/microsoft/posts/?viewAsMember=true",
                section_name="posts",
            )

        mock_page.wait_for_function.assert_awaited_once()
        _, kwargs = mock_scroll.call_args
        assert kwargs["max_scrolls"] == 10

    async def test_non_activity_non_details_page_skips_wait_and_uses_fast_scroll(
        self, mock_page
    ):
        """Plain profile URLs (not activity, search, or details) skip wait_for_function."""
        mock_page.evaluate = AsyncMock(
            return_value={"source": "root", "text": "Profile text", "references": []}
        )
        mock_page.wait_for_function = AsyncMock()
        extractor = LinkedInExtractor(mock_page)
        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.scroll_to_bottom",
                new_callable=AsyncMock,
            ) as mock_scroll,
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            await extractor._extract_page_once(
                "https://www.linkedin.com/in/billgates/",
                section_name="main_profile",
            )

        mock_page.wait_for_function.assert_not_awaited()
        mock_scroll.assert_awaited_once()
        _, kwargs = mock_scroll.call_args
        assert kwargs["pause_time"] == 0.5
        assert kwargs["max_scrolls"] == 5

    async def test_details_page_waits_for_panel_content(self, mock_page):
        """Detail pages (/details/experience/ etc.) call wait_for_function to wait for the panel."""
        mock_page.evaluate = AsyncMock(
            return_value={
                "source": "root",
                "text": "Experience\nSoftware Engineer",
                "references": [],
            }
        )
        mock_page.wait_for_function = AsyncMock()
        extractor = LinkedInExtractor(mock_page)
        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.scroll_to_bottom",
                new_callable=AsyncMock,
            ) as mock_scroll,
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            await extractor._extract_page_once(
                "https://www.linkedin.com/in/billgates/details/experience/",
                section_name="experience",
            )

        mock_page.wait_for_function.assert_awaited_once()
        mock_scroll.assert_awaited_once()
        _, kwargs = mock_scroll.call_args
        assert kwargs["pause_time"] == 0.5
        assert kwargs["max_scrolls"] == 5

    async def test_max_scrolls_override_passed_to_scroll_to_bottom(self, mock_page):
        """Custom max_scrolls on a detail page overrides the default of 5."""
        mock_page.evaluate = AsyncMock(
            return_value={
                "source": "root",
                "text": "Experience\nSoftware Engineer",
                "references": [],
            }
        )
        mock_page.wait_for_function = AsyncMock()
        extractor = LinkedInExtractor(mock_page)
        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.scroll_to_bottom",
                new_callable=AsyncMock,
            ) as mock_scroll,
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            await extractor._extract_page_once(
                "https://www.linkedin.com/in/billgates/details/certifications/",
                section_name="certifications",
                max_scrolls=20,
            )

        mock_scroll.assert_awaited_once()
        _, kwargs = mock_scroll.call_args
        assert kwargs["max_scrolls"] == 20

    async def test_default_scrolls_without_max_scrolls_override(self, mock_page):
        """Without max_scrolls, detail pages use the default of 5."""
        mock_page.evaluate = AsyncMock(
            return_value={
                "source": "root",
                "text": "Experience\nSoftware Engineer",
                "references": [],
            }
        )
        mock_page.wait_for_function = AsyncMock()
        extractor = LinkedInExtractor(mock_page)
        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.scroll_to_bottom",
                new_callable=AsyncMock,
            ) as mock_scroll,
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            await extractor._extract_page_once(
                "https://www.linkedin.com/in/billgates/details/certifications/",
                section_name="certifications",
            )

        mock_scroll.assert_awaited_once()
        _, kwargs = mock_scroll.call_args
        assert kwargs["max_scrolls"] == 5

    async def test_details_page_clicks_show_more_until_gone(self, mock_page):
        """Detail pages click 'Show more' in a loop until the button disappears."""
        mock_page.evaluate = AsyncMock(
            return_value={"source": "root", "text": "text", "references": []}
        )
        mock_page.wait_for_function = AsyncMock()

        show_more = MagicMock()
        # count() returns 1, 1, 0 across iterations — button disappears on 3rd check
        show_more.count = AsyncMock(side_effect=[1, 1, 0])
        show_more.is_visible = AsyncMock(return_value=True)
        show_more.scroll_into_view_if_needed = AsyncMock()
        show_more.click = AsyncMock()
        show_more.first = show_more
        show_more.filter = MagicMock(return_value=show_more)

        def locator_side_effect(selector):
            if selector == "main button":
                return show_more
            return MagicMock(count=AsyncMock(return_value=0))

        mock_page.locator = MagicMock(side_effect=locator_side_effect)
        extractor = LinkedInExtractor(mock_page)

        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.scroll_to_bottom",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            await extractor._extract_page_once(
                "https://www.linkedin.com/in/billgates/details/certifications/",
                section_name="certifications",
            )

        assert show_more.click.await_count == 2

    async def test_details_page_show_more_respects_max_scrolls_budget(self, mock_page):
        """When 'Show more' never disappears, loop exits after max_scrolls clicks."""
        mock_page.evaluate = AsyncMock(
            return_value={"source": "root", "text": "text", "references": []}
        )
        mock_page.wait_for_function = AsyncMock()

        show_more = MagicMock()
        show_more.count = AsyncMock(return_value=1)  # always present
        show_more.is_visible = AsyncMock(return_value=True)
        show_more.scroll_into_view_if_needed = AsyncMock()
        show_more.click = AsyncMock()
        show_more.first = show_more
        show_more.filter = MagicMock(return_value=show_more)

        def locator_side_effect(selector):
            if selector == "main button":
                return show_more
            return MagicMock(count=AsyncMock(return_value=0))

        mock_page.locator = MagicMock(side_effect=locator_side_effect)
        extractor = LinkedInExtractor(mock_page)

        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.scroll_to_bottom",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            await extractor._extract_page_once(
                "https://www.linkedin.com/in/billgates/details/experience/",
                section_name="experience",
                max_scrolls=3,
            )

        assert show_more.click.await_count == 3

    async def test_non_details_page_does_not_click_show_more(self, mock_page):
        """Non-details URLs (main profile, activity) skip the Show more loop."""
        mock_page.evaluate = AsyncMock(
            return_value={"source": "root", "text": "text", "references": []}
        )
        mock_page.wait_for_function = AsyncMock()

        show_more = MagicMock()
        show_more.count = AsyncMock(return_value=1)
        show_more.click = AsyncMock()
        show_more.first = show_more
        show_more.filter = MagicMock(return_value=show_more)

        def locator_side_effect(selector):
            if selector == "main button":
                return show_more
            return MagicMock(count=AsyncMock(return_value=0))

        mock_page.locator = MagicMock(side_effect=locator_side_effect)
        extractor = LinkedInExtractor(mock_page)

        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.scroll_to_bottom",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            await extractor._extract_page_once(
                "https://www.linkedin.com/in/billgates/",
                section_name="main_profile",
            )

        show_more.click.assert_not_awaited()

    async def test_activity_page_timeout_proceeds_gracefully(self, mock_page):
        """When activity feed content never loads, extraction proceeds with available text."""
        from patchright.async_api import TimeoutError as PlaywrightTimeoutError

        tab_headers = "All activity\nPosts\nComments\nVideos\nImages"
        mock_page.evaluate = AsyncMock(
            return_value={"source": "root", "text": tab_headers, "references": []}
        )
        mock_page.wait_for_function = AsyncMock(
            side_effect=PlaywrightTimeoutError("Timeout")
        )
        extractor = LinkedInExtractor(mock_page)
        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.scroll_to_bottom",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            result = await extractor._extract_page_once(
                "https://www.linkedin.com/in/billgates/recent-activity/all/",
                section_name="posts",
            )

        # Should return whatever text is available, not crash
        assert result.text == tab_headers


class TestCompanyPeopleExtraction:
    """Tests for /company/<slug>/people/ hydration wait in _extract_page_once."""

    async def test_waits_for_listing_with_5s_timeout(self, mock_page):
        """Company /people/ pages call wait_for_function so the employee
        listing has hydrated before scroll/extract. Empty/restricted listings
        are common, so the timeout is 5s rather than the 10s pattern shared
        with is_search/is_details."""
        mock_page.evaluate = AsyncMock(
            return_value={
                "source": "root",
                "text": "Anthropic\nFollowing\nHome\nAbout\nPeople",
                "references": [],
            }
        )
        mock_page.wait_for_function = AsyncMock()
        extractor = LinkedInExtractor(mock_page)
        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.scroll_to_bottom",
                new_callable=AsyncMock,
            ) as mock_scroll,
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            await extractor._extract_page_once(
                "https://www.linkedin.com/company/anthropicresearch/people/",
                section_name="employees",
            )

        mock_page.wait_for_function.assert_awaited_once()
        wait_predicate = mock_page.wait_for_function.call_args[0][0]
        wait_kwargs = mock_page.wait_for_function.call_args.kwargs
        assert "/in/" in wait_predicate
        assert "querySelectorAll" in wait_predicate
        assert wait_kwargs["timeout"] == 5000
        mock_scroll.assert_awaited_once()

    async def test_continues_extraction_on_wait_timeout(self, mock_page):
        """When the hydration wait times out (genuinely empty listing), the
        extractor swallows PlaywrightTimeoutError and still scrolls + extracts
        rather than propagating the error to the caller."""
        from patchright.async_api import TimeoutError as PlaywrightTimeoutError

        mock_page.evaluate = AsyncMock(
            return_value={
                "source": "root",
                "text": "Empty company page",
                "references": [],
            }
        )
        mock_page.wait_for_function = AsyncMock(
            side_effect=PlaywrightTimeoutError("Timeout")
        )
        extractor = LinkedInExtractor(mock_page)
        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.scroll_to_bottom",
                new_callable=AsyncMock,
            ) as mock_scroll,
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            result = await extractor._extract_page_once(
                "https://www.linkedin.com/company/anthropicresearch/people/",
                section_name="employees",
            )

        mock_scroll.assert_awaited_once()
        assert result.text  # non-empty placeholder text from the mock


class TestSearchResultsExtraction:
    """Tests for search results page detection and wait behavior in _extract_page_once."""

    async def test_search_results_page_waits_for_content(self, mock_page):
        """Search results URLs should call wait_for_function to wait for content."""
        mock_page.evaluate = AsyncMock(
            return_value={
                "source": "root",
                "text": "Search results for John Doe. " * 10,
                "references": [],
            }
        )
        mock_page.wait_for_function = AsyncMock()
        extractor = LinkedInExtractor(mock_page)
        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.scroll_to_bottom",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            result = await extractor._extract_page_once(
                "https://www.linkedin.com/search/results/people/?keywords=John+Doe",
                section_name="search_results",
            )

        mock_page.wait_for_function.assert_awaited_once()
        assert len(result.text) > 100

    async def test_non_search_page_does_not_wait_for_search_content(self, mock_page):
        """Non-search URLs should not trigger the search results wait."""
        mock_page.evaluate = AsyncMock(
            return_value={"source": "root", "text": "Profile text", "references": []}
        )
        mock_page.wait_for_function = AsyncMock()
        extractor = LinkedInExtractor(mock_page)
        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.scroll_to_bottom",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            await extractor._extract_page_once(
                "https://www.linkedin.com/in/billgates/",
                section_name="main_profile",
            )

        mock_page.wait_for_function.assert_not_awaited()

    async def test_search_results_timeout_proceeds_gracefully(self, mock_page):
        """When search results never load, extraction proceeds with available text."""
        from patchright.async_api import TimeoutError as PlaywrightTimeoutError

        placeholder = "Search results for John Doe. No results found"
        mock_page.evaluate = AsyncMock(
            return_value={"source": "root", "text": placeholder, "references": []}
        )
        mock_page.wait_for_function = AsyncMock(
            side_effect=PlaywrightTimeoutError("Timeout")
        )
        extractor = LinkedInExtractor(mock_page)
        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.scroll_to_bottom",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            result = await extractor._extract_page_once(
                "https://www.linkedin.com/search/results/people/?keywords=John+Doe",
                section_name="search_results",
            )

        assert result.text == placeholder


class TestScrapePersonCallbacks:
    """Test that scrape_person invokes callbacks at each stage."""

    async def test_scrape_person_calls_callbacks(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        cb = MagicMock(spec=ProgressCallback)
        cb.on_start = AsyncMock()
        cb.on_progress = AsyncMock()
        cb.on_complete = AsyncMock()
        cb.on_error = AsyncMock()

        with (
            patch.object(
                extractor,
                "extract_page",
                new_callable=AsyncMock,
                return_value=extracted("text"),
            ),
            patch.object(
                extractor,
                "_extract_overlay",
                new_callable=AsyncMock,
                return_value=extracted("overlay text"),
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            await extractor.scrape_person(
                "testuser", {"experience", "education"}, callbacks=cb
            )

        cb.on_start.assert_awaited_once()
        assert cb.on_start.call_args[0][0] == "person profile"

        # 3 sections: main_profile (always) + experience + education
        assert cb.on_progress.await_count == 3
        messages = [c.args[0] for c in cb.on_progress.call_args_list]
        assert messages == [
            "Scraped main_profile (1/3)",
            "Scraped experience (2/3)",
            "Scraped education (3/3)",
        ]
        # Last section should be at 95%
        assert cb.on_progress.call_args_list[-1].args[1] == 95

        cb.on_complete.assert_awaited_once()
        assert cb.on_complete.call_args[0][0] == "person profile"
        cb.on_error.assert_not_awaited()

    async def test_scrape_person_no_callbacks_by_default(self, mock_page):
        """Without callbacks, scrape_person works identically to before."""
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "extract_page",
                new_callable=AsyncMock,
                return_value=extracted("text"),
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.scrape_person("testuser", {"main_profile"})

        assert "main_profile" in result["sections"]

    async def test_scrape_person_calls_on_error(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        cb = MagicMock(spec=ProgressCallback)
        cb.on_start = AsyncMock()
        cb.on_progress = AsyncMock()
        cb.on_complete = AsyncMock()
        cb.on_error = AsyncMock()

        with (
            patch.object(
                extractor,
                "extract_page",
                new_callable=AsyncMock,
                side_effect=LinkedInScraperException("boom"),
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            with pytest.raises(LinkedInScraperException):
                await extractor.scrape_person(
                    "testuser", {"main_profile"}, callbacks=cb
                )

        cb.on_start.assert_awaited_once()
        cb.on_error.assert_awaited_once()
        error_arg = cb.on_error.call_args[0][0]
        assert isinstance(error_arg, LinkedInScraperException)
        assert "boom" in str(error_arg)
        cb.on_complete.assert_not_awaited()


class TestMainProfileAlreadyLoaded:
    """Reuse path for scrape_person when get_my_profile already loaded the page."""

    async def test_get_my_profile_passes_already_loaded_flag(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        mock_page.url = "https://www.linkedin.com/in/realuser/"
        with (
            patch.object(extractor, "_navigate_to_page", new_callable=AsyncMock) as nav,
            patch.object(
                extractor,
                "scrape_person",
                new_callable=AsyncMock,
                return_value={"url": "...", "sections": {}},
            ) as scrape,
        ):
            await extractor.get_my_profile(sections={"main_profile"})

        nav.assert_awaited_once_with("https://www.linkedin.com/in/me/")
        assert scrape.await_count == 1
        assert scrape.call_args.kwargs["main_profile_already_loaded"] is True

    async def test_scrape_person_already_loaded_skips_navigation(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        mock_page.url = "https://www.linkedin.com/in/foo/"
        with (
            patch.object(
                extractor,
                "_extract_loaded_section",
                new_callable=AsyncMock,
                return_value=extracted("reused"),
            ) as loaded,
            patch.object(
                extractor, "extract_page", new_callable=AsyncMock
            ) as extract_page,
            patch.object(extractor, "_navigate_to_page", new_callable=AsyncMock) as nav,
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            await extractor.scrape_person(
                "foo", {"main_profile"}, main_profile_already_loaded=True
            )

        loaded.assert_awaited_once()
        extract_page.assert_not_awaited()
        nav.assert_not_awaited()

    async def test_scrape_person_already_loaded_url_mismatch_falls_back(
        self, mock_page
    ):
        extractor = LinkedInExtractor(mock_page)
        mock_page.url = "https://www.linkedin.com/feed/"
        with (
            patch.object(
                extractor,
                "extract_page",
                new_callable=AsyncMock,
                return_value=extracted("fallback"),
            ) as extract_page,
            patch.object(
                extractor,
                "_extract_loaded_section",
                new_callable=AsyncMock,
            ) as loaded,
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            await extractor.scrape_person(
                "foo", {"main_profile"}, main_profile_already_loaded=True
            )

        extract_page.assert_awaited_once()
        loaded.assert_not_awaited()

    async def test_scrape_person_already_loaded_rate_limit_falls_back(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        mock_page.url = "https://www.linkedin.com/in/foo/"

        from linkedin_mcp_server.scraping.extractor import _RATE_LIMITED_MSG

        # main_profile success path now stores a structured dict (see
        # scraping/main_profile.py); mock the structured extractor with a
        # sentinel so this test remains focused on the rate-limit fallback.
        sentinel_profile = {"name": "retry succeeded"}
        with (
            patch.object(
                extractor,
                "_extract_loaded_section",
                new_callable=AsyncMock,
                return_value=extracted(_RATE_LIMITED_MSG),
            ) as loaded,
            patch.object(
                extractor,
                "extract_page",
                new_callable=AsyncMock,
                return_value=extracted("retry succeeded"),
            ) as extract_page,
            patch(
                "linkedin_mcp_server.scraping.extractor.extract_main_profile",
                new_callable=AsyncMock,
                return_value=sentinel_profile,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.scrape_person(
                "foo", {"main_profile"}, main_profile_already_loaded=True
            )

        loaded.assert_awaited_once()
        extract_page.assert_awaited_once()
        assert result["sections"]["main_profile"] == sentinel_profile


class TestScrapeCompanyCallbacks:
    """Test that scrape_company invokes callbacks at each stage."""

    async def test_scrape_company_calls_callbacks(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        cb = MagicMock(spec=ProgressCallback)
        cb.on_start = AsyncMock()
        cb.on_progress = AsyncMock()
        cb.on_complete = AsyncMock()
        cb.on_error = AsyncMock()

        with (
            patch.object(
                extractor,
                "extract_page",
                new_callable=AsyncMock,
                return_value=extracted("text"),
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            await extractor.scrape_company(
                "testcorp", {"about", "posts", "jobs"}, callbacks=cb
            )

        cb.on_start.assert_awaited_once()
        assert cb.on_start.call_args[0][0] == "company profile"

        # 3 sections: about + posts + jobs
        assert cb.on_progress.await_count == 3
        messages = [c.args[0] for c in cb.on_progress.call_args_list]
        assert messages == [
            "Scraped about (1/3)",
            "Scraped posts (2/3)",
            "Scraped jobs (3/3)",
        ]
        assert cb.on_progress.call_args_list[-1].args[1] == 95

        cb.on_complete.assert_awaited_once()
        assert cb.on_complete.call_args[0][0] == "company profile"
        cb.on_error.assert_not_awaited()


class TestGetSidebarProfiles:
    async def test_returns_sidebar_profiles_from_all_sections(self, mock_page):
        """Happy path: extracts profiles from all sections, merges Show all results."""
        sidebar_js_result = {
            "sections": {
                "more_profiles_for_you": ["/in/alice/", "/in/bob/"],
                "explore_premium_profiles": ["/in/carol/"],
                "people_you_may_know": ["/in/dave/"],
            },
            "showAllUrls": {
                "more_profiles_for_you": "https://www.linkedin.com/search/results/people/?keywords=test",
            },
        }
        show_all_js_result = ["/in/alice/", "/in/eve/", "/in/frank/"]

        mock_page.evaluate = AsyncMock(
            side_effect=[sidebar_js_result, show_all_js_result]
        )
        mock_page.url = "https://www.linkedin.com/in/testuser/"

        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(extractor, "_navigate_to_page", new_callable=AsyncMock),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.get_sidebar_profiles("testuser")

        assert result["url"] == "https://www.linkedin.com/in/testuser/"
        mpfy = result["sidebar_profiles"]["more_profiles_for_you"]
        # sidebar links first, then show_all expansion, deduped
        assert mpfy == ["/in/alice/", "/in/bob/", "/in/eve/", "/in/frank/"]
        assert result["sidebar_profiles"]["explore_premium_profiles"] == ["/in/carol/"]
        assert result["sidebar_profiles"]["people_you_may_know"] == ["/in/dave/"]

    async def test_skips_show_all_when_url_contains_premium(self, mock_page):
        """Show all URL containing /premium is skipped without navigation."""
        sidebar_js_result = {
            "sections": {"explore_premium_profiles": ["/in/carol/"]},
            "showAllUrls": {
                "explore_premium_profiles": "https://www.linkedin.com/premium/products/"
            },
        }
        mock_page.evaluate = AsyncMock(return_value=sidebar_js_result)
        mock_page.url = "https://www.linkedin.com/in/testuser/"

        extractor = LinkedInExtractor(mock_page)
        navigate_mock = AsyncMock()
        with (
            patch.object(extractor, "_navigate_to_page", navigate_mock),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            result = await extractor.get_sidebar_profiles("testuser")

        navigate_mock.assert_awaited_once()  # only the initial profile navigation
        mock_page.evaluate.assert_awaited_once()  # no show_all JS call
        assert result["sidebar_profiles"]["explore_premium_profiles"] == ["/in/carol/"]

    async def test_skips_show_all_when_page_redirects_to_premium(self, mock_page):
        """If navigating to Show all lands on a /premium URL, skip that section."""
        sidebar_js_result = {
            "sections": {"more_profiles_for_you": ["/in/alice/"]},
            "showAllUrls": {
                "more_profiles_for_you": "https://www.linkedin.com/search/results/people/?keywords=test"
            },
        }
        mock_page.evaluate = AsyncMock(return_value=sidebar_js_result)
        mock_page.url = "https://www.linkedin.com/in/testuser/"

        navigate_call_count = 0

        async def fake_navigate(url: str) -> None:
            nonlocal navigate_call_count
            navigate_call_count += 1
            if navigate_call_count >= 2:
                mock_page.url = "https://www.linkedin.com/premium/grow-your-network/"

        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(extractor, "_navigate_to_page", side_effect=fake_navigate),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.get_sidebar_profiles("testuser")

        mock_page.evaluate.assert_awaited_once()  # sidebar JS only, no show_all expansion
        assert result["sidebar_profiles"]["more_profiles_for_you"] == ["/in/alice/"]

    async def test_returns_empty_sidebar_profiles_when_no_sections_found(
        self, mock_page
    ):
        """No matching sidebar headings -> empty sidebar_profiles dict."""
        mock_page.evaluate = AsyncMock(return_value={"sections": {}, "showAllUrls": {}})
        mock_page.url = "https://www.linkedin.com/in/testuser/"

        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(extractor, "_navigate_to_page", new_callable=AsyncMock),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            result = await extractor.get_sidebar_profiles("testuser")

        assert result == {
            "url": "https://www.linkedin.com/in/testuser/",
            "sidebar_profiles": {},
        }


class TestExtractProfileUrn:
    async def test_returns_urn_from_compose_href(self, mock_page):
        """Extracts the recipient URN from the messaging compose link."""
        mock_page.evaluate = AsyncMock(
            return_value="/messaging/compose/?recipient=ACoAAB1IelEBLEkqTkNbZ-a1D8mq5R-6C1ihSEk&lipi=urn..."
        )

        extractor = LinkedInExtractor(mock_page)
        result = await extractor._extract_profile_urn()

        assert result == "ACoAAB1IelEBLEkqTkNbZ-a1D8mq5R-6C1ihSEk"

    async def test_returns_none_when_no_compose_button(self, mock_page):
        """Returns None when no messaging compose link is found."""
        mock_page.evaluate = AsyncMock(return_value=None)

        extractor = LinkedInExtractor(mock_page)
        result = await extractor._extract_profile_urn()

        assert result is None

    async def test_returns_none_when_no_recipient_param(self, mock_page):
        """Returns None when the compose href has no recipient query param."""
        mock_page.evaluate = AsyncMock(
            return_value="/messaging/compose/?someOtherParam=value"
        )

        extractor = LinkedInExtractor(mock_page)
        result = await extractor._extract_profile_urn()

        assert result is None


class TestScrapePersonProfileUrn:
    async def test_includes_profile_urn_in_result_when_found(self, mock_page):
        """scrape_person includes profile_urn in result when _extract_profile_urn returns a value."""
        urn = "ACoAAB1IelEBLEkqTkNbZ-a1D8mq5R-6C1ihSEk"
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "extract_page",
                new_callable=AsyncMock,
                return_value=extracted("profile text"),
            ),
            patch.object(
                extractor,
                "_extract_profile_urn",
                new_callable=AsyncMock,
                return_value=urn,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.scrape_person("testuser", {"main_profile"})

        assert result["profile_urn"] == urn

    async def test_omits_profile_urn_when_not_found(self, mock_page):
        """scrape_person omits profile_urn key when _extract_profile_urn returns None."""
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "extract_page",
                new_callable=AsyncMock,
                return_value=extracted("profile text"),
            ),
            patch.object(
                extractor,
                "_extract_profile_urn",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.scrape_person("testuser", {"main_profile"})

        assert "profile_urn" not in result


class TestGetInbox:
    async def test_returns_inbox_section(self, mock_page):
        """get_inbox returns sections with inbox key."""
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "_navigate_to_page",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
            ),
            patch.object(
                extractor,
                "_wait_for_main_text",
                new_callable=AsyncMock,
            ),
            patch.object(
                extractor,
                "_scroll_main_scrollable_region",
                new_callable=AsyncMock,
            ),
            patch.object(
                extractor,
                "_extract_root_content",
                new_callable=AsyncMock,
                return_value={
                    "text": "Conversation A\nConversation B",
                    "references": [],
                },
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.strip_linkedin_noise",
                return_value="Conversation A\nConversation B",
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.build_references",
                return_value=[],
            ),
            patch.object(
                extractor,
                "_extract_conversation_thread_refs",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = await extractor.get_inbox(limit=10)

        assert "sections" in result
        assert "inbox" in result["sections"]
        assert "Conversation A" in result["sections"]["inbox"]

    async def test_empty_inbox(self, mock_page):
        """get_inbox returns empty sections when page has no content."""
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(extractor, "_navigate_to_page", new_callable=AsyncMock),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
            ),
            patch.object(extractor, "_wait_for_main_text", new_callable=AsyncMock),
            patch.object(
                extractor, "_scroll_main_scrollable_region", new_callable=AsyncMock
            ),
            patch.object(
                extractor,
                "_extract_root_content",
                new_callable=AsyncMock,
                return_value={"text": "", "references": []},
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.strip_linkedin_noise",
                return_value="",
            ),
            patch.object(
                extractor,
                "_extract_conversation_thread_refs",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = await extractor.get_inbox(limit=5)

        assert result["sections"] == {}

    async def test_includes_conversation_thread_refs(self, mock_page):
        """get_inbox includes conversation thread references from click extraction."""
        extractor = LinkedInExtractor(mock_page)
        thread_refs = [
            {
                "kind": "conversation",
                "url": "/messaging/thread/2-abc123/",
                "text": "Tony Chan",
                "context": "inbox",
            },
            {
                "kind": "conversation",
                "url": "/messaging/thread/2-def456/",
                "text": "Paul Jasper",
                "context": "inbox",
            },
        ]
        with (
            patch.object(extractor, "_navigate_to_page", new_callable=AsyncMock),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
            ),
            patch.object(extractor, "_wait_for_main_text", new_callable=AsyncMock),
            patch.object(
                extractor, "_scroll_main_scrollable_region", new_callable=AsyncMock
            ),
            patch.object(
                extractor,
                "_extract_root_content",
                new_callable=AsyncMock,
                return_value={
                    "text": "Tony Chan\nPaul Jasper",
                    "references": [],
                },
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.strip_linkedin_noise",
                return_value="Tony Chan\nPaul Jasper",
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.build_references",
                return_value=[],
            ),
            patch.object(
                extractor,
                "_extract_conversation_thread_refs",
                new_callable=AsyncMock,
                return_value=thread_refs,
            ),
        ):
            result = await extractor.get_inbox(limit=10)

        assert "references" in result
        refs = result["references"]["inbox"]
        assert len(refs) == 2
        assert refs[0]["kind"] == "conversation"
        assert refs[0]["url"] == "/messaging/thread/2-abc123/"
        assert refs[0]["text"] == "Tony Chan"

    async def test_merges_inbox_tabs_and_dedupes_thread_refs(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        mock_page.wait_for_selector = AsyncMock()
        tabs = MagicMock()

        async def count_after_hydration():
            mock_page.wait_for_selector.assert_awaited_once()
            return 2

        tabs.count = AsyncMock(side_effect=count_after_hydration)
        focused_tab = MagicMock()
        focused_tab.get_attribute = AsyncMock(side_effect=["true", "false"])
        focused_tab.click = AsyncMock()
        other_tab = MagicMock()
        other_tab.get_attribute = AsyncMock(side_effect=["false", "false"])
        other_tab.click = AsyncMock()
        tabs.nth.side_effect = lambda index: [focused_tab, other_tab][index]
        mock_page.locator.return_value = tabs

        focused_ref = {
            "kind": "conversation",
            "url": "/messaging/thread/2-focused/",
            "context": "inbox",
        }
        other_ref = {
            "kind": "conversation",
            "url": "/messaging/thread/2-other/",
            "context": "inbox",
        }

        with (
            patch.object(extractor, "_navigate_to_page", new_callable=AsyncMock),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
            ),
            patch.object(extractor, "_wait_for_main_text", new_callable=AsyncMock),
            patch.object(
                extractor,
                "_scroll_main_scrollable_region",
                new_callable=AsyncMock,
            ) as scroll_mock,
            patch.object(
                extractor,
                "_extract_root_content",
                new_callable=AsyncMock,
                side_effect=[
                    {"text": "Focused conversation", "references": []},
                    {"text": "Other conversation", "references": []},
                ],
            ) as root_mock,
            patch(
                "linkedin_mcp_server.scraping.extractor.strip_linkedin_noise",
                side_effect=lambda text: text,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.build_references",
                return_value=[],
            ),
            patch.object(
                extractor,
                "_extract_conversation_thread_refs",
                new_callable=AsyncMock,
            ) as refs_mock,
        ):
            ref_batches = iter([[focused_ref], [focused_ref, other_ref]])

            async def refs_after_text(*_args, **_kwargs):
                assert root_mock.await_count == 2
                return next(ref_batches)

            refs_mock.side_effect = refs_after_text
            result = await extractor.get_inbox(limit=10)

        mock_page.wait_for_selector.assert_awaited_once_with(
            'main [role="tablist"] [role="tab"], main li div[class*="listitem__link"]',
            state="attached",
            timeout=10000,
        )
        assert result["sections"]["inbox"] == (
            "Focused conversation\n\nOther conversation"
        )
        assert [ref["url"] for ref in result["references"]["inbox"]] == [
            "/messaging/thread/2-focused/",
            "/messaging/thread/2-other/",
        ]
        focused_tab.click.assert_awaited_once()
        assert other_tab.click.await_count == 2
        assert scroll_mock.await_count == 4


class TestInvitationManagement:
    def test_invitation_card_script_uses_visual_card_order(self):
        assert "for (const button of actionControls(root))" in _INVITATION_CARDS_JS
        assert "getBoundingClientRect()" in _INVITATION_CARDS_JS
        assert "cards.sort" in _INVITATION_CARDS_JS
        assert 'a[href*="/school/"]' in _INVITATION_CARDS_JS
        assert "if (kind === 'sent')" in _INVITATION_CARDS_JS
        assert "recipient:" in _INVITATION_CARDS_JS
        assert "sent|envoyé|envoyée" in _INVITATION_CARDS_JS
        assert "a[aria-label][href]" in _INVITATION_CARDS_JS
        assert "profile\\s+(?:photo|picture)" in _INVITATION_CARDS_JS
        assert "return senderNameFromProfileLink(profileLink);" in _INVITATION_CARDS_JS
        assert "let candidate = card;" in _INVITATION_CARDS_JS
        assert "if (actions.length > 4) break;" in _INVITATION_CARDS_JS
        assert "messageLink?.path" in _INVITATION_CARDS_JS
        assert "^\\/messaging\\/compose\\/" in _INVITATION_CARDS_JS

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("", 0),
            ("Hugo Attal mutual connection", 1),
            ("3 mutual connections", 3),
            ("Hugo Attal and 2 other mutual connections", 3),
            ("Hugo Attal et 2 relations en commun", 3),
            ("3 relations en commun", 3),
        ],
    )
    def test_connection_request_mutual_count_rules(self, text, expected):
        invitation = _normalize_structured_invitation(
            {
                "type": "connection_request",
                "sender": {"name": "Ayoub Chalabi", "url": "/in/ayoub-chalabi/"},
                "text": text,
            }
        )

        assert invitation is not None
        assert invitation["sender"]["mutual_connections"] == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("1 hour ago", "1h"),
            ("23 minutes ago", "23min"),
            ("Il y a 18 heures", "18h"),
            ("5 days ago", "5d"),
            ("Il y a 5 jours", "5d"),
            ("1 month ago", "1mo"),
            ("Il y a 1 mois", "1mo"),
            ("1m", "1mo"),
        ],
    )
    def test_invitation_age_rules(self, text, expected):
        invitation = _normalize_structured_invitation(
            {
                "type": "connection_request",
                "sender": {"name": "Ayoub Chalabi", "url": "/in/ayoub-chalabi/"},
                "text": text,
            }
        )

        assert invitation is not None
        assert invitation["invitation_age"] == expected

    async def test_get_pending_invitations_received(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        invitations = [
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
            },
            {
                "type": "connection_request",
                "invitation_age": "18h",
                "sender": {
                    "name": "Ayoub Chalabi",
                    "url": "/in/ayoub-chalabi/",
                    "headline": "Co-founder & CTO @ Learnrithm AI (SCV X26)",
                    "mutual_connections": 3,
                },
                "note": None,
                "target": None,
                "message_url": "/messaging/compose/?recipient=ayoub-chalabi",
            },
        ]
        with (
            patch.object(
                extractor,
                "_navigate_to_page",
                new_callable=AsyncMock,
            ) as mock_nav,
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
            ),
            patch.object(extractor, "_wait_for_main_text", new_callable=AsyncMock),
            patch.object(
                extractor, "_scroll_invitation_manager_down", new_callable=AsyncMock
            ),
            patch.object(
                extractor, "_expand_invitation_note_toggles", new_callable=AsyncMock
            ),
            patch.object(
                extractor,
                "_extract_invitation_cards",
                new_callable=AsyncMock,
                return_value=invitations,
            ) as mock_cards,
        ):
            result = await extractor.get_pending_invitations(limit=2)

        mock_nav.assert_awaited_once_with(
            "https://www.linkedin.com/mynetwork/invitation-manager/received/"
        )
        mock_cards.assert_awaited_once_with(kind="received", limit=2)
        assert list(result) == ["url", "invitations"]
        assert result["invitations"] == invitations

    async def test_get_pending_invitations_collects_before_scrolling(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        first_rows = [
            {
                "type": "connection_request",
                "invitation_age": "1w",
                "recipient": {
                    "name": "Laurent SORBIER",
                    "url": "/in/laurent-sorbier/",
                    "headline": "chargé d’affaires chez belectric",
                },
            },
            {
                "type": "connection_request",
                "invitation_age": "1w",
                "recipient": {
                    "name": "Hugues Jouffroy",
                    "url": "/in/hugues-jouffroy/",
                    "headline": "Directeur Général Délégué aux Opérations",
                },
            },
        ]
        second_rows = [
            first_rows[1],
            {
                "type": "connection_request",
                "invitation_age": "1w",
                "recipient": {
                    "name": "Rémi SACHOT",
                    "url": "/in/remisachotenr/",
                    "headline": "Directeur Exploitation chez TSE Energy",
                },
            },
        ]
        with (
            patch.object(extractor, "_navigate_to_page", new_callable=AsyncMock),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
            ),
            patch.object(extractor, "_wait_for_main_text", new_callable=AsyncMock),
            patch.object(
                extractor,
                "_scroll_invitation_manager_down",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_scroll,
            patch.object(
                extractor, "_expand_invitation_note_toggles", new_callable=AsyncMock
            ),
            patch.object(
                extractor,
                "_extract_invitation_cards",
                new_callable=AsyncMock,
                side_effect=[first_rows, second_rows],
            ),
        ):
            result = await extractor.get_pending_invitations(limit=6, kind="sent")

        assert result["invitations"] == [
            first_rows[0],
            first_rows[1],
            second_rows[1],
        ]
        mock_scroll.assert_awaited_once()

    async def test_get_pending_invitations_sent_url(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(
                extractor,
                "_navigate_to_page",
                new_callable=AsyncMock,
            ) as mock_nav,
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
            ),
            patch.object(extractor, "_wait_for_main_text", new_callable=AsyncMock),
            patch.object(
                extractor, "_scroll_invitation_manager_down", new_callable=AsyncMock
            ),
            patch.object(
                extractor, "_expand_invitation_note_toggles", new_callable=AsyncMock
            ),
            patch.object(
                extractor,
                "_extract_invitation_cards",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = await extractor.get_pending_invitations(limit=5, kind="sent")

        mock_nav.assert_awaited_once_with(
            "https://www.linkedin.com/mynetwork/invitation-manager/sent/"
        )
        assert result == {
            "url": "https://www.linkedin.com/mynetwork/invitation-manager/sent/",
            "invitations": [],
        }

    async def test_extract_received_invitation_cards_returns_structured_payload(
        self, mock_page
    ):
        extractor = LinkedInExtractor(mock_page)
        mock_page.evaluate = AsyncMock(
            return_value=[
                {
                    "type": "page_follow",
                    "invitation_age": "1 hour ago",
                    "sender": {
                        "name": "Juan Manuel M. Pérez",
                        "url": "/in/juanmanuelperez/",
                        "headline": "Ignored headline",
                        "mutual_connections": 4,
                    },
                    "note": "Ignored note",
                    "target": {
                        "page": {
                            "name": "Magical Potion Consulting",
                            "url": "/company/magical-potion-consulting/",
                        }
                    },
                    "message_url": "/messaging/compose/?recipient=alice",
                },
                {
                    "type": "page_follow",
                    "invitation_age": "1 hour ago",
                    "sender": {
                        "name": "Juan Manuel M. Pérez",
                        "url": "/in/juanmanuelperez/",
                    },
                    "target": {
                        "page": {
                            "name": "Magical Potion Consulting",
                            "url": "/company/magical-potion-consulting/",
                        }
                    },
                },
                {
                    "type": "connection_request",
                    "invitation_age": None,
                    "sender": {
                        "name": "Ayoub Chalabi",
                        "url": "/in/ayoub-chalabi/",
                        "headline": "Co-founder & CTO @ Learnrithm AI (SCV X26)",
                    },
                    "text": "Ayoub Chalabi follows you and is inviting you to connect Co-founder & CTO @ Learnrithm AI (SCV X26) Hugo Attal and 2 other mutual connections 18 hours ago",
                    "target": {"page": {"name": "Ignored", "url": "/company/x/"}},
                    "message_url": "/messaging/compose/?recipient=ACoAAAqI0hgB6hz3TEqrc9e4jwzIth2jHkAWxjk&invitation=urn%3Ali%3Afsd_invitation%3A7464720462661468161&contextEntityUrn=urn%3Ali%3Afsd_invitation%3A7464720462661468161&recipients=List%28urn%3Ali%3Afsd_profile%3AACoAAAqI0hgB6hz3TEqrc9e4jwzIth2jHkAWxjk%29",
                },
                {
                    "type": "newsletter_subscription",
                    "text": "The Example Brief 1 month ago",
                    "sender": {
                        "name": "CentraleSupélec",
                        "url": "/school/centralesupelec/",
                        "mutual_connections": None,
                    },
                    "target": {
                        "newsletter": {
                            "title": "The Example Brief",
                            "url": "/newsletters/the-example-brief-123/",
                        }
                    },
                },
                {"type": "connection_request", "sender": {}},
                "bad row",
            ]
        )

        cards = await extractor._extract_invitation_cards(kind="received", limit=10)

        evaluate_args = mock_page.evaluate.await_args
        assert evaluate_args is not None
        assert evaluate_args.args[1] == {"kind": "received", "limit": 20}
        assert cards == [
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
            },
            {
                "type": "connection_request",
                "invitation_age": "18h",
                "sender": {
                    "name": "Ayoub Chalabi",
                    "url": "/in/ayoub-chalabi/",
                    "headline": "Co-founder & CTO @ Learnrithm AI (SCV X26)",
                    "mutual_connections": 3,
                },
                "note": None,
                "target": None,
                "message_url": "/messaging/compose/?recipient=ACoAAAqI0hgB6hz3TEqrc9e4jwzIth2jHkAWxjk&invitation=urn%3Ali%3Afsd_invitation%3A7464720462661468161&contextEntityUrn=urn%3Ali%3Afsd_invitation%3A7464720462661468161&recipients=List%28urn%3Ali%3Afsd_profile%3AACoAAAqI0hgB6hz3TEqrc9e4jwzIth2jHkAWxjk%29",
            },
            {
                "type": "newsletter_subscription",
                "invitation_age": "1mo",
                "sender": {
                    "name": "CentraleSupélec",
                    "url": "/school/centralesupelec/",
                    "headline": None,
                    "mutual_connections": None,
                },
                "note": None,
                "target": {
                    "page": None,
                    "newsletter": {
                        "title": "The Example Brief",
                        "url": "/newsletters/the-example-brief-123/",
                    },
                },
                "message_url": None,
            },
        ]

    async def test_extract_sent_invitation_cards_returns_recipient_payload(
        self, mock_page
    ):
        extractor = LinkedInExtractor(mock_page)
        mock_page.evaluate = AsyncMock(
            return_value=[
                {
                    "type": "connection_request",
                    "invitation_age": "Sent 1 week ago",
                    "recipient": {
                        "name": "Laurent SORBIER",
                        "url": "/in/laurent-sorbier/",
                        "headline": "chargé d’affaires chez belectric",
                    },
                    "sender": {
                        "name": "Ignored sender",
                        "url": "/in/ignored/",
                        "headline": "Ignored headline",
                        "mutual_connections": 7,
                    },
                    "note": "Ignored note",
                    "target": {"page": {"name": "Ignored", "url": "/company/x/"}},
                    "message_url": "/messaging/compose/?recipient=ignored",
                },
                {
                    "type": "connection_request",
                    "invitation_age": "Sent 1 week ago",
                    "recipient": {
                        "name": "Laurent SORBIER",
                        "url": "/in/laurent-sorbier/",
                        "headline": "chargé d’affaires chez belectric",
                    },
                },
                {
                    "type": "connection_request",
                    "text": "Hugues Jouffroy\nDirecteur Général Délégué aux Opérations\nSent 1 week ago\nWithdraw",
                    "recipient": {
                        "name": "Hugues Jouffroy",
                        "url": "/in/hugues-jouffroy/",
                        "headline": "Directeur Général Délégué aux Opérations",
                    },
                },
                {"type": "page_follow", "recipient": {"name": "Ignored"}},
                "bad row",
            ]
        )

        cards = await extractor._extract_invitation_cards(kind="sent", limit=10)

        evaluate_args = mock_page.evaluate.await_args
        assert evaluate_args is not None
        assert evaluate_args.args[1] == {"kind": "sent", "limit": 20}
        assert cards == [
            {
                "type": "connection_request",
                "invitation_age": "1w",
                "recipient": {
                    "name": "Laurent SORBIER",
                    "url": "/in/laurent-sorbier/",
                    "headline": "chargé d’affaires chez belectric",
                },
            },
            {
                "type": "connection_request",
                "invitation_age": "1w",
                "recipient": {
                    "name": "Hugues Jouffroy",
                    "url": "/in/hugues-jouffroy/",
                    "headline": "Directeur Général Délégué aux Opérations",
                },
            },
        ]
        assert "sender" not in cards[0]
        assert "target" not in cards[0]
        assert "note" not in cards[0]
        assert "message_url" not in cards[0]

    async def test_expand_invitation_note_toggles_runs_second_pass(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        mock_page.evaluate = AsyncMock(side_effect=[1, 0])
        with patch(
            "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
            new_callable=AsyncMock,
        ) as mock_sleep:
            await extractor._expand_invitation_note_toggles()

        assert mock_page.evaluate.await_count == 2
        mock_sleep.assert_awaited_once_with(0.5)

    # --- act_on_invitation: accept/ignore via profile ----------------------

    @staticmethod
    def _patch_incoming_profile_pipeline(
        extractor,
        *,
        state: str,
        verified_state: str | None = None,
        click_clicked: bool = True,
    ):
        """Patch the scrape→signals→click→verify pipeline used by
        ``_respond_to_incoming_invitation``. Returns the ExitStack (entered
        by caller), the ``detect_connection_state`` MagicMock, and the
        ``page.evaluate`` AsyncMock so tests can override its return value
        to exercise diagnostic surfacing.

        Verification polls (up to 10 iterations) so detect/signals/evaluate
        mocks must tolerate arbitrary call counts. The first
        ``detect_connection_state`` call returns ``state``; subsequent calls
        return ``verified_state``. The first ``page.evaluate`` returns the
        click result; subsequent calls (fresh-text fetches during polling)
        return empty string. Tests that need to inject diagnostics into the
        click result mutate ``evaluate_mock.click_result``.
        """
        from contextlib import ExitStack

        stack = ExitStack()
        e = stack.enter_context

        # The lean gating path: _load_profile_for_state replaces the
        # heavier scrape_person + _read_action_signals pair on write
        # paths. Mock it directly so tests don't need to know about the
        # internal navigation + innertext + signals decomposition.
        e(
            patch.object(
                extractor,
                "_load_profile_for_state",
                new_callable=AsyncMock,
                return_value=("Alice\n--\nLondon", MagicMock()),
            )
        )
        # The polling verification still calls _read_action_signals
        # directly. Any number of calls return a fresh MagicMock;
        # detect_mock drives the state transition.
        e(
            patch.object(
                extractor,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            )
        )

        detect_calls = {"n": 0}
        final_state = verified_state if verified_state is not None else state

        def _detect_side_effect(*args, **kwargs):
            detect_calls["n"] += 1
            return state if detect_calls["n"] == 1 else final_state

        detect_mock = MagicMock(side_effect=_detect_side_effect)
        e(
            patch(
                "linkedin_mcp_server.scraping.connection.detect_connection_state",
                detect_mock,
            )
        )

        # evaluate_mock.click_result is the dict returned by the FIRST
        # evaluate (the click). Subsequent calls (polling for fresh main
        # innerText) return an empty string so the verification loop reads
        # a non-incoming-request text and exits as soon as detect_mock
        # transitions.
        evaluate_calls = {"n": 0}

        async def _evaluate_side_effect(*args, **kwargs):
            evaluate_calls["n"] += 1
            if evaluate_calls["n"] == 1:
                return evaluate_mock.click_result
            return ""

        evaluate_mock = AsyncMock(side_effect=_evaluate_side_effect)
        evaluate_mock.click_result = {
            "found": click_clicked,
            "clicked": click_clicked,
        }
        extractor._page.evaluate = evaluate_mock
        e(
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            )
        )
        return stack, detect_mock, evaluate_mock

    @pytest.mark.parametrize(
        "action, verified_state, status",
        [
            ("accept", "already_connected", "accepted"),
            # After ignore, the incoming-request signal disappears — the
            # state typically reverts to connectable (the Ignore button
            # row goes away, leaving Connect as the available action).
            ("ignore", "connectable", "ignored"),
        ],
    )
    async def test_respond_to_incoming_success_path(
        self, mock_page, action, verified_state, status
    ):
        extractor = LinkedInExtractor(mock_page)
        stack, _, evaluate_mock = self._patch_incoming_profile_pipeline(
            extractor,
            state="incoming_request",
            verified_state=verified_state,
        )
        with stack:
            result = await extractor.act_on_invitation("alice", action)
        assert result["status"] == status
        assert result["action"] == action
        assert result["performed"] is True
        assert result["linkedin_username"] == "alice"
        assert result["profile_url"] == "/in/alice/"
        assert result["url"] == "https://www.linkedin.com/in/alice/"
        # The click JS was the FIRST evaluate call; subsequent calls fetch
        # fresh main innerText during verification polling.
        first_call = evaluate_mock.await_args_list[0]
        js_args = first_call.args[1]
        assert js_args["action"] == action
        # Labels come from INCOMING_REQUEST_LABELS — at least the English
        # pair must be present in target/other label arrays.
        if action == "accept":
            assert "Accept" in js_args["target_labels"]
            assert "Ignore" in js_args["other_labels"]
        else:
            assert "Ignore" in js_args["target_labels"]
            assert "Accept" in js_args["other_labels"]

    async def test_act_on_invitation_not_found_without_username(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        # No pipeline patching — short-circuits before any scrape.
        with patch.object(extractor, "scrape_person", new_callable=AsyncMock) as scrape:
            result = await extractor.act_on_invitation("", "accept")
        scrape.assert_not_awaited()
        assert result["status"] == "not_found"
        assert result["action"] == "accept"
        assert result["performed"] is False

    async def test_respond_to_incoming_already_connected_short_circuits(
        self, mock_page
    ):
        extractor = LinkedInExtractor(mock_page)
        stack, _, evaluate_mock = self._patch_incoming_profile_pipeline(
            extractor, state="already_connected"
        )
        with stack:
            result = await extractor.act_on_invitation("alice", "accept")
        assert result["status"] == "already_connected"
        assert result["performed"] is False
        evaluate_mock.assert_not_awaited()

    async def test_respond_to_incoming_no_request_returns_not_found(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        stack, _, evaluate_mock = self._patch_incoming_profile_pipeline(
            extractor, state="connectable"
        )
        with stack:
            result = await extractor.act_on_invitation("alice", "ignore")
        assert result["status"] == "not_found"
        assert "No incoming connection request" in result["message"]
        assert result["performed"] is False
        evaluate_mock.assert_not_awaited()

    async def test_pending_profile_falls_back_to_received_invitations(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        stack, _, evaluate_mock = self._patch_incoming_profile_pipeline(
            extractor, state="pending"
        )
        invitation = {"sender": {"url": "/in/alice/"}}
        with (
            stack,
            patch.object(
                extractor,
                "get_pending_invitations",
                new_callable=AsyncMock,
                side_effect=[
                    {"invitations": [invitation]},
                    {"invitations": []},
                ],
            ) as get_pending,
        ):
            result = await extractor.act_on_invitation("alice", "ignore")

        assert result["status"] == "ignored"
        assert result["performed"] is True
        assert result["url"].endswith("/invitation-manager/received/")
        assert get_pending.await_count == 2
        get_pending.assert_awaited_with(limit=100, kind="received")
        assert evaluate_mock.await_args.args[1]["username"] == "alice"
        assert evaluate_mock.await_args.args[1]["action"] == "ignore"

    async def test_respond_to_incoming_action_unavailable_when_click_fails(
        self, mock_page
    ):
        extractor = LinkedInExtractor(mock_page)
        stack, _, evaluate_mock = self._patch_incoming_profile_pipeline(
            extractor, state="incoming_request"
        )
        evaluate_mock.click_result = {
            "found": True,
            "clicked": False,
            "reason": "no_buttons",
            "action_buttons": [],
        }
        with stack:
            result = await extractor.act_on_invitation("alice", "ignore")
        assert result["status"] == "action_unavailable"
        assert result["performed"] is False
        assert "ignore" in result["message"].lower()
        # Diagnostics propagate even on the failure path.
        assert result.get("action_buttons") == []

    async def test_respond_to_incoming_surfaces_click_diagnostics(self, mock_page):
        """The button enumeration and chosen strategy must propagate to the
        tool result so misroutes are debuggable from the JSON-RPC response."""
        extractor = LinkedInExtractor(mock_page)
        stack, _, evaluate_mock = self._patch_incoming_profile_pipeline(
            extractor,
            state="incoming_request",
            verified_state="already_connected",
        )
        click_diagnostics = {
            "found": True,
            "clicked": True,
            "button_count": 2,
            "match_strategy": "design_system_class",
            "clicked_button": {
                "tag": "button",
                "aria_label": None,
                "text": "Accept",
                "data_control": None,
                "data_test": None,
                "data_view": None,
                "classes": "artdeco-button artdeco-button--primary",
            },
            "action_buttons": [
                {"tag": "button", "text": "Ignore", "classes": "artdeco-button"},
                {
                    "tag": "button",
                    "text": "Accept",
                    "classes": "artdeco-button--primary",
                },
            ],
        }
        evaluate_mock.click_result = click_diagnostics
        with stack:
            result = await extractor.act_on_invitation("alice", "accept")

        assert result["status"] == "accepted"
        assert result["match_strategy"] == "design_system_class"
        assert result["clicked_button"]["text"] == "Accept"
        assert len(result["action_buttons"]) == 2
        # button_count is exposed as action_count for symmetry with the
        # previous tool contract.
        assert result["action_count"] == 2

    def test_click_incoming_action_js_declares_strategy_layers(self):
        """Regression guard: the incoming-action JS must keep its four
        documented strategy layers (label_text → attr → design-system class
        → position) and filter out icon-only/expanded buttons so we never
        land on a More menu opener or chevron button.

        The label_text strategy is essential because LinkedIn does not
        always render a Message button alongside an incoming request —
        the compose-anchor action-root walk can return null even when
        Accept + Ignore are clearly in the DOM. Scanning <main> for
        buttons whose visible text matches INCOMING_REQUEST_LABELS is
        the robust catch-all per CLAUDE.md's text-fallback exception.
        """
        from linkedin_mcp_server.scraping.extractor import _CLICK_INCOMING_ACTION_JS

        # Strategy 0: locale-table label scan.
        assert "'label_text'" in _CLICK_INCOMING_ACTION_JS
        assert "target_labels" in _CLICK_INCOMING_ACTION_JS
        # Strategy 1: stable attr substrings.
        assert "'attr_' + action" in _CLICK_INCOMING_ACTION_JS
        # Strategy 2: design-system primary class.
        assert "'design_system_class'" in _CLICK_INCOMING_ACTION_JS
        assert "artdeco-button--primary" in _CLICK_INCOMING_ACTION_JS
        # Strategy 3: position fallback.
        assert "strategy = 'position'" in _CLICK_INCOMING_ACTION_JS
        # Position constants: [Ignore, Accept] — Accept at index 1.
        assert "action === 'accept' ? 1 : 0" in _CLICK_INCOMING_ACTION_JS
        # Filters: icon-only (no visible text) + More menu (aria-expanded).
        assert "isTextBearing" in _CLICK_INCOMING_ACTION_JS
        assert "aria-expanded" in _CLICK_INCOMING_ACTION_JS
        # Diagnostic enumeration on every return path.
        assert "action_buttons:" in _CLICK_INCOMING_ACTION_JS
        assert "main_buttons:" in _CLICK_INCOMING_ACTION_JS

    # --- withdraw via profile page ----------------------------------------

    @staticmethod
    def _patch_withdraw_profile_pipeline(
        extractor,
        *,
        state: str,
        verified_state: str | None = None,
        click_clicked: bool = True,
        dialog_open: bool = True,
    ):
        """Patch the scrape→signals→click→dialog→confirm→verify pipeline used
        by ``_withdraw_outgoing_invitation``. Returns the ExitStack (entered
        by caller), the ``detect_connection_state`` MagicMock, and the
        ``page.evaluate`` AsyncMock (so tests can override its side_effect
        to exercise confirm-click failures)."""
        from contextlib import ExitStack

        stack = ExitStack()
        e = stack.enter_context

        # Lean gating: _load_profile_for_state replaces scrape_person +
        # _read_action_signals on write paths.
        e(
            patch.object(
                extractor,
                "_load_profile_for_state",
                new_callable=AsyncMock,
                return_value=("Alice\n--\nLondon", MagicMock()),
            )
        )
        # Withdraw also re-reads signals for verification (call 2).
        e(
            patch.object(
                extractor,
                "_read_action_signals",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            )
        )
        detect_mock = MagicMock(
            side_effect=[state, verified_state if verified_state is not None else state]
        )
        e(
            patch(
                "linkedin_mcp_server.scraping.connection.detect_connection_state",
                detect_mock,
            )
        )
        # Two evaluate calls happen on the happy path: the Pending anchor
        # click and the dialog confirm click. Both return success by default.
        evaluate_mock = AsyncMock(
            return_value={"found": click_clicked, "clicked": click_clicked}
        )
        extractor._page.evaluate = evaluate_mock
        e(
            patch.object(
                extractor,
                "_dialog_is_open",
                new_callable=AsyncMock,
                return_value=dialog_open,
            )
        )
        extractor._page.wait_for_selector = AsyncMock()
        return stack, detect_mock, evaluate_mock

    async def test_withdraw_via_profile_success(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        stack, _, _ = self._patch_withdraw_profile_pipeline(
            extractor, state="pending", verified_state="connectable"
        )
        with stack:
            result = await extractor.act_on_invitation("alice", "withdraw")
        assert result["status"] == "withdrawn"
        assert result["action"] == "withdraw"
        assert result["performed"] is True
        assert result["url"] == "https://www.linkedin.com/in/alice/"
        assert result["profile_url"] == "/in/alice/"

    async def test_withdraw_via_profile_already_connected_short_circuits(
        self, mock_page
    ):
        extractor = LinkedInExtractor(mock_page)
        stack, _, evaluate_mock = self._patch_withdraw_profile_pipeline(
            extractor, state="already_connected"
        )
        with stack:
            result = await extractor.act_on_invitation("alice", "withdraw")
        assert result["status"] == "already_connected"
        assert result["performed"] is False
        # No dialog interaction should have occurred — short-circuit before click.
        evaluate_mock.assert_not_awaited()

    async def test_withdraw_via_profile_not_pending_returns_not_found(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        stack, _, evaluate_mock = self._patch_withdraw_profile_pipeline(
            extractor, state="connectable"
        )
        with stack:
            result = await extractor.act_on_invitation("bob", "withdraw")
        assert result["status"] == "not_found"
        assert "No outgoing connection request found" in result["message"]
        assert result["performed"] is False
        evaluate_mock.assert_not_awaited()

    async def test_withdraw_via_profile_dialog_does_not_open(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        stack, _, _ = self._patch_withdraw_profile_pipeline(
            extractor, state="pending", dialog_open=False
        )
        with stack:
            result = await extractor.act_on_invitation("alice", "withdraw")
        assert result["status"] == "action_unavailable"
        # We clicked Pending but the dialog never opened -> performed=True so
        # callers can see that a side effect did happen even though it
        # didn't complete the full withdraw flow.
        assert result["performed"] is True

    async def test_withdraw_via_profile_verification_failed(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        # Initial state=pending, verification still sees pending -> failed.
        stack, _, _ = self._patch_withdraw_profile_pipeline(
            extractor, state="pending", verified_state="pending"
        )
        with stack:
            result = await extractor.act_on_invitation("alice", "withdraw")
        assert result["status"] == "verification_failed"
        assert result["performed"] is True

    async def test_withdraw_via_profile_surfaces_confirm_diagnostics(self, mock_page):
        """The dialog enumeration from the confirm JS must propagate to the
        tool result so misroutes are debuggable from the JSON-RPC response."""
        extractor = LinkedInExtractor(mock_page)
        stack, _, evaluate_mock = self._patch_withdraw_profile_pipeline(
            extractor, state="pending", verified_state="connectable"
        )
        # First evaluate call (Pending anchor click): minimal success result.
        # Second evaluate call (Withdraw confirm): rich diagnostics that
        # must round-trip into the tool result unchanged.
        confirm_diagnostics = {
            "found": True,
            "clicked": True,
            "match_strategy": "design_system_primary",
            "clicked_button": {
                "tag": "button",
                "aria_label": None,
                "text": "Withdraw",
                "data_control": None,
                "data_test": None,
                "data_view": None,
                "classes": "artdeco-button artdeco-button--primary",
            },
            "dialog_buttons": [
                {"tag": "button", "text": "Cancel", "classes": "artdeco-button"},
                {
                    "tag": "button",
                    "text": "Withdraw",
                    "classes": "artdeco-button--primary",
                },
            ],
        }
        evaluate_mock.side_effect = [
            {"found": True, "clicked": True},  # Pending anchor click
            confirm_diagnostics,  # Withdraw confirm click
        ]
        with stack:
            result = await extractor.act_on_invitation("alice", "withdraw")

        assert result["status"] == "withdrawn"
        assert result["match_strategy"] == "design_system_primary"
        assert result["clicked_button"]["text"] == "Withdraw"
        assert len(result["dialog_buttons"]) == 2
        assert result["dialog_buttons"][1]["text"] == "Withdraw"

    async def test_withdraw_via_profile_confirm_click_fails(self, mock_page):
        """If the JS confirm click doesn't land, we return action_unavailable
        without dismissing the dialog — a dismiss would actively close the
        modal without withdrawing, which is the original bug we are fixing."""
        extractor = LinkedInExtractor(mock_page)
        stack, _, evaluate_mock = self._patch_withdraw_profile_pipeline(
            extractor, state="pending"
        )
        # Pending click succeeds; confirm click fails.
        evaluate_mock.side_effect = [
            {"found": True, "clicked": True},
            {"found": True, "clicked": False, "reason": "no_buttons"},
        ]
        with stack:
            result = await extractor.act_on_invitation("alice", "withdraw")
        assert result["status"] == "action_unavailable"
        assert result["performed"] is True
        assert "confirm" in result["message"].lower()

    def test_click_withdraw_confirm_js_strategy_layers(self):
        """Regression guard: the withdraw-confirm JS must keep its three
        documented strategy layers. Past misroutes:

        * Without the artdeco-modal__dismiss filter, a "last visible
          button" position fallback can land on the close X and dismiss
          the dialog without performing the withdraw.
        * Matching 'confirm' as an attr substring misroutes onto the
          Cancel button when LinkedIn names it "modal-confirm-cancel" or
          similar — observed live, caused verification_failed.
        * Without the artdeco-button--primary class strategy, accounts
          where withdraw isn't in the data-control-name fall through to
          the position fallback even when the design system already
          identifies the primary action.
        """
        from linkedin_mcp_server.scraping.extractor import _CLICK_WITHDRAW_CONFIRM_JS

        # Close X must be filtered structurally.
        assert "artdeco-modal__dismiss" in _CLICK_WITHDRAW_CONFIRM_JS

        # Strategy 1: narrow attr match on 'withdraw'.
        assert "'attr_withdraw'" in _CLICK_WITHDRAW_CONFIRM_JS
        assert "includes('withdraw')" in _CLICK_WITHDRAW_CONFIRM_JS

        # Strategy 2: design-system primary class.
        assert "'design_system_primary'" in _CLICK_WITHDRAW_CONFIRM_JS
        assert "artdeco-button--primary" in _CLICK_WITHDRAW_CONFIRM_JS

        # Strategy 3: position fallback as last resort.
        assert "strategy = 'position'" in _CLICK_WITHDRAW_CONFIRM_JS

        # Diagnostics: every return path must expose dialog_buttons so
        # misroutes are visible in the tool result without extra logging.
        assert "dialog_buttons:" in _CLICK_WITHDRAW_CONFIRM_JS
        assert "clicked_button:" in _CLICK_WITHDRAW_CONFIRM_JS

    async def test_respond_to_incoming_verification_failed(self, mock_page):
        """Click landed but the state still shows incoming_request after
        the re-read — we report verification_failed without claiming
        success."""
        extractor = LinkedInExtractor(mock_page)
        stack, _, _ = self._patch_incoming_profile_pipeline(
            extractor,
            state="incoming_request",
            verified_state="incoming_request",
        )
        with stack:
            result = await extractor.act_on_invitation("alice", "accept")
        assert result["status"] == "verification_failed"
        assert result["performed"] is True

    async def test_respond_to_incoming_verification_polls_fresh_text(self, mock_page):
        """Live regression: verification must re-fetch <main> innerText
        rather than reusing the original scrape's page_text. The original
        text still contains the "Accept\\nIgnore" lines and would always
        re-classify as incoming_request, falsely reporting
        verification_failed even after a successful click.

        This test forces detect_connection_state to flip to
        already_connected on the second call — verifying the polling
        loop actually re-evaluates rather than caching the initial state.
        """
        extractor = LinkedInExtractor(mock_page)
        stack, detect_mock, evaluate_mock = self._patch_incoming_profile_pipeline(
            extractor,
            state="incoming_request",
            verified_state="already_connected",
        )
        with stack:
            result = await extractor.act_on_invitation("alice", "accept")

        assert result["status"] == "accepted"
        assert result["performed"] is True
        # detect_connection_state must be called at least twice: once for
        # the pre-click state, once for the post-click verification.
        assert detect_mock.call_count >= 2
        # page.evaluate must be awaited at least twice: once for the
        # click, once for the fresh-text fetch during verification.
        assert evaluate_mock.await_count >= 2

    async def test_act_on_invitation_rejects_unknown_action(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        # Bypass static-type narrowing to exercise the runtime guard.
        act = getattr(extractor, "act_on_invitation")
        with pytest.raises(ValueError, match="Unknown invitation action"):
            await act("alice", "explode")

    async def test_accept_incoming_invitation_delegates_to_act_on_invitation(
        self, mock_page
    ):
        """Slim wrapper preserves connect_with_person's call shape."""
        extractor = LinkedInExtractor(mock_page)
        with patch.object(
            extractor,
            "act_on_invitation",
            new_callable=AsyncMock,
            return_value={"status": "accepted"},
        ) as mock_act:
            result = await extractor._accept_incoming_invitation("alice")
        mock_act.assert_awaited_once_with("alice", "accept")
        assert result == {"status": "accepted"}


# ----------------------------------------------------------------------
# Conversation test helpers.
# ----------------------------------------------------------------------
def _event(**kw):
    """Build a JS-extractor event dict with null defaults.

    Lets test cases name only the fields they care about; the JS-side
    contract requires every key to be present (or null) per
    ``scraping/conversation.py``.
    """
    return {
        "day_heading": None,
        "time_text": None,
        "sender_url": None,
        "sender_name": None,
        "body_text": None,
        "shared_url": None,
        **kw,
    }


def _raw(events, viewer_urn=None):
    """Wrap events into the JS-extractor result dict shape."""
    out = {"events": events}
    if viewer_urn is not None:
        out["viewer_urn"] = viewer_urn
    return out


@contextmanager
def _patched_extractor(
    mock_page,
    *,
    extract_return=([], []),
    thread_urls=None,
    display_name=None,
):
    """Stand up a LinkedInExtractor with the standard get_conversation
    patches applied. Yields ``(extractor, nav_mock, scroll_mock,
    extract_mock)``.

    Pass ``thread_urls`` and ``display_name`` to also patch the
    username-resolution path.
    """
    extractor = LinkedInExtractor(mock_page)
    mock_page.wait_for_selector = AsyncMock()
    nav_mock = AsyncMock()
    scroll_mock = AsyncMock()
    extract_mock = AsyncMock(return_value=extract_return)
    with ExitStack() as stack:
        e = stack.enter_context
        e(patch.object(extractor, "_navigate_to_page", nav_mock))
        e(
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            )
        )
        e(
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
            )
        )
        e(patch.object(extractor, "_wait_for_main_text", new_callable=AsyncMock))
        e(patch.object(extractor, "_scroll_main_scrollable_region", scroll_mock))
        e(
            patch(
                "linkedin_mcp_server.scraping.extractor.extract_conversation",
                extract_mock,
            )
        )
        if display_name is not None:
            e(
                patch.object(
                    extractor,
                    "_read_profile_display_name",
                    new_callable=AsyncMock,
                    return_value=display_name,
                )
            )
        if thread_urls is not None:
            e(
                patch.object(
                    extractor,
                    "_resolve_conversation_thread_urls",
                    new_callable=AsyncMock,
                    return_value=thread_urls,
                )
            )
        yield extractor, nav_mock, scroll_mock, extract_mock


_JACKI_URLS = [
    "https://www.linkedin.com/messaging/thread/2-newer/",
    "https://www.linkedin.com/messaging/thread/2-older/",
]


class TestGetConversation:
    async def test_returns_structured_messages_by_thread_id(self, mock_page):
        """get_conversation returns sections.messages + sections.members."""
        sample_messages = [
            {
                "timestamp": "2026-02-10T15:17:00",
                "status": "sent",
                "sender": 0,
                "content": "Hello!",
            },
            {
                "timestamp": "2026-02-10T15:18:00",
                "status": "deleted",
                "sender": 0,
                "content": None,
            },
        ]
        sample_members = [
            {"kind": "person", "url": "/in/alice/", "name": "Alice", "is_self": True},
            {"kind": "person", "url": "/in/bob/", "name": "Bob", "is_self": False},
        ]
        with _patched_extractor(
            mock_page, extract_return=(sample_messages, sample_members)
        ) as (ext, nav_mock, _scroll, _extract):
            result = await ext.get_conversation(thread_id="abc123")

        nav_mock.assert_awaited_once_with(
            "https://www.linkedin.com/messaging/thread/abc123/"
        )
        assert result["sections"]["messages"] == sample_messages
        assert result["sections"]["members"] == sample_members
        # Old single-string "conversation" key must not leak through.
        assert "conversation" not in result["sections"]

    async def test_max_scrolls_zero_skips_scroll(self, mock_page):
        """max_scrolls=0 bypasses the back-scroll loop entirely."""
        with _patched_extractor(mock_page) as (ext, _nav, scroll_mock, _extract):
            await ext.get_conversation(thread_id="abc123", max_scrolls=0)
        scroll_mock.assert_not_called()

    async def test_max_scrolls_threads_attempts(self, mock_page):
        """max_scrolls=N invokes the scroll helper with attempts=N."""
        with _patched_extractor(mock_page) as (ext, _nav, scroll_mock, _extract):
            await ext.get_conversation(thread_id="abc123", max_scrolls=7)
        scroll_mock.assert_awaited_once()
        assert scroll_mock.await_args.kwargs["attempts"] == 7

    async def test_raises_when_no_identifier(self, mock_page):
        """get_conversation raises LinkedInScraperException with no args."""
        extractor = LinkedInExtractor(mock_page)
        with pytest.raises(LinkedInScraperException):
            await extractor.get_conversation()

    async def test_reads_invitation_conversation_from_compose_dialog(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        message_url = "/messaging/compose/?recipient=ACoAAB&invitation=urn"
        messages = [
            {
                "timestamp": "2026-07-27T13:13:00",
                "status": "sent",
                "sender": 1,
                "content": "Hello!",
            }
        ]
        members = [
            {"kind": "person", "is_self": True},
            {
                "kind": "person",
                "url": "/in/gautier/",
                "name": "Gautier",
                "is_self": False,
            },
        ]
        compose_root = MagicMock()

        with (
            patch.object(
                extractor, "_navigate_to_page", new_callable=AsyncMock
            ) as navigate,
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
            ),
            patch.object(extractor, "_wait_for_main_text", new_callable=AsyncMock),
            patch.object(
                extractor,
                "_read_profile_display_name",
                new_callable=AsyncMock,
                return_value="Gautier",
            ),
            patch.object(
                extractor,
                "_open_invitation_message_compose",
                new_callable=AsyncMock,
                return_value=True,
            ) as open_compose,
            patch.object(
                extractor,
                "_wait_for_message_surface",
                new_callable=AsyncMock,
                return_value="composer",
            ),
            patch.object(
                extractor,
                "_resolve_recipient_message_compose_box",
                new_callable=AsyncMock,
                return_value=compose_root,
            ),
            patch.object(
                extractor,
                "_scroll_main_scrollable_region",
                new_callable=AsyncMock,
            ) as scroll,
            patch(
                "linkedin_mcp_server.scraping.extractor.extract_conversation",
                new_callable=AsyncMock,
                return_value=(messages, members),
            ) as extract,
        ):
            result = await extractor.get_conversation(
                linkedin_username="gautier",
                message_url=message_url,
            )

        navigate.assert_awaited_once_with("https://www.linkedin.com/in/gautier/")
        open_compose.assert_awaited_once_with("gautier", message_url)
        scroll.assert_awaited_once_with(
            position="top",
            attempts=3,
            pause_time=0.5,
            root=compose_root,
        )
        extract.assert_awaited_once_with(
            mock_page,
            root=compose_root,
        )
        assert result["sections"] == {"messages": messages, "members": members}

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            (
                {"message_url": "/messaging/compose/?recipient=ACoAAB"},
                "linkedin_username is required",
            ),
            (
                {
                    "linkedin_username": "gautier",
                    "thread_id": "abc123",
                    "message_url": "/messaging/compose/?recipient=ACoAAB",
                },
                "cannot be combined",
            ),
            (
                {
                    "linkedin_username": "gautier",
                    "message_url": "https://example.com/messaging/compose/?recipient=x",
                },
                "relative /messaging/compose/",
            ),
        ],
    )
    async def test_rejects_invalid_invitation_conversation_args(
        self, mock_page, kwargs, message
    ):
        extractor = LinkedInExtractor(mock_page)
        with pytest.raises(LinkedInScraperException, match=message):
            await extractor.get_conversation(**kwargs)

    async def test_by_username_default_index_picks_first_thread(self, mock_page):
        """get_conversation by username opens the 0th matching thread by default."""
        with _patched_extractor(
            mock_page,
            display_name="Jacki McMahan",
            thread_urls=_JACKI_URLS,
        ) as (ext, nav_mock, _scroll, _extract):
            await ext.get_conversation(linkedin_username="jacki-old")
        target = [
            c.args[0]
            for c in nav_mock.call_args_list
            if c.args and "/messaging/thread/" in c.args[0]
        ]
        assert target == ["https://www.linkedin.com/messaging/thread/2-newer/"]

    async def test_by_username_index_picks_specified_thread(self, mock_page):
        """get_conversation by username + index opens the i-th matching thread."""
        with _patched_extractor(
            mock_page,
            display_name="Jacki McMahan",
            thread_urls=_JACKI_URLS,
        ) as (ext, nav_mock, _scroll, _extract):
            await ext.get_conversation(linkedin_username="jacki-old", index=1)
        target = [
            c.args[0]
            for c in nav_mock.call_args_list
            if c.args and "/messaging/thread/" in c.args[0]
        ]
        assert target == ["https://www.linkedin.com/messaging/thread/2-older/"]

    async def test_by_username_index_out_of_range_raises(self, mock_page):
        """get_conversation raises when index exceeds the number of threads."""
        with _patched_extractor(
            mock_page,
            display_name="Jacki McMahan",
            thread_urls=["https://www.linkedin.com/messaging/thread/2-only/"],
        ) as (ext, *_):
            with pytest.raises(LinkedInScraperException, match="out of range"):
                await ext.get_conversation(linkedin_username="jacki-old", index=5)

    async def test_by_username_no_threads_raises_could_not_find(self, mock_page):
        """get_conversation raises 'Could not find a conversation' when none exist."""
        with _patched_extractor(
            mock_page,
            display_name="Jacki McMahan",
            thread_urls=[],
        ) as (ext, *_):
            with pytest.raises(
                LinkedInScraperException, match="Could not find a conversation"
            ):
                await ext.get_conversation(linkedin_username="jacki-old")


class TestArchiveConversation:
    async def test_opens_then_archives(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        mock_page.evaluate = AsyncMock(
            return_value={
                "clicked": True,
                "verified": True,
                "alreadyArchived": False,
            }
        )

        with patch.object(
            extractor,
            "_open_conversation_surface",
            new_callable=AsyncMock,
            return_value=None,
        ) as open_conversation:
            result = await extractor.archive_conversation(thread_id="thread-123")

        assert result["status"] == "archived"
        assert result["archived"] is True
        assert result["already_archived"] is False
        open_conversation.assert_awaited_once_with(
            linkedin_username=None,
            thread_id="thread-123",
            message_url=None,
            index=0,
        )

    async def test_already_archived_is_success(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        mock_page.evaluate = AsyncMock(
            return_value={
                "clicked": False,
                "verified": True,
                "alreadyArchived": True,
            }
        )

        with patch.object(
            extractor,
            "_open_conversation_surface",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await extractor.archive_conversation(thread_id="thread-123")

        assert result["status"] == "archived"
        assert result["performed"] is False
        assert result["already_archived"] is True
        assert result["message"] == "Conversation was already archived."

    async def test_invitation_archive_uses_resolved_dialog(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        compose_root = MagicMock()
        compose_root.evaluate = AsyncMock(
            return_value={
                "clicked": True,
                "verified": True,
                "alreadyArchived": False,
            }
        )

        with patch.object(
            extractor,
            "_open_conversation_surface",
            new_callable=AsyncMock,
            return_value=compose_root,
        ):
            result = await extractor.archive_conversation(
                linkedin_username="gautier",
                message_url="/messaging/compose/?profileUrn=urn%3Ali%3Afsd_profile%3A1",
            )

        assert result["status"] == "archived"
        compose_root.evaluate.assert_awaited_once_with(_ARCHIVE_CONVERSATION_JS)
        mock_page.evaluate.assert_not_awaited()
        assert "anchor?.closest('[role=\"dialog\"]')" in _ARCHIVE_CONVERSATION_JS
        assert "const event = root.querySelector(" in _ARCHIVE_CONVERSATION_JS
        assert "Array.from(root.querySelectorAll(" in _ARCHIVE_CONVERSATION_JS


class TestConversationParserHelpers:
    """Unit tests for the conversation parser helpers (no browser needed).

    Covers timestamp reconstruction, status classification, profile URL
    normalization, and quoted-reply flattening — the en-US, locale-aware
    pieces of the new ``get_conversation`` pipeline.
    """

    async def test_extract_conversation_uses_dialog_locator(self, mock_page):
        dialog_locator = MagicMock()
        dialog_locator.evaluate = AsyncMock(
            return_value={"events": [], "viewer_urn": None}
        )
        mock_page.evaluate = AsyncMock()

        result = await conversation_parser.extract_conversation(
            mock_page, root=dialog_locator
        )

        assert result == ([], [])
        dialog_locator.evaluate.assert_awaited_once()
        mock_page.evaluate.assert_not_awaited()

    def test_normalize_profile_url_strips_origin_and_query(self):
        assert (
            conversation_parser.normalize_profile_url(
                "https://www.linkedin.com/in/ACoAA123/?miniProfileUrn=foo"
            )
            == "/in/ACoAA123/"
        )

    def test_normalize_profile_url_handles_relative(self):
        assert (
            conversation_parser.normalize_profile_url("/in/alice/?bar=baz")
            == "/in/alice/"
        )

    def test_normalize_profile_url_returns_none_for_non_profile(self):
        assert conversation_parser.normalize_profile_url("/jobs/view/42") is None
        assert conversation_parser.normalize_profile_url(None) is None

    def test_parse_day_heading_month_day(self):
        assert conversation_parser.parse_day_heading("Feb 10") == (2, 10, None)

    def test_parse_day_heading_with_explicit_year(self):
        assert conversation_parser.parse_day_heading("Feb 10, 2024") == (2, 10, 2024)

    def test_parse_day_heading_uppercase_via_css_transform(self):
        """innerText returns 'FEB 10' when LinkedIn applies CSS uppercase."""
        assert conversation_parser.parse_day_heading("FEB 10") == (2, 10, None)

    def test_parse_day_heading_today_resolves_to_anchor(self):
        from datetime import datetime as _dt

        today = _dt(2026, 5, 25, 18, 0, 0)
        assert conversation_parser.parse_day_heading("Today", today=today) == (
            5,
            25,
            2026,
        )

    def test_parse_day_heading_yesterday_resolves_to_anchor_minus_one(self):
        from datetime import datetime as _dt

        today = _dt(2026, 5, 25, 18, 0, 0)
        assert conversation_parser.parse_day_heading("Yesterday", today=today) == (
            5,
            24,
            2026,
        )

    def test_parse_day_heading_yesterday_crosses_month(self):
        from datetime import datetime as _dt

        today = _dt(2026, 3, 1, 0, 30, 0)
        # March 1 - 1 day → Feb 28 (2026 is not a leap year)
        assert conversation_parser.parse_day_heading("Yesterday", today=today) == (
            2,
            28,
            2026,
        )

    def test_parse_day_heading_today_case_insensitive(self):
        from datetime import datetime as _dt

        today = _dt(2026, 5, 25)
        assert conversation_parser.parse_day_heading("TODAY", today=today) == (
            5,
            25,
            2026,
        )

    def test_parse_day_heading_unknown_format_returns_none(self):
        assert conversation_parser.parse_day_heading("Whenever") is None

    def test_normalize_shared_url_feed_update(self):
        out = conversation_parser.normalize_shared_url(
            "/feed/update/urn:li:activity:7440422212802826240/"
        )
        assert out == "/feed/update/urn:li:activity:7440422212802826240/"

    def test_normalize_shared_url_absolute_strips_origin(self):
        out = conversation_parser.normalize_shared_url(
            "https://www.linkedin.com/jobs/view/4371001486"
        )
        assert out == "/jobs/view/4371001486"

    def test_normalize_shared_url_posts_slug(self):
        out = conversation_parser.normalize_shared_url("/posts/some-slug-abc123")
        assert out == "/posts/some-slug-abc123"

    def test_normalize_shared_url_rejects_non_content(self):
        # Profile and company links aren't shared-card permalinks.
        assert conversation_parser.normalize_shared_url("/in/someone/") is None
        assert conversation_parser.normalize_shared_url("/company/foo/") is None
        assert conversation_parser.normalize_shared_url(None) is None

    def test_build_iso_timestamp_pm(self):
        out = conversation_parser.build_iso_timestamp("Feb 10", "3:17 PM", 2026)
        assert out == "2026-02-10T15:17:00"

    def test_build_iso_timestamp_am(self):
        out = conversation_parser.build_iso_timestamp("Feb 10", "9:05 AM", 2026)
        assert out == "2026-02-10T09:05:00"

    def test_build_iso_timestamp_midnight(self):
        out = conversation_parser.build_iso_timestamp("Feb 10", "12:30 AM", 2026)
        assert out == "2026-02-10T00:30:00"

    def test_build_iso_timestamp_noon(self):
        out = conversation_parser.build_iso_timestamp("Feb 10", "12:30 PM", 2026)
        assert out == "2026-02-10T12:30:00"

    def test_build_iso_timestamp_falls_back_when_unparseable(self):
        """When format is unrecognized, return the raw concatenation."""
        out = conversation_parser.build_iso_timestamp("Whenever", "3:17 PM", 2026)
        assert "3:17 PM" in out

    def test_classify_status_deleted_en_us(self):
        status, content = conversation_parser.classify_status(
            "This message has been deleted."
        )
        assert status == "deleted"
        assert content is None

    def test_classify_status_normal_message(self):
        status, content = conversation_parser.classify_status("Hello, world!")
        assert status == "sent"
        assert content == "Hello, world!"

    def test_classify_status_other_locale_falls_through_to_sent(self):
        """Non-en-US deleted markers fall through to 'sent' (documented)."""
        status, content = conversation_parser.classify_status(
            "Ce message a été supprimé."
        )
        assert status == "sent"
        assert content == "Ce message a été supprimé."


class TestNormalizeConversationEvents:
    """Integration of the JS-side dump shape into Message/Member lists."""

    def test_normal_message_flow(self):
        """Two distinct senders, no viewer URN → no one is self, ordering
        follows first-appearance order in events."""
        raw = _raw(
            [
                _event(
                    day_heading="Feb 10, 2024",
                    time_text="3:17 PM",
                    sender_url="https://www.linkedin.com/in/alice/",
                    sender_name="Alice",
                    body_text="Hello there!",
                ),
                _event(
                    time_text="3:18 PM",
                    sender_url="/in/bob/",
                    sender_name="Bob",
                    body_text="Hi Alice!",
                ),
            ]
        )
        messages, members = conversation_parser.normalize_conversation_events(raw)
        assert messages == [
            {
                "timestamp": "2024-02-10T15:17:00",
                "status": "sent",
                "sender": 0,
                "content": "Hello there!",
            },
            {
                "timestamp": "2024-02-10T15:18:00",
                "status": "sent",
                "sender": 1,
                "content": "Hi Alice!",
            },
        ]
        assert members == [
            {"kind": "person", "url": "/in/alice/", "name": "Alice", "is_self": False},
            {"kind": "person", "url": "/in/bob/", "name": "Bob", "is_self": False},
        ]

    def test_deleted_message_emits_none_content(self):
        raw = _raw(
            [
                _event(
                    day_heading="May 10",
                    time_text="5:50 PM",
                    sender_url="/in/alice/",
                    sender_name="Alice",
                    body_text="This message has been deleted.",
                ),
            ]
        )
        messages, _ = conversation_parser.normalize_conversation_events(raw)
        assert messages[0]["status"] == "deleted"
        assert messages[0]["content"] is None
        # Sender index still set on tombstones — points at the original author.
        assert isinstance(messages[0]["sender"], int)

    def test_running_day_persists_across_events(self):
        """Subsequent events without their own day_heading inherit the prior."""
        raw = _raw(
            [
                _event(
                    day_heading="Feb 10, 2024",
                    time_text="3:17 PM",
                    sender_url="/in/alice/",
                    sender_name="Alice",
                    body_text="first",
                ),
                _event(
                    time_text="3:18 PM",
                    sender_url="/in/alice/",
                    sender_name="Alice",
                    body_text="second",
                ),
                _event(
                    day_heading="Feb 11, 2024",
                    time_text="9:00 AM",
                    sender_url="/in/bob/",
                    sender_name="Bob",
                    body_text="next day",
                ),
            ]
        )
        messages, _ = conversation_parser.normalize_conversation_events(raw)
        assert [m["timestamp"] for m in messages] == [
            "2024-02-10T15:17:00",
            "2024-02-10T15:18:00",
            "2024-02-11T09:00:00",
        ]

    def test_self_sender_sentinel(self):
        """An event without a resolvable sender URL maps to index 0 (self)
        when viewer URN is known."""
        raw = _raw(
            [
                _event(
                    day_heading="Feb 10",
                    time_text="3:17 PM",
                    sender_name="You",
                    body_text="hello",
                )
            ],
            viewer_urn="ACoAA_VIEWER",
        )
        messages, members = conversation_parser.normalize_conversation_events(raw)
        assert messages[0]["sender"] == 0
        assert members[0]["is_self"] is True

    def test_time_text_inherited_across_same_minute_group(self):
        """Consecutive same-sender messages within a minute share one <time>.
        Subsequent events inherit the prior clock value until a new
        day-heading resets it."""
        raw = _raw(
            [
                _event(
                    day_heading="Feb 10, 2024",
                    time_text="4:36 PM",
                    sender_url="/in/alice/",
                    sender_name="Alice",
                    body_text="first",
                ),
                _event(
                    sender_url="/in/alice/", sender_name="Alice", body_text="second"
                ),
                _event(sender_url="/in/alice/", sender_name="Alice", body_text="third"),
            ]
        )
        messages, _ = conversation_parser.normalize_conversation_events(raw)
        assert [m["timestamp"] for m in messages] == [
            "2024-02-10T16:36:00",
            "2024-02-10T16:36:00",
            "2024-02-10T16:36:00",
        ]
        assert [m["content"] for m in messages] == ["first", "second", "third"]

    def test_running_time_resets_on_new_day_heading(self):
        """Running time should not bleed across day boundaries."""
        raw = _raw(
            [
                _event(
                    day_heading="Feb 10, 2024",
                    time_text="4:36 PM",
                    sender_url="/in/alice/",
                    sender_name="Alice",
                    body_text="day 1",
                ),
                # New day, no time_text — should NOT inherit 4:36 PM.
                _event(
                    day_heading="Feb 11, 2024",
                    sender_url="/in/alice/",
                    sender_name="Alice",
                    body_text="day 2 no time",
                ),
            ]
        )
        messages, _ = conversation_parser.normalize_conversation_events(raw)
        assert messages[0]["timestamp"] == "2024-02-10T16:36:00"
        # Second message has no clock — falls back to the raw day heading.
        assert messages[1]["timestamp"] == "Feb 11, 2024"

    def test_link_card_event_renders_url_as_content(self):
        """A message that is purely a shared link card (no <p> body) emits
        the card's permalink as content rather than being skipped."""
        raw = _raw(
            [
                _event(
                    day_heading="Mar 19, 2026",
                    time_text="5:22 PM",
                    sender_url="/in/alice/",
                    sender_name="Alice",
                    shared_url="/feed/update/urn:li:activity:7440422212802826240/",
                ),
            ]
        )
        messages, _ = conversation_parser.normalize_conversation_events(raw)
        assert len(messages) == 1
        assert messages[0]["status"] == "sent"
        assert (
            messages[0]["content"]
            == "/feed/update/urn:li:activity:7440422212802826240/"
        )

    def test_link_card_absolute_url_normalized_to_relative(self):
        """Absolute LinkedIn URLs in a card href become relative paths."""
        raw = _raw(
            [
                _event(
                    day_heading="Mar 19, 2026",
                    time_text="5:22 PM",
                    sender_url="/in/alice/",
                    sender_name="Alice",
                    shared_url="https://www.linkedin.com/jobs/view/4371001486?ref=foo",
                ),
            ]
        )
        messages, _ = conversation_parser.normalize_conversation_events(raw)
        assert messages[0]["content"] == "/jobs/view/4371001486"

    def test_body_text_wins_over_shared_url(self):
        """When a comment-with-share has both text and a card URL, body wins."""
        raw = _raw(
            [
                _event(
                    day_heading="Feb 16, 2026",
                    time_text="9:01 AM",
                    sender_url="/in/alice/",
                    sender_name="Alice",
                    body_text="Look at this!",
                    shared_url="/feed/update/urn:li:activity:7428718265717190656/",
                ),
            ]
        )
        messages, _ = conversation_parser.normalize_conversation_events(raw)
        assert messages[0]["content"] == "Look at this!"

    def test_skips_event_without_body_unless_deleted(self):
        """Attachment-only / system events (no body_text, not a tombstone)
        are dropped per V1 scope. Their sender still surfaces as a member."""
        raw = _raw(
            [
                _event(
                    day_heading="Feb 10, 2024",
                    time_text="3:17 PM",
                    sender_url="/in/alice/",
                    sender_name="Alice",
                    body_text="real message",
                ),
                # Image-only or similar: no text body, not deleted.
                _event(time_text="3:18 PM", sender_url="/in/bob/", sender_name="Bob"),
            ]
        )
        messages, members = conversation_parser.normalize_conversation_events(raw)
        assert len(messages) == 1
        assert messages[0]["sender"] == 0  # Alice is at index 0 (no viewer URN)
        # Both Alice and Bob still appear as participants — we recorded
        # Bob's name from the dropped event before skipping it.
        urls = {m["url"]: m.get("name") for m in members}
        assert urls == {"/in/alice/": "Alice", "/in/bob/": "Bob"}

    def test_self_always_at_index_zero(self):
        """When viewer URN is known, the authenticated user is members[0]
        regardless of order of appearance in events."""
        raw = _raw(
            [
                # Other person speaks first.
                _event(
                    day_heading="Feb 16, 2026",
                    time_text="9:01 AM",
                    sender_url="/in/ACoAA_OTHER/",
                    sender_name="Other Person",
                    body_text="hi",
                ),
                # Viewer replies.
                _event(
                    time_text="9:02 AM",
                    sender_url="https://www.linkedin.com/in/ACoAA_VIEWER/",
                    sender_name="Me",
                    body_text="hello back",
                ),
            ],
            viewer_urn="ACoAA_VIEWER",
        )
        messages, members = conversation_parser.normalize_conversation_events(raw)
        # Self lives at index 0 even though "other" appears first in events.
        assert members[0]["is_self"] is True
        assert members[0]["url"] == "/in/ACoAA_VIEWER/"
        assert members[1]["is_self"] is False
        assert members[1]["url"] == "/in/ACoAA_OTHER/"
        # Sender indices reference the ordered list: 0 for self, 1 for other.
        assert [m["sender"] for m in messages] == [1, 0]

    def test_self_synthesized_when_viewer_urn_unobserved(self):
        """If viewer URN is known but the viewer never appears as a sender
        anchor (rare), the self member still occupies index 0 but carries
        NO url field. fsd_profile URN isn't a guaranteed vanity slug."""
        raw = _raw(
            [
                _event(
                    day_heading="Feb 16, 2026",
                    time_text="9:01 AM",
                    body_text="I sent this with no anchor",
                ),
                _event(
                    time_text="9:02 AM",
                    sender_url="/in/ACoAA_OTHER/",
                    sender_name="Other Person",
                    body_text="reply",
                ),
            ],
            viewer_urn="ACoAA_VIEWER",
        )
        messages, members = conversation_parser.normalize_conversation_events(raw)
        # Self has is_self True but NO url.
        assert members[0] == {"kind": "person", "is_self": True}
        assert "url" not in members[0]
        assert members[1]["is_self"] is False
        assert members[1]["url"] == "/in/ACoAA_OTHER/"
        assert [m["sender"] for m in messages] == [0, 1]

    def test_is_self_false_on_everyone_when_viewer_urn_absent(self):
        """If JS couldn't determine the viewer URN, is_self is False
        everywhere — but the field is still always present."""
        raw = _raw(
            [
                _event(
                    day_heading="Feb 16, 2026",
                    time_text="9:01 AM",
                    sender_url="/in/ACoAA_SOMEONE/",
                    sender_name="Someone",
                    body_text="hi",
                ),
            ]
        )
        _, members = conversation_parser.normalize_conversation_events(raw)
        assert all(m["is_self"] is False for m in members)

    def test_orphan_event_without_attribution_is_skipped(self):
        """An event with no sender_url AND no viewer URN can't be attributed
        and is dropped — consistent with V1 attachment-skip behavior."""
        raw = _raw(
            [
                _event(
                    day_heading="Feb 10, 2024", time_text="3:17 PM", body_text="orphan"
                ),
            ]
        )
        messages, _ = conversation_parser.normalize_conversation_events(raw)
        assert messages == []


class TestStripSelectConversationPrefix:
    def test_strips_en_us_prefix(self):
        """Best-effort strip removes the en-US 'Select conversation with ' prefix."""
        assert (
            LinkedInExtractor._strip_select_conversation_prefix(
                "Select conversation with Jacki McMahan"
            )
            == "Jacki McMahan"
        )

    def test_case_insensitive(self):
        assert (
            LinkedInExtractor._strip_select_conversation_prefix(
                "select conversation with jacki mcmahan"
            )
            == "jacki mcmahan"
        )

    def test_returns_full_aria_when_prefix_absent(self):
        """In a non-en-US locale the verb prefix won't match; return as-is so
        downstream matching can endsWith / endswith on the participant name."""
        assert (
            LinkedInExtractor._strip_select_conversation_prefix(
                "Konversation auswählen mit Jacki McMahan"
            )
            == "Konversation auswählen mit Jacki McMahan"
        )

    def test_empty_input(self):
        assert LinkedInExtractor._strip_select_conversation_prefix("") == ""


class TestResolveConversationThreadUrls:
    async def test_inbox_enumeration_and_exact_aria_match(self, mock_page):
        """_resolve_conversation_thread_urls enumerates the plain inbox and
        matches participant by exact aria-label rather than substring."""
        extractor = LinkedInExtractor(mock_page)
        nav_mock = AsyncMock()
        thread_refs = [
            {
                "kind": "conversation",
                "url": "/messaging/thread/2-aaa/",
                "text": "Jacki McMahan",  # exact match
                "context": "search",
            },
            {
                "kind": "conversation",
                "url": "/messaging/thread/2-bbb/",
                "text": "Jacki McMahan-Group",  # extra suffix → not exact
                "context": "search",
            },
            {
                "kind": "conversation",
                "url": "/messaging/thread/2-ccc/",
                "text": "Jacki McMahan",  # second exact match (multi-thread case)
                "context": "search",
            },
        ]
        with (
            patch.object(extractor, "_navigate_to_page", nav_mock),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
            ),
            patch.object(extractor, "_wait_for_main_text", new_callable=AsyncMock),
            patch.object(
                extractor, "_scroll_main_scrollable_region", new_callable=AsyncMock
            ),
            patch.object(
                extractor,
                "_extract_conversation_thread_refs",
                new_callable=AsyncMock,
                return_value=thread_refs,
            ),
        ):
            urls = await extractor._resolve_conversation_thread_urls("Jacki McMahan")

        nav_mock.assert_awaited_once_with("https://www.linkedin.com/messaging/")
        assert urls == [
            "https://www.linkedin.com/messaging/thread/2-aaa/",
            "https://www.linkedin.com/messaging/thread/2-ccc/",
        ]

    async def test_resolver_passes_name_filter_to_enumerator(self, mock_page):
        """_resolve_conversation_thread_urls scopes the click side effect by
        forwarding name_filter so only the participant's row is clicked."""
        extractor = LinkedInExtractor(mock_page)
        refs_mock = AsyncMock(
            return_value=[
                {
                    "kind": "conversation",
                    "url": "/messaging/thread/2-aaa/",
                    "text": "Jacki McMahan",
                    "context": "inbox",
                },
            ]
        )
        with (
            patch.object(extractor, "_navigate_to_page", new_callable=AsyncMock),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
            ),
            patch.object(extractor, "_wait_for_main_text", new_callable=AsyncMock),
            patch.object(
                extractor, "_scroll_main_scrollable_region", new_callable=AsyncMock
            ),
            patch.object(extractor, "_extract_conversation_thread_refs", refs_mock),
        ):
            urls = await extractor._resolve_conversation_thread_urls("Jacki McMahan")

        refs_mock.assert_awaited_once_with(
            limit=ANY, context="inbox", name_filter="Jacki McMahan"
        )
        assert urls == ["https://www.linkedin.com/messaging/thread/2-aaa/"]

    async def test_resolver_falls_back_to_search_when_inbox_empty(self, mock_page):
        """When the inbox scan finds no match, resolution falls back to the
        messaging search for threads buried below the inbox window."""
        extractor = LinkedInExtractor(mock_page)
        nav_mock = AsyncMock()
        # First call (inbox) finds nothing; second call (search) finds the thread.
        refs_mock = AsyncMock(
            side_effect=[
                [],
                [
                    {
                        "kind": "conversation",
                        "url": "/messaging/thread/2-ddd/",
                        "text": "Jacki McMahan",
                        "context": "search",
                    },
                ],
            ]
        )
        with (
            patch.object(extractor, "_navigate_to_page", nav_mock),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
            ),
            patch.object(extractor, "_wait_for_main_text", new_callable=AsyncMock),
            patch.object(
                extractor, "_scroll_main_scrollable_region", new_callable=AsyncMock
            ),
            patch.object(extractor, "_extract_conversation_thread_refs", refs_mock),
        ):
            urls = await extractor._resolve_conversation_thread_urls("Jacki McMahan")

        assert nav_mock.await_args_list[0].args == (
            "https://www.linkedin.com/messaging/",
        )
        assert nav_mock.await_args_list[1].args == (
            "https://www.linkedin.com/messaging/?searchTerm=Jacki+McMahan",
        )
        assert refs_mock.await_count == 2
        assert urls == ["https://www.linkedin.com/messaging/thread/2-ddd/"]

    async def test_extract_refs_threads_name_filter_into_evaluate(self, mock_page):
        """_extract_conversation_thread_refs forwards name_filter into the
        in-browser click loop so non-matching rows are never clicked."""
        extractor = LinkedInExtractor(mock_page)
        mock_page.wait_for_selector = AsyncMock()
        captured: dict[str, object] = {}

        async def fake_evaluate(_js: str, arg: dict | None = None) -> list:
            captured["arg"] = arg
            return []

        mock_page.evaluate = fake_evaluate

        await extractor._extract_conversation_thread_refs(
            limit=50, context="inbox", name_filter="Jacki McMahan"
        )

        assert captured["arg"] == {"limit": 50, "nameFilter": "Jacki McMahan"}

    async def test_extract_refs_keeps_rows_without_participant_labels(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        mock_page.wait_for_selector = AsyncMock()
        mock_page.evaluate = AsyncMock(
            return_value=[{"threadId": "2-request", "ariaLabel": ""}]
        )

        refs = await extractor._extract_conversation_thread_refs(
            limit=50, context="inbox"
        )

        assert refs == [
            {
                "kind": "conversation",
                "url": "/messaging/thread/2-request/",
                "context": "inbox",
            }
        ]
        mock_page.wait_for_selector.assert_awaited_once_with(
            'main li div[class*="listitem__link"]',
            state="attached",
            timeout=10000,
        )
        evaluate_call = mock_page.evaluate.await_args
        assert evaluate_call is not None
        script = evaluate_call.args[0]
        assert "'main li div[class*=\"listitem__link\"]'" in script
        assert "clickTarget.closest('li')" in script


class TestSearchConversations:
    async def test_returns_search_results(self, mock_page):
        """search_conversations returns search_results section."""
        extractor = LinkedInExtractor(mock_page)
        nav_mock = AsyncMock()

        with (
            patch.object(extractor, "_navigate_to_page", nav_mock),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
            ),
            patch.object(extractor, "_wait_for_main_text", new_callable=AsyncMock),
            patch.object(
                extractor,
                "_extract_root_content",
                new_callable=AsyncMock,
                return_value={"text": "Result 1\nResult 2", "references": []},
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.strip_linkedin_noise",
                return_value="Result 1\nResult 2",
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.build_references",
                return_value=[],
            ),
            patch.object(
                extractor,
                "_extract_conversation_thread_refs",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = await extractor.search_conversations("hello world")

        assert "search_results" in result["sections"]
        assert "Result 1" in result["sections"]["search_results"]
        # Search must be driven by the searchTerm URL parameter, not by typing
        # into the searchbox -- the URL form is reliable across SPA mounts and
        # preserves the search filter across click-to-capture navigations.
        nav_mock.assert_awaited_once_with(
            "https://www.linkedin.com/messaging/?searchTerm=hello+world"
        )

    async def test_includes_conversation_thread_refs(self, mock_page):
        """search_conversations exposes per-result thread URLs as references."""
        extractor = LinkedInExtractor(mock_page)
        thread_refs = [
            {
                "kind": "conversation",
                "url": "/messaging/thread/2-abc/",
                "text": "Jacki McMahan",
                "context": "search_results",
            },
            {
                "kind": "conversation",
                "url": "/messaging/thread/2-def/",
                "text": "Jacki McMahan",
                "context": "search_results",
            },
        ]
        with (
            patch.object(extractor, "_navigate_to_page", new_callable=AsyncMock),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
            ),
            patch.object(extractor, "_wait_for_main_text", new_callable=AsyncMock),
            patch.object(
                extractor,
                "_extract_root_content",
                new_callable=AsyncMock,
                return_value={"text": "Jacki McMahan\nJacki McMahan", "references": []},
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.strip_linkedin_noise",
                return_value="Jacki McMahan\nJacki McMahan",
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.build_references",
                return_value=[],
            ),
            patch.object(
                extractor,
                "_extract_conversation_thread_refs",
                new_callable=AsyncMock,
                return_value=thread_refs,
            ) as mock_refs,
        ):
            result = await extractor.search_conversations("Jacki")

        mock_refs.assert_awaited_once_with(limit=20, context="search_results")
        refs = result["references"]["search_results"]
        assert len(refs) == 2
        assert {ref["url"] for ref in refs} == {
            "/messaging/thread/2-abc/",
            "/messaging/thread/2-def/",
        }


class TestSendMessage:
    async def test_dry_run_returns_confirmation_required(self, mock_page):
        """send_message with confirm_send=False returns confirmation_required status."""
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(extractor, "_navigate_to_page", new_callable=AsyncMock),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
            ),
            patch.object(
                extractor,
                "_read_profile_display_name",
                new_callable=AsyncMock,
                return_value="Test User",
            ),
            patch.object(
                extractor,
                "_resolve_message_compose_href",
                new_callable=AsyncMock,
                return_value="https://www.linkedin.com/messaging/compose/?recipient=ACoAAB",
            ),
            patch.object(
                extractor,
                "_wait_for_message_surface",
                new_callable=AsyncMock,
                return_value="composer",
            ),
            patch.object(
                extractor,
                "_resolve_message_compose_box",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch.object(
                extractor,
                "_compose_page_matches_recipient",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                extractor,
                "_dismiss_message_ui",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.send_message(
                "testuser", "Hello!", confirm_send=False
            )

        assert result["status"] == "confirmation_required"
        assert result["sent"] is False

    async def test_message_unavailable_when_no_compose_href(self, mock_page):
        """send_message returns message_unavailable when no compose URL found."""
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(extractor, "_navigate_to_page", new_callable=AsyncMock),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
            ),
            patch.object(
                extractor,
                "_read_profile_display_name",
                new_callable=AsyncMock,
                return_value="Test User",
            ),
            patch.object(
                extractor,
                "_resolve_message_compose_href",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await extractor.send_message(
                "testuser", "Hello!", confirm_send=True
            )

        assert result["status"] == "message_unavailable"
        assert result["sent"] is False

    async def test_uses_profile_urn_when_provided(self, mock_page):
        """send_message builds compose URL from profile_urn without Message-button lookup."""
        extractor = LinkedInExtractor(mock_page)
        with (
            patch.object(extractor, "_navigate_to_page", new_callable=AsyncMock),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
            ),
            patch.object(
                extractor,
                "_read_profile_display_name",
                new_callable=AsyncMock,
                return_value="Test User",
            ),
            patch.object(
                extractor,
                "_resolve_message_compose_href",
                new_callable=AsyncMock,
                return_value=None,
            ) as mock_resolve_href,
            patch.object(
                extractor,
                "_wait_for_message_surface",
                new_callable=AsyncMock,
                return_value="composer",
            ),
            patch.object(
                extractor,
                "_resolve_message_compose_box",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch.object(
                extractor,
                "_compose_page_matches_recipient",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                extractor,
                "_dismiss_message_ui",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.send_message(
                "testuser",
                "Hello!",
                confirm_send=False,
                profile_urn="ACoAAB1IelEB",
            )

        # _resolve_message_compose_href should NOT be called when profile_urn given
        mock_resolve_href.assert_not_awaited()
        assert result["status"] == "confirmation_required"

    async def test_profile_urn_compose_url_includes_full_params(self, mock_page):
        """send_message with profile_urn builds URL with profileUrn, screenContext, interop."""
        extractor = LinkedInExtractor(mock_page)
        navigate_calls = []

        async def capture_navigate(url):
            navigate_calls.append(url)

        with (
            patch.object(
                extractor,
                "_navigate_to_page",
                new_callable=AsyncMock,
                side_effect=capture_navigate,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
            ),
            patch.object(
                extractor,
                "_read_profile_display_name",
                new_callable=AsyncMock,
                return_value="Test User",
            ),
            patch.object(
                extractor,
                "_wait_for_message_surface",
                new_callable=AsyncMock,
                return_value="composer",
            ),
            patch.object(
                extractor,
                "_resolve_message_compose_box",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch.object(
                extractor,
                "_compose_page_matches_recipient",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                extractor,
                "_dismiss_message_ui",
                new_callable=AsyncMock,
            ),
        ):
            await extractor.send_message(
                "testuser",
                "Hello!",
                confirm_send=False,
                profile_urn="ACoAAB1IelEB",
            )

        # Second navigate call is the compose URL (first is the profile page)
        compose_url = navigate_calls[1]
        assert "profileUrn=" in compose_url
        assert "urn%3Ali%3Afsd_profile%3AACoAAB1IelEB" in compose_url
        assert "recipient=ACoAAB1IelEB" in compose_url
        assert "screenContext=NON_SELF_PROFILE_VIEW" in compose_url
        assert "interop=msgOverlay" in compose_url

    async def test_uses_invitation_compose_url_when_provided(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        invitation_url = (
            "/messaging/compose/?recipient=ACoAAB"
            "&invitation=urn%3Ali%3Afsd_invitation%3A123"
        )

        with (
            patch.object(extractor, "_navigate_to_page", new_callable=AsyncMock),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
            ),
            patch.object(
                extractor,
                "_read_profile_display_name",
                new_callable=AsyncMock,
                return_value="Test User",
            ),
            patch.object(
                extractor,
                "_open_invitation_message_compose",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_open_invitation_compose,
            patch.object(
                extractor,
                "_resolve_message_compose_href",
                new_callable=AsyncMock,
            ) as mock_resolve_href,
            patch.object(
                extractor,
                "_wait_for_message_surface",
                new_callable=AsyncMock,
                return_value="composer",
            ),
            patch.object(
                extractor,
                "_resolve_message_compose_box",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch.object(
                extractor,
                "_compose_page_matches_recipient",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                extractor,
                "_dismiss_message_ui",
                new_callable=AsyncMock,
            ),
        ):
            result = await extractor.send_message(
                "testuser",
                "Hello!",
                confirm_send=False,
                profile_urn="ignored",
                compose_url=invitation_url,
            )

        assert result["status"] == "confirmation_required"
        mock_open_invitation_compose.assert_awaited_once_with(
            "testuser",
            invitation_url,
        )
        mock_resolve_href.assert_not_awaited()

    async def test_opens_matching_invitation_message_modal(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        invitation_url = (
            "/messaging/compose/?recipient=ACoAAB"
            "&invitation=urn%3Ali%3Afsd_invitation%3A123"
        )
        message_link = MagicMock()
        message_link.click = AsyncMock()
        message_links = MagicMock()
        message_links.nth.return_value = message_link
        mock_page.locator = MagicMock(return_value=message_links)
        mock_page.evaluate = AsyncMock(return_value=0)
        mock_page.wait_for_timeout = AsyncMock()

        with (
            patch.object(extractor, "_navigate_to_page", new_callable=AsyncMock),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch.object(
                extractor,
                "_wait_for_main_text",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
            ),
            patch.object(
                extractor,
                "_extract_invitation_cards",
                new_callable=AsyncMock,
                return_value=[
                    {
                        "type": "connection_request",
                        "sender": {"url": "/in/testuser/"},
                        "message_url": invitation_url,
                    }
                ],
            ),
        ):
            opened = await extractor._open_invitation_message_compose(
                "testuser",
                invitation_url,
            )

        assert opened is True
        message_links.nth.assert_called_once_with(0)
        message_link.click.assert_awaited_once()

    async def test_rejects_invitation_url_for_different_sender(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        invitation_url = "/messaging/compose/?recipient=ACoAAB&invitation=urn"
        mock_page.evaluate = AsyncMock()
        mock_page.wait_for_timeout = AsyncMock()

        with (
            patch.object(extractor, "_navigate_to_page", new_callable=AsyncMock),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch.object(
                extractor,
                "_wait_for_main_text",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
            ),
            patch.object(
                extractor,
                "_extract_invitation_cards",
                new_callable=AsyncMock,
                return_value=[
                    {
                        "type": "connection_request",
                        "sender": {"url": "/in/someone-else/"},
                        "message_url": invitation_url,
                    }
                ],
            ),
            patch.object(
                extractor,
                "_scroll_invitation_manager_down",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            opened = await extractor._open_invitation_message_compose(
                "testuser",
                invitation_url,
            )

        assert opened is False
        mock_page.evaluate.assert_not_awaited()

    @pytest.mark.parametrize(
        "compose_url",
        [
            "https://example.com/messaging/compose/?recipient=ACoAAB",
            "/in/testuser/",
            "/messaging/compose/",
        ],
    )
    async def test_rejects_invalid_invitation_compose_url(self, mock_page, compose_url):
        extractor = LinkedInExtractor(mock_page)
        with patch.object(
            extractor,
            "_navigate_to_page",
            new_callable=AsyncMock,
        ) as mock_navigate:
            result = await extractor.send_message(
                "testuser",
                "Hello!",
                confirm_send=True,
                compose_url=compose_url,
            )

        assert result["status"] == "message_unavailable"
        assert result["sent"] is False
        mock_navigate.assert_not_awaited()


class TestResolveMessageComposeBox:
    async def test_returns_locator_when_count_positive(self, mock_page):
        """_resolve_message_compose_box returns locator.last when count() > 0."""
        extractor = LinkedInExtractor(mock_page)
        mock_locator = MagicMock()
        mock_locator.count = AsyncMock(return_value=1)
        sentinel = MagicMock(name="last_locator")
        sentinel.wait_for = AsyncMock()
        mock_locator.last = sentinel
        mock_locator.wait_for = AsyncMock()
        mock_page.locator = MagicMock(return_value=mock_locator)

        result = await extractor._resolve_message_compose_box()

        assert result is sentinel
        # wait_for should NOT be called on the early-return path
        sentinel.wait_for.assert_not_called()
        mock_locator.wait_for.assert_not_called()

    async def test_returns_none_when_all_selectors_miss(self, mock_page):
        """_resolve_message_compose_box returns None when no selector matches."""
        from patchright.async_api import TimeoutError as PlaywrightTimeoutError

        extractor = LinkedInExtractor(mock_page)
        mock_locator = MagicMock()
        mock_locator.count = AsyncMock(return_value=0)
        mock_locator.last = MagicMock()
        mock_locator.last.wait_for = AsyncMock(
            side_effect=PlaywrightTimeoutError("timeout")
        )
        mock_page.locator = MagicMock(return_value=mock_locator)

        result = await extractor._resolve_message_compose_box()

        assert result is None

    async def test_falls_through_when_count_raises(self, mock_page):
        """_resolve_message_compose_box handles count() exceptions gracefully."""
        from patchright.async_api import TimeoutError as PlaywrightTimeoutError

        extractor = LinkedInExtractor(mock_page)
        mock_locator = MagicMock()
        mock_locator.count = AsyncMock(side_effect=Exception("detached"))
        mock_locator.last = MagicMock()
        mock_locator.last.wait_for = AsyncMock(
            side_effect=PlaywrightTimeoutError("timeout")
        )
        mock_page.locator = MagicMock(return_value=mock_locator)

        result = await extractor._resolve_message_compose_box()

        assert result is None


class TestSendMessageComposerInteraction:
    """Tests for the page.evaluate + keyboard.type send path (patchright workaround)."""

    def _patch_send_message_to_compose(self, extractor, mock_page):
        """Return a context manager that patches send_message up to the compose step."""
        self.compose_box = MagicMock()
        self.compose_box.evaluate = AsyncMock(side_effect=[True, True])
        return (
            patch.object(extractor, "_navigate_to_page", new_callable=AsyncMock),
            patch(
                "linkedin_mcp_server.scraping.extractor.detect_rate_limit",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.handle_modal_close",
                new_callable=AsyncMock,
            ),
            patch.object(
                extractor,
                "_read_profile_display_name",
                new_callable=AsyncMock,
                return_value="Test User",
            ),
            patch.object(
                extractor,
                "_resolve_message_compose_href",
                new_callable=AsyncMock,
                return_value="https://www.linkedin.com/messaging/compose/?recipient=ACoAAB",
            ),
            patch.object(
                extractor,
                "_wait_for_message_surface",
                new_callable=AsyncMock,
                return_value="composer",
            ),
            patch.object(
                extractor,
                "_resolve_message_compose_box",
                new_callable=AsyncMock,
                return_value=self.compose_box,
            ),
            patch.object(
                extractor,
                "_compose_page_matches_recipient",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch.object(
                extractor,
                "_dismiss_message_ui",
                new_callable=AsyncMock,
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        )

    async def test_focus_and_type_via_evaluate_and_keyboard(self, mock_page):
        """send_message uses page.evaluate to focus and page.keyboard.type to type."""
        extractor = LinkedInExtractor(mock_page)
        mock_keyboard = MagicMock()
        mock_keyboard.type = AsyncMock()
        mock_keyboard.press = AsyncMock()
        mock_page.keyboard = mock_keyboard
        mock_page.evaluate = AsyncMock(return_value=True)
        patches = self._patch_send_message_to_compose(extractor, mock_page)

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
            patches[9],
            patch.object(
                extractor,
                "_message_text_visible",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = await extractor.send_message(
                "testuser", "Hello!", confirm_send=True
            )

        assert result["status"] == "sent"
        assert result["sent"] is True
        assert self.compose_box.evaluate.await_count == 2
        # Verify keyboard.type was used (not press_sequentially)
        mock_keyboard.type.assert_awaited_once_with("Hello!", delay=15)

    async def test_compose_interact_failed_when_focus_fails(self, mock_page):
        """send_message returns compose_interact_failed when JS focus fails."""
        extractor = LinkedInExtractor(mock_page)
        mock_keyboard = MagicMock()
        mock_keyboard.type = AsyncMock()
        mock_page.keyboard = mock_keyboard
        patches = self._patch_send_message_to_compose(extractor, mock_page)
        self.compose_box.evaluate.side_effect = None
        self.compose_box.evaluate.return_value = False

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
            patches[9],
        ):
            result = await extractor.send_message(
                "testuser", "Hello!", confirm_send=True
            )

        assert result["status"] == "compose_interact_failed"
        assert result["sent"] is False

    async def test_enter_fallback_when_send_button_not_found(self, mock_page):
        """send_message falls back to Enter key when JS cannot find send button."""
        extractor = LinkedInExtractor(mock_page)
        mock_keyboard = MagicMock()
        mock_keyboard.type = AsyncMock()
        mock_keyboard.press = AsyncMock()
        mock_page.keyboard = mock_keyboard
        patches = self._patch_send_message_to_compose(extractor, mock_page)
        self.compose_box.evaluate.side_effect = [True, False]

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            patches[8],
            patches[9],
            patch.object(
                extractor,
                "_message_text_visible",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = await extractor.send_message(
                "testuser", "Hello!", confirm_send=True
            )

        assert result["status"] == "sent"
        # Enter was pressed as fallback
        mock_keyboard.press.assert_awaited_once_with("Enter")

    async def test_sent_message_verification_uses_resolved_composer(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        compose_box = MagicMock()
        compose_box.evaluate = AsyncMock(side_effect=[RuntimeError, True])

        with patch(
            "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            visible = await extractor._message_text_visible(
                "Hello!",
                compose_box,
            )

        assert visible is True
        assert compose_box.evaluate.await_count == 2
        verification_script = compose_box.evaluate.await_args_list[0].args[0]
        assert "editor.closest('[role=\"dialog\"]')" in verification_script
        assert "editor.getRootNode()" not in verification_script

    async def test_recipient_verification_uses_resolved_composer(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        compose_box = MagicMock()
        compose_box.evaluate = AsyncMock(return_value=True)

        matched = await extractor._compose_page_matches_recipient(
            compose_box,
            "Test User",
        )

        assert matched is True
        verification_script = compose_box.evaluate.await_args_list[0].args[0]
        assert "editor.closest('[role=\"dialog\"]')" in verification_script
        assert "document.querySelector('main')" not in verification_script

    async def test_waits_for_recipient_composer_to_hydrate(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        wrong_compose_box = MagicMock()
        recipient_compose_box = MagicMock()

        with (
            patch.object(
                extractor,
                "_resolve_message_compose_box",
                new_callable=AsyncMock,
                side_effect=[wrong_compose_box, recipient_compose_box],
            ),
            patch.object(
                extractor,
                "_compose_page_matches_recipient",
                new_callable=AsyncMock,
                side_effect=[False, True],
            ),
            patch(
                "linkedin_mcp_server.scraping.extractor.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            resolved = await extractor._resolve_recipient_message_compose_box(
                "Test User"
            )

        assert resolved is recipient_compose_box


class TestBuildFeedReferences:
    """Tests for _build_feed_references SDUI-capture / DOM-anchor merging."""

    def test_sdui_urls_become_relative_feed_post_references(self):
        captured = [
            "https://www.linkedin.com/posts/alice_some-slug-ugcPost-1-xx",
            "https://www.linkedin.com/posts/bob_other-post-share-2-yy",
        ]
        refs = _build_feed_references([], captured)
        assert refs == [
            {
                "kind": "feed_post",
                "url": "/posts/alice_some-slug-ugcPost-1-xx",
                "context": "feed",
            },
            {
                "kind": "feed_post",
                "url": "/posts/bob_other-post-share-2-yy",
                "context": "feed",
            },
        ]

    def test_duplicate_sdui_urls_are_deduped(self):
        captured = [
            "https://www.linkedin.com/posts/alice_x-ugcPost-1-xx",
            "https://www.linkedin.com/posts/alice_x-ugcPost-1-xx",
        ]
        refs = _build_feed_references([], captured)
        assert len(refs) == 1
        assert refs[0]["url"] == "/posts/alice_x-ugcPost-1-xx"

    def test_dom_anchor_feed_update_passes_through(self):
        # DOM anchors that classify_link recognises as feed_post survive
        # the merge alongside SDUI captures.
        raw_anchors = [
            {
                "href": "https://www.linkedin.com/feed/update/urn:li:activity:1234567890/",
                "text": "View post",
            }
        ]
        refs = _build_feed_references(raw_anchors, [])
        assert any(
            r["url"] == "/feed/update/urn:li:activity:1234567890/"
            and r["kind"] == "feed_post"
            for r in refs
        )

    def test_non_posts_paths_in_sdui_capture_are_skipped(self):
        # Defensive: only /posts/<slug> shapes count for SDUI append.
        captured = [
            "https://www.linkedin.com/in/someuser/",
            "https://www.linkedin.com/posts/alice_x-ugcPost-1-xx",
        ]
        refs = _build_feed_references([], captured)
        assert [r["url"] for r in refs] == ["/posts/alice_x-ugcPost-1-xx"]

    def test_cap_matches_num_posts_ceiling(self):
        captured = [
            f"https://www.linkedin.com/posts/p{i}-ugcPost-{i}-xx" for i in range(60)
        ]
        refs = _build_feed_references([], captured)
        # Cap is 50, mirroring _REFERENCE_CAPS["feed"] / num_posts <= 50.
        assert len(refs) == 50

    def test_non_feed_post_dom_anchors_are_filtered(self):
        # Sidebar profile / company / external anchors must not crowd
        # out SDUI permalinks — references["feed"] is feed_post-only.
        raw_anchors = [
            {
                "href": "https://www.linkedin.com/in/sidebar-user/",
                "text": "Sidebar User",
            },
            {
                "href": "https://www.linkedin.com/company/some-corp/",
                "text": "Some Corp",
            },
            {
                "href": "https://example.com/external/",
                "text": "External Link",
            },
        ]
        refs = _build_feed_references(raw_anchors, [])
        assert refs == []

    def test_feed_post_dom_anchors_coexist_with_sdui_captures(self):
        # The two sources fold into the same feed_post kind without
        # collapsing across URL shapes pointing at the same post.
        raw_anchors = [
            {
                "href": "https://www.linkedin.com/feed/update/urn:li:activity:111/",
                "text": "View post",
            }
        ]
        captured = ["https://www.linkedin.com/posts/alice_x-ugcPost-1-xx"]
        refs = _build_feed_references(raw_anchors, captured)
        urls = [r["url"] for r in refs]
        kinds = {r["kind"] for r in refs}
        assert urls == [
            "/feed/update/urn:li:activity:111/",
            "/posts/alice_x-ugcPost-1-xx",
        ]
        assert kinds == {"feed_post"}


class TestConnectionList:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Connected on May 25, 2026", "2026-05-25"),
            ("Connected on January 1, 2020", "2020-01-01"),
            ("Connected on Dec 31, 1999", "1999-12-31"),
            ("connected on jul 4, 2024", "2024-07-04"),
            ("Some prefix · Connected on Feb 9, 2024 · suffix", "2024-02-09"),
        ],
    )
    def test_parse_connected_on_en_us(self, text, expected):
        assert _parse_connected_on(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            None,
            "",
            "   ",
            "Mise en relation le 25 mai 2026",
            "Connected on May 2026",
            "Connected on Mai 25, 2026",
            "Connected on May 32, 2026",
            # Impossible date — caught by ``datetime.date`` validation.
            "Connected on Feb 30, 2024",
            "Connected on Apr 31, 2024",
        ],
    )
    def test_parse_connected_on_returns_none_for_unparseable(self, text):
        assert _parse_connected_on(text) is None

    def test_normalize_connection_full_record(self):
        raw = {
            "name": "  Rob Choy  ",
            "profile_url": "/in/robchoy/",
            "headline": "Founder & investor",
            "connected_on_text": "Connected on May 25, 2026",
        }
        assert _normalize_connection(raw) == {
            "name": "Rob Choy",
            "url": "/in/robchoy/",
            "headline": "Founder & investor",
            "connected_on": "2026-05-25",
        }

    def test_normalize_connection_absolute_url_is_normalized(self):
        raw = {
            "name": "Santiago Moreno",
            "profile_url": "https://www.linkedin.com/in/santiago-moreno-7098138b?miniProfileUrn=urn:li:fsd_profile:XYZ",
            "headline": "Regional Operations Manager - RWE Renewables France",
            "connected_on_text": "Connected on May 24, 2026",
        }
        result = _normalize_connection(raw)
        assert result is not None
        assert result["url"] == "/in/santiago-moreno-7098138b/"
        assert result["connected_on"] == "2026-05-24"

    def test_normalize_connection_drops_record_without_profile_url(self):
        assert _normalize_connection({"name": "Anon", "profile_url": None}) is None
        assert _normalize_connection({"name": "Anon"}) is None
        assert _normalize_connection("not-a-dict") is None

    def test_normalize_connection_surfaces_none_for_unparseable_date(self):
        raw = {
            "name": "Foreign Friend",
            "profile_url": "/in/foreign/",
            "headline": "Headline",
            "connected_on_text": "Mise en relation le 24 mai 2026",
        }
        result = _normalize_connection(raw)
        assert result is not None
        assert result["connected_on"] is None

    def test_connection_identity_key_dedupes_on_url(self):
        a = {"url": "/in/robchoy/", "name": "Rob Choy"}
        b = {"url": "/in/robchoy/", "name": "Robert Choy"}
        c = {"url": "/in/santiago-moreno-7098138b/", "name": "Santiago Moreno"}
        assert _connection_identity_key(a) == _connection_identity_key(b)
        assert _connection_identity_key(a) != _connection_identity_key(c)

    def test_connection_cards_script_uses_locale_independent_signals(self):
        # Per AGENTS.md scraping rules: anchor-href URL patterns, not class
        # names; minimal selectors; innerText-driven line classification.
        assert 'a[href*="/in/"]' in _CONNECTION_CARDS_JS
        assert "profileSlug" in _CONNECTION_CARDS_JS
        assert "linesFrom" in _CONNECTION_CARDS_JS
        assert "connected_on_text" in _CONNECTION_CARDS_JS
        # V1 contract: only rows whose ancestor exposes the en-US
        # "Connected on/since" line are surfaced — the regex must be
        # present in the JS source.
        assert "connectedOnRe" in _CONNECTION_CARDS_JS
        assert r"connected\s+(?:on|since)\b" in _CONNECTION_CARDS_JS
        # Headline / date selection must not depend on class names.
        assert "className" not in _CONNECTION_CARDS_JS
        assert "[class" not in _CONNECTION_CARDS_JS


class TestExtractFeedPosts:
    """The JS -> Python boundary of the feed post extractor.

    Covers the failure surface (``page.evaluate`` raising or returning a
    non-list) and the valid-post limit, which the JS extractor and
    ``_normalize_feed_post`` cannot exercise on their own.
    """

    def _raw_post(self, name: str = "Alice") -> dict:
        return {
            "url": "/feed/update/urn:li:activity:1/",
            "post_age": "5h",
            "author": {
                "name": name,
                "profile_url": "/in/alice/",
                "headline": "Engineer",
                "degree": "1st",
            },
            "content": "hello",
            "is_promoted": False,
            "media": None,
            "reactions_count": 2,
            "comment_count": 0,
            "repost_count": 0,
        }

    def test_fallback_uses_locale_independent_action_structure(self):
        assert "button[aria-label]" in _FEED_POSTS_JS
        assert "aria-label*=" not in _FEED_POSTS_JS

    async def test_evaluate_exception_propagates(self, mock_page):
        mock_page.evaluate = AsyncMock(side_effect=RuntimeError("boom"))
        extractor = LinkedInExtractor(mock_page)
        with pytest.raises(RuntimeError, match="boom"):
            await extractor._extract_feed_posts(10)

    async def test_non_list_result_raises(self, mock_page):
        mock_page.evaluate = AsyncMock(return_value={"not": "a list"})
        extractor = LinkedInExtractor(mock_page)
        with pytest.raises(TypeError, match="non-list"):
            await extractor._extract_feed_posts(10)

    async def test_respects_limit(self, mock_page):
        raws = [self._raw_post(f"P{i}") for i in range(5)]
        mock_page.evaluate = AsyncMock(return_value=raws)
        extractor = LinkedInExtractor(mock_page)
        posts = await extractor._extract_feed_posts(2)
        assert [p["author"]["name"] for p in posts] == ["P0", "P1"]

    async def test_chrome_cards_not_counted_toward_limit(self, mock_page):
        """Cards that normalize to None must not consume the limit budget."""
        chrome = {"url": None, "content": None, "author": {"name": None}}
        raws = [
            chrome,
            self._raw_post("A"),
            chrome,
            self._raw_post("B"),
            self._raw_post("C"),
        ]
        mock_page.evaluate = AsyncMock(return_value=raws)
        extractor = LinkedInExtractor(mock_page)
        posts = await extractor._extract_feed_posts(2)
        assert [p["author"]["name"] for p in posts] == ["A", "B"]


class TestProxyNavigationFailures:
    """A proxy outage during an ordinary tool call is reported as itself."""

    async def test_proxy_error_is_raised_instead_of_a_scraping_failure(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        mock_page.goto = AsyncMock(
            side_effect=Exception("net::ERR_PROXY_CONNECTION_FAILED at …")
        )

        with pytest.raises(ProxyConnectionError):
            await extractor._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

    async def test_proxy_error_is_converted_before_it_reaches_a_trace(self, mock_page):
        # The trace records the raw exception text, which for a proxy failure
        # can quote the proxy URL and put a password into trace.jsonl.
        extractor = LinkedInExtractor(mock_page)
        mock_page.goto = AsyncMock(
            side_effect=Exception("net::ERR_TUNNEL_CONNECTION_FAILED")
        )

        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.record_page_trace",
                new_callable=AsyncMock,
            ) as mock_trace,
            pytest.raises(ProxyConnectionError),
        ):
            await extractor._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        recorded = [call.args[1] for call in mock_trace.await_args_list]
        assert "extractor-navigation-error" not in recorded

    async def test_ordinary_navigation_failure_is_unaffected(self, mock_page):
        extractor = LinkedInExtractor(mock_page)
        mock_page.goto = AsyncMock(side_effect=Exception("net::ERR_ABORTED"))

        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.resolve_remember_me_prompt",
                new_callable=AsyncMock,
                return_value=False,
            ),
            pytest.raises(Exception) as excinfo,
        ):
            await extractor._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        assert not isinstance(excinfo.value, ProxyConnectionError)


class TestNavigationFailureLogRedaction:
    """The navigation-failure log must not carry proxy credentials.

    It reaches the log even for errors the marker check does not recognise as
    proxy faults, and that log is what users paste into issue reports.
    """

    async def test_credentials_are_redacted_from_the_log(
        self, mock_page, monkeypatch, caplog
    ):
        import logging

        from linkedin_mcp_server.config.schema import AppConfig

        config = AppConfig()
        config.browser.proxy_server = "http://gate.example:7000"
        config.browser.proxy_username = "acctzone9"
        config.browser.proxy_password = "s3cr3t"
        monkeypatch.setattr("linkedin_mcp_server.config.get_config", lambda: config)

        extractor = LinkedInExtractor(mock_page)
        # No proxy marker, so it is not converted and reaches the logger.
        mock_page.goto = AsyncMock(
            side_effect=Exception(
                "failed via http://acctzone9:s3cr3t@gate.example:7000"
            )
        )

        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.resolve_remember_me_prompt",
                new_callable=AsyncMock,
                return_value=False,
            ),
            caplog.at_level(logging.WARNING),
            pytest.raises(Exception),
        ):
            await extractor._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        assert "s3cr3t" not in caplog.text
        assert "acctzone9" not in caplog.text


class TestNavigationFailureCrossesTheToolBoundaryClean:
    """The re-raised exception itself must be credential-free.

    Redacting the extractor's own trace and log is not enough: everything
    downstream logs the exception too, starting with the catch-all in
    error_handler and FastMCP's handler above it.
    """

    async def test_reraised_exception_carries_no_credentials(
        self, mock_page, monkeypatch
    ):
        from linkedin_mcp_server.config.schema import AppConfig

        config = AppConfig()
        config.browser.proxy_server = "http://gate.example:7000"
        config.browser.proxy_username = "acctzone9"
        config.browser.proxy_password = "s3cr3t"
        monkeypatch.setattr("linkedin_mcp_server.config.get_config", lambda: config)

        extractor = LinkedInExtractor(mock_page)
        mock_page.goto = AsyncMock(
            side_effect=Exception(
                "failed via http://acctzone9:s3cr3t@gate.example:7000"
            )
        )

        with (
            patch(
                "linkedin_mcp_server.scraping.extractor.resolve_remember_me_prompt",
                new_callable=AsyncMock,
                return_value=False,
            ),
            pytest.raises(Exception) as excinfo,
        ):
            await extractor._goto_with_auth_checks(
                "https://www.linkedin.com/in/testuser/"
            )

        assert "s3cr3t" not in str(excinfo.value)
        assert "acctzone9" not in str(excinfo.value)
        # The raw error must not survive as a cause either: the handlers
        # downstream print the whole chain.
        assert excinfo.value.__cause__ is None
