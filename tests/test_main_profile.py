"""Tests for the structured ``main_profile`` parser.

Mirrors ``tests/test_messaging_extractor.py`` in spirit: unit-test the
Python normalization functions against hand-built raw-extraction dicts,
plus a single mocked ``extract_main_profile`` integration test that
patches ``page.evaluate`` to return the same fixtures.

The fixture below is shaped to match the expected output for
``linkedin.com/in/alexandrelebrun`` so a regression in any of the
normalizers will surface here (5 experiences / 2 educations,
``connection_count = None`` from "500+", ``follower_count = 48516``,
``mutual_connection_count = 37``).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from linkedin_mcp_server.scraping.main_profile import (
    extract_main_profile,
    normalize_main_profile,
    parse_connection_count,
    parse_education_entry,
    parse_experience_entry,
    parse_follower_count,
    parse_mutual_connection_count,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _alexandrelebrun_raw() -> dict[str, Any]:
    """Hand-built raw extraction dict resembling alexandrelebrun's profile.

    Mirrors the new ``_EXTRACT_SCRIPT`` shape (verified live 2026-05-26):
    LinkedIn no longer renders ``<h1>`` for the name and dropped the
    ``<div id="about|experience|education">`` anchor divs, so the
    extractor returns only ``name``, ``top_card_text``, ``mutual_text``,
    ``profile_picture_url``, ``about``, ``experience_raw``, and
    ``education_raw``. Headline, location, main_organization, and
    main_education are derived in Python from ``top_card_text`` line
    walking.
    """
    return {
        "name": "Alex LeBrun",
        "profile_picture_url": (
            "https://media.licdn.com/dms/image/v2/D4E03AQE0xheLWQ1z9g/"
            "profile-displayphoto-shrink_800_800/B4EZPZMFzrHkAg-/0/"
            "1734515646135?e=1781136000&v=beta&t=abc"
        ),
        "top_card_text": (
            "Alex LeBrun\n"
            "· 2nd\n"
            "Building at AMI & Nabla\n"
            "Paris, Ile-de-France, France\n"
            "·\n"
            "Contact info\n"
            "AMI - Advanced Machine Intelligence\n"
            "Ecole polytechnique\n"
            "48,516 followers\n"
            "500+ connections\n"
            "John Doe, Jane Smith, and 35 other mutual connections\n"
            "Follow\n"
            "Connect"
        ),
        "mutual_text": "John Doe, Jane Smith, and 35 other mutual connections",
        "about": (
            "Cofounder of\n"
            "☛ Nabla (2019, AI assistant supporting 85,000 physicians)\n"
            "☛ Wit.ai (2013, YC W14, first AI startup with a .𝗮𝗶 domain 😎 "
            "acquired by Facebook)"
        ),
        "experience_raw": [
            {
                "text": (
                    "Cofounder\n"
                    "AMI - Advanced Machine Intelligence · Full-time\n"
                    "Jan 2025 - Present · 11 mos\n"
                    "Paris, France · Hybrid\n"
                    "Building the next-gen AI assistant."
                ),
                "anchor_href": "/company/ami-ai/",
            },
            {
                "text": (
                    "Cofounder\n"
                    "Nabla · Full-time\n"
                    "2019 - 2024 · 5 yrs\n"
                    "Paris, France\n"
                    "AI assistant for physicians."
                ),
                "anchor_href": "/company/nabla/",
            },
            {
                "text": (
                    "Director, AI Research\n"
                    "Meta · Full-time\n"
                    "2015 - 2019 · 4 yrs\n"
                    "Menlo Park, California"
                ),
                "anchor_href": "/company/meta/",
            },
            {
                "text": (
                    "Cofounder & CEO\n"
                    "Wit.ai (acquired by Facebook) · Full-time\n"
                    "2013 - 2015 · 2 yrs\n"
                    "Palo Alto, California"
                ),
                "anchor_href": "/company/wit-ai/",
            },
            {
                "text": (
                    "Founder & CEO\n"
                    "VirtuOz · Full-time\n"
                    "2002 - 2013 · 11 yrs\n"
                    "Paris Area, France"
                ),
                "anchor_href": "/company/virtuoz/",
            },
        ],
        "education_raw": [
            {
                "text": (
                    "Ecole polytechnique\n"
                    "Engineering degree, Mathematics and Computer Science\n"
                    "1994 - 1997"
                ),
                "anchor_href": "/school/ecole-polytechnique/",
            },
            {
                "text": (
                    "Lycée Louis-le-Grand\nClasses préparatoires, MP\n1992 - 1994"
                ),
                "anchor_href": "/school/lycee-louis-le-grand/",
            },
        ],
    }


# ---------------------------------------------------------------------------
# parse_connection_count
# ---------------------------------------------------------------------------


class TestParseConnectionCount:
    def test_returns_none_for_plus_threshold(self) -> None:
        # LinkedIn's "500+" is a privacy threshold, not an exact count.
        assert parse_connection_count("500+ connections") is None

    def test_returns_int_for_exact_count(self) -> None:
        assert parse_connection_count("237 connections") == 237

    def test_handles_comma_separated_thousands(self) -> None:
        assert parse_connection_count("1,234 connections") == 1234

    def test_returns_none_for_no_match(self) -> None:
        assert parse_connection_count("48,516 followers") is None

    def test_returns_none_for_empty_input(self) -> None:
        assert parse_connection_count("") is None
        assert parse_connection_count(None) is None

    def test_singular_connection(self) -> None:
        assert parse_connection_count("1 connection") == 1


# ---------------------------------------------------------------------------
# parse_follower_count
# ---------------------------------------------------------------------------


class TestParseFollowerCount:
    def test_parses_comma_separated_thousands(self) -> None:
        assert parse_follower_count("48,516 followers") == 48516

    def test_parses_small_count(self) -> None:
        assert parse_follower_count("12 followers") == 12

    def test_returns_none_when_absent(self) -> None:
        assert parse_follower_count("500+ connections") is None
        assert parse_follower_count(None) is None


# ---------------------------------------------------------------------------
# parse_mutual_connection_count
# ---------------------------------------------------------------------------


class TestParseMutualConnectionCount:
    def test_two_named_plus_other(self) -> None:
        """alexandrelebrun's expected case: 35 + 2 named = 37."""
        text = "John Doe, Jane Smith, and 35 other mutual connections"
        assert parse_mutual_connection_count(text) == 37

    def test_one_named_plus_other(self) -> None:
        assert (
            parse_mutual_connection_count("John Doe and 16 other mutual connections")
            == 17
        )

    def test_singleton(self) -> None:
        assert parse_mutual_connection_count("Jane Doe is a mutual connection") == 1

    def test_exact_two_plural(self) -> None:
        assert (
            parse_mutual_connection_count(
                "John Doe and Jane Smith are mutual connections"
            )
            == 2
        )

    def test_returns_none_for_no_match(self) -> None:
        assert parse_mutual_connection_count("48,516 followers") is None
        assert parse_mutual_connection_count(None) is None
        assert parse_mutual_connection_count("") is None


# ---------------------------------------------------------------------------
# parse_experience_entry
# ---------------------------------------------------------------------------


class TestParseExperienceEntry:
    def test_full_entry(self) -> None:
        raw = {
            "text": (
                "Cofounder\n"
                "AMI - Advanced Machine Intelligence · Full-time\n"
                "Jan 2025 - Present · 11 mos\n"
                "Paris, France · Hybrid\n"
                "Building the next-gen AI assistant."
            ),
            "anchor_href": "/company/ami-ai/",
        }
        entry = parse_experience_entry(raw)
        assert entry is not None
        assert entry["title"] == "Cofounder"
        assert entry["organization"] == "AMI - Advanced Machine Intelligence"
        assert entry["organization_url"] == "/company/ami-ai/"
        assert entry["dates"] == "Jan 2025 - Present"
        assert entry["location"] == "Paris, France"
        assert entry["description"] == "Building the next-gen AI assistant."

    def test_no_description(self) -> None:
        raw = {
            "text": (
                "Director, AI Research\n"
                "Meta · Full-time\n"
                "2015 - 2019 · 4 yrs\n"
                "Menlo Park, California"
            ),
            "anchor_href": "/company/meta/",
        }
        entry = parse_experience_entry(raw)
        assert entry is not None
        assert entry["title"] == "Director, AI Research"
        assert entry["organization"] == "Meta"
        assert entry["dates"] == "2015 - 2019"
        assert entry["location"] == "Menlo Park, California"
        assert "description" not in entry

    def test_dedupes_adjacent_duplicates(self) -> None:
        # LinkedIn often pairs visible text with visually-hidden duplicates.
        raw = {
            "text": ("Cofounder\nCofounder\nNabla\nNabla\n2019 - 2024\n"),
            "anchor_href": "/company/nabla/",
        }
        entry = parse_experience_entry(raw)
        assert entry is not None
        assert entry["title"] == "Cofounder"
        assert entry["organization"] == "Nabla"
        assert entry["dates"] == "2019 - 2024"

    def test_returns_none_for_empty_text(self) -> None:
        assert parse_experience_entry({"text": "", "anchor_href": None}) is None
        assert parse_experience_entry({"text": "   ", "anchor_href": None}) is None


# ---------------------------------------------------------------------------
# parse_education_entry
# ---------------------------------------------------------------------------


class TestParseEducationEntry:
    def test_full_entry(self) -> None:
        raw = {
            "text": (
                "Ecole polytechnique\n"
                "Engineering degree, Mathematics and Computer Science\n"
                "1994 - 1997"
            ),
            "anchor_href": "/school/ecole-polytechnique/",
        }
        entry = parse_education_entry(raw)
        assert entry is not None
        assert entry["school"] == "Ecole polytechnique"
        assert entry["school_url"] == "/school/ecole-polytechnique/"
        assert entry["degree"] == "Engineering degree"
        assert entry["field_of_study"] == "Mathematics and Computer Science"
        assert entry["dates"] == "1994 - 1997"

    def test_degree_only(self) -> None:
        raw = {
            "text": "Stanford University\nMaster of Science\n2010 - 2012",
            "anchor_href": None,
        }
        entry = parse_education_entry(raw)
        assert entry is not None
        assert entry["school"] == "Stanford University"
        assert entry["degree"] == "Master of Science"
        assert "field_of_study" not in entry
        assert entry["dates"] == "2010 - 2012"

    def test_minimum_school_only(self) -> None:
        raw = {"text": "Some University", "anchor_href": None}
        entry = parse_education_entry(raw)
        assert entry is not None
        assert entry["school"] == "Some University"
        assert "dates" not in entry


# ---------------------------------------------------------------------------
# normalize_main_profile
# ---------------------------------------------------------------------------


class TestNormalizeMainProfile:
    def test_alexandrelebrun_shape(self) -> None:
        result = normalize_main_profile(_alexandrelebrun_raw())

        assert result["name"] == "Alex LeBrun"
        assert result["headline"] == "Building at AMI & Nabla"
        assert result["location"] == "Paris, Ile-de-France, France"
        assert result["profile_picture_url"] is not None
        assert result["profile_picture_url"].startswith(
            "https://media.licdn.com/dms/image/"
        )
        # "500+ connections" is not an exact count.
        assert result["connection_count"] is None
        assert result["follower_count"] == 48516
        # "John, Jane, and 35 other" → 35 + 2 = 37
        assert result["mutual_connection_count"] == 37
        assert result["main_organization"] == "AMI - Advanced Machine Intelligence"
        assert result["main_education"] == "Ecole polytechnique"
        assert result["about"] is not None
        assert "Cofounder of" in result["about"]
        assert len(result["experience"]) == 5
        assert len(result["education"]) == 2

    def test_exact_connection_count(self) -> None:
        raw = _alexandrelebrun_raw()
        raw["top_card_text"] = "Alex LeBrun\n237 connections · 12 followers"
        result = normalize_main_profile(raw)
        assert result["connection_count"] == 237
        assert result["follower_count"] == 12

    def test_default_avatar_emits_none(self) -> None:
        """Profiles with the default avatar SVG must emit ``None``."""
        raw = _alexandrelebrun_raw()
        # JS extractor would emit None when no media.licdn.com displayphoto
        # <img src> is found; the normalizer should also defensively
        # reject anything that doesn't match the host filter.
        raw["profile_picture_url"] = None
        assert normalize_main_profile(raw)["profile_picture_url"] is None

        raw["profile_picture_url"] = "https://static.linkedin.com/some-avatar.svg"
        assert normalize_main_profile(raw)["profile_picture_url"] is None

    def test_all_keys_present_on_empty_input(self) -> None:
        """Even an empty raw dict yields a fully-populated MainProfile."""
        result = normalize_main_profile({})
        expected_keys = {
            "name",
            "headline",
            "location",
            "profile_picture_url",
            "connection_count",
            "follower_count",
            "mutual_connection_count",
            "main_organization",
            "main_education",
            "about",
            "experience",
            "education",
        }
        assert set(result.keys()) == expected_keys
        # Text fields default to None, lists to [].
        assert result["name"] is None
        assert result["experience"] == []
        assert result["education"] == []

    def test_none_input_yields_empty_shape(self) -> None:
        result = normalize_main_profile(None)
        assert result["name"] is None
        assert result["experience"] == []

    def test_skips_unparseable_entries(self) -> None:
        raw = _alexandrelebrun_raw()
        raw["experience_raw"] = [
            {"text": "", "anchor_href": None},
            {"text": "Valid Entry\nOrg\n2020 - Present", "anchor_href": None},
        ]
        result = normalize_main_profile(raw)
        assert len(result["experience"]) == 1
        assert result["experience"][0]["title"] == "Valid Entry"


# ---------------------------------------------------------------------------
# extract_main_profile (integration with mocked page.evaluate)
# ---------------------------------------------------------------------------


class TestExtractMainProfile:
    async def test_scroll_sweep_then_expand_then_extract(self) -> None:
        """The extractor runs an incremental scroll sweep, then a
        per-section scroll-into-view, then expand, then extract.
        """
        raw = _alexandrelebrun_raw()
        call_order: list[str] = []
        # Simulate the page reaching a stable scrollHeight after 3 steps
        # so the incremental sweep exits promptly. Each step returns
        # an increasing y until clamped.
        step_returns = [
            {"y": 400, "height": 2000, "h2_count": 1},
            {"y": 800, "height": 2000, "h2_count": 2},
            {"y": 1200, "height": 2000, "h2_count": 3},
            {"y": 1600, "height": 2000, "h2_count": 3},
            {"y": 2000, "height": 2000, "h2_count": 3},
        ]
        step_iter = iter(step_returns)

        async def fake_evaluate(script: str, *args: Any) -> Any:
            # Four script kinds:
            # - incremental scroll step: window.scrollTo(0, ...)
            # - per-section scroll: scrollIntoView(...)
            # - expand: aria-expanded clicks
            # - extract: returns dict with top_card_text
            if "window.scrollTo" in script:
                call_order.append("scroll_step")
                return next(step_iter, {"y": 2000, "height": 2000, "h2_count": 3})
            if "scrollIntoView" in script:
                call_order.append("scroll_section")
                return True
            if "aria-expanded" in script and "top_card_text" not in script:
                call_order.append("expand")
                return []
            if "top_card_text" in script:
                call_order.append("extract")
                return raw
            call_order.append("unknown")
            return None

        page = MagicMock()
        page.evaluate = AsyncMock(side_effect=fake_evaluate)

        result = await extract_main_profile(page)

        # Incremental sweep runs at least a few steps, then exactly one
        # scroll-section call per labeled section (about/experience/
        # education), then expand, then extract.
        assert call_order[-5:] == [
            "scroll_section",
            "scroll_section",
            "scroll_section",
            "expand",
            "extract",
        ]
        assert call_order.count("scroll_step") >= 3
        assert result["name"] == "Alex LeBrun"
        assert len(result["experience"]) == 5

    async def test_continues_when_expansion_fails(self) -> None:
        """A failed expand call must not break extraction."""
        raw = _alexandrelebrun_raw()
        calls: list[str] = []

        async def fake_evaluate(script: str, *args: Any) -> Any:
            if "window.scrollTo" in script:
                calls.append("scroll_step")
                # Report immediate stability so the loop exits fast.
                return {"y": 0, "height": 0, "h2_count": 0}
            if "scrollIntoView" in script:
                calls.append("scroll_section")
                return False  # heading not found; skip hydration pause
            if "aria-expanded" in script and "top_card_text" not in script:
                calls.append("expand")
                raise RuntimeError("stale element")
            calls.append("extract")
            return raw

        page = MagicMock()
        page.evaluate = AsyncMock(side_effect=fake_evaluate)

        result = await extract_main_profile(page)

        # Tail must end with the section scrolls + expand + extract.
        assert calls[-5:] == [
            "scroll_section",
            "scroll_section",
            "scroll_section",
            "expand",
            "extract",
        ]
        assert result["name"] == "Alex LeBrun"

    async def test_handles_null_evaluate_result(self) -> None:
        """When the page has no <main>, the extractor returns ``None`` from JS."""
        page = MagicMock()
        page.evaluate = AsyncMock(return_value=None)

        result = await extract_main_profile(page)

        # All fields should be the empty defaults.
        assert result["name"] is None
        assert result["experience"] == []
        assert result["education"] == []
