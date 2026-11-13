"""Compatibility access to fork-specific workflows not in the upstream facade."""

from typing import Any


def _extractor_for(extractor: Any) -> Any:
    """Use the retained scraper for fork-only tools, leaving test doubles intact."""
    # Mocks gain arbitrary attributes, so only wrap the real facade class.
    if type(extractor).__module__ != "linkedin_mcp_server.linkedin.extractor":
        return extractor
    from linkedin_mcp_server.scraping.extractor import (
        LinkedInExtractor as LegacyExtractor,
    )

    content = getattr(extractor, "_content")
    session = getattr(content, "_session")
    return LegacyExtractor(getattr(session, "page"))
