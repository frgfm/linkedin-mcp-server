"""LinkedIn main-profile page parser.

Encapsulates the in-browser JS extraction script and the Python
normalization step that build the structured ``main_profile`` payload
for ``get_person_profile``. Lives in its own module so the (long)
extractor module isn't weighed down by feature-specific parsing logic,
and so DOM/parser tests have a focused import target.

This is the second exception to the ``{section_name: raw_text}``
contract documented in ``CLAUDE.md`` — alongside ``get_conversation``,
``main_profile`` returns a structured dict instead of innerText so the
LLM doesn't re-parse the same top-card / about / experience / education
shape on every call.

Locale-dependence: count parsing ("500+ connections", "48,516
followers", "37 other mutual connections"), the "Present" date
sentinel, the "Contact info" / "Show all" line filters, and the H2
section-heading labels ("About", "Experience", "Education") all assume
en-US. BrowserManager forces en-US, and the limitation is documented on
the ``MainProfile`` TypedDict. Additional locales would extend the
labels tables in this module.

LinkedIn restricts content based on the viewer's relationship to the
profile owner: on 2nd/3rd-degree profiles, the About / Experience /
Education sections may not be rendered inline on the main profile page
at all. The parser returns ``None`` / empty lists in those cases — the
caller can request the dedicated ``experience`` and ``education``
sections (raw text from ``/in/<user>/details/…``) to recover the data.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlparse

from patchright.async_api import Page

from linkedin_mcp_server.scraping.link_metadata import (
    Education,
    Experience,
    MainProfile,
)

logger = logging.getLogger(__name__)

# Detect an explicit "X+ connections" so we can return None instead of
# parsing the integer (X is a floor, not a count). LinkedIn uses "500+"
# as the canonical privacy-respecting threshold.
_CONNECTION_PLUS_RE = re.compile(r"\b\d[\d,]*\+\s*connection", re.IGNORECASE)
_CONNECTION_EXACT_RE = re.compile(r"\b(\d[\d,]*)\s*connections?\b", re.IGNORECASE)
_FOLLOWER_RE = re.compile(r"\b(\d[\d,]*)\s*follower", re.IGNORECASE)
_MUTUAL_OTHER_RE = re.compile(
    r"\b(\d[\d,]*)\s*other\s+mutual\s+connection", re.IGNORECASE
)

# Date-line classifier for experience/education entries. Matches the
# en-US patterns LinkedIn renders ("Jan 2023 - Present", "2018 - 2022").
_DATE_LINE_RE = re.compile(
    r"^(?:[A-Za-z]{3,9}\s+\d{4}|\d{4})\s*[-–]\s*"
    r"(?:[A-Za-z]{3,9}\s+\d{4}|\d{4}|Present)",
    re.IGNORECASE,
)
_DURATION_TAIL_RE = re.compile(r"\s*[·•]\s*\d+\s*(?:yrs?|mos?).*$", re.IGNORECASE)

# A profile-picture image LinkedIn serves from media.licdn.com under
# the displayphoto namespace.
_PROFILE_PHOTO_HOST_RE = re.compile(
    r"^https?://[^/]*media\.licdn\.com/.*profile-displayphoto",
    re.IGNORECASE,
)

# en-US line marker placed by LinkedIn between the location and the
# current employer/school in the top card. Used by the line-by-line
# top-card parser as a structural anchor.
_CONTACT_INFO_MARKER_EN_US = "Contact info"
# en-US connection-degree markers ("1st", "2nd", "3rd", "3rd+"). Used to
# skip the degree line that LinkedIn renders directly under the name.
_DEGREE_LINE_RE = re.compile(r"^[·•]?\s*\d(?:st|nd|rd)\+?$", re.IGNORECASE)
# Anchor lines that bound the top card on the rendered page (en-US).
# After any of these labels the rest of innerText is sidebar / footer
# content and the line walker must stop. Pure "·" / "•" separators are
# also stops.
_TOP_CARD_END_MARKERS_EN_US = {
    "more profiles for you",
    "explore premium profiles",
    "people you may know",
    "show all",
    "about",  # site footer link
    "accessibility",
}

# Section heading labels for the H2-based section finder used by the JS
# extractor. LinkedIn's current layout dropped the
# ``<div id="about|experience|education">`` anchors, so we match on the
# ``<h2>`` heading text inside each ``<section>``.
_SECTION_LABELS_EN_US: dict[str, list[str]] = {
    "about": ["about"],
    "experience": ["experience"],
    "education": ["education"],
}

# Expand "see more" / "show more" buttons before extracting. The button
# text is locale-dependent, but inside the about / experience /
# education sections every collapsible content control LinkedIn renders
# carries ``aria-expanded="false"``. That attribute is structural and
# locale-independent — clicking them all reveals the truncated copy
# without depending on the rendered label. Sections are identified by
# their ``<h2>`` heading text (locale table) since LinkedIn no longer
# exposes the ``<div id="about|...">`` anchor divs.
_EXPAND_BUTTONS_JS = r"""
    (sectionLabels) => {
        const expanded = [];
        const main = document.querySelector('main');
        if (!main) return expanded;
        const labelsAll = Object.values(sectionLabels).flat().map(l => l.toLowerCase());
        const clean = s => (s || '').replace(/\s+/g, ' ').trim();
        const sections = Array.from(main.querySelectorAll('section'));
        for (const s of sections) {
            const heading = s.querySelector('h2');
            if (!heading) continue;
            const text = clean(heading.textContent).toLowerCase();
            if (!labelsAll.some(l => text === l)) continue;
            const buttons = s.querySelectorAll('button[aria-expanded="false"]');
            buttons.forEach(b => {
                try { b.click(); expanded.push(text); } catch (e) {}
            });
        }
        return expanded;
    }
"""

_EXTRACT_SCRIPT = r"""
    (sectionLabels) => {
        const clean = s => (s || '').replace(/\s+/g, ' ').trim();
        const main = document.querySelector('main');
        if (!main) return null;

        // ----- Top card -----
        // The first <section> in <main> is the top card on the
        // logged-in profile view. It contains name, headline, location,
        // current employer/school, counts, and action buttons.
        const topCard = main.querySelector('section');
        const cardText = topCard ? (topCard.innerText || '') : '';

        // Name: first <h2> inside <main>. LinkedIn dropped <h1> from
        // profile pages; the H2 at the top of the top card is now the
        // canonical name anchor.
        const h2 = main.querySelector('h2');
        const name = h2 ? clean(h2.textContent) : null;

        // Profile picture URL: the <img> rendered inside the top-card
        // avatar control. The avatar is the only top-card image with a
        // media.licdn.com displayphoto src; the default-avatar SVG has
        // no <img> at all (LinkedIn renders an inline <svg> icon).
        let profilePictureUrl = null;
        if (topCard) {
            const imgs = topCard.querySelectorAll('img');
            for (const img of imgs) {
                const src = img.getAttribute('src') || '';
                if (/profile-displayphoto/i.test(src)) {
                    profilePictureUrl = src;
                    break;
                }
            }
        }

        // Mutual-connection text: structural via anchor href to the
        // people-search facet that lists this profile's connections.
        let mutualText = null;
        if (topCard) {
            const mutualAnchor = topCard.querySelector(
                'a[href*="connectionOf"], a[href*="facetConnectionOf"]'
            );
            if (mutualAnchor) {
                mutualText = clean(mutualAnchor.innerText || mutualAnchor.textContent);
            }
        }

        // ----- Section finder by H2 heading text -----
        // LinkedIn's current layout identifies About / Experience /
        // Education only by their <h2> heading text. Enumerate every
        // <section> inside <main> and key off the first <h2>'s text
        // using the locale table passed in from Python.
        const sectionByLabels = (labels) => {
            const labelsLower = (labels || []).map(l => l.toLowerCase());
            const sections = Array.from(main.querySelectorAll('section'));
            for (const s of sections) {
                const heading = s.querySelector('h2');
                if (!heading) continue;
                const text = clean(heading.textContent).toLowerCase();
                if (labelsLower.some(l => text === l)) return s;
            }
            return null;
        };

        // ----- About -----
        let about = null;
        const aboutSection = sectionByLabels(sectionLabels.about);
        if (aboutSection) {
            const heading = aboutSection.querySelector('h2');
            let body = aboutSection.innerText || '';
            if (heading) {
                const headingText = (heading.innerText || heading.textContent || '').trim();
                if (headingText && body.startsWith(headingText)) {
                    body = body.slice(headingText.length).trimStart();
                }
            }
            // Strip trailing "...see more" / "see less" toggle remnants.
            body = body.replace(/\n[…\s]*see\s+(more|less)\s*$/i, '').trim();
            about = body || null;
        }

        // ----- Experience / Education list extraction -----
        // Anchor-first approach (resilient to layout changes): every
        // entry in Experience / Education carries a /company/ or
        // /school/ anchor pointing at the org/school page. We find
        // each unique anchor inside the section and walk up the DOM
        // tree to the nearest ancestor that "looks like an entry
        // container" — either a direct <li> child of the section's
        // main list, or the first ancestor whose innerText is at least
        // 30 chars. De-dup on the entry container so multi-role
        // companies (one anchor per role under the same parent) only
        // contribute one entry.
        const extractEntries = (sectionEl, anchorSelector) => {
            if (!sectionEl) return [];
            const anchors = Array.from(sectionEl.querySelectorAll(anchorSelector));
            const seen = new Set();
            const entries = [];
            for (const a of anchors) {
                // Walk up to the nearest <li> ancestor that's a direct
                // child of a <ul>, OR to a div with substantial text.
                let container = a;
                let chosen = null;
                while (container && container !== sectionEl) {
                    if (container.tagName === 'LI'
                        && container.parentElement
                        && container.parentElement.tagName === 'UL') {
                        chosen = container;
                        break;
                    }
                    container = container.parentElement;
                }
                if (!chosen) {
                    // Fall back: nearest ancestor with >= 30 chars of text.
                    let walker = a.parentElement;
                    while (walker && walker !== sectionEl) {
                        const t = (walker.innerText || '').trim();
                        if (t.length >= 30) { chosen = walker; break; }
                        walker = walker.parentElement;
                    }
                }
                if (!chosen) continue;
                if (seen.has(chosen)) continue;
                seen.add(chosen);
                entries.push({
                    text: (chosen.innerText || '').trim(),
                    anchor_href: a.getAttribute('href'),
                });
            }
            return entries;
        };

        const experienceRaw = extractEntries(
            sectionByLabels(sectionLabels.experience),
            'a[href*="/company/"]'
        );
        const educationRaw = extractEntries(
            sectionByLabels(sectionLabels.education),
            'a[href*="/school/"]'
        );

        return {
            name: name,
            top_card_text: cardText,
            mutual_text: mutualText,
            profile_picture_url: profilePictureUrl,
            about: about,
            experience_raw: experienceRaw,
            education_raw: educationRaw,
        };
    }
"""


def _parse_int(token: str) -> int | None:
    """Parse a comma-separated integer token; return None on failure."""
    cleaned = token.replace(",", "").strip()
    if not cleaned.isdigit():
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


def parse_connection_count(text: str | None) -> int | None:
    """Parse the connection count from top-card text.

    "500+ connections" -> ``None`` (LinkedIn's privacy threshold; not an
    exact count). "237 connections" -> ``237``. Anything else -> ``None``.
    """
    if not text:
        return None
    if _CONNECTION_PLUS_RE.search(text):
        return None
    match = _CONNECTION_EXACT_RE.search(text)
    if not match:
        return None
    return _parse_int(match.group(1))


def parse_follower_count(text: str | None) -> int | None:
    """Parse the follower count from top-card text (en-US)."""
    if not text:
        return None
    match = _FOLLOWER_RE.search(text)
    if not match:
        return None
    return _parse_int(match.group(1))


def parse_mutual_connection_count(mutual_text: str | None) -> int | None:
    """Parse the mutual-connection count from the mutual anchor text.

    LinkedIn renders the mutual indicator in a few shapes; this parser
    handles each:

    * ``"X, Y, and N other mutual connections"`` -> ``N + 2`` (LinkedIn
      always names the first one or two mutuals before the "other"
      count, so the parser counts the named prefix by comma/``and``
      separators and adds it to ``N``).
    * ``"X and N other mutual connections"`` -> ``N + 1``.
    * ``"X is a mutual connection"`` (singleton) -> ``1``.
    * ``"X and Y are mutual connections"`` (exact two, no "other") ->
      ``2``.

    Returns ``None`` when no mutual-connection anchor was found.
    """
    if not mutual_text:
        return None
    match = _MUTUAL_OTHER_RE.search(mutual_text)
    if match:
        others = _parse_int(match.group(1))
        if others is None:
            return None
        before = mutual_text[: match.start()]
        before = re.sub(r"\s*\band\b\s*$", "", before, flags=re.IGNORECASE).strip()
        before = before.rstrip(",").strip()
        named = (
            [n.strip() for n in before.split(",") if n.strip()] if before else []
        )
        return others + len(named)

    plural_match = re.search(
        r"\bare\s+mutual\s+connection", mutual_text, re.IGNORECASE
    )
    if plural_match:
        before = mutual_text[: plural_match.start()].strip()
        parts = [
            p.strip()
            for p in re.split(r"\s*(?:,|\band\b)\s*", before, flags=re.IGNORECASE)
            if p.strip()
        ]
        return len(parts) if parts else None

    if re.search(r"\bmutual\s+connection", mutual_text, re.IGNORECASE):
        return 1
    return None


def _dedupe_lines(text: str) -> list[str]:
    """Split innerText, drop empties and adjacent duplicate lines.

    LinkedIn's innerText often pairs visible text with visually-hidden
    accessibility duplicates ("Cofounder\\nCofounder"). Adjacent dedupe
    catches those without flattening the document.
    """
    lines: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if lines and lines[-1] == line:
            continue
        lines.append(line)
    return lines


def _strip_duration_tail(text: str) -> str:
    """Drop a trailing duration fragment ("- 3 yrs 2 mos") from a date line."""
    return _DURATION_TAIL_RE.sub("", text).strip()


def _normalize_href(href: str | None) -> str | None:
    """Strip origin and query/fragment from a LinkedIn relative path."""
    if not href:
        return None
    parsed = urlparse(href)
    path = parsed.path
    if not path:
        return None
    if not path.endswith("/"):
        path = path + "/"
    return path


def _parse_top_card_lines(top_card_text: str | None) -> dict[str, str | None]:
    """Parse headline / location / main_organization / main_education
    from the top card's innerText.

    LinkedIn's current layout drops every structural anchor for these
    fields (no ``<a href="/company/...">`` in the top card, no
    ``<h1>`` for the name). The innerText is the only reliable source.
    Lines in the en-US top card render in this order:

    * line 0: name (also captured via the H2 selector)
    * line 1: optional connection degree ("· 2nd")
    * line 2: headline
    * line 3: location (comma-separated geo)
    * line 4: ``·`` separator
    * line 5: "Contact info" marker
    * line 6: main organization (current employer text)
    * line 7: main education (current school text)
    * line 8+: follower/connection/mutual counts, action buttons,
      sidebar content

    The parser stops at the first "Show all" / "More profiles for you"
    / "Explore Premium profiles" line (everything after that is
    sidebar / footer content).
    """
    result: dict[str, str | None] = {
        "headline": None,
        "location": None,
        "main_organization": None,
        "main_education": None,
    }
    if not top_card_text:
        return result

    raw_lines = _dedupe_lines(top_card_text)
    # Truncate at any top-card end marker.
    lines: list[str] = []
    for line in raw_lines:
        if line.lower() in _TOP_CARD_END_MARKERS_EN_US:
            break
        lines.append(line)

    if not lines:
        return result

    # Drop the name line (line 0) — surfaced separately via the H2.
    cursor = 1
    # Skip a connection-degree line ("· 2nd") if present.
    while cursor < len(lines) and (
        _DEGREE_LINE_RE.match(lines[cursor]) or lines[cursor] in {"·", "•"}
    ):
        cursor += 1

    # Headline: next non-skip line.
    if cursor < len(lines):
        result["headline"] = lines[cursor]
        cursor += 1

    # Location: next non-empty line, optionally containing a comma.
    # Strip a trailing " · Contact info" if LinkedIn collapsed it onto
    # the same line.
    while cursor < len(lines) and lines[cursor] in {"·", "•"}:
        cursor += 1
    if cursor < len(lines):
        loc_line = lines[cursor]
        if loc_line.lower() != _CONTACT_INFO_MARKER_EN_US.lower():
            loc_line = re.split(r"\s*[·•]\s*Contact info", loc_line, flags=re.IGNORECASE)[0].strip()
            if loc_line:
                result["location"] = loc_line
            cursor += 1

    # Skip "·" separators and the "Contact info" line itself.
    while cursor < len(lines) and (
        lines[cursor] in {"·", "•"}
        or lines[cursor].lower() == _CONTACT_INFO_MARKER_EN_US.lower()
    ):
        cursor += 1

    # main_organization: next line that doesn't look like a count.
    def _is_count_line(line: str) -> bool:
        lower = line.lower()
        return (
            "follower" in lower
            or "connection" in lower
            or "mutual" in lower
        )

    if cursor < len(lines) and not _is_count_line(lines[cursor]):
        result["main_organization"] = lines[cursor]
        cursor += 1

    # main_education: next non-count line.
    while cursor < len(lines) and lines[cursor] in {"·", "•"}:
        cursor += 1
    if cursor < len(lines) and not _is_count_line(lines[cursor]):
        # Only accept this as education if the surrounding lines look
        # plausibly like the top-card slot (i.e. not an action button).
        candidate = lines[cursor]
        if candidate.lower() not in {"follow", "connect", "message", "more", "pending"}:
            result["main_education"] = candidate

    return result


def parse_experience_entry(raw: dict[str, Any]) -> Experience | None:
    """Convert one raw experience-entry dict into a structured Experience."""
    text = (raw.get("text") or "").strip()
    if not text:
        return None
    lines = _dedupe_lines(text)
    if not lines:
        return None

    entry: Experience = {}

    if lines:
        entry["title"] = lines[0]

    if len(lines) >= 2:
        org_line = lines[1]
        org = org_line.split(" · ", 1)[0].strip()
        if org:
            entry["organization"] = org

    org_url = _normalize_href(raw.get("anchor_href"))
    if org_url:
        entry["organization_url"] = org_url

    date_idx: int | None = None
    for i, line in enumerate(lines):
        if _DATE_LINE_RE.match(line):
            date_idx = i
            break
    if date_idx is not None:
        entry["dates"] = _strip_duration_tail(lines[date_idx])

    if date_idx is not None and date_idx + 1 < len(lines):
        loc_candidate = lines[date_idx + 1]
        if not _DATE_LINE_RE.match(loc_candidate):
            loc_clean = re.sub(
                r"\s*·\s*(?:Hybrid|Remote|On-site).*$",
                "",
                loc_candidate,
                flags=re.IGNORECASE,
            ).strip()
            if loc_clean:
                entry["location"] = loc_clean

    desc_start: int | None = None
    if date_idx is not None:
        desc_start = date_idx + 1
        if "location" in entry and desc_start < len(lines):
            desc_start += 1
    if desc_start is not None and desc_start < len(lines):
        description_lines = lines[desc_start:]
        if description_lines and re.match(
            r"^(?:…?\s*)?see\s+(more|less)$",
            description_lines[-1],
            re.IGNORECASE,
        ):
            description_lines = description_lines[:-1]
        description = "\n".join(description_lines).strip()
        if description:
            entry["description"] = description

    return entry


def parse_education_entry(raw: dict[str, Any]) -> Education | None:
    """Convert one raw education-entry dict into a structured Education."""
    text = (raw.get("text") or "").strip()
    if not text:
        return None
    lines = _dedupe_lines(text)
    if not lines:
        return None

    entry: Education = {}

    if lines:
        entry["school"] = lines[0]

    school_url = _normalize_href(raw.get("anchor_href"))
    if school_url:
        entry["school_url"] = school_url

    if len(lines) >= 2 and not _DATE_LINE_RE.match(lines[1]):
        degree_line = lines[1]
        if "," in degree_line:
            degree, _, field = degree_line.partition(",")
            degree = degree.strip()
            field = field.strip()
            if degree:
                entry["degree"] = degree
            if field:
                entry["field_of_study"] = field
        else:
            entry["degree"] = degree_line

    date_idx: int | None = None
    for i, line in enumerate(lines):
        if _DATE_LINE_RE.match(line):
            date_idx = i
            break
    if date_idx is not None:
        entry["dates"] = _strip_duration_tail(lines[date_idx])

    if date_idx is not None and date_idx + 1 < len(lines):
        description_lines = lines[date_idx + 1 :]
        if description_lines and re.match(
            r"^(?:…?\s*)?see\s+(more|less)$",
            description_lines[-1],
            re.IGNORECASE,
        ):
            description_lines = description_lines[:-1]
        description = "\n".join(description_lines).strip()
        if description:
            entry["description"] = description

    return entry


def normalize_main_profile(raw: dict[str, Any] | None) -> MainProfile:
    """Convert the JS dump into the structured ``MainProfile`` payload.

    Always returns a fully-populated dict (all keys present); missing
    signals become ``None`` for text fields and ``[]`` for the lists.
    """
    if not raw:
        return {
            "name": None,
            "headline": None,
            "location": None,
            "profile_picture_url": None,
            "connection_count": None,
            "follower_count": None,
            "mutual_connection_count": None,
            "main_organization": None,
            "main_education": None,
            "about": None,
            "experience": [],
            "education": [],
        }

    top_card_text: str | None = raw.get("top_card_text")
    top_card_fields = _parse_top_card_lines(top_card_text)

    profile_picture_url: str | None = raw.get("profile_picture_url")
    if profile_picture_url and not _PROFILE_PHOTO_HOST_RE.match(profile_picture_url):
        profile_picture_url = None

    experience: list[Experience] = []
    for entry_raw in raw.get("experience_raw") or []:
        entry = parse_experience_entry(entry_raw)
        if entry is not None:
            experience.append(entry)

    education: list[Education] = []
    for entry_raw in raw.get("education_raw") or []:
        entry = parse_education_entry(entry_raw)
        if entry is not None:
            education.append(entry)

    return {
        "name": raw.get("name") or None,
        "headline": top_card_fields["headline"],
        "location": top_card_fields["location"],
        "profile_picture_url": profile_picture_url,
        "connection_count": parse_connection_count(top_card_text),
        "follower_count": parse_follower_count(top_card_text),
        "mutual_connection_count": parse_mutual_connection_count(
            raw.get("mutual_text")
        ),
        "main_organization": top_card_fields["main_organization"],
        "main_education": top_card_fields["main_education"],
        "about": raw.get("about") or None,
        "experience": experience,
        "education": education,
    }


async def extract_main_profile(page: Page) -> MainProfile:
    """Pull the structured main-profile payload from the current page.

    The caller is responsible for navigating to the profile root and
    waiting for the page to hydrate (scroll budget exhausted, lazy
    sections loaded). This function expands the "see more" / "show
    more" toggles in the about / experience / education sections via
    a single ``aria-expanded="false"`` sweep, then runs the in-page
    extractor and normalizes its output.
    """
    try:
        await page.evaluate(_EXPAND_BUTTONS_JS, _SECTION_LABELS_EN_US)
    except Exception as e:
        logger.debug("see-more expansion failed (continuing): %s", e)

    raw = await page.evaluate(_EXTRACT_SCRIPT, _SECTION_LABELS_EN_US)
    return normalize_main_profile(raw)
