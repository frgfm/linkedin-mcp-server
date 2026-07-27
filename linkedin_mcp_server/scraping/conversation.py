"""LinkedIn message-thread DOM parser.

Encapsulates the in-browser JS extraction script and the Python
normalization step that build the structured ``get_conversation``
payload. Lives in its own module so the (long) extractor module isn't
weighed down by feature-specific parsing logic, and so DOM/parser
tests have a focused import target.

LinkedIn renders message events without ``<time datetime>`` and
without locale-independent status markers, so two structural signals
carry the contract:

* ``data-event-urn`` (prefix ``urn:li:msg_message:``) identifies a
  real message event ``<li>`` and filters chrome ``<li>``s (loader,
  top-of-list, quick-reply chips). The tuple's first element is the
  *viewer*'s ``fsd_profile`` URN, constant across the thread.
* A ``<time>`` element directly inside the ``<li>`` is a day heading
  ("Feb 10"); a ``<time>`` deeper in the event is the per-message
  clock ("3:17 PM").

Timestamp parsing and deleted-status detection rely on en-US text.
BrowserManager forces en-US, and the limitation is documented in the
docstring of ``get_conversation`` and in ``AGENTS.md``.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

from patchright.async_api import Page

from linkedin_mcp_server.scraping.link_metadata import (
    Member,
    Message,
    MessageStatus,
)

# en-US body text emitted for a recalled message. Guarded behind a
# documented locale assumption per the project's scraping rules.
_DELETED_BODY_TEXTS_EN_US: frozenset[str] = frozenset(
    {"This message has been deleted."}
)

_MONTH_ABBREVS_EN_US: dict[str, int] = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_DAY_HEADING_RE = re.compile(r"^\s*([A-Za-z]{3,9})\s+(\d{1,2})(?:,\s*(\d{4}))?\s*$")
# en-US "Today" / "Yesterday" relative-day headings. Resolved against
# ``datetime.now()`` in the normalizer; documented in the en-US locale
# caveats alongside the deleted-status marker.
_TODAY_HEADING_RE = re.compile(r"^\s*today\s*$", re.IGNORECASE)
_YESTERDAY_HEADING_RE = re.compile(r"^\s*yesterday\s*$", re.IGNORECASE)
_CLOCK_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})\s*([AaPp][Mm])?\s*$")

# Anchors LinkedIn renders inside a shared-post / shared-job card embedded
# in a message. When a participant shares a link the message has no <p>
# body — only the card — so we surface this URL as the message content.
# Match relative or absolute LinkedIn URLs; emit as a relative path.
_SHARED_LINK_RE = re.compile(
    r"^(?:https?://[^/]*linkedin\.com)?(/(?:feed/update/[^/?#]+/?|posts/[^/?#]+/?|jobs/view/\d+/?|pulse/[^/?#]+/?))",
    re.IGNORECASE,
)

_EXTRACT_SCRIPT = r"""
    start => {
        const clean = s => (s || '').replace(/\s+/g, ' ').trim();
        const root = start
            ? (start.closest('[role="dialog"]') || start.closest('main') || start)
            : document.querySelector('main');
        const inRoot = sel => {
            return root ? Array.from(root.querySelectorAll(sel)) : [];
        };

        // Real message events carry a data-event-urn attribute on an
        // inner element. Locale-independent and stable across the chrome
        // <li>s (loader, top-of-list, quick-reply chips) which have none.
        const eventLis = inRoot('li')
            .filter(li => li.querySelector('[data-event-urn^="urn:li:msg_message:"]'));

        const events = eventLis.map(li => {
            // Day heading: <time> element as a direct child of the <li>.
            const dayTime = Array.from(li.children)
                .find(c => c.tagName === 'TIME');
            const dayHeading = dayTime ? clean(dayTime.textContent) : null;

            // Per-message clock time: any <time> NOT directly under <li>.
            const innerTime = Array.from(li.querySelectorAll('time'))
                .find(t => t.parentElement !== li);
            const timeText = innerTime ? clean(innerTime.textContent) : null;

            // Sender link: the first /in/ anchor whose text is the display
            // name (not the a11y "View X's profile" link wrapping the
            // avatar). Heuristic: pick the /in/ anchor that does NOT
            // contain an <img>.
            let senderUrl = null;
            let senderName = null;
            const personAnchors = Array.from(li.querySelectorAll('a[href*="/in/"]'));
            const nameAnchor = personAnchors.find(a => !a.querySelector('img'));
            const avatarAnchor = personAnchors.find(a => a.querySelector('img'));
            if (nameAnchor) {
                senderUrl = nameAnchor.getAttribute('href');
                senderName = clean(nameAnchor.textContent);
            } else if (avatarAnchor) {
                senderUrl = avatarAnchor.getAttribute('href');
                const img = avatarAnchor.querySelector('img');
                senderName = img ? clean(img.getAttribute('alt') || img.getAttribute('title')) : null;
            }

            // Body: <p> directly inside the event item; LinkedIn renders
            // both normal and recalled messages as <p>. innerText is the
            // displayed text including line breaks.
            const eventItem = li.querySelector('[data-event-urn^="urn:li:msg_message:"]');
            const bodyEl = eventItem ? eventItem.querySelector('p') : null;
            const bodyText = bodyEl ? (bodyEl.innerText || bodyEl.textContent || '').trim() : null;

            // Shared-post / shared-job card: any anchor inside the event
            // whose href matches a content-permalink pattern (feed update,
            // posts slug, job posting, pulse article). We surface only
            // the first match — LinkedIn's card embeds repeat the link
            // across the title, image, and CTA.
            let sharedUrl = null;
            if (eventItem) {
                const cardAnchor = Array.from(eventItem.querySelectorAll('a[href]'))
                    .map(a => a.getAttribute('href') || '')
                    .find(h => /\/(?:feed\/update|posts|jobs\/view|pulse)\//i.test(h));
                if (cardAnchor) sharedUrl = cardAnchor;
            }

            return {
                day_heading: dayHeading,
                time_text: timeText,
                sender_url: senderUrl,
                sender_name: senderName,
                body_text: bodyText,
                shared_url: sharedUrl,
            };
        });

        // Viewer URN: every event item's data-event-urn carries the
        // authenticated user's fsd_profile URN as its first tuple
        // component (regardless of who authored the message). Pull it
        // from the first available event so the normalizer can rewrite
        // sender to "self" wherever LinkedIn renders the viewer's
        // profile anchor.
        let viewerUrn = null;
        const firstEventItem = eventLis.length > 0
            ? eventLis[0].querySelector('[data-event-urn^="urn:li:msg_message:"]')
            : null;
        if (firstEventItem) {
            const urn = firstEventItem.getAttribute('data-event-urn') || '';
            const m = urn.match(/urn:li:fsd_profile:([^,)]+)/);
            if (m) viewerUrn = m[1];
        }

        return { events, viewer_urn: viewerUrn };
    }
"""


def normalize_profile_url(raw: str | None) -> str | None:
    """Strip origin and query/fragment, return a ``/in/<slug>/`` path."""
    if not raw:
        return None
    # LinkedIn anchors are sometimes absolute (https://...) and sometimes
    # carry trailing query/fragment (?miniProfileUrn=...). Normalize.
    match = re.search(r"/in/([^/?#]+)", raw)
    if not match:
        return None
    return f"/in/{match.group(1)}/"


def parse_day_heading(
    text: str, today: datetime | None = None
) -> tuple[int, int, int | None] | None:
    """Parse 'Feb 10' / 'Feb 10, 2024' / 'Today' / 'Yesterday' → (month, day, year_or_None).

    en-US only. ``today`` is injectable for deterministic tests;
    defaults to ``datetime.now()`` when ``Today``/``Yesterday`` is
    encountered. Returns None if the heading does not match the
    expected pattern (caller falls back to no date context).
    """
    if _TODAY_HEADING_RE.match(text):
        anchor = today or datetime.now()
        return anchor.month, anchor.day, anchor.year
    if _YESTERDAY_HEADING_RE.match(text):
        anchor = (today or datetime.now()) - timedelta(days=1)
        return anchor.month, anchor.day, anchor.year
    m = _DAY_HEADING_RE.match(text)
    if not m:
        return None
    month_str, day_str, year_str = m.groups()
    month = _MONTH_ABBREVS_EN_US.get(month_str[:3].lower())
    if month is None:
        return None
    try:
        day = int(day_str)
    except ValueError:
        return None
    year = int(year_str) if year_str else None
    return month, day, year


def build_iso_timestamp(
    day_heading: str | None,
    time_text: str | None,
    reference_year: int,
) -> str:
    """Best-effort ISO 8601 from LinkedIn's split day-heading + clock text.

    Falls back to the raw concatenated text when either piece is
    missing or unparseable. ``reference_year`` is the current year —
    used to fill in dates LinkedIn renders without a year; if the
    resulting date would be in the future, the previous year is used.
    """
    parsed_day = parse_day_heading(day_heading) if day_heading else None
    clock = _CLOCK_RE.match(time_text or "")
    if not parsed_day or not clock:
        joined = " ".join(filter(None, [day_heading, time_text]))
        return joined or ""

    month, day, year = parsed_day
    hour = int(clock.group(1))
    minute = int(clock.group(2))
    meridiem = (clock.group(3) or "").lower()
    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0

    if year is None:
        year = reference_year
        try:
            candidate = datetime(year, month, day, hour, minute)
            if candidate > datetime.now():
                year -= 1
        except ValueError:
            return f"{day_heading} {time_text}"

    try:
        return datetime(year, month, day, hour, minute).strftime("%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return f"{day_heading} {time_text}"


def classify_status(
    body_text: str | None,
) -> tuple[MessageStatus, str | None]:
    """Map body text to (status, normalized_content).

    en-US heuristic: equality with a known "recalled" marker →
    ``deleted`` (content emitted as ``None``). All other states
    default to ``sent``; LinkedIn does not expose read/delivered to
    the sender in the message list DOM, so v1 ships those literals
    only via the type signature.
    """
    if body_text is None:
        return "sent", None
    if body_text.strip() in _DELETED_BODY_TEXTS_EN_US:
        return "deleted", None
    return "sent", body_text


def normalize_shared_url(raw: str | None) -> str | None:
    """Normalize a link-card href to a LinkedIn relative path.

    Strips origin and query/fragment so the emitted URL matches the
    relative form used elsewhere in the codebase (``references``
    entries, ``feed`` permalinks).
    """
    if not raw:
        return None
    m = _SHARED_LINK_RE.match(raw)
    return m.group(1) if m else None


def normalize_conversation_events(
    raw: dict[str, Any],
) -> tuple[list[Message], list[Member]]:
    """Convert the JS dump into structured Message/Member lists.

    Two-pass algorithm. The first pass walks the events to collect
    every distinct participant (url → display name) and identifies
    the authenticated user via the viewer URN embedded in
    ``data-event-urn``. The second pass emits the Message list with
    each ``sender`` field as the integer index into the ordered
    Member list (self always at index 0 when detectable).
    """
    events: list[dict[str, Any]] = raw.get("events") or []
    viewer_urn: str | None = raw.get("viewer_urn") or None

    # ------------------------------------------------------------
    # Pass 1: collect distinct participants and their display names.
    # ------------------------------------------------------------
    # Dict insertion order is the natural "first appearance" order.
    # Members are derived only from senders observed in events — a
    # silent participant (recipient who hasn't sent a visible message
    # yet) won't appear. The header-profiles scan that previously
    # tried to surface them relied on a broad "/in/ anchors outside
    # the message list" heuristic that picks up sidebar entries; see
    # PR review #4 for the rationale on dropping it.
    candidate_members: dict[str, str | None] = {}

    def _record_member(url: str | None, name: str | None) -> None:
        if not url:
            return
        existing = candidate_members.get(url)
        if url not in candidate_members:
            candidate_members[url] = name or None
        elif existing is None and name:
            candidate_members[url] = name

    for event in events:
        _record_member(
            normalize_profile_url(event.get("sender_url")),
            event.get("sender_name"),
        )

    # Resolve the self entry. Prefer an existing url whose path
    # contains the viewer URN; otherwise track an internal sentinel
    # so the authenticated user still sits at members[0]. The
    # sentinel is never emitted as a URL — fsd_profile IDs from
    # data-event-urn aren't guaranteed valid vanity paths, so we
    # surface ``is_self: true`` alone rather than a misleading URL.
    self_key: str | None = None
    self_key_synthesized = False
    if viewer_urn:
        self_key = next((u for u in candidate_members if viewer_urn in u), None)
        if not self_key:
            self_key = f"__self__{viewer_urn}"
            self_key_synthesized = True
            candidate_members[self_key] = None

    ordered_keys: list[str] = []
    if self_key:
        ordered_keys.append(self_key)
    ordered_keys.extend(u for u in candidate_members if u != self_key)

    members: list[Member] = []
    for key in ordered_keys:
        member: Member = {
            "kind": "person",
            "is_self": key == self_key,
        }
        # Only emit url when we observed an actual /in/ anchor in
        # the DOM. The synthesized self sentinel never gets a URL.
        if not (self_key_synthesized and key == self_key):
            member["url"] = key
        name = candidate_members.get(key)
        if name:
            member["name"] = name
        members.append(member)

    url_to_index: dict[str, int] = {key: i for i, key in enumerate(ordered_keys)}

    # ------------------------------------------------------------
    # Pass 2: emit messages with integer sender indices.
    # ------------------------------------------------------------
    reference_year = datetime.now().year
    running_day: str | None = None
    # Per-minute message groups share a single rendered <time>;
    # subsequent events in the group emit no time_text of their own.
    # Inherit the last observed clock value within a day, and reset
    # whenever a new day-heading lands so we never bleed times across
    # day boundaries.
    running_time: str | None = None
    messages: list[Message] = []

    for event in events:
        day_heading = event.get("day_heading")
        if day_heading:
            running_day = day_heading
            running_time = None
        time_text = event.get("time_text") or running_time
        if event.get("time_text"):
            running_time = event["time_text"]
        timestamp = build_iso_timestamp(running_day, time_text, reference_year)

        body_text = event.get("body_text")
        shared_url = normalize_shared_url(event.get("shared_url"))
        status, content_base = classify_status(body_text)

        # Resolve sender to an integer member index. Three signals
        # cover every case:
        #   1. anchor whose URL contains viewer URN  → self (idx 0)
        #   2. anchor whose URL is in the members map → that index
        #   3. no anchor at all                       → self (idx 0)
        # Events that can't be attributed (no anchor + no viewer URN)
        # fall through and are skipped — consistent with the V1
        # attachment-skip behavior.
        raw_sender_url = normalize_profile_url(event.get("sender_url"))
        sender_idx: int | None
        if raw_sender_url and viewer_urn and viewer_urn in raw_sender_url:
            sender_idx = url_to_index.get(self_key) if self_key else None
        elif raw_sender_url:
            sender_idx = url_to_index.get(raw_sender_url)
        elif self_key:
            sender_idx = url_to_index.get(self_key)
        else:
            sender_idx = None

        if sender_idx is None:
            continue

        # Drop events with no extractable text body that aren't
        # tombstones AND don't carry a shared link card. Covers
        # image / file / voice-only messages and any system event
        # we don't yet model in V1.
        if status != "deleted" and not body_text and not shared_url:
            continue

        if not body_text and shared_url and status != "deleted":
            content = shared_url
        else:
            content = content_base
        messages.append(
            {
                "timestamp": timestamp,
                "status": status,
                "sender": sender_idx,
                "content": content,
            }
        )

    return messages, members


async def extract_conversation(
    page: Page, root: Any | None = None
) -> tuple[list[Message], list[Member]]:
    """Pull structured messages + members from the current thread page.

    The caller is responsible for navigating to the thread URL and
    waiting for the message list to hydrate. This function only runs
    the in-page extraction script and normalizes its output. ``root`` may be
    a locator inside a messaging dialog; otherwise the parser uses ``main``.
    """
    raw = (
        await root.evaluate(_EXTRACT_SCRIPT)
        if root is not None
        else await page.evaluate(_EXTRACT_SCRIPT, None)
    )
    return normalize_conversation_events(raw)
