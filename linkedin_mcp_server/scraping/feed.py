"""Home-feed scraping with SDUI permalink capture."""

from __future__ import annotations

from typing import Any

import asyncio
import logging

import anyio
import anyio.lowlevel
from patchright.async_api import TimeoutError as PlaywrightTimeoutError
from patchright.async_api import Page

from linkedin_mcp_server.core.exceptions import LinkedInScraperException
from linkedin_mcp_server.error_diagnostics import build_issue_diagnostics
from linkedin_mcp_server.scraping.content import PageContentReader
from linkedin_mcp_server.scraping.contracts import (
    RATE_LIMITED_SECTION_TEXT,
    ExtractedSection,
)
from linkedin_mcp_server.scraping.feed_payload import (
    POST_SLUG_URL_RE,
    build_feed_references,
    is_feed_payload_response,
    normalize_feed_post,
)
from linkedin_mcp_server.scraping.navigation import PageNavigator
from linkedin_mcp_server.scraping.session import ScrapingSession
from linkedin_mcp_server.scraping.text import (
    truncate_linkedin_noise,
)

logger = logging.getLogger(__name__)

_FEED_POSTS_JS = r"""
({ limit }) => {
  const root = document.querySelector('main') || document.body;
  if (!root) return [];

  const normalize = v => (v || '').replace(/\s+/g, ' ').trim();
  const linesFrom = el => {
    const t = el ? (el.innerText || el.textContent || '') : '';
    return t.split('\n').map(normalize).filter(Boolean);
  };
  const linkedInPath = href => {
    try { const u = new URL(href, location.origin); return `${u.pathname}${u.search}${u.hash}`; }
    catch { return ''; }
  };
  const absUrl = href => { try { return new URL(href, location.origin).href; } catch { return null; } };
  const isExternal = href => {
    try {
      const h = new URL(href, location.origin).host.toLowerCase();
      // endsWith('linkedin.com') alone would treat notlinkedin.com as internal.
      return !!h && h !== 'linkedin.com' && !h.endsWith('.linkedin.com');
    } catch { return false; }
  };
  const bestAnchorText = a => {
    const t = normalize(a.textContent);
    if (t) return t;
    return normalize(a.querySelector('img[alt]')?.getAttribute('alt')) || null;
  };
  const isImageOnlyAnchor = a => !normalize(a.textContent) && !!a.querySelector('img[alt]');
  const cleanName = text => {
    if (!text) return null;
    const c = normalize(text)
      .replace(/^view\s+/i, '')
      .replace(/’/g, "'")
      .replace(/\s*'s\s+(?:profile|page)\b.*$/i, '')
      .replace(/\s+profile\s+(?:photo|picture)$/i, '')
      .replace(/^[•·]+\s*/, '')
      .trim();
    return c || null;
  };

  const DEGREE_RE = /^[·•]?\s*(\d(?:st|nd|rd)\+?)$/i;
  const AGE_RE = /(?:^|\s)(\d+)\s*(mo|min|m|hr|hrs|h|d|w|wk|y|yr)\b/i;
  const FOLLOWERS_RE = /^\d[\d,.\s]*\s+followers$/i;
  // Reaction/social-proof headers render above the actor as "<Name> <verb> this"
  // — the verb spans every reaction (like/love/celebrate/support/funny/insightful)
  // plus repost/comment/follow. en-US only (BrowserManager locks the locale).
  const SOCIAL_PROOF_RE = /(?:likes?|loves?|celebrates?|supports?|reposted|shared|commented on|reacted(?:\s+to)?)\s+this$|\bfinds this (?:funny|insightful|helpful)$|follows? this page$|\breacted$/i;
  const BADGE_RE = /^(?:premium|verified)\s+(?:profile|member|account)$/i;
  const MORE_RE = /^(?:…|\.\.\.)\s*more$|^see more$/i;
  const NOISE = new Set([
    'Feed post', 'Suggested', 'Promoted', 'Following', 'Follow', '+ Follow',
    'Connect', 'Message', 'Like', 'Comment', 'Repost', 'Send', 'More',
    'Show translation', 'Save', 'Saved', 'Report post', 'Copy link to post',
  ]);

  const ageToken = (n, raw) => {
    const u = raw.toLowerCase();
    const unit = u.startsWith('mo') ? 'mo'
      : (u === 'm' || u.startsWith('min')) ? 'min'
      : u.startsWith('h') ? 'h'
      : u.startsWith('d') ? 'd'
      : u.startsWith('w') ? 'w'
      : 'y';
    return `${n}${unit}`;
  };
  const feedAge = lines => {
    // The header age line is short and carries the "•" separator; prefer it
    // before scanning the rest so body text like "5h of work" can't win.
    for (const L of lines.slice(0, 10)) {
      if (!/[•·]/.test(L) && L.length > 24) continue;
      const m = L.match(AGE_RE); if (m) return ageToken(m[1], m[2]);
    }
    for (const L of lines.slice(0, 10)) { const m = L.match(AGE_RE); if (m) return ageToken(m[1], m[2]); }
    return null;
  };

  const intFrom = s => { const n = parseInt(String(s).replace(/[^\d]/g, ''), 10); return Number.isFinite(n) ? n : null; };
  const countFromLabels = (card, re) => {
    // querySelectorAll is DOM order, so the post-level social bar (which
    // precedes any expanded comment thread) is matched before comment counts.
    for (const el of card.querySelectorAll('[aria-label]')) {
      const m = normalize(el.getAttribute('aria-label')).match(re);
      if (m) return intFrom(m[1]);
    }
    return null;
  };
  const countFromLines = (lines, re) => {
    for (const L of lines) { const m = L.match(re); if (m) return intFrom(m[1]); }
    return null;
  };

  // ---- Container discovery: one outermost element per post, by URN shape ----
  const URN_RE = /urn:li:(?:activity|ugcPost|share):\d+/i;
  const urnOf = el => el.getAttribute('data-urn') || el.getAttribute('data-id') || '';
  let nodes = Array.from(root.querySelectorAll('[data-urn], [data-id]'))
    .filter(el => URN_RE.test(urnOf(el)) || /sponsoredCreative/i.test(urnOf(el)));
  nodes = nodes.filter(el => !nodes.some(o => o !== el && o.contains(el)));
  if (!nodes.length) {
    // Fallback: LinkedIn renders a visually-hidden "Feed post" heading once
    // per update — climb from it to the post root (has an actor anchor and at
    // least four labeled actions).
    const climbed = [];
    for (const h of Array.from(root.querySelectorAll('h2, [role="heading"]'))
      .filter(h => /^feed post$/i.test(normalize(h.textContent)))) {
      let el = h;
      for (let i = 0; i < 8 && el && el !== root; i++) {
        const actor = el.querySelector('a[href*="/in/"], a[href*="/company/"], a[href*="/school/"], a[href*="/showcase/"]');
        const actions = el.querySelectorAll('button[aria-label], [role="button"][aria-label]').length;
        if (actor && actions >= 4 && normalize(el.innerText).length > 80) { climbed.push(el); break; }
        el = el.parentElement;
      }
    }
    nodes = climbed.filter(el => !climbed.some(o => o !== el && o.contains(el)));
  }
  nodes.sort((a, b) => {
    const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
    return (ra.top - rb.top) || (ra.left - rb.left);
  });

  const result = [];
  for (const card of nodes) {
    const lines = linesFrom(card);
    const urn = urnOf(card);
    const anchors = Array.from(card.querySelectorAll('a[href]')).map(a => ({
      a,
      path: linkedInPath(a.getAttribute('href') || a.href),
      href: a.getAttribute('href') || a.href,
      text: bestAnchorText(a),
      image_only: isImageOnlyAnchor(a),
    }));
    const buttonTexts = new Set(
      Array.from(card.querySelectorAll('button, [role="button"]')).flatMap(linesFrom)
    );

    const is_promoted = /sponsoredCreative/i.test(urn) || lines.some(L => /^promoted$/i.test(L));

    // ---- degree (structural: bullet + ordinal; people only) ----
    let degree = null, degreeIdx = -1;
    for (let i = 0; i < lines.length && i < 14; i++) {
      const m = lines[i].match(DEGREE_RE);
      if (m) { degree = m[1].toLowerCase(); degreeIdx = i; break; }
    }

    // The actor header sits above the degree/age line. Skip past any
    // social-proof header ("<Reactor> likes this" — which LinkedIn may split
    // across two lines, "<Reactor>" then "likes this") so the reactor's name
    // can't be mistaken for the author or headline. Proof is matched only in
    // this header window, never the footer's "N others reacted".
    const ageLineIdx = lines.findIndex(
      L => AGE_RE.test(L) && (/[•·]/.test(L) || L.length <= 24)
    );
    const headerEnd = degreeIdx >= 0
      ? degreeIdx
      : (ageLineIdx >= 0 ? ageLineIdx : Math.min(6, lines.length));
    let headerStart = 0;
    for (let i = 0; i < headerEnd; i++) {
      if (SOCIAL_PROOF_RE.test(lines[i])) headerStart = i + 1;
    }

    // ---- author name from lines: the name sits just above the degree badge,
    // else the first identity-ish line after the social-proof header ----
    let authorName = null;
    if (degreeIdx > 0) {
      for (let i = degreeIdx - 1; i >= headerStart; i--) {
        const L = lines[i];
        if (!L || NOISE.has(L) || buttonTexts.has(L) || BADGE_RE.test(L)) continue;
        if (SOCIAL_PROOF_RE.test(L)) break;
        authorName = cleanName(L); if (authorName) break;
      }
    }
    if (!authorName) {
      for (let i = headerStart; i < lines.length; i++) {
        const L = lines[i];
        if (NOISE.has(L) || buttonTexts.has(L) || BADGE_RE.test(L)) continue;
        if (SOCIAL_PROOF_RE.test(L) || DEGREE_RE.test(L) || FOLLOWERS_RE.test(L)) continue;
        if (AGE_RE.test(L) && /[•·]/.test(L)) continue;
        authorName = cleanName(L); if (authorName) break;
      }
    }

    // ---- actor anchor (for profile_url): prefer the one whose text matches
    // the resolved name, so social-proof reactor anchors aren't picked ----
    const nameNorm = authorName ? normalize(authorName) : null;
    const matchA = nameNorm && anchors.find(x =>
      /^\/(?:in|company|showcase|school)\/[^/?#]+/.test(x.path) &&
      x.text && normalize(cleanName(x.text)) === nameNorm);
    const personA = anchors.find(x => !x.image_only && /^\/in\/[^/?#]+/.test(x.path));
    const pageA = anchors.find(x => !x.image_only && /^\/(?:company|showcase|school)\/[^/?#]+/.test(x.path));
    const actorA = matchA || personA || pageA || null;
    let profile_url = actorA ? actorA.path.split('?')[0] : null;
    if (profile_url) {
      // Canonicalize to the base entity path: the actor anchor sometimes
      // points at /company/<slug>/posts/ or similar deep links.
      const baseMatch = profile_url.match(/^(\/(?:in|company|showcase|school)\/[^/?#]+)/);
      if (baseMatch) profile_url = baseMatch[1] + '/';
    }
    const isPage = !!profile_url && /^\/(?:company|showcase|school)\//.test(profile_url);

    // ---- headline ----
    let headline = null;
    if (isPage) {
      headline = lines.find(L => FOLLOWERS_RE.test(L)) || null;
    } else {
      // Bound to the header window [headerStart, age) so neither a split
      // social-proof reactor name above nor the post body below can leak in.
      const headlineEnd = ageLineIdx >= 0 ? ageLineIdx : lines.length;
      for (let i = headerStart; i < headlineEnd; i++) {
        const L = lines[i];
        if (nameNorm && (L === nameNorm || L.startsWith(nameNorm))) continue;
        if (NOISE.has(L) || buttonTexts.has(L) || BADGE_RE.test(L)) continue;
        if (DEGREE_RE.test(L) || SOCIAL_PROOF_RE.test(L) || FOLLOWERS_RE.test(L)) continue;
        headline = L; break;
      }
    }

    const post_age = feedAge(lines);

    // ---- url (relative permalink) ----
    let url = null;
    const permA = anchors.find(x => /^\/feed\/update\/[^/?#]+/.test(x.path))
               || anchors.find(x => /^\/posts\/[^/?#]+/.test(x.path));
    if (permA) url = permA.path.split('?')[0].split('#')[0];
    if (!url) {
      const m = urn.match(/urn:li:(?:activity|ugcPost|share):\d+/i);
      if (m) url = `/feed/update/${m[0]}/`;
    }

    // ---- content: body lines between the header and the "see more"/footer.
    // Cutting at "see more" drops the trailing link-card CTA text cleanly. ----
    const moreIdx = lines.findIndex(L => MORE_RE.test(L));
    const footerIdx = lines.findIndex(L =>
      /^\d[\d,.\s]*\s+(?:reactions?|comments?|reposts?)$/i.test(L) ||
      /\breacted$/i.test(L) || L === 'Like' || L === 'Comment' || L === 'Repost' || L === 'Send');
    let startIdx = 0;
    const ageIdx = ageLineIdx;  // computed once in the header window above
    if (ageIdx >= 0) {
      startIdx = ageIdx + 1;
    } else {
      const promIdx = lines.findIndex(L => /^promoted$/i.test(L));
      const folIdx = lines.findIndex(L => FOLLOWERS_RE.test(L));
      startIdx = Math.max(promIdx, folIdx, degreeIdx) + 1;
      if (startIdx <= 0 && authorName) {
        const nIdx = lines.findIndex(L => normalize(L) === nameNorm);
        startIdx = nIdx >= 0 ? nIdx + 1 : 0;
      }
    }
    let endIdx = moreIdx >= 0 ? moreIdx : (footerIdx >= 0 ? footerIdx : lines.length);
    if (endIdx < startIdx) endIdx = lines.length;
    const body = lines.slice(startIdx, endIdx).filter(L =>
      !NOISE.has(L) && !buttonTexts.has(L) && !BADGE_RE.test(L));
    const content = body.join('\n').trim() || null;

    // ---- media: first match wins, video > link > image ----
    let media = null;
    const video = card.querySelector('video');
    if (video) {
      const v = video.getAttribute('src') || video.querySelector('source')?.getAttribute('src') || null;
      if (v) media = { type: 'video', url: absUrl(v) };
    }
    if (!media) {
      // Link card = external anchor carrying a preview <img>. The <img> guard
      // avoids matching inline body links (bare <a>, no thumbnail).
      const linkCard = anchors.find(x => x.href && isExternal(x.href) && x.a.querySelector('img'));
      if (linkCard) media = { type: 'link', url: absUrl(linkCard.href) };
    }
    if (!media) {
      const img = Array.from(card.querySelectorAll('img')).find(im => {
        const s = im.currentSrc || im.src || '';
        return /media\.licdn\.com/i.test(s)
          && /(feedshare|image-shrink|article-cover|media-proxy)/i.test(s)
          && (im.naturalWidth || im.width || 0) >= 160;
      });
      if (img) media = { type: 'image', url: img.currentSrc || img.src };
    }

    // ---- engagement counts: aria-label first, then footer text ----
    let reactions_count = countFromLabels(card, /\b(\d[\d,.\s]*)\s+reactions?\b/i)
      ?? countFromLines(lines, /^(\d[\d,.\s]*)\s+reactions?$/i);
    if (reactions_count == null && lines.some(L => /reacted/i.test(L))) {
      // "Name and N others reacted" -> N + 1 (the named reactor counts too).
      const rl = lines.find(L => /\band\s+\d[\d,.\s]*\s+others?\b/i.test(L));
      const m = rl && rl.match(/\band\s+(\d[\d,.\s]*)\s+others?\b/i);
      const n = m ? intFrom(m[1]) : null;
      reactions_count = n == null ? null : n + 1;
    }
    const comment_count = countFromLabels(card, /\b(\d[\d,.\s]*)\s+comments?\b/i)
      ?? countFromLines(lines, /^(\d[\d,.\s]*)\s+comments?$/i);
    const repost_count = countFromLabels(card, /\b(\d[\d,.\s]*)\s+reposts?\b/i)
      ?? countFromLines(lines, /^(\d[\d,.\s]*)\s+reposts?$/i);

    // Drop chrome that slipped through discovery.
    if (!authorName && !content && !url) continue;

    result.push({
      url, post_age,
      author: { name: authorName, profile_url, headline, degree },
      content, is_promoted, media,
      reactions_count, comment_count, repost_count,
    });
    if (limit && result.length >= limit) break;
  }
  return result;
}
"""


class FeedScraper:
    """Scrape the home feed and the post permalinks its SDUI payloads carry."""

    def __init__(
        self,
        session: ScrapingSession,
        navigator: PageNavigator,
        content: PageContentReader,
    ):
        self._session = session
        self._navigator = navigator
        self._content = content

    @staticmethod
    async def _drain_listener_tasks(pending: list[asyncio.Task[None]]) -> None:
        """Bounded teardown for fire-and-forget response listener tasks.

        The feed scroll loop appends a read task per matching response;
        those tasks must finish (or be cancelled) before we leave the
        extractor or the event loop's "Task exception was never retrieved"
        warnings will surface unrelated errors. The caps below let a stuck
        ``resp.body()`` call burn at most three seconds of teardown budget.
        """
        if not pending:
            return
        try:
            await asyncio.wait(pending, timeout=2.0)
        finally:
            # Cancel on *every* exit of that wait, the caller's own cancellation
            # included. The response listener is unsubscribed before we get here,
            # so no one else will ever ask these reads to stop; returning through
            # the cancelled path without asking left a real ``resp.body()``
            # running with no cancellation requested at all.
            for task in pending:
                if not task.done():
                    task.cancel()
            # FastMCP wraps each tool call in ``anyio.fail_after``, whose scope
            # re-delivers its cancellation on every loop iteration until the task
            # leaves it. Unshielded, the wait below would be cancelled before the
            # reads it watches can act on the cancel above, which is the case the
            # budget exists for. The shield covers a bounded wait only, and the
            # outer cancellation resumes as soon as the scope closes.
            with anyio.CancelScope(shield=True):
                try:
                    await asyncio.wait(pending, timeout=1.0)
                finally:
                    # A shield only holds off AnyIO's own delivery, so a second
                    # plain ``Task.cancel()`` still cuts that wait short. Read
                    # and report the reads as they actually stand, or a failure
                    # that arrived before the cancel is left for the loop to
                    # report and a task still running is left unmentioned.
                    leftover = [task for task in pending if not task.done()]
                    for task in pending:
                        if task.done() and not task.cancelled():
                            # Consume the failure; unretrieved, it reaches the
                            # loop's handler long after the feed call returned.
                            task.exception()
                    if leftover:
                        logger.warning(
                            "SDUI feed listener tasks did not drain after cancel; leaking %d task(s)",
                            len(leftover),
                        )
        # A deadline that first comes due inside the shield has nowhere to land:
        # AnyIO skips a shielded scope while delivering, and the restart on the
        # way out runs in this very task, so it can only schedule delivery for the
        # next turn. get_feed's next step is report_progress, which never suspends
        # when the client sent no progress token, and the expired call would then
        # return a result. This unshielded checkpoint is that next turn. It is
        # outside the block above so that a cancellation already on its way keeps
        # propagating without waiting on anything.
        await anyio.lowlevel.checkpoint()

    async def extract_feed(
        self,
        num_posts: int = 10,
    ) -> ExtractedSection:
        """Scrape the LinkedIn home feed, scrolling until *num_posts* are loaded."""
        try:
            return await self._extract_feed_once(num_posts)
        except LinkedInScraperException:
            raise
        except Exception as e:
            logger.warning("Failed to extract feed: %s", e)
            return ExtractedSection(
                text="",
                references=[],
                error=build_issue_diagnostics(e, context="extract_feed"),
            )

    async def _extract_feed_once(
        self,
        num_posts: int,
    ) -> ExtractedSection:
        """Single attempt: navigate, scroll until post count, extract."""
        url = "https://www.linkedin.com/feed/"
        page = self._session.page

        # Post permalinks live in the SDUI pagination response (field:
        # "postSlugUrl"). The initial /feed/ HTML embeds the same data in
        # an RSC flight payload. Listen for both during the whole scroll
        # loop. ``seen_urls`` doubles as the locale-independent scroll
        # progress signal, replacing the previous "Feed post" innerText
        # marker that broke on non-English UIs.
        captured_urls: list[str] = []
        seen_urls: set[str] = set()
        pending_reads: list[asyncio.Task[None]] = []

        def _handle_response(resp: Any) -> None:
            if not is_feed_payload_response(resp.url):
                return

            async def _read() -> None:
                try:
                    body = await resp.body()
                except Exception:
                    return
                if not body:
                    return
                text = body.decode("utf-8", errors="replace")
                for match in POST_SLUG_URL_RE.finditer(text):
                    post_url = f"https://www.linkedin.com/posts/{match.group('slug')}"
                    if post_url not in seen_urls:
                        seen_urls.add(post_url)
                        captured_urls.append(post_url)

            pending_reads.append(asyncio.create_task(_read()))

        page.on("response", _handle_response)
        try:
            return await self._extract_feed_body(
                url, num_posts, captured_urls, pending_reads
            )
        finally:
            try:
                # The very object that was registered, never a fresh equivalent:
                # Playwright matches a listener by identity, so a re-created
                # closure removes nothing and leaves the read subscribed for the
                # rest of the page's life. The drain below runs either way,
                # because a removal that raised is exactly the case where the
                # reads still need stopping.
                page.remove_listener("response", _handle_response)
            except Exception:
                pass
            await self._drain_listener_tasks(pending_reads)

    async def _extract_feed_body(
        self,
        url: str,
        num_posts: int,
        captured_urls: list[str],
        pending_reads: list[asyncio.Task[None]],
    ) -> ExtractedSection:
        page = self._session.page
        await self._navigator._navigate_to_page(url)
        await self._session.check_rate_limit()

        try:
            await page.wait_for_selector("main")
        except PlaywrightTimeoutError:
            logger.debug("No <main> element found on %s", url)

        await self._session.dismiss_modal()

        try:
            await page.wait_for_function(
                """() => {
                    const main = document.querySelector('main');
                    if (!main) return false;
                    return main.innerText.length > 200;
                }""",
                timeout=10000,
            )
        except PlaywrightTimeoutError:
            logger.debug("Feed content did not appear on %s", url)

        # The feed has its own scroll container — window.scrollTo is a no-op.
        # mouse.wheel over the viewport center triggers the real scroll.
        _MAX_SCROLLS = 12
        _MAX_STALE = 3
        _BATCH_WAIT = 6.0
        _WHEEL_DELTA = 2000
        _IN_LOOP_DRAIN_TIMEOUT = 1.0
        stale_count = 0

        viewport = page.viewport_size or {"width": 1280, "height": 720}
        cx, cy = viewport["width"] // 2, viewport["height"] // 2
        await page.mouse.move(cx, cy)

        for i in range(_MAX_SCROLLS):
            count = len(captured_urls)
            logger.debug("Feed scroll %d: %d permalinks captured", i, count)
            if count >= num_posts:
                break

            await page.mouse.wheel(0, _WHEEL_DELTA)

            new_count = count
            for _ in range(int(_BATCH_WAIT)):
                await self._session.delay(1.0)
                # Drain in-flight response reads so captured_urls reflects
                # everything Playwright already delivered. Without this,
                # the count comparison races: the wheel fires a network
                # response, the listener creates a read task, and the loop
                # sleeps and re-checks before _read() finishes appending —
                # producing false-stale verdicts.
                if pending_reads:
                    done, _still = await asyncio.wait(
                        pending_reads, timeout=_IN_LOOP_DRAIN_TIMEOUT
                    )
                    if done:
                        # Surface unexpected exceptions. _read() catches
                        # expected playwright errors, but a parser bug
                        # would otherwise vanish into the loop. Log them
                        # rather than raising so a single bad response
                        # doesn't abort the whole scroll session.
                        for result in await asyncio.gather(
                            *done, return_exceptions=True
                        ):
                            if isinstance(result, BaseException):
                                logger.warning(
                                    "Unhandled error in feed _read task: %r",
                                    result,
                                )
                    pending_reads[:] = [t for t in pending_reads if not t.done()]
                new_count = len(captured_urls)
                if new_count > count:
                    break

            if new_count > count:
                stale_count = 0
            else:
                stale_count += 1
                logger.debug(
                    "Feed stale scroll %d/%d (still at %d permalinks)",
                    stale_count,
                    _MAX_STALE,
                    new_count,
                )
                if stale_count >= _MAX_STALE:
                    logger.debug("Feed stopped producing new posts")
                    break

        # Give any in-flight response reads a beat to finish recording URLs.
        await self._session.delay(0.2)

        raw_result = await self._content._extract_root_content(["main"])
        raw = raw_result["text"]

        if not raw:
            return ExtractedSection(text="", references=[])
        if not truncate_linkedin_noise(raw) and raw.strip():
            logger.warning(
                "Page %s returned only LinkedIn chrome (likely rate-limited)", url
            )
            return ExtractedSection(text=RATE_LIMITED_SECTION_TEXT, references=[])
        raw_posts = []
        if isinstance(page, Page):
            try:
                raw_posts = await page.evaluate(_FEED_POSTS_JS, {"limit": num_posts})
            except Exception:
                logger.debug("Structured feed post extraction failed", exc_info=True)
        if not isinstance(raw_posts, list):
            raw_posts = []
        posts = [
            post
            for raw_post in raw_posts
            for post in [normalize_feed_post(raw_post)]
            if post is not None
        ][:num_posts]
        return ExtractedSection(
            text="",
            references=build_feed_references(raw_result["references"], captured_urls),
            posts=posts,
        )
