"""Dump LinkedIn conversation-thread DOM for parser development.

Opens a real conversation against the persisted login profile, scrolls a few
times to populate the virtualized list, then dumps the structural shape of
the message-list region: container, per-event ``<li>`` outerHTML (trimmed),
``<time datetime>`` attributes, ``a[href^="/in/"]`` anchors, and header
participants.

Used while implementing structured ``get_conversation`` parsing
(see issue stickerdaniel/linkedin-mcp-server#442).

Run: uv run python scripts/dump_conversation_dom.py <thread_url_or_id> [scrolls]
"""

import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Capture our own argv before the project's CLI parser sees it on import.
_SCRIPT_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]

from linkedin_mcp_server.drivers.browser import (  # noqa: E402
    close_browser,
    ensure_authenticated,
    get_or_create_browser,
    set_headless,
)

OUTPUT_DIR = Path(__file__).parent / "snapshot_dumps"


_THREAD_URL_RE = re.compile(r"/messaging/thread/([^/?#]+)")


def normalize_thread_url(arg: str) -> str:
    match = _THREAD_URL_RE.search(arg)
    thread_id = match.group(1) if match else arg
    return f"https://www.linkedin.com/messaging/thread/{thread_id}/"


DUMP_SCRIPT = r"""
() => {
    const clip = (s, n = 4000) => (s || '').slice(0, n);

    // Heuristic: the message list is the longest <ul> inside <main> whose
    // children are <li>. Fall back to the longest <ul> in <main> otherwise.
    const main = document.querySelector('main');
    const uls = main ? Array.from(main.querySelectorAll('ul')) : [];
    let bestUl = null;
    let bestCount = 0;
    for (const ul of uls) {
        const lis = Array.from(ul.children).filter(c => c.tagName === 'LI');
        if (lis.length > bestCount) {
            bestUl = ul;
            bestCount = lis.length;
        }
    }

    const eventEntries = [];
    if (bestUl) {
        const lis = Array.from(bestUl.children).filter(c => c.tagName === 'LI');
        // Sample up to 8 events: first 3, last 3, and any that look like deleted/tombstone
        const sample = new Set();
        lis.slice(0, 3).forEach(li => sample.add(li));
        lis.slice(-3).forEach(li => sample.add(li));
        for (const li of lis) {
            const txt = (li.innerText || '').toLowerCase();
            if (txt.includes('deleted') || txt.includes('unsent')) sample.add(li);
        }
        for (const li of sample) {
            const time = li.querySelector('time');
            const personLinks = Array.from(li.querySelectorAll('a[href]'))
                .map(a => a.getAttribute('href'))
                .filter(h => h && (h.startsWith('/in/') || h.includes('/in/')));
            const ariaAttrs = {};
            for (const attr of li.attributes) {
                if (attr.name.startsWith('aria-') || attr.name.startsWith('data-')) {
                    ariaAttrs[attr.name] = attr.value;
                }
            }
            eventEntries.push({
                index: lis.indexOf(li),
                aria_data_attrs: ariaAttrs,
                time_datetime: time ? time.getAttribute('datetime') : null,
                time_text: time ? (time.innerText || '').trim() : null,
                person_links: personLinks,
                inner_text: clip(li.innerText, 800),
                outer_html: clip(li.outerHTML, 4000),
            });
        }
    }

    // Header / participants region: try aria-label of the conversation root
    // and any /in/ links sitting near the top of <main>.
    const headerCandidates = [];
    if (main) {
        const headings = Array.from(main.querySelectorAll('h1, h2, h3')).slice(0, 5);
        for (const h of headings) {
            headerCandidates.push({
                tag: h.tagName.toLowerCase(),
                text: (h.innerText || '').trim(),
                outer_html: clip(h.outerHTML, 800),
            });
        }
    }

    const topProfileLinks = main
        ? Array.from(main.querySelectorAll('a[href*="/in/"]'))
              .slice(0, 12)
              .map(a => ({
                  href: a.getAttribute('href'),
                  text: (a.innerText || '').trim(),
                  aria_label: a.getAttribute('aria-label'),
              }))
        : [];

    return {
        url: location.href,
        main_present: Boolean(main),
        ul_count: uls.length,
        best_ul_event_count: bestCount,
        best_ul_outer_html_head: bestUl ? clip(bestUl.outerHTML, 2000) : null,
        events_sampled: eventEntries,
        header_candidates: headerCandidates,
        top_profile_links: topProfileLinks,
    };
}
"""


async def main():
    if not _SCRIPT_ARGV:
        print(
            "Usage: uv run python scripts/dump_conversation_dom.py <thread_url_or_id> [scrolls]"
        )
        sys.exit(1)

    thread_url = normalize_thread_url(_SCRIPT_ARGV[0])
    scrolls = int(_SCRIPT_ARGV[1]) if len(_SCRIPT_ARGV) >= 2 else 3

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"conversation_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    headless_env = os.environ.get("DUMP_HEADLESS", "true").lower() != "false"
    set_headless(headless_env)
    print(f"--- headless={headless_env} ---")

    try:
        await ensure_authenticated()
        browser = await get_or_create_browser()
        page = browser.page

        print(f"--- Navigating to {thread_url} ---")
        await page.goto(thread_url, wait_until="domcontentloaded")
        await page.wait_for_selector("main", timeout=15000)
        # Let LinkedIn hydrate the messaging SPA
        await asyncio.sleep(3)

        for i in range(scrolls):
            await page.evaluate(
                """() => {
                    const main = document.querySelector('main');
                    if (!main) return;
                    const scrollable = Array.from(main.querySelectorAll('*'))
                        .find(el => el.scrollHeight > el.clientHeight + 50 && getComputedStyle(el).overflowY !== 'visible');
                    if (scrollable) scrollable.scrollTop = 0;
                }"""
            )
            await asyncio.sleep(1.5)
            print(f"  scroll {i + 1}/{scrolls} done")

        dump = await page.evaluate(DUMP_SCRIPT)

        (run_dir / "dump.json").write_text(
            json.dumps(dump, indent=2, ensure_ascii=False)
        )
        # Save the best <ul> outer HTML head separately for easy inspection
        if dump.get("best_ul_outer_html_head"):
            (run_dir / "best_ul_head.html").write_text(dump["best_ul_outer_html_head"])
        for event in dump.get("events_sampled", []):
            idx = event["index"]
            (run_dir / f"event_{idx}.html").write_text(event["outer_html"])

        print(f"\nDump saved to {run_dir}/")
        print(f"  events sampled: {len(dump.get('events_sampled', []))}")
        print(f"  best_ul_event_count: {dump.get('best_ul_event_count')}")
        print(f"  top_profile_links: {len(dump.get('top_profile_links', []))}")

    finally:
        await close_browser()


if __name__ == "__main__":
    asyncio.run(main())
