"""Dump every message event from the open thread (ALL events, not sampled).

Used to verify edge cases the sampled dump misses: link-card events,
attachment events, system events. Sister to dump_conversation_dom.py.

Run: uv run python scripts/probe_all_events.py <thread_url_or_id>
"""

import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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


PROBE_SCRIPT = r"""
() => {
    const clip = (s, n = 6000) => (s || '').slice(0, n);
    const main = document.querySelector('main');
    const uls = main ? Array.from(main.querySelectorAll('ul')) : [];
    const messageList = uls.find(ul => ul.querySelector('[data-event-urn^="urn:li:msg_message:"]'));

    if (!messageList) return { error: 'no message list found' };

    const events = [];
    const lis = Array.from(messageList.children).filter(c => c.tagName === 'LI');
    for (let i = 0; i < lis.length; i++) {
        const li = lis[i];
        const hasEventUrn = !!li.querySelector('[data-event-urn^="urn:li:msg_message:"]');
        const dayTime = Array.from(li.children).find(c => c.tagName === 'TIME');
        const innerTime = Array.from(li.querySelectorAll('time')).find(t => t.parentElement !== li);
        const personHrefs = Array.from(li.querySelectorAll('a[href*="/in/"]'))
            .map(a => a.getAttribute('href'));
        const allHrefs = Array.from(li.querySelectorAll('a[href]'))
            .map(a => ({ href: a.getAttribute('href'), text: (a.textContent || '').trim().slice(0, 80) }))
            .filter(x => x.href && !x.href.startsWith('#'));
        const ariaHeading = li.querySelector('.msg-s-event-listitem--group-a11y-heading');

        events.push({
            idx: i,
            has_event_urn: hasEventUrn,
            day_time: dayTime ? (dayTime.textContent || '').trim() : null,
            inner_time: innerTime ? (innerTime.textContent || '').trim() : null,
            person_hrefs: Array.from(new Set(personHrefs)),
            all_hrefs: allHrefs.slice(0, 6),
            aria_heading: ariaHeading ? (ariaHeading.textContent || '').trim().slice(0, 120) : null,
            inner_text: clip((li.innerText || '').trim(), 400),
            outer_html: clip(li.outerHTML, 6000),
        });
    }

    return { count: lis.length, events };
}
"""


async def main():
    if not _SCRIPT_ARGV:
        print("Usage: uv run python scripts/probe_all_events.py <thread_url_or_id>")
        sys.exit(1)

    thread_url = normalize_thread_url(_SCRIPT_ARGV[0])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"probe_all_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    set_headless(os.environ.get("DUMP_HEADLESS", "true").lower() != "false")

    try:
        await ensure_authenticated()
        browser = await get_or_create_browser()
        page = browser.page

        print(f"--- Navigating to {thread_url} ---")
        await page.goto(thread_url, wait_until="domcontentloaded")
        await page.wait_for_selector("main", timeout=15000)
        await asyncio.sleep(3)

        # Scroll up several times to load older history
        for i in range(5):
            await page.evaluate(
                """() => {
                    const main = document.querySelector('main');
                    if (!main) return;
                    const s = Array.from(main.querySelectorAll('*'))
                        .find(el => el.scrollHeight > el.clientHeight + 50 && getComputedStyle(el).overflowY !== 'visible');
                    if (s) s.scrollTop = 0;
                }"""
            )
            await asyncio.sleep(1.2)

        probe = await page.evaluate(PROBE_SCRIPT)
        (run_dir / "probe.json").write_text(
            json.dumps(probe, indent=2, ensure_ascii=False)
        )
        print(f"\nProbe saved to {run_dir}/probe.json")
        print(f"  count: {probe.get('count')}")
    finally:
        await close_browser()


if __name__ == "__main__":
    asyncio.run(main())
