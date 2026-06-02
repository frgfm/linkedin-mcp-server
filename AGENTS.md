# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

- Use `uv` for dependency management: `uv sync` (dev: `uv sync --group dev`)
- Lint: `uv run ruff check .` (auto-fix with `--fix`)
- Format: `uv run ruff format .`
- Type check: `uv run ty check` (using ty, not mypy)
- Tests: `uv run pytest` (with coverage: `uv run pytest --cov`)
- Pre-commit: `uv run pre-commit install` then `uv run pre-commit run --all-files`
- Run server locally: `uv run -m linkedin_mcp_server --no-headless`
- Run via uvx (PyPI/package verification only): `uvx linkedin-scraper-mcp`
- Docker build: `docker build -t linkedin-mcp-server .`
- Install browser: `uv run patchright install chromium`

## Scraping Rules

- **One section = one navigation.** Each entry in `PERSON_SECTIONS` / `COMPANY_SECTIONS` (`scraping/fields.py`) maps to exactly one page navigation. Never combine multiple URLs behind a single section.
- **Minimize DOM dependence.** Prefer innerText and URL navigation over DOM selectors. When DOM access is unavoidable, use minimal generic selectors (`a[href*="/jobs/view/"]`) — never class names tied to LinkedIn's layout.
- **Detection must be locale-independent.** Classification logic — connection state, action availability, button identity — must rely on URL patterns (`/preload/custom-invite/?vanityName=USER`, `/in/USER/edit/intro/`, `/messaging/compose/`), attribute *presence* (`aria-label` exists, `aria-expanded` exists, `aria-disabled` exists), or structural counts — never on text values like "Connect", "Follow", "Message", "1st", "Pending". The verb in an `aria-label` is locale-dependent; whether the attribute exists is not. Where text is genuinely the only signal, guard it behind an explicit per-locale table and document the limitation in code.

## Tool Return Format

All scraping tools return: `{url, sections: {name: raw_text}}`.

Optional additional keys:

- `references: {section_name: [{kind, url, text?, context?, value?}]}` — LinkedIn URLs are relative paths; `value` carries non-URL identifiers (e.g. company URN id for `kind: "company_urn"`)
- `section_errors: {section_name: {error_type, error_message, issue_template_path, runtime, ...}}`
- `unknown_sections: [name, ...]`
- `job_ids: [id, ...]` (search_jobs only)
- `references["feed"]` (get_feed only) — every entry is `kind: "feed_post"`; non-post anchors (sidebar profiles, employer logos) are filtered. URLs may carry either `/feed/update/<urn>/` (DOM-anchor-derived) or `/posts/<slug>` (SDUI-derived) form; both are valid LinkedIn permalinks. Cap is 50 entries, matching `get_feed`'s `num_posts` ceiling.

`get_feed`'s `sections["feed"]`, `get_conversation`, and `get_person_profile`'s `main_profile` section break the `{section_name: raw_text}` shape.

`get_feed` returns `sections["feed"]` as a list of structured posts (not raw text): `[{url, post_age, author: {name, profile_url, headline, degree}, content, is_promoted, media, reactions_count, comment_count, repost_count}, ...]`. `url` is a relative permalink (`/feed/update/<urn>/` from the container's `data-urn`, or `/posts/<slug>`) or `null` when none is exposed. `post_age` is the short relative-age token in the invitation format ("21min", "15h", "1d", "2mo", "1y") — the browser-side parser resolves the feed's locale-locked unit (bare `m` = minutes, `mo` = months); `null` when unparsed. `author.profile_url` is relative (`/in/<slug>` for people, `/company|showcase|school/<slug>` for Pages). `author.degree` is the connection-distance badge ("1st" / "2nd" / "3rd+"), detected structurally (bullet + ordinal); the ordinal suffix itself is en-US (BrowserManager lock); `null` for Pages, promoted, and self-authored posts. `author.headline` is the actor subline — a person's tagline or a Page's "<N> followers" text. `content` is the visible body up to the "see more" fold (trailing link-card CTA text is excluded); truncated bodies are not auto-expanded — fetch the `url` permalink for full text — and `content` is `null` when the post has no text body. `is_promoted` flags sponsored posts (detected via `data-urn` `sponsoredCreativeId` or the en-US "Promoted" marker). `media` is `null` or `{type: "link" | "image" | "video", url}` — only the first attachment is surfaced (priority video > link > image); `link` carries the external card destination as-is (`lnkd.in` shortlinks are not expanded), `image`/`video` carry the `media.licdn.com` source. `reactions_count` / `comment_count` / `repost_count` are integers (preferring the count buttons' `aria-label`, falling back to footer text; the named reactor in "X and N others reacted" counts toward the total → N+1) or `null` when LinkedIn renders no count. `sections["feed"]` is `[]` when extraction succeeded but the page rendered no posts — a rate-limited page surfaces `section_errors["feed"]` instead, never an empty list. Locale caveat: `is_promoted` and the engagement-count *text* fallback assume en-US (BrowserManager forces en-US); `degree` and `post_age` are structural/numeric in form (the digit and the named-reactor + N arithmetic are locale-independent) but their ordinal/age-unit tokens are en-US, safe under the same lock. `references["feed"]` is still emitted independently and unchanged.

`get_conversation` returns `sections` as `{"messages": [{timestamp, status, sender, content}, ...], "members": [{kind: "person", url?, name?, is_self}, ...]}`. Members are ordered with the authenticated user at index 0 (identified via the viewer URN harvested from `data-event-urn`); other participants follow in first-appearance order. `is_self` is always present and is `true` only on the authenticated user. `url` is omitted on the self member when they never appear as a sender anchor — the viewer URN is an internal `fsd_profile` identifier and not a guaranteed vanity slug, so we surface `is_self: true` alone rather than a synthetic URL. Message `sender` is an integer index into the `members` list — `0` is always self when detectable. `timestamp` is best-effort ISO 8601 reconstructed from LinkedIn's split day-heading + clock text (en-US only — LinkedIn does not expose `<time datetime>` for message events; "Today" / "Yesterday" are resolved against `datetime.now()`); per-minute message groups share one `<time>` in the DOM, and the parser inherits the running clock value within a day so later events in a group still get a precise timestamp. `status` is one of `sent` / `read` / `delivered` / `deleted`, with only `sent` and `deleted` reliably emitted today (deleted detection is en-US text equality on the recalled-body marker). `content` is the message body; messages that are purely a shared LinkedIn link card (no text body) emit the card's permalink (`/feed/update/<urn>/`, `/posts/<slug>`, `/jobs/view/<id>`, `/pulse/<slug>`) as `content`. Attachment-only events, quoted-reply enrichment, and unmodeled system events are deferred from V1.

`get_person_profile`'s `sections["main_profile"]` is a structured dict: `{name, headline, location, profile_picture_url, connection_count, follower_count, mutual_connection_count, main_organization, main_education, about, experience, education}`. Text fields are `null` when LinkedIn doesn't render them; counts are `null` when LinkedIn shows a non-exact value (e.g. `"500+ connections"` → `connection_count: null`). `mutual_connection_count` is the total of named-plus-other mutuals (LinkedIn renders this as "X and N other mutual connections"). `main_organization` and `main_education` are the textual names from the top-card buttons (`null` when not present). `experience` and `education` are lists of structured entries parsed from the entries visible on the main profile page after a deep scroll — for the full lists, request the `experience` and `education` sections (still raw text). Each experience entry carries `{title, organization, organization_url?, dates, location?, description?}`; each education entry carries `{school, school_url?, degree?, field_of_study?, dates?, description?}`. Count parsing and the "Present" date sentinel are en-US (BrowserManager forces en-US). Profile-picture URLs are absolute `media.licdn.com` displayphoto URLs; default-avatar profiles emit `null`. All other person sections (`experience`, `education`, `interests`, …) still return raw `innerText`.

## Verifying Bug Reports

Always verify scraping bugs end-to-end against live LinkedIn, not just code analysis. Use `uv run`, not `uvx`, so the running process reflects your workspace. Use `uvx` only for packaged distribution verification. For live Docker investigations, refresh the source session first with `uv run -m linkedin_mcp_server --login` before testing each materially different approach. Assume a valid login profile already exists at `~/.linkedin-mcp/profile/`.

```bash
# Start server
uv run -m linkedin_mcp_server --transport streamable-http --log-level DEBUG

# Initialize MCP session (grab Mcp-Session-Id from response headers)
curl -s -D /tmp/mcp-headers -X POST http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'

# Extract the session ID from saved headers
SESSION_ID=$(grep -i 'Mcp-Session-Id' /tmp/mcp-headers | awk '{print $2}' | tr -d '\r')

# Call a tool
curl -s -X POST http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Mcp-Session-Id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_person_profile","arguments":{"linkedin_username":"williamhgates","sections":"posts"}}}'
```

## Release Process

```bash
git checkout main && git pull
uv version --bump minor          # or: major, patch — updates pyproject.toml AND uv.lock
gt create -m "chore: Bump version to X.Y.Z"
gt submit                        # merge PR to trigger release workflow
```

The CI release workflow automatically updates `manifest.json` and `docker-compose.yml` with the new version — do not update them manually.

After the workflow completes, file a PR in the MCP registry to update the version.

## Commit Messages

- Follow conventional commits: `type(scope): subject`
- Types: feat, fix, docs, style, refactor, test, chore, perf, ci
- Keep subject <50 chars, imperative mood

## Development Workflow

Always read [`CONTRIBUTING.md`](CONTRIBUTING.md) before filing an issue or working on this repository.

- Write a short synthetic prompt that would reproduce the PR diff if given to a fresh Claude Code session. Don't copy the user's first message — distill the conversation into a single instruction that captures the full scope of changes. This tells the maintainer what was intended, which is often more useful than reviewing the full diff. Use a Markdown blockquote under a `## Synthetic prompt` heading, followed by the model attribution:
  ```
  ## Synthetic prompt

  > Add `skills` and `projects` sections to `get_person_profile`, following the certifications PR pattern. Update fields, tests, docs, and manifest.

  Generated with <model name and version>
  ```
- When implementing a new feature/fix:
  1. Check open issues. If no issue exists, create one following the templates in `.github/ISSUE_TEMPLATE/`. Fill in every section; delete optional sections if not applicable.
  2. Branch from `main`: `feature/issue-number-short-description`
  3. Implement and test
  4. Update README.md and docs/docker-hub.md if relevant
  5. Create a draft PR; only convert to regular PR when ready to merge
  6. Review with AI agents first, then manual review. Do not squash commits.

## PR Reviews

Greptile posts initial reviews as PR review comments, but follow-ups as **issue comments**. Always check both.

```bash
gh api repos/{owner}/{repo}/pulls/{pr}/reviews    # initial reviews
gh api repos/{owner}/{repo}/pulls/{pr}/comments   # inline comments
gh api repos/{owner}/{repo}/issues/{pr}/comments   # follow-up reviews
```

## btca

When you need up-to-date information about technologies used in this project, use btca to query source repositories directly.

```bash
btca resources                           # list available resources
btca ask -r <resource> -q "<question>"
btca ask -r fastmcp -r playwright -q "How do I set up browser context with FastMCP tools?"
```
