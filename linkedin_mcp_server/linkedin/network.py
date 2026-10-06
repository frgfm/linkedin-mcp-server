"""Network invitation and connection listing workflows."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import date
from typing import Any, Literal
from urllib.parse import quote_plus, urlparse

from patchright.async_api import TimeoutError as PlaywrightTimeoutError

from linkedin_mcp_server.core.utils import detect_rate_limit, handle_modal_close
from linkedin_mcp_server.linkedin.connection import ActionSignals
from linkedin_mcp_server.linkedin.conversation import (
    normalize_profile_url,
    parse_day_heading,
)
from linkedin_mcp_server.linkedin.navigation import PageNavigator
from linkedin_mcp_server.linkedin.session import ScrapingSession

logger = logging.getLogger(__name__)

_DIALOG_SELECTOR = 'dialog[open], [role="dialog"]'
_READ_MAIN_INNERTEXT_JS = """() => {
  const main = document.querySelector('main');
  return main ? (main.innerText || main.textContent || '') : '';
}"""

_INVITATION_CARDS_JS = r"""
({ kind, limit }) => {
  const root = document.querySelector('main') || document.body;
  if (!root) return [];

  const normalize = value => (value || '').replace(/\s+/g, ' ').trim();
  const ageUnits = 'min|mins|minute|minutes|h|hr|hrs|hour|hours|heure|heures|d|day|days|j|jour|jours|w|week|weeks|sem|semaine|semaines|m|mo|month|months|mois';
  const cardText = card => normalize(card.innerText || card.textContent);
  const linesFrom = el => {
    const text = el ? (el.innerText || el.textContent || '') : '';
    return text
      .split('\n')
      .map(normalize)
      .filter(Boolean);
  };
  const linkedInPath = href => {
    try {
      const url = new URL(href, location.origin);
      return `${url.pathname}${url.search}${url.hash}`;
    } catch {
      return '';
    }
  };
  const ageLineMatch = line => line.match(
    new RegExp(`^(?:(?:sent|envoyé|envoyée)\\s+)?(?:il\\s+y\\s+a\\s+)?(\\d+)\\s*(${ageUnits})(?:\\s+ago)?$`, 'i')
  );
  const ageFromLines = lines => {
    const exact = lines
      .map(ageLineMatch)
      .find(Boolean);
    const found = exact || normalize(lines.join(' ')).match(new RegExp(`\\b(\\d+)\\s*(${ageUnits})(?:\\s+ago)?\\b`, 'i'));
    if (!found) return null;
    const rawUnit = found[2].toLowerCase();
    const unit = rawUnit.startsWith('min')
      ? 'min'
      : rawUnit.startsWith('h')
      ? 'h'
      : rawUnit.startsWith('d') || rawUnit.startsWith('j')
        ? 'd'
        : rawUnit.startsWith('w') || rawUnit.startsWith('sem')
          ? 'w'
          : 'mo';
    return `${found[1]}${unit}`;
  };
  const mutualCountFromText = text => {
    const normalized = normalize(text);
    if (!/\b(mutual|relations?\s+en\s+commun)\b/i.test(normalized)) return 0;
    const otherMatch = normalized.match(
      /(\d[\d,.\s]*)\s+other(?:s)?(?:\s+mutual)?/i
    );
    if (otherMatch) {
      const count = Number.parseInt(otherMatch[1].replace(/[^\d]/g, ''), 10);
      return Number.isFinite(count) ? count + 1 : 1;
    }
    const countMatch = normalized.match(/(\d[\d,.\s]*)\s+mutual/i);
    if (countMatch) {
      const count = Number.parseInt(countMatch[1].replace(/[^\d]/g, ''), 10);
      return Number.isFinite(count) ? count : 0;
    }
    const frenchOtherMatch = normalized.match(
      /\bet\s+(\d[\d,.\s]*)\s+relations?\s+en\s+commun/i
    );
    if (frenchOtherMatch) {
      const count = Number.parseInt(frenchOtherMatch[1].replace(/[^\d]/g, ''), 10);
      return Number.isFinite(count) ? count + 1 : 1;
    }
    const frenchCountMatch = normalized.match(
      /(\d[\d,.\s]*)\s+relations?\s+en\s+commun/i
    );
    if (frenchCountMatch) {
      const count = Number.parseInt(frenchCountMatch[1].replace(/[^\d]/g, ''), 10);
      return Number.isFinite(count) ? count : 0;
    }
    if (/\brelations?\s+en\s+commun\b/i.test(normalized)) return 1;
    return 1;
  };
  const bestAnchorText = anchor => {
    const directText = normalize(anchor.textContent);
    if (directText) return directText;
    const imgAlt = anchor.querySelector('img[alt]')?.getAttribute('alt');
    return normalize(imgAlt) || null;
  };
  const isImageOnlyAnchor = anchor => {
    return !normalize(anchor.textContent) && !!anchor.querySelector('img[alt]');
  };
  const profileNameFromText = text => {
    if (!text) return null;
    const cleaned = normalize(text)
      .replace(/^profile photo of\s+/i, '')
      .replace(/^photo de profil de\s+/i, '')
      .replace(/\s*(?:'s|’s)\s+profile\s+(?:photo|picture)$/i, '')
      .replace(/\s+profile\s+(?:photo|picture)$/i, '')
      .replace(/\s+follows you\b.*$/i, '')
      .replace(/\s+is inviting you\b.*$/i, '')
      .replace(/\s+invited you\b.*$/i, '')
      .trim();
    return cleaned || null;
  };
  const senderNameFromProfileLink = link => {
    return profileNameFromText(link?.text);
  };
  const noteText = (card, buttonTexts) => {
    const noteRoot = card.querySelector(
      '[data-testid="expandable-text"], [data-testid="expandable-text-box"]'
    );
    if (!noteRoot) return null;
    const note = linesFrom(noteRoot)
      .filter(line => !buttonTexts.has(line))
      .join('\n');
    return note || null;
  };
  const headlineFromLines = (lines, senderName, note, buttonTexts) => {
    for (const line of lines) {
      if (line === senderName) continue;
      if (senderName && line.startsWith(senderName)) continue;
      if (line === note) continue;
      if (buttonTexts.has(line)) continue;
      if (ageLineMatch(line)) continue;
      if (/\b(mutual|relations?\s+en\s+commun)\b/i.test(line)) continue;
      if (/\b(follows you|invit|vous suit)\b/i.test(line)) continue;
      return line;
    }
    return null;
  };
  const personNameFromLines = (lines, profileLink, buttonTexts) => {
    for (const line of lines) {
      if (buttonTexts.has(line)) continue;
      if (ageLineMatch(line)) continue;
      if (/\b(mutual|relations?\s+en\s+commun)\b/i.test(line)) continue;
      if (/\b(follows you|invit|vous suit)\b/i.test(line)) continue;
      const candidate = profileNameFromText(line);
      if (candidate) return candidate;
    }
    return senderNameFromProfileLink(profileLink);
  };
  const visible = el => {
    if (!el || !el.getClientRects) return false;
    if (el.disabled) return false;
    const rects = el.getClientRects();
    if (!rects.length) return false;
    const style = window.getComputedStyle ? window.getComputedStyle(el) : null;
    return !style || (style.display !== 'none' && style.visibility !== 'hidden');
  };
  const identitySelector =
    'a[href*="/in/"], a[href*="/company/"], a[href*="/showcase/"], a[href*="/school/"], a[href*="/newsletters/"]';
  const actionControls = el => Array.from(
    el.querySelectorAll('button, [role="button"], a[aria-label][href]')
  )
    .filter(visible)
    .filter(button => !button.matches(identitySelector))
    .filter(button => button.getAttribute('data-testid') !== 'expandable-text-button')
    .filter(button => !button.closest('[data-testid="expandable-text-box"]'))
    .filter(button => button.matches('a[href]') || !button.closest('a[href]'))
    .filter(button => linesFrom(button).length > 0 || button.hasAttribute('aria-label'));
  const hasInvitationIdentity = el => !!el.querySelector(identitySelector);
  const cardForActionControl = button => {
    let el = button.parentElement;
    while (el && el !== root) {
      const actions = actionControls(el);
      if (
        visible(el) &&
        hasInvitationIdentity(el) &&
        cardText(el) &&
        actions.length > 0 &&
        actions.length <= 4
      ) {
        return el;
      }
      el = el.parentElement;
    }
    return null;
  };
  const outerInvitationCard = card => {
    if (!card) return null;
    // No-note connection requests render the Message link as a sibling of the
    // action row, so keep climbing until the parent contains another invite.
    let candidate = card;
    let el = card.parentElement;
    while (el && el !== root) {
      const actions = actionControls(el);
      if (actions.length > 4) break;
      if (
        visible(el) &&
        hasInvitationIdentity(el) &&
        cardText(el) &&
        actions.length > 0
      ) {
        candidate = el;
      }
      el = el.parentElement;
    }
    return candidate;
  };

  const cards = [];
  const seen = new WeakSet();
  for (const button of actionControls(root)) {
    const card = outerInvitationCard(cardForActionControl(button));
    if (!card || seen.has(card)) continue;
    seen.add(card);
    const rect = card.getBoundingClientRect();
    cards.push({ card, rect, index: cards.length });
  }
  cards.sort((a, b) => (
    (a.rect.top - b.rect.top) ||
    (a.rect.left - b.rect.left) ||
    (a.index - b.index)
  ));

  const result = [];
  for (const { card } of cards) {
    const cardLines = linesFrom(card);
    const buttonTexts = new Set(
      Array.from(card.querySelectorAll('button, [role="button"]')).flatMap(linesFrom)
    );
    const links = Array.from(card.querySelectorAll('a[href]')).map(link => ({
      anchor: link,
      path: linkedInPath(link.getAttribute('href') || link.href),
      text: bestAnchorText(link),
      image_only: isImageOnlyAnchor(link),
    }));
    const bestLink = predicate => {
      const matches = links.filter(predicate);
      return matches.find(link => !link.image_only) || matches[0];
    };
    const profileLink = bestLink(link => /^\/in\/[^/?#]+\/?/.test(link.path));
    const organizationLink = bestLink(link => /^\/(?:company|showcase|school)\/[^/?#]+\/?/.test(link.path));
    const pageLink = bestLink(link => /^\/(?:company|showcase)\/[^/?#]+\/?/.test(link.path));
    const newsletterLink = bestLink(link => /^\/newsletters\/[^/?#]+\/?/.test(link.path));
    const messageLink = links.find(link => /^\/messaging\/compose\//.test(link.path));
    const text = cardText(card);

    if (kind === 'sent') {
      if (!profileLink) continue;
      const recipientName = personNameFromLines(cardLines, profileLink, buttonTexts);
      result.push({
        type: 'connection_request',
        invitation_age: ageFromLines(cardLines),
        text,
        recipient: {
          name: recipientName,
          url: profileLink.path,
          headline: headlineFromLines(cardLines, recipientName, null, buttonTexts),
        },
      });
      if (limit && result.length >= limit) break;
      continue;
    }

    const type = newsletterLink
      ? 'newsletter_subscription'
      : pageLink
        ? 'page_follow'
        : 'connection_request';
    if (type === 'connection_request' && !profileLink) continue;
    if (type === 'page_follow' && !pageLink) continue;
    if (type === 'newsletter_subscription' && !newsletterLink) continue;

    const note = type === 'connection_request' ? noteText(card, buttonTexts) : null;
    const senderLink = type === 'newsletter_subscription'
      ? (organizationLink || profileLink)
      : profileLink;
    const senderName = type === 'newsletter_subscription'
      ? (senderLink?.text || null)
      : senderNameFromProfileLink(senderLink);
    result.push({
      type,
      invitation_age: ageFromLines(cardLines),
      text,
      sender: {
        name: senderName,
        url: senderLink?.path || null,
        headline: type === 'connection_request'
          ? headlineFromLines(cardLines, senderName, note, buttonTexts)
          : null,
        mutual_connections: type === 'connection_request'
          ? mutualCountFromText(text)
          : null,
      },
      note,
      target: type === 'connection_request'
        ? null
        : {
            page: type === 'page_follow'
              ? { name: pageLink.text, url: pageLink.path }
              : null,
            newsletter: type === 'newsletter_subscription'
              ? { title: newsletterLink.text, url: newsletterLink.path }
              : null,
          },
      message_url: type === 'connection_request' ? (messageLink?.path || null) : null,
    });
    if (limit && result.length >= limit) break;
  }
  return result;
}
"""

_ARCHIVE_CONVERSATION_JS = r"""
async (anchor) => {
  const visible = el => !el.disabled && el.getClientRects().length > 0;
  const text = el =>
    (el.getAttribute('aria-label') || el.innerText || el.textContent || '')
      .replace(/\s+/g, ' ').trim().toLowerCase();
  const root = anchor?.closest('[role="dialog"]') || document.querySelector('main');
  if (!root) return { clicked: false, verified: false, reason: 'no_root' };
  const event = root.querySelector('[data-event-urn^="urn:li:msg_message:"]');
  if (!event) return { clicked: false, verified: false, reason: 'no_messages' };

  const eventRect = event.getBoundingClientRect();
  // BrowserManager forces en-US, so LinkedIn's menu labels are stable here.
  const actionItems = () => Array.from(root.querySelectorAll(
    '[role="menuitem"], [role="menu"] button, [role="button"]'
  )).filter(visible);
  const findAction = action => actionItems().find(item =>
    text(item) === action || text(item) === `${action} conversation`
  );
  const menuButtons = Array.from(root.querySelectorAll(
    'button[aria-haspopup="menu"], button[aria-expanded]'
  ))
    .filter(visible)
    .filter(button => {
      const rect = button.getBoundingClientRect();
      return rect.left + rect.width / 2 >= eventRect.left;
    });
  for (const button of menuButtons) {
    if (findAction('restore')) {
      return {
        clicked: false,
        verified: true,
        alreadyArchived: true,
      };
    }
    button.click();
    for (let waits = 0; waits < 10; waits++) {
      await new Promise(resolve => setTimeout(resolve, 100));
      if (findAction('restore')) {
        return {
          clicked: false,
          verified: true,
          alreadyArchived: true,
        };
      }
      const archive = findAction('archive');
      if (!archive) continue;

      archive.click();
      let reopened = false;
      for (let verifies = 0; verifies < 20; verifies++) {
        await new Promise(resolve => setTimeout(resolve, 100));
        if (findAction('restore')) {
          return { clicked: true, verified: true, alreadyArchived: false };
        }
        if (
          !reopened &&
          verifies >= 2 &&
          button.isConnected &&
          button.getAttribute('aria-expanded') !== 'true'
        ) {
          button.click();
          reopened = true;
        }
      }
      return { clicked: true, verified: false };
    }
    if (button.getAttribute('aria-expanded') === 'true') button.click();
  }
  return { clicked: false, verified: false, reason: 'archive_action_not_found' };
}
"""

# Shared JS function that walks up from any /messaging/compose/ anchor
# inside <main> to find the smallest ancestor that satisfies the
# action-root predicate (>=2 interactive children, >=1 button). This is
# the top-card action row regardless of LinkedIn's class names.
#
# Inlined into both _ACTION_SIGNALS_JS and _OPEN_MORE_BUTTON_JS so a
# single change to the heuristic propagates to both call sites.
_FIND_ACTION_ROOT_FN_JS = r"""
function findActionRoot(main) {
  const composeAnchors = main.querySelectorAll('a[href*="/messaging/compose/"]');
  for (const a of composeAnchors) {
    let el = a.parentElement;
    while (el && el !== main) {
      const interactive = el.querySelectorAll('button, a').length;
      const buttons = el.querySelectorAll('button').length;
      if (interactive >= 2 && buttons >= 1) {
        return el;
      }
      el = el.parentElement;
    }
  }
  return null;
}
"""

# Shared JS function that fingerprints the incoming-request action row.
# Incoming-request profiles render no Message button in the top card, so
# findActionRoot (compose-anchor walk) cannot locate their action row and
# would mis-anchor on sidebar mutual-connection cards instead. This walk
# anchors on button[aria-expanded] (the More button) and validates the
# smallest multi-button ancestor against the fingerprint verified live
# 2026-06-11 on two German-locale incoming-request profiles:
#
#   [button aria-label (Accept)] [button aria-label (Ignore)]
#   [button aria-expanded, no aria-label (More)]
#
# All checks are attribute presence and structural counts per the
# AGENTS.md Scraping Rules — no label values are read. Every guard kills
# a known false positive: total-button-count === 3 and labeled === 2
# exclude video-player control bars (play/mute/captions all carry
# aria-label); the unlabeled-expander check excludes player settings
# expanders (the profile More button never carries aria-label); the
# DOM-order guard excludes bars with trailing labeled buttons; the
# compose/invite/labeled-anchor exclusions kill follow_only, pending,
# connected top cards and sidebar cards. The scan continues over ALL
# expander candidates because cover-video profiles render the player's
# expander before the top-card row in DOM order.
#
# The search is scoped to the top card — the first <section> of <main>
# (falling back to main's first child, then main). Profile pages render
# the action row in the top card; feed, "people also viewed", and other
# widgets live in later sections. Without the scope an unrelated widget
# elsewhere in main with the same button shape could be misclassified and
# its first labeled button clicked.
#
# Inlined into _ACTION_SIGNALS_JS and _CLICK_INCOMING_ACCEPT_JS so a
# single change to the fingerprint propagates to both call sites.
_FIND_INCOMING_ACTION_ROW_FN_JS = r"""
function findIncomingActionRow(main) {
  const scope = main.querySelector('section') || main.firstElementChild || main;
  const matches = [];
  for (const expander of scope.querySelectorAll('button[aria-expanded]')) {
    let el = expander.parentElement;
    while (el && el !== scope && el !== main) {
      if (el.querySelectorAll('button').length >= 2) {
        const buttons = el.querySelectorAll('button');
        const labeled = el.querySelectorAll('button[aria-label]');
        const expanders = el.querySelectorAll('button[aria-expanded]');
        if (
          buttons.length === 3 &&
          labeled.length === 2 &&
          expanders.length === 1 &&
          !expanders[0].hasAttribute('aria-label') &&
          expanders[0].compareDocumentPosition(labeled[1]) &
            Node.DOCUMENT_POSITION_PRECEDING &&
          !el.querySelector('a[href*="/messaging/compose/"]') &&
          !el.querySelector('a[href*="/preload/custom-invite/"]') &&
          !el.querySelector('a[aria-label]')
        ) {
          matches.push(el);
        }
        break;
      }
      el = el.parentElement;
    }
  }
  // Require a unique match: a profile's top card has exactly one action
  // row. Ambiguity (two rows matching the shape) is treated as no match so
  // the irreversible Accept click never fires on a guessed control.
  return matches.length === 1 ? matches[0] : null;
}
"""

# Locale-independent connection-state probe. Returns four booleans;
# per AGENTS.md Scraping Rules, every signal is based on URL patterns
# or ARIA-attribute *presence* — never on label text values.
#
# - hasInvite: vanityName-scoped invite anchor anywhere in document.
#   Searches document (not main) so a post-More-menu reread sees
#   portal-rendered menu items. The vanityName parameter is unique to
#   the target user, so document-wide search has no false-positive risk.
# - hasComposeInActionRoot: any /messaging/compose/ anchor exists inside
#   the action root. Scoped to main (not document) to avoid the More
#   menu's "Send profile in a message" anchor, which is a compose URL
#   but lives outside the action area.
# - hasEditIntro: edit-intro URL exists, only rendered on own profile.
# - hasLabeledActionButton: at least one <button[aria-label]> inside the
#   action root. Primary action buttons (Follow / Connect /
#   Save in Sales Navigator) carry aria-label for screen readers; the
#   profile More button uses aria-expanded instead and is not counted.
# - hasLabeledActionAnchor: at least one <a[aria-label]> inside the
#   action root. LinkedIn renders the Pending state as an anchor (linking
#   back to the profile URL) carrying aria-label like "Pending, click to
#   withdraw…". The Message anchor has only aria-disabled, so a labeled
#   anchor is the locale-independent Pending signal.
# - hasIncomingActionRow: the incoming-request fingerprint matched (see
#   _FIND_INCOMING_ACTION_ROW_FN_JS). Computed independently of
#   findActionRoot, which cannot locate the top-card row on incoming
#   profiles (no compose anchor there) and would mis-anchor on sidebar
#   cards.
#
# The username is CSS-escaped before interpolation into attribute
# selectors to defend against malformed inputs containing characters
# that would otherwise break the selector syntax (quotes, brackets).
_ACTION_SIGNALS_JS = (
    r"""
((username) => {
"""
    + _FIND_ACTION_ROOT_FN_JS
    + _FIND_INCOMING_ACTION_ROW_FN_JS
    + r"""
  const main = document.querySelector('main');
  if (!main) return null;

  const safe = CSS.escape(username);
  const inviteSel = `a[href*="/preload/custom-invite/?vanityName=${safe}"]`;
  const editSel = `a[href*="/in/${safe}/edit/intro/"]`;

  const hasInvite = !!document.querySelector(inviteSel);
  const hasEditIntro = !!main.querySelector(editSel);

  const actionRoot = findActionRoot(main);

  let hasComposeInActionRoot = false;
  let hasLabeledActionButton = false;
  let hasLabeledActionAnchor = false;
  if (actionRoot) {
    hasComposeInActionRoot =
      !!actionRoot.querySelector('a[href*="/messaging/compose/"]');
    for (const b of actionRoot.querySelectorAll('button')) {
      if (b.hasAttribute('aria-label')) {
        hasLabeledActionButton = true;
        break;
      }
    }
    for (const a of actionRoot.querySelectorAll('a')) {
      if (a.hasAttribute('aria-label')) {
        hasLabeledActionAnchor = true;
        break;
      }
    }
  }

  return {
    hasInvite,
    hasComposeInActionRoot,
    hasEditIntro,
    hasLabeledActionButton,
    hasLabeledActionAnchor,
    hasIncomingActionRow: !!findIncomingActionRow(main),
  };
})
"""
)

# Open the profile's More button, located inside the action root via the
# aria-expanded attribute. The aria-expanded attribute uniquely identifies
# the menu opener without text labels (the More button has no aria-label,
# while Follow/Connect/Pending buttons do — the inverse pattern). Returns
# true iff the click landed; the caller waits for [role='menu'] visibility
# before re-scanning signals.
_OPEN_MORE_BUTTON_JS = (
    r"""
(() => {
"""
    + _FIND_ACTION_ROOT_FN_JS
    + r"""
  const main = document.querySelector('main');
  if (!main) return false;
  const actionRoot = findActionRoot(main);
  if (!actionRoot) return false;
  const moreBtn = actionRoot.querySelector('button[aria-expanded]');
  if (!moreBtn) return false;
  moreBtn.click();
  return true;
})
"""
)

# Click the Pending control on the recipient's profile to open the
# withdraw-confirm dialog. The Pending state is rendered as the labeled
# action <a> inside the action root (see _ACTION_SIGNALS_JS comment block);
# clicking it is what LinkedIn's UI does when the user clicks "Pending" on
# a profile where they have an outstanding outgoing invitation. Caller
# verifies that the labeled anchor really is Pending (and not e.g. Message)
# by reading the connection state first via detect_connection_state.
_CLICK_PENDING_WITHDRAW_JS = (
    r"""
(() => {
"""
    + _FIND_ACTION_ROOT_FN_JS
    + r"""
  const main = document.querySelector('main');
  if (!main) return { found: false, clicked: false, reason: 'no_main' };
  const actionRoot = findActionRoot(main);
  if (!actionRoot) {
    return { found: false, clicked: false, reason: 'no_action_root' };
  }
  const anchor = actionRoot.querySelector('a[aria-label][href]');
  if (!anchor) {
    return { found: false, clicked: false, reason: 'no_labeled_anchor' };
  }
  anchor.click();
  return { found: true, clicked: true };
})
"""
)

# Click Accept on an incoming-request profile. Accept is the FIRST labeled
# button in the fingerprinted row — primary actions render first in
# top-card action rows (Connect/Message lead on other profile states; the
# inverse of dialogs, where the primary button renders last). Clicking the
# second button would silently and irreversibly Ignore the request, so the
# click only fires when the full fingerprint matched.
_CLICK_INCOMING_ACCEPT_JS = (
    r"""
(() => {
"""
    + _FIND_INCOMING_ACTION_ROW_FN_JS
    + r"""
  const main = document.querySelector('main');
  if (!main) return false;
  const row = findIncomingActionRow(main);
  if (!row) return false;
  row.querySelectorAll('button[aria-label]')[0].click();
  return true;
})
"""
)

# Click the Withdraw button inside the confirm dialog opened by clicking
# Pending. Three layered strategies, in order:
#   1. attr 'withdraw' — narrow, never matches Cancel.
#      (We deliberately do NOT match 'confirm' as an attr substring:
#       LinkedIn names the Cancel button with attrs like
#       "modal-confirm-cancel" or "confirm-dialog-cancel" — using
#       'confirm' as a needle silently misroutes the click onto Cancel.)
#   2. artdeco-button--primary class — LinkedIn's stable design-system
#      class for "this is the primary action of the modal". Locale-
#      independent. This is the most reliable signal once the close X is
#      out of the way.
#   3. Position: last visible non-dismiss button. Safe fallback when both
#      attribute-based strategies miss.
#
# The dialog selector prefers .artdeco-modal (LinkedIn's design-system
# class) and excludes aria-hidden containers — a defensive guard against
# matching dormant dialog wrappers that remain in the DOM after closing.
#
# The close X (.artdeco-modal__dismiss) is filtered before scoring, so no
# strategy can ever land on it.
#
# Every return path includes a diagnostic enumeration of the dialog's
# visible buttons (`dialog_buttons`) and the chosen target
# (`clicked_button`) — surfaced through to the tool result so misroutes
# are visible without re-running with extra logging.
_CLICK_WITHDRAW_CONFIRM_JS = r"""
() => {
  const visible = el => {
    if (el.disabled) return false;
    const rects = el.getClientRects ? el.getClientRects() : [];
    return rects.length > 0;
  };
  const stableAttrs = el => [
    el.getAttribute('data-control-name'),
    el.getAttribute('data-tracking-control-name'),
    el.getAttribute('data-test-id'),
    el.getAttribute('data-testid'),
    el.getAttribute('data-view-name'),
    el.getAttribute('name'),
    el.getAttribute('id'),
  ].filter(Boolean).join(' ').toLowerCase();
  const describe = el => ({
    tag: el.tagName.toLowerCase(),
    aria_label: el.getAttribute('aria-label'),
    text: ((el.innerText || el.textContent || '').trim()).slice(0, 80),
    data_control: el.getAttribute('data-control-name'),
    data_test:
      el.getAttribute('data-test-id') || el.getAttribute('data-testid'),
    data_view: el.getAttribute('data-view-name'),
    classes:
      typeof el.className === 'string' ? el.className.slice(0, 120) : null,
  });
  // Icon-only buttons (close X, dismiss chevrons, etc.) have an
  // aria-label but no visible text node. The withdraw/cancel actions
  // always carry a visible text label — filter icon-only buttons so we
  // never land on the close X even when LinkedIn obfuscates its class
  // name to a hash like `_94255844 _06957f2c`.
  const isTextBearing = b =>
    ((b.innerText || b.textContent || '').trim()).length > 0;
  const actionButtonsIn = dialog =>
    Array.from(dialog.querySelectorAll('button, [role="button"]'))
      .filter(visible)
      .filter(b => !b.classList.contains('artdeco-modal__dismiss'))
      .filter(isTextBearing);

  // LinkedIn often has multiple open dialogs in the DOM at once
  // (coachmarks, a11y skip-link wrappers, the actual modal). Score
  // candidates by the count of text-bearing action buttons and pick
  // the richest — that is, the one with actionable choices like
  // Withdraw + Cancel, never the chrome shell with only an X.
  const candidates = [
    ...document.querySelectorAll(
      '.artdeco-modal[role="dialog"]:not([aria-hidden="true"])'
    ),
    ...document.querySelectorAll('dialog[open]'),
    ...document.querySelectorAll('[role="dialog"]:not([aria-hidden="true"])'),
  ].filter(d => {
    const rects = d.getClientRects ? d.getClientRects() : [];
    return rects.length > 0;
  });
  const seen = new Set();
  const unique = [];
  for (const d of candidates) {
    if (seen.has(d)) continue;
    seen.add(d);
    unique.push(d);
  }

  const scored = unique
    .map(d => ({ dialog: d, buttons: actionButtonsIn(d) }))
    .sort((a, b) => b.buttons.length - a.buttons.length);

  const dialogEnumeration = scored.map(({ dialog, buttons }) => ({
    button_count: buttons.length,
    classes:
      typeof dialog.className === 'string'
        ? dialog.className.slice(0, 120)
        : null,
    aria_label: dialog.getAttribute('aria-label'),
    aria_labelledby: dialog.getAttribute('aria-labelledby'),
  }));

  if (scored.length === 0) {
    return {
      found: false,
      clicked: false,
      reason: 'no_dialog',
      candidate_dialogs: dialogEnumeration,
    };
  }

  const pick = scored[0];
  const buttons = pick.buttons;
  const enumeration = buttons.map(describe);

  if (buttons.length === 0) {
    // The richest dialog has no text-bearing actions yet — the modal
    // body has not rendered. The Python caller should retry.
    return {
      found: true,
      clicked: false,
      reason: 'no_buttons',
      dialog_buttons: enumeration,
      candidate_dialogs: dialogEnumeration,
    };
  }

  let target = buttons.find(b => stableAttrs(b).includes('withdraw'));
  let strategy = target ? 'attr_withdraw' : null;
  if (!target) {
    target = buttons.find(b =>
      b.classList.contains('artdeco-button--primary')
    );
    if (target) strategy = 'design_system_primary';
  }
  if (!target) {
    target = buttons[buttons.length - 1];
    strategy = 'position';
  }

  target.click();
  return {
    found: true,
    clicked: true,
    button_count: buttons.length,
    match_strategy: strategy,
    clicked_button: describe(target),
    dialog_buttons: enumeration,
    candidate_dialogs: dialogEnumeration,
  };
}
"""

# it is not a candidate). Diagnostics (`action_buttons`, `clicked_button`,
# `match_strategy`) are surfaced in every return path so misroutes are
# visible from the tool result without re-running with extra logging.
_CLICK_INCOMING_ACTION_JS = (
    r"""
((args) => {
  const { action, target_labels, other_labels } = args;
"""
    + _FIND_ACTION_ROOT_FN_JS
    + r"""
  const main = document.querySelector('main');
  if (!main) return { found: false, clicked: false, reason: 'no_main' };

  const visible = el => {
    if (el.disabled) return false;
    const rects = el.getClientRects ? el.getClientRects() : [];
    return rects.length > 0;
  };
  const isTextBearing = b =>
    ((b.innerText || b.textContent || '').trim()).length > 0;
  const buttonText = b =>
    ((b.innerText || b.textContent || '').trim()).toLowerCase();
  const stableAttrs = el => [
    el.getAttribute('data-control-name'),
    el.getAttribute('data-tracking-control-name'),
    el.getAttribute('data-test-id'),
    el.getAttribute('data-testid'),
    el.getAttribute('data-view-name'),
    el.getAttribute('name'),
    el.getAttribute('id'),
  ].filter(Boolean).join(' ').toLowerCase();
  const describe = el => ({
    tag: el.tagName.toLowerCase(),
    aria_label: el.getAttribute('aria-label'),
    text: ((el.innerText || el.textContent || '').trim()).slice(0, 80),
    data_control: el.getAttribute('data-control-name'),
    data_test:
      el.getAttribute('data-test-id') || el.getAttribute('data-testid'),
    data_view: el.getAttribute('data-view-name'),
    classes:
      typeof el.className === 'string' ? el.className.slice(0, 120) : null,
  });

  const targetSet = new Set(
    (target_labels || []).map(s => s.toLowerCase())
  );
  const otherSet = new Set(
    (other_labels || []).map(s => s.toLowerCase())
  );

  // --- Strategy 0: locale-table label scan over the entire <main> ----------
  // LinkedIn does not always render a Message button alongside an incoming
  // request, so a compose-anchor walk (findActionRoot) can return null
  // even when Accept + Ignore are clearly in the DOM. This first pass
  // sidesteps that: scan every visible text-bearing <button> in main and
  // pick the one whose normalized text matches one of the target action's
  // labels from INCOMING_REQUEST_LABELS. Per CLAUDE.md, this is the
  // sanctioned text-fallback path — gated through the same locale table
  // already used by detect_connection_state.
  const mainButtons = Array.from(
    main.querySelectorAll('button, [role="button"]')
  )
    .filter(visible)
    .filter(isTextBearing)
    .filter(b => !b.hasAttribute('aria-expanded'));

  const mainEnumeration = mainButtons.map(describe);
  let target = null;
  let strategy = null;

  if (targetSet.size > 0) {
    target = mainButtons.find(b => targetSet.has(buttonText(b)));
    if (target) strategy = 'label_text';
  }

  // --- Strategies 1-3: structural fallbacks scoped to the action root ------
  // If the label scan misses (locale not in the table, or LinkedIn
  // changed the label slightly), fall back to action-root structural
  // detection so we still have a chance of clicking the right button.
  let rootButtons = [];
  if (!target) {
    const actionRoot = findActionRoot(main);
    if (actionRoot) {
      rootButtons = Array.from(actionRoot.querySelectorAll('button'))
        .filter(visible)
        .filter(isTextBearing)
        .filter(b => !b.hasAttribute('aria-expanded'));

      // Exclude buttons whose text matches the *other* action's labels —
      // protects against accidentally clicking Ignore when we wanted Accept.
      const candidates = otherSet.size > 0
        ? rootButtons.filter(b => !otherSet.has(buttonText(b)))
        : rootButtons;

      // 1) Stable engineering attr substring.
      const token = action === 'accept' ? 'accept' : 'ignore';
      target = candidates.find(b => stableAttrs(b).includes(token));
      if (target) strategy = 'attr_' + action;

      // 2) Design-system class: Accept is primary, Ignore is the non-primary
      //    text-bearing sibling.
      if (!target && candidates.length >= 2) {
        if (action === 'accept') {
          target = candidates.find(b =>
            b.classList.contains('artdeco-button--primary')
          );
        } else {
          target = candidates.find(b =>
            !b.classList.contains('artdeco-button--primary')
          );
        }
        if (target) strategy = 'design_system_class';
      }

      // 3) Position fallback: documented UX layout [Ignore, Accept].
      if (!target && candidates.length === 2) {
        target = candidates[action === 'accept' ? 1 : 0];
        strategy = 'position';
      }
    }
  }

  const rootEnumeration = rootButtons.map(describe);

  if (!target) {
    return {
      found: false,
      clicked: false,
      reason: targetSet.size === 0 && rootButtons.length === 0
        ? 'no_action_root'
        : 'action_unavailable',
      main_buttons: mainEnumeration,
      action_buttons: rootEnumeration,
      target_labels: Array.from(targetSet),
    };
  }

  target.click();
  return {
    found: true,
    clicked: true,
    button_count: (strategy === 'label_text' ? mainButtons : rootButtons).length,
    match_strategy: strategy,
    clicked_button: describe(target),
    action_buttons: strategy === 'label_text' ? mainEnumeration : rootEnumeration,
    main_buttons: mainEnumeration,
  };
})
"""
)

_CLICK_RECEIVED_INVITATION_ACTION_JS = r"""
({ username, action, target_labels }) => {
  const main = document.querySelector('main');
  if (!main) return { found: false, clicked: false, reason: 'no_main' };

  const expected = `/in/${username.replace(/^\/+|\/+$/g, '')}/`;
  const visible = el => !el.disabled && el.getClientRects().length > 0;
  const text = el => (el.innerText || el.textContent || '').trim();
  const labels = new Set((target_labels || []).map(label => label.toLowerCase()));

  const profileLinks = Array.from(main.querySelectorAll('a[href*="/in/"]'))
    .filter(link => {
      try {
        return new URL(link.href, location.origin).pathname.startsWith(expected);
      } catch {
        return false;
      }
    });

  for (const link of profileLinks) {
    let card = link.parentElement;
    while (card && card !== main) {
      const buttons = Array.from(card.querySelectorAll('button, [role="button"]'))
        .filter(visible)
        .filter(button => text(button))
        .filter(button => !button.hasAttribute('aria-expanded'));
      if (buttons.length === 2) {
        // LinkedIn's invitation manager renders [Ignore, Accept]; the exact
        // two-button guard keeps this observed fallback scoped to that row.
        const target = buttons.find(button => labels.has(text(button).toLowerCase()))
          || buttons[action === 'accept' ? 1 : 0];
        target.click();
        return {
          found: true,
          clicked: true,
          match_strategy: labels.has(text(target).toLowerCase())
            ? 'label_text'
            : 'position',
          clicked_button: text(target),
          button_count: buttons.length,
        };
      }
      card = card.parentElement;
    }
  }
  return { found: false, clicked: false, reason: 'invitation_card_not_found' };
}
"""

_EXPAND_INVITATION_NOTES_JS = r"""
() => {
  const buttons = Array.from(
    document.querySelectorAll('[data-testid="expandable-text-button"]')
  );
  let clicked = 0;
  for (const button of buttons) {
    if (button.getAttribute('data-mcp-clicked') === '1') continue;
    if (button.getAttribute('aria-expanded') === 'true') continue;

    button.setAttribute('data-mcp-clicked', '1');
    button.style.pointerEvents = 'auto';
    button.dispatchEvent(new MouseEvent('click', {
      bubbles: true,
      cancelable: true,
      view: window,
    }));
    clicked += 1;
  }
  return clicked;
}
"""

_CONNECTION_CARDS_JS = r"""
({ limit }) => {
  const root = document.querySelector('main') || document.body;
  if (!root) return [];

  const normalize = value => (value || '').replace(/\s+/g, ' ').trim();
  const linesFrom = el => {
    const text = el ? (el.innerText || el.textContent || '') : '';
    return text.split('\n').map(normalize).filter(Boolean);
  };
  const linkedInPath = href => {
    try {
      const url = new URL(href, location.origin);
      return `${url.pathname}${url.search}${url.hash}`;
    } catch {
      return '';
    }
  };
  const visible = el => {
    if (!el || !el.getClientRects) return false;
    const rects = el.getClientRects();
    if (!rects.length) return false;
    const style = window.getComputedStyle ? window.getComputedStyle(el) : null;
    return !style || (style.display !== 'none' && style.visibility !== 'hidden');
  };
  const profileSlug = path => {
    const m = path && path.match(/^\/in\/([^/?#]+)/);
    return m ? decodeURIComponent(m[1]) : '';
  };
  const connectedOnRe = /\bconnected\s+(?:on|since)\b/i;

  // Climb from each profile anchor up to the smallest visible ancestor
  // whose text contains the en-US "Connected on/since ..." line and at
  // most one /in/ slug (this slug). The Connected-on line is the V1
  // contract: anchors whose surroundings lack it are not connection rows
  // (e.g. People-You-May-Know rails) and are skipped.
  const cardForAnchor = (anchor, slug) => {
    let el = anchor.parentElement;
    let depth = 0;
    while (el && el !== root && depth < 10) {
      if (visible(el)) {
        const text = el.innerText || el.textContent || '';
        if (text && text.length < 800 && connectedOnRe.test(text)) {
          const otherProfiles = Array.from(el.querySelectorAll('a[href*="/in/"]'))
            .filter(a => {
              const otherSlug = profileSlug(linkedInPath(a.getAttribute('href') || a.href));
              return otherSlug && otherSlug !== slug;
            });
          if (otherProfiles.length === 0) return el;
        }
      }
      el = el.parentElement;
      depth += 1;
    }
    return null;
  };

  const cards = [];
  const seenSlugs = new Set();
  for (const anchor of Array.from(root.querySelectorAll('a[href*="/in/"]')).filter(visible)) {
    const path = linkedInPath(anchor.getAttribute('href') || anchor.href);
    const slug = profileSlug(path);
    if (!slug) continue;
    if (seenSlugs.has(slug)) continue;
    const card = cardForAnchor(anchor, slug);
    if (!card) continue;

    // Identity is the URL; name comes from visible anchor text. If no
    // anchor in this card has visible text, skip rather than fall back
    // to ``img[alt]`` labels — those carry locale-specific cleanup rules
    // (English possessive, French "photo de profil de …") that the V1
    // en-US contract does not support.
    const nameAnchor = Array.from(card.querySelectorAll('a[href*="/in/"]'))
      .filter(a => profileSlug(linkedInPath(a.getAttribute('href') || a.href)) === slug && visible(a))
      .find(a => linesFrom(a).length > 0);
    if (!nameAnchor) continue;

    const nameAnchorLines = linesFrom(nameAnchor);
    const name = nameAnchorLines[0];
    // The connections page renders the headline as subsequent
    // ``innerText`` lines inside the same profile anchor. Anchor-line
    // parsing is sufficient on its own — no card-wide line fallback.
    const headline = nameAnchorLines.slice(1)
      .filter(line => line && line !== name)
      .join(' • ') || null;
    const connectedLine = linesFrom(card).find(line => connectedOnRe.test(line)) || null;

    seenSlugs.add(slug);
    cards.push({
      name,
      profile_url: `/in/${slug}/`,
      headline,
      connected_on_text: connectedLine,
    });
    if (limit && cards.length >= limit) break;
  }
  return cards;
}
"""


def _connection_result(
    url: str,
    status: str,
    message: str,
    *,
    note_sent: bool = False,
    profile: str = "",
) -> dict[str, Any]:
    """Build a structured response for a profile connection attempt."""
    result: dict[str, Any] = {
        "url": url,
        "status": status,
        "message": message,
        "note_sent": note_sent,
    }
    if profile:
        result["profile"] = profile
    return result


def _invitation_action_result(
    url: str,
    status: str,
    message: str,
    *,
    action: str,
    linkedin_username: str,
    profile_url: str = "",
    performed: bool = False,
    action_count: int | None = None,
    match_strategy: str | None = None,
) -> dict[str, Any]:
    """Build the response for an invitation action (accept / ignore / withdraw)."""
    result: dict[str, Any] = {
        "url": url,
        "status": status,
        "message": message,
        "action": action,
        "linkedin_username": linkedin_username,
        "performed": performed,
    }
    if profile_url:
        result["profile_url"] = profile_url
    if action_count is not None:
        result["action_count"] = action_count
    if match_strategy:
        result["match_strategy"] = match_strategy
    return result


def _normalize_invitation_username(value: str) -> str:
    """Normalize a username, /in/ path, or LinkedIn URL to the vanity username."""
    raw = value.strip()
    if not raw:
        return ""

    path_or_username = raw
    if "://" in raw:
        parsed = urlparse(raw)
        path_or_username = parsed.path
    elif raw.startswith("/"):
        path_or_username = urlparse(raw).path

    match = re.search(r"(?:^|/)in/([^/?#]+)/?", path_or_username)
    if match:
        return match.group(1).strip("/")
    return path_or_username.split("?", 1)[0].split("#", 1)[0].strip("/")


def _is_invitation_message_url(value: str) -> bool:
    """Return whether value is a relative invitation compose URL."""
    parsed = urlparse(value)
    return bool(
        not parsed.scheme
        and not parsed.netloc
        and parsed.path == "/messaging/compose/"
        and parsed.query
        and not parsed.fragment
    )


def _invitation_manager_url(kind: Literal["received", "sent"]) -> str:
    return f"https://www.linkedin.com/mynetwork/invitation-manager/{kind}/"


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_non_negative_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _normalize_invitation_age(*values: Any) -> str | None:
    for value in values:
        text = _optional_text(value)
        if not text:
            continue
        match = re.search(
            r"\b(\d+)\s*"
            r"(min|mins|minute|minutes|h|hr|hrs|hour|hours|heure|heures|"
            r"d|day|days|j|jour|jours|w|week|weeks|sem|semaine|semaines|"
            r"m|mo|month|months|mois)"
            r"(?:\s+ago)?\b",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        raw_unit = match.group(2).lower()
        if raw_unit.startswith("h"):
            unit = "h"
        elif raw_unit.startswith("min"):
            unit = "min"
        elif raw_unit.startswith(("d", "j")):
            unit = "d"
        elif raw_unit.startswith(("w", "sem")):
            unit = "w"
        else:
            unit = "mo"
        return f"{match.group(1)}{unit}"
    return None


def _invitation_mutual_connections(
    invitation_type: str, raw_sender: dict[str, Any], raw_text: Any
) -> int | None:
    if invitation_type != "connection_request":
        return None

    text = _optional_text(raw_text)
    if text and re.search(
        r"\b(mutual|relations?\s+en\s+commun)\b", text, flags=re.IGNORECASE
    ):
        other_match = re.search(
            r"(\d[\d,.\s]*)\s+other(?:s)?(?:\s+mutual)?",
            text,
            flags=re.IGNORECASE,
        )
        if other_match:
            return _coerce_non_negative_int(other_match.group(1).replace(",", "")) + 1

        count_match = re.search(
            r"(\d[\d,.\s]*)\s+mutual",
            text,
            flags=re.IGNORECASE,
        )
        if count_match:
            return _coerce_non_negative_int(count_match.group(1).replace(",", ""))

        french_other_match = re.search(
            r"\bet\s+(\d[\d,.\s]*)\s+relations?\s+en\s+commun",
            text,
            flags=re.IGNORECASE,
        )
        if french_other_match:
            return (
                _coerce_non_negative_int(french_other_match.group(1).replace(",", ""))
                + 1
            )

        french_count_match = re.search(
            r"(\d[\d,.\s]*)\s+relations?\s+en\s+commun",
            text,
            flags=re.IGNORECASE,
        )
        if french_count_match:
            return _coerce_non_negative_int(
                french_count_match.group(1).replace(",", "")
            )

        return 1

    explicit = raw_sender.get("mutual_connections")
    if explicit is not None:
        return _coerce_non_negative_int(explicit)

    return 0


def _invitation_entity(
    value: Any, *, label_key: Literal["name", "title"]
) -> dict[str, str | None] | None:
    if not isinstance(value, dict):
        return None
    label = _optional_text(
        value.get(label_key) or value.get("name") or value.get("title")
    )
    url = _optional_text(value.get("url"))
    if not label and not url:
        return None
    return {label_key: label, "url": url}


def _normalize_sent_structured_invitation(raw: dict[str, Any]) -> dict[str, Any] | None:
    invitation_type = _optional_text(raw.get("type"))
    if invitation_type != "connection_request":
        return None

    raw_recipient_value = raw.get("recipient")
    if isinstance(raw_recipient_value, dict):
        raw_recipient: dict[str, Any] = raw_recipient_value
    else:
        raw_sender_value = raw.get("sender")
        raw_recipient = raw_sender_value if isinstance(raw_sender_value, dict) else {}
    recipient = {
        "name": _optional_text(raw_recipient.get("name")),
        "url": _optional_text(raw_recipient.get("url")),
        "headline": _optional_text(raw_recipient.get("headline")),
    }
    if not (recipient["name"] or recipient["url"]):
        return None

    return {
        "type": "connection_request",
        "invitation_age": _normalize_invitation_age(
            raw.get("invitation_age"),
            raw.get("text"),
        ),
        "recipient": recipient,
    }


def _normalize_structured_invitation(
    raw: Any,
    *,
    kind: Literal["received", "sent"] = "received",
) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None

    if kind == "sent":
        return _normalize_sent_structured_invitation(raw)

    invitation_type = _optional_text(raw.get("type"))
    if invitation_type not in {
        "connection_request",
        "page_follow",
        "newsletter_subscription",
    }:
        return None

    raw_sender = raw.get("sender") if isinstance(raw.get("sender"), dict) else {}
    sender = {
        "name": _optional_text(raw_sender.get("name")),
        "url": _optional_text(raw_sender.get("url")),
        "headline": (
            _optional_text(raw_sender.get("headline"))
            if invitation_type == "connection_request"
            else None
        ),
        "mutual_connections": _invitation_mutual_connections(
            invitation_type,
            raw_sender,
            raw.get("text"),
        ),
    }

    raw_target = raw.get("target") if isinstance(raw.get("target"), dict) else {}
    page = _invitation_entity(raw_target.get("page"), label_key="name")
    newsletter = _invitation_entity(raw_target.get("newsletter"), label_key="title")
    target = (
        None
        if invitation_type == "connection_request"
        else {
            "page": page if invitation_type == "page_follow" else None,
            "newsletter": (
                newsletter if invitation_type == "newsletter_subscription" else None
            ),
        }
    )

    has_identity = (
        (invitation_type == "connection_request" and (sender["name"] or sender["url"]))
        or (invitation_type == "page_follow" and page is not None)
        or (invitation_type == "newsletter_subscription" and newsletter is not None)
    )
    if not has_identity:
        return None

    return {
        "type": invitation_type,
        "invitation_age": _normalize_invitation_age(
            raw.get("invitation_age"),
            raw.get("text"),
        ),
        "sender": sender,
        "note": (
            _optional_text(raw.get("note"))
            if invitation_type == "connection_request"
            else None
        ),
        "target": target,
        "message_url": (
            _optional_text(raw.get("message_url"))
            if invitation_type == "connection_request"
            else None
        ),
    }


def _invitation_identity_key(invitation: dict[str, Any]) -> tuple[str, str, str, str]:
    raw_sender = invitation.get("sender")
    sender: dict[str, Any] = raw_sender if isinstance(raw_sender, dict) else {}
    raw_recipient = invitation.get("recipient")
    recipient: dict[str, Any] = raw_recipient if isinstance(raw_recipient, dict) else {}
    raw_target = invitation.get("target")
    target: dict[str, Any] = raw_target if isinstance(raw_target, dict) else {}
    raw_page = target.get("page")
    page: dict[str, Any] = raw_page if isinstance(raw_page, dict) else {}
    raw_newsletter = target.get("newsletter")
    newsletter: dict[str, Any] = (
        raw_newsletter if isinstance(raw_newsletter, dict) else {}
    )
    return (
        str(invitation.get("type") or ""),
        str(recipient.get("url") or sender.get("url") or ""),
        str(page.get("url") or ""),
        str(newsletter.get("url") or ""),
    )


_CONNECTIONS_URL = "https://www.linkedin.com/mynetwork/invite-connect/connections/"

# en-US "Connected on/since" prefix followed by a strict day-heading
# shape (``Month DD`` with optional ``, YYYY``). Bounded capture stops at
# the heading itself so trailing separators (``·``, em-dash, etc.) do not
# leak into :func:`conversation.parse_day_heading`. Locale scope matches
# conversation.py's en-US assumption: dates we cannot parse surface as
# ``connected_on: null`` rather than raise.
_CONNECTED_ON_RE_EN_US = re.compile(
    r"\bconnected\s+(?:on|since)\s+([A-Za-z]{3,9}\s+\d{1,2}(?:,\s*\d{4})?)",
    flags=re.IGNORECASE,
)


def _parse_connected_on(value: Any) -> str | None:
    """Parse "Connected on Month DD, YYYY" → "YYYY-MM-DD" (en-US).

    Reuses :func:`conversation.parse_day_heading` for the day-heading tail
    so the en-US month table lives in exactly one place. Validation is via
    :class:`datetime.date`, which rejects impossible dates like Feb 30.
    Returns None for missing, non-en-US, or invalid values — never raises.
    """
    text = _optional_text(value)
    if not text:
        return None
    match = _CONNECTED_ON_RE_EN_US.search(text)
    if not match:
        return None
    parsed = parse_day_heading(match.group(1).strip())
    if not parsed:
        return None
    month, day, year = parsed
    if year is None:
        return None
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _normalize_connection(raw: Any) -> dict[str, Any] | None:
    """Normalize a raw connection card into the public record shape.

    Returns ``{name, url, headline, connected_on}`` or ``None`` when the
    profile URL is missing — every connection must be addressable.
    """
    if not isinstance(raw, dict):
        return None
    url = normalize_profile_url(_optional_text(raw.get("profile_url")))
    if not url:
        return None
    return {
        "name": _optional_text(raw.get("name")),
        "url": url,
        "headline": _optional_text(raw.get("headline")),
        "connected_on": _parse_connected_on(raw.get("connected_on_text")),
    }


def _connection_identity_key(connection: dict[str, Any]) -> tuple[str]:
    return (str(connection.get("url") or ""),)


def _normalize_csv(value: str, mapping: dict[str, str]) -> str:
    """Normalize a comma-separated filter value using the provided mapping."""
    parts = [v.strip() for v in value.split(",")]
    return ",".join(mapping.get(p, p) for p in parts)


def _encode_list_facet(values: list[str]) -> str:
    """Encode a list of string values for a LinkedIn people-search list facet.

    LinkedIn's people-search URL uses JSON-list encoded facets of the form
    ``["A","B"]``. This helper URL-encodes the rendered JSON so the final URL
    contains e.g. ``%5B%22F%22%5D`` for ``["F"]``.
    """
    return quote_plus(json.dumps(values, separators=(",", ":")))


class NetworkScraper:
    def __init__(self, session: ScrapingSession, navigator: PageNavigator):
        self._session = session
        self._navigator = navigator
        self._page = session.page

    async def _navigate_to_page(self, url: str) -> None:
        await self._navigator._navigate_to_page(url)

    async def _wait_for_main_text(self, *, log_context: str) -> None:
        try:
            await self._page.wait_for_function(
                "() => { const main = document.querySelector('main'); return !!main && main.innerText.length > 100; }",
                timeout=10000,
            )
        except PlaywrightTimeoutError:
            logger.debug("%s content did not appear", log_context)

    async def _scroll_main_scrollable_region(
        self,
        *,
        position: Literal["top", "bottom"],
        attempts: int = 3,
        pause_time: float = 0.5,
    ) -> None:
        script = """({position}) => {
            const main = document.querySelector('main');
            if (!main) return;
            const nodes = [main, ...main.querySelectorAll('*')];
            const target = nodes
                .filter(node => {
                    const style = getComputedStyle(node);
                    return ['auto', 'scroll'].includes(style.overflowY)
                        && node.scrollHeight > node.clientHeight + 20;
                })
                .sort((a, b) => b.scrollHeight - a.scrollHeight)[0] || main;
            target.scrollTop = position === 'top' ? 0 : target.scrollHeight;
        }"""
        for _ in range(attempts):
            await self._page.evaluate(script, {"position": position})
            await asyncio.sleep(pause_time)

    async def _read_main_innertext(self) -> str:
        try:
            value = await self._page.evaluate(_READ_MAIN_INNERTEXT_JS)
        except Exception:
            return ""
        return value if isinstance(value, str) else ""

    async def _load_profile_for_state(self, username: str) -> tuple[str, ActionSignals]:
        await self._navigate_to_page(f"https://www.linkedin.com/in/{username}/")
        await self._wait_for_main_text(log_context=f"Profile {username}")
        return await self._read_main_innertext(), await self._read_action_signals(
            username
        )

    async def _read_action_signals(self, username: str) -> ActionSignals:
        data = await self._page.evaluate(_ACTION_SIGNALS_JS, username)
        if not isinstance(data, dict):
            return ActionSignals(
                has_invite_anchor=False,
                has_compose_anchor_in_action_root=False,
                has_edit_intro_anchor=False,
                has_labeled_action_button=False,
                has_labeled_action_anchor=False,
                has_incoming_action_row=False,
            )
        return ActionSignals(
            has_invite_anchor=bool(data.get("hasInvite")),
            has_compose_anchor_in_action_root=bool(data.get("hasComposeInActionRoot")),
            has_edit_intro_anchor=bool(data.get("hasEditIntro")),
            has_labeled_action_button=bool(data.get("hasLabeledActionButton")),
            has_labeled_action_anchor=bool(data.get("hasLabeledActionAnchor")),
            has_incoming_action_row=bool(data.get("hasIncomingActionRow")),
        )

    async def _dialog_is_open(self, *, timeout: int = 1000) -> bool:
        locator = self._page.locator(_DIALOG_SELECTOR)
        try:
            if await locator.count() == 0:
                return False
            await locator.first.wait_for(state="visible", timeout=timeout)
            return True
        except Exception:
            return False

    async def get_pending_invitations(
        self,
        limit: int = 20,
        kind: Literal["received", "sent"] = "received",
    ) -> dict[str, Any]:
        """List pending LinkedIn network invitations (received or sent)."""
        url = _invitation_manager_url(kind)
        await self._navigate_to_page(url)
        await detect_rate_limit(self._page)
        await self._wait_for_main_text(log_context=f"Invitations ({kind})")
        await handle_modal_close(self._page)
        await self._expand_invitation_note_toggles()

        invitations: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str, str, str]] = set()
        max_scrolls = max(0, (limit + 4) // 5 - 1)
        for attempt in range(max_scrolls + 1):
            for invitation in await self._extract_invitation_cards(
                kind=kind,
                limit=limit,
            ):
                key = _invitation_identity_key(invitation)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                invitations.append(invitation)
                if len(invitations) >= limit:
                    break

            if len(invitations) >= limit or attempt >= max_scrolls:
                break

            moved = await self._scroll_invitation_manager_down()
            if not moved:
                break
            await self._expand_invitation_note_toggles()

        return {"url": url, "invitations": invitations}

    async def _scroll_invitation_manager_down(self) -> bool:
        """Scroll one viewport in the invitation manager without skipping rows."""
        try:
            moved = await self._page.evaluate(
                """() => {
                    const main = document.querySelector('main');
                    if (!main) return false;

                    const isScrollable = element => {
                        const style = window.getComputedStyle(element);
                        return (
                            (style.overflowY === 'auto' || style.overflowY === 'scroll') &&
                            element.scrollHeight > element.clientHeight + 20
                        );
                    };

                    const candidates = [main, ...main.querySelectorAll('*')].filter(isScrollable);
                    const target = candidates.sort(
                        (left, right) => right.scrollHeight - left.scrollHeight
                    )[0] || main;
                    const before = target.scrollTop || window.scrollY || 0;
                    const step = Math.max(Math.floor((target.clientHeight || window.innerHeight) * 0.85), 320);
                    if (target === main && !isScrollable(main)) {
                        window.scrollBy(0, step);
                        return window.scrollY > before;
                    }
                    target.scrollTop = Math.min(target.scrollTop + step, target.scrollHeight);
                    return target.scrollTop > before;
                }"""
            )
        except Exception:
            logger.debug("Invitation manager scroll failed", exc_info=True)
            return False
        await asyncio.sleep(0.5)
        return bool(moved)

    async def _expand_invitation_note_toggles(self) -> None:
        """Reveal truncated invitation notes using locale-independent test ids.

        LinkedIn renders invite notes as ordinary text after the inline
        expandable-text button is triggered. DOM access is unavoidable here:
        innerText alone only exposes the truncated preview, while the button's
        `data-testid` is stable across locales and avoids visible text matching.
        """
        for _ in range(2):
            try:
                clicked = await self._page.evaluate(_EXPAND_INVITATION_NOTES_JS)
            except Exception:
                logger.debug("Invitation note expansion failed", exc_info=True)
                return
            if not clicked:
                return
            await asyncio.sleep(0.5)

    async def _extract_invitation_cards(
        self,
        *,
        kind: Literal["received", "sent"],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Extract structured invitation cards from the invitation manager.

        DOM access is needed because invitation type, notes, message links, and
        page/newsletter targets are sibling elements inside each card. The
        classifier uses LinkedIn URL shapes instead of localized button text.
        """
        try:
            raw_cards = await self._page.evaluate(
                _INVITATION_CARDS_JS,
                {"kind": kind, "limit": min(limit * 2, 200)},
            )
        except Exception:
            logger.debug("Invitation card extraction failed", exc_info=True)
            return []
        if not isinstance(raw_cards, list):
            return []

        cards: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str, str, str]] = set()
        for raw in raw_cards:
            invitation = _normalize_structured_invitation(raw, kind=kind)
            if invitation:
                key = _invitation_identity_key(invitation)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                cards.append(invitation)
                if len(cards) >= limit:
                    break
        return cards

    async def get_connections(self, limit: int = 20) -> dict[str, Any]:
        """List the authenticated user's most recently added 1st-degree connections.

        Returns ``{url, connections}`` where each connection is
        ``{name, url, headline, connected_on}``. ``connected_on`` is an
        ISO date (``YYYY-MM-DD``) parsed from the en-US "Connected on
        Month DD, YYYY" line, or ``None`` for other locales / unparseable
        text. ``url`` is the relative ``/in/<slug>/`` profile path.
        """
        url = _CONNECTIONS_URL
        await self._navigate_to_page(url)
        await detect_rate_limit(self._page)
        await self._wait_for_main_text(log_context="Connections")
        await handle_modal_close(self._page)

        connections: list[dict[str, Any]] = []
        seen_keys: set[tuple[str]] = set()
        # Generous scroll cap; the ``len(connections) >= limit`` and
        # ``not moved`` guards below handle early termination so this cap
        # does not bake in a per-viewport card-density assumption.
        max_scrolls = 20
        for attempt in range(max_scrolls + 1):
            for connection in await self._extract_connection_cards(limit=limit):
                key = _connection_identity_key(connection)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                connections.append(connection)
                if len(connections) >= limit:
                    break

            if len(connections) >= limit or attempt >= max_scrolls:
                break

            moved = await self._scroll_connections_down()
            if not moved:
                break

        return {"url": url, "connections": connections}

    async def _scroll_connections_down(self) -> bool:
        """Scroll the connections page one viewport down.

        Aliases :meth:`_scroll_invitation_manager_down`: today both pages
        share the same "find the largest scrollable container inside
        ``<main>``" heuristic. Keeping a named entry point makes the
        connections call site read honestly and gives the two pages a
        cheap divergence path if their scroll containers ever differ.
        """
        return await self._scroll_invitation_manager_down()

    async def _extract_connection_cards(
        self,
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Extract structured connection cards from the connections page.

        DOM access is needed because each row's headline and "Connected on"
        line are siblings of the profile anchor, not attributes. Card
        boundaries are inferred by climbing the anchor's parents until a
        bounded-size ancestor containing only this connection's ``/in/``
        anchors is found — locale-independent.
        """
        try:
            raw_cards = await self._page.evaluate(
                _CONNECTION_CARDS_JS,
                {"limit": min(limit * 2, 200)},
            )
        except Exception:
            logger.debug("Connection card extraction failed", exc_info=True)
            return []
        if not isinstance(raw_cards, list):
            return []

        # ``_CONNECTION_CARDS_JS`` already dedupes by ``profileSlug`` so we
        # do not re-check identity here. Cross-scroll dedup is handled by
        # ``get_connections`` via ``_connection_identity_key``.
        cards: list[dict[str, Any]] = []
        for raw in raw_cards:
            connection = _normalize_connection(raw)
            if connection is None:
                continue
            cards.append(connection)
            if len(cards) >= limit:
                break
        return cards

    # Accept/ignore use the received invitation manager. Withdraw bypasses the
    # invitation manager entirely (see _withdraw_outgoing_invitation) because
    # the sent page is paginated and accounts with many outgoing requests can
    # require many scrolls to surface a specific recipient — the profile page
    # is one navigation and exposes the Pending control directly.
    _INVITATION_ACTIONS: frozenset[str] = frozenset({"accept", "ignore", "withdraw"})
    _INCOMING_SUCCESS_STATUS: dict[str, str] = {
        "accept": "accepted",
        "ignore": "ignored",
    }

    async def act_on_invitation(
        self,
        linkedin_username: str,
        action: Literal["accept", "ignore", "withdraw"],
    ) -> dict[str, Any]:
        """Accept, ignore, or withdraw a pending invitation.

        All three actions navigate to the counterparty's profile page and
        click the relevant top-card action button. Withdraw uses the
        Pending anchor (labeled <a> in the action root); accept/ignore
        use the Accept / Ignore buttons LinkedIn renders in the action
        root when the target has sent us an invitation. Profile
        navigation sidesteps the invitation-manager pagination cost for
        accounts with many outstanding requests in either direction.
        """
        if action not in self._INVITATION_ACTIONS:
            raise ValueError(
                f"Unknown invitation action: {action!r} "
                f"(expected one of: {sorted(self._INVITATION_ACTIONS)})"
            )

        if action == "withdraw":
            return await self._withdraw_outgoing_invitation(linkedin_username)
        return await self._respond_to_incoming_invitation(linkedin_username, action)

    async def _accept_incoming_invitation(
        self,
        linkedin_username: str,
    ) -> dict[str, Any]:
        """Accept a received invitation. Thin wrapper used by ``connect_with_person``."""
        return await self.act_on_invitation(linkedin_username, "accept")

    async def _withdraw_outgoing_invitation(
        self,
        linkedin_username: str,
    ) -> dict[str, Any]:
        """Withdraw an outgoing invitation via the recipient's profile page.

        One navigation: load ``/in/{username}/``, confirm the connection state
        is ``pending``, click the Pending action anchor (labeled <a> in the
        action root — locale-independent signal documented in
        :func:`_read_action_signals`), then click the primary button in the
        confirm dialog. Avoids the sent-invitations-manager pagination cost
        for accounts with many outstanding requests.
        """
        from linkedin_mcp_server.linkedin.connection import detect_connection_state

        username = _normalize_invitation_username(linkedin_username)
        url = f"https://www.linkedin.com/in/{username}/"
        profile_url = f"/in/{username}/" if username else ""

        if not username:
            return _invitation_action_result(
                url,
                "not_found",
                "LinkedIn username is required.",
                action="withdraw",
                linkedin_username=username,
                profile_url=profile_url,
            )

        # Lean gating: this is a write path that only needs the top-card
        # text (for the locale-table fallbacks in detect_connection_state)
        # and the action signals — not the full section extraction
        # pipeline. Avoid scrape_person's scroll + structured-extraction
        # + references cost; navigate, wait for main, read fresh
        # innerText + signals directly.
        page_text, signals = await self._load_profile_for_state(username)
        if not page_text:
            return _invitation_action_result(
                url,
                "not_found",
                f"Could not load profile for {username}.",
                action="withdraw",
                linkedin_username=username,
                profile_url=profile_url,
            )

        state = detect_connection_state(signals)
        if state == "already_connected":
            return _invitation_action_result(
                url,
                "already_connected",
                f"{username} is already a 1st-degree connection; nothing to withdraw.",
                action="withdraw",
                linkedin_username=username,
                profile_url=profile_url,
            )
        if state != "pending":
            return _invitation_action_result(
                url,
                "not_found",
                f"No outgoing connection request found for {username} (state={state}).",
                action="withdraw",
                linkedin_username=username,
                profile_url=profile_url,
            )

        try:
            click_result = await self._page.evaluate(_CLICK_PENDING_WITHDRAW_JS)
        except Exception:
            logger.debug("Pending anchor click failed", exc_info=True)
            click_result = {"found": False, "clicked": False}
        if not (isinstance(click_result, dict) and click_result.get("clicked")):
            return _invitation_action_result(
                url,
                "action_unavailable",
                "Could not find or click the Pending control on the profile.",
                action="withdraw",
                linkedin_username=username,
                profile_url=profile_url,
            )

        if not await self._dialog_is_open(timeout=5000):
            return _invitation_action_result(
                url,
                "action_unavailable",
                "Pending was clicked but the confirm dialog did not open.",
                action="withdraw",
                linkedin_username=username,
                profile_url=profile_url,
                performed=True,
            )

        # The dialog is open, but `_dialog_is_open` only checks for *any*
        # role=dialog node — LinkedIn's chrome layer can have empty
        # coachmark/wrapper dialogs in the DOM that match the selector
        # before our actual modal body mounts. Poll until at least one
        # candidate dialog contains a text-bearing, non-dismiss action
        # button (Withdraw / Cancel). This avoids clicking the close X
        # of a wrapper dialog that has only icon buttons.
        try:
            await self._page.wait_for_function(
                """() => {
                  const visible = el => {
                    if (el.disabled) return false;
                    const rects = el.getClientRects
                      ? el.getClientRects() : [];
                    return rects.length > 0;
                  };
                  const dialogs = [
                    ...document.querySelectorAll(
                      '.artdeco-modal[role="dialog"]:not([aria-hidden="true"])'
                    ),
                    ...document.querySelectorAll('dialog[open]'),
                    ...document.querySelectorAll(
                      '[role="dialog"]:not([aria-hidden="true"])'
                    ),
                  ];
                  for (const d of dialogs) {
                    const rects = d.getClientRects ? d.getClientRects() : [];
                    if (rects.length === 0) continue;
                    const buttons = Array.from(
                      d.querySelectorAll('button, [role="button"]')
                    )
                      .filter(visible)
                      .filter(b =>
                        !b.classList.contains('artdeco-modal__dismiss')
                      )
                      .filter(b =>
                        ((b.innerText || b.textContent || '').trim()).length
                          > 0
                      );
                    if (buttons.length >= 1) return true;
                  }
                  return false;
                }""",
                timeout=5000,
            )
        except PlaywrightTimeoutError:
            logger.debug(
                "Withdraw confirm action buttons never rendered for %s",
                username,
            )

        # Confirm via the targeted withdraw JS helper. Playwright's actionability
        # checks race
        # with the modal mount animation here (the same reason send_message
        # clicks Send via JS). Do not call _dismiss_dialog on failure: if
        # the confirm click didn't land, dismissing would actively close the
        # dialog without withdrawing. Let LinkedIn time out the modal.
        #
        # Retry loop: LinkedIn streams the modal body (Withdraw + Cancel
        # buttons) asynchronously after the dialog wrapper mounts. The JS
        # helper refuses to click when only icon-only buttons (the close
        # X) are visible — it returns `reason: 'no_buttons'` after filtering
        # text-bearing candidates. We re-evaluate every 500 ms until the
        # action buttons render or we exhaust the budget. 6 × 500 ms ≈ 3 s
        # on top of the 5 s wait_for_function above gives ~8 s total for
        # the modal body to mount, which has been enough on all profiles
        # observed in testing while keeping the failure path fast.
        confirm_result: dict[str, Any] = {"found": False, "clicked": False}
        for attempt in range(6):
            try:
                evaluated = await self._page.evaluate(_CLICK_WITHDRAW_CONFIRM_JS)
            except Exception:
                logger.debug(
                    "Withdraw confirm click failed (attempt %d)",
                    attempt,
                    exc_info=True,
                )
                evaluated = None
            if isinstance(evaluated, dict):
                confirm_result = evaluated
                # Success or a non-retryable failure (e.g. no_dialog) — stop.
                if confirm_result.get("clicked"):
                    break
                if confirm_result.get("reason") != "no_buttons":
                    break
            await asyncio.sleep(0.5)

        def _attach_confirm_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
            """Surface dialog enumeration so misroutes are visible without
            re-running with extra logging."""
            for key in (
                "match_strategy",
                "clicked_button",
                "dialog_buttons",
                "candidate_dialogs",
            ):
                value = confirm_result.get(key)
                if value is not None:
                    result[key] = value
            return result

        # Log the strategy + every visible button in the dialog so the
        # next layout change is debuggable from server logs alone.
        logger.info(
            "Withdraw confirm result for %s: strategy=%s clicked=%s buttons=%s",
            username,
            confirm_result.get("match_strategy"),
            confirm_result.get("clicked_button"),
            confirm_result.get("dialog_buttons"),
        )

        if not confirm_result.get("clicked"):
            return _attach_confirm_diagnostics(
                _invitation_action_result(
                    url,
                    "action_unavailable",
                    "Could not click the Withdraw confirm button in the dialog.",
                    action="withdraw",
                    linkedin_username=username,
                    profile_url=profile_url,
                    performed=True,
                )
            )

        try:
            await self._page.wait_for_selector(
                _DIALOG_SELECTOR, state="hidden", timeout=5000
            )
        except PlaywrightTimeoutError:
            logger.debug("Withdraw confirm dialog did not close")

        verified_signals = await self._read_action_signals(username)
        verified_state = detect_connection_state(verified_signals)
        if verified_state == "pending":
            return _attach_confirm_diagnostics(
                _invitation_action_result(
                    url,
                    "verification_failed",
                    f"Clicked withdraw, but {username} still shows as pending.",
                    action="withdraw",
                    linkedin_username=username,
                    profile_url=profile_url,
                    performed=True,
                )
            )

        return _attach_confirm_diagnostics(
            _invitation_action_result(
                url,
                "withdrawn",
                "Invitation withdrawn.",
                action="withdraw",
                linkedin_username=username,
                profile_url=profile_url,
                performed=True,
            )
        )

    async def _respond_via_received_invitations(
        self,
        username: str,
        action: Literal["accept", "ignore"],
    ) -> dict[str, Any]:
        """Handle profile states that render incoming invitations as Pending."""
        from linkedin_mcp_server.linkedin.connection import INCOMING_REQUEST_LABELS

        url = _invitation_manager_url("received")
        profile_url = f"/in/{username}/"
        pending = await self.get_pending_invitations(limit=100, kind="received")

        def has_target(invitations: Any) -> bool:
            return isinstance(invitations, list) and any(
                isinstance(invitation, dict)
                and _normalize_invitation_username(
                    ((invitation.get("sender") or {}).get("url") or "")
                )
                == username
                for invitation in invitations
            )

        if not has_target(pending.get("invitations")):
            return _invitation_action_result(
                url,
                "not_found",
                f"No received connection request found for {username}.",
                action=action,
                linkedin_username=username,
                profile_url=profile_url,
            )

        labels = [
            pair[0 if action == "accept" else 1]
            for pair in INCOMING_REQUEST_LABELS.values()
        ]
        try:
            click_result = await self._page.evaluate(
                _CLICK_RECEIVED_INVITATION_ACTION_JS,
                {
                    "username": username,
                    "action": action,
                    "target_labels": labels,
                },
            )
        except Exception:
            logger.debug("Invitation-manager %s click failed", action, exc_info=True)
            click_result = {"clicked": False}
        if not isinstance(click_result, dict) or not click_result.get("clicked"):
            return _invitation_action_result(
                url,
                "action_unavailable",
                f"Could not click the {action} button for {username}.",
                action=action,
                linkedin_username=username,
                profile_url=profile_url,
            )

        await asyncio.sleep(0.5)
        refreshed = await self.get_pending_invitations(limit=100, kind="received")
        if not has_target(refreshed.get("invitations")):
            return _invitation_action_result(
                url,
                self._INCOMING_SUCCESS_STATUS[action],
                f"Invitation {self._INCOMING_SUCCESS_STATUS[action]}.",
                action=action,
                linkedin_username=username,
                profile_url=profile_url,
                performed=True,
            )

        return _invitation_action_result(
            url,
            "verification_failed",
            f"Clicked {action}, but {username} still appears in received invitations.",
            action=action,
            linkedin_username=username,
            profile_url=profile_url,
            performed=True,
        )

    async def _respond_to_incoming_invitation(
        self,
        linkedin_username: str,
        action: Literal["accept", "ignore"],
    ) -> dict[str, Any]:
        """Accept or ignore an incoming invitation via the sender's profile.

        Mirrors ``_withdraw_outgoing_invitation``. Navigate to ``/in/{user}/``,
        verify the connection state is ``incoming_request`` (LinkedIn renders
        Accept + Ignore in the top-card action root when the target has sent
        us an invitation), and click the matching button via the layered
        strategies in :data:`_CLICK_INCOMING_ACTION_JS` (stable attrs →
        design-system primary class → documented [Ignore, Accept] position).
        Verification re-reads the connection state — after Accept it should
        flip to ``already_connected``; after Ignore the incoming-request
        signal should disappear.

        Profile navigation sidesteps invitation-manager pagination — same
        rationale as withdraw.
        """
        from linkedin_mcp_server.linkedin.connection import detect_connection_state

        success_status = self._INCOMING_SUCCESS_STATUS[action]
        username = _normalize_invitation_username(linkedin_username)
        url = f"https://www.linkedin.com/in/{username}/"
        profile_url = f"/in/{username}/" if username else ""

        if not username:
            return _invitation_action_result(
                url,
                "not_found",
                "LinkedIn username is required.",
                action=action,
                linkedin_username=username,
                profile_url=profile_url,
            )

        # Lean gating: write path; same rationale as
        # _withdraw_outgoing_invitation. Avoid scrape_person's full
        # section-extraction pipeline.
        page_text, signals = await self._load_profile_for_state(username)
        if not page_text:
            return _invitation_action_result(
                url,
                "not_found",
                f"Could not load profile for {username}.",
                action=action,
                linkedin_username=username,
                profile_url=profile_url,
            )
        state = detect_connection_state(signals)
        if state == "already_connected":
            # Accept on an already-connected profile is a no-op success;
            # ignore on the same is meaningless — both states return the
            # same short-circuit message.
            return _invitation_action_result(
                url,
                "already_connected",
                f"{username} is already a 1st-degree connection; "
                f"no incoming invitation to {action}.",
                action=action,
                linkedin_username=username,
                profile_url=profile_url,
            )
        if state == "pending":
            return await self._respond_via_received_invitations(username, action)
        if state != "incoming_request":
            return _invitation_action_result(
                url,
                "not_found",
                f"No incoming connection request found for {username} (state={state}).",
                action=action,
                linkedin_username=username,
                profile_url=profile_url,
            )

        # Pass the locale-gated label table to the JS so it can match
        # buttons by visible text (the strongest signal — works even when
        # the compose-anchor walk fails because LinkedIn renders the
        # incoming-request top-card without a Message button).
        from linkedin_mcp_server.linkedin.connection import (
            INCOMING_REQUEST_LABELS,
        )

        accept_labels = [pair[0] for pair in INCOMING_REQUEST_LABELS.values()]
        ignore_labels = [pair[1] for pair in INCOMING_REQUEST_LABELS.values()]
        target_labels = accept_labels if action == "accept" else ignore_labels
        other_labels = ignore_labels if action == "accept" else accept_labels

        try:
            click_result = await self._page.evaluate(
                _CLICK_INCOMING_ACTION_JS,
                {
                    "action": action,
                    "target_labels": target_labels,
                    "other_labels": other_labels,
                },
            )
        except Exception:
            logger.debug("Incoming-invitation %s click failed", action, exc_info=True)
            click_result = {"found": False, "clicked": False}
        if not isinstance(click_result, dict):
            click_result = {"found": False, "clicked": False}

        def _attach_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
            for key in (
                "match_strategy",
                "clicked_button",
                "action_buttons",
                "main_buttons",
                "button_count",
            ):
                value = click_result.get(key)
                if value is not None:
                    if key == "button_count" and isinstance(value, int):
                        result["action_count"] = value
                    else:
                        result[key] = value
            return result

        logger.info(
            "Incoming-invitation %s click for %s: strategy=%s clicked=%s buttons=%s",
            action,
            username,
            click_result.get("match_strategy"),
            click_result.get("clicked_button"),
            click_result.get("action_buttons"),
        )

        if not click_result.get("clicked"):
            return _attach_diagnostics(
                _invitation_action_result(
                    url,
                    "action_unavailable",
                    f"Could not click the {action} button on {username}'s profile.",
                    action=action,
                    linkedin_username=username,
                    profile_url=profile_url,
                )
            )

        # Verify the click landed by polling fresh page state. The DOM
        # does not update synchronously after click() — LinkedIn streams
        # the Accept/Ignore button removal — so we must re-read text and
        # signals on each attempt. Critically, we *re-fetch* main's
        # innerText: the original page_text variable is a frozen snapshot
        # from the first scrape, and reusing it would make
        # detect_connection_state return incoming_request forever
        # because _has_incoming_request_text is text-only and would
        # always see the stale Accept/Ignore lines.
        #
        # Budget: ~5 s (10 × 0.5 s) — empirically enough on every profile
        # observed in testing. Fast path: first iteration after the
        # immediate state flip exits the loop in <1 s.
        verified_state: str = "incoming_request"
        for _ in range(10):
            try:
                fresh_text = await self._page.evaluate(_READ_MAIN_INNERTEXT_JS)
            except Exception:
                fresh_text = ""
            if not isinstance(fresh_text, str):
                fresh_text = ""
            verified_signals = await self._read_action_signals(username)
            verified_state = detect_connection_state(verified_signals)
            if verified_state != "incoming_request":
                break
            await asyncio.sleep(0.5)

        if verified_state == "incoming_request":
            return _attach_diagnostics(
                _invitation_action_result(
                    url,
                    "verification_failed",
                    f"Clicked {action}, but {username} still shows an "
                    f"incoming invitation.",
                    action=action,
                    linkedin_username=username,
                    profile_url=profile_url,
                    performed=True,
                )
            )

        return _attach_diagnostics(
            _invitation_action_result(
                url,
                success_status,
                f"Invitation {success_status}.",
                action=action,
                linkedin_username=username,
                profile_url=profile_url,
                performed=True,
            )
        )
