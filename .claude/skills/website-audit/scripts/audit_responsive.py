#!/usr/bin/env python3
"""Audit responsive and mobile behaviour against the rendered page.

Implements the verification procedure in ``references/mobile.md`` §5: the V1-V8
viewport matrix, the Tier 1 hard failures from §1, and — the point of the whole
exercise — **every finding is attributed to the nearest identifiable section or
landmark**.

``mobile.md`` §5 states the reason plainly:

    A page can pass every page-level gate ... while three of its eleven sections
    are individually broken. Page-level ``scrollWidth`` is masked by any
    ancestor with ``overflow-x: hidden``, which most site shells set. **Every
    finding must name a section**, or it is not actionable and the developer
    will close it as "works on my phone."

Checks:

* §1.1 meta viewport (page-level, once)
* §1.2 horizontal overflow, with the offending elements named, scoped per section
* §1.3 reflow — 320 px content diffed against 1280 px to catch hidden content
* §1.4 target size — WCAG 2.2 SC 2.5.8 (24 x 24 CSS px) with the spacing and
  inline exceptions implemented, plus separate advisory counts at 44/48
* §1.5 text at 200% — root ``font-size: 32px`` re-run of the clipping detector
* §1.8 hover-only affordances, CSSOM candidates confirmed by a dispatched touch
* §1.9 inputs below 16 px that trigger iOS focus auto-zoom
* §2.1 text below the readability floor
* V7 — 1280 px at 400% page scale, to confirm V1 findings are zoom findings

Usage:
    ./audit_responsive.py https://example.com
    ./audit_responsive.py https://example.com --viewports V1,V3 --out ./shots
    ./audit_responsive.py https://example.com --json out/responsive.json
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _common import (
    Finding,
    Report,
    TargetUnreachable,
    base_parser,
    finish,
    import_playwright,
    launch_chromium,
    normalise_url,
    run_cli,
)

# --------------------------------------------------------------------------- #
# Thresholds — every number cited to references/mobile.md
# --------------------------------------------------------------------------- #

TARGET_SIZE_AA = 24        # §1.4 — WCAG 2.2 SC 2.5.8 Target Size (Minimum), AA
TARGET_SIZE_HIG = 44       # §1.4 — Apple HIG (advisory) / WCAG 2.5.5 AAA
TARGET_SIZE_MATERIAL = 48  # §1.4 — Material Design 3 (advisory)
TEXT_FAIL_PX = 14          # §2.1 — "fail body text < 14 px"
TEXT_WARN_PX = 16          # §2.1 — "warn 14-15 px, pass >= 16 px"
INPUT_MIN_PX = 16          # §1.9 — iOS Safari auto-zooms inputs below 16 CSS px
REFLOW_FLOOR_PX = 320      # §1.3 — WCAG 1.4.10 vertical-scroll floor
REFLOW_REFERENCE_PX = 1280  # §1.3 — 320 px == 1280 px at 400% zoom
ZOOM_FACTOR = 4            # §5.2 V7 — 400% page zoom

MOBILE_USER_AGENT = (
    "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/143.0.0.0 Mobile Safari/537.36"
)


@dataclass(frozen=True, slots=True)
class Viewport:
    """One row of the ``mobile.md`` §5.2 viewport matrix."""

    key: str
    width: int
    height: int
    dpr: float
    mobile: bool
    why: str
    root_font_size: str | None = None  # V8 only
    page_scale: int = 1                # V7 only

    @property
    def label(self) -> str:
        """Short human label used in the ``viewport`` field of every finding."""
        parts = [f"{self.width}x{self.height}", f"@{self.dpr:g}x"]
        if self.mobile:
            parts.append("isMobile,hasTouch")
        if self.page_scale != 1:
            parts.append(f"zoom {self.page_scale * 100}%")
        if self.root_font_size:
            parts.append(f"root {self.root_font_size}")
        return " ".join(parts)


# mobile.md §5.2, verbatim.
VIEWPORT_MATRIX: dict[str, Viewport] = {
    "V1": Viewport("V1", 320, 512, 2, True,
                   "WCAG 1.4.10 floor; equivalent to 1280 px at 400% zoom"),
    "V2": Viewport("V2", 360, 640, 3, True,
                   "most common Android logical width worldwide"),
    "V3": Viewport("V3", 390, 844, 3, True,
                   "iPhone 12-16 baseline; the 'it looks fine to me' width"),
    "V4": Viewport("V4", 412, 915, 2.6, True,
                   "large Android (Pixel class)"),
    "V5": Viewport("V5", 568, 320, 2, True,
                   "landscape; orientation (1.3.4) and the 256 px floor in 1.4.10"),
    "V6": Viewport("V6", 768, 1024, 2, True,
                   "tablet portrait"),
    "V7": Viewport("V7", 1280, 1024, 1, False,
                   "desktop equivalence for 1.4.10", page_scale=ZOOM_FACTOR),
    "V8": Viewport("V8", 390, 844, 3, True,
                   "WCAG 1.4.4 at 200% text", root_font_size="32px"),
}

DEFAULT_VIEWPORTS = ("V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8")


# --------------------------------------------------------------------------- #
# In-page JavaScript
#
# Each snippet is adapted from the detection code in references/mobile.md. The
# adaptation is always the same: results carry a `section` field resolved by
# `__auditSection`, which is installed by SECTION_SETUP_JS.
# --------------------------------------------------------------------------- #

# mobile.md §5.1 — the environment sanity gate. If touch/hover emulation is not
# active, every hover and target finding is invalid.
SANITY_GATE_JS = r"""() => ({
  layoutWidth: document.documentElement.clientWidth,
  innerWidth: window.innerWidth,
  dpr: devicePixelRatio,
  emulatingTouch: matchMedia('(pointer: coarse)').matches,
  emulatingNoHover: matchMedia('(hover: none)').matches,
  anyHover: matchMedia('(any-hover: hover)').matches,
  reducedMotion: matchMedia('(prefers-reduced-motion: reduce)').matches
})"""

# mobile.md §1.1 — meta viewport parsing.
VIEWPORT_META_JS = r"""() => {
  const m = document.querySelector('meta[name="viewport" i]');
  if (!m) return { pass: false, errors: ['MISSING_VIEWPORT_META'], warnings: [] };
  const p = {};
  for (const kv of m.content.toLowerCase().split(',')) {
    const [k, v] = kv.split('=').map(s => (s || '').trim());
    if (k) p[k] = v;
  }
  const errors = [], warnings = [];
  if (p.width !== 'device-width') errors.push('WIDTH_NOT_DEVICE_WIDTH:' + p.width);
  if (p['user-scalable'] === 'no' || p['user-scalable'] === '0')
    errors.push('USER_SCALABLE_NO');
  const max = parseFloat(p['maximum-scale']);
  if (!isNaN(max) && max < 2) errors.push('MAXIMUM_SCALE_LT_2:' + max);
  if (!isNaN(max) && max >= 2 && max < 5) warnings.push('MAXIMUM_SCALE_LT_5:' + max);
  const min = parseFloat(p['minimum-scale']);
  if (!isNaN(min) && min > 1) warnings.push('MINIMUM_SCALE_GT_1:' + min);
  const init = parseFloat(p['initial-scale']);
  if (!isNaN(init) && init !== 1) warnings.push('INITIAL_SCALE_NOT_1:' + init);
  return { content: m.content, parts: p, errors, warnings,
           optedIntoCover: p['viewport-fit'] === 'cover',
           pass: errors.length === 0 };
}"""

# mobile.md §5.0 — enumerate the sections once, freeze the list, and tag each
# root so every later check can resolve its own nearest section.
SECTION_SETUP_JS = r"""() => {
  const roots = [...document.querySelectorAll(
    'main > *, section, article, aside, header, footer, nav, [role=region], ' +
    '[data-section], form')]
    .filter(el => { const r = el.getBoundingClientRect();
      return r.height > 40 && getComputedStyle(el).display !== 'none'; });
  const seen = [], out = [];
  for (const el of roots) {
    if (seen.some(a => a.contains(el))) continue;   // drop nested duplicates
    seen.push(el);
    const r = el.getBoundingClientRect();
    const selector = el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') +
      (el.className && typeof el.className === 'string'
        ? '.' + el.className.trim().split(/\s+/).slice(0, 2).join('.') : '');
    el.setAttribute('data-audit-section', String(out.length));
    out.push({ index: out.length, selector,
      heading: el.querySelector('h1,h2,h3')?.innerText.trim().slice(0, 60) || null,
      topPx: Math.round(r.top + scrollY), heightPx: Math.round(r.height) });
  }
  window.__auditSections = out;
  window.__auditSection = el => {
    const host = el && el.closest ? el.closest('[data-audit-section]') : null;
    if (!host) return null;
    const i = Number(host.getAttribute('data-audit-section'));
    const s = out[i];
    return s ? (s.heading ? s.selector + ' — "' + s.heading + '"' : s.selector) : null;
  };
  window.__auditPath = el => { const seg = [];
    for (let n = el; n && n.nodeType === 1 && seg.length < 5; n = n.parentElement) {
      let s = n.tagName.toLowerCase();
      if (n.id) { seg.unshift(s + '#' + n.id); break; }
      if (n.classList.length) s += '.' + [...n.classList].slice(0, 3).join('.');
      seg.unshift(s);
    } return seg.join(' > '); };
  return out;
}"""

# mobile.md §1.2 — horizontal overflow. The ancestor walk is what stops the
# naive version producing false positives on visually clipped children.
OVERFLOW_JS = r"""() => {
  const de = document.documentElement, limit = de.clientWidth;
  const clipsX = el => /^(hidden|clip|auto|scroll)$/.test(getComputedStyle(el).overflowX);
  const culprits = [];
  for (const el of document.querySelectorAll('body *')) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) continue;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') continue;
    const over = Math.max(r.right - limit, -r.left);
    if (over <= 1) continue;
    let clipped = false;
    for (let p = el.parentElement; p && p !== de; p = p.parentElement)
      if (clipsX(p)) { clipped = true; break; }
    if (clipped) continue;                       // contained: not a real overflow
    culprits.push({
      selector: window.__auditPath(el), section: window.__auditSection(el),
      tag: el.tagName.toLowerCase(),
      rect: { left: Math.round(r.left), right: Math.round(r.right),
              width: Math.round(r.width) },
      overflowPx: Math.round(over),
      cssWidth: cs.width, cssMinWidth: cs.minWidth, cssMaxWidth: cs.maxWidth,
      position: cs.position,
      suspectMinWidthAuto: (cs.minWidth === 'auto' || cs.minWidth === '0px') &&
        /flex|grid/.test(getComputedStyle(el.parentElement || de).display),
      suspect100vw: /100vw/.test(el.getAttribute('style') || '')
    });
  }
  culprits.sort((a, b) => b.overflowPx - a.overflowPx);
  return { documentOverflows: de.scrollWidth > limit + 1,
           scrollWidth: de.scrollWidth, clientWidth: limit,
           excessPx: de.scrollWidth - limit, culprits: culprits.slice(0, 40) };
}"""

# mobile.md §1.4 — target size. Implements the Spacing and Inline exceptions;
# Equivalent, User Agent Control and Essential need human judgement and are
# surfaced as "review" rather than "fail".
TARGET_SIZE_JS = r"""(MIN) => {
  const SEL = 'a[href], button, input:not([type=hidden]), select, textarea, summary,' +
    '[role=button], [role=link], [role=checkbox], [role=radio], [role=tab],' +
    '[role=menuitem], [role=switch], [role=option], [onclick], [tabindex]:not([tabindex="-1"])';
  const visible = el => { const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' ||
        cs.pointerEvents === 'none') return false;
    if (parseFloat(cs.opacity) === 0) return false;
    const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
  const isInlineInText = el => { const cs = getComputedStyle(el);
    if (!/^inline/.test(cs.display)) return false;
    const p = el.parentElement; if (!p) return false;
    const own = (el.textContent || '').trim().length;
    const all = (p.textContent || '').trim().length;
    return all > own + 20; };
  const NATIVE_REVIEW = /^(date|datetime-local|month|week|time|color|file|range)$/;
  const targets = [...document.querySelectorAll(SEL)].filter(visible).map(el => {
    const r = el.getBoundingClientRect();
    return { el, r, cx: r.left + r.width / 2, cy: r.top + r.height / 2,
             undersized: r.width < MIN || r.height < MIN, inline: isInlineInText(el),
             nativeControl: el.tagName === 'INPUT' && NATIVE_REVIEW.test(el.type || '') };
  });
  const intersectsRect = (cx, cy, rad, r) =>
    Math.hypot(Math.max(r.left - cx, 0, cx - r.right),
               Math.max(r.top - cy, 0, cy - r.bottom)) < rad;
  const failures = [], review = [];
  for (const t of targets) {
    if (!t.undersized || t.inline) continue;
    const r0 = MIN / 2;                                   // 24px-diameter circle
    const conflicts = targets.filter(o => {
      if (o === t) return false;
      if (o.el.contains(t.el) || t.el.contains(o.el)) return false;
      if (o.undersized) return Math.hypot(o.cx - t.cx, o.cy - t.cy) < MIN;
      return intersectsRect(t.cx, t.cy, r0, o.r);
    });
    if (conflicts.length === 0) continue;                 // passes via Spacing
    const record = { tag: t.el.tagName.toLowerCase(),
      selector: window.__auditPath(t.el), section: window.__auditSection(t.el),
      text: (t.el.innerText || t.el.getAttribute('aria-label') || '').trim().slice(0, 60),
      size: [Math.round(t.r.width), Math.round(t.r.height)],
      conflicts: conflicts.length };
    (t.nativeControl ? review : failures).push(record);
  }
  const below44 = targets.filter(t =>
    !t.undersized && !t.inline && (t.r.width < 44 || t.r.height < 44));
  const below48 = targets.filter(t =>
    !t.undersized && !t.inline && (t.r.width < 48 || t.r.height < 48));
  return { total: targets.length, failures, review,
           below44Advisory: below44.length, below48Advisory: below48.length };
}"""

# mobile.md §2.1 — computed text size and the share of visible text affected.
TEXT_SIZE_JS = r"""() => {
  const seen = new Map();
  const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let n; (n = w.nextNode()); ) {
    const t = n.nodeValue.trim(); if (t.length < 4) continue;
    const el = n.parentElement;
    if (!el || /^(SCRIPT|STYLE|NOSCRIPT)$/.test(el.tagName)) continue;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') continue;
    const px = parseFloat(cs.fontSize);
    const section = window.__auditSection(el);
    const key = el.tagName + '|' + px + '|' + cs.fontWeight + '|' + section;
    const rec = seen.get(key) || { tag: el.tagName.toLowerCase(), px, section,
      selector: window.__auditPath(el), chars: 0, sample: t.slice(0, 60) };
    rec.chars += t.length; seen.set(key, rec);
  }
  const all = [...seen.values()], total = all.reduce((a, r) => a + r.chars, 0) || 1;
  all.forEach(r => r.pctOfText = +(100 * r.chars / total).toFixed(1));
  return { runs: all.filter(r => r.px < 16).sort((a, b) => b.pctOfText - a.pctOfText),
           totalChars: total };
}"""

# mobile.md §1.9 — form controls whose rendered font-size triggers iOS auto-zoom.
INPUT_FONT_JS = r"""(MIN) => [...document.querySelectorAll(
  'input:not([type=hidden]):not([type=checkbox]):not([type=radio])' +
  ':not([type=range]):not([type=color]), select, textarea')]
  .filter(el => getComputedStyle(el).display !== 'none')
  .map(el => ({ tag: el.tagName.toLowerCase(), type: el.type || null,
                name: el.name || el.id || null,
                selector: window.__auditPath(el), section: window.__auditSection(el),
                fontSizePx: parseFloat(getComputedStyle(el).fontSize),
                transform: getComputedStyle(el).transform !== 'none'
                  ? getComputedStyle(el).transform : null }))
  .filter(r => r.fontSizePx < MIN)"""

# mobile.md §1.8 — hover-only affordances, from the CSSOM.
HOVER_ONLY_JS = r"""() => {
  const VISUAL = /^(display|visibility|opacity|max-height|height|transform|clip-path|pointer-events|content-visibility|width|max-width)$/;
  const rules = [];
  for (const sheet of document.styleSheets) {
    let list; try { list = sheet.cssRules; } catch { continue; }   // cross-origin
    const walk = rs => { for (const r of rs) {
      if (r.cssRules) { walk(r.cssRules); continue; }
      if (r.selectorText) rules.push(r);
    }}; walk(list);
  }
  const hoverRules = rules.filter(r => /:hover\b/.test(r.selectorText));
  const focusSelectors = new Set(rules
    .filter(r => /:focus(-within|-visible)?\b|\[aria-expanded|\[data-(open|expanded|active)|\.(is-|has-)?(open|active|expanded)/.test(r.selectorText))
    .map(r => r.selectorText.replace(/:focus(-within|-visible)?|\[aria-expanded[^\]]*\]|\[data-[^\]]*\]|\.(is-|has-)?(open|active|expanded)/g, '').trim()));
  const findings = [];
  for (const r of hoverRules) {
    const props = [...r.style].filter(p => VISUAL.test(p));
    if (!props.length) continue;                   // ignore decorative colour hovers
    const base = r.selectorText.replace(/:hover/g, '').trim();
    const hasFocusEquivalent = [...focusSelectors]
      .some(f => f && (f.includes(base) || base.includes(f)));
    const reveals = props.some(p => { const v = r.style.getPropertyValue(p);
      return (p === 'display' && v !== 'none') || (p === 'visibility' && v === 'visible') ||
             (p === 'opacity' && parseFloat(v) > 0) || (p === 'max-height' && v !== '0px') ||
             (p === 'pointer-events' && v === 'auto'); });
    if (!reveals || hasFocusEquivalent) continue;
    let matched = [];
    try { matched = [...document.querySelectorAll(base)].slice(0, 5); } catch { }
    findings.push({ selector: r.selectorText, properties: props,
                    matchedElements: matched.length,
                    sections: [...new Set(matched.map(e => window.__auditSection(e))
                                                 .filter(Boolean))],
                    href: r.parentStyleSheet?.href || 'inline' });
  }
  const inlineHandlers = [...document.querySelectorAll('[onmouseover],[onmouseenter]')]
    .filter(el => !el.onclick && !el.getAttribute('onfocus') &&
                  !el.getAttribute('ontouchstart'))
    .map(el => ({ selector: window.__auditPath(el),
                  section: window.__auditSection(el) }));
  return { hoverOnlyRules: findings, mouseOnlyInlineHandlers: inlineHandlers };
}"""

# mobile.md §1.3 — visible text, for the 320 vs 1280 content diff.
VISIBLE_TEXT_JS = r"""() => [...document.body.querySelectorAll('*')]
  .filter(e => { const cs = getComputedStyle(e);
    return cs.display !== 'none' && cs.visibility !== 'hidden' &&
           parseFloat(cs.opacity) !== 0; })
  .map(e => (e.childNodes[0]?.nodeType === 3 ? e.childNodes[0].nodeValue.trim() : ''))
  .filter(Boolean)"""

# mobile.md §1.5 — clipping and overflow deltas at 200% text.
TEXT_200_JS = r"""() => {
  const de = document.documentElement, prev = de.style.fontSize;
  const clipped = () => [...document.querySelectorAll('*')].filter(e =>
    (e.scrollHeight > e.clientHeight + 2 || e.scrollWidth > e.clientWidth + 2) &&
    /hidden|clip/.test(getComputedStyle(e).overflow));
  const before = { count: clipped().length, sw: de.scrollWidth };
  de.style.fontSize = '32px'; void document.body.offsetHeight;
  const afterEls = clipped();
  const after = { count: afterEls.length, sw: de.scrollWidth };
  const sections = [...new Set(afterEls.map(e => window.__auditSection(e))
                                       .filter(Boolean))].slice(0, 20);
  de.style.fontSize = prev;
  return { before, after, newlyClipped: after.count - before.count,
           newHorizontalOverflow: after.sw > before.sw + 1, sections };
}"""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def slugify(text: str) -> str:
    """Reduce *text* to something safe for a filename or a finding id."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "page"


def open_page(browser: Any, viewport: Viewport, url: str, timeout: float) -> Any:
    """Create a context for *viewport*, navigate, and return the page.

    Raises:
        TargetUnreachable: If navigation fails.
    """
    context = browser.new_context(
        viewport={"width": viewport.width, "height": viewport.height},
        device_scale_factor=viewport.dpr,
        is_mobile=viewport.mobile,
        has_touch=viewport.mobile,
        user_agent=MOBILE_USER_AGENT if viewport.mobile else None,
    )
    page = context.new_page()
    try:
        page.goto(url, wait_until="load", timeout=timeout * 1000)
    except Exception as exc:  # noqa: BLE001 - normalised into an audit error
        context.close()
        raise TargetUnreachable(url, str(exc).splitlines()[0]) from exc

    if viewport.page_scale != 1:
        # mobile.md §5.2 V7 — 400% page zoom via CDP, the desktop equivalence
        # test for WCAG 1.4.10.
        session = context.new_cdp_session(page)
        session.send("Emulation.setPageScaleFactor", {"pageScaleFactor": viewport.page_scale})
    if viewport.root_font_size:
        # mobile.md §5.2 V8 — WCAG 1.4.4 at 200%.
        page.evaluate("size => { document.documentElement.style.fontSize = size; }",
                      viewport.root_font_size)

    page.wait_for_timeout(400)  # let post-load layout and web fonts settle
    page.evaluate(SECTION_SETUP_JS)
    return page


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def check_viewport_meta(report: Report, page: Any, url: str, viewport: Viewport) -> None:
    """mobile.md §1.1 — page-level, recorded once."""
    result = page.evaluate(VIEWPORT_META_JS)
    fix = ('Use `<meta name="viewport" content="width=device-width, initial-scale=1">`. '
           "Do not disable zoom; the fix for iOS input auto-zoom is 16 px inputs, not "
           "user-scalable=no.")
    for error in result.get("errors", []):
        code = error.split(":")[0]
        severity, sc = "blocker", None
        if code in ("USER_SCALABLE_NO", "MAXIMUM_SCALE_LT_2"):
            sc = "1.4.4 Resize Text (AA)"
        report.add(Finding(
            id=f"responsive.viewport-meta.{slugify(code)}",
            severity=severity, rule="meta-viewport", wcag_sc=sc, url=url,
            viewport=viewport.label, selector='meta[name="viewport"]', section="document head",
            message=f"Viewport meta failure: {error}.",
            evidence={"content": result.get("content"), "code": error},
            how_to_fix=fix,
        ))
    for warning in result.get("warnings", []):
        report.add(Finding(
            id=f"responsive.viewport-meta.{slugify(warning)}",
            severity="advisory", rule="meta-viewport", wcag_sc=None, url=url,
            viewport=viewport.label, selector='meta[name="viewport"]',
            section="document head",
            message=f"Viewport meta smell: {warning}.",
            evidence={"content": result.get("content"), "code": warning},
            how_to_fix=fix,
        ))


def check_overflow(report: Report, page: Any, url: str, viewport: Viewport) -> dict[str, Any]:
    """mobile.md §1.2 — one finding per offending section, naming the top culprit."""
    result = page.evaluate(OVERFLOW_JS)
    culprits = result["culprits"]
    if not culprits:
        return result

    # 320 px is the WCAG 1.4.10 floor: overflow there is a violation, not a smell.
    severity = "blocker" if viewport.width <= REFLOW_FLOOR_PX else "major"
    by_section: dict[str, list[dict[str, Any]]] = {}
    for culprit in culprits:
        by_section.setdefault(culprit["section"] or "(unsectioned)", []).append(culprit)

    for section, items in by_section.items():
        worst = max(items, key=lambda c: c["overflowPx"])
        hints = []
        if worst["suspectMinWidthAuto"]:
            hints.append("flex/grid child with min-width:auto — add `min-width: 0`")
        if worst["suspect100vw"]:
            hints.append("100vw ignores the scrollbar — use 100% instead")
        report.add(Finding(
            id=f"responsive.overflow.{slugify(section)}.{viewport.key.lower()}",
            severity=severity, rule="horizontal-overflow",
            wcag_sc="1.4.10 Reflow (AA)", url=url, viewport=viewport.label,
            selector=worst["selector"], section=section,
            message=(f"{len(items)} element(s) in this section extend past the viewport; "
                     f"the worst overflows by {worst['overflowPx']} px."),
            evidence={
                "overflowPx": worst["overflowPx"],
                "threshold": "element right edge <= clientWidth + 1",
                "documentScrollWidth": result["scrollWidth"],
                "documentClientWidth": result["clientWidth"],
                "documentOverflows": result["documentOverflows"],
                "cssWidth": worst["cssWidth"], "cssMinWidth": worst["cssMinWidth"],
                "cssMaxWidth": worst["cssMaxWidth"],
                "otherCulprits": [c["selector"] for c in items[1:6]],
            },
            how_to_fix="; ".join(hints) or
                       "Give the element `max-width: 100%`, or `min-width: 0` plus "
                       "`overflow-wrap: anywhere` if it is a flex/grid child holding "
                       "long unbroken content.",
        ))
    if not result["documentOverflows"] and culprits:
        report.note(
            f"At {viewport.label} the document does not scroll horizontally, but "
            f"{len(culprits)} element(s) still extend past the viewport — an ancestor "
            "with overflow-x:hidden is masking them. This is exactly the case a "
            "page-level check misses."
        )
    return result


def check_target_size(report: Report, page: Any, url: str, viewport: Viewport) -> None:
    """mobile.md §1.4 — WCAG 2.5.8 failures plus the advisory platform floors."""
    result = page.evaluate(TARGET_SIZE_JS, TARGET_SIZE_AA)

    by_section: dict[str, list[dict[str, Any]]] = {}
    for failure in result["failures"]:
        by_section.setdefault(failure["section"] or "(unsectioned)", []).append(failure)

    for section, items in by_section.items():
        smallest = min(items, key=lambda f: f["size"][0] * f["size"][1])
        report.add(Finding(
            id=f"responsive.target-size.{slugify(section)}.{viewport.key.lower()}",
            severity="major", rule="target-size",
            wcag_sc="2.5.8 Target Size (Minimum) (AA)", url=url,
            viewport=viewport.label, selector=smallest["selector"], section=section,
            message=(f"{len(items)} target(s) are below {TARGET_SIZE_AA}x{TARGET_SIZE_AA} "
                     f"CSS px and are not saved by the spacing exception; the smallest "
                     f"is {smallest['size'][0]}x{smallest['size'][1]} px."),
            evidence={
                "measured": smallest["size"],
                # mobile.md §1.4 — 24x24 is the AA number. 44x44 is 2.5.5 AAA.
                "threshold": f"{TARGET_SIZE_AA}x{TARGET_SIZE_AA} CSS px (SC 2.5.8, AA)",
                "conflictingTargetsWithin24px": smallest["conflicts"],
                "label": smallest["text"],
                "others": [f["selector"] for f in items[1:6]],
            },
            how_to_fix="Pad the control to at least 24x24 CSS px, or space it so a "
                       "24 px-diameter circle centred on it touches no other target. "
                       "`min-height`/`min-width` plus padding keeps the visual size "
                       "unchanged (WCAG technique C42).",
        ))

    if result["review"]:
        report.add(Finding(
            id=f"responsive.target-size.review.{viewport.key.lower()}",
            severity="advisory", rule="target-size-review", wcag_sc="2.5.8 (AA)",
            url=url, viewport=viewport.label, selector=None, section="page",
            message=f"{len(result['review'])} undersized native control(s) need human "
                    f"judgement — the User Agent Control exception applies unless the "
                    f"author restyled them.",
            evidence={"controls": [r["selector"] for r in result["review"][:10]]},
            how_to_fix="Confirm whether the control is rendered by the browser "
                       "unmodified. If the author restyled it, the exception is lost.",
        ))

    if result["below44Advisory"]:
        report.add(Finding(
            id=f"responsive.target-size.below-44.{viewport.key.lower()}",
            severity="advisory", rule="target-size-platform", wcag_sc=None, url=url,
            viewport=viewport.label, selector=None, section="page",
            message=(f"{result['below44Advisory']} target(s) pass WCAG AA but sit below "
                     f"the {TARGET_SIZE_HIG} px Apple HIG floor "
                     f"({result['below48Advisory']} below Material's {TARGET_SIZE_MATERIAL} dp)."),
            evidence={"below44": result["below44Advisory"],
                      "below48": result["below48Advisory"],
                      "totalTargets": result["total"],
                      "status": "platform guidance, advisory — not a violation"},
            how_to_fix="Report as a recommendation with the source named. Do not call "
                       "it a WCAG violation: 44x44 is SC 2.5.5, Level AAA.",
        ))


def check_text_size(report: Report, page: Any, url: str, viewport: Viewport) -> None:
    """mobile.md §2.1 — text below the readability floor, per section."""
    result = page.evaluate(TEXT_SIZE_JS)
    failing = [r for r in result["runs"] if r["px"] < TEXT_FAIL_PX]
    warning = [r for r in result["runs"] if TEXT_FAIL_PX <= r["px"] < TEXT_WARN_PX]

    by_section: dict[str, list[dict[str, Any]]] = {}
    for run in failing:
        by_section.setdefault(run["section"] or "(unsectioned)", []).append(run)

    for section, runs in by_section.items():
        share = round(sum(r["pctOfText"] for r in runs), 1)
        smallest = min(runs, key=lambda r: r["px"])
        report.add(Finding(
            id=f"responsive.text-size.{slugify(section)}.{viewport.key.lower()}",
            severity="minor", rule="text-size", wcag_sc=None, url=url,
            viewport=viewport.label, selector=smallest["selector"], section=section,
            message=(f"Text renders at {smallest['px']:g} px here — {share}% of the "
                     f"page's visible text."),
            # mobile.md §2.1 audit thresholds: fail < 14, warn 14-15, pass >= 16.
            evidence={"smallestPx": smallest["px"], "threshold": f"< {TEXT_FAIL_PX} px fails",
                      "pctOfPageText": share, "sample": smallest["sample"],
                      "note": "readability guidance, not a Google or WCAG requirement"},
            how_to_fix="Raise body copy to 16 px. Lighthouse removed its font-size "
                       "audit in v13, so this stands on readability and iOS input "
                       "auto-zoom grounds only.",
        ))

    if warning:
        total = round(sum(r["pctOfText"] for r in warning), 1)
        report.add(Finding(
            id=f"responsive.text-size.warn.{viewport.key.lower()}",
            severity="advisory", rule="text-size", wcag_sc=None, url=url,
            viewport=viewport.label, selector=None, section="page",
            message=f"{total}% of visible text renders between {TEXT_FAIL_PX} and "
                    f"{TEXT_WARN_PX} px.",
            evidence={"sections": sorted({r["section"] for r in warning if r["section"]})},
            how_to_fix="Move body copy to the 16 px browser default.",
        ))


def check_input_font_size(report: Report, page: Any, url: str, viewport: Viewport) -> None:
    """mobile.md §1.9 — inputs below 16 px trigger iOS focus auto-zoom."""
    for control in page.evaluate(INPUT_FONT_JS, INPUT_MIN_PX):
        section = control["section"] or "(unsectioned)"
        report.add(Finding(
            id=f"responsive.ios-autozoom.{slugify(section)}."
               f"{slugify(control['selector'])}.{viewport.key.lower()}",
            severity="major", rule="ios-input-autozoom", wcag_sc=None, url=url,
            viewport=viewport.label, selector=control["selector"], section=section,
            message=(f"Form control renders at {control['fontSizePx']:g} px. iOS Safari "
                     f"zooms the page on focus and does not reliably zoom back out."),
            # mobile.md §1.9 — threshold is 16 CSS px, rendered size.
            evidence={"fontSizePx": control["fontSizePx"], "threshold": INPUT_MIN_PX,
                      "type": control["type"], "name": control["name"],
                      "transform": control["transform"],
                      "tier": "deterministic platform defect; no WCAG SC maps directly"},
            how_to_fix="Set the control's font-size to 16 px. If the design demands a "
                       "smaller control, declare 16 px and shrink visually with "
                       "transform: scale() plus compensating margins — and verify on a "
                       "real device. Never fix this with user-scalable=no; that is a "
                       "WCAG 1.4.4 violation.",
        ))


def check_hover_only(report: Report, page: Any, url: str, viewport: Viewport) -> None:
    """mobile.md §1.8 — hover-only reveals, confirmed by a dispatched touch.

    A CSSOM finding alone is a hypothesis. This dispatches a real pointer/touch
    sequence at the element centre and asserts the content becomes visible, so
    the finding ships with behavioural proof.
    """
    result = page.evaluate(HOVER_ONLY_JS)
    for rule in result["hoverOnlyRules"]:
        if not rule["matchedElements"]:
            continue
        confirmed = _confirm_touch_reveal(page, rule["selector"])
        sections = rule["sections"] or ["(unsectioned)"]
        report.add(Finding(
            id=f"responsive.hover-only.{slugify(rule['selector'])}.{viewport.key.lower()}",
            severity="major", rule="hover-only-affordance",
            wcag_sc="2.1.1 Keyboard (A); 1.4.13 Content on Hover or Focus (AA)",
            url=url, viewport=viewport.label, selector=rule["selector"],
            section="; ".join(sections),
            message=(f"`{rule['selector']}` reveals content on hover with no focus or "
                     f"tap equivalent; {rule['matchedElements']} element(s) match."),
            evidence={"properties": rule["properties"], "stylesheet": rule["href"],
                      "touchDispatchRevealedContent": confirmed,
                      "matchedElements": rule["matchedElements"]},
            how_to_fix="Add a `:focus-visible` / `:focus-within` equivalent, or drive "
                       "the reveal from an `aria-expanded` toggle. Gate the hover "
                       "enhancement additively behind "
                       "`@media (hover: hover) and (pointer: fine)` — never use "
                       "`@media (hover: none)` to remove function, because hybrid "
                       "devices report both.",
        ))

    for handler in result["mouseOnlyInlineHandlers"]:
        report.add(Finding(
            id=f"responsive.hover-only.inline.{slugify(handler['selector'])}",
            severity="minor", rule="mouse-only-handler", wcag_sc="2.1.1 Keyboard (A)",
            url=url, viewport=viewport.label, selector=handler["selector"],
            section=handler["section"] or "(unsectioned)",
            message="Element has an inline mouseover/mouseenter handler and no click, "
                    "focus or touch handler.",
            evidence={"attributes": "onmouseover / onmouseenter"},
            how_to_fix="Add an equivalent focus and click/pointerdown path.",
        ))


def _confirm_touch_reveal(page: Any, hover_selector: str) -> bool | None:
    """Dispatch a touch sequence at the first matching element and report the result.

    Returns:
        ``True`` if the tap revealed something, ``False`` if it did not, and
        ``None`` if the element could not be located or driven.
    """
    base = hover_selector.replace(":hover", "").strip()
    try:
        return page.evaluate(
            r"""(sel) => {
              const el = document.querySelector(sel);
              if (!el) return null;
              const before = el.getBoundingClientRect();
              const x = before.left + before.width / 2;
              const y = before.top + before.height / 2;
              const opts = { bubbles: true, cancelable: true, clientX: x, clientY: y };
              for (const type of ['pointerdown', 'touchstart', 'touchend',
                                  'pointerup', 'click'])
                el.dispatchEvent(new (type.startsWith('touch') ? Event : MouseEvent)(
                  type, opts));
              const after = el.getBoundingClientRect();
              return after.height > before.height + 1 || after.width > before.width + 1;
            }""",
            base,
        )
    except Exception:  # noqa: BLE001 - an invalid selector is not an audit failure
        return None


def check_text_200(report: Report, page: Any, url: str, viewport: Viewport) -> None:
    """mobile.md §1.5 — WCAG 1.4.4, text resized to 200%."""
    result = page.evaluate(TEXT_200_JS)
    if result["newlyClipped"] <= 0 and not result["newHorizontalOverflow"]:
        return
    report.add(Finding(
        id=f"responsive.text-200.{viewport.key.lower()}",
        severity="blocker", rule="resize-text-200",
        wcag_sc="1.4.4 Resize Text (AA)", url=url, viewport=viewport.label,
        selector=None, section="; ".join(result["sections"]) or "page",
        message=(f"Doubling the root font-size newly clips {result['newlyClipped']} "
                 f"element(s)"
                 + (" and introduces horizontal overflow."
                    if result["newHorizontalOverflow"] else ".")),
        evidence={"clippedBefore": result["before"]["count"],
                  "clippedAfter": result["after"]["count"],
                  "scrollWidthBefore": result["before"]["sw"],
                  "scrollWidthAfter": result["after"]["sw"],
                  "threshold": "no new clipping or overflow at 2x root font-size",
                  "affectedSections": result["sections"]},
        how_to_fix="Remove fixed heights with `overflow: hidden` on text containers; "
                   "size in rem rather than px; make sure no `font-size` is expressed "
                   "purely in viewport units, which never grow under zoom.",
    ))


def check_reflow_content_loss(narrow: list[str], wide: list[str]) -> list[str]:
    """mobile.md §1.3 — text present at 1280 px but absent at 320 px.

    WCAG 1.4.10 bans content that is *hidden or clipped* at 320 px, not just
    content that scrolls; a media query that removes information is a failure,
    not a responsive technique.

    Returns:
        The missing strings, so the caller can attach them as evidence.
    """
    narrow_set = {s for s in narrow if len(s) > 12}
    missing = [s for s in wide if len(s) > 12 and s not in narrow_set]
    # Ignore near-matches (truncation, whitespace normalisation) to cut noise.
    missing = [
        s for s in missing
        if not difflib.get_close_matches(s, list(narrow_set), n=1, cutoff=0.9)
    ]
    return missing


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def build_parser():
    """Construct the argument parser."""
    parser = base_parser(
        "audit_responsive.py",
        "Audit responsive/mobile behaviour across the mobile.md V1-V8 viewport "
        "matrix, attributing every finding to a section.",
        epilog=(
            "Viewport matrix (references/mobile.md §5.2):\n"
            + "\n".join(f"  {v.key}  {v.width:>4}x{v.height:<4} @{v.dpr:g}x  {v.why}"
                        for v in VIEWPORT_MATRIX.values())
        ),
    )
    parser.add_argument(
        "--viewports", default=",".join(DEFAULT_VIEWPORTS), metavar="LIST",
        help="Comma-separated viewport keys to run (default: all of V1-V8)",
    )
    parser.add_argument(
        "--out", default="audit-output/responsive", metavar="DIR",
        help="Directory for per-breakpoint screenshots (default: audit-output/responsive)",
    )
    parser.add_argument(
        "--no-screenshots", action="store_true",
        help="Skip screenshot capture",
    )
    parser.add_argument(
        "--headful", action="store_true",
        help="Run the browser headed (useful when debugging the harness itself)",
    )
    return parser


def main() -> int:
    """Run the responsive audit and return the process exit code."""
    args = build_parser().parse_args()
    url = normalise_url(args.url)

    keys = [k.strip().upper() for k in args.viewports.split(",") if k.strip()]
    unknown = [k for k in keys if k not in VIEWPORT_MATRIX]
    if unknown:
        build_parser().error(f"unknown viewport key(s): {', '.join(unknown)}")
    viewports = [VIEWPORT_MATRIX[k] for k in keys]

    out_dir = Path(args.out)
    report = Report(tool="audit_responsive", target=url)
    report.meta["viewports"] = ", ".join(v.key for v in viewports)

    sync_playwright = import_playwright()
    narrow_text: list[str] = []
    wide_text: list[str] = []
    overflow_by_viewport: dict[str, dict[str, Any]] = {}

    with sync_playwright() as playwright:
        browser = launch_chromium(playwright, headless=not args.headful)
        try:
            for index, viewport in enumerate(viewports):
                page = open_page(browser, viewport, url, args.timeout)
                context = page.context
                try:
                    gate = page.evaluate(SANITY_GATE_JS)
                    if viewport.mobile and not (gate["emulatingTouch"]
                                                and gate["emulatingNoHover"]):
                        # mobile.md §5.1 — without touch/hover emulation every
                        # hover and target finding below is invalid.
                        report.note(
                            f"{viewport.key}: touch/hover emulation did not take "
                            f"(pointer:coarse={gate['emulatingTouch']}, "
                            f"hover:none={gate['emulatingNoHover']}). Hover and "
                            "target-size findings at this viewport are not reliable."
                        )

                    sections = page.evaluate("window.__auditSections")
                    if index == 0:
                        report.meta["sections_enumerated"] = len(sections)
                        if not sections:
                            report.add(Finding(
                                id="responsive.no-sections",
                                severity="minor", rule="landmarks", wcag_sc="1.3.1 (A)",
                                url=url, viewport=viewport.label, selector="body",
                                section="page",
                                message="No usable sectioning elements or landmarks "
                                        "were found, so findings cannot be attributed "
                                        "to a section.",
                                evidence={"selectorsTried":
                                          "main > *, section, article, aside, header, "
                                          "footer, nav, [role=region], form"},
                                how_to_fix="Add real sectioning elements and landmark "
                                           "roles; see references/ada/html-core.md.",
                            ))
                        check_viewport_meta(report, page, url, viewport)
                        check_hover_only(report, page, url, viewport)

                    overflow_by_viewport[viewport.key] = check_overflow(
                        report, page, url, viewport)
                    check_target_size(report, page, url, viewport)
                    check_text_size(report, page, url, viewport)
                    check_input_font_size(report, page, url, viewport)

                    if viewport.root_font_size:
                        check_text_200(report, page, url, viewport)
                    if viewport.key == "V1":
                        narrow_text = page.evaluate(VISIBLE_TEXT_JS)
                    if viewport.key == "V6":
                        wide_text = page.evaluate(VISIBLE_TEXT_JS)

                    if not args.no_screenshots:
                        out_dir.mkdir(parents=True, exist_ok=True)
                        shot = out_dir / f"{viewport.key}-{viewport.width}x{viewport.height}.png"
                        page.screenshot(path=str(shot), full_page=True)
                finally:
                    context.close()

            # mobile.md §1.3 — the 320 vs 1280 content diff needs its own render.
            if "V1" in keys:
                reference = Viewport("REF", REFLOW_REFERENCE_PX, 1024, 1, False,
                                     "1280 px reference render for the 1.4.10 diff")
                page = open_page(browser, reference, url, args.timeout)
                try:
                    wide_text = page.evaluate(VISIBLE_TEXT_JS)
                finally:
                    page.context.close()

                missing = check_reflow_content_loss(narrow_text, wide_text)
                if missing:
                    report.add(Finding(
                        id="responsive.reflow.content-loss",
                        severity="blocker", rule="reflow-content-loss",
                        wcag_sc="1.4.10 Reflow (AA)", url=url,
                        viewport=f"{REFLOW_FLOOR_PX} px vs {REFLOW_REFERENCE_PX} px",
                        selector=None, section="page",
                        message=(f"{len(missing)} text string(s) visible at "
                                 f"{REFLOW_REFERENCE_PX} px are not visible at "
                                 f"{REFLOW_FLOOR_PX} px — loss of information, not "
                                 f"reflow."),
                        evidence={"examples": missing[:10],
                                  "threshold": "no content may be hidden or clipped "
                                               "at 320 CSS px"},
                        how_to_fix="Reflow the content instead of hiding it. A "
                                   "`@media (max-width: ...) { display: none }` that "
                                   "removes information is a 1.4.10 failure.",
                    ))
        finally:
            browser.close()

    # mobile.md §5.3 step 6 — a failure that reproduces at V1 and V7 is unambiguous.
    if "V1" in overflow_by_viewport and "V7" in overflow_by_viewport:
        v1_sections = {c["section"] for c in overflow_by_viewport["V1"]["culprits"]}
        v7_sections = {c["section"] for c in overflow_by_viewport["V7"]["culprits"]}
        both = sorted(s for s in v1_sections & v7_sections if s)
        if both:
            report.meta["reproduced_at_320_and_400pct_zoom"] = "; ".join(both)

    report.note(
        "Automated emulation cannot observe three things: safe-area insets (DevTools "
        "device mode always returns 0), iOS input auto-zoom, and iOS modal "
        "scroll-locking. Confirm those on a real device and say in the report which "
        "findings were device-confirmed and which are emulation or static-signal only."
    )
    report.note(
        "Target-size findings implement the Spacing and Inline exceptions only. The "
        "Equivalent, User Agent Control and Essential exceptions require human "
        "judgement; native date/color/file inputs are surfaced as 'review'."
    )
    if not args.no_screenshots:
        report.meta["screenshots"] = str(out_dir)
    return finish(report, args)


if __name__ == "__main__":
    run_cli(main)
