#!/usr/bin/env python3
"""Crawl a site and audit the mechanical SEO checks against the rendered page.

Every page is fetched twice — raw HTML over HTTP, and the rendered DOM through
Playwright — because ``references/seo/L1-foundations.md`` §1.4 states the rule
this whole skill is built on:

    **Rule: test the rendered output. Never infer rendering strategy from
    ``package.json``.**

and its highest-value machine check:

    ``<title>``, meta description, canonical, and JSON-LD exist in the raw HTML,
    not injected by JS ... Head tags injected client-side are the most common
    partial-CSR failure.

Per page: title, meta description, canonical, robots meta and ``X-Robots-Tag``,
Open Graph and Twitter cards, heading outline, hreflang, structured data with a
type inventory, image alt coverage, and the raw-vs-rendered diff.

Site-wide: the internal link graph with click depth and orphan pages, broken
links and redirect chains, duplicate titles and descriptions, sitemap and
robots.txt validity, and the soft-404 probe.

One thing below is **not** specified by the reference documents and is labelled
as a script decision in the output rather than as a doc-backed threshold: the
numeric raw-vs-rendered text ratio. The references are silent on it.

Usage:
    ./audit_seo.py https://example.com
    ./audit_seo.py https://example.com --max-pages 100 --delay 0.5
    ./audit_seo.py https://example.com --check-external --json out/seo.json
"""

from __future__ import annotations

import re
import time
import urllib.error
import urllib.request
import urllib.robotparser
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable
from urllib.parse import urldefrag, urljoin, urlparse

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
# Thresholds — cited to references/seo/*
# --------------------------------------------------------------------------- #

TITLE_MIN_CHARS = 50        # L4-onpage.md: "~50-60 characters"
TITLE_MAX_CHARS = 60        # L4-onpage.md: Screaming Frog flags "Over 60 Characters"
DESCRIPTION_MIN_CHARS = 140  # L4-onpage.md: "~140-160 characters"
DESCRIPTION_MAX_CHARS = 160
MAX_REDIRECT_HOPS = 1       # L2-technical-performance.md: "at most one 301 before 200"
MAX_CLICK_DEPTH = 3         # L3-architecture.md: "all important URLs at depth <= 3"
DEEP_PAGE_DEPTH = 5         # L3-architecture.md fail signature: "money pages at depth 5+"
BOILERPLATE_LINK_LIMIT = 200  # L3-architecture.md: "200+ identical footer links"
CANONICAL_SAMPLE = 10       # L1-foundations.md: sample 10 URLs, expect 10 distinct
SITEMAP_MAX_URLS = 50_000   # L1-foundations.md: "under 50,000 URLs and 50MB"
RAW_HTML_WINDOW = 2000      # L1-foundations.md test 1: `head -c 2000`

# L1-foundations.md §1.4: the doc compares raw and rendered qualitatively
# ("raw ~= rendered -> SSR; raw a small fraction of rendered -> CSR") and gives
# no number. This is a SCRIPT DECISION, reported as such.
CSR_TEXT_RATIO_THRESHOLD = 0.25

# L3-architecture.md: the three literal stop-phrases to count.
GENERIC_ANCHOR_TEXT = ("click here", "read more", "learn more")

# 00-map.md: title reject patterns.
PLACEHOLDER_TITLES = re.compile(r"^(untitled|home|page|document|new page|index)$", re.I)

# L3-architecture.md: ISO 639-1 language, optional ISO 3166-1 Alpha-2 region.
# `en-UK` is invalid (the region code is GB); `uk` is Ukrainian.
HREFLANG_RE = re.compile(r"^(x-default|[a-z]{2,3}(-[A-Za-z]{4})?(-[A-Za-z]{2})?)$")
KNOWN_BAD_HREFLANG = {"en-uk"}

# L6-structured-data.md §6.1 — retired from search appearance. Still valid
# vocabulary: report as informational, and do not tell anyone to rip it out.
RETIRED_SCHEMA_TYPES = {
    "FAQPage": "FAQ rich results stopped appearing 7 May 2026; Search Console "
               "reporting and the Rich Results Test dropped support June 2026",
    "HowTo": "HowTo rich results were retired in 2023",
}

#: L6-structured-data.md §6.1 — required properties per prioritised type.
REQUIRED_SCHEMA_PROPERTIES: dict[str, tuple[str, ...]] = {
    "Organization": ("name", "url", "logo", "sameAs"),
    "Article": ("headline", "author", "datePublished", "dateModified", "publisher"),
    "BlogPosting": ("headline", "author", "datePublished", "dateModified", "publisher"),
    "Product": ("offers",),
    "LocalBusiness": ("address", "geo", "openingHoursSpecification"),
}

#: L4-onpage.md §4.3 — the four properties the Open Graph protocol marks
#: required. Absent, the parser falls back to its own guess.
OG_REQUIRED_TAGS = ("og:title", "og:type", "og:image", "og:url")

#: L4-onpage.md §4.3 — recommended, and each has a named consequence when
#: absent: no feed copy, no card at all on X, no alt text on the shared image.
OG_RECOMMENDED_TAGS = ("og:description", "twitter:card", "og:image:alt")

#: L4-onpage.md §4.3 — "advisory" tier: a rendering-latency and polish
#: optimisation, not a broken share.
OG_ADVISORY_TAGS = ("og:image:width", "og:image:height", "og:site_name")

#: L4-onpage.md §4.3 image constraints, verified 2026-08-19. Facebook: file
#: size must not exceed 8 MB, minimum dimension 200x200, 600x315 for the
#: large-image layout, 1200x630 recommended, ~1.91:1. X summary_large_image:
#: under 5 MB, 300x157 to 4096x4096, 2:1, and SVG is not supported.
OG_IMAGE_MIN_WIDTH = 600
OG_IMAGE_MIN_HEIGHT = 315
OG_IMAGE_RECOMMENDED = (1200, 630)

#: SCRIPT DECISION — the references give no crawl-politeness number.
DEFAULT_DELAY_SECONDS = 1.0
DEFAULT_MAX_PAGES = 50

USER_AGENT = (
    "Mozilla/5.0 (compatible; website-audit/audit_seo.py; +local audit tooling)"
)


# --------------------------------------------------------------------------- #
# In-page extraction
# --------------------------------------------------------------------------- #

EXTRACT_JS = r"""() => {
  const attr = (sel, name) => document.querySelector(sel)?.getAttribute(name) || null;
  const metas = {};
  for (const m of document.querySelectorAll('meta[property], meta[name]')) {
    const key = (m.getAttribute('property') || m.getAttribute('name') || '').toLowerCase();
    if (key && !(key in metas)) metas[key] = m.getAttribute('content');
  }
  const headings = [...document.querySelectorAll('h1,h2,h3,h4,h5,h6')]
    .map(h => ({ level: Number(h.tagName[1]),
                 text: (h.innerText || '').trim().slice(0, 90) }));
  const links = [...document.querySelectorAll('a[href]')].map(a => ({
    href: a.href,
    text: (a.innerText || a.getAttribute('aria-label') || '').trim().slice(0, 80),
    rel: a.getAttribute('rel') || '',
    inFooter: !!a.closest('footer'),
    inNav: !!a.closest('nav, header')
  }));
  const images = [...document.querySelectorAll('img')].map(img => ({
    src: (img.currentSrc || img.getAttribute('src') || '').slice(-100),
    hasAlt: img.hasAttribute('alt'),
    alt: img.getAttribute('alt'),
    filename: (img.getAttribute('src') || '').split('/').pop() || ''
  }));
  const jsonld = [];
  for (const node of document.querySelectorAll('script[type="application/ld+json"]')) {
    try { jsonld.push(JSON.parse(node.textContent)); }
    catch (e) { jsonld.push({ __parseError: String(e).slice(0, 120) }); }
  }
  const hreflang = [...document.querySelectorAll('link[rel=alternate][hreflang]')]
    .map(l => ({ hreflang: l.getAttribute('hreflang'), href: l.href }));
  const breadcrumb = [...document.querySelectorAll(
    'nav[aria-label*="readcrumb" i] a, .breadcrumb a, [class*="breadcrumb"] a')]
    .map(a => (a.innerText || '').trim()).filter(Boolean);
  return {
    title: document.title || null,
    canonical: attr('link[rel=canonical]', 'href'),
    canonicalCount: document.querySelectorAll('link[rel=canonical]').length,
    metas, headings, links, images, jsonld, hreflang, breadcrumb,
    lang: document.documentElement.getAttribute('lang'),
    microdataItemprops: document.querySelectorAll('body [itemprop]').length,
    textLength: (document.body.innerText || '').replace(/\s+/g, ' ').trim().length
  };
}"""


# --------------------------------------------------------------------------- #
# Page model
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Page:
    """Everything collected for one crawled URL."""

    url: str
    status: int
    depth: int
    headers: dict[str, str] = field(default_factory=dict)
    raw_html: str = ""
    rendered: dict[str, Any] = field(default_factory=dict)
    redirect_hops: list[str] = field(default_factory=list)

    @property
    def description(self) -> str | None:
        """The meta description from the rendered DOM."""
        return (self.rendered.get("metas") or {}).get("description")

    @property
    def robots_meta(self) -> str:
        """The robots meta content, lower-cased."""
        return ((self.rendered.get("metas") or {}).get("robots") or "").lower()


# --------------------------------------------------------------------------- #
# HTTP helpers (stdlib — the raw HTML half of the comparison)
# --------------------------------------------------------------------------- #


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Stop at the first redirect so hops can be counted."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


def fetch_raw(url: str, timeout: float, follow: bool = True) -> tuple[int, dict[str, str], str]:
    """Fetch *url* over plain HTTP and return ``(status, headers, body)``.

    Non-2xx statuses are returned, not raised — a 404 is data, not an error.
    """
    opener = urllib.request.build_opener(*([] if follow else [_NoRedirect()]))
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with opener.open(request, timeout=timeout) as response:
            body = response.read(1_000_000).decode("utf-8", errors="replace")
            headers = {k.lower(): v for k, v in response.headers.items()}
            return response.status, headers, body
    except urllib.error.HTTPError as exc:
        headers = {k.lower(): v for k, v in exc.headers.items()} if exc.headers else {}
        return exc.code, headers, ""
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise TargetUnreachable(url, str(getattr(exc, "reason", exc))) from exc


def status_of(url: str, timeout: float) -> tuple[int, list[str]]:
    """Return the final status and the redirect chain for *url*.

    Returns:
        ``(status, hops)`` where ``hops`` lists each intermediate Location.
    """
    hops: list[str] = []
    current = url
    for _ in range(6):
        try:
            status, headers, _ = fetch_raw(current, timeout, follow=False)
        except TargetUnreachable:
            return 0, hops
        if status in (301, 302, 303, 307, 308) and headers.get("location"):
            current = urljoin(current, headers["location"])
            hops.append(current)
            continue
        return status, hops
    return 0, hops  # redirect loop


# --------------------------------------------------------------------------- #
# Crawling
# --------------------------------------------------------------------------- #


def load_robots(origin: str, timeout: float) -> tuple[urllib.robotparser.RobotFileParser,
                                                      list[str], int | None]:
    """Fetch and parse ``/robots.txt``.

    Returns:
        ``(parser, sitemap_urls, status)``. ``status`` is ``None`` when the file
        could not be fetched at all.
    """
    robots_url = urljoin(origin, "/robots.txt")
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(robots_url)
    try:
        status, _, body = fetch_raw(robots_url, timeout)
    except TargetUnreachable:
        parser.parse([])
        return parser, [], None
    if status != 200:
        # A 404 is survivable (everything is allowed); a 5xx is not — crawlers
        # may back off entirely (L1-foundations.md §1.2).
        parser.parse([])
        return parser, [], status
    parser.parse(body.splitlines())
    sitemaps = [line.split(":", 1)[1].strip()
                for line in body.splitlines()
                if line.lower().startswith("sitemap:")]
    return parser, sitemaps, status


def normalise(url: str) -> str:
    """Strip the fragment so ``/a`` and ``/a#b`` are one page."""
    clean, _ = urldefrag(url)
    return clean


def crawl(browser: Any, start: str, *, max_pages: int, delay: float, timeout: float,
          robots: urllib.robotparser.RobotFileParser,
          obey_robots: bool) -> tuple[dict[str, Page], dict[str, set[str]]]:
    """Breadth-first crawl of same-origin pages.

    Returns:
        ``(pages, inlinks)`` where ``inlinks`` maps a URL to the set of URLs
        that link to it — the data the orphan and depth reports need.
    """
    origin_host = urlparse(start).hostname
    pages: dict[str, Page] = {}
    inlinks: dict[str, set[str]] = {}
    queue: deque[tuple[str, int]] = deque([(normalise(start), 0)])
    seen: set[str] = set()

    context = browser.new_context(user_agent=USER_AGENT)
    page_handle = context.new_page()
    try:
        while queue and len(pages) < max_pages:
            url, depth = queue.popleft()
            if url in seen:
                continue
            seen.add(url)
            if obey_robots and not robots.can_fetch(USER_AGENT, url):
                continue

            status, headers, raw = fetch_raw(url, timeout)
            final_status, hops = (status, []) if status < 300 or status >= 400 \
                else status_of(url, timeout)

            rendered: dict[str, Any] = {}
            if status == 200:
                try:
                    page_handle.goto(url, wait_until="load", timeout=timeout * 1000)
                    page_handle.wait_for_timeout(300)
                    rendered = page_handle.evaluate(EXTRACT_JS)
                except Exception:  # noqa: BLE001 - a render failure is recorded, not fatal
                    rendered = {}

            pages[url] = Page(url=url, status=final_status or status, depth=depth,
                              headers=headers, raw_html=raw, rendered=rendered,
                              redirect_hops=hops)

            for link in rendered.get("links", []):
                target = normalise(link["href"])
                parsed = urlparse(target)
                if parsed.scheme not in ("http", "https"):
                    continue
                inlinks.setdefault(target, set()).add(url)
                if parsed.hostname == origin_host and target not in seen:
                    queue.append((target, depth + 1))

            time.sleep(delay)
    finally:
        context.close()
    return pages, inlinks


def parse_sitemap_urls(sitemap_urls: Iterable[str], timeout: float,
                       limit: int = 5000) -> list[str]:
    """Fetch sitemaps (following one level of sitemap index) and return ``<loc>`` URLs."""
    found: list[str] = []
    pending = list(sitemap_urls)
    visited: set[str] = set()
    while pending and len(found) < limit:
        current = pending.pop(0)
        if current in visited:
            continue
        visited.add(current)
        try:
            status, _, body = fetch_raw(current, timeout)
        except TargetUnreachable:
            continue
        if status != 200:
            continue
        locations = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", body, re.IGNORECASE)
        if "<sitemapindex" in body.lower():
            pending.extend(locations[:50])
        else:
            found.extend(locations)
    return found[:limit]


# --------------------------------------------------------------------------- #
# Per-page checks
# --------------------------------------------------------------------------- #


def check_title(report: Report, page: Page) -> None:
    """L4-onpage.md — unique, descriptive, ~50-60 characters."""
    title = (page.rendered.get("title") or "").strip()
    if not title:
        report.add(Finding(
            id=f"seo.title.missing.{page.url}",
            severity="major", rule="title", wcag_sc=None, url=page.url,
            viewport=None, selector="title", section="head",
            message="Page has no <title>.",
            evidence={"measured": None, "target": f"{TITLE_MIN_CHARS}-{TITLE_MAX_CHARS} chars"},
            how_to_fix="Write a unique title with the primary topic near the front and "
                       "the brand at the end.",
        ))
        return
    if PLACEHOLDER_TITLES.match(title):
        report.add(Finding(
            id=f"seo.title.placeholder.{page.url}",
            severity="major", rule="title", wcag_sc=None, url=page.url,
            viewport=None, selector="title", section="head",
            message=f"Title is a placeholder or template default: {title!r}.",
            evidence={"title": title},
            how_to_fix="Replace with the page's actual subject.",
        ))
    if len(title) > TITLE_MAX_CHARS:
        report.add(Finding(
            id=f"seo.title.too-long.{page.url}",
            severity="minor", rule="title-length", wcag_sc=None, url=page.url,
            viewport=None, selector="title", section="head",
            message=f"Title is {len(title)} characters, over {TITLE_MAX_CHARS}.",
            evidence={"measured": len(title), "target": f"{TITLE_MIN_CHARS}-{TITLE_MAX_CHARS}",
                      "title": title},
            how_to_fix="Trim to roughly 50-60 characters so it is not truncated in the "
                       "SERP.",
        ))
    elif len(title) < TITLE_MIN_CHARS:
        report.add(Finding(
            id=f"seo.title.too-short.{page.url}",
            severity="advisory", rule="title-length", wcag_sc=None, url=page.url,
            viewport=None, selector="title", section="head",
            message=f"Title is {len(title)} characters, under {TITLE_MIN_CHARS}.",
            evidence={"measured": len(title),
                      "target": f"{TITLE_MIN_CHARS}-{TITLE_MAX_CHARS}", "title": title},
            how_to_fix="Add descriptive detail; short titles waste SERP real estate.",
        ))


def check_description(report: Report, page: Page) -> None:
    """L4-onpage.md — unique, ~140-160 characters. Not a ranking factor."""
    description = (page.description or "").strip()
    if not description:
        report.add(Finding(
            id=f"seo.description.missing.{page.url}",
            severity="minor", rule="meta-description", wcag_sc=None, url=page.url,
            viewport=None, selector='meta[name=description]', section="head",
            message="No meta description.",
            evidence={"target": f"{DESCRIPTION_MIN_CHARS}-{DESCRIPTION_MAX_CHARS} chars",
                      "note": "not a ranking factor, but it is the ad copy for the "
                              "listing; Google rewrites it often and a written one "
                              "still wins more often than an absent one"},
            how_to_fix="Write a unique description per page.",
        ))
        return
    if not DESCRIPTION_MIN_CHARS <= len(description) <= DESCRIPTION_MAX_CHARS:
        report.add(Finding(
            id=f"seo.description.length.{page.url}",
            severity="advisory", rule="meta-description-length", wcag_sc=None,
            url=page.url, viewport=None, selector='meta[name=description]',
            section="head",
            message=f"Meta description is {len(description)} characters.",
            evidence={"measured": len(description),
                      "target": f"{DESCRIPTION_MIN_CHARS}-{DESCRIPTION_MAX_CHARS}"},
            how_to_fix="Aim for roughly 140-160 characters.",
        ))


def check_canonical(report: Report, page: Page) -> None:
    """L1-foundations.md §1.3 — self-referencing, absolute, exact match."""
    canonical = page.rendered.get("canonical")
    if page.rendered.get("canonicalCount", 0) > 1:
        report.add(Finding(
            id=f"seo.canonical.multiple.{page.url}",
            severity="major", rule="canonical", wcag_sc=None, url=page.url,
            viewport=None, selector="link[rel=canonical]", section="head",
            message=f"{page.rendered['canonicalCount']} canonical tags on one page; "
                    f"search engines will ignore all of them.",
            evidence={"count": page.rendered["canonicalCount"], "first": canonical},
            how_to_fix="Emit exactly one canonical link.",
        ))
        return
    if not canonical:
        report.add(Finding(
            id=f"seo.canonical.missing.{page.url}",
            severity="minor", rule="canonical", wcag_sc=None, url=page.url,
            viewport=None, selector="link[rel=canonical]", section="head",
            message="No rel=canonical.",
            evidence={"expected": "an absolute URL equal to the page's own URL"},
            how_to_fix="Add a self-referencing absolute canonical.",
        ))
        return
    # Compare protocol, host and trailing slash exactly; do not normalise them away.
    if canonical.rstrip() != page.url:
        report.add(Finding(
            id=f"seo.canonical.mismatch.{page.url}",
            severity="minor", rule="canonical", wcag_sc=None, url=page.url,
            viewport=None, selector="link[rel=canonical]", section="head",
            message="Canonical does not exactly match the page's own absolute URL.",
            evidence={"canonical": canonical, "pageUrl": page.url,
                      "compared": "protocol, host and trailing slash, exactly"},
            how_to_fix="Emit the page's own absolute URL, matching protocol, host and "
                       "trailing slash. If the difference is deliberate, confirm the "
                       "target is the intended canonical and is itself indexable.",
        ))


def check_robots_directives(report: Report, page: Page) -> None:
    """L1-foundations.md §1.2 — meta robots *and* the X-Robots-Tag header."""
    header = page.headers.get("x-robots-tag", "").lower()
    for source, value, selector in (("meta robots", page.robots_meta,
                                     "meta[name=robots]"),
                                    ("X-Robots-Tag header", header, None)):
        if "noindex" in value:
            report.add(Finding(
                id=f"seo.robots.noindex.{source.split()[0]}.{page.url}",
                severity="blocker", rule="noindex", wcag_sc=None, url=page.url,
                viewport=None, selector=selector, section="indexability",
                message=f"{source} contains `noindex`; this page cannot be indexed.",
                evidence={"value": value, "source": source,
                          "note": "header-level noindex is invisible in view-source "
                                  "and is a classic missed cause of a de-indexed site"},
                how_to_fix="Remove noindex if the page should rank. Staging protection "
                           "needs HTTP auth or a 403/404 — noindex alone leaks.",
            ))


def check_headings(report: Report, page: Page) -> None:
    """L4-onpage.md — exactly one H1; no skipped levels."""
    headings = page.rendered.get("headings", [])
    h1s = [h for h in headings if h["level"] == 1]

    # The reference's pass condition is literally 1; it names 0 and 3+ as
    # failures and is silent on 2, so 2 is reported at a lower severity.
    if len(h1s) == 0:
        report.add(Finding(
            id=f"seo.headings.no-h1.{page.url}",
            severity="major", rule="heading-outline", wcag_sc="1.3.1 (A)",
            url=page.url, viewport=None, selector="h1", section="content",
            message="No <h1>. This is common when a logo replaced it.",
            evidence={"measured": 0, "target": 1},
            how_to_fix="Add one <h1> matching the page's actual subject.",
        ))
    elif len(h1s) >= 3:
        report.add(Finding(
            id=f"seo.headings.many-h1.{page.url}",
            severity="minor", rule="heading-outline", wcag_sc="1.3.1 (A)",
            url=page.url, viewport=None, selector="h1", section="content",
            message=f"{len(h1s)} <h1> elements; heading levels are being chosen for "
                    f"their font size.",
            evidence={"measured": len(h1s), "target": 1,
                      "texts": [h["text"] for h in h1s[:5]]},
            how_to_fix="Keep one <h1> and demote the rest to <h2>/<h3> according to "
                       "the real structure.",
        ))
    elif len(h1s) == 2:
        report.add(Finding(
            id=f"seo.headings.two-h1.{page.url}",
            severity="advisory", rule="heading-outline", wcag_sc="1.3.1 (A)",
            url=page.url, viewport=None, selector="h1", section="content",
            message="Two <h1> elements.",
            evidence={"measured": 2, "target": 1,
                      "texts": [h["text"] for h in h1s]},
            how_to_fix="Demote one unless the second genuinely opens a separate "
                       "top-level document section.",
        ))

    skips = []
    previous = 0
    for heading in headings:
        if previous and heading["level"] > previous + 1:
            skips.append(f"h{previous} -> h{heading['level']}: {heading['text']!r}")
        previous = heading["level"]
    if skips:
        report.add(Finding(
            id=f"seo.headings.skipped-levels.{page.url}",
            severity="minor", rule="heading-outline", wcag_sc="1.3.1 (A)",
            url=page.url, viewport=None, selector=None, section="content",
            message=f"{len(skips)} skipped heading level(s).",
            evidence={"skips": skips[:8],
                      "outline": [f"h{h['level']} {h['text']}" for h in headings[:20]]},
            how_to_fix="Pick heading levels for structure, not font size. The outline "
                       "should read as a genuine table of contents — that half is a "
                       "human judgement this script cannot make.",
        ))


#: L4-onpage.md §4.3 — the reference every social-card finding cites.
SOCIAL_SOURCE = "../references/seo/L4-onpage.md §4.3"

#: L4-onpage.md §4.3 escalation rule the script cannot apply on its own: it has
#: no audit profile, so it emits the baseline severity and carries the rule.
SOCIAL_ESCALATION = (
    "L4-onpage.md §4.3: escalate to major where distribution is part of the "
    "site's stated job (publisher, marketing, campaign, seo.priority: full). "
    "This script has no audit profile and emits the baseline severity."
)


def check_social(report: Report, page: Page) -> None:
    """Open Graph and Twitter cards — L4-onpage.md §4.3.

    Severities are the ones that section defines: the four OGP-required
    properties and the named-consequence recommendations are minor, the
    polish tier is advisory, and anything that makes the share render
    *visibly broken* rather than merely plain is major.
    """
    metas = page.rendered.get("metas", {})
    selector = "meta[property^=og], meta[name^=twitter]"

    # --- Tier 1: presence, by the section's three severity bands ----------- #
    for tags, severity, band, fix in (
        (OG_REQUIRED_TAGS, "minor", "Open Graph required",
         "Add the four properties the Open Graph protocol marks required. Without "
         "them the parser guesses — usually the <title>, the meta description and "
         "the first large image in the document, which is often a logo or a "
         "tracking pixel."),
        (OG_RECOMMENDED_TAGS, "minor", "recommended",
         "og:description is the feed copy; twitter:card is the switch that makes X "
         "render a card at all; og:image:alt is the alt text for an image shown "
         "outside your DOM (ogp.me: if the page specifies an og:image it should "
         "specify og:image:alt)."),
        (OG_ADVISORY_TAGS, "advisory", "polish",
         "og:image:width/height let a crawler render the card immediately instead "
         "of downloading and measuring the file first."),
    ):
        missing = [tag for tag in tags if tag not in metas]
        if not missing:
            continue
        report.add(Finding(
            id=f"seo.social.missing.{band.split()[0].lower()}.{page.url}",
            severity=severity, rule="social-cards", wcag_sc=None, url=page.url,
            viewport=None, selector=selector, section="head",
            message=f"{len(missing)} {band} social-card tag(s) missing: "
                    f"{', '.join(missing)}.",
            evidence={"missing": missing, "present": sorted(
                          k for k in metas if k.startswith(("og:", "twitter:"))),
                      "source": SOCIAL_SOURCE,
                      "escalation": SOCIAL_ESCALATION},
            how_to_fix=fix,
        ))

    # --- Tier 2: values that make the share render visibly broken ---------- #
    broken: list[str] = []
    image = (metas.get("og:image") or "").strip()
    og_url = (metas.get("og:url") or "").strip()
    canonical = (page.rendered.get("canonical") or "").strip()

    if image and not image.lower().startswith(("http://", "https://")):
        broken.append(f"og:image is not an absolute URL ({image[:80]!r}); "
                      f"relative values are silently dropped")
    if image.lower().startswith("http://") and page.url.lower().startswith("https://"):
        broken.append("og:image is served over http:// on an https:// page; use an "
                      "https URL, or og:image:secure_url")
    if image and urlparse(image).path.lower().endswith(".svg"):
        broken.append("og:image is an SVG; X does not support SVG for cards")
    if og_url and not og_url.lower().startswith(("http://", "https://")):
        broken.append(f"og:url is not an absolute URL ({og_url[:80]!r})")
    if og_url and canonical and normalise(og_url).rstrip("/") != normalise(canonical).rstrip("/"):
        broken.append(f"og:url ({og_url[:80]}) disagrees with rel=canonical "
                      f"({canonical[:80]}); the share points somewhere the sharer "
                      f"did not intend")

    declared = {}
    for key in ("og:image:width", "og:image:height"):
        try:
            declared[key] = int(str(metas.get(key, "")).strip())
        except (TypeError, ValueError):
            pass
    if len(declared) == 2 and (declared["og:image:width"] < OG_IMAGE_MIN_WIDTH
                               or declared["og:image:height"] < OG_IMAGE_MIN_HEIGHT):
        broken.append(
            f"declared og:image dimensions {declared['og:image:width']}x"
            f"{declared['og:image:height']} are below the {OG_IMAGE_MIN_WIDTH}x"
            f"{OG_IMAGE_MIN_HEIGHT} floor for the large-image layout "
            f"({OG_IMAGE_RECOMMENDED[0]}x{OG_IMAGE_RECOMMENDED[1]} recommended)")

    if broken:
        report.add(Finding(
            id=f"seo.social.broken-value.{page.url}",
            severity="major", rule="social-cards", wcag_sc=None, url=page.url,
            viewport=None, selector=selector, section="head",
            message=f"{len(broken)} social-card value(s) will render a broken share.",
            evidence={"problems": broken, "ogImage": image or None,
                      "ogUrl": og_url or None, "canonical": canonical or None,
                      "source": SOCIAL_SOURCE,
                      "note": "Reachability of og:image is NOT checked here — fetch "
                              "it and read status, content-type and content-length "
                              "against the constraint table in " + SOCIAL_SOURCE},
            how_to_fix="Use absolute https URLs for og:image and og:url; keep og:url "
                       "equal to rel=canonical; ship a raster image (JPG/PNG/WEBP) "
                       "at 1200x630.",
        ))

    # --- Tier 3: injected client-side, so no social crawler ever sees it --- #
    raw = page.raw_html or ""
    injected = [tag for tag in OG_REQUIRED_TAGS
                if tag in metas and f'"{tag}"' not in raw and f"'{tag}'" not in raw]
    if injected:
        report.add(Finding(
            id=f"seo.social.client-side-only.{page.url}",
            severity="major", rule="social-cards", wcag_sc=None, url=page.url,
            viewport=None, selector=selector, section="head",
            message=f"{len(injected)} Open Graph tag(s) exist only in the rendered "
                    f"DOM, not in the raw HTML.",
            evidence={"clientSideOnly": injected, "source": SOCIAL_SOURCE,
                      "note": "Social crawlers do not execute JavaScript, so these "
                              "tags do not exist as far as any sharing surface is "
                              "concerned. Same failure class as a client-side "
                              "<title> — see L1-foundations.md §1.4."},
            how_to_fix="Emit og: tags server-side or at build time. Verify with "
                       "curl, not DevTools.",
        ))


def check_social_duplicates(report: Report, pages: dict[str, Page], start: str) -> None:
    """L4-onpage.md §4.3 — a site-wide default is a baseline, not a target.

    One brand image in the base template beats nothing. It becomes a finding
    when the templates that actually get shared all render the same card.
    """
    for key, label, severity in (("og:title", "og:title", "minor"),
                                 ("og:image", "og:image", "advisory")):
        buckets: dict[str, list[str]] = {}
        for page in pages.values():
            value = ((page.rendered.get("metas") or {}).get(key) or "").strip()
            if value:
                buckets.setdefault(value, []).append(page.url)
        duplicates = {value: urls for value, urls in buckets.items() if len(urls) > 1}
        if not duplicates:
            continue
        report.add(Finding(
            id=f"seo.social.duplicate.{key.replace(':', '-')}",
            severity=severity, rule="social-cards", wcag_sc=None, url=start,
            viewport=None, selector=f'meta[property="{key}"]', section="site",
            message=f"{len(duplicates)} {label} value(s) are reused across pages.",
            evidence={"duplicates": {value[:70]: urls[:5]
                                     for value, urls in list(duplicates.items())[:8]},
                      "pagesCrawled": len(pages), "source": SOCIAL_SOURCE,
                      "note": "A shared og:image across a whole site is a legitimate "
                              "baseline. A shared og:title means every link reads as "
                              "the homepage."},
            how_to_fix=f"Give every template that gets shared its own {label}.",
        ))


def check_hreflang(report: Report, page: Page, all_pages: dict[str, Page]) -> None:
    """L3-architecture.md §3.5 — reciprocal, one x-default, valid codes."""
    entries = page.rendered.get("hreflang", [])
    if not entries:
        return
    codes = [entry["hreflang"] for entry in entries]

    invalid = [code for code in codes
               if not HREFLANG_RE.match(code) or code.lower() in KNOWN_BAD_HREFLANG]
    if invalid:
        report.add(Finding(
            id=f"seo.hreflang.invalid-codes.{page.url}",
            severity="major", rule="hreflang", wcag_sc=None, url=page.url,
            viewport=None, selector="link[rel=alternate][hreflang]", section="head",
            message=f"Invalid hreflang code(s): {', '.join(invalid)}.",
            evidence={"invalid": invalid, "allCodes": codes,
                      "rule": "ISO 639-1 language, optional ISO 3166-1 Alpha-2 region; "
                              "en-UK is invalid (the region is GB) and uk is Ukrainian"},
            how_to_fix="Use valid codes such as en, en-GB, es-MX.",
        ))

    if "x-default" not in [c.lower() for c in codes]:
        report.add(Finding(
            id=f"seo.hreflang.no-x-default.{page.url}",
            severity="minor", rule="hreflang", wcag_sc=None, url=page.url,
            viewport=None, selector="link[rel=alternate][hreflang]", section="head",
            message="The hreflang cluster has no x-default.",
            evidence={"codes": codes, "expected": "exactly one x-default"},
            how_to_fix="Add one x-default pointing at the language-selection or "
                       "fallback page.",
        ))

    # Reciprocity: an alternate must point back. Only alternates that were
    # crawled can be checked; the rest are reported as unverified.
    unverified, non_reciprocal = [], []
    for entry in entries:
        target = normalise(entry["href"])
        alternate = all_pages.get(target)
        if alternate is None:
            unverified.append(entry["href"])
            continue
        back = {normalise(e["href"]) for e in alternate.rendered.get("hreflang", [])}
        if page.url not in back:
            non_reciprocal.append(entry["href"])
    if non_reciprocal:
        report.add(Finding(
            id=f"seo.hreflang.non-reciprocal.{page.url}",
            severity="major", rule="hreflang", wcag_sc=None, url=page.url,
            viewport=None, selector="link[rel=alternate][hreflang]", section="head",
            message=f"{len(non_reciprocal)} hreflang alternate(s) do not link back. "
                    f"Non-reciprocal hreflang is silently ignored.",
            evidence={"nonReciprocal": non_reciprocal[:8],
                      "unverified": unverified[:8]},
            how_to_fix="Every language pair must reference the other in both "
                       "directions, and hreflang must point at each alternate's "
                       "canonical URL.",
        ))


def check_structured_data(report: Report, page: Page) -> dict[str, int]:
    """L6-structured-data.md — JSON-LD presence, required properties, type inventory.

    Returns:
        A ``{schema type: count}`` inventory for the site-level summary.
    """
    blocks = page.rendered.get("jsonld", [])
    inventory: dict[str, int] = {}

    if not blocks:
        report.add(Finding(
            id=f"seo.schema.none.{page.url}",
            severity="minor", rule="structured-data", wcag_sc=None, url=page.url,
            viewport=None, selector='script[type="application/ld+json"]', section="head",
            message="No JSON-LD structured data.",
            evidence={"jsonLdBlocks": 0, "expected": ">= 1"},
            how_to_fix="Add JSON-LD for the types that apply. Note that structured "
                       "data is not required for AI Overviews or AI Mode, and there is "
                       "no special markup to add for them.",
        ))

    if page.rendered.get("microdataItemprops", 0) > 0:
        report.add(Finding(
            id=f"seo.schema.microdata.{page.url}",
            severity="advisory", rule="structured-data", wcag_sc=None, url=page.url,
            viewport=None, selector="[itemprop]", section="content",
            message=f"{page.rendered['microdataItemprops']} itemprop attribute(s) "
                    f"scattered through the body.",
            evidence={"itempropCount": page.rendered["microdataItemprops"]},
            how_to_fix="Prefer JSON-LD in the head over inline microdata.",
        ))

    for index, block in enumerate(blocks):
        if isinstance(block, dict) and "__parseError" in block:
            report.add(Finding(
                id=f"seo.schema.parse-error.{page.url}.{index}",
                severity="major", rule="structured-data", wcag_sc=None, url=page.url,
                viewport=None, selector='script[type="application/ld+json"]',
                section="head",
                message="A JSON-LD block is not valid JSON and will be ignored.",
                evidence={"error": block["__parseError"]},
                how_to_fix="Fix the JSON syntax.",
            ))
            continue
        for node in _iter_schema_nodes(block):
            types = node.get("@type")
            for schema_type in ([types] if isinstance(types, str) else (types or [])):
                inventory[schema_type] = inventory.get(schema_type, 0) + 1
                _check_schema_node(report, page, schema_type, node, index)

    _check_breadcrumb_agreement(report, page, blocks)
    return inventory


def _iter_schema_nodes(block: Any) -> Iterable[dict[str, Any]]:
    """Yield every object in a JSON-LD block, walking @graph and arrays."""
    if isinstance(block, list):
        for item in block:
            yield from _iter_schema_nodes(item)
    elif isinstance(block, dict):
        if "@graph" in block:
            yield from _iter_schema_nodes(block["@graph"])
        if "@type" in block:
            yield block


def _check_schema_node(report: Report, page: Page, schema_type: str,
                       node: dict[str, Any], index: int) -> None:
    """Validate one JSON-LD node against the reference's required properties."""
    if schema_type in RETIRED_SCHEMA_TYPES:
        report.add(Finding(
            id=f"seo.schema.retired.{schema_type}.{page.url}",
            severity="advisory", rule="structured-data-retired", wcag_sc=None,
            url=page.url, viewport=None,
            selector='script[type="application/ld+json"]', section="head",
            message=f"{schema_type} markup is present but its rich result is retired.",
            evidence={"type": schema_type, "reason": RETIRED_SCHEMA_TYPES[schema_type]},
            how_to_fix="Informational only. Do not rip out existing markup — the "
                       "vocabulary is still valid. Just do not build anything new on "
                       "it.",
        ))

    required = REQUIRED_SCHEMA_PROPERTIES.get(schema_type)
    if required:
        missing = [prop for prop in required if not node.get(prop)]
        if missing:
            report.add(Finding(
                id=f"seo.schema.missing-properties.{schema_type}.{page.url}.{index}",
                severity="minor", rule="structured-data", wcag_sc=None, url=page.url,
                viewport=None, selector='script[type="application/ld+json"]',
                section="head",
                message=f"{schema_type} is missing required properties: "
                        f"{', '.join(missing)}.",
                evidence={"type": schema_type, "missing": missing,
                          "required": list(required)},
                how_to_fix=f"Populate every required property for {schema_type}.",
            ))

    # L6 §6.1: "author must be a Person entity object, not a bare text string".
    if schema_type in ("Article", "BlogPosting") and isinstance(node.get("author"), str):
        report.add(Finding(
            id=f"seo.schema.author-string.{page.url}.{index}",
            severity="minor", rule="structured-data", wcag_sc=None, url=page.url,
            viewport=None, selector='script[type="application/ld+json"]', section="head",
            message="Article `author` is a bare text string rather than a Person "
                    "entity object.",
            evidence={"author": node["author"]},
            how_to_fix='Use {"@type": "Person", "name": ..., "url": ..., '
                       '"sameAs": [...]}.',
        ))

    if schema_type == "Product":
        offers = node.get("offers")
        offer = offers[0] if isinstance(offers, list) and offers else offers
        if isinstance(offer, dict):
            missing = [prop for prop in ("price", "priceCurrency", "availability")
                       if not offer.get(prop)]
            if missing:
                report.add(Finding(
                    id=f"seo.schema.product-offers.{page.url}.{index}",
                    severity="minor", rule="structured-data", wcag_sc=None,
                    url=page.url, viewport=None,
                    selector='script[type="application/ld+json"]', section="head",
                    message=f"Product offers is missing {', '.join(missing)}.",
                    evidence={"missing": missing},
                    how_to_fix="Populate offers.price, offers.priceCurrency and "
                               "offers.availability, and keep them matching what the "
                               "page displays right now — mismatched markup is a "
                               "policy violation, not a technicality.",
                ))


def _check_breadcrumb_agreement(report: Report, page: Page,
                                blocks: list[Any]) -> None:
    """L6 §6.1 / L3 — BreadcrumbList must match the visible trail exactly."""
    visible = [text for text in page.rendered.get("breadcrumb", []) if text]
    if not visible:
        return
    marked: list[str] = []
    for block in blocks:
        for node in _iter_schema_nodes(block):
            if node.get("@type") != "BreadcrumbList":
                continue
            for item in node.get("itemListElement", []) or []:
                name = item.get("name") if isinstance(item, dict) else None
                if not name and isinstance(item, dict):
                    inner = item.get("item")
                    name = inner.get("name") if isinstance(inner, dict) else None
                if name:
                    marked.append(str(name).strip())
    if marked and marked != visible:
        report.add(Finding(
            id=f"seo.schema.breadcrumb-mismatch.{page.url}",
            severity="minor", rule="structured-data", wcag_sc=None, url=page.url,
            viewport=None, selector='script[type="application/ld+json"]', section="head",
            message="BreadcrumbList markup does not match the visible breadcrumb trail.",
            evidence={"markup": marked, "visible": visible,
                      "rule": "identical order and labels"},
            how_to_fix="Generate the markup from the same data as the visible trail. "
                       "Markup that disagrees with the page is a policy violation.",
        ))


def check_images(report: Report, page: Page) -> tuple[int, int]:
    """L4-onpage.md — alt coverage, and the missing-vs-empty distinction.

    Returns:
        ``(total, missing_alt)`` for the site-level coverage summary.
    """
    images = page.rendered.get("images", [])
    missing = [img for img in images if not img["hasAlt"]]
    if missing:
        report.add(Finding(
            id=f"seo.images.missing-alt.{page.url}",
            severity="minor", rule="image-alt", wcag_sc="1.1.1 (A)", url=page.url,
            viewport=None, selector="img", section="content",
            message=f"{len(missing)} of {len(images)} image(s) have no alt attribute.",
            evidence={"missing": len(missing), "total": len(images),
                      "examples": [img["src"] for img in missing[:8]],
                      "rule": 'a missing alt is not the same as alt="" — decorative '
                              'images get an explicitly empty alt'},
            how_to_fix='Add alt describing the image\'s content and function; use '
                       'alt="" for decorative images.',
        ))
    generic_filenames = [img for img in images
                         if re.match(r"^(img|dsc|image|photo)[-_]?\d+\.",
                                     img["filename"], re.IGNORECASE)]
    if generic_filenames:
        report.add(Finding(
            id=f"seo.images.filenames.{page.url}",
            severity="advisory", rule="image-filename", wcag_sc=None, url=page.url,
            viewport=None, selector="img", section="content",
            message=f"{len(generic_filenames)} image(s) use camera-default filenames.",
            evidence={"examples": [img["filename"] for img in generic_filenames[:8]]},
            how_to_fix="Rename to descriptive filenames such as "
                       "roof-repair-flashing-detail.webp.",
        ))
    return len(images), len(missing)


def check_rendering(report: Report, page: Page) -> str:
    """L1-foundations.md §1.4 — is this page server-rendered or client-rendered?

    Returns:
        ``"ssr"``, ``"partial-csr"`` or ``"csr"`` for the per-route summary the
        reference asks for ("Per-route if routes differ").
    """
    raw = page.raw_html
    raw_text = re.sub(r"<script[^>]*>.*?</script>", " ", raw,
                      flags=re.DOTALL | re.IGNORECASE)
    raw_text = re.sub(r"<[^>]*>", " ", raw_text)
    raw_text = re.sub(r"\s+", " ", raw_text).strip()
    rendered_length = page.rendered.get("textLength", 0) or 1
    ratio = round(len(raw_text) / rendered_length, 3)

    # The reference's highest-value machine check: the head tags must be in the
    # raw response, not injected client-side.
    head_tags = {
        "title": bool(re.search(r"<title[^>]*>\s*\S", raw, re.IGNORECASE)),
        "meta description": bool(re.search(
            r'<meta[^>]+name=["\']description["\']', raw, re.IGNORECASE)),
        "canonical": bool(re.search(r'rel=["\']canonical["\']', raw, re.IGNORECASE)),
        "json-ld": "application/ld+json" in raw.lower(),
    }
    injected = [name for name, present in head_tags.items()
                if not present and _rendered_has(page, name)]

    verdict = "ssr"
    if ratio < CSR_TEXT_RATIO_THRESHOLD:
        verdict = "csr"
    elif injected:
        verdict = "partial-csr"

    if verdict != "ssr":
        report.add(Finding(
            id=f"seo.rendering.{verdict}.{page.url}",
            severity="major" if verdict == "csr" else "minor",
            rule="client-side-rendering", wcag_sc=None, url=page.url,
            viewport=None, selector=None, section="rendering",
            message=(f"Raw HTML carries only {int(ratio * 100)}% of the rendered text"
                     + (f"; these head tags are injected by JavaScript: "
                        f"{', '.join(injected)}" if injected else "") + "."),
            evidence={
                "rawTextChars": len(raw_text),
                "renderedTextChars": rendered_length,
                "ratio": ratio,
                "ratioThreshold": f"{CSR_TEXT_RATIO_THRESHOLD} — SCRIPT DECISION; the "
                                  f"reference compares raw and rendered qualitatively "
                                  f"and states no number",
                "headTagsInRawHtml": head_tags,
                "rawHtmlSample": raw_text[:RAW_HTML_WINDOW][:300],
            },
            how_to_fix="Server-render or pre-render the primary content and the head "
                       "tags. Non-Google consumers — Bing, ChatGPT's crawlers, "
                       "Perplexity, in-chat fetchers and link previews — execute far "
                       "less JavaScript than Googlebot does; none of the major AI "
                       "crawlers render it at all. Confirm with Search Console's URL "
                       "Inspection > Test Live URL > View Tested Page > HTML, which is "
                       "the definitive answer for Google.",
        ))
    return verdict


def _rendered_has(page: Page, tag_name: str) -> bool:
    """Whether the rendered DOM carries the named head tag."""
    return {
        "title": bool(page.rendered.get("title")),
        "meta description": bool(page.description),
        "canonical": bool(page.rendered.get("canonical")),
        "json-ld": bool(page.rendered.get("jsonld")),
    }[tag_name]


# --------------------------------------------------------------------------- #
# Site-level checks
# --------------------------------------------------------------------------- #


def check_duplicates(report: Report, pages: dict[str, Page], start: str) -> None:
    """L4-onpage.md — zero duplicate titles or descriptions."""
    for label, extractor, rule in (
        ("title", lambda p: (p.rendered.get("title") or "").strip(), "title"),
        ("meta description", lambda p: (p.description or "").strip(),
         "meta-description"),
    ):
        buckets: dict[str, list[str]] = {}
        for page in pages.values():
            value = extractor(page)
            if value:
                buckets.setdefault(value, []).append(page.url)
        duplicates = {value: urls for value, urls in buckets.items() if len(urls) > 1}
        if duplicates:
            report.add(Finding(
                id=f"seo.duplicate.{rule}",
                severity="major" if rule == "title" else "minor",
                rule=f"duplicate-{rule}", wcag_sc=None, url=start,
                viewport=None, selector=None, section="site",
                message=f"{len(duplicates)} {label} value(s) are reused across pages.",
                evidence={"duplicates": {value[:70]: urls[:5]
                                         for value, urls in list(duplicates.items())[:8]},
                          "pagesCrawled": len(pages)},
                how_to_fix=f"Give every page a distinct {label}.",
            ))


def check_canonical_sample(report: Report, pages: dict[str, Page], start: str) -> None:
    """L1-foundations.md §1.3 — the "all canonicals point at /" misconfiguration."""
    sample = [page for page in list(pages.values())[:CANONICAL_SAMPLE]
              if page.rendered.get("canonical")]
    if len(sample) < 2:
        return
    distinct = {page.rendered["canonical"] for page in sample}
    if len(distinct) == 1:
        report.add(Finding(
            id="seo.canonical.all-point-home",
            severity="blocker", rule="canonical", wcag_sc=None, url=start,
            viewport=None, selector="link[rel=canonical]", section="site",
            message=f"All {len(sample)} sampled pages share one canonical URL; this "
                    f"de-indexes the whole site.",
            evidence={"sampledPages": len(sample), "distinctCanonicals": 1,
                      "canonical": distinct.pop(),
                      "expected": f"{len(sample)} distinct canonicals"},
            how_to_fix="Fix the CMS/theme setting emitting a single hardcoded "
                       "canonical.",
        ))


def check_link_graph(report: Report, pages: dict[str, Page],
                     inlinks: dict[str, set[str]], sitemap_urls: list[str],
                     start: str) -> None:
    """L3-architecture.md — click depth, orphans, boilerplate bloat, anchor text."""
    deep = {url: page.depth for url, page in pages.items()
            if page.depth >= DEEP_PAGE_DEPTH}
    over_target = {url: page.depth for url, page in pages.items()
                   if MAX_CLICK_DEPTH < page.depth < DEEP_PAGE_DEPTH}
    if deep:
        report.add(Finding(
            id="seo.link-graph.deep-pages",
            severity="major", rule="click-depth", wcag_sc=None, url=start,
            viewport=None, selector=None, section="architecture",
            message=f"{len(deep)} crawled page(s) sit {DEEP_PAGE_DEPTH}+ clicks from "
                    f"the homepage.",
            evidence={"threshold": f"important pages at depth <= {MAX_CLICK_DEPTH}",
                      "deepest": sorted(deep.items(), key=lambda kv: -kv[1])[:10],
                      "alsoOverTarget": len(over_target)},
            how_to_fix="Link commercially important pages from hub pages so nothing "
                       "important is more than three clicks from the homepage. Depth "
                       "5+ is typically a page buried under a blog archive.",
        ))

    if sitemap_urls:
        crawled = set(pages)
        orphans = [url for url in sitemap_urls
                   if url not in crawled and not inlinks.get(url)]
        if orphans:
            report.add(Finding(
                id="seo.link-graph.orphans",
                severity="minor", rule="orphan-pages", wcag_sc=None, url=start,
                viewport=None, selector=None, section="architecture",
                message=f"{len(orphans)} sitemap URL(s) have no internal links pointing "
                        f"at them within the crawled sample.",
                evidence={"examples": orphans[:10], "sitemapUrls": len(sitemap_urls),
                          "pagesCrawled": len(pages),
                          "definition": "orphans are in the sitemap with zero internal "
                                        "links pointing at them",
                          "caveat": "a partial crawl over-reports orphans; raise "
                                    "--max-pages before acting on this"},
                how_to_fix="Link them from a relevant hub, or remove them from the "
                           "sitemap if they should not be indexed.",
            ))

    for page in pages.values():
        links = page.rendered.get("links", [])
        boilerplate = [link for link in links if link["inFooter"] or link["inNav"]]
        if len(boilerplate) > BOILERPLATE_LINK_LIMIT:
            report.add(Finding(
                id=f"seo.link-graph.boilerplate.{page.url}",
                severity="minor", rule="link-volume", wcag_sc=None, url=page.url,
                viewport=None, selector="footer a, nav a", section="architecture",
                message=f"{len(boilerplate)} boilerplate nav/footer links on this page, "
                        f"which flattens the priority signal.",
                evidence={"boilerplateLinks": len(boilerplate),
                          "totalLinks": len(links),
                          "threshold": BOILERPLATE_LINK_LIMIT},
                how_to_fix="Cut the repeated link block down to what users actually "
                           "need; prefer contextual links.",
            ))
            break  # one finding is enough; it is the same block on every page

    generic = []
    for page in pages.values():
        for link in page.rendered.get("links", []):
            if link["text"].strip().lower() in GENERIC_ANCHOR_TEXT:
                generic.append({"page": page.url, "href": link["href"],
                                "text": link["text"]})
    if generic:
        report.add(Finding(
            id="seo.link-graph.generic-anchors",
            severity="advisory", rule="anchor-text", wcag_sc="2.4.4 (A)", url=start,
            viewport=None, selector="a", section="architecture",
            message=f"{len(generic)} link(s) use generic anchor text.",
            evidence={"phrases": list(GENERIC_ANCHOR_TEXT),
                      "examples": generic[:8]},
            how_to_fix="Describe the destination in the link text.",
        ))


def check_broken_links(report: Report, pages: dict[str, Page],
                       inlinks: dict[str, set[str]], start: str, timeout: float,
                       check_external: bool, limit: int) -> None:
    """L2-technical-performance.md §2.3 — internal 4xx and redirect chains."""
    origin_host = urlparse(start).hostname
    candidates = [url for url in inlinks
                  if url not in pages
                  and (check_external or urlparse(url).hostname == origin_host)]
    broken: list[dict[str, Any]] = []
    chains: list[dict[str, Any]] = []
    for url in candidates[:limit]:
        status, hops = status_of(url, timeout)
        internal = urlparse(url).hostname == origin_host
        if status == 0 or 400 <= status < 600:
            broken.append({"url": url, "status": status, "internal": internal,
                           "linkedFrom": sorted(inlinks[url])[:3]})
        elif len(hops) > MAX_REDIRECT_HOPS:
            chains.append({"url": url, "hops": hops, "internal": internal})

    internal_broken = [item for item in broken if item["internal"]]
    if internal_broken:
        report.add(Finding(
            id="seo.links.broken-internal",
            severity="major", rule="broken-link", wcag_sc=None, url=start,
            viewport=None, selector=None, section="site",
            message=f"{len(internal_broken)} internal link(s) return an error status.",
            evidence={"threshold": "zero internal 4xx", "links": internal_broken[:10]},
            how_to_fix="Fix or remove the links. Internal 4xx is a high-priority "
                       "finding; external 4xx is lower priority but still worth fixing.",
        ))
    external_broken = [item for item in broken if not item["internal"]]
    if external_broken:
        report.add(Finding(
            id="seo.links.broken-external",
            severity="minor", rule="broken-link", wcag_sc=None, url=start,
            viewport=None, selector=None, section="site",
            message=f"{len(external_broken)} external link(s) return an error status.",
            evidence={"links": external_broken[:10]},
            how_to_fix="Update or remove them.",
        ))
    if chains:
        report.add(Finding(
            id="seo.links.redirect-chains",
            severity="minor", rule="redirect-chain", wcag_sc=None, url=start,
            viewport=None, selector=None, section="site",
            message=f"{len(chains)} link(s) go through more than one redirect.",
            evidence={"threshold": f"at most {MAX_REDIRECT_HOPS} hop before the 200",
                      "chains": chains[:8]},
            how_to_fix="Point internal links at the final URL; every hop is wasted "
                       "crawl and diluted signal.",
        ))


def check_robots_txt(report: Report, start: str, status: int | None,
                     robots_body_sitemaps: list[str]) -> None:
    """L1-foundations.md §1.2 — robots.txt reachability and sitemap declaration."""
    if status is None or status >= 500:
        report.add(Finding(
            id="seo.robots-txt.unreachable",
            severity="blocker", rule="robots-txt", wcag_sc=None, url=start,
            viewport=None, selector=None, section="crawlability",
            message=f"/robots.txt returned {status or 'a connection error'}.",
            evidence={"status": status,
                      "note": "a 404 is survivable (everything is allowed); a 5xx is "
                              "not — crawlers may back off entirely"},
            how_to_fix="Serve robots.txt with a 200 and content-type text/plain.",
        ))
    elif not robots_body_sitemaps:
        report.add(Finding(
            id="seo.robots-txt.no-sitemap",
            severity="minor", rule="robots-txt", wcag_sc=None, url=start,
            viewport=None, selector=None, section="crawlability",
            message="robots.txt declares no Sitemap: URL.",
            evidence={"expected": "at least one absolute Sitemap: URL that itself "
                                  "returns 200"},
            how_to_fix="Add an absolute `Sitemap:` line.",
        ))


def check_sitemap(report: Report, start: str, sitemap_urls: list[str],
                  timeout: float, sample: int = 20) -> None:
    """L1-foundations.md §1.3 — sitemap size and status codes."""
    if not sitemap_urls:
        return
    if len(sitemap_urls) >= SITEMAP_MAX_URLS:
        report.add(Finding(
            id="seo.sitemap.too-large",
            severity="minor", rule="sitemap", wcag_sc=None, url=start,
            viewport=None, selector=None, section="crawlability",
            message=f"A sitemap carries {len(sitemap_urls)} URLs.",
            evidence={"threshold": SITEMAP_MAX_URLS,
                      "alsoLimited": "50 MB uncompressed"},
            how_to_fix="Split into multiple sitemaps behind a sitemap index.",
        ))
    bad: list[dict[str, Any]] = []
    for url in sitemap_urls[:sample]:
        status, hops = status_of(url, timeout)
        if status != 200:
            bad.append({"url": url, "status": status, "hops": hops})
    if bad:
        report.add(Finding(
            id="seo.sitemap.non-200",
            severity="minor", rule="sitemap", wcag_sc=None, url=start,
            viewport=None, selector=None, section="crawlability",
            message=f"{len(bad)} of {min(sample, len(sitemap_urls))} sampled sitemap "
                    f"URLs do not return 200.",
            evidence={"threshold": "every <loc> returns 200", "urls": bad[:10]},
            how_to_fix="A sitemap should contain only canonical, 200-status, indexable "
                       "URLs. Any 301, 404 or noindex URL in it is a wasted crawl "
                       "signal.",
        ))


def check_soft_404(report: Report, start: str, timeout: float) -> None:
    """L1-foundations.md — a definitely-missing URL must return 404."""
    probe = urljoin(start, "/definitely-not-real-xyz-audit-probe")
    status, hops = status_of(probe, timeout)
    if status == 200:
        report.add(Finding(
            id="seo.soft-404",
            severity="major", rule="soft-404", wcag_sc=None, url=probe,
            viewport=None, selector=None, section="crawlability",
            message="A URL that cannot exist returns 200 — a soft 404.",
            evidence={"status": 200, "expected": 404, "redirectHops": hops},
            how_to_fix="Return a real 404 (or 410). Redirecting everything to the "
                       "homepage is the same defect.",
        ))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def build_parser():
    """Construct the argument parser."""
    parser = base_parser(
        "audit_seo.py",
        "Crawl a site and audit the mechanical SEO checks against the rendered page.",
        epilog=(
            "Thresholds come from references/seo/L1-foundations.md, L3-architecture.md,\n"
            "L4-onpage.md (including the social-card set and severities in §4.3) and\n"
            "L6-structured-data.md. One item is a script decision rather than a doc\n"
            "threshold and is labelled as such in the output: the raw-vs-rendered\n"
            "text ratio."
        ),
    )
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES,
                        help=f"Maximum pages to crawl (default: {DEFAULT_MAX_PAGES})")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS,
                        metavar="SECONDS",
                        help=f"Delay between requests (default: {DEFAULT_DELAY_SECONDS}; "
                             f"the references state no crawl-rate guidance, so this is "
                             f"a script default)")
    parser.add_argument("--ignore-robots", action="store_true",
                        help="Crawl URLs robots.txt disallows (use only on sites you own)")
    parser.add_argument("--check-external", action="store_true",
                        help="Also status-check outbound external links")
    parser.add_argument("--link-check-limit", type=int, default=200, metavar="N",
                        help="Maximum links to status-check (default: 200)")
    return parser


def main() -> int:
    """Run the SEO crawl and return the process exit code."""
    args = build_parser().parse_args()
    start = normalise_url(args.url)

    report = Report(tool="audit_seo", target=start)
    robots, sitemap_locations, robots_status = load_robots(start, args.timeout)
    check_robots_txt(report, start, robots_status, sitemap_locations)
    sitemap_urls = parse_sitemap_urls(sitemap_locations, args.timeout)
    report.meta["sitemap_urls_found"] = len(sitemap_urls)

    sync_playwright = import_playwright()
    with sync_playwright() as playwright:
        browser = launch_chromium(playwright)
        try:
            pages, inlinks = crawl(
                browser, start, max_pages=args.max_pages, delay=args.delay,
                timeout=args.timeout, robots=robots,
                obey_robots=not args.ignore_robots,
            )
        finally:
            browser.close()

    if not pages:
        raise TargetUnreachable(start, "no pages could be crawled")

    report.meta["pages_crawled"] = len(pages)
    report.meta["max_pages"] = args.max_pages
    report.meta["robots_txt_obeyed"] = not args.ignore_robots

    schema_inventory: dict[str, int] = {}
    rendering_by_route: dict[str, str] = {}
    images_total = images_missing = 0

    for page in pages.values():
        if page.status != 200 or not page.rendered:
            continue
        check_title(report, page)
        check_description(report, page)
        check_canonical(report, page)
        check_robots_directives(report, page)
        check_headings(report, page)
        check_social(report, page)
        check_hreflang(report, page, pages)
        for schema_type, count in check_structured_data(report, page).items():
            schema_inventory[schema_type] = schema_inventory.get(schema_type, 0) + count
        total, missing = check_images(report, page)
        images_total += total
        images_missing += missing
        rendering_by_route[urlparse(page.url).path or "/"] = check_rendering(report, page)

    check_duplicates(report, pages, start)
    check_social_duplicates(report, pages, start)
    check_canonical_sample(report, pages, start)
    check_link_graph(report, pages, inlinks, sitemap_urls, start)
    check_broken_links(report, pages, inlinks, start, args.timeout,
                       args.check_external, args.link_check_limit)
    check_sitemap(report, start, sitemap_urls, args.timeout)
    check_soft_404(report, start, args.timeout)

    report.meta["schema_type_inventory"] = dict(sorted(schema_inventory.items()))
    report.meta["rendering_by_route"] = rendering_by_route
    report.meta["image_alt_coverage"] = (
        f"{images_total - images_missing}/{images_total} images carry an alt attribute"
        if images_total else "no images found"
    )
    report.meta["max_click_depth_seen"] = max((p.depth for p in pages.values()),
                                              default=0)

    report.note(
        f"{len(pages)} page(s) were crawled out of a --max-pages limit of "
        f"{args.max_pages}. A finding count without a denominator is meaningless; "
        "state the sample in the written report, and treat orphan and depth findings "
        "as provisional until the crawl covers the whole site."
    )
    report.note(
        "Structured data was checked at the vocabulary level only. Validate in both "
        "Google's Rich Results Test and validator.schema.org before signing anything "
        "off — Google's tool only reports on types it supports and silently ignores "
        "valid-but-unsupported markup."
    )
    report.note(
        "Not checkable from outside: whether the heading outline reads as a genuine "
        "table of contents, whether structured data matches what the page actually "
        "displays over time, and anything behind Search Console (index coverage, "
        "crawl stats, International Targeting, Enhancements errors)."
    )
    report.note(
        "Social cards were checked for presence, absolute-URL form, canonical "
        "agreement and raw-HTML availability. NOT checked: whether og:image is "
        "actually fetchable, its real pixel dimensions, its MIME type or its byte "
        "size — fetch it and read those against the constraint table in "
        "references/seo/L4-onpage.md §4.3, then confirm the resolved card in the "
        "Facebook Sharing Debugger and the LinkedIn Post Inspector, which are also "
        "the only way to clear a stale per-URL cache."
    )
    report.note(
        "The definitive rendering answer for Google is Search Console's URL Inspection "
        "> Test Live URL > View Tested Page > HTML tab. This script's raw-vs-rendered "
        "ratio is a proxy and its threshold is a script decision, not a documented one."
    )
    return finish(report, args)


if __name__ == "__main__":
    run_cli(main)
