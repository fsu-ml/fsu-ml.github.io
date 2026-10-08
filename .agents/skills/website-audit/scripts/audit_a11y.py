#!/usr/bin/env python3
"""Run axe-core against the rendered page, and say what axe could not test.

The engine is axe-core driven through Playwright, which is what
``references/ada/testing.md`` §1 prescribes. The part that matters more is the
honesty layer, from the same section:

    **Automated tools detect roughly 30-40% of WCAG failures.** ... Deque
    reports that axe-core catches around **57% of issues by volume**, which is
    a different and more flattering measurement ... Neither number licenses
    "we ran axe and it passed."

    > **Automated testing is a regression net, not an audit.**

So every run of this script reports three counts, not one:

1. violations — what axe found
2. incomplete — what axe found but could not decide (needs review)
3. **criteria not automatable at all** — the success criteria no scanner can
   evaluate, enumerated from ``testing.md`` and ``wcag22-new.md``

Without (3) in the output, a clean report implies coverage that does not exist.

Beyond axe this script performs the sweeps the ADA references name explicitly:

* ``--images``       image accessible names; missing alt, filename-like alt,
                     alt starting with "image of" (``ada/html-core.md`` §5)
* ``--contrast``     the automatable contrast sweep, with the incomplete cases
                     surfaced rather than silently passed (``html-core.md`` §8)
* ``--forms``        the label / autocomplete / required sweep
                     (``ada/html-forms.md``)
* ``--target-size``  WCAG 2.2 SC 2.5.8 (``ada/wcag22-new.md``)
* ``--inventory-documents``
                     crawl for linked PDF/Office documents and report which
                     document-accessibility references need loading
                     (``ada/00-map.md``)

Usage:
    ./audit_a11y.py https://example.com
    ./audit_a11y.py https://example.com --all --standard wcag22aa
    ./audit_a11y.py https://example.com --inventory-documents --max-pages 50
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urldefrag, urlparse

from _common import (
    Finding,
    MissingDependency,
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
# axe-core
# --------------------------------------------------------------------------- #

#: Pinned so two runs of the audit are comparable. axe-core has had the
#: ``target-size`` rule since 4.5 (``mobile.md`` §3.5).
DEFAULT_AXE_VERSION = "4.10.2"
AXE_CDN = "https://cdn.jsdelivr.net/npm/axe-core@{version}/axe.min.js"

#: WCAG level/version selector -> axe tag set.
#: ``ada/00-map.md`` spells the profile targets with a hyphen (``wcag22-aa``);
#: axe's own tags have none. Both spellings are accepted on the command line.
#: ``wcag20aa`` is a real profile target, not a legacy spelling: Section 508
#: incorporates **WCAG 2.0 Level AA** by reference (``ada/targets.md`` §1.3), as
#: does the Air Carrier Access Act (§1.5).
STANDARD_TAGS: dict[str, list[str]] = {
    "wcag20aa": ["wcag2a", "wcag2aa"],
    "wcag21aa": ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"],
    "wcag22aa": ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"],
}
STANDARD_ALIASES = {
    "wcag20-aa": "wcag20aa",
    "wcag21-aa": "wcag21aa",
    "wcag22-aa": "wcag22aa",
}

#: ``ada/program.md`` §4 Tier 1 — "blocks access entirely". Tier must be
#: assigned from the criterion, not from axe's impact field: axe rates
#: ``color-contrast`` as *serious*, but the ADA references classify contrast as
#: Tier 2 and unlabelled controls / missing functional alt as Tier 1.
TIER1_RULES = frozenset({
    "image-alt", "input-image-alt", "area-alt", "object-alt", "role-img-alt",
    "svg-img-alt", "label", "label-title-only", "form-field-multiple-labels",
    "select-name", "button-name", "link-name", "input-button-name",
    "aria-input-field-name", "aria-toggle-field-name", "frame-title",
    "server-side-image-map", "blink", "marquee", "no-autoplay-audio",
    "video-caption", "bypass", "accesskeys", "scrollable-region-focusable",
})

#: axe impact -> severity, used for everything not in :data:`TIER1_RULES`.
IMPACT_TO_SEVERITY = {
    "critical": "major",
    "serious": "major",
    "moderate": "minor",
    "minor": "advisory",
    None: "minor",
}

#: ``ada/testing.md`` line 180 — the criteria that appear in a report only if
#: the manual layers were actually run. "If every finding maps to an axe rule
#: ID, you ran Layer 1 only."
MANUAL_ONLY_CRITERIA: tuple[tuple[str, str], ...] = (
    ("1.3.2", "Meaningful Sequence (A) — is the reading order coherent?"),
    ("2.4.3", "Focus Order (A) — is the focus order logical?"),
    ("2.5.7", "Dragging Movements (AA) — is there a single-pointer alternative?"),
    ("3.2.6", "Consistent Help (A) — is help in the same place on every page?"),
    ("3.3.3", "Error Suggestion (AA) — is the error message actually helpful?"),
    ("3.3.7", "Redundant Entry (A) — is anything re-entered in the same process?"),
)

#: ``ada/wcag22-new.md`` line 7 — of the six new WCAG 2.2 criteria, only 2.5.8
#: is reliably automatable.
WCAG22_NEW_CRITERIA = (
    "2.4.11 Focus Not Obscured (Minimum)",
    "2.5.7 Dragging Movements",
    "2.5.8 Target Size (Minimum)",
    "3.2.6 Consistent Help",
    "3.3.7 Redundant Entry",
    "3.3.8 Accessible Authentication (Minimum)",
)

# --------------------------------------------------------------------------- #
# Contrast thresholds — ada/html-core.md §8
# --------------------------------------------------------------------------- #

CONTRAST_NORMAL_TEXT = 4.5     # 1.4.3 Contrast (Minimum), AA
CONTRAST_LARGE_TEXT = 3.0      # 1.4.3 AA — large = >=18pt/24px, or >=14pt/18.66px bold
CONTRAST_NON_TEXT = 3.0        # 1.4.11 Non-text Contrast, AA

# --------------------------------------------------------------------------- #
# Document inventory — ada/00-map.md, ada/documents-*.md
# --------------------------------------------------------------------------- #

#: extension -> (profile key, reference file to load)
DOCUMENT_TYPES: dict[str, tuple[str, str]] = {
    ".pdf": ("content.has_pdfs", "references/ada/documents-pdf.md"),
    ".doc": ("content.has_office_docs", "references/ada/documents-office.md"),
    ".docx": ("content.has_office_docs", "references/ada/documents-office.md"),
    ".ppt": ("content.has_office_docs", "references/ada/documents-office.md"),
    ".pptx": ("content.has_office_docs", "references/ada/documents-office.md"),
    ".xls": ("content.has_office_docs", "references/ada/documents-office.md"),
    ".xlsx": ("content.has_office_docs", "references/ada/documents-office.md"),
    ".epub": ("content.has_epub", "references/ada/documents-office.md"),
    ".tex": ("content.has_latex_pdfs", "references/ada/documents-latex.md"),
}

#: ``ada/documents-office.md`` line 5 and ``documents-pdf.md``: the PDF file
#: carries the WCAG2ICT rules the other two depend on.
DOCUMENT_PREREQUISITE = "references/ada/documents-pdf.md"

#: ``ada/testing.md`` line 95 — "PDFs and documents linked from the top 50 pages".
DEFAULT_MAX_PAGES = 50


# --------------------------------------------------------------------------- #
# In-page JavaScript
# --------------------------------------------------------------------------- #

# ada/html-core.md §5 — every image with its computed accessible name.
IMAGES_JS = r"""() => {
  const nameOf = el => {
    const labelledby = el.getAttribute('aria-labelledby');
    if (labelledby) {
      const text = labelledby.split(/\s+/)
        .map(id => document.getElementById(id)?.textContent?.trim() || '')
        .filter(Boolean).join(' ');
      if (text) return { name: text, from: 'aria-labelledby' };
    }
    const label = el.getAttribute('aria-label');
    if (label && label.trim()) return { name: label.trim(), from: 'aria-label' };
    if (el.tagName === 'IMG' || el.tagName === 'INPUT' || el.tagName === 'AREA') {
      if (el.hasAttribute('alt')) return { name: el.getAttribute('alt'), from: 'alt' };
    }
    const title = el.getAttribute('title');
    if (title && title.trim()) return { name: title.trim(), from: 'title' };
    const caption = el.closest('figure')?.querySelector('figcaption')?.textContent;
    if (caption && caption.trim()) return { name: caption.trim(), from: 'figcaption' };
    return { name: null, from: null };
  };
  const path = el => { const seg = [];
    for (let n = el; n && n.nodeType === 1 && seg.length < 4; n = n.parentElement) {
      let s = n.tagName.toLowerCase();
      if (n.id) { seg.unshift(s + '#' + n.id); break; }
      if (n.classList.length) s += '.' + [...n.classList].slice(0, 2).join('.');
      seg.unshift(s);
    } return seg.join(' > '); };
  const sectionOf = el => {
    const host = el.closest('section, article, header, footer, nav, aside, main, form');
    if (!host) return null;
    return host.tagName.toLowerCase() + (host.id ? '#' + host.id : '');
  };
  const nodes = [...document.querySelectorAll(
    'img, [role=img], svg, input[type=image], area')];
  return nodes.map(el => {
    const { name, from } = nameOf(el);
    const r = el.getBoundingClientRect();
    return {
      tag: el.tagName.toLowerCase(), selector: path(el), section: sectionOf(el),
      src: (el.currentSrc || el.getAttribute('src') || '').slice(-100) || null,
      hasAltAttribute: el.hasAttribute('alt'),
      accessibleName: name, nameSource: from,
      decorativeMarkup: el.getAttribute('role') === 'presentation' ||
                        el.getAttribute('role') === 'none' ||
                        el.getAttribute('aria-hidden') === 'true',
      insideLinkOrButton: !!el.closest('a[href], button, [role=button], [role=link]'),
      size: [Math.round(r.width), Math.round(r.height)]
    };
  });
}"""

# ada/html-forms.md — the automatable label / autocomplete / required sweep.
FORMS_JS = r"""() => {
  const PERSONAL = /name|email|phone|tel|address|city|state|zip|postal|country|card|cc-|birth|dob|company|organi|username|password/i;
  const path = el => { const seg = [];
    for (let n = el; n && n.nodeType === 1 && seg.length < 4; n = n.parentElement) {
      let s = n.tagName.toLowerCase();
      if (n.id) { seg.unshift(s + '#' + n.id); break; }
      if (n.classList.length) s += '.' + [...n.classList].slice(0, 2).join('.');
      seg.unshift(s);
    } return seg.join(' > '); };
  const sectionOf = el => {
    const host = el.closest('form, section, article, main, header, footer, nav');
    if (!host) return null;
    return host.tagName.toLowerCase() + (host.id ? '#' + host.id : '');
  };
  return [...document.querySelectorAll('input:not([type=hidden]), select, textarea')]
    .filter(el => getComputedStyle(el).display !== 'none')
    .map(el => {
      const type = (el.getAttribute('type') || el.tagName.toLowerCase()).toLowerCase();
      const name = el.name || el.id || el.getAttribute('aria-label') || el.placeholder || '';
      const labelled = !!(el.labels?.length || el.getAttribute('aria-label') ||
                          el.getAttribute('aria-labelledby'));
      const issues = [];
      if (!labelled) issues.push('NO_ACCESSIBLE_NAME');
      if (!el.labels?.length && el.placeholder && !el.getAttribute('aria-label'))
        issues.push('PLACEHOLDER_AS_LABEL');
      if (PERSONAL.test(name) && !el.getAttribute('autocomplete'))
        issues.push('MISSING_AUTOCOMPLETE');
      if ((el.getAttribute('autocomplete') || '').toLowerCase() === 'off' &&
          PERSONAL.test(name))
        issues.push('AUTOCOMPLETE_OFF_ON_PERSONAL_FIELD');
      if (el.getAttribute('aria-required') === 'true' && !el.hasAttribute('required'))
        issues.push('ARIA_REQUIRED_WITHOUT_REQUIRED');
      if (type === 'number' && /phone|tel|zip|postal|code|otp|card|ssn|pin/i.test(name))
        issues.push('TYPE_NUMBER_MISUSE');
      return { selector: path(el), section: sectionOf(el), name, type,
               autocomplete: el.getAttribute('autocomplete'),
               required: el.hasAttribute('required'), issues };
    })
    .filter(r => r.issues.length);
}"""


# --------------------------------------------------------------------------- #
# axe-core sourcing
# --------------------------------------------------------------------------- #


def load_axe_source(explicit: str | None, version: str, timeout: float) -> str:
    """Return the axe-core bundle source.

    Resolution order: ``--axe-source``, then a local ``node_modules`` install,
    then the pinned CDN build.

    Raises:
        MissingDependency: If none of the three is available. The message names
            all three so an offline machine has a path forward.
    """
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise MissingDependency(
                f"axe-core bundle at {explicit}",
                "Point --axe-source at an existing axe.min.js",
            )
        return path.read_text(encoding="utf-8")

    for candidate in (
        Path.cwd() / "node_modules" / "axe-core" / "axe.min.js",
        Path(__file__).resolve().parent / "node_modules" / "axe-core" / "axe.min.js",
        Path(__file__).resolve().parent / "axe.min.js",
    ):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")

    url = AXE_CDN.format(version=version)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise MissingDependency(
            f"axe-core {version} (no local copy, and {url} is unreachable: {exc})",
            "npm install axe-core            # then re-run from the same directory\n"
            "# or download axe.min.js and pass --axe-source /path/to/axe.min.js",
        ) from exc


def resolve_standard(value: str) -> tuple[str, list[str]]:
    """Map a ``--standard`` value to its canonical name and axe tag list."""
    key = STANDARD_ALIASES.get(value.lower(), value.lower())
    if key not in STANDARD_TAGS:
        raise ValueError(f"unknown standard {value!r}")
    return key, list(STANDARD_TAGS[key])


# --------------------------------------------------------------------------- #
# axe result conversion
# --------------------------------------------------------------------------- #


def severity_for(rule_id: str, impact: str | None) -> str:
    """Assign severity from the criterion first, axe's impact second."""
    if rule_id in TIER1_RULES:
        return "blocker"
    return IMPACT_TO_SEVERITY.get(impact, "minor")


def wcag_sc_from_tags(tags: Iterable[str]) -> str | None:
    """Turn axe's ``wcag143``-style tags into ``1.4.3``-style criterion numbers."""
    criteria = []
    for tag in tags:
        match = re.fullmatch(r"wcag(\d)(\d)(\d+)", tag)
        if match:
            criteria.append(".".join(match.groups()))
    return ", ".join(sorted(set(criteria))) or None


def findings_from_axe(results: dict[str, Any], url: str,
                      viewport: str) -> list[Finding]:
    """Convert an axe ``violations`` array into report findings."""
    findings: list[Finding] = []
    for violation in results.get("violations", []):
        rule_id = violation["id"]
        severity = severity_for(rule_id, violation.get("impact"))
        for index, node in enumerate(violation.get("nodes", [])):
            selector = " ".join(_flatten(node.get("target", [])))
            findings.append(Finding(
                id=f"a11y.axe.{rule_id}.{index}",
                severity=severity, rule=rule_id,
                wcag_sc=wcag_sc_from_tags(violation.get("tags", [])),
                url=url, viewport=viewport, selector=selector,
                section=_section_hint(selector),
                message=violation.get("help", rule_id),
                evidence={
                    "impact": node.get("impact") or violation.get("impact"),
                    "failureSummary": (node.get("failureSummary") or "").strip()[:400],
                    "html": (node.get("html") or "")[:240],
                    "helpUrl": violation.get("helpUrl"),
                    "tags": violation.get("tags", []),
                },
                how_to_fix=violation.get("description", ""),
            ))
    return findings


def findings_from_incomplete(results: dict[str, Any], url: str,
                             viewport: str) -> list[Finding]:
    """Convert axe ``incomplete`` results into explicit review items.

    These are the cases axe detected but could not decide — most commonly text
    over a gradient or an image, where ``html-core.md`` §8 warns "the DevTools
    number is wrong there". Silently passing them is how an automated report
    overstates its coverage.
    """
    findings: list[Finding] = []
    for item in results.get("incomplete", []):
        rule_id = item["id"]
        findings.append(Finding(
            id=f"a11y.axe.incomplete.{rule_id}",
            severity="advisory", rule=f"{rule_id} (needs review)",
            wcag_sc=wcag_sc_from_tags(item.get("tags", [])),
            url=url, viewport=viewport, selector=None, section="page",
            message=(f"axe could not decide `{rule_id}` for "
                     f"{len(item.get('nodes', []))} element(s); a human must check "
                     f"them."),
            evidence={"nodeCount": len(item.get("nodes", [])),
                      "examples": [" ".join(_flatten(n.get("target", [])))
                                   for n in item.get("nodes", [])[:5]],
                      "helpUrl": item.get("helpUrl")},
            how_to_fix=item.get("description", ""),
        ))
    return findings


def _flatten(target: Any) -> list[str]:
    """axe targets can nest for shadow DOM; flatten to a list of strings."""
    if isinstance(target, str):
        return [target]
    result: list[str] = []
    for item in target or []:
        result.extend(_flatten(item))
    return result


def _section_hint(selector: str | None) -> str | None:
    """Best-effort section label from an axe CSS selector."""
    if not selector:
        return None
    for landmark in ("header", "nav", "main", "footer", "aside", "form",
                     "section", "article"):
        if re.search(rf"\b{landmark}\b", selector):
            return landmark
    return None


# --------------------------------------------------------------------------- #
# Non-axe sweeps
# --------------------------------------------------------------------------- #

FILENAME_ALT = re.compile(r"^[\w\-. ]+\.(png|jpe?g|gif|svg|webp|avif)$", re.IGNORECASE)
IMAGE_OF_PREFIX = re.compile(r"^\s*(image|picture|photo|graphic|icon)\s+of\b", re.IGNORECASE)


def check_images(report: Report, page: Any, url: str, viewport: str) -> None:
    """ada/html-core.md §5 — what ``--images`` is specified to report."""
    images = page.evaluate(IMAGES_JS)
    report.meta["images_examined"] = len(images)
    named = 0

    for index, image in enumerate(images):
        section = image["section"] or "page"
        name = image["accessibleName"]

        # "alt="" and a missing alt are completely different. Missing alt causes
        # screen readers to announce the filename." (html-core.md §5)
        if image["tag"] == "img" and not image["hasAltAttribute"] \
                and not image["decorativeMarkup"] and name is None:
            report.add(Finding(
                id=f"a11y.images.missing-alt.{index}",
                severity="blocker", rule="image-missing-alt", wcag_sc="1.1.1 (A)",
                url=url, viewport=viewport, selector=image["selector"], section=section,
                message="Image has no alt attribute at all, so screen readers announce "
                        "the filename.",
                evidence={"src": image["src"], "size": image["size"],
                          "insideLinkOrButton": image["insideLinkOrButton"]},
                how_to_fix='Add alt text describing the information or function. If the '
                           'image is decorative, use alt="" — explicitly empty, not '
                           'missing.',
            ))
            continue

        if name is None:
            continue
        named += 1

        if FILENAME_ALT.match(name):
            report.add(Finding(
                id=f"a11y.images.filename-alt.{index}",
                severity="major", rule="image-filename-alt", wcag_sc="1.1.1 (A)",
                url=url, viewport=viewport, selector=image["selector"], section=section,
                message=f"Alt text is a filename: {name!r}.",
                evidence={"accessibleName": name, "nameSource": image["nameSource"]},
                how_to_fix="Replace with a description of the information or function "
                           "the image conveys.",
            ))
        elif IMAGE_OF_PREFIX.match(name):
            report.add(Finding(
                id=f"a11y.images.image-of-prefix.{index}",
                severity="minor", rule="image-alt-redundant-prefix", wcag_sc="1.1.1 (A)",
                url=url, viewport=viewport, selector=image["selector"], section=section,
                message=f"Alt text starts with a redundant role announcement: {name!r}.",
                evidence={"accessibleName": name},
                how_to_fix='Drop the "Image of" / "Graphic of" prefix — the role is '
                           'already announced.',
            ))

        if image["insideLinkOrButton"] and not name.strip():
            report.add(Finding(
                id=f"a11y.images.empty-alt-in-link.{index}",
                severity="blocker", rule="image-functional-empty-alt",
                wcag_sc="1.1.1 (A), 2.4.4 (A)", url=url, viewport=viewport,
                selector=image["selector"], section=section,
                message="Image is the content of a link or button but has empty alt, "
                        "leaving the control with no accessible name.",
                evidence={"src": image["src"]},
                how_to_fix="Set alt to the action or destination, not the picture.",
            ))

    report.note(
        f"{named} image(s) have a non-empty accessible name. Automation can tell you "
        "an alt exists; it cannot tell you the alt is *right*. Read every non-empty "
        "alt against the decision tree in references/ada/html-core.md §5. Charts, "
        "infographics and images containing text need adjacent text or a data table — "
        'alt="Bar chart of revenue" fails 1.1.1.'
    )


def check_forms(report: Report, page: Any, url: str, viewport: str) -> None:
    """ada/html-forms.md — the label / autocomplete / required sweep."""
    issue_meta = {
        "NO_ACCESSIBLE_NAME": ("blocker", "1.3.1 (A), 4.1.2 (A)",
                               "Control has no accessible name.",
                               "Give it a real <label for>, or aria-label / "
                               "aria-labelledby."),
        "PLACEHOLDER_AS_LABEL": ("major", "3.3.2 (A)",
                                 "Control is labelled only by its placeholder, which "
                                 "disappears on focus.",
                                 "A placeholder is not a label. Add a visible, "
                                 "associated <label>."),
        "MISSING_AUTOCOMPLETE": ("major", "1.3.5 Identify Input Purpose (AA)",
                                 "Personal-data field has no autocomplete token.",
                                 "Add the correct autocomplete token from the WCAG "
                                 "input-purpose list."),
        "AUTOCOMPLETE_OFF_ON_PERSONAL_FIELD": ("minor", "1.3.5 (AA)",
                                               "autocomplete=off on a personal-data "
                                               "field.",
                                               "Remove it; browsers largely ignore it "
                                               "and it is an accessibility negative."),
        "ARIA_REQUIRED_WITHOUT_REQUIRED": ("minor", "3.3.2 (A)",
                                           "aria-required without the native required "
                                           "attribute.",
                                           "Use the native `required` attribute; "
                                           "aria-required is a fallback, not a "
                                           "replacement."),
        "TYPE_NUMBER_MISUSE": ("minor", None,
                               "type=number used for a phone, OTP, postal code or "
                               "card field.",
                               'Use type="text" inputmode="numeric"; type=number '
                               "strips leading zeros and mutates on scroll."),
    }
    controls = page.evaluate(FORMS_JS)
    report.meta["form_controls_with_issues"] = len(controls)
    for index, control in enumerate(controls):
        for issue in control["issues"]:
            severity, sc, message, fix = issue_meta[issue]
            report.add(Finding(
                id=f"a11y.forms.{issue.lower()}.{index}",
                severity=severity, rule=f"form-{issue.lower().replace('_', '-')}",
                wcag_sc=sc, url=url, viewport=viewport, selector=control["selector"],
                section=control["section"] or "page",
                message=f"{message} (field: {control['name'] or control['type']})",
                evidence={"type": control["type"],
                          "autocomplete": control["autocomplete"],
                          "required": control["required"]},
                how_to_fix=fix,
            ))
    report.note(
        "Form automation covers labels, autocomplete and required only. Error "
        "identification, association and suggestion (3.3.1, 3.3.3, 4.1.3) need the "
        "manual pass: submit the form empty, then with one bad field, and check the "
        "error is in text, associated with the field, announced, and actually helpful."
    )


def summarise_contrast(report: Report, results: dict[str, Any]) -> None:
    """Record the contrast thresholds and the undecidable cases (html-core.md §8)."""
    incomplete = [i for i in results.get("incomplete", [])
                  if i["id"].startswith("color-contrast")]
    undecidable = sum(len(i.get("nodes", [])) for i in incomplete)
    report.meta["contrast_thresholds"] = (
        f"{CONTRAST_NORMAL_TEXT}:1 normal text, {CONTRAST_LARGE_TEXT}:1 large text "
        f"(>=18pt/24px, or >=14pt/18.66px bold), {CONTRAST_NON_TEXT}:1 UI boundaries "
        f"and meaningful graphics"
    )
    if undecidable:
        report.note(
            f"axe could not compute a contrast ratio for {undecidable} element(s) — "
            "typically text over a gradient, image or video. Test against the actual "
            "rendered background; the computed background-color of a parent is not "
            "necessarily what is behind the glyphs."
        )
    report.note(
        "Contrast automation cannot evaluate 1.4.1 Use of Color: links in body text "
        "must be underlined, or carry >=3:1 contrast against the surrounding text plus "
        "a non-colour cue on hover and focus."
    )


# --------------------------------------------------------------------------- #
# Document inventory
# --------------------------------------------------------------------------- #


def inventory_documents(report: Report, browser: Any, start_url: str,
                        max_pages: int, timeout: float) -> None:
    """Crawl same-origin pages and inventory linked documents.

    ``ada/00-map.md``: the document references are *not* loaded for a site that
    serves none of those formats — "Confirm by crawl, not by assumption".
    """
    origin = urlparse(start_url)
    seen: set[str] = set()
    queue: deque[str] = deque([start_url])
    documents: dict[str, list[str]] = {}
    pages_visited = 0

    context = browser.new_context()
    page = context.new_page()
    try:
        while queue and pages_visited < max_pages:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            try:
                page.goto(current, wait_until="domcontentloaded", timeout=timeout * 1000)
            except Exception:  # noqa: BLE001 - an unreachable page is not fatal here
                continue
            pages_visited += 1

            for href in page.eval_on_selector_all(
                "a[href]", "els => els.map(e => e.href)"
            ):
                clean, _ = urldefrag(href)
                parsed = urlparse(clean)
                if parsed.scheme not in ("http", "https"):
                    continue
                extension = Path(parsed.path).suffix.lower()
                if extension in DOCUMENT_TYPES:
                    documents.setdefault(extension, [])
                    if clean not in documents[extension]:
                        documents[extension].append(clean)
                elif parsed.netloc == origin.netloc and clean not in seen:
                    queue.append(clean)
    finally:
        context.close()

    report.meta["pages_crawled_for_documents"] = pages_visited
    if not documents:
        report.note(
            f"No linked PDF or Office documents found across {pages_visited} page(s). "
            "The document-accessibility references need not be loaded; record "
            "content.has_pdfs / has_office_docs / has_latex_pdfs as false."
        )
        return

    references_needed = sorted({DOCUMENT_TYPES[ext][1] for ext in documents})
    profile_keys = sorted({DOCUMENT_TYPES[ext][0] for ext in documents})
    total = sum(len(urls) for urls in documents.values())

    report.add(Finding(
        id="a11y.documents.inventory",
        severity="major", rule="document-inventory",
        wcag_sc="1.1.1, 1.3.1, 2.4.2 (documents are in scope as content)",
        url=start_url, viewport=None, selector=None, section="site",
        message=(f"{total} linked document(s) across {len(documents)} format(s) are in "
                 f"scope and were not tested by this script."),
        evidence={
            "byExtension": {ext: len(urls) for ext, urls in sorted(documents.items())},
            "examples": {ext: urls[:5] for ext, urls in sorted(documents.items())},
            "profileKeysToSet": profile_keys,
            "referencesToLoad": [DOCUMENT_PREREQUISITE] + [
                r for r in references_needed if r != DOCUMENT_PREREQUISITE
            ],
            "pagesCrawled": pages_visited,
        },
        how_to_fix=(
            "Load " + ", ".join([DOCUMENT_PREREQUISITE]
                                + [r for r in references_needed
                                   if r != DOCUMENT_PREREQUISITE])
            + " (documents-pdf.md carries the WCAG2ICT rules the other two depend on) "
              "and run the document pass: PAC 2024 + veraPDF + the Acrobat checker for "
              "PDF, the built-in Accessibility Checker plus a manual reading-order "
              "review for Office. Use two validators plus a manual screen-reader pass — "
              "one published comparison of 155 files found four leading validators "
              "disagreeing in over half of cases."
        ),
    ))


# --------------------------------------------------------------------------- #
# Coverage reporting
# --------------------------------------------------------------------------- #


def report_coverage(report: Report, results: dict[str, Any], standard: str) -> None:
    """Add the counts that stop a clean scan implying a clean site."""
    violations = results.get("violations", [])
    incomplete = results.get("incomplete", [])
    passes = results.get("passes", [])
    inapplicable = results.get("inapplicable", [])

    not_testable = len(MANUAL_ONLY_CRITERIA)
    report.meta["axe_rules_run"] = len(violations) + len(passes) + len(incomplete)
    report.meta["axe_rules_inapplicable"] = len(inapplicable)
    report.meta["axe_needs_review"] = sum(len(i.get("nodes", [])) for i in incomplete)
    report.meta["criteria_automation_cannot_test"] = not_testable
    report.meta["standard"] = standard

    report.note(
        "Coverage: automated tools detect roughly 30-40% of WCAG failures measured by "
        "criterion. Deque reports axe-core catches around 57% of issues by volume — a "
        "different and more flattering denominator that counts repeat instances of the "
        "same detectable rule. This run used the by-criterion framing. Automated "
        "testing is a regression net, not an audit."
    )
    report.note(
        f"{not_testable} success criteria in scope cannot be evaluated by any scanner "
        "and do not appear in the findings above: "
        + "; ".join(f"{number} {title}" for number, title in MANUAL_ONLY_CRITERIA)
        + ". If every finding in a report maps to an axe rule id, only Layer 1 was run."
    )
    if standard == "wcag22aa":
        report.note(
            "Of the six criteria new in WCAG 2.2 ("
            + ", ".join(WCAG22_NEW_CRITERIA)
            + "), only 2.5.8 Target Size is reliably automatable. An automated scan "
              "reporting 'no WCAG 2.2 issues' has checked one of six."
        )
    report.note(
        f"{report.meta['axe_needs_review']} element(s) landed in axe's `incomplete` "
        "bucket: detected, but undecidable without a human. They are reported as "
        "advisory findings, not as passes."
    )
    report.note(
        "Engines disagree. Run a second engine (IBM Equal Access or WAVE) alongside "
        "axe; a violation only one reports is worth a manual look before publishing it."
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def build_parser():
    """Construct the argument parser."""
    parser = base_parser(
        "audit_a11y.py",
        "Run axe-core through Playwright and report what it could not test.",
        epilog=(
            "With no mode flag the full axe scan runs. Mode flags add the sweeps the\n"
            "ADA references name explicitly, and can be combined.\n\n"
            "NOTE: this script does not read the audit profile, so severity is\n"
            "assigned without profile context. Re-check every severity against\n"
            "references/reporting.md §2 — WCAG A/AA failures escalate to blocker\n"
            "only when compliance.regime != none.\n\n"
            "Sources: references/ada/testing.md, html-core.md, html-forms.md,\n"
            "wcag22-new.md, 00-map.md."
        ),
    )
    parser.add_argument(
        "--standard", default="wcag22aa",
        choices=sorted(set(STANDARD_TAGS) | set(STANDARD_ALIASES)),
        help="WCAG level/version to test against: wcag20aa (Section 508 / ACAA), "
             "wcag21aa (ADA Title II, HHS 504), or wcag22aa (default). The "
             "hyphenated profile spellings wcag20-aa / wcag21-aa / wcag22-aa "
             "are accepted too",
    )
    parser.add_argument("--images", action="store_true",
                        help="Image accessible-name sweep (ada/html-core.md §5)")
    parser.add_argument("--contrast", action="store_true",
                        help="Contrast sweep, including the undecidable cases")
    parser.add_argument("--forms", action="store_true",
                        help="Label / autocomplete / required sweep")
    parser.add_argument("--target-size", action="store_true", dest="target_size",
                        help="WCAG 2.2 SC 2.5.8 target-size check")
    parser.add_argument("--inventory-documents", action="store_true",
                        dest="inventory_documents",
                        help="Crawl for linked .pdf/.docx/.pptx/.xlsx and report which "
                             "document-accessibility references need loading")
    parser.add_argument("--all", action="store_true", dest="run_all",
                        help="Run every mode above")
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES,
                        help=f"Crawl limit for --inventory-documents "
                             f"(default: {DEFAULT_MAX_PAGES})")
    parser.add_argument("--axe-source", metavar="PATH",
                        help="Path to a local axe.min.js instead of the CDN build")
    parser.add_argument("--axe-version", default=DEFAULT_AXE_VERSION,
                        help=f"axe-core version to fetch (default: {DEFAULT_AXE_VERSION})")
    parser.add_argument("--viewport", default="1280x1024", metavar="WxH",
                        help="Viewport for the scan (default: 1280x1024)")
    return parser


def main() -> int:
    """Run the accessibility audit and return the process exit code."""
    args = build_parser().parse_args()
    url = normalise_url(args.url)
    standard, tags = resolve_standard(args.standard)

    width, _, height = args.viewport.partition("x")
    viewport_label = f"{width}x{height}"

    want_images = args.images or args.run_all
    want_contrast = args.contrast or args.run_all
    want_forms = args.forms or args.run_all
    want_target_size = args.target_size or args.run_all
    want_documents = args.inventory_documents or args.run_all

    report = Report(tool="audit_a11y", target=url)
    report.meta["axe_tags"] = ", ".join(tags)

    axe_source = load_axe_source(args.axe_source, args.axe_version, args.timeout)
    sync_playwright = import_playwright()

    with sync_playwright() as playwright:
        browser = launch_chromium(playwright)
        try:
            context = browser.new_context(
                viewport={"width": int(width), "height": int(height or 1024)}
            )
            page = context.new_page()
            try:
                try:
                    page.goto(url, wait_until="load", timeout=args.timeout * 1000)
                except Exception as exc:  # noqa: BLE001
                    raise TargetUnreachable(url, str(exc).splitlines()[0]) from exc
                page.wait_for_timeout(500)  # let late-rendered UI settle
                page.add_script_tag(content=axe_source)

                # target-size is a WCAG 2.2 rule; ask for it explicitly so the
                # check also works when --standard is wcag21aa.
                run_tags = list(tags)
                if want_target_size and "wcag22aa" not in run_tags:
                    run_tags.append("wcag22aa")

                # No `resultTypes` filter: the passes and inapplicable arrays are
                # what make the coverage counts in report_coverage() honest.
                results = page.evaluate(
                    """async (tags) => await axe.run(document, {
                         runOnly: { type: 'tag', values: tags }
                       })""",
                    run_tags,
                )
                report.meta["axe_version"] = results.get("testEngine", {}).get(
                    "version", args.axe_version)

                report.extend(findings_from_axe(results, url, viewport_label))
                report.extend(findings_from_incomplete(results, url, viewport_label))

                if want_contrast:
                    summarise_contrast(report, results)
                if want_target_size:
                    ran = any(r["id"] == "target-size"
                              for group in ("violations", "passes", "incomplete",
                                            "inapplicable")
                              for r in results.get(group, []))
                    report.meta["target_size_rule_ran"] = ran
                    report.note(
                        "Target size checked against WCAG 2.2 SC 2.5.8: 24x24 CSS px, "
                        "or spaced so a 24 px-diameter circle centred on the target "
                        "does not intersect another target's circle. Exceptions for "
                        "inline links in text, browser-default controls and essential "
                        "sizing are applied by axe; 44x44 is SC 2.5.5 Level AAA and is "
                        "not a conformance failure."
                    )
                if want_images:
                    check_images(report, page, url, viewport_label)
                if want_forms:
                    check_forms(report, page, url, viewport_label)

                report_coverage(report, results, standard)
            finally:
                context.close()

            if want_documents:
                inventory_documents(report, browser, url, args.max_pages, args.timeout)
        finally:
            browser.close()

    report.note(
        "This run scanned one page. State the sample in the written report — 30-50 "
        "pages is a typical audit sample for a mid-size site, and a finding count "
        "without a denominator is meaningless."
    )
    return finish(report, args)


if __name__ == "__main__":
    run_cli(main)
