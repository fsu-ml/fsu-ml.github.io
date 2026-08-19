#!/usr/bin/env python3
"""Measure load performance against a budget, and fail the build when it breaks.

Playwright drives a throttled Chromium; a ``PerformanceObserver`` installed
before navigation collects the metrics; CDP supplies byte-accurate transfer
sizes and the Coverage API. Lighthouse is optional and additive.

Everything here follows ``references/performance.md``:

* §1.1 thresholds — LCP 2.5 s, CLS 0.1, INP 200 ms, FCP 1.8 s, TTFB 0.8 s,
  TBT 200 ms — and the doc's own lab/field labelling, which this script
  reproduces rather than flattens. **INP is field-only**; what a lab run can
  give you is TBT, and this script reports it as a proxy and says so.
* §4.1 JS budgets, §6 the budget table.
* §4.4 third-party origin inventory with byte cost.
* §4.5 Coverage — "any file > 60% unused is a pruning candidate", carrying the
  doc's caveat that coverage identifies candidates and does not prove
  deletability.
* §3.1 render-blocking resources.

The reference budget in §6 is a fill-in Markdown table, not machine-readable.
``--init-budget`` writes the JSON equivalent, with the two rows the doc
deliberately leaves blank ("set explicitly": total image weight and total page
weight) left as ``null`` for the operator to fill in.

Usage:
    ./audit_performance.py https://example.com
    ./audit_performance.py https://example.com --budget budget.json --json out/perf.json
    ./audit_performance.py https://example.com --lighthouse --desktop
    ./audit_performance.py --init-budget budget.json
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from _common import (
    EXIT_OK,
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
# Thresholds — every number cited to references/performance.md unless noted
# --------------------------------------------------------------------------- #

# §1.1 "Thresholds (unchanged, authoritative)". Type column preserved: LCP, CLS,
# FCP and TTFB are field+lab; INP is field only; TBT is lab only.
CWV_GOOD = {
    "lcp_ms": 2500,
    "cls": 0.1,
    "inp_ms": 200,
    "fcp_ms": 1800,
    "ttfb_ms": 800,
    "tbt_ms": 200,
}

# mobile.md §3.1 — the Lighthouse mobile "Slow 4G" preset: 150 ms RTT,
# 1.6 Mbps down, 750 Kbps up, no packet loss. CDP wants bytes per second.
SLOW_4G = {
    "rtt_ms": 150,
    "download_bytes_per_s": 1_600_000 // 8,   # 1.6 Mbps
    "upload_bytes_per_s": 750_000 // 8,       # 750 Kbps
}
CPU_THROTTLE_MOBILE = 4      # constant 4x slowdown multiplier
MOBILE_VIEWPORT = (412, 915)  # large-Android class, matches mobile.md V4
DESKTOP_VIEWPORT = (1350, 940)

# §4.5 / §6 row 25 — "Any file > 60% unused is a pruning candidate".
MAX_UNUSED_PCT_PER_FILE = 60

#: Default budget. ``None`` means "the reference deliberately declines to fix a
#: number" (§6 rows 12-14 say "set explicitly"); those are reported as
#: measurements only until the operator supplies a value.
DEFAULT_BUDGET: dict[str, dict[str, Any]] = {
    "timings": {
        "lcp_ms": CWV_GOOD["lcp_ms"],      # §1.1
        "cls": CWV_GOOD["cls"],            # §1.1
        "fcp_ms": CWV_GOOD["fcp_ms"],      # §1.1 diagnostic
        "ttfb_ms": CWV_GOOD["ttfb_ms"],    # §1.1 diagnostic, §6 row 7
        "tbt_ms": CWV_GOOD["tbt_ms"],      # §6 row 8, lab
    },
    "resource_bytes": {
        "script": 200_000,      # §4.1 "< 200 KB compressed initial JS"; 50 KB content sites
        "stylesheet": 150_000,  # §6 row 11 (mobile.md quotes a stricter 50 KB heuristic)
        "font": None,           # §6 row 12 — payload budget not fixed by the doc
        "image": None,          # §6 row 13 — "set explicitly" (median web: 1,059 KB)
        "media": None,
        "document": None,
        "total": None,          # §6 row 14 — "set explicitly" (median web: 2.56 MB mobile)
    },
    "counts": {
        "third_party_origins": 9,          # §6 row 19 "<= 9 (median)"
        "render_blocking_scripts": 0,      # §6 row 16 "0 blocking JS in <head>"
        "render_blocking_stylesheets": 1,  # §6 row 16 "<= 1 blocking CSS"
        "render_blocking_third_parties": 0,  # §6 row 20
        "preload": 4,                      # §3.2 "cap at 2-4"
        "preconnect": 4,                   # §3.2 "cap at 2-4"
        "dom_elements": 1500,              # §6 row 15 "< 1,500"
        "console_errors": 0,               # §6 row 23
        "failed_requests": 0,              # §6 row 24
    },
    "coverage": {
        "max_unused_pct_per_file": MAX_UNUSED_PCT_PER_FILE,  # §4.5
    },
}

#: §5.1 — "Lighthouse 13 removed or renamed most performance audit IDs.
#: Hard-code the ``*-insight`` IDs."
LIGHTHOUSE_INSIGHT_IDS = (
    "lcp-phases-insight",
    "render-blocking-insight",
    "dom-size-insight",
    "third-parties-insight",
    "duplicated-javascript-insight",
    "legacy-javascript-insight",
    "modern-http-insight",
    "document-latency-insight",
    "non-composited-animations",
)


# --------------------------------------------------------------------------- #
# In-page instrumentation
# --------------------------------------------------------------------------- #

# Installed before navigation so nothing is missed. CLS uses the session-window
# algorithm from performance.md §2.3: max 5 s gap, 5 s cap, hadRecentInput
# excluded. TBT is derived from long tasks after FCP (performance.md §1.1 —
# TBT is the lab proxy for INP; Lighthouse cannot measure INP and neither can
# an uninteracted Playwright run).
METRICS_INIT_JS = r"""
window.__audit = {
  lcp: null, lcpElement: null,
  fcp: null,
  cls: 0, clsSources: [],
  longTasks: [],
  interactions: [],
  supported: {}
};

const A = window.__audit;
const safeObserve = (type, cb, extra) => {
  try {
    if (!PerformanceObserver.supportedEntryTypes?.includes(type)) {
      A.supported[type] = false; return;
    }
    new PerformanceObserver(cb).observe(Object.assign({ type, buffered: true }, extra || {}));
    A.supported[type] = true;
  } catch (e) { A.supported[type] = false; }
};

safeObserve('largest-contentful-paint', list => {
  for (const e of list.getEntries()) {
    A.lcp = e.startTime;
    A.lcpElement = e.element
      ? e.element.tagName.toLowerCase() +
        (e.element.id ? '#' + e.element.id : '') +
        (e.url ? ' [' + e.url.slice(-70) + ']' : '')
      : (e.url || null);
  }
});

safeObserve('paint', list => {
  for (const e of list.getEntries())
    if (e.name === 'first-contentful-paint') A.fcp = e.startTime;
});

// CLS session window: largest burst, 5 s gap / 5 s cap.
let sessionValue = 0, sessionEntries = [];
safeObserve('layout-shift', list => {
  for (const e of list.getEntries()) {
    if (e.hadRecentInput) continue;                       // user-initiated: excluded
    const first = sessionEntries[0], last = sessionEntries[sessionEntries.length - 1];
    if (sessionValue && e.startTime - last.startTime < 1000 &&
        e.startTime - first.startTime < 5000) {
      sessionValue += e.value; sessionEntries.push(e);
    } else {
      sessionValue = e.value; sessionEntries = [e];
    }
    if (sessionValue > A.cls) {
      A.cls = sessionValue;
      A.clsSources = sessionEntries.flatMap(x => (x.sources || []).map(s =>
        s.node ? s.node.nodeName.toLowerCase() +
                 (s.node.id ? '#' + s.node.id : '') : 'unknown'));
    }
  }
});

safeObserve('longtask', list => {
  for (const e of list.getEntries())
    A.longTasks.push({ start: e.startTime, duration: e.duration,
                       attribution: (e.attribution || [])
                         .map(a => a.name || a.containerName || '').filter(Boolean) });
});

// Present only if the harness or a real user interacts; reported as a proxy,
// never as INP, which is field-only.
safeObserve('event', list => {
  for (const e of list.getEntries()) {
    if (!e.interactionId) continue;
    A.interactions.push({
      type: e.name, duration: e.duration,
      inputDelay: e.processingStart - e.startTime,
      processing: e.processingEnd - e.processingStart,
      presentationDelay: e.startTime + e.duration - e.processingEnd
    });
  }
}, { durationThreshold: 16 });
"""

# performance.md §3.1 — what counts as render-blocking, read from the rendered DOM.
RENDER_BLOCKING_JS = r"""() => {
  const head = document.head;
  const scripts = [...head.querySelectorAll('script[src]')]
    .filter(s => !s.defer && !s.async && s.type !== 'module')
    .map(s => ({ url: s.src, type: 'script' }));
  const styles = [...document.querySelectorAll('link[rel~=stylesheet]')]
    .filter(l => !l.disabled && !/print/i.test(l.media || ''))
    .map(l => ({ url: l.href, type: 'stylesheet', media: l.media || 'all' }));
  return {
    scripts, styles,
    preload: document.querySelectorAll('link[rel=preload]').length,
    preconnect: document.querySelectorAll('link[rel=preconnect]').length,
    domElements: document.getElementsByTagName('*').length,
    nextHopProtocol:
      performance.getEntriesByType('navigation')[0]?.nextHopProtocol || null
  };
}"""

COLLECT_METRICS_JS = r"""() => {
  const A = window.__audit;
  const nav = performance.getEntriesByType('navigation')[0] || {};
  // performance.md §1.1: TBT is the lab proxy for INP. Long tasks after FCP,
  // each contributing (duration - 50) ms of blocking time.
  const fcp = A.fcp || 0;
  const tbt = A.longTasks
    .filter(t => t.start + t.duration > fcp)
    .reduce((sum, t) => sum + Math.max(0, t.duration - 50), 0);
  return {
    lcp_ms: A.lcp, lcpElement: A.lcpElement,
    fcp_ms: A.fcp,
    cls: +A.cls.toFixed(4), clsSources: [...new Set(A.clsSources)].slice(0, 10),
    ttfb_ms: nav.responseStart ? +nav.responseStart.toFixed(1) : null,
    tbt_ms: +tbt.toFixed(1),
    longTaskCount: A.longTasks.length,
    longestTask_ms: A.longTasks.reduce((m, t) => Math.max(m, t.duration), 0),
    interactions: A.interactions,
    observerSupport: A.supported
  };
}"""


# --------------------------------------------------------------------------- #
# Budget handling
# --------------------------------------------------------------------------- #


def load_budget(path: str | None) -> dict[str, dict[str, Any]]:
    """Load a budget file, filling any absent key from :data:`DEFAULT_BUDGET`."""
    budget = {section: dict(values) for section, values in DEFAULT_BUDGET.items()}
    if not path:
        return budget
    file = Path(path)
    if not file.is_file():
        raise MissingDependency(
            f"budget file {path}",
            f"Create one with:  ./audit_performance.py --init-budget {path}",
        )
    try:
        supplied = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MissingDependency(f"a valid JSON budget at {path} ({exc})",
                                "Fix the JSON syntax, or regenerate with --init-budget")
    for section, values in supplied.items():
        if section in budget and isinstance(values, dict):
            budget[section].update(values)
    return budget


def write_budget_template(path: str) -> None:
    """Write the JSON equivalent of the performance.md §6 budget table."""
    document = {
        "_source": "references/performance.md §6 — agree these with the client "
                   "before remediation and enforce them in CI. Anything without a "
                   "number is not a budget.",
        "_device_profile": "mid-tier Android (Moto G-class / Snapdragon 695-class), "
                           "Slow 4G (150 ms RTT / 1.6 Mbps down / 750 Kbps up), "
                           "4x CPU throttle",
        "_null_means": "the reference deliberately declines to fix a number "
                       "('set explicitly'); supply your own or the check is skipped",
        "_median_web_anchors_kb": {"script": 697, "image": 1059, "font": 139,
                                   "stylesheet": 82, "document": 22,
                                   "total_mobile": 2560},
    }
    document.update(DEFAULT_BUDGET)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def compare(report: Report, url: str, key: str, measured: float | None,
            limit: float | None, unit: str, rule: str, section: str,
            how_to_fix: str, severity: str = "major") -> None:
    """Record a finding when *measured* exceeds *limit*.

    A ``None`` limit means the budget deliberately has no number for this row;
    the measurement is still surfaced in ``report.meta`` by the caller.
    """
    if measured is None or limit is None or measured <= limit:
        return
    report.add(Finding(
        id=f"performance.budget.{key}",
        severity=severity, rule=rule, wcag_sc=None, url=url,
        viewport=report.meta.get("emulation"), selector=None, section=section,
        message=f"{key} is {measured:g} {unit}, over the budget of {limit:g} {unit}.",
        evidence={"measured": measured, "budget": limit, "unit": unit,
                  "over_by": round(measured - limit, 3)},
        how_to_fix=how_to_fix,
    ))


# --------------------------------------------------------------------------- #
# Network and coverage collection via CDP
# --------------------------------------------------------------------------- #


class NetworkRecorder:
    """Collect per-request transfer sizes and resource types from CDP events.

    Resource Timing's ``transferSize`` is zero for cross-origin responses
    without ``Timing-Allow-Origin``, which would silently under-count exactly
    the third-party bytes §4.4 cares about. CDP's ``encodedDataLength`` does
    not have that blind spot.
    """

    def __init__(self) -> None:
        self.by_request: dict[str, dict[str, Any]] = {}
        self.failed: list[str] = []

    def attach(self, session: Any) -> None:
        """Subscribe to the CDP network events."""
        session.on("Network.responseReceived", self._on_response)
        session.on("Network.loadingFinished", self._on_finished)
        session.on("Network.loadingFailed", self._on_failed)

    def _on_response(self, event: dict[str, Any]) -> None:
        response = event.get("response", {})
        self.by_request[event["requestId"]] = {
            "url": response.get("url", ""),
            "type": (event.get("type") or "Other").lower(),
            "status": response.get("status", 0),
            "mime": response.get("mimeType", ""),
            "protocol": response.get("protocol"),
            "encoding": (response.get("headers", {}) or {}).get("content-encoding"),
            "bytes": 0,
        }

    def _on_finished(self, event: dict[str, Any]) -> None:
        record = self.by_request.get(event["requestId"])
        if record is not None:
            record["bytes"] = int(event.get("encodedDataLength", 0))

    def _on_failed(self, event: dict[str, Any]) -> None:
        record = self.by_request.get(event["requestId"])
        if record is not None:
            self.failed.append(record["url"])

    def resources(self) -> list[dict[str, Any]]:
        """All recorded responses."""
        return list(self.by_request.values())


def collect_coverage(session: Any, stylesheet_urls: dict[str, str]) -> dict[str, Any]:
    """Stop the CDP coverage trackers and summarise unused bytes per file.

    Args:
        session: The active CDP session.
        stylesheet_urls: ``styleSheetId -> source URL``, gathered from
            ``CSS.styleSheetAdded`` events while the page loaded.

    Returns:
        ``{"js": [...], "css": [...], "available": bool}``. ``available`` is
        False when the CDP domains refused, so the caller can say so rather
        than silently reporting zero unused bytes.
    """
    result: dict[str, Any] = {"js": [], "css": [], "available": True}
    try:
        js_coverage = session.send("Profiler.takePreciseCoverage")
        session.send("Profiler.stopPreciseCoverage")
        for script in js_coverage.get("result", []):
            url = script.get("url") or ""
            if not url.startswith("http"):
                continue
            # Block coverage nests ranges, so covered bytes must be the *union*
            # of executed offsets, not their sum.
            covered: list[tuple[int, int]] = []
            total = 0
            for function in script.get("functions", []):
                for entry in function.get("ranges", []):
                    total = max(total, entry["endOffset"])
                    if entry["count"] > 0:
                        covered.append((entry["startOffset"], entry["endOffset"]))
            if not total:
                continue
            used = _merged_length(covered)
            result["js"].append({"url": url, "total": total, "used": used,
                                 "unused_pct": round(100 * (1 - used / total), 1)})

        css_coverage = session.send("CSS.stopRuleUsageTracking")
        sheets: dict[str, dict[str, Any]] = {}
        for usage in css_coverage.get("ruleUsage", []):
            sheet_id = usage["styleSheetId"]
            sheet = sheets.setdefault(sheet_id, {"used": 0, "total": 0})
            length = usage["endOffset"] - usage["startOffset"]
            sheet["total"] += length
            if usage.get("used"):
                sheet["used"] += length
        for sheet_id, sheet in sheets.items():
            if not sheet["total"]:
                continue
            result["css"].append({
                "url": stylesheet_urls.get(sheet_id) or "(inline stylesheet)",
                "total": sheet["total"], "used": sheet["used"],
                "unused_pct": round(100 * (1 - sheet["used"] / sheet["total"]), 1),
            })
    except Exception:  # noqa: BLE001 - coverage is a bonus, never a hard failure
        result["available"] = False
    return result


def _merged_length(ranges: list[tuple[int, int]]) -> int:
    """Total length covered by *ranges*, merging overlaps."""
    if not ranges:
        return 0
    ordered = sorted(ranges)
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    return total + (current_end - current_start)


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def check_timings(report: Report, url: str, metrics: dict[str, Any],
                  budget: dict[str, dict[str, Any]]) -> None:
    """Compare the collected metrics with the timing budget."""
    timings = budget["timings"]
    compare(report, url, "lcp_ms", metrics["lcp_ms"], timings.get("lcp_ms"), "ms",
            "core-web-vital", "loading",
            "Break LCP into TTFB, resource load delay, load duration and render delay; "
            "the usual win is making the LCP resource discoverable in the initial HTML "
            "with fetchpriority=high and no lazy loading.")
    compare(report, url, "cls", metrics["cls"], timings.get("cls"), "",
            "core-web-vital", "visual stability",
            "Reserve space for images, embeds and ad slots; measure with the consent "
            "banner present, not dismissed.")
    compare(report, url, "tbt_ms", metrics["tbt_ms"], timings.get("tbt_ms"), "ms",
            "lab-proxy", "responsiveness",
            "Break up long tasks and yield; TBT is the lab proxy for INP, so this is "
            "the number to move when field INP is poor.")
    # §1.1: "TTFB and FCP are not Core Web Vitals and are not ranking inputs ...
    # Do not report them as failures; report them as causes."
    compare(report, url, "ttfb_ms", metrics["ttfb_ms"], timings.get("ttfb_ms"), "ms",
            "diagnostic", "server response",
            "Diagnostic, not a Core Web Vital. Investigate origin response time, "
            "redirects and CDN coverage — only 33% of HTML documents are CDN-served.",
            severity="minor")
    compare(report, url, "fcp_ms", metrics["fcp_ms"], timings.get("fcp_ms"), "ms",
            "diagnostic", "loading",
            "Diagnostic, not a Core Web Vital. Usually downstream of render-blocking "
            "resources.", severity="minor")


def check_resources(report: Report, url: str, resources: list[dict[str, Any]],
                    budget: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Weight breakdown by type, plus the third-party origin inventory (§4.4)."""
    page_host = urlparse(url).hostname or ""
    by_type: dict[str, int] = {}
    by_origin: dict[str, dict[str, Any]] = {}
    for resource in resources:
        by_type[resource["type"]] = by_type.get(resource["type"], 0) + resource["bytes"]
        host = urlparse(resource["url"]).hostname or ""
        if host and host != page_host:
            origin = by_origin.setdefault(host, {"bytes": 0, "requests": 0,
                                                 "types": set()})
            origin["bytes"] += resource["bytes"]
            origin["requests"] += 1
            origin["types"].add(resource["type"])

    total_bytes = sum(by_type.values())
    limits = budget["resource_bytes"]
    for kind, measured in sorted(by_type.items()):
        compare(report, url, f"bytes.{kind}", measured, limits.get(kind), "bytes",
                "resource-weight", "page weight",
                f"Reduce {kind} transfer. Bytes are a proxy: the pass criterion is "
                f"TBT < 200 ms and field INP <= 200 ms, but byte budgets are what CI "
                f"can enforce.")
    compare(report, url, "bytes.total", total_bytes, limits.get("total"), "bytes",
            "resource-weight", "page weight",
            "Set and enforce a total transfer budget.")

    third_parties = sorted(by_origin.items(), key=lambda kv: -kv[1]["bytes"])
    origin_limit = budget["counts"].get("third_party_origins")
    if origin_limit is not None and len(third_parties) > origin_limit:
        report.add(Finding(
            id="performance.third-parties.origin-count",
            severity="major", rule="third-party-inventory", wcag_sc=None, url=url,
            viewport=report.meta.get("emulation"), selector=None,
            section="third parties",
            message=(f"{len(third_parties)} third-party origins are contacted, over the "
                     f"budget of {origin_limit}."),
            evidence={
                "originCount": len(third_parties), "budget": origin_limit,
                "topByBytes": [
                    {"origin": host, "bytes": data["bytes"],
                     "requests": data["requests"], "types": sorted(data["types"])}
                    for host, data in third_parties[:12]
                ],
                "totalThirdPartyBytes": sum(d["bytes"] for _, d in third_parties),
            },
            how_to_fix="For each origin record who owns it, what business value it "
                       "delivers, what it costs in main-thread ms and bytes, and when "
                       "it was last reviewed. Delete the ones nobody can name an owner "
                       "for — in practice 20-40% of tags on a mature site. Defer the "
                       "rest past load, behind interaction, or behind consent.",
        ))

    return {
        "by_type": by_type,
        "total_bytes": total_bytes,
        "third_party_origins": len(third_parties),
        "third_party_bytes": sum(d["bytes"] for _, d in third_parties),
        "third_parties": [
            {"origin": host, "bytes": data["bytes"], "requests": data["requests"]}
            for host, data in third_parties
        ],
    }


def check_render_blocking(report: Report, url: str, dom: dict[str, Any],
                          page_host: str, budget: dict[str, dict[str, Any]]) -> None:
    """performance.md §3.1 and §6 rows 16-18."""
    counts = budget["counts"]
    scripts = dom["scripts"]
    styles = dom["styles"]

    if counts.get("render_blocking_scripts") is not None and \
            len(scripts) > counts["render_blocking_scripts"]:
        report.add(Finding(
            id="performance.render-blocking.scripts",
            severity="major", rule="render-blocking", wcag_sc=None, url=url,
            viewport=report.meta.get("emulation"), selector="head > script[src]",
            section="critical path",
            message=f"{len(scripts)} parser-blocking script(s) in <head>.",
            evidence={"scripts": [s["url"] for s in scripts][:10],
                      "budget": counts["render_blocking_scripts"]},
            how_to_fix="Add `defer` for first-party app code (executes in document "
                       "order before DOMContentLoaded) or `async` for independent "
                       "third-party scripts. `type=module` is deferred by default.",
        ))

    if counts.get("render_blocking_stylesheets") is not None and \
            len(styles) > counts["render_blocking_stylesheets"]:
        report.add(Finding(
            id="performance.render-blocking.stylesheets",
            severity="minor", rule="render-blocking", wcag_sc=None, url=url,
            viewport=report.meta.get("emulation"), selector="link[rel=stylesheet]",
            section="critical path",
            message=f"{len(styles)} render-blocking stylesheet(s).",
            evidence={"stylesheets": [s["url"] for s in styles][:10],
                      "budget": counts["render_blocking_stylesheets"],
                      "context": "only 13-15% of pages pass Lighthouse's "
                                 "render-blocking audit, so this is common and not a "
                                 "differentiator on its own"},
            how_to_fix="Inline critical CSS for above-the-fold content and load the "
                       "rest with `media=\"print\" onload=\"this.media='all'\"` or "
                       "`rel=preload as=style` plus a swap.",
        ))

    third_party_blocking = [
        item["url"] for item in scripts + styles
        if (urlparse(item["url"]).hostname or page_host) != page_host
    ]
    if third_party_blocking and counts.get("render_blocking_third_parties") == 0:
        report.add(Finding(
            id="performance.render-blocking.third-parties",
            severity="major", rule="render-blocking-third-party", wcag_sc=None, url=url,
            viewport=report.meta.get("emulation"), selector=None,
            section="third parties",
            message=f"{len(third_party_blocking)} render-blocking third-party "
                    f"resource(s).",
            evidence={"resources": third_party_blocking[:10], "budget": 0},
            how_to_fix="Third-party scripts must be `async`, never synchronous, and "
                       "never sync-injected CSS. Run the SPOF test: does the page "
                       "still render and the core task still complete if the origin "
                       "never responds?",
        ))

    for key, measured in (("preload", dom["preload"]), ("preconnect", dom["preconnect"]),
                          ("dom_elements", dom["domElements"])):
        compare(report, url, key, measured, counts.get(key), "",
                "budget", "critical path",
                {"preload": "Preload only the 2-4 truly critical resources; pages with "
                            "10+ preloads consistently load slower, and DevTools warns "
                            "about any preload unused within 3 s.",
                 "preconnect": "Cap preconnect at 2-4; an unused preconnect holds a "
                               "connection for about 10 seconds.",
                 "dom_elements": "Reduce DOM size; large trees inflate style "
                                 "recalculation and layout on every frame."}[key],
                severity="minor")


def check_coverage(report: Report, url: str, coverage: dict[str, Any],
                   budget: dict[str, dict[str, Any]]) -> None:
    """performance.md §4.5 — files more than 60% unused."""
    if not coverage["available"]:
        report.note("The Coverage API was unavailable in this run; unused CSS/JS was "
                    "not measured.")
        return
    limit = budget["coverage"].get("max_unused_pct_per_file", MAX_UNUSED_PCT_PER_FILE)
    for kind in ("js", "css"):
        for entry in coverage[kind]:
            if entry["unused_pct"] <= limit:
                continue
            report.add(Finding(
                id=f"performance.coverage.{kind}.{entry['url'][-60:]}",
                severity="minor", rule="unused-code", wcag_sc=None, url=url,
                viewport=report.meta.get("emulation"),
                selector=entry["url"], section="page weight",
                message=(f"{entry['unused_pct']}% of this {kind.upper()} file is unused "
                         f"on first load ({entry['total'] - entry['used']} of "
                         f"{entry['total']} bytes)."),
                evidence={"unused_pct": entry["unused_pct"], "threshold": limit,
                          "totalBytes": entry["total"], "usedBytes": entry["used"]},
                how_to_fix="Split the bundle by route, or purge dead CSS. Coverage is "
                           "per-session: code used only on other routes shows as "
                           "unused, so this identifies candidates and does not prove "
                           "deletability.",
            ))
    report.note(
        "Coverage was measured on first load only. performance.md §4.5 also asks you "
        "to click through the key interactions before reading it; do that manually in "
        "DevTools before deleting anything."
    )


# --------------------------------------------------------------------------- #
# Lighthouse (optional)
# --------------------------------------------------------------------------- #


def run_lighthouse(report: Report, url: str, mobile: bool, timeout: float) -> None:
    """Run the Lighthouse CLI and surface its ``*-insight`` audits as leads.

    Raises:
        MissingDependency: If the ``lighthouse`` binary is not on PATH.
    """
    binary = shutil.which("lighthouse")
    if not binary:
        raise MissingDependency(
            "the Lighthouse CLI (requested by --lighthouse)",
            "npm install -g lighthouse   # requires Node >= 22.19 for Lighthouse 13",
        )
    with tempfile.TemporaryDirectory() as workdir:
        output = Path(workdir) / "lh.json"
        command = [
            binary, url,
            "--quiet",
            "--only-categories=performance",
            f"--form-factor={'mobile' if mobile else 'desktop'}",
            "--throttling-method=simulate",
            "--output=json", f"--output-path={output}",
            '--chrome-flags=--headless=new --no-sandbox',
        ]
        try:
            subprocess.run(command, check=True, capture_output=True,
                           timeout=max(120.0, timeout * 4))
        except subprocess.TimeoutExpired:
            report.note("Lighthouse timed out; its results are not included.")
            return
        except subprocess.CalledProcessError as exc:
            report.note("Lighthouse exited non-zero; its results are not included. "
                        f"stderr: {exc.stderr.decode('utf-8', 'replace')[:300]}")
            return
        data = json.loads(output.read_text(encoding="utf-8"))

    audits = data.get("audits", {})
    failing = []
    for audit_id in LIGHTHOUSE_INSIGHT_IDS:
        audit = audits.get(audit_id)
        if audit and audit.get("score") is not None and audit["score"] < 1:
            failing.append({"id": audit_id, "title": audit.get("title"),
                            "displayValue": audit.get("displayValue")})
    report.meta["lighthouse_version"] = data.get("lighthouseVersion")
    report.meta["lighthouse_benchmark_index"] = data.get("environment", {}).get(
        "benchmarkIndex")
    if failing:
        report.add(Finding(
            id="performance.lighthouse.insights",
            severity="advisory", rule="lighthouse-insight", wcag_sc=None, url=url,
            viewport=report.meta.get("emulation"), selector=None, section="page",
            message=f"{len(failing)} Lighthouse insight audit(s) are failing.",
            evidence={"insights": failing},
            how_to_fix="Treat these as leads, not evidence. Lighthouse cannot measure "
                       "INP, and its score is 55% TBT+CLS — never present it as a Core "
                       "Web Vitals result.",
        ))
    report.note(
        "Lighthouse ran in simulated throttling mode. Its score is not a Core Web "
        "Vitals result and lab and field can move in opposite directions; the "
        "defensible number is CrUX p75 over a 28-day window, segmented by form factor."
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def build_parser():
    """Construct the argument parser."""
    parser = base_parser(
        "audit_performance.py",
        "Collect load metrics, weight, third parties, render-blocking resources and "
        "code coverage; compare against a budget and exit non-zero on breach.",
        epilog=(
            "Thresholds come from references/performance.md §1.1, §3-4 and the §6\n"
            "budget table. Run --init-budget to write the JSON form of that table.\n\n"
            "Reminder from §5.2: field data comes first. This script is step 7 of the\n"
            "nine-step workflow, not step 1 — it automates the deterministic checks and\n"
            "regression-gates them. It does not tell you whether real users are slow."
        ),
    )
    parser.set_defaults(fail_on="major")  # a budget breach should gate a deploy
    parser.add_argument("--budget", metavar="PATH",
                        help="Budget JSON to enforce (see --init-budget)")
    parser.add_argument("--init-budget", metavar="PATH", dest="init_budget",
                        help="Write a starter budget file to PATH and exit")
    parser.add_argument("--desktop", action="store_true",
                        help="Emulate desktop instead of the mobile config "
                             "(Slow 4G + 4x CPU). Desktop lab numbers are "
                             "structurally optimistic; say so in the report.")
    parser.add_argument("--no-throttle", action="store_true",
                        help="Disable network and CPU throttling (headless CI machines "
                             "are unrealistically fast — use only for A/B debugging)")
    parser.add_argument("--lighthouse", action="store_true",
                        help="Additionally run the Lighthouse CLI and surface its "
                             "*-insight audits")
    parser.add_argument("--settle", type=float, default=3.0, metavar="SECONDS",
                        help="Idle time after load before reading metrics (default: 3)")
    return parser


def main() -> int:
    """Run the performance audit and return the process exit code."""
    parser = build_parser()
    # --init-budget is a standalone action; the positional url is not required.
    if "--init-budget" in sys.argv:
        index = sys.argv.index("--init-budget")
        try:
            path = sys.argv[index + 1]
        except IndexError:
            parser.error("--init-budget requires a path")
        write_budget_template(path)
        print(f"Wrote budget template to {path}")
        return EXIT_OK

    args = parser.parse_args()
    url = normalise_url(args.url)
    budget = load_budget(args.budget)

    width, height = DESKTOP_VIEWPORT if args.desktop else MOBILE_VIEWPORT
    emulation = (f"{width}x{height} "
                 + ("desktop, unthrottled" if args.no_throttle else
                    ("desktop" if args.desktop
                     else "mobile, Slow 4G (150 ms / 1.6 Mbps / 750 Kbps), 4x CPU")))

    report = Report(tool="audit_performance", target=url)
    report.meta["emulation"] = emulation
    report.meta["budget_file"] = args.budget or "(built-in defaults)"

    sync_playwright = import_playwright()
    recorder = NetworkRecorder()

    with sync_playwright() as playwright:
        browser = launch_chromium(playwright)
        try:
            context = browser.new_context(
                viewport={"width": width, "height": height},
                device_scale_factor=2.6 if not args.desktop else 1,
                is_mobile=not args.desktop,
                has_touch=not args.desktop,
            )
            context.add_init_script(METRICS_INIT_JS)
            page = context.new_page()
            console_errors: list[str] = []
            page.on("console", lambda msg: console_errors.append(msg.text)
                    if msg.type == "error" else None)

            session = context.new_cdp_session(page)
            session.send("Network.enable")
            recorder.attach(session)
            if not args.no_throttle:
                session.send("Network.emulateNetworkConditions", {
                    "offline": False,
                    "latency": SLOW_4G["rtt_ms"],
                    "downloadThroughput": SLOW_4G["download_bytes_per_s"],
                    "uploadThroughput": SLOW_4G["upload_bytes_per_s"],
                })
                session.send("Emulation.setCPUThrottlingRate",
                             {"rate": 1 if args.desktop else CPU_THROTTLE_MOBILE})
            stylesheet_urls: dict[str, str] = {}
            try:
                session.send("Profiler.enable")
                session.send("Profiler.startPreciseCoverage",
                             {"callCount": False, "detailed": True})
                session.send("DOM.enable")
                session.send("CSS.enable")
                session.on("CSS.styleSheetAdded", lambda event: stylesheet_urls.update(
                    {event["header"]["styleSheetId"]:
                     event["header"].get("sourceURL") or ""}))
                session.send("CSS.startRuleUsageTracking")
            except Exception:  # noqa: BLE001 - coverage is optional
                pass

            try:
                try:
                    page.goto(url, wait_until="load", timeout=args.timeout * 1000)
                except Exception as exc:  # noqa: BLE001
                    raise TargetUnreachable(url, str(exc).splitlines()[0]) from exc
                page.wait_for_timeout(int(args.settle * 1000))

                metrics = page.evaluate(COLLECT_METRICS_JS)
                dom = page.evaluate(RENDER_BLOCKING_JS)
                coverage = collect_coverage(session, stylesheet_urls)
            finally:
                context.close()

            if args.lighthouse:
                run_lighthouse(report, url, mobile=not args.desktop,
                               timeout=args.timeout)
        finally:
            browser.close()

    # --- record measurements, then compare against the budget ---------------
    report.meta["metrics"] = {
        "lcp_ms": metrics["lcp_ms"], "lcp_element": metrics["lcpElement"],
        "fcp_ms": metrics["fcp_ms"], "cls": metrics["cls"],
        "ttfb_ms": metrics["ttfb_ms"], "tbt_ms": metrics["tbt_ms"],
        "longest_task_ms": round(metrics["longestTask_ms"], 1),
        "long_task_count": metrics["longTaskCount"],
        "cls_sources": metrics["clsSources"],
        "next_hop_protocol": dom["nextHopProtocol"],
    }
    check_timings(report, url, metrics, budget)
    weight = check_resources(report, url, recorder.resources(), budget)
    report.meta["weight"] = weight
    check_render_blocking(report, url, dom,
                          urlparse(url).hostname or "", budget)
    check_coverage(report, url, coverage, budget)

    if recorder.failed and budget["counts"].get("failed_requests") == 0:
        report.add(Finding(
            id="performance.network.failed-requests",
            severity="minor", rule="failed-request", wcag_sc=None, url=url,
            viewport=emulation, selector=None, section="network",
            message=f"{len(recorder.failed)} subresource request(s) failed.",
            evidence={"urls": recorder.failed[:10]},
            how_to_fix="Remove or fix the broken references; every failed request is "
                       "wasted connection setup on a slow link.",
        ))
    if console_errors and budget["counts"].get("console_errors") == 0:
        report.add(Finding(
            id="performance.console.errors",
            severity="minor", rule="console-error", wcag_sc=None, url=url,
            viewport=emulation, selector=None, section="runtime",
            message=f"{len(console_errors)} console error(s) during load.",
            evidence={"examples": console_errors[:6]},
            how_to_fix="Fix or silence them; a noisy console hides real regressions.",
        ))

    unsupported = [name for name, ok in metrics["observerSupport"].items() if not ok]
    if unsupported:
        report.note(f"These PerformanceObserver entry types were unsupported in this "
                    f"browser build and were not measured: {', '.join(unsupported)}.")
    report.note(
        "INP is a field-only metric and is not reported here. TBT is the lab proxy; "
        "an uninteracted page load cannot produce an INP value, and Lighthouse cannot "
        "either. Get INP from CrUX p75 or RUM, and check p95 as well — jank "
        "concentrates in the tail."
    )
    report.note(
        "Lab numbers do not predict field numbers. Measure CLS with the consent banner "
        "present, not dismissed, and expect no field movement for up to 28 days after "
        "a deploy."
    )
    for key, value in budget["resource_bytes"].items():
        if value is None and weight["by_type"].get(key):
            report.note(
                f"No budget is set for {key} weight ({weight['by_type'][key]} bytes "
                f"measured). performance.md §6 asks you to set this one explicitly."
            )
    return finish(report, args)


if __name__ == "__main__":
    run_cli(main)
