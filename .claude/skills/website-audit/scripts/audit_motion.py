#!/usr/bin/env python3
"""Audit animation cost, scroll jank, reduced-motion honesty and fail-open reveals.

Four things Lighthouse cannot tell you, all measured against the rendered page:

1. **Scroll jank.** Long Animation Frames and long tasks are observed while the
   page is scrolled at a controlled rate under CPU throttling, and every long
   frame is reported *with the scroll position where it happened*. From
   ``references/animation-and-motion.md`` §8.1: "A page that loads in 800 ms and
   drops 40% of frames on every scroll scores 100."

2. **The fail-open violation.** ``references/dynamic-loading.md`` §3 calls this
   "the single highest-value check in this entire skill": a static
   ``opacity: 0`` undone only by JS is a permanent-blank-page hazard. The script
   loads the page twice — once with JavaScript enabled, once disabled — and
   diffs the *visible* text, so content that never appears is named explicitly.

3. **Layout-triggering animations.** Every animated property is checked against
   the §2 decision table and the Chromium ``kCompositableProperties`` list.

4. **Reduced-motion honesty.** The page is driven twice, with
   ``prefers-reduced-motion: no-preference`` and ``reduce``, and the two runs
   are diffed. A ``@media (prefers-reduced-motion: reduce)`` block that only
   sets ``animation: none`` while JS scroll animations or WAAPI calls keep
   running is a **failed** implementation (§8.3), and that is only visible by
   comparing behaviour, not source.

Usage:
    ./audit_motion.py https://example.com
    ./audit_motion.py https://example.com --headful --cpu-throttle 6
    ./audit_motion.py https://example.com --json out/motion.json --scroll-seconds 8
"""

from __future__ import annotations

import difflib
import json
import tempfile
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
# Thresholds — cited to references/animation-and-motion.md and dynamic-loading.md
# --------------------------------------------------------------------------- #

# animation-and-motion.md §10.6 F2: "<5% red (dropped) frames".
DROPPED_FRAME_BUDGET_PCT = 5.0

# animation-and-motion.md §8.5: the LoAF threshold is 50 ms.
LOAF_THRESHOLD_MS = 50

# performance.md §8 C: "No long task > 200 ms during page load" — reused here as
# the escalation point for a single frame during scroll.
LONG_TASK_SEVERE_MS = 200

# animation-and-motion.md §8.6: from a high-end desktop host, 4x targets mid-tier
# mobile (Lighthouse default) and 10x targets low-end mobile. The doc's warning
# stands: never hardcode 4x without calibrating benchmarkIndex into 125-800.
DEFAULT_CPU_THROTTLE = 4

# animation-and-motion.md §1.4 — Chromium kCompositableProperties, and the audit
# rule that only transform/translate/rotate/scale/opacity are compositor-only
# across all engines.
COMPOSITOR_SAFE = frozenset({"transform", "translate", "rotate", "scale", "opacity"})
COMPOSITOR_CONDITIONAL = frozenset({"filter", "backdrop-filter", "background-color",
                                    "clip-path"})
# §2 decision table — properties whose deepest pipeline stage is Layout.
LAYOUT_TRIGGERING = frozenset({
    "width", "height", "top", "left", "right", "bottom",
    "margin", "margin-top", "margin-right", "margin-bottom", "margin-left",
    "padding", "padding-top", "padding-right", "padding-bottom", "padding-left",
    "min-width", "min-height", "max-width", "max-height",
    "font-size", "line-height", "border-width", "flex-basis", "grid-template-rows",
    # SVG geometry: "orders of magnitude worse than the equivalent translate()"
    "cx", "cy", "r", "x", "y", "viewBox", "points", "d",
})
# §2 decision table — deepest stage is Paint.
PAINT_TRIGGERING = frozenset({
    "box-shadow", "border-radius", "background-position", "background-size",
    "color", "fill", "stroke", "stroke-dashoffset", "stop-color", "text-shadow",
})

# animation-and-motion.md §8.7 — the trace categories that emit DroppedFrame /
# DrawFrame. `Commit` is on the plain category while the frame events are on
# `.frame`, so both are required.
FRAME_CATEGORIES = [
    "-*", "devtools.timeline",
    "disabled-by-default-devtools.timeline",
    "disabled-by-default-devtools.timeline.frame",
    "toplevel", "blink.user_timing", "latencyInfo",
]


# --------------------------------------------------------------------------- #
# In-page instrumentation
# --------------------------------------------------------------------------- #

# LoAF (Chrome 123+) with a `longtask` fallback, plus a rAF sampler that records
# frame deltas alongside scrollY so every long frame can be located on the page.
#
# The doc's warnings are honoured: `longtask` is a fallback only ("a site that
# moves heavy work into rAF shows zero long tasks and still drops every frame"),
# and the rAF delta is calibrated to the observed median rather than assuming
# 60 Hz, because there is no screen.refreshRate API.
MOTION_INIT_JS = r"""
window.__motion = {
  loaf: [], longTasks: [], frames: [], supportsLoaf: false, supportsLongTask: false
};
const M = window.__motion;

if (PerformanceObserver.supportedEntryTypes?.includes('long-animation-frame')) {
  M.supportsLoaf = true;
  new PerformanceObserver(list => {
    for (const f of list.getEntries()) {
      M.loaf.push({
        startTime: f.startTime, duration: f.duration,
        blockingDuration: f.blockingDuration,
        renderDuration: f.renderStart
          ? f.startTime + f.duration - f.renderStart : null,
        styleLayoutDuration: f.styleAndLayoutStart
          ? f.startTime + f.duration - f.styleAndLayoutStart : null,
        scrollY: window.scrollY,
        scripts: (f.scripts || []).map(s => ({
          invokerType: s.invokerType, invoker: s.invoker,
          duration: s.duration,
          forcedStyleAndLayoutDuration: s.forcedStyleAndLayoutDuration,
          sourceURL: s.sourceURL, sourceFunctionName: s.sourceFunctionName,
          windowAttribution: s.windowAttribution
        }))
      });
    }
  }).observe({ type: 'long-animation-frame', buffered: true });
}

if (PerformanceObserver.supportedEntryTypes?.includes('longtask')) {
  M.supportsLongTask = true;
  new PerformanceObserver(list => {
    for (const t of list.getEntries())
      M.longTasks.push({ startTime: t.startTime, duration: t.duration,
                         scrollY: window.scrollY });
  }).observe({ type: 'longtask', buffered: true });
}

// Portable frame sampler. Caveats stated in the report: it measures when the
// callback ran, not when the frame was presented, and it is blind to
// compositor-thread-only jank.
(() => {
  let last = null;
  const tick = now => {
    if (last !== null) M.frames.push({ t: now, delta: now - last, y: window.scrollY });
    last = now;
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
})();
"""

# Visible text, used for the JS-enabled vs JS-disabled fail-open diff. Filtering
# on opacity is the whole point: a static `.reveal { opacity: 0 }` undone only by
# JS shows up here as content that exists in the DOM but is never visible.
VISIBLE_TEXT_JS = r"""() => [...document.body.querySelectorAll('*')]
  .filter(e => { const cs = getComputedStyle(e);
    return cs.display !== 'none' && cs.visibility !== 'hidden' &&
           parseFloat(cs.opacity) > 0.01; })
  .map(e => (e.childNodes[0]?.nodeType === 3 ? e.childNodes[0].nodeValue.trim() : ''))
  .filter(s => s.length > 12)"""

# dynamic-loading.md §8.1 check 1 — static hidden states in the stylesheet, the
# fail-closed anti-pattern. A hit is only a finding when it is *not* inside a
# @keyframes block, a @supports block, or scoped to a JS-applied class.
STATIC_HIDDEN_JS = r"""() => {
  const hits = [];
  const HIDDEN = /(^|;)\s*(opacity\s*:\s*0(\.0+)?\s*(;|$)|visibility\s*:\s*hidden|transform\s*:\s*[^;]*scale\(\s*0\s*\))/i;
  const JS_SCOPED = /\.(js|no-js|is-|has-|wf-)[\w-]*/;
  for (const sheet of document.styleSheets) {
    let rules; try { rules = sheet.cssRules; } catch { continue; }   // cross-origin
    const walk = (list, insideSafeContext) => {
      for (const rule of list) {
        const safe = insideSafeContext ||
          rule.type === CSSRule.KEYFRAMES_RULE ||
          rule.type === CSSRule.SUPPORTS_RULE ||
          (rule.conditionText || '').includes('prefers-reduced-motion');
        if (rule.cssRules) { walk(rule.cssRules, safe); continue; }
        if (safe || !rule.selectorText || !rule.style) continue;
        if (!HIDDEN.test(rule.style.cssText)) continue;
        if (JS_SCOPED.test(rule.selectorText)) continue;   // applied by JS: fails open
        let matched = 0;
        try { matched = document.querySelectorAll(rule.selectorText).length; } catch {}
        if (!matched) continue;
        hits.push({ selector: rule.selectorText,
                    declaration: rule.style.cssText.slice(0, 160),
                    matchedElements: matched,
                    stylesheet: rule.parentStyleSheet?.href || 'inline' });
      }
    };
    walk(rules, false);
  }
  return hits.slice(0, 30);
}"""

# animation-and-motion.md §1.4 / §2 — every animation currently attached to the
# document, read from the rendered page rather than from source.
ANIMATED_PROPERTIES_JS = r"""() => {
  const path = el => { const seg = [];
    for (let n = el; n && n.nodeType === 1 && seg.length < 4; n = n.parentElement) {
      let s = n.tagName.toLowerCase();
      if (n.id) { seg.unshift(s + '#' + n.id); break; }
      if (n.classList.length) s += '.' + [...n.classList].slice(0, 2).join('.');
      seg.unshift(s);
    } return seg.join(' > '); };
  const sectionOf = el => {
    const host = el.closest && el.closest('section, article, header, footer, nav, main');
    return host ? host.tagName.toLowerCase() + (host.id ? '#' + host.id : '') : null;
  };
  const out = [];
  let animations = [];
  try { animations = document.getAnimations(); } catch { return out; }
  for (const anim of animations) {
    const target = anim.effect?.target;
    if (!target || !target.getBoundingClientRect) continue;
    let properties = [];
    try {
      const keyframes = anim.effect.getKeyframes();
      properties = [...new Set(keyframes.flatMap(k =>
        Object.keys(k).filter(p =>
          !['offset', 'composite', 'computedOffset', 'easing'].includes(p))))];
    } catch { continue; }
    // WAAPI reports camelCase; the reference tables use kebab-case.
    properties = properties.map(p => p.replace(/[A-Z]/g, c => '-' + c.toLowerCase()));
    out.push({ selector: path(target), section: sectionOf(target),
               properties, animationName: anim.animationName || null,
               playState: anim.playState,
               iterations: anim.effect.getTiming().iterations,
               duration: anim.effect.getTiming().duration });
  }
  return out;
}"""

# Behavioural signals that must change when prefers-reduced-motion is honoured.
MOTION_STATE_JS = r"""() => {
  let running = 0, names = [];
  try {
    for (const a of document.getAnimations())
      if (a.playState === 'running') { running++;
        if (a.animationName) names.push(a.animationName); }
  } catch {}
  return {
    reducedMotionMatches: matchMedia('(prefers-reduced-motion: reduce)').matches,
    runningAnimations: running,
    animationNames: [...new Set(names)].slice(0, 20),
    scrollBehaviorSmooth:
      getComputedStyle(document.documentElement).scrollBehavior === 'smooth' ||
      getComputedStyle(document.body).scrollBehavior === 'smooth',
    smoothScrollLibrary: !!(window.Lenis || window.lenis ||
                            window.LocomotiveScroll || window.locomotive ||
                            document.documentElement.classList.contains('lenis') ||
                            document.querySelector('[data-scroll-container]')),
    frameSamples: window.__motion.frames.length,
    loafCount: window.__motion.loaf.length
  };
}"""

COLLECT_MOTION_JS = r"""() => {
  const M = window.__motion;
  const deltas = M.frames.map(f => f.delta).filter(d => d > 0).sort((a, b) => a - b);
  // Calibrate to the observed median rather than assuming 60 Hz: there is no
  // screen.refreshRate API and displays range from 30 to 144 Hz.
  const median = deltas.length ? deltas[Math.floor(deltas.length / 2)] : null;
  const longFrames = median
    ? M.frames.filter(f => f.delta > median * 1.5)
        .map(f => ({ delta: +f.delta.toFixed(1), scrollY: Math.round(f.y),
                     missed: Math.max(0, Math.round(f.delta / median) - 1) }))
    : [];
  return {
    supportsLoaf: M.supportsLoaf, supportsLongTask: M.supportsLongTask,
    loaf: M.loaf, longTasks: M.longTasks,
    frameCount: M.frames.length,
    medianFrameDeltaMs: median ? +median.toFixed(2) : null,
    longFrames: longFrames.sort((a, b) => b.delta - a.delta).slice(0, 40),
    longFrameCount: longFrames.length,
    missedFrames: longFrames.reduce((sum, f) => sum + f.missed, 0)
  };
}"""


# --------------------------------------------------------------------------- #
# Browser driving
# --------------------------------------------------------------------------- #


def scroll_pass(page: Any, session: Any, seconds: float, step_px: int) -> None:
    """Scroll the page at a controlled rate using real wheel input.

    ``animation-and-motion.md`` §10.6 F9 is explicit: drive the scroll with
    ``page.mouse.wheel``, **never** ``window.scrollTo`` inside ``evaluate`` —
    the latter bypasses the input pipeline and under-reports scroll jank.

    Args:
        page: The page to scroll.
        session: CDP session (unused directly, kept so callers pass one).
        seconds: How long to scroll for. §8.2 suggests 5-10 s traces.
        step_px: Wheel delta per step.
    """
    del session  # documented for symmetry with the tracing caller
    viewport = page.viewport_size or {"width": 800, "height": 600}
    page.mouse.move(viewport["width"] // 2, viewport["height"] // 2)
    steps = max(1, int(seconds * 1000 / 16))
    for _ in range(steps):
        page.mouse.wheel(0, step_px)
        page.wait_for_timeout(16)


def run_motion_pass(browser: Any, url: str, *, reduced_motion: str,
                    cpu_throttle: int, timeout: float, seconds: float,
                    step_px: int, trace_path: Path | None) -> dict[str, Any]:
    """Load the page, scroll it under throttling, and return the collected data.

    Args:
        reduced_motion: ``"no-preference"`` or ``"reduce"``.
        cpu_throttle: CDP CPU slowdown multiplier.
        trace_path: When set, a Chrome trace is captured over the scroll so
            ``DroppedFrame``/``DrawFrame`` can be counted.
    """
    context = browser.new_context(
        viewport={"width": 412, "height": 823},
        device_scale_factor=2.6, is_mobile=True, has_touch=True,
        reduced_motion=reduced_motion,
    )
    context.add_init_script(MOTION_INIT_JS)
    page = context.new_page()
    session = context.new_cdp_session(page)
    try:
        session.send("Emulation.setCPUThrottlingRate", {"rate": cpu_throttle})
        try:
            page.goto(url, wait_until="load", timeout=timeout * 1000)
        except Exception as exc:  # noqa: BLE001
            raise TargetUnreachable(url, str(exc).splitlines()[0]) from exc
        page.wait_for_timeout(1000)  # let post-load work settle before tracing

        tracing = False
        if trace_path is not None:
            try:
                # start_tracing mutates the categories list, so hand it a copy.
                browser.start_tracing(page=page, path=str(trace_path),
                                      categories=list(FRAME_CATEGORIES))
                tracing = True
            except Exception:  # noqa: BLE001 - tracing is a bonus
                tracing = False

        scroll_pass(page, session, seconds, step_px)

        if tracing:
            try:
                browser.stop_tracing()
            except Exception:  # noqa: BLE001
                tracing = False

        data = page.evaluate(COLLECT_MOTION_JS)
        data["state"] = page.evaluate(MOTION_STATE_JS)
        data["animations"] = page.evaluate(ANIMATED_PROPERTIES_JS)
        data["staticHidden"] = page.evaluate(STATIC_HIDDEN_JS)
        data["visibleText"] = page.evaluate(VISIBLE_TEXT_JS)
        data["traced"] = tracing
        data["scrollHeight"] = page.evaluate("document.documentElement.scrollHeight")
        return data
    finally:
        context.close()


def parse_frame_trace(path: Path) -> dict[str, Any] | None:
    """Count ``DroppedFrame`` vs ``DrawFrame`` in a Chrome trace.

    Implements the counting rules in ``animation-and-motion.md`` §8.7: only
    events carrying ``frameSeqId`` count, only the page's own ``layerTreeId``
    counts, and ``hasPartialUpdate`` is its own bucket rather than a drop.

    Returns:
        A summary dict, or ``None`` if the trace is unreadable or empty.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    events = raw.get("traceEvents", raw if isinstance(raw, list) else [])
    if not events:
        return None

    layer_tree_id = None
    for event in events:
        if event.get("name") == "SetLayerTreeId":
            layer_tree_id = (event.get("args", {}).get("data", {}) or {}).get("layerTreeId")
            break

    def mine(event: dict[str, Any]) -> bool:
        args = event.get("args") or {}
        if "frameSeqId" not in args:
            return False
        return layer_tree_id is None or args.get("layerTreeId") == layer_tree_id

    dropped = [e for e in events if e.get("name") == "DroppedFrame" and mine(e)]
    drawn = [e for e in events if e.get("name") == "DrawFrame" and mine(e)]
    partial = [e for e in dropped if (e.get("args") or {}).get("hasPartialUpdate")]
    total = len(dropped) + len(drawn)
    if total == 0:
        return None
    return {
        "drawnFrames": len(drawn),
        "droppedFrames": len(dropped),
        "partialUpdates": len(partial),
        "droppedPct": round(100 * len(dropped) / total, 2),
    }


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def check_frames(report: Report, url: str, data: dict[str, Any],
                 trace: dict[str, Any] | None, viewport: str) -> None:
    """Long frames and dropped frames, each located by scroll position."""
    if trace:
        report.meta["frames"] = trace
        if trace["droppedPct"] > DROPPED_FRAME_BUDGET_PCT:
            report.add(Finding(
                id="motion.frames.dropped",
                severity="major", rule="dropped-frames", wcag_sc=None, url=url,
                viewport=viewport, selector=None, section="scroll",
                message=(f"{trace['droppedPct']}% of frames were dropped during the "
                         f"scroll pass."),
                # animation-and-motion.md §10.6 F2: "<5% red (dropped) frames"
                evidence={"droppedPct": trace["droppedPct"],
                          "threshold": DROPPED_FRAME_BUDGET_PCT,
                          "drawnFrames": trace["drawnFrames"],
                          "droppedFrames": trace["droppedFrames"],
                          "partialUpdates": trace["partialUpdates"]},
                how_to_fix="Open the Performance panel, record the same scroll, and "
                           "read the Frames track. Sustained yellow (partially "
                           "presented) frames mean main-thread animation work is "
                           "losing to a compositor-driven scroll.",
            ))
    else:
        report.note(
            "No usable frame trace was captured, so the authoritative dropped-frame "
            "count is unavailable. The rAF-based long-frame numbers below are the "
            "portable fallback: they measure when the callback ran, not when the frame "
            "was presented, and they are blind to compositor-thread-only jank."
        )

    if data["longFrameCount"]:
        worst = data["longFrames"][0]
        report.add(Finding(
            id="motion.frames.long",
            severity="major" if data["missedFrames"] > 10 else "minor",
            rule="long-frames", wcag_sc=None, url=url, viewport=viewport,
            selector=None, section="scroll",
            message=(f"{data['longFrameCount']} frame(s) ran over 1.5x the calibrated "
                     f"median of {data['medianFrameDeltaMs']} ms during scroll; the "
                     f"worst was {worst['delta']} ms at scrollY {worst['scrollY']}."),
            evidence={"medianFrameDeltaMs": data["medianFrameDeltaMs"],
                      "frameSamples": data["frameCount"],
                      "estimatedMissedFrames": data["missedFrames"],
                      "worstFramesByScrollPosition": data["longFrames"][:12],
                      "scrollHeight": data["scrollHeight"]},
            how_to_fix="Scroll to each listed scrollY, record a trace there, and read "
                       "the Animations track for red triangles and the Frames track "
                       "for red and yellow frames.",
        ))

    for index, frame in enumerate(sorted(data["loaf"],
                                         key=lambda f: -f["blockingDuration"])[:8]):
        if frame["blockingDuration"] <= 0:
            continue
        forced = max((s["forcedStyleAndLayoutDuration"] for s in frame["scripts"]),
                     default=0)
        severity = "major" if frame["duration"] >= LONG_TASK_SEVERE_MS else "minor"
        culprit = frame["scripts"][0] if frame["scripts"] else {}
        report.add(Finding(
            id=f"motion.loaf.{index}",
            severity=severity, rule="long-animation-frame", wcag_sc=None, url=url,
            viewport=viewport,
            selector=culprit.get("invoker"), section="scroll",
            message=(f"Long animation frame: {frame['duration']:.0f} ms "
                     f"({frame['blockingDuration']:.0f} ms blocking) at scrollY "
                     f"{frame['scrollY']}."),
            evidence={
                "durationMs": round(frame["duration"], 1),
                "blockingDurationMs": round(frame["blockingDuration"], 1),
                "threshold": LOAF_THRESHOLD_MS,
                "styleLayoutDurationMs": frame["styleLayoutDuration"],
                "forcedStyleAndLayoutDurationMs": forced,
                "scrollY": frame["scrollY"],
                "scripts": frame["scripts"][:3],
                "diagnosis": ("layout thrashing" if forced > 0
                              else "expensive selectors or a large DOM"
                              if (frame["styleLayoutDuration"] or 0) > frame["duration"] / 2
                              else "script work"),
            },
            how_to_fix=("Batch DOM reads before writes — a non-zero "
                        "forcedStyleAndLayoutDuration is forced synchronous reflow."
                        if forced > 0 else
                        "Break the work up and yield; move animation work off the "
                        "main thread by animating only transform and opacity."),
        ))

    if not data["supportsLoaf"]:
        report.note(
            "The Long Animation Frames API was unavailable, so long-task timing was "
            "used instead. Long tasks are an incomplete measure: a site that moves "
            "heavy work into requestAnimationFrame shows zero long tasks and still "
            "drops every frame."
        )


def check_animated_properties(report: Report, url: str, animations: list[dict[str, Any]],
                              viewport: str) -> None:
    """§2 decision table applied to every animation attached to the document."""
    for index, animation in enumerate(animations):
        layout = sorted(set(animation["properties"]) & LAYOUT_TRIGGERING)
        paint = sorted(set(animation["properties"]) & PAINT_TRIGGERING)
        conditional = sorted(set(animation["properties"]) & COMPOSITOR_CONDITIONAL)
        if not (layout or paint or conditional):
            continue

        looping = animation["iterations"] in (None, "Infinity") or \
            (isinstance(animation["iterations"], (int, float))
             and animation["iterations"] > 3)

        if layout:
            report.add(Finding(
                id=f"motion.animated-property.layout.{index}",
                severity="major" if looping else "minor",
                rule="layout-triggering-animation", wcag_sc=None, url=url,
                viewport=viewport, selector=animation["selector"],
                section=animation["section"] or "page",
                message=(f"Animates {', '.join(layout)}, which runs Layout on every "
                         f"frame."),
                evidence={"properties": animation["properties"],
                          "layoutTriggering": layout,
                          "animationName": animation["animationName"],
                          "playState": animation["playState"],
                          "iterations": animation["iterations"],
                          "reference": "web.dev measured 50% dropped frames vs 1% for "
                                       "the transform equivalent"},
                how_to_fix="Replace with transform. For width use `transform: scaleX()` "
                           "on a wrapper; for an accordion use "
                           "`grid-template-rows: 0fr -> 1fr`. Re-verify in the "
                           "Animations track: it must show no red triangle.",
            ))
        if paint:
            report.add(Finding(
                id=f"motion.animated-property.paint.{index}",
                severity="minor" if looping else "advisory",
                rule="paint-triggering-animation", wcag_sc=None, url=url,
                viewport=viewport, selector=animation["selector"],
                section=animation["section"] or "page",
                message=f"Animates {', '.join(paint)}, which repaints every frame.",
                evidence={"properties": animation["properties"],
                          "paintTriggering": paint,
                          "iterations": animation["iterations"]},
                how_to_fix="Acceptable for a short one-shot, not for a loop. For "
                           "box-shadow, animate the opacity of a pseudo-element "
                           "instead. Confirm with Rendering > Paint flashing.",
            ))
        if conditional:
            report.add(Finding(
                id=f"motion.animated-property.conditional.{index}",
                severity="advisory", rule="conditionally-composited-animation",
                wcag_sc=None, url=url, viewport=viewport,
                selector=animation["selector"], section=animation["section"] or "page",
                message=(f"Animates {', '.join(conditional)}, which is composited only "
                         f"under conditions."),
                evidence={"properties": conditional,
                          "notes": "filter: blur() and drop-shadow() are pixel-moving "
                                   "and are NOT composited; clip-path animation is "
                                   "still experimental and off by default in stable "
                                   "Chrome; background-color compositing is "
                                   "Chromium-only"},
                how_to_fix="Only transform and opacity are compositor-only across all "
                           "engines. Check the Animations track Summary for the "
                           "compositing-failure reason.",
            ))


def check_fail_open(report: Report, url: str, with_js: list[str],
                    without_js: list[str], static_hidden: list[dict[str, Any]],
                    viewport: str) -> None:
    """dynamic-loading.md §3 — the fail-open requirement, blocker severity."""
    visible_without = set(without_js)
    missing = [
        text for text in with_js
        if text not in visible_without
        and not difflib.get_close_matches(text, list(visible_without), n=1, cutoff=0.9)
    ]
    if missing:
        report.add(Finding(
            id="motion.fail-open.content-requires-js",
            severity="blocker", rule="fail-open-reveal", wcag_sc="1.3.2 (A) context",
            url=url, viewport=viewport, selector=None, section="page",
            message=(f"{len(missing)} block(s) of text are visible only when JavaScript "
                     f"runs; with JavaScript disabled they exist in the DOM but are "
                     f"never visible."),
            evidence={"examples": missing[:10],
                      "visibleWithJs": len(with_js),
                      "visibleWithoutJs": len(without_js),
                      "consequence": "JS 404s, CSP blocks, ad-blocker interference, a "
                                     "slow-network timeout, a non-rendering crawler, a "
                                     "misfiring observer, an element already in the "
                                     "viewport, or a deep link mid-page all leave this "
                                     "content invisible forever; none of the major AI "
                                     "crawlers render JavaScript"},
            how_to_fix="Never put the hidden state in a static base rule. Put it in a "
                       "@keyframes `from` block and let animation-fill-mode apply it "
                       "only when the animation exists, gated behind both "
                       "`@media (prefers-reduced-motion: no-preference)` and "
                       "`@supports (animation-timeline: view())`. If JS must hide it, "
                       "have JS apply the hiding class in <head>, with a <noscript> "
                       "override.",
        ))

    for index, hit in enumerate(static_hidden):
        report.add(Finding(
            id=f"motion.fail-open.static-hidden.{index}",
            severity="major", rule="static-hidden-state", wcag_sc=None, url=url,
            viewport=viewport, selector=hit["selector"], section="stylesheet",
            message=(f"`{hit['selector']}` ships a static hidden state in the "
                     f"stylesheet, matching {hit['matchedElements']} element(s)."),
            evidence={"declaration": hit["declaration"],
                      "stylesheet": hit["stylesheet"],
                      "matchedElements": hit["matchedElements"]},
            how_to_fix="Move the hidden state into a @keyframes `from` block, or scope "
                       "it to a class that JavaScript adds. Also check for phantom "
                       "focus: opacity:0 elements stay in the accessibility tree and "
                       "stay tabbable, so a keyboard user tabs into nowhere.",
        ))


def check_reduced_motion(report: Report, url: str, normal: dict[str, Any],
                         reduced: dict[str, Any], viewport: str) -> None:
    """Run twice and diff — the only way to see whether `reduce` changes anything."""
    normal_state = normal["state"]
    reduced_state = reduced["state"]

    if not reduced_state["reducedMotionMatches"]:
        report.note("prefers-reduced-motion emulation did not take in the second pass; "
                    "the reduced-motion comparison below is not reliable.")
        return

    report.meta["reduced_motion"] = {
        "running_animations_normal": normal_state["runningAnimations"],
        "running_animations_reduced": reduced_state["runningAnimations"],
        "long_frames_normal": normal["longFrameCount"],
        "long_frames_reduced": reduced["longFrameCount"],
    }

    unchanged = (reduced_state["runningAnimations"] >= normal_state["runningAnimations"]
                 and normal_state["runningAnimations"] > 0)
    if unchanged:
        report.add(Finding(
            id="motion.reduced-motion.ignored",
            severity="major", rule="reduced-motion-not-honoured",
            wcag_sc="2.3.3 Animation from Interactions (AAA); 2.2.2 Pause, Stop, Hide (A)",
            url=url, viewport=viewport, selector=None, section="page",
            message=(f"Setting prefers-reduced-motion: reduce did not reduce motion: "
                     f"{reduced_state['runningAnimations']} animation(s) still running "
                     f"versus {normal_state['runningAnimations']} without the "
                     f"preference."),
            evidence={"runningNormal": normal_state["runningAnimations"],
                      "runningReduced": reduced_state["runningAnimations"],
                      "animationNamesReduced": reduced_state["animationNames"],
                      "method": "two full scroll passes, diffed"},
            how_to_fix="The CSS media query does not reach WAAPI, JS animation "
                       "libraries, Lottie, SMIL, <video autoplay>, canvas/rAF loops or "
                       "smooth-scroll libraries. Add a matchMedia listener that stops "
                       "each of those, and substitute a cross-fade rather than "
                       "deleting the transition entirely.",
        ))

    if reduced_state["scrollBehaviorSmooth"]:
        report.add(Finding(
            id="motion.reduced-motion.smooth-scroll-still-on",
            severity="minor", rule="reduced-motion-not-honoured",
            wcag_sc="2.3.3 (AAA)", url=url, viewport=viewport,
            selector=":root", section="page",
            message="`scroll-behavior: smooth` is still applied under "
                    "prefers-reduced-motion: reduce.",
            evidence={"computedScrollBehavior": "smooth"},
            how_to_fix="Gate scroll-behavior behind "
                       "`@media (prefers-reduced-motion: no-preference)`.",
        ))

    if reduced_state["smoothScrollLibrary"]:
        report.add(Finding(
            id="motion.smooth-scroll-library",
            severity="major", rule="smooth-scroll-library",
            wcag_sc="2.3.3 (AAA)", url=url, viewport=viewport, selector=None,
            section="page",
            message="A smooth-scroll library appears to be running while "
                    "prefers-reduced-motion: reduce is set.",
            evidence={"detected": "Lenis / Locomotive-style scroll wrapper",
                      "note": "Lenis honours prefers-reduced-motion only from 1.3.26; "
                              "lenis/framer and Locomotive v5.0.1 still do not"},
            how_to_fix="Check the shipped version. Below Lenis 1.3.26, add a manual "
                       "matchMedia teardown. Also verify `anchors: true` — anchor "
                       "links are broken by default — and that Space / PageDown / "
                       "Home / End still work.",
        ))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def build_parser():
    """Construct the argument parser."""
    parser = base_parser(
        "audit_motion.py",
        "Measure scroll jank, animated-property cost, reduced-motion honesty and the "
        "fail-open reveal requirement.",
        epilog=(
            "Sources: references/animation-and-motion.md §8 and §10.6,\n"
            "references/dynamic-loading.md §3 and §8.1.\n\n"
            "Run with --headful where you can: a headful browser has a real\n"
            "vsync-driven compositor, and headless frame counts under-report jank."
        ),
    )
    parser.set_defaults(fail_on="major")
    parser.add_argument("--cpu-throttle", type=int, default=DEFAULT_CPU_THROTTLE,
                        metavar="N",
                        help=f"CDP CPU slowdown multiplier (default: "
                             f"{DEFAULT_CPU_THROTTLE}, which targets mid-tier mobile "
                             f"from a high-end desktop host)")
    parser.add_argument("--scroll-seconds", type=float, default=6.0, metavar="SECONDS",
                        help="Duration of each scroll pass (default: 6)")
    parser.add_argument("--scroll-step", type=int, default=60, metavar="PX",
                        help="Wheel delta per scroll step (default: 60)")
    parser.add_argument("--headful", action="store_true",
                        help="Run the browser headed; strongly recommended for frame "
                             "measurement")
    parser.add_argument("--skip-reduced-motion", action="store_true",
                        help="Skip the second (prefers-reduced-motion) pass")
    parser.add_argument("--skip-no-js", action="store_true",
                        help="Skip the JavaScript-disabled fail-open comparison")
    return parser


def main() -> int:
    """Run the motion audit and return the process exit code."""
    args = build_parser().parse_args()
    url = normalise_url(args.url)
    viewport = f"412x823 @2.6x isMobile, CPU {args.cpu_throttle}x"

    report = Report(tool="audit_motion", target=url)
    report.meta["emulation"] = viewport
    report.meta["headless"] = not args.headful

    sync_playwright = import_playwright()
    with sync_playwright() as playwright:
        browser = launch_chromium(playwright, headless=not args.headful,
                                  extra_args=["--enable-gpu"])
        try:
            with tempfile.TemporaryDirectory() as workdir:
                trace_path = Path(workdir) / "trace.json"
                normal = run_motion_pass(
                    browser, url, reduced_motion="no-preference",
                    cpu_throttle=args.cpu_throttle, timeout=args.timeout,
                    seconds=args.scroll_seconds, step_px=args.scroll_step,
                    trace_path=trace_path,
                )
                trace = parse_frame_trace(trace_path) if normal["traced"] else None

                reduced = None
                if not args.skip_reduced_motion:
                    reduced = run_motion_pass(
                        browser, url, reduced_motion="reduce",
                        cpu_throttle=args.cpu_throttle, timeout=args.timeout,
                        seconds=args.scroll_seconds, step_px=args.scroll_step,
                        trace_path=None,
                    )

            # None means "not measured"; an empty list is itself the worst-case
            # result (nothing at all is visible without JavaScript).
            without_js: list[str] | None = None
            if not args.skip_no_js:
                context = browser.new_context(
                    viewport={"width": 412, "height": 823},
                    java_script_enabled=False,
                )
                page = context.new_page()
                try:
                    page.goto(url, wait_until="load", timeout=args.timeout * 1000)
                    page.wait_for_timeout(500)
                    without_js = page.evaluate(VISIBLE_TEXT_JS)
                except Exception:  # noqa: BLE001 - a no-JS failure is itself a signal
                    report.note("The page could not be loaded with JavaScript "
                                "disabled; the fail-open diff was skipped.")
                finally:
                    context.close()
        finally:
            browser.close()

    check_frames(report, url, normal, trace, viewport)
    check_animated_properties(report, url, normal["animations"], viewport)
    if without_js is not None:
        check_fail_open(report, url, normal["visibleText"], without_js,
                        normal["staticHidden"], viewport)
    if reduced is not None:
        check_reduced_motion(report, url, normal, reduced, viewport)

    report.meta["scroll_pass"] = {
        "seconds": args.scroll_seconds, "step_px": args.scroll_step,
        "driven_by": "page.mouse.wheel (real input pipeline)",
        "frame_samples": normal["frameCount"],
        "scroll_height_px": normal["scrollHeight"],
    }
    if not args.headful:
        report.note(
            "This run was headless. A headful browser has a real vsync-driven "
            "compositor; headless frame counts under-report jank. Re-run with "
            "--headful before quoting a dropped-frame percentage in a report."
        )
    report.note(
        f"CPU throttling was fixed at {args.cpu_throttle}x. Throttling is relative to "
        "the host, not absolute — calibrate against Lighthouse's benchmarkIndex and "
        "pick a multiplier that lands in the 125-800 mid-tier mobile band rather than "
        "hardcoding a number."
    )
    report.note(
        "Not covered here, and manual: whether the reduced-motion substitute is a "
        "sensible cross-fade rather than a deletion; whether scroll-jacking disorients "
        "users; Ctrl+F and screen-reader behaviour inside content-visibility subtrees; "
        "and whether an infinite-scroll footer is reachable."
    )
    return finish(report, args)


if __name__ == "__main__":
    run_cli(main)
