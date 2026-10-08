# Visualization & animation libraries

**Covers:** which visualization, mathematics and animation libraries to use when a site needs them, with pinned versions, verified CDN paths and licence status.
**Load when:** `content.has_visualizations: true` or `content.has_animation_layer: true` in the audit profile — or whenever a site is about to hand-roll a chart, a 3D scene, or a choreographed sequence.
**Source:** the Viz Bench survey (28 libraries, 7 subject areas, all instrumented), 17 August 2026. Every version and CDN path below was read out of the published package, not out of documentation.

---

## 1. The rule

**Default to the shortlist. Do not hand-roll, and do not reach for a general charting library.**

Two failure modes account for nearly all bad visualization work:

| Failure | What it looks like | Why it happens |
|---|---|---|
| Hand-rolled draw loop | Animations feel lackluster; every widget re-implements update ordering; nothing can be scrubbed, reversed or retimed | A bare canvas loop has **no dependency model** and **no timeline**. Those are the two things worth buying. |
| Reaching for a charting library | 460 KB gzipped to draw one contour plot; the abstraction fights you the moment the picture is a *construction* rather than a *chart* | Charting libraries solve a problem most of these sites do not have |

Canvas is not the problem — two of the four core picks draw to canvas. The missing pieces are a **dependency graph** (JSXGraph) and a **timeline** (GSAP).

**Performance is the selection criterion, not a tiebreaker.** The benchmark's meter measures time from "parameter changed" to "layout complete", forcing synchronous layout before stopping the clock so DOM/SVG libraries are charged for the reflow they actually cause. Under one 60 Hz frame (16.7 ms) means the picture stays attached to the cursor. Over two frames means the user stops exploring and starts waiting. Unless the profile says otherwise, **pick the fastest option that covers the requirement.**

---

## 2. The shortlist

**Total runtime cost, computed from the table below: ~647 KB gzipped** for all five entries (JSXGraph 250 + Three.js 184 + KaTeX 95 + D3 90 + GSAP 28). No build step, no server.

Nothing on this list is loaded unconditionally. Realistic per-page totals are much lower, because Three.js only ships on 3D routes and KaTeX only where there is mathematics:

| What you actually load | gzip |
|---|---|
| 2D maths page — JSXGraph + GSAP + D3 | **368 KB** |
| 2D maths with typesetting — the above + KaTeX | **463 KB** |
| 3D page — Three.js + GSAP + D3 | **302 KB** |
| Everything at once | **647 KB** |

**On the "~430 KB core four" figure.** The source survey this file is derived from reports ~430 KB as its headline number for a four-library core. It does not reconcile with the per-library sizes in the table: the four non-typesetting libraries (JSXGraph + Three.js + GSAP + D3) come to **552 KB**, and no four-library subset of the table sums to 430. Treat 430 KB as the survey's headline, not as a derived total, and **quote the table figures in any report** — they are the ones a client can verify against a network panel.

| Library | Version | Licence | gzip | Global | Use it for |
|---|---|---|---|---|---|
| **JSXGraph** | 1.13.2 | MIT | 250 KB | `JXG` | Interactive 2D mathematics. The only permissively-licensed library with a real constraint/dependency graph. Ships RK4, slope and vector fields, Lagrange, Neville, cardinal splines, B-splines, NURBS, regression polynomials, Riemann sums, `fminbr` and Newton. |
| **Three.js** | 0.185.1 | MIT | 184 KB | ESM only | Everything 3D. The only 3D option that scales to smooth animation and custom shading. You write the axes and labels yourself. |
| **GSAP** | 3.15.0 | free, **not OSI** | ~28 KB core | `gsap` | The timeline layer. Nested, scrubbable, retimeable sequences; SVG attribute animation including `d`. |
| **D3** | 7.9.0 | ISC | 90 KB | `d3` | Fields and the escape hatch — used as primitives drawn **to canvas**, not as a charting library: `d3.contours`, `d3.Delaunay`, `d3.scale*`, colour ramps. |
| **KaTeX** | 0.18.4 | MIT | ~95 KB | `katex` | Typesetting. Chosen for **synchronous** rendering — inside a timeline you need a label's dimensions immediately. |

### Situational — add per panel, never globally

| Library | Version | Licence | gzip | Global | Add it when |
|---|---|---|---|---|---|
| **uPlot** | 1.6.32 | MIT | 21 KB | `uPlot` | The panel is curves over a shared ascending x. Fastest 2D line renderer on the web by a distance. Pure profit at that size. |
| **ml-matrix** | 6.15.0 | MIT | ~24 KB | `mlMatrix` | Eigen, SVD, QR, Cholesky, LU. Faithful JAMA port, actively maintained. |
| **simple-statistics** | 7.10.1 | ISC | ~9 KB | `ss` | Regression, KDE, t-tests, quantiles. No dependencies. |
| **Rough.js** | 4.6.6 | MIT | ~10 KB | `rough` | Annotation that should read as annotation — signals "sketch, not measurement". A *generator*, not an animator; pair it with GSAP or anime.js. |
| **Observable Plot** | 0.6.17 | ISC | — | `Plot` | Statistical figures rendered on demand, lazily. **Never behind a drag** — `Plot.plot()` returns a new detached SVG every call and there is no patch API by design. |
| **math.js** | 15.2.0 | Apache-2.0 | 170 KB | `math` | Only on routes that accept user-typed expressions. |
| **anime.js** | 4.5.0 | MIT | — | `anime` | The OSI-licensed substitute for GSAP. See §4. |

### Verified CDN paths

Pin exact versions. **Never load any of these as `@latest`.**

```html
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/jsxgraph@1.13.2/distrib/jsxgraph.css">
<script src="https://cdn.jsdelivr.net/npm/jsxgraph@1.13.2/distrib/jsxgraphcore.js"></script>
<script src="https://cdn.jsdelivr.net/npm/gsap@3.15.0/dist/gsap.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.18.4/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.18.4/dist/katex.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/uplot@1.6.32/dist/uPlot.min.css">
<script src="https://cdn.jsdelivr.net/npm/uplot@1.6.32/dist/uPlot.iife.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/ml-matrix@6.15.0/matrix.umd.js"></script>
<script src="https://cdn.jsdelivr.net/npm/simple-statistics@7.10.1/dist/simple-statistics.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/roughjs@4.6.6/bundled/rough.js"></script>
<script src="https://cdn.jsdelivr.net/npm/animejs@4.5.0/dist/bundles/anime.umd.min.js"></script>
```

Three.js is **ESM only** — there has been no UMD build since r161 and there is no way around that:

```html
<script type="importmap">
{ "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.185.1/build/three.module.min.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.185.1/examples/jsm/"
} }
</script>
<script type="module">import * as THREE from 'three';</script>
```

> **Production sites should vendor these locally, not load from a CDN.** A third-party origin on the critical path is a performance liability, a privacy disclosure, and a single point of failure. See `performance.md` § third-party scripts, and `security-and-hygiene.md` for SRI if a CDN is unavoidable.

---

## 3. Choosing — decision table

| The panel is… | Use | Not |
|---|---|---|
| A construction: draggable points, gliders on curves, tangents attached to gliders | **JSXGraph** | any chart library — the abstraction is wrong |
| Curves over a shared ascending x (convergence, trajectories, error vs iteration) | **uPlot** | ECharts/Chart.js — 15–20× the bytes for less speed |
| A scalar field, contour, heatmap or raster | **d3-contour → canvas** | Plotly (renders tens of thousands of SVG paths) |
| Scattered data needing interpolation or nearest-neighbour | **d3.Delaunay** | — |
| Anything in 3D, animated or shaded | **Three.js** | Babylon.js (1.7 MB), vtk.js (625 KB, verbatim C++ API) |
| A static 3D figure that needs axes, ticks and a colourbar more than it needs dragging | **Plotly 3D** — the one justified Plotly use | Three.js (you'd rebuild the chrome) |
| A choreographed sequence: sequence, overlap, hold, scrub backwards, retime | **GSAP** (or anime.js v4 if OSI required) | a hand-rolled rAF loop |
| A statistical figure rendered once, on demand | **Observable Plot**, lazily | anything behind a drag |
| Mathematical typesetting inside a timeline | **KaTeX** | MathJax — asynchronous, and an `await` mid-choreography fights the model |
| Eigen/SVD/QR/Cholesky | **ml-matrix** | numeric.js (2012, broken licence metadata), numjs, odex, jStat |
| Genuine FEM meshes | vtk.js, accepting 625 KB | anything else |

---

## 4. Licence trap — read before committing

**GSAP is free, not open source.** Webflow acquired GreenSock in October 2024 and, effective 30 April 2025, made the entire toolset — core plus every formerly paid plugin — free for everyone including commercial work. That is real. But it is a **no-charge grant under a proprietary licence**, revocable for non-compliance, with terms Webflow may amend unilaterally. The one prohibited use is building a tool that lets end users author animations visually without code.

**If the client or institution requires OSI-approved licences, GSAP fails that test despite costing nothing.**

| | GSAP 3.15.0 | anime.js 4.5.0 |
|---|---|---|
| Licence | free, proprietary | **MIT** |
| Timeline | nested, scrubbable | `createTimeline` |
| Path draw | DrawSVG | `svg.createDrawable` |
| Morph `d` | MorphSVG | `svg.morphTo` (lower quality) |
| Motion path | MotionPath | `svg.createMotionPath` |
| FLIP | Flip | **none** |

anime.js v4 is a real answer, not a consolation prize — you lose FLIP and some morph quality, and that is roughly the whole difference.

**Also check:** Desmos and GeoGebra have the best interaction quality in the field and both restrict free use to non-commercial or individual classroom contexts. Desmos has no self-hostable bundle, so every visitor's browser calls out to `desmos.com` with your API key. JSXGraph is the licence-clean substitute.

---

## 5. Rejected, with reasons

Audit finding: if a site is using one of these, ask why.

| Library | Why not |
|---|---|
| **Plotly 3.7.0** | 460 KB gzipped minimum for 2D contour, 526 KB for 3D; 2D contours render as tens of thousands of SVG paths; 4.0 in RC with breaking changes. Justified only for a static 3D figure needing full chart chrome. |
| **Desmos · GeoGebra** | Licence-restricted; Desmos phones home with your API key. |
| **MathBox 2.3.1** | The most expressive tool ever built for this, and abandoned. Bootstrap imports `WebGL1Renderer`, removed from three at r163. Works only against three r160.1 UMD — forecloses WebGPU permanently. |
| **Vega / Vega-Lite** | The reactive signal model is architecturally the most correct thing in the survey. 272 KB plus a JSON DSL plus a compile-once-drive-signals discipline is a lot to hand the next maintainer, and anything outside the grammar means writing signal-expression strings. |
| **ECharts 6.1.0** | Genuinely good, 359 KB, **no contour series** — so it does not remove the need for `d3-contour`. Take it only if several panels want heatmaps or large scatter. |
| **Chart.js · p5.js · Two.js · Paper.js** | Each fine at its job, none a fit. p5 is 277 KB, LGPL-2.1, no plotting concepts. Paper.js is Canvas-only, last released July 2024 — keep it solely for Bézier intersection and boolean path ops. |
| **Babylon.js** | 1.7 MB gzipped. Technically excellent; centre of gravity is games. |
| **Motion 13** | A UI motion library, not a choreography library. No rich timeline object, no path drawing, no morphing. Use for chrome *around* a visualization, never for the visualization. |
| **Lottie** | Plays pre-authored After Effects files. Cannot react to computed data, cannot be driven by a slider, cannot morph to a computed shape. Fine for a spinner. |
| **Motion Canvas 3.17.2** | Essentially a JavaScript Manim and genuinely excellent — but owns its own canvas and render loop and targets **video**. Last stable Dec 2024; active fork is Revideo. Use only if the deliverable is a rendered video. |
| **Snap.svg · Vivus · Popmotion · Velocity · Theatre.js** | Dead (Adobe stopped 2017), superseded, wound down into Motion, v2 never shipped, and no release in 27 months respectively. Theatre.js's studio is AGPL. |
| **numeric.js · numjs · odex · jStat** | 2012 code with broken licence metadata; native Node deps; a three-year-old RC with no UMD build; four years unmaintained. |
| **Python in the browser** (Pyodide, Dash, Panel, Shiny, Streamlit) | Rejected as *runtime*, not as tools. Python belongs in the authoring pipeline — precompute with NumPy/SciPy/PyVista/Manim/Matplotlib and ship the result as data. A Flask back end is usually not needed and makes the page worse. |

---

## 6. Structural rules

These are architecture, not preference. Check them.

**Precompute rather than compute in the browser.** Anything expensive and deterministic — mesh generation, PDE solutions, high-resolution fields — is computed once at build time by a Python step and shipped as data. The browser interpolates and draws.

**One factory per engine, themed in one place.** A board factory (JSXGraph defaults), a scene factory (Three.js camera/lights/axes/orbit/resize), a motion helper (timeline reveal, path draw, scrub, **reduced-motion**), a field helper (d3-contour → canvas, colour ramps, quiver). Per-widget theming is how a site drifts out of visual consistency.

**Theme both ways before shipping.** JSXGraph, function-plot and Plotly assume a light page and look wrong on a dark one until you write CSS for them. Flipping the site's theme is the fastest way to find which library will cost you styling work. *Check: toggle the site's colour scheme and re-screenshot every visualization panel.* A panel legible in only one theme is a **1.4.3 / 1.4.11 failure against that theme**, not a styling preference — the full both-themes audit procedure, including how to force each state and the `forced-colors` third rendering, is `code-quality.md` §10.

**Instrument the render cost.** Keep a rolling average of the time from "parameter changed" to "layout complete" during development, reading `offsetHeight` before stopping the clock to force synchronous layout — this charges DOM/SVG libraries for the reflow they actually cause. It does not capture GPU compositing, so WebGL flatters itself slightly. Numbers move with the machine; **the ordering transfers.**

**Every library is a dependency you will maintain.** Pin versions. Vendor them. Record licence and gzip size in the audit profile.

---

## How to verify

| Check | How | Pass |
|---|---|---|
| No `@latest` or unpinned CDN references | `grep -rnE 'cdn\.jsdelivr\.net/npm/[^@"]+["/]' <site>` and search for `@latest` | Every URL carries an exact version |
| Total viz payload is known and budgeted | `python3 ../scripts/audit_performance.py <url> --json out.json` → read the resource breakdown by origin | Viz libraries appear as a named line in the performance budget, not a surprise |
| Interaction stays under one frame | `python3 ../scripts/audit_motion.py <url>` with CPU throttling; drive each interactive control | No long-animation-frame entries attributable to the widget |
| Reduced motion is honoured by the animation layer | Run `audit_motion.py` twice, with and without `prefers-reduced-motion: reduce`, and diff | Behaviour measurably differs; the visualization still conveys its content |
| Visualizations are accessible | `../scripts/audit_a11y.py <url> --all`, plus manual: does each canvas/SVG panel have a text alternative or adjacent prose conveying the same finding? | No canvas or SVG carries information available nowhere else. See `ada/media-and-motion.md`. |
| Licence obligations are recorded | Check the audit profile's library inventory against §2 | Every library has a recorded licence; no OSI-required project ships GSAP |
| Theme parity | Toggle colour scheme, screenshot every panel | No panel renders with unreadable or default-light chrome |

---

## Related

- `animation-and-motion.md` — the rendering-cost rules any library's output must still obey. A library does not exempt you from the compositor-only property list.
- `performance.md` — third-party origin cost, bundle budgets, and why a §2 total of this size (302–647 KB gzipped depending on which panels load) must be a deliberate decision, not an accident. It is multiples of the ≤ ~150–200 KB JS budget; justify it per route or lazy-load it per `dynamic-loading.md`.
- `dynamic-loading.md` — lazy-loading a visualization panel, and the fail-open requirement if it never initialises.
- `ada/media-and-motion.md` — the criteria a moving or graphical panel must meet.
- `site-categories.md` — whether this site's motion budget justifies a timeline layer at all.
