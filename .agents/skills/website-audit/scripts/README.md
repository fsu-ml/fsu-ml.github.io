# website-audit scripts

Six runnable verification scripts. They exist because the skill's thesis is that
**quality must be verified against the rendered page, not assumed from source
code** — sites are built on unpredictable frameworks and library stacks, and only
the rendered output is comparable between them. These scripts are how that
verification actually happens.

Every threshold in this directory is taken from a reference document in
`../references/` and cited in a comment next to the number. Where a reference is
silent, the script says so in its output rather than inventing authority — see
[What these scripts cannot detect](#what-these-scripts-cannot-detect).

---

## Install

Python **3.11 or newer**.

```bash
cd website-audit/scripts
pip install -r requirements.txt
playwright install chromium          # required — the pip install alone is not enough
```

Optional extras:

```bash
npm install -g lighthouse            # enables audit_performance.py --lighthouse
npm install axe-core                 # lets audit_a11y.py run without CDN access
```

`check_headers.py` needs **none of this**. It is standard library only, on
purpose, so the 60-second security baseline always works.

---

## The scripts

| Script | Needs | What it verifies |
|---|---|---|
| [`check_headers.py`](#check_headerspy) | nothing | Security, caching and hygiene headers, graded on policy |
| [`audit_a11y.py`](#audit_a11ypy) | Playwright + axe-core | axe-core scan, plus the count of what axe could not test |
| [`audit_responsive.py`](#audit_responsivepy) | Playwright | The V1–V8 viewport matrix, findings scoped per section |
| [`audit_performance.py`](#audit_performancepy) | Playwright (+ Lighthouse) | Metrics, weight, third parties, coverage, budget gate |
| [`audit_motion.py`](#audit_motionpy) | Playwright | Scroll jank, animated properties, reduced motion, fail-open |
| [`audit_seo.py`](#audit_seopy) | Playwright | Crawl, on-page, structured data, link graph, CSR detection |

`_common.py` holds the shared finding schema, report renderer, CLI conventions
and graceful-failure plumbing. It is standard library only so `check_headers.py`
can import it.

---

## Shared conventions

Every script takes a URL as its first positional argument and supports the same
core flags:

```
--json PATH        Write the machine-readable report to PATH
--fail-on LEVEL    Exit 1 at this severity or worse (blocker|major|minor|advisory)
--show-all         Print advisory findings in the terminal summary too
--quiet            Suppress the human summary (use with --json in CI)
--timeout SECONDS  Per-navigation timeout (default 30)
--help             Full flag list for that script
```

Default `--fail-on` is `blocker`, except `audit_performance.py` and
`audit_motion.py`, which default to `major` so a budget breach or a jank
regression gates a deploy.

### Output

Human-readable summary on stdout by default, grouped most-severe-first, with
coverage notes at the end. `--json` additionally writes:

```json
{
  "tool": "audit_responsive",
  "target": "https://example.com",
  "started_at": "2026-08-19T10:04:11+00:00",
  "schema": "website-audit/finding@1",
  "meta":     { "...": "run configuration and measurements" },
  "counts":   { "blocker": 0, "major": 3, "minor": 7, "advisory": 12 },
  "notes":    [ "coverage caveats that belong in the written report" ],
  "findings": [ "..." ]
}
```

### Finding schema

Every finding from every script has exactly these keys. Nullable fields are
always present, never omitted.

| Field | Notes |
|---|---|
| `id` | Stable identifier for this instance; used for dedup |
| `severity` | `blocker` \| `major` \| `minor` \| `advisory` |
| `rule` | The check, or the upstream engine's rule id (`color-contrast`) |
| `wcag_sc` | Dotted criterion with level, e.g. `"1.4.10 Reflow (AA)"`, or `null` |
| `url` | The page it was observed on |
| `viewport` | Emulation context, e.g. `"320x512 @2x isMobile,hasTouch"`, or `null` |
| `selector` | CSS-ish path to the offending element, or `null` |
| `section` | Nearest identifiable section or landmark, or `null` |
| `message` | One sentence: what was measured, and why it fails |
| `evidence` | Measured value, threshold, and supporting detail |
| `how_to_fix` | Concrete remediation |

`wcag_sc` is `null` rather than stretched when a finding maps to no criterion.
`references/mobile.md` §0 is explicit about why: mixing a Tier 1 WCAG violation
with a Tier 2 platform recommendation is the fastest way to get an entire audit
report dismissed.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Ran cleanly; nothing at or above `--fail-on` |
| `1` | Ran cleanly; findings at or above `--fail-on`, or a budget breach |
| `2` | Bad command line |
| `3` | Missing dependency — stdout names it and how to install it |
| `4` | Target unreachable, or a required network resource failed |
| `5` | Unexpected internal error, reported without a traceback |

No script ever prints a traceback at a user. Codes 3 and 4 are distinct so CI can
tell "the harness is broken" from "the site is broken".

---

## `check_headers.py`

Grades the baseline in `../references/security-and-hygiene.md`. **Standard
library only.**

```bash
./check_headers.py https://example.com
./check_headers.py https://example.com --path /pricing --path /app/dashboard
./check_headers.py https://example.com --json out/headers.json --fail-on major
```

| Flag | Purpose |
|---|---|
| `--path PATH` | Additional path to check (repeatable). Header config often differs between CDN-cached and origin-served routes — check at least one of each |
| `--no-probe` | Skip the exposed-path and source-map probes |
| `--insecure` | Do not verify TLS certificates (staging hosts) |

Checks HTTPS redirect (301, at most one hop), HSTS, CSP, `X-Content-Type-Options`,
`Referrer-Policy`, `Permissions-Policy`, framing, COOP/CORP, version disclosure,
HTML caching, cookie flags, mixed content, Subresource Integrity, and probes
`/.env`, `/.env.production`, `/.git/HEAD`, `/.git/config`, `/.DS_Store`,
`/backup.zip` and reachable source maps.

**It grades policy, not presence.** The load-bearing consequence:

- `script-src 'unsafe-inline'` **with** a nonce or hash present is *not* flagged.
  Browsers that understand nonces ignore `'unsafe-inline'` entirely, so it is a
  backwards-compatibility fallback, not a hole.
- `script-src 'unsafe-inline'` **without** a nonce or hash *is* flagged, as
  "CSP present but non-functional for XSS".
- A nonce identical across two consecutive responses is reported as equivalent
  to `'unsafe-inline'`, because a CDN-cached nonce is exactly that.

Output includes a letter grade weighted towards CSP effect, so a site cannot
score an A by adding six cheap headers around a broken policy.

---

## `audit_a11y.py`

axe-core through Playwright, plus the sweeps the ADA references name explicitly,
plus the honesty layer.

```bash
./audit_a11y.py https://example.com
./audit_a11y.py https://example.com --all --standard wcag22aa
./audit_a11y.py https://example.com --images --forms --json out/a11y.json
./audit_a11y.py https://example.com --inventory-documents --max-pages 50
```

| Flag | Purpose |
|---|---|
| `--standard wcag20aa\|wcag21aa\|wcag22aa` | WCAG level/version selector (default `wcag22aa`). `wcag20aa` is a real target, not a legacy spelling — Section 508 and the Air Carrier Access Act incorporate WCAG 2.0 AA by reference (`../references/ada/targets.md` §1.3, §1.5). The hyphenated profile spellings `wcag20-aa` / `wcag21-aa` / `wcag22-aa` are accepted too |
| `--images` | Image accessible names; missing alt, filename-like alt, alt starting with "image of" |
| `--contrast` | Contrast sweep, with axe's undecidable cases surfaced rather than passed |
| `--forms` | Label / autocomplete / required sweep |
| `--target-size` | WCAG 2.2 SC 2.5.8 (24 × 24 CSS px) |
| `--inventory-documents` | Crawl for linked `.pdf`/`.docx`/`.pptx`/`.xlsx` and report which document-accessibility references need loading and which profile keys to set |
| `--all` | Every mode above |
| `--axe-source PATH` | Local `axe.min.js` instead of the CDN build |
| `--max-pages N` | Crawl limit for `--inventory-documents` (default 50) |

**The count of what axe could not test is part of the output, always.** Three
numbers are reported, not one: violations found, elements axe could not decide
(`incomplete`), and the success criteria no scanner can evaluate at all. Coverage
notes state that automated tools detect roughly 30–40% of WCAG failures measured
by criterion — and that the more flattering 57%-by-volume figure counts repeat
instances of the same detectable rule, which is a different denominator. Under
`--standard wcag22aa` the report also states that of the six criteria new in
WCAG 2.2, only 2.5.8 is reliably automatable: an automated scan reporting "no
WCAG 2.2 issues" has checked one of six.

Severity is assigned from the criterion, not from axe's `impact` field. axe rates
`color-contrast` as *serious*, but unlabelled controls and missing functional alt
block access entirely and are reported as `blocker`.

---

## `audit_responsive.py`

The verification procedure from `../references/mobile.md` §5.

```bash
./audit_responsive.py https://example.com
./audit_responsive.py https://example.com --viewports V1,V3,V7 --out ./shots
./audit_responsive.py https://example.com --json out/responsive.json
```

| Flag | Purpose |
|---|---|
| `--viewports LIST` | Comma-separated matrix keys (default: all of V1–V8) |
| `--out DIR` | Screenshot directory (default `audit-output/responsive`) |
| `--no-screenshots` | Skip screenshot capture |
| `--headful` | Run the browser headed |

Viewport matrix:

| Key | Viewport | DPR | Why |
|---|---|---|---|
| V1 | 320 × 512 | 2 | WCAG 1.4.10 floor; equals 1280 px at 400% zoom |
| V2 | 360 × 640 | 3 | Most common Android logical width worldwide |
| V3 | 390 × 844 | 3 | iPhone 12–16; the "it looks fine to me" width |
| V4 | 412 × 915 | 2.6 | Large Android; catches layouts tuned to 375–390 only |
| V5 | 568 × 320 | 2 | Landscape; orientation and the 256 px floor |
| V6 | 768 × 1024 | 2 | Tablet portrait |
| V7 | 1280 × 1024 @ 400% zoom | 1 | Desktop equivalence for 1.4.10 |
| V8 | 390 × 844, root `32px` | 3 | WCAG 1.4.4 at 200% text |

Checks meta viewport, horizontal overflow with the offending elements named,
touch targets against SC 2.5.8 (24 px, with the Spacing and Inline exceptions
implemented and the 44/48 platform floors reported separately as advisory),
hover-only affordances confirmed by a dispatched touch sequence, reflow content
loss diffed between 320 px and 1280 px, text below the readability floor, and
inputs below 16 px that trigger iOS focus auto-zoom. Full-page screenshots are
captured per breakpoint.

**Every finding names a section.** The script enumerates sectioning elements and
landmarks once, freezes the list, reuses it at every viewport, and attributes
each finding to the nearest one. This is the whole point: page-level `scrollWidth`
is masked by any ancestor with `overflow-x: hidden`, which most site shells set,
so a page can pass every page-level gate while three of its sections are broken.
When that masking is detected, the script says so explicitly in the notes.

---

## `audit_performance.py`

```bash
./audit_performance.py https://example.com
./audit_performance.py https://example.com --budget budget.json --json out/perf.json
./audit_performance.py https://example.com --lighthouse --desktop
./audit_performance.py --init-budget budget.json
```

| Flag | Purpose |
|---|---|
| `--budget PATH` | Budget JSON to enforce; exits 1 on breach |
| `--init-budget PATH` | Write a starter budget and exit |
| `--desktop` | Emulate desktop instead of the mobile config |
| `--no-throttle` | Disable network and CPU throttling |
| `--lighthouse` | Also run the Lighthouse CLI and surface its `*-insight` audits |
| `--settle SECONDS` | Idle time after load before reading metrics (default 3) |

Default emulation is the Lighthouse mobile config: Slow 4G (150 ms RTT,
1.6 Mbps down, 750 Kbps up) plus a 4× CPU slowdown, 412 × 915.

Collects LCP (with its element), CLS (session-window, `hadRecentInput` excluded,
with source nodes), FCP, TTFB and TBT via a `PerformanceObserver` installed
before navigation; the resource weight breakdown by type from CDP
`encodedDataLength` (not Resource Timing, which reads zero for cross-origin
responses and would silently under-count third-party bytes); the third-party
origin inventory with byte cost; the render-blocking resource list; and unused
CSS/JS from the Coverage API.

`--init-budget` writes the JSON form of the `../references/performance.md` §6 budget table. The
two rows the reference deliberately leaves blank — total image weight and total
page weight, both marked "set explicitly" — are emitted as `null`, and the script
reports the measurement plus a note rather than inventing a threshold.

**INP is not reported, deliberately.** It is a field-only metric; an
uninteracted page load cannot produce one and neither can Lighthouse. TBT is
reported as the lab proxy and labelled as such. TTFB and FCP are reported as
diagnostics at lower severity, because they are not Core Web Vitals and are not
ranking inputs — they are causes, not failures.

---

## `audit_motion.py`

```bash
./audit_motion.py https://example.com
./audit_motion.py https://example.com --headful --cpu-throttle 6
./audit_motion.py https://example.com --scroll-seconds 8 --json out/motion.json
```

| Flag | Purpose |
|---|---|
| `--cpu-throttle N` | CDP CPU slowdown multiplier (default 4) |
| `--scroll-seconds SECONDS` | Duration of each scroll pass (default 6) |
| `--scroll-step PX` | Wheel delta per step (default 60) |
| `--headful` | Run headed — **recommended**, see below |
| `--skip-reduced-motion` | Skip the second pass |
| `--skip-no-js` | Skip the JavaScript-disabled comparison |

Four things Lighthouse cannot see:

1. **Scroll jank.** LoAF and long tasks are observed while the page is scrolled
   at a controlled rate under CPU throttling, and every long frame is reported
   *with the scroll position where it occurred*, so a developer can go straight
   to it. A Chrome trace additionally counts `DroppedFrame` vs `DrawFrame`
   against the 5% budget. Scroll is driven with `page.mouse.wheel`, never
   `window.scrollTo` inside `evaluate` — the latter bypasses the input pipeline
   and under-reports jank.
2. **The fail-open violation.** The page is loaded with JavaScript enabled and
   disabled, and the *visible* text is diffed. Content that exists in the DOM
   but never becomes visible is reported as a blocker, along with any static
   `opacity: 0` / `visibility: hidden` base rule that is not inside a
   `@keyframes`, a `@supports`, or a JS-applied class.
3. **Layout-triggering animations.** Every animation attached to the document is
   read via `document.getAnimations()` and its properties checked against the
   pipeline-stage table. Only `transform` and `opacity` are compositor-only
   across all engines; conditional cases (`filter`, `backdrop-filter`,
   `background-color`, `clip-path`) are flagged with the specific caveat.
4. **Reduced-motion honesty.** The whole scroll pass runs twice, once with
   `prefers-reduced-motion: no-preference` and once with `reduce`, and the two
   are diffed. A media query that only sets `animation: none` while WAAPI, JS
   libraries or a smooth-scroll wrapper keep running is a failed implementation,
   and that is visible only by comparing behaviour.

**Run with `--headful` before quoting a dropped-frame percentage.** A headful
browser has a real vsync-driven compositor; headless frame counts under-report.
The script says so in its notes when run headless.

---

## `audit_seo.py`

```bash
./audit_seo.py https://example.com
./audit_seo.py https://example.com --max-pages 100 --delay 0.5
./audit_seo.py https://example.com --check-external --json out/seo.json
```

| Flag | Purpose |
|---|---|
| `--max-pages N` | Crawl limit (default 50) |
| `--delay SECONDS` | Delay between requests (default 1.0) |
| `--ignore-robots` | Crawl disallowed URLs — only on sites you own |
| `--check-external` | Also status-check outbound external links |
| `--link-check-limit N` | Maximum links to status-check (default 200) |

Respects `robots.txt` by default and rate-limits itself.

Per page: title, meta description, canonical, robots meta **and** `X-Robots-Tag`
(header-level `noindex` is invisible in view-source and is a classic missed cause
of a de-indexed site), Open Graph and Twitter cards, heading outline, hreflang
with reciprocity verification, structured data with required-property validation
and a type inventory, and image alt coverage.

Site-wide: the internal link graph with click depth and orphan pages, broken
links and redirect chains, duplicate titles and descriptions, the
all-canonicals-point-at-`/` misconfiguration, sitemap validity, robots.txt
reachability, and the soft-404 probe.

Each page is fetched **twice** — raw over HTTP and rendered through Playwright —
and diffed, so client-side-rendering dependence is detected rather than assumed.
The verdict is emitted **per route**, because a static homepage plus a
client-rendered `/products` is one site with two answers and only the second
needs the deep dive.

---

## What these scripts cannot detect

Stating this is not modesty; it is what stops a clean run being read as a clean
site. Each script repeats the relevant items in its own coverage notes.

**Automation ceiling.** Automated accessibility tools detect roughly 30–40% of
WCAG failures by criterion. No scanner can evaluate whether the reading order
makes sense, whether alt text is *right*, whether focus order is logical, whether
an error message is actually helpful, or whether a custom widget's keyboard model
is coherent. If every finding in a report maps to an axe rule id, only the
automated layer was run.

**Emulation ceiling.** Headless Chrome cannot observe safe-area insets (DevTools
device mode always returns 0), iOS input auto-zoom, or iOS modal scroll-locking.
Those are static-signal checks here and must be confirmed on a real device; the
report must say which findings were device-confirmed.

**Lab is not field.** Nothing here tells you whether real users are slow. INP is
field-only. Lab and field can move in opposite directions, and no field movement
should be expected for up to 28 days after a deploy. The defensible number is
CrUX or RUM at the 75th percentile over a 28-day window, segmented by form
factor — and check p95 too, because jank concentrates in the tail.

**Headless frame timing is optimistic.** Dropped-frame counts from a headless run
under-report. The only ground truth is a trace from a headful browser, read in
the Performance panel's Frames and Animations tracks.

**Coverage identifies candidates, not dead code.** Coverage is per-session: code
used only on other routes shows as unused. It does not prove deletability.

**Partial crawls over-report.** Orphan pages and click depth are only as good as
the crawl. A finding count without a denominator is meaningless — state the
sample in the report.

**Header grading is not a penetration test.** Nothing in `check_headers.py`
proves an application is secure. It also does not evaluate TLS versions, ciphers
or certificate chains (use `testssl.sh`), cannot read `HttpOnly` cookies set by
JavaScript, and cannot analyse a host-allowlist CSP for JSONP endpoints or
arbitrary-path library CDNs (use Google's CSP Evaluator).

**Structured data is checked at the vocabulary level only.** Whether markup
matches what the page actually displays, and whether Google accepts it, requires
the Rich Results Test and `validator.schema.org` — both.

**One threshold is a script decision, not a documented one**, and is labelled as
such in the JSON output: the raw-vs-rendered text ratio used to classify a route
as client-rendered. The SEO references are silent on it. The definitive rendering
answer for Google is Search Console's URL Inspection → Test Live URL → View
Tested Page → HTML tab.

**Social cards are now doc-backed.** The tag set, the three severity bands and
the image constraints come from `../references/seo/L4-onpage.md` §4.3, which
every social finding cites. What the script does **not** do is fetch `og:image`:
reachability, real pixel dimensions, MIME type and byte size have to be checked
by hand against the constraint table there, and the resolved card confirmed in
the Facebook Sharing Debugger and the LinkedIn Post Inspector — which are also
the only way to clear a stale per-URL cache.

**No script reads the audit profile.** Nothing here loads
`templates/audit-profile.yaml`, so every severity is assigned without profile
context. That matters most for accessibility: `../references/reporting.md` §2
escalates a WCAG A/AA failure to **blocker** only when `compliance.regime` is
something other than `none`, and `audit_a11y.py` cannot know the regime. The same
gap applies to every other profile-conditional rule — `stack.rendering`,
`audience.intent`, `content.has_data_tables` and the rest are invisible to the
scripts. Treat script severities as provisional and re-grade them against
`reporting.md` §2 with the profile in hand.

**Nothing here is a substitute for using the page.** Keyboard-only operation,
screen-reader behaviour, Ctrl+F inside `content-visibility` subtrees, whether an
infinite-scroll footer is reachable, and whether a reduced-motion substitute is a
sensible cross-fade rather than a deletion, all require a human.
