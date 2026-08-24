# Code quality — the cleanest possible website

Framework-agnostic engineering hygiene, **checkable from the outside** with a browser, `curl`, and a headless script. Nothing here requires repo access; everything here is stronger with it.
Load: any "is this well built?" question, any redesign/rebuild assessment, or `code-quality: full`.
Companions: `performance.md` (metrics, delivery, JS budgets), `animation-and-motion.md` (motion mechanics, reduced-motion), `dynamic-loading.md` (loading states and route transitions), `security-and-hygiene.md` (headers, TLS, cookies), `ada/` (accessibility — semantics overlap but the criteria live there), `seo/L2-technical-performance.md`.
Scripts: `../scripts/audit_performance.py`, `../scripts/check_headers.py`.

**Reliability key.** `[P]` primary source · `[S]` secondary, directionally right · `[?]` contested — do not assert.

**Rule for every finding in this file: state the check.** "Use semantic HTML" is not a finding. "37 of 41 clickable elements are `<div onclick>`; counted via `document.querySelectorAll('[onclick]:not(button):not(a)')`" is.

---

## 1. Semantic HTML — the foundation

Semantics are **not only an accessibility concern**. They drive default behaviour, bfcache eligibility, form functionality without JS, keyboard/pointer conventions, crawlability, and **how much CSS and JS you have to write at all**. A div-soup site pays for every one of those in code.

### 1.1 Smells, and how to detect each

| Smell | Detection |
|---|---|
| `<div onclick>` instead of `<button>` / `<a href>` | `document.querySelectorAll('[onclick]').length` and `$$('[onclick]:not(button):not(a):not(input)')`; also grep the bundle for `addEventListener('click'` bound to non-interactive selectors |
| Navigation that isn't `<nav><ul><li><a href>` with **real** `href`s | `$$('nav a:not([href]), nav a[href="#"], nav a[href^="javascript:"]')`. JS-only routing links break middle-click, ⌘-click, "open in new tab", and crawlers |
| No `<main>`, no landmark structure | `$$('main, nav, header, footer, aside, section[aria-label]').length` |
| Heading levels chosen by font size | `$$('h1,h2,h3,h4,h5,h6').map(h=>h.tagName)` — check for skipped levels and multiple `h1` |
| Custom dropdown/disclosure/modal where `<select>`, `<details>`, `<dialog>` or `popover` would do | grep for widget library names; `$$('[role="dialog"]:not(dialog), [role="listbox"]')` |
| Forms without `<form>` / `<label>` / `name` / `type` / `autocomplete` | `$$('input:not([name]), input:not([type])')`; `$$('form:not([action])')`; `$$('input,select,textarea').filter(e=>!e.labels?.length && !e.getAttribute('aria-label'))` |
| `<div>`-based tables | `$$('[class*="table"] [class*="row"]').length` vs `$$('table').length` |

### 1.2 The div-soup metric — say the number

```js
// Paste in DevTools console. Report both numbers in the audit.
const all = document.querySelectorAll('*').length;
const soup = document.querySelectorAll('div, span').length;
const semantic = document.querySelectorAll(
  'main,nav,header,footer,aside,article,section,h1,h2,h3,h4,h5,h6,p,ul,ol,li,dl,dt,dd,' +
  'table,thead,tbody,tr,th,td,form,label,input,select,textarea,button,a,figure,figcaption,' +
  'blockquote,time,address,details,summary,dialog,picture,img,video,audio,fieldset,legend'
).length;
({ all, soup, semantic,
   soupRatio: +(soup / all).toFixed(2),
   semanticRatio: +(semantic / all).toFixed(2) });
```

| Reading | Interpretation |
|---|---|
| `soupRatio > 0.90` | **Div soup.** Report as a structural finding, not a nitpick |
| `soupRatio 0.70 – 0.90` | Typical component-framework output; look at the specific smells in §1.1 instead |
| `soupRatio < 0.70` | Deliberately semantic |
| `all > 1500` | **DOM size finding.** Lighthouse `dom-size-insight`. DOM size correlates directly with layout cost and INP — see `performance.md` §2.2 |

Keep total elements **well under ~1,500**.

---

## 2. Modern CSS platform features — 2026 support and what it means for the audit

**Baseline "Widely available"** = 30 months in all three engines. Treat that as *use unguarded*. Everything else needs a `@supports` guard **or** must degrade harmlessly, and the audit question is always the same two parts:

> **(a)** Is there a `@supports` guard or harmless degradation?
> **(b)** Does the **core task still complete** in the non-supporting engine?

**Test in real Safari and real Firefox, not just Chrome.** This is the single most-skipped step in web QA.

### 2.1 Safe to use unguarded — Baseline Widely available

| Feature | Notes |
|---|---|
| `aspect-ratio` | Chrome/Edge 88, FF 89, Safari 15 |
| **Container queries (size)** | Newly available **Feb 2023** → widely available. Correct replacement for viewport media queries inside reusable components — see `mobile.md` |
| **`:has()`** | Newly available **Dec 2023** → **Widely available June 2026** `[S]` |
| **CSS nesting** | Newly available **Aug 2023** |
| **Cascade layers `@layer`** | Chrome/Edge 99, FF 97, Safari 15.4 |
| **Subgrid** | **Baseline Widely available 15 Mar 2026** `[S]` |
| **`color-mix()`, `oklch()`** | OKLCH: Chrome/Edge 111, Safari 15.4, FF 113 — **~93–95% global** |
| `clamp()`, logical properties | Long-settled |
| **`<dialog>`** | Universal. Use `showModal()` for top-layer + backdrop — replaces an entire class of modal JS |
| **`content-visibility` / `contain-intrinsic-size`** | Chrome 85, FF 125, **Safari 18** — see `performance.md` §3.9 for the mandatory `contain-intrinsic-size` caveat |
| `clip-path`, `:user-invalid`, `lh`/`rlh` units, `Navigator.userActivation` | **Reached Widely available May 2026** `[P]` |
| **Scroll-snap** | Universal |

### 2.2 Safe **with** a `@supports` guard or graceful degradation

| Feature | Status | Audit stance |
|---|---|---|
| **`popover` attribute** | Baseline **Newly available Jan 2025** — all major engines. `ToggleEvent.source` newly available **May 2026** `[P]` | Use it; it retires tooltip/menu JS. Guard only if you support long-tail old browsers |
| **CSS anchor positioning** | Chrome/Edge **125**, Firefox **132**, **Safari 26.0** (Sept 2025; `flip-x`/`flip-y` in `position-try-fallback` and last-successful-fallback memory added in **26.2**). Global ~**88–91%**. Still marked *Limited availability* in some Baseline readings | ⚠️ **Correction to flag: the widely-repeated "Safari 18.2" figure is wrong — WebKit shipped it in Safari 26.** `[P]` **Use `@supports (anchor-name: --a)` and keep a static fallback position** |
| **Relative color syntax** `rgb(from …)` | Chrome/Edge 119, Safari 16.4; **Firefox lagging** — ~**89.6% global** (Feb 2026) `[S]` | **Guard it** |
| **`@property`** | Widely shipped but **Baseline status contested** `[?]`. Enables animatable custom properties | Degrades to a plain custom property — safe as enhancement |
| **`field-sizing: content`** | **Baseline Newly available 16 June 2026** — Chrome/Edge 123, **Safari 26.2** (Dec 2025), **Firefox 152** (June 2026) `[S]` | Pure enhancement; degrades to a fixed-size control. **Kills the auto-growing-textarea JS** |
| **`text-wrap: balance`** | Chrome 114+, Safari 17.5+, Firefox 121+ | Safe, purely cosmetic. **Only applies to ≤ ~6 lines** — use on headings, not body copy |
| **`text-wrap: pretty`** | Chrome/Edge 117 (refined in 130+), **Safari 26+**, **Firefox: no support as of early 2026** `[S]` | Pure enhancement, no layout breakage |
| **Container *style* queries** | **Baseline Newly available May 2026** `[P]` | Guard |
| **`:open` pseudo-class** | **Baseline Newly available May 2026** `[P]` | Guard |
| `image-rendering`, `text-decoration-skip-ink: all`, `SharedWorker` | **Baseline Newly available May 2026** `[P]` | Guard |
| **`scheduler.yield()`** | Chrome/Edge 129, FF 142, **Safari: none** | **Always feature-detect with a `setTimeout` fallback** — `performance.md` §2.2 |
| **Speculation Rules** | **Chromium only**; Safari 26.2 off by default; Firefox none | Silently ignored elsewhere — enhancement only, never load-bearing |
| **Scroll-driven animations** | Chromium; Safari/Firefox partial | **Must** be optional — `animation-and-motion.md` |

**Guard pattern:**
```css
.tooltip { position: absolute; top: 100%; left: 0; }        /* fallback */
@supports (anchor-name: --trigger) {
  .trigger { anchor-name: --trigger; }
  .tooltip { position: absolute; position-anchor: --trigger;
             position-area: block-end span-inline-end;
             position-try-fallbacks: flip-block, flip-inline; top: auto; left: auto; }
}
```

**Check for unguarded use:** fetch every stylesheet and grep for the guarded-tier features, then check for a nearby `@supports`:
`curl -s SITE/assets/app.css | grep -nE 'anchor-name|position-anchor|rgb\(from|text-wrap: *pretty|field-sizing|@container .*style\(' ` and `grep -c '@supports'`.

---

## 3. View Transitions — the "feels high-tech" lever

The highest perceived-quality-per-line feature currently available, and the reason an MPA can now feel like an SPA (see `performance.md` §4.3).

| Mode | Support |
|---|---|
| **Same-document (SPA)** | `document.startViewTransition(cb)` — Chrome **111+**, Safari **18+**, **Firefox shipped it (Firefox 144+ region; treat as broadly available in 2026 with a progressive-enhancement guard)** `[S]` |
| **Cross-document (MPA)** | Chrome **126+**, **Safari 18.2+**, Firefox **partial/in progress across 146–151** `[S]` |

```css
/* required on BOTH the outgoing and incoming page */
@view-transition { navigation: auto; }

@media (prefers-reduced-motion: reduce) {
  @view-transition { navigation: none; }
}
```

**Correct use:**
- `view-transition-name` must be **unique per snapshot**. **Duplicates on one page abort the transition entirely** — the single most common bug. (`view-transition-name: match-element` exists in newer Chrome to auto-generate uniqueness — **verify support before relying on it** `[?]`.)
- `types` descriptor + `:active-view-transition-type()` lets you vary the animation by navigation direction (forward/back, deeper/shallower).
- Same-origin only.

**Pitfalls to audit for:**
- **Transitions add latency if the next page isn't ready.** Pair cross-document VT with Speculation Rules prerender/prefetch (`performance.md` §3.3) or the "polish" makes the site feel slower.
- The snapshot is a **static image** — animating huge regions is expensive; scope names to the elements that actually move.
- `@view-transition` must be present on **both** pages or nothing happens.

**Accessibility caveat — non-negotiable.** View transitions are motion. They must honour `prefers-reduced-motion: reduce` (the `navigation: none` block above, and `animation: none` on `::view-transition-group(*)` for same-document transitions). Vestibular-disorder triggers include large-area movement, scaling, and parallax — full-page cross-fades and morphing hero images are exactly that. Details and the full motion criteria live in `animation-and-motion.md` and `ada/media-and-motion.md`.
**Check:** DevTools → Rendering → **Emulate CSS media feature `prefers-reduced-motion: reduce`**, then navigate. Any remaining transition is a fail.

**Always progressive enhancement — the site must be fully usable without it.**

---

## 4. Design tokens, scales, and magic values

Three-tier token architecture is the 2026 consensus `[S]`:

**primitive** (`--blue-500: #3b82f6`) → **semantic** (`--color-interactive: var(--blue-500)`) → **component** (`--button-bg: var(--color-interactive)`)

Major design systems all ship tokens-first. The **W3C Design Tokens Format** is stable enough in 2026 that Figma Variables, Style Dictionary, Theo, and Token Transformer all support it. `[S]`

| Rule | Detail | Check |
|---|---|---|
| Spacing scale | 4 px or 8 px base with a consistent ratio: `--space-1: 4px … --space-8: 32px` | `curl -s CSS \| grep -oE '(margin\|padding)[^:]*: *[0-9]+px' \| sort \| uniq -c \| sort -rn` — a long tail of one-off values is the finding |
| Type scale | `clamp()` for fluid type; **no arbitrary `px` at call sites** | `grep -c 'font-size: *[0-9]' CSS` vs `grep -c 'font-size: *var(' CSS` |
| **Magic values** | hardcoded hex, `margin-top: 37px`, `z-index: 99999`, three near-identical greys | `grep -oE '#[0-9a-fA-F]{3,8}' CSS \| sort \| uniq -c \| sort -rn \| wc -l` — report the count of **distinct raw hex literals** outside the token block. Report `z-index` values: `grep -oE 'z-index: *[0-9]+' CSS \| sort -u` |
| `!important` | should be **≈ 0** in application (non-vendor) CSS | `grep -c '!important' CSS` |
| Dark mode | token remapping + `color-scheme` + `light-dark()`, **not a second stylesheet** | `grep -c 'color-scheme\|light-dark(' CSS`; check for a duplicated dark stylesheet in Network. Full treatment, including the both-themes contrast pass: **§10** |

**Smell summary to report as a single number:** *distinct raw hex literals + distinct one-off px spacing values + `!important` count*. Falling that number is a concrete refactor goal; "use design tokens" is not.

---

## 5. Specificity and CSS architecture

- **Cascade layers** make cascade order explicit and end specificity arms races: `@layer reset, base, tokens, components, utilities;` — **layer order beats specificity**.
- `:where()` for **zero-specificity** resets; `:is()` for grouping.
- **Utility-first vs component CSS is a team-scale tradeoff, not a correctness one.** Utilities give a bounded CSS payload and no naming overhead but bloat HTML and resist theming without a token layer. Component CSS reads better and themes cleanly but **grows unboundedly without discipline**.
  **The failure mode to flag is both at once with no boundary** — utility classes *and* a parallel bespoke stylesheet fighting each other. Detect: presence of a utility framework's generated CSS **plus** a large hand-written stylesheet **plus** `!important` in the hand-written one.
- **Median CSS payload is 82 KB.** Anything much above **~150 KB** warrants a Coverage check (`performance.md` §4.5). Unused CSS **< 40%** on first load.
  **Two CSS budgets exist and both are legitimate.** ~150 KB here is the *general* budget. `mobile.md` §3.4 uses **≤ ~50 KB compressed** — the *mobile device-floor* figure, what a mid-tier phone on Slow 4G absorbs without pushing TBT past 200 ms. Audit mobile against 50, the site as a whole against 150, and state which you used. **`performance.md` §6 governs the budget file**; if the numbers disagree in a report, the agreed budget file wins.
- **Check for layers:** `grep -c '@layer' CSS`. Absence is not a fail on its own — an equivalent **documented** ordering strategy is acceptable; *no* strategy plus rising `!important` counts is the fail.

---

## 6. Progressive enhancement — what must work without JS

**The baseline test, and it is one action: turn JavaScript off. What still works?**
DevTools → `Ctrl+Shift+P` → "Disable JavaScript", or `curl -s URL | sed 's/<[^>]*>//g' | wc -w`, or Playwright `browser.newContext({ javaScriptEnabled: false })`.

**Must pass with JS off** (or the gap must be **explicitly documented and justified**):

| Requirement | Check |
|---|---|
| Content renders | Word count of the JS-off render is within ~80% of the JS-on render |
| Links navigate | Every `<a>` in nav has a resolvable `href` |
| Forms submit | `<form action method>` present; server handles the non-JS submit |
| Primary nav works | Navigate two levels deep with JS off |

**How to decide what needs JS at all.** In order:
1. Can the **platform** do it? `<dialog>`, `popover` + anchor positioning, `<details>`, `<input type=date>`, `field-sizing`, `scroll-snap`, CSS scroll effects, native form validation (`required`, `type="email"`, `pattern`, `:user-invalid`).
2. Can **CSS** do it? Transitions, `:has()`, container queries, view transitions.
3. Only then, JS — and then as **enhancement over a working baseline**.

- **Feature-detect, never UA-sniff:** `@supports`, `'x' in window`, `CSS.supports()`. Check: `grep -i 'navigator.userAgent' bundle.js` — any UA branching for feature decisions is a finding.
- **Never `<noscript>` a blank page or an "enable JavaScript" wall for content.** Check: `curl -s URL | grep -A3 '<noscript'`.
- **Run a quarterly platform-reclamation pass** `[S]`: list every JS dependency and check what has since become Baseline and can be handed back to the platform — tooltips → `popover` + anchor positioning; modals → `<dialog>`; auto-grow textarea → `field-sizing`; sticky/scroll effects → CSS; date picker → `<input type=date>`. This is the cheapest recurring JS reduction available, and it directly serves the main-thread problem in `performance.md` §0.

---

## 7. Required states — most sites ship only the happy path

Every async surface needs these, and **their absence is the most reliable quality tell in the whole audit** because they are what separates a demo from a product.

| State | Requirement | How to force it |
|---|---|---|
| **Loading** | Skeleton with the **same dimensions** as the loaded content — otherwise you traded a spinner for CLS. **Delay spinners ~200–300 ms** to avoid flash | DevTools → Network → **Slow 4G** / custom 100 kbps profile |
| **Empty** | Explanatory text **+ an action**, not a blank box | Search for a nonsense string; filter to zero results; log in as a fresh account |
| **Error** | **What failed, what to do, and a retry affordance** | DevTools → Network → **request blocking** on the API origin; or override the response to 500 |
| **Offline** | Service worker offline page at minimum — **don't leave the browser's dinosaur** | DevTools → Network → **Offline**, then reload |
| **Partial / stale** | Needed for optimistic UI and `stale-while-revalidate` | Slow the API and interact before it resolves |
| **Interaction states** | hover, `:focus-visible`, active, disabled, **loading-on-submit with a double-submit guard** | Keyboard-tab the page; double-click submit |

**Report format:** a grid of *surface × state*, with pass/fail per cell. A site with five async surfaces and no error state on any of them is a five-cell failure, not one.
Loading-state mechanics and route-transition patterns → `dynamic-loading.md`. Focus visibility criteria → `ada/`.

---

## 8. Console and network hygiene — fast, high-signal

The cheapest audit in this file and one of the most damning. Collect automatically with Playwright (`page.on('console')`, `page.on('pageerror')`, `page.on('response')`) across load **and** the primary user flow.

| Finding | Severity | Note |
|---|---|---|
| Any console **`error`** | **Automatic fail** | No qualification |
| Console **warnings** | Note and enumerate | Framework key warnings, deprecated API warnings |
| **Hydration mismatch warnings** | **Fail** | A direct CLS **and** INP risk — the DOM is being rebuilt after paint |
| **`console.log` spam in production** | Build hygiene failure | §9 |
| **404/5xx on any subresource** | Fail | images, fonts, JS, CSS, favicon, **source maps** |
| **Mixed content** (any `http://` subresource on an HTTPS page) | Fail | Blocked or upgraded, and always a defect. See `security-and-hygiene.md` |
| **CORS errors, CSP violations, deprecated-feature reports** | Fail / note | CSP violations → `security-and-hygiene.md` |
| **Uncaught promise rejections** | Fail | |
| Duplicate requests for the same asset | Note | |
| Requests to **dev/staging hosts** or `localhost` references | Fail | grep the bundle: `grep -oE 'https?://[a-z0-9.-]*(localhost\|staging\|dev\|test)[a-z0-9.:/-]*'` |

---

## 9. Build and deploy hygiene — exact checks

### 9.1 Source map leakage

**Correct pattern: generate source maps, upload them to your error tracker, delete them from the public bundle.** If you must serve them, gate behind auth or an allowlist.

The canonical cautionary tale `[S]`: the **March 2026** incident in which a **59.8 MB `cli.js.map`** shipped in a public npm package, reconstructing **1,906 TypeScript source files**.

**Check:**
```bash
# 1. every script's sibling .map
curl -s https://example.com | grep -oE 'src="[^"]+\.js"' | cut -d'"' -f2 | while read -r s; do
  u="https://example.com${s#https://example.com}"
  printf '%s %s\n' "$(curl -s -o /dev/null -w '%{http_code}' "${u}.map")" "${u}.map"
done
# 2. the inline pointer
curl -s https://example.com/assets/app.js | tail -c 200 | grep -o 'sourceMappingURL=.*'
```
Any non-404 is a finding.

### 9.2 Exposed VCS, config, and dev artifacts

Probe each; **any 200 is a finding**:
`/.git/config`, `/.git/HEAD`, `/.git/index`, `/.git/refs/heads/main`, `/.env`, `/.env.local`, `/.env.production`, `/.svn/`, `/.DS_Store`, `/config.json`, `/backup.zip`, `/package.json`, `/composer.lock`

```bash
for p in /.git/config /.git/HEAD /.git/index /.git/refs/heads/main /.env /.env.local \
         /.env.production /.svn/ /.DS_Store /config.json /backup.zip /package.json /composer.lock; do
  printf '%s %s\n' "$(curl -s -o /dev/null -w '%{http_code}' "https://example.com$p")" "$p"
done
```
Automate via `../scripts/check_headers.py`.

### 9.3 Secrets in the client bundle

```bash
curl -s https://example.com/assets/app.js | grep -oE 'sk_live[A-Za-z0-9_-]+|AKIA[0-9A-Z]{16}|-----BEGIN[A-Z ]+-----|Bearer [A-Za-z0-9._-]{20,}'
```
Framework env prefixes (`NEXT_PUBLIC_`, `VITE_`, `REACT_APP_`, and equivalents) are **shipped to the browser by design** — enumerate every one in the bundle and verify nothing sensitive uses them.

### 9.4 Minification, dev artifacts, and headers

| Check | Command / method |
|---|---|
| JS/CSS/HTML minified | Newlines-per-KB heuristic: `curl -s FILE \| awk 'END{print NR}'` against byte size; or eyeball for indentation |
| Text compression on all text responses | `curl -sI -H 'Accept-Encoding: br,zstd,gzip' URL \| grep -i content-encoding` (Lighthouse `document-latency-insight` covers this now) |
| No `debugger` statements | `curl -s BUNDLE \| grep -c '\bdebugger\b'` |
| No `console.log` from first-party code | `page.on('console')` in Playwright, filtered to first-party stack frames |
| No `.map` files | §9.1 |
| No TODO/FIXME in shipped strings | `curl -s BUNDLE \| grep -oE 'TODO\|FIXME\|XXX:' \| wc -l` |
| No commented-out blocks in shipped HTML | `curl -s URL \| grep -c '<!--'` |
| No devtools hooks in prod builds | `'__REACT_DEVTOOLS_GLOBAL_HOOK__' in window`, `window.__VUE_DEVTOOLS_GLOBAL_HOOK__`, `window.__NUXT__.dev` etc. |
| No stack traces on error pages | `curl -s https://example.com/definitely-not-a-page` and read the body |
| No verbose `Server` / `X-Powered-By` | `curl -sI URL \| grep -iE 'server:\|x-powered-by'` — see `security-and-hygiene.md` |
| Test IDs in production markup | `$$('[data-testid],[data-test],[data-cy]').length` — **acceptable if intentional and documented**; report the count, don't auto-fail |
| Reproducible builds | lockfile committed, runtime version pinned, CI-enforced lint/typecheck/test/**budget** gates (requires repo access) |

---

## 10. Colour schemes — audit every theme the site can render

**Why this section exists.** Contrast gets audited once, in whichever theme happened to load. A site with a dark theme has **two or more** rendered colour states, and WCAG 1.4.3 / 1.4.11 apply to each of them independently. A dark theme that fails 4.5:1 on body text is the same blocker the light one would be, and an audit run at default OS settings never sees it.

The ratios and the criteria live in `ada/html-core.md` §8. This section is about **how many themes exist, how to force each one, and the defects that only appear when you do.**

### 10.1 Does the site have a theme at all? — detect before you audit

Three mechanisms, frequently combined. Detect all three, because which one is in use decides how you switch themes in §10.2.

| Mechanism | What it looks like | Detection |
|---|---|---|
| **Media query only** — OS-driven, no toggle | `@media (prefers-color-scheme: dark) { … }` | CSSOM scan below. Baseline **widely available since 15 Jul 2022** (Chrome 76, Firefox 67, Safari 12.1 / iOS 13) `[P]` — no guard needed |
| **Class / attribute toggle** — user-driven, usually persisted | `<html class="dark">`, `<html data-theme="night">` | `document.documentElement.className` and `.dataset` before/after clicking the control; `Object.keys(localStorage)` for a `theme` / `mode` / `color-scheme` key |
| **`color-scheme` only** — UA-rendered dark, no author CSS | `:root { color-scheme: light dark }` or `<meta name="color-scheme">` | `getComputedStyle(document.documentElement).colorScheme` |

```js
// Paste in DevTools. Answers "how many colour states does this page have?"
(() => {
  const out = { prefersColorSchemeRules: 0, colorSchemeDecls: 0, lightDarkUses: 0,
                forcedColorsRules: 0, themeSelectors: new Set(), stylesheetsUnreadable: 0 };
  for (const sheet of document.styleSheets) {
    let rs; try { rs = sheet.cssRules; } catch { out.stylesheetsUnreadable++; continue; }
    const walk = list => { for (const r of list) {
      const cond = r.conditionText || r.media?.mediaText || '';
      if (/prefers-color-scheme/.test(cond)) out.prefersColorSchemeRules++;
      if (/forced-colors/.test(cond))        out.forcedColorsRules++;
      if (r.style) {
        if (r.style.getPropertyValue('color-scheme')) out.colorSchemeDecls++;
        if (/light-dark\(/.test(r.cssText))           out.lightDarkUses++;
      }
      // Matches `.dark`, `.theme-night`, `[data-theme="dark"]`, `[data-theme]`.
      // Deliberately does NOT match `.lightbox` or `.highlight`.
      if (r.selectorText &&
          (/(\.|\[[\w-]+[~^$*|]?=\s*["']?)(dark|light|theme|night)\b/i.test(r.selectorText) ||
           /\[data-(theme|scheme|color-scheme|mode)\b/i.test(r.selectorText)))
        out.themeSelectors.add(r.selectorText.slice(0, 80));
      if (r.cssRules) walk(r.cssRules);
    }}; walk(rs);
  }
  return { ...out, themeSelectors: [...out.themeSelectors].slice(0, 20),
    computedColorScheme: getComputedStyle(document.documentElement).colorScheme,
    metaColorScheme: document.querySelector('meta[name="color-scheme" i]')?.content || null,
    osPrefersDark: matchMedia('(prefers-color-scheme: dark)').matches,
    persistedThemeKeys: Object.keys(localStorage).filter(k => /theme|scheme|mode|dark/i.test(k)),
    toggleCandidates: [...document.querySelectorAll(
      '[data-theme-toggle],[aria-label*="theme" i],[aria-label*="dark mode" i],' +
      '[title*="dark mode" i],[class*="theme-toggle" i]')].length };
})()
```

- `stylesheetsUnreadable > 0` means cross-origin CSS you could not scan. Say so in the report rather than concluding "no dark theme".
- **All counters zero and no toggle → the site has one theme.** That is not a defect. Record it, note that the contrast pass therefore covers the whole site, and move on. Only flag it if the profile or brief asked for a dark theme.

### 10.2 Force each state, and re-run the whole colour pass in each

| State | How to force it |
|---|---|
| Light (author) | DevTools → Rendering → **Emulate CSS media feature `prefers-color-scheme: light`**; Playwright `browser.newContext({ colorScheme: 'light' })` |
| Dark (author) | Same, `dark`. **This does not flip a class-toggle theme** unless the site's JS listens for the media-query change |
| Every toggle position | Operate the site's own control, including any explicit "system" option — that is a third state, not a duplicate |
| `forced-colors: active` | DevTools → Rendering → **Emulate CSS media feature `forced-colors: active`**, or Windows **Settings → Accessibility → Contrast themes**. See §10.6 |

**Re-run in each state:** the contrast sweep (`../scripts/audit_a11y.py --contrast`), focus-indicator contrast (3:1, 1.4.11), placeholder / secondary / disabled / error text, form-control borders, `::selection`, chart and data-viz colours, and a section-by-section screenshot for the image defects in §10.4.

**Every finding must name its theme.** *"1.4.3 failure, dark theme only — `.card__meta` `#8a8a8a` on `#1c1c1c` = 3.6:1"* is reproducible. "Low contrast on the meta text" is not.

**The trap that invalidates the whole pass:** the mechanisms disagree. A site that persists `theme=light` renders light *even when the OS says dark*, so emulating the media feature proves nothing and you will report a clean dark theme you never rendered. Clear site data between states, or drive the site's own toggle, and record which you did. Confirm the state actually changed by diffing `getComputedStyle(document.body).backgroundColor` before and after — if it did not move, the emulation did not take.

### 10.3 `color-scheme` — the one declaration that fixes UA-rendered widgets

`color-scheme` tells the UA which schemes the page supports, so it can render **form controls, scrollbars, `::selection`, spellcheck underlines, the canvas default background and `<input type=date>` internals** in the matching scheme. Baseline **widely available since 3 Aug 2024** (Chrome 98, Firefox 96, Safari 13) `[P]` — use it unguarded.

```html
<meta name="color-scheme" content="light dark">   <!-- applies before CSS arrives -->
```
```css
:root { color-scheme: light dark; }               /* both, follow the OS */
.force-light-widget { color-scheme: light; }      /* opt a subtree back out */
```

| Symptom you will see in a dark theme | Cause | Check |
|---|---|---|
| White input/select/textarea boxes in an otherwise dark form; dark placeholder on white | no `color-scheme` | `getComputedStyle(el).colorScheme` on `:root` — expect `light dark` (or `dark`), not `normal` |
| Light scrollbars framing a dark page | same | visual, at a viewport where the page scrolls |
| Flash of white before CSS parses | `color-scheme` set only in a late stylesheet | `curl -s URL \| grep -i 'name="color-scheme"'` — the meta must be in the initial HTML |
| Native date/colour/file pickers rendering light | same | open each picker in the dark state |
| Author CSS re-styled the control but left the **dropdown popup / picker internals** UA-rendered | partial restyle | open every `<select>` and date input in the dark state; the popup is not the control |

**Both the meta and the CSS property.** The meta applies during the initial paint; the CSS property survives the meta being stripped by a framework head manager. Neither is a substitute for author contrast — `color-scheme` fixes the *widgets*, not your palette.

### 10.4 Images and SVG that assume a light background

The most common visible dark-theme defect, and the one no automated contrast tool reports, because the tool measures text against a background, not a black logo against black.

| Asset | Failure | Check |
|---|---|---|
| Transparent PNG/SVG logo drawn in near-black ink | invisible on dark | screenshot each state and diff; or list transparent-background images: `[...document.images].map(i => i.currentSrc).filter(s => /\.(png\|svg\|webp\|avif)$/i.test(s))` and inspect each in the dark render |
| **`<img src="…svg">`** with hardcoded `fill="#000"` | External SVG **cannot inherit page CSS or `currentColor`** — it renders identically in both themes no matter what your stylesheet says. Inline the SVG, or ship a `<picture>` with a `media="(prefers-color-scheme: dark)"` source | `curl -s SVG_URL \| grep -oE 'fill="[^"]*"\|stroke="[^"]*"' \| sort -u` — any literal dark hex is the finding |
| Inline SVG icons with baked `fill` instead of `currentColor` | same, but fixable in place | `$$('svg [fill]:not([fill="currentColor"]):not([fill="none"])').length` |
| Screenshots, diagrams, charts, code samples exported as light-mode raster | a white rectangle punched into a dark page — also a 1.4.3 risk if text inside it is the site's own content | visual, per section, in the dark state |
| Photographs / hero images tuned for a light overlay | overlay gradient inverts, text lands on the bright part | contrast sweep against the **rendered** background per §10.2 |

The correct pattern for a theme-swapped raster asset, which needs no JS and no flash:
```html
<picture>
  <source srcset="logo-dark.avif" media="(prefers-color-scheme: dark)" type="image/avif">
  <img src="logo-light.avif" width="180" height="40" alt="Acme">
</picture>
```
**Caveat to verify, not assume:** `<picture media>` follows the *OS/media-query* state. On a site whose theme is a persisted class toggle, this markup will show the wrong asset whenever the toggle disagrees with the OS. On those sites the asset must be swapped by the same mechanism as everything else (CSS `background-image` under the theme selector, or two `<img>`s toggled by CSS). **Check both mechanisms against each other; a mismatched logo is the tell.**

### 10.5 `light-dark()` — use it, but it is a guard-tier feature

`light-dark(<light-value>, <dark-value>)` picks a value from the element's used colour scheme, so one declaration replaces a media-query block. It **only works where `color-scheme` is set** to include both — that is the most common reason it appears to do nothing.

```css
:root { color-scheme: light dark; }
body   { background: light-dark(#fff, #101014); color: light-dark(#1a1a1a, #ececec); }
```

| | |
|---|---|
| Baseline | **Newly available 13 May 2024** — Chrome/Edge 123, Firefox 120, Safari 17.5 `[P]` |
| Widely available? | **No, not yet as of Aug 2026** — the 30-month mark falls around **Nov 2026**. Do not describe it as widely available in a report |
| Web-platform-test scores | Firefox and Safari are **below 1.0 on stable** `[P]` — a support claim is not a correctness claim; verify the specific values you use in real Safari and real Firefox |
| Audit stance | **Guard it** (`@supports (color: light-dark(#000, #fff))`) or make it a pure enhancement over a working single-theme value. A `light-dark()` value on a non-supporting engine is an invalid declaration and the property falls back to its inherited/initial value — which is frequently unreadable, not merely unstyled |
| Interaction with a class toggle | `light-dark()` reads `color-scheme`, not your `.dark` class. Sites using a class toggle **must also set `color-scheme` on the same selector** (`.dark { color-scheme: dark }`) or every `light-dark()` value stays on its light branch |

**Check:** `curl -s CSS | grep -c 'light-dark('` against `grep -c '@supports'`, then set `color-scheme` to each value in DevTools and confirm the rendered colour actually moves.

### 10.6 `forced-colors` — a third rendering, not a darker second one

Windows Contrast Themes (and equivalents) replace the author palette wholesale with a user-chosen system palette. Baseline **widely available since 12 Mar 2025** (Chrome/Edge 79–89, Firefox 89, Safari 16) `[P]` — but the **web-platform-test scores sit around 0.20–0.32 across every engine** `[P]`, so behaviour differs meaningfully between browsers. Test in more than one.

This is **not** the dark theme. It is orthogonal, and a site can pass both light and dark and still fail here.

| Failure | Why | Check |
|---|---|---|
| Borders, focus rings or dividers drawn with `box-shadow` or `background-image` **vanish** | forced-colors does not repaint those | Emulate `forced-colors: active` and Tab through every section — a focus indicator that disappears is **2.4.7 / 1.4.11** |
| Icons built from `background-image` disappear | same | visual sweep in the forced state |
| State conveyed only by background colour (selected tab, active toggle, error field) collapses | all backgrounds become the system background | operate every stateful control in the forced state |
| `forced-color-adjust: none` used to "fix" the above | It **opts out of the user's chosen palette**, which is the accessibility feature. Legitimate only for content whose colour carries meaning (a colour picker swatch, a brand-accurate logo, a map key) | `curl -s CSS \| grep -n 'forced-color-adjust'` — every occurrence needs a stated reason |

Use the **system colour keywords** (`CanvasText`, `Canvas`, `LinkText`, `ButtonText`, `Highlight`) inside `@media (forced-colors: active)` rather than hex, and re-draw missing borders with `border: 1px solid` / `outline`. Keyboard and screen-reader procedure: `ada/testing.md`.

### 10.7 Third-party and visualisation surfaces

Anything you did not style yourself gets its own pass in every theme: embedded maps, video players, chat widgets, payment iframes, comment systems, and the consent banner (which is often the only white rectangle on a dark page — see `security-and-hygiene.md` §6).

**Visualisation libraries are a known offender.** `viz-libraries.md` records that JSXGraph, function-plot and Plotly **assume a light page** and render with default-light chrome on a dark one until CSS is written for them, and it carries the "theme both ways before shipping" rule and the theme-parity check row. Toggle the theme and re-screenshot **every** visualisation panel; a chart that is legible only in one theme is a 1.4.3 / 1.4.11 finding against that theme, not a styling preference.

---

## 11. RTL and bidirectional text

**Placed here rather than in `mobile.md` on purpose:** the root cause of nearly all RTL breakage is a CSS authoring decision — physical properties where logical ones belong — and it breaks identically at 320 px and at 1920 px. It is an architecture defect, not a viewport defect, and it maps to no WCAG success criterion of its own, so it does not fit `mobile.md`'s Tier 1 / Tier 2 (violation vs recommendation) frame. `mobile.md` §1.10 cross-references back here.

**Load this section when `site.multilingual: true`** — or whenever the site serves, or intends to serve, any right-to-left script: Arabic (`ar`), Hebrew (`he`), Persian/Farsi (`fa`), Urdu (`ur`), Pashto (`ps`), Sorani Kurdish (`ckb`), Divehi (`dv`), Syriac (`syr`), N'Ko (`nqo`), Yiddish (`yi`). **Skip it entirely for a single-locale LTR site.** Confirm from the profile and the rendered page — `link[rel=alternate][hreflang]`, the `lang` attribute, and the presence of a language switcher — not from an assumption about the client's market.

### 11.1 `dir` correctness — markup, not CSS

| Rule | Why | Check |
|---|---|---|
| `dir` on `<html>`, paired with `lang` | Direction is a property of the content. `<html lang="ar" dir="rtl">` | `document.documentElement.dir` and `.lang` on every locale's URL — a locale page serving `dir="ltr"` is the finding |
| `dir` set in **HTML**, not only CSS `direction` | CSS may not load, may be overridden by a user stylesheet, and does not travel with the content when it is copied, syndicated or read by a tool that ignores CSS | `getComputedStyle(html).direction === 'rtl'` **and** `html.getAttribute('dir') === 'rtl'` — the second missing is the finding |
| `dir` on the element whose *content* is in that direction, not blanket-applied | A single `dir="rtl"` on `<body>` flips embedded LTR blocks (code samples, tables of Latin data, English quotations) that should stay LTR | `$$('pre, code, [lang]:not([lang^="ar"]):not([lang^="he"])').map(e => [e.tagName, getComputedStyle(e).direction])` |
| `dir="auto"` on user-generated strings of unknown direction | Uses the first strong directional character to decide. Correct for names, comments, search queries, filenames | `$$('input, textarea, [contenteditable]').filter(e => !e.getAttribute('dir'))` on any surface that renders user text |
| Language switcher itself works in both directions | It is the one control every RTL user touches first | operate it in both directions and confirm the `dir` and `lang` both change |

Direction and language are independent: an English quotation inside an Arabic page keeps `lang="en" dir="ltr"`; an Arabic phrase inside an English page needs `lang="ar" dir="rtl"`. Setting one and not the other is the routine bug.

### 11.2 Logical properties — the actual root cause

An RTL layout does not break because of RTL. It breaks because the stylesheet said `left` when it meant `start`. Every physical property below is a latent RTL bug; every logical one flips for free. Logical properties are long-settled Baseline (§2.1).

| Physical — flags an RTL risk | Logical — flips automatically |
|---|---|
| `margin-left` / `margin-right` | `margin-inline-start` / `margin-inline-end` |
| `padding-left` / `padding-right` | `padding-inline-start` / `padding-inline-end` |
| `left` / `right` (positioned elements) | `inset-inline-start` / `inset-inline-end` |
| `border-left` / `border-right` | `border-inline-start` / `border-inline-end` |
| `text-align: left \| right` | `text-align: start \| end` |
| `float: left \| right` | `float: inline-start \| inline-end` |
| `border-top-left-radius` etc. | `border-start-start-radius` etc. |
| `width` / `height` on a flow-sensitive box | `inline-size` / `block-size` |
| `overflow-x` / `overflow-y` | `overflow-inline` / `overflow-block` |

**Report the ratio, not an adjective** — the same discipline as the div-soup metric in §1.2:

```js
// Counts physical vs logical inline-axis declarations across readable stylesheets.
(() => {
  // Property NAMES only (the CSSOM has already expanded shorthands), so
  // `border-left-width` and `margin-left` both count, and `inline-start` never does.
  const PHYS = /(^|-)(left|right)(-|$)|^(float|clear)$/;
  const LOGI = /(^|-)(inline|block)(-|$)|start-start|start-end|end-start|end-end/;
  let physical = 0, logical = 0, unreadable = 0;
  const physicalSamples = [];
  for (const sheet of document.styleSheets) {
    let rs; try { rs = sheet.cssRules; } catch { unreadable++; continue; }
    const walk = list => { for (const r of list) {
      if (r.cssRules) { walk(r.cssRules); continue; }
      if (!r.style) continue;
      for (const p of r.style) {
        if (LOGI.test(p)) { logical++; continue; }
        if (PHYS.test(p)) {
          physical++;
          if (physicalSamples.length < 25)
            physicalSamples.push(r.selectorText + ' { ' + p + ': ' + r.style.getPropertyValue(p) + ' }');
        }
      }
      const ta = r.style.getPropertyValue('text-align');
      if (/^(left|right)$/.test(ta)) { physical++;
        if (physicalSamples.length < 25) physicalSamples.push(r.selectorText + ' { text-align: ' + ta + ' }'); }
    }}; walk(rs);
  }
  return { physical, logical, unreadableStylesheets: unreadable,
           logicalRatio: +(logical / (physical + logical || 1)).toFixed(2), physicalSamples };
})()
```

Also physical, and missed by the scan above because they take numeric values:

- **`transform: translateX(±n)`** — slide-in drawers, off-canvas menus, carousel tracks. Does **not** flip. Must be conditioned on direction, or expressed as a logical inset.
- **`scrollLeft` arithmetic** — RTL scroll coordinates are engine-dependent and have historically been negative or inverted. Any carousel, tab strip or virtualised list doing `el.scrollLeft += n` is a probable RTL defect. Prefer `scrollIntoView()` / `scrollBy({ left })` with `dir` set, and **verify by scrolling, not by reading the code.**
- **`background-position: left/right`**, `linear-gradient(to right, …)`, `box-shadow` x-offsets, `text-shadow` x-offsets, `clip-path` polygons.
- **`::before` / `::after` decorative content** positioned with `left`/`right` — bullet glyphs, chevrons, quote marks.
- **`transform: scaleX(-1)` applied to a container to "make it RTL"** — an anti-pattern. It mirrors the *text glyphs and images* too. Any occurrence is a finding.

### 11.3 Icons and directional affordances

Mirror what expresses direction of *flow*. Do not mirror what represents a physical object, a fixed convention, or a real-world orientation. `[S]` — this is platform convention (Material and Apple bidirectionality guidance), not a normative standard; report deviations as recommendations unless the result is unusable.

| Must mirror | Must **not** mirror |
|---|---|
| Back / forward, previous / next, breadcrumb chevrons | Media playback controls — play, fast-forward, rewind, the scrub timeline `[S]` |
| Carousel and pagination arrows | Clock faces, and anything reading a real dial |
| Indent / outdent, list bullets and numbering side, tree-view disclosure triangles | Checkmarks, and most brand logos |
| Progress trackers, steppers, breadcrumb order, slider fill and direction | Photographs and images of real objects or people |
| Off-canvas drawer entry side; nav order; the "close" X position | Numerals themselves — Western Arabic digits render LTR inside RTL text |
| Text alignment, table column order, form label/field order, tooltip and popover flip side | Phone numbers, URLs, email addresses, code, version strings — all LTR runs |
| Sort indicators, resize handles, drag affordances | Musical notation; chemical and mathematical notation |

**Check:** screenshot the same section in both directions, place them side by side, and walk the icon inventory. `$$('svg, [class*="icon"]')` gives you the list; the judgement is human. An arrow pointing the wrong way in an RTL locale is a comprehension failure, not a cosmetic one.

### 11.4 Mixed-direction content — where the subtle bugs live

The Unicode bidirectional algorithm resolves direction per paragraph using the first strong directional character, then re-orders. Neutral characters — punctuation, spaces, digits, brackets, `@`, `/`, `#` — take direction from what surrounds them. That is where the bugs are, and they are usually in **user-generated or database-sourced strings**, which is exactly the content nobody screenshots.

| Symptom | Cause | Fix |
|---|---|---|
| A Latin product name, username or URL at the end of an RTL sentence jumps to the wrong side | the interpolated string's direction leaks into the surrounding run | wrap the interpolated value in **`<bdi>`** — it isolates the value's direction from its context and is the correct element for *any* string of unknown direction |
| Trailing `?`, `!`, `.` or `)` lands at the wrong end of the line | neutral character resolved against the wrong run | `<bdi>` around the variable part, or CSS `unicode-bidi: isolate` |
| A phone number, IP address or version string reads scrambled | digits and separators are neutral/weak | `<span dir="ltr">` or `unicode-bidi: isolate` on the run |
| Concatenated translated strings break only in one locale | string built by concatenation rather than by a placeholder | flag the concatenation; it is a localisation defect, not a CSS one |

- **`<bdi>`** — isolate, direction auto-detected. **The default choice for interpolated content.**
- **`<bdo dir="…">`** — a hard *override*. Rare and usually wrong; it is for displaying text whose direction must be forced, not for fixing layout.
- **CSS `unicode-bidi`** — `isolate` (equivalent to `<bdi>`), `isolate-override`, `plaintext` (auto-detect per paragraph; useful on `<pre>` and message bodies). Prefer the HTML element where you control the markup; the CSS is for content you do not.

**Check:** find every place a variable is interpolated into a translated sentence and confirm it is isolated. From the outside: `$$('bdi').length` against the number of visibly interpolated values, plus a rendered read of the pages carrying user-generated content — comments, reviews, search results, profile names, file listings.

### 11.5 How to verify — force RTL and re-screenshot

RTL breakage is a **visual** finding. The procedure is to render the site in RTL and compare.

```js
// Crude but effective: flips the document without translating it.
(() => { const de = document.documentElement;
  const before = de.getAttribute('dir');
  de.setAttribute('dir', de.getAttribute('dir') === 'rtl' ? 'ltr' : 'rtl');
  return { was: before, now: de.getAttribute('dir'),
           scrollWidth: de.scrollWidth, clientWidth: de.clientWidth,
           newHorizontalOverflow: de.scrollWidth > de.clientWidth + 1 };
})()
```

```js
// Playwright: render both directions and screenshot per section.
// await page.addInitScript(() => document.documentElement.setAttribute('dir','rtl'));
```

**State the limitation in the report:** forcing `dir="rtl"` on English text is a *layout* probe, not a localisation test. It finds hardcoded physical CSS, unmirrored icons and broken scroll maths. It does not find translation length overflow, font fallback for the target script, or line-breaking bugs. **If a real RTL locale exists, audit that locale's URLs directly** and say which method produced each finding.

Walk this list against a side-by-side screenshot pair, per section (`mobile.md` §5.0 gives you the section list):

- [ ] Horizontal overflow appears in RTL that was absent in LTR — run `mobile.md` §1.2 scoped per section in both directions
- [ ] Any element that did **not** move: hardcoded `left`/`right`/`translateX`
- [ ] Asymmetric padding/margin now on the wrong side — icon-to-label gaps, list indents, card gutters
- [ ] Text still left-aligned inside a flipped container (`text-align: left` rather than `start`)
- [ ] Directional icons pointing the wrong way, per the §11.3 table
- [ ] Border radii, shadows and gradients still weighted to the old side
- [ ] Off-canvas menus, drawers and modals entering from the wrong edge, or now off-screen
- [ ] Carousels, tab strips and sliders that will not scroll, scroll backwards, or land on the wrong item
- [ ] Sticky/fixed chrome, close buttons, FABs and toasts on the wrong side or overlapping
- [ ] Form label/field/hint order, checkbox-to-label side, and error-message alignment
- [ ] Tables: column order, and numeric columns that must stay LTR
- [ ] Scrollbar side, and any custom scrollbar pinned physically
- [ ] Focus order still matches the *visual* order after the flip (**WCAG 1.3.2 / 2.4.3** — this one is a real SC, and reordering by `flex-direction: row-reverse` or `order` is the usual cause; `ada/html-core.md` §9)
- [ ] Mixed-direction strings render correctly, per §11.4
- [ ] Fonts: the target script has a real font in the stack, with a `unicode-range`-scoped `@font-face` rather than a Latin font falling back to a system default at a mismatched size

---

## 12. Testable checklist

`[AUTO]` = fully automatable via Playwright / curl / Lighthouse JSON.

### A. HTML & semantics
- [ ] `[AUTO]` `<html lang>` set; single `<h1>`; heading levels sequential; `<main>`/`<nav>`/`<header>`/`<footer>` present
- [ ] `[AUTO]` **Zero** `onclick` handlers on non-interactive elements; every clickable is `<button>` or `<a href>`
- [ ] `[AUTO]` Every nav link has a real, resolvable `href` (middle-click / ⌘-click works)
- [ ] `[AUTO]` **Semantic-element ratio recorded** (div-soup metric, §1.2) — report the number, not an adjective
- [ ] `[AUTO]` Forms use `<form>` with `action`/`method`, `<label for>`, correct `type`, and `autocomplete`
- [ ] `[AUTO]` Total DOM element count under threshold (**~1,500**)

### B. Platform features & CSS quality
- [ ] `[AUTO]` Every **non-Baseline-widely-available** feature has a `@supports` guard or degrades harmlessly (§2)
- [ ] **Core task completes in current Safari and current Firefox**, not just Chrome — run the site's single primary task end to end (the one named in `audience.success_metric`) in each browser, on a real build, and record browser name + version and the step at which it diverges. Chrome DevTools device emulation is **not** a substitute: it is Blink either way. On Linux, use a real macOS/iOS device or a hosted lab for WebKit; `epiphany`/WebKitGTK is close but not Safari. Pass: the task completes in all three, with no console error introduced by the non-Chrome engine (§D)
- [ ] `[AUTO]` `prefers-reduced-motion` honoured (view transitions, scroll-driven animations, autoplay video, parallax)
- [ ] `[AUTO]` If view transitions are used: `@view-transition` present on **both** pages; **no duplicate `view-transition-name`**; reduced-motion opt-out present; paired with prerender/prefetch
- [ ] `[AUTO]` Colour, spacing, radius, shadow, and type values come from custom properties — **count raw hex/px literals in component CSS as a smell metric**
- [ ] `[AUTO]` `!important` count in application (non-vendor) CSS ≈ **0**
- [ ] `[AUTO]` `z-index` values come from a defined scale; **no values > 100 outside the token set**
- [ ] `[AUTO]` Total CSS ≤ **~150 KB** compressed (general budget; **≤ ~50 KB** on the mobile device floor per `mobile.md` §3.4 — `performance.md` §6 governs the budget file); unused CSS **< 40%** on first load
- [ ] `[AUTO]` Cascade layers (`@layer`) **or** an equivalent documented ordering strategy in use

### C. Resilience & states
- [ ] `[AUTO]` **With JS disabled:** content renders, nav works, primary form submits — or the gap is **explicitly documented and justified**
- [ ] Loading skeletons match final content dimensions (**no CLS on state change**) — throttle the network (DevTools → Network → Slow 4G) so the skeleton is visible for several seconds, then measure rather than eyeball it: with Performance recording, or with a live CLS observer, `new PerformanceObserver(l => l.getEntries().filter(e => !e.hadRecentInput).forEach(e => console.log(e.value, e.sources))).observe({type:'layout-shift', buffered:true})` — trigger the load and read any shift attributed to the skeleton's container. Pass: **CLS contribution 0** across the skeleton → content swap. Screenshot the skeleton and the settled state at the same viewport and diff the bounding boxes if the observer reports nothing
- [ ] **Empty state, error state, and offline behaviour verified for each async surface** (surface × state grid) — enumerate every surface that fetches (list, search, feed, dashboard panel, autocomplete) and force each state deliberately: **empty** — use a query or account with no results, or return `[]` via DevTools → Network → request → Override content; **error** — Network → Block request URL, or override the response to a 500; **offline** — Network → Offline preset, then reload *and* interact. Pass: each cell of the grid shows purposeful copy with a recovery path — never an infinite spinner, a blank region, or a raw stack trace. Record the grid in the report; an unfilled cell is "not verified", not "passes"
- [ ] `[AUTO]` Offline: service worker offline fallback present, or its absence is a conscious choice
- [ ] Slow-3G run: page is **usable, not a spinner-forever**
- [ ] No UA-sniffing used for feature decisions (`grep navigator.userAgent`)

### D. Console, network & build
- [ ] `[AUTO]` **Zero** console errors on load **and** during the primary user flow
- [ ] `[AUTO]` Console warnings enumerated; **hydration mismatches = fail**
- [ ] `[AUTO]` Zero `console.log` from first-party code in production
- [ ] `[AUTO]` Zero 404/5xx subresources; zero mixed content; zero uncaught promise rejections; zero CSP violations
- [ ] `[AUTO]` **No `.map` files reachable** — fetch every `<script src>` + `.map` and check for a non-404
- [ ] `[AUTO]` `/.git/config`, `/.git/HEAD`, `/.env`, `/.env.production`, `/.DS_Store` **all return 404**
- [ ] `[AUTO]` No secrets (`sk_live`, `AKIA`, `-----BEGIN`, bearer tokens) in the built bundle
- [ ] `[AUTO]` JS/CSS/HTML minified; no `debugger` statements; no `X-Powered-By` / verbose `Server` header
- [ ] `[AUTO]` No references to dev/staging hosts or `localhost` in shipped code

### E. Colour schemes (§10)

Run **only after** §10.1 has established how many colour states exist, and repeat every colour check once per state.

- [ ] `[AUTO]` Theme inventory recorded — §10.1 snippet — report the number of colour states and how each is switched, or state that the site has one theme
- [ ] `[AUTO]` **Contrast sweep run in every theme**, not once — `../scripts/audit_a11y.py --contrast` per state — zero 1.4.3 / 1.4.11 failures **in each** (`ada/html-core.md` §8)
- [ ] Theme actually changed before each pass — `getComputedStyle(document.body).backgroundColor` differs between states — guards against a persisted toggle overriding the emulated media query
- [ ] `[AUTO]` `color-scheme` declared on `:root` **and** as `<meta name="color-scheme">` in the initial HTML — §10.3 checks — computed value is not `normal`
- [ ] Native form controls, `<select>` popups, date/colour/file pickers, scrollbars and `::selection` render in the active scheme — open each in the dark state — no white-on-dark widgets
- [ ] Logos, inline SVG, external `<img src="…svg">`, diagrams and screenshots legible in every theme — §10.4, section screenshots per state — no black-on-black, no white raster panels
- [ ] `[AUTO]` `light-dark()` guarded or degrading harmlessly, **and** `color-scheme` set on the same selector as any class toggle — §10.5 — `grep -c 'light-dark('` has a matching `@supports` or a working fallback value
- [ ] Focus indicators, state colours and icon borders survive `forced-colors: active` — §10.6, emulate and Tab every section — nothing vanishes; **2.4.7 / 1.4.11**
- [ ] `[AUTO]` Every `forced-color-adjust: none` has a stated reason — `grep -n 'forced-color-adjust'` — zero unexplained occurrences
- [ ] Third-party surfaces themed: embeds, players, chat, payment iframes, consent banner — visual, per state — no unstyled light rectangle on a dark page
- [ ] Every visualisation panel legible in both themes — toggle and re-screenshot each panel (`viz-libraries.md`) — no default-light chrome

### F. RTL and bidirectional text (§11) — only when `site.multilingual: true`

- [ ] `[AUTO]` `<html dir>` **and** `lang` correct on every locale's URL — §11.1 checks — `dir` present in the HTML, not only via CSS `direction`
- [ ] `[AUTO]` Logical-property ratio recorded — §11.2 snippet — report `logicalRatio`; a low ratio with many `physicalSamples` **is** the finding
- [ ] `[AUTO]` No `transform: scaleX(-1)` used to fake RTL; `translateX`, `scrollLeft` arithmetic and directional gradients accounted for — §11.2 — zero occurrences of the scaleX anti-pattern
- [ ] Site re-rendered with `dir="rtl"` and screenshotted **per section**, both directions — §11.5 procedure — the 15-item walk-through completed
- [ ] No horizontal overflow introduced by the flip — `mobile.md` §1.2 scoped per section, in both directions — `scrollWidth ≤ clientWidth + 1`
- [ ] Directional icons mirror correctly; non-directional ones do not — §11.3 table, side-by-side screenshots — advisory `[S]`, escalate only where meaning is lost
- [ ] Off-canvas menus, drawers, modals, carousels, sliders and tab strips operate correctly in RTL — behavioural, not visual — each opens from the correct edge and scrolls the correct way
- [ ] Interpolated and user-generated strings isolated with `<bdi>` / `unicode-bidi: isolate` — §11.4 — names, URLs, phone numbers and version strings read correctly inside RTL sentences
- [ ] Focus order still matches visual order after the flip — Tab through each section — **WCAG 1.3.2 / 2.4.3**
- [ ] A real font covers the target script, scoped by `unicode-range` — CSSOM scan of `@font-face` — no silent fallback to a system default at a mismatched size
- [ ] Findings state **which** method produced them — forced-`dir` probe vs a real localised URL — the probe cannot find translation-length or font-fallback defects

Headers, TLS, CSP, cookies and consent → `security-and-hygiene.md`. Metrics and budgets → `performance.md`. Contrast ratios and criteria → `ada/html-core.md` §8.
