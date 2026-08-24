# HTML core: structure, images, colour, focus, ARIA

**Covers:** document skeleton, landmarks, headings, lists, images and the alt decision tree, links vs buttons, data tables, colour and contrast ratios, focus management, zoom/reflow/text spacing, ARIA rules and live regions.
**Load when:** every HTML audit. This is the default working reference.
**Siblings:** forms → `html-forms.md` · video/audio/motion → `media-and-motion.md` · WCAG 2.2 additions → `wcag22-new.md`.

Roughly **70% of WCAG failures in HTML come from four things**: missing or wrong semantics, missing labels, insufficient contrast, and broken keyboard operation. Get the foundation right and the criteria list mostly takes care of itself.

---

## 1. Document skeleton

```html
<!DOCTYPE html>
<html lang="en">            <!-- 3.1.1 Language of Page -->
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <!-- Never: user-scalable=no or maximum-scale=1 → breaks 1.4.4 -->
  <title>Renew a vehicle registration — City of Example</title>
  <!-- 2.4.2: unique, descriptive, most-specific-information-first -->
</head>
<body>
  <a class="skip-link" href="#main">Skip to main content</a>  <!-- 2.4.1 -->

  <header>
    <nav aria-label="Primary">…</nav>       <!-- 2.4.1, 3.2.3 -->
  </header>

  <main id="main" tabindex="-1">            <!-- 2.4.1 -->
    <h1>Renew a vehicle registration</h1>   <!-- exactly one h1 -->
    …
  </main>

  <footer>…</footer>
</body>
</html>
```

The skip link must be **visible on focus**. A skip link permanently hidden with `display:none` is removed from the tab order entirely — a common and silent failure.

```css
.skip-link {
  position: absolute; left: -9999px;
  background: #fff; color: #000; padding: .75rem 1rem; z-index: 100;
}
.skip-link:focus { left: 0; top: 0; }
```

| SC | Level | Implementation |
|---|---|---|
| 3.1.1 Language of Page | A | `<html lang>` |
| 3.1.2 Language of Parts | AA | `lang` on foreign-language passages |
| 2.4.2 Page Titled | A | Unique descriptive `<title>`, specific-to-general |
| 2.4.1 Bypass Blocks | A | Skip link **and** landmarks |
| 1.4.4 Resize Text | AA | Relative units; no `user-scalable=no` |

**How to verify:** `curl -s <url> | head -40` and read the `<head>`. Then in DevTools console: `document.documentElement.lang`, `document.title`, `document.querySelectorAll('h1').length`. Tab once from the address bar — the first stop should be the skip link and it should become visible.

---

## 2. Landmarks

Use native elements; they carry implicit ARIA roles for free.

| Element | Implicit role | Notes |
|---|---|---|
| `<header>` (page-level) | `banner` | Only when **not** nested in `article`/`section`/`main` |
| `<nav>` | `navigation` | Label each one if there is more than one: `aria-label="Primary"` |
| `<main>` | `main` | **Exactly one per page** |
| `<aside>` | `complementary` | |
| `<footer>` (page-level) | `contentinfo` | |
| `<form aria-label="…">` | `form` | Only becomes a landmark when named |
| `<section aria-label="…">` | `region` | Only becomes a landmark when named |
| `<search>` | `search` | Newer element; `role="search"` on a wrapper is the fallback |

Do not add redundant roles (`<nav role="navigation">`). Do not label a landmark with its own type — `aria-label="Navigation"` on a `<nav>` announces "navigation navigation".

**How to verify:** DevTools → Elements → Accessibility pane → Full-page accessibility tree, or run in console:
`[...document.querySelectorAll('header,nav,main,aside,footer,section[aria-label],form[aria-label],[role]')].map(e=>[e.tagName,e.getAttribute('role'),e.getAttribute('aria-label')])`.
Exactly one `main`. Multiple `nav`s each with a distinct label. In NVDA, press <kbd>D</kbd> to cycle landmarks and confirm the announced names make sense.

---

## 3. Headings

- Exactly one `<h1>`, matching the page's main purpose.
- **Never skip levels going down.** h2 → h4 is a failure of 1.3.1 in practice.
- Headings describe the section that follows (2.4.6 Headings and Labels, AA).
- Never use a heading element for visual size. Never fake a heading with `<p class="big-bold">`.
- `<div role="heading" aria-level="3">` is a legitimate last resort in constrained component systems, but native is always better.

**How to verify:** console —
`[...document.querySelectorAll('h1,h2,h3,h4,h5,h6,[role=heading]')].map(h=>h.getAttribute('aria-level')||h.tagName[1]).join(' ')`
— read the sequence for gaps. In NVDA press <kbd>H</kbd> repeatedly; in VoiceOver use the rotor (<kbd>VO</kbd>+<kbd>U</kbd>) → Headings. If the heading list alone does not describe the page, 2.4.6 fails.

---

## 4. Lists, quotes, emphasis

```html
<ul> / <ol> / <dl>          <!-- real lists, not • characters in <p> -->
<blockquote cite="…">       <!-- not indentation -->
<em> <strong>               <!-- semantic, not <i>/<b> for meaning -->
<abbr title="Web Content Accessibility Guidelines">WCAG</abbr>
<code> <pre> <kbd> <samp>
```

Fake lists are a 1.3.1 failure: a screen reader announces "list, 5 items" for a real list and nothing at all for five paragraphs starting with a bullet character.

**How to verify:** search the rendered HTML for bullet glyphs outside `<li>`:
`curl -s <url> | grep -nE '<p[^>]*>\s*[•·▪◦*-]\s'`. Also check `document.querySelectorAll('ul li, ol li').length` against the number of visible bullets on screen.

---

## 5. Images — 1.1.1 Non-text Content (A)

```html
<!-- Informative -->
<img src="chart.png" alt="Applications rose from 1,200 in 2024 to 4,800 in 2026.">

<!-- Decorative — empty alt, NOT a missing alt attribute -->
<img src="divider.svg" alt="">

<!-- Functional (image is the control) — alt describes the ACTION -->
<a href="/cart"><img src="cart.svg" alt="Shopping cart, 3 items"></a>

<!-- Complex — short alt + long description -->
<figure>
  <img src="flow.png" alt="Permit approval workflow, described below.">
  <figcaption id="flowdesc">
    Applications enter intake, are routed to zoning review…
  </figcaption>
</figure>

<!-- Inline SVG -->
<svg role="img" aria-labelledby="t1"><title id="t1">Downward trend</title>…</svg>
<svg aria-hidden="true" focusable="false">…</svg>   <!-- decorative icon -->
```

### Alt decision tree

| Ask | Then |
|---|---|
| Is the image the **only content of a link or button**? | alt = the **action or destination**, not the picture. `alt="Shopping cart, 3 items"`. |
| Does it carry **information not in the surrounding text**? | alt = that information, as concisely as it can be stated. |
| Is it a **chart, graph or infographic**? | Short alt naming what it is + the underlying **data** available adjacent as a table or text summary. `alt="Bar chart of revenue"` **fails 1.1.1**. |
| Does it **contain text**? | The text appears in the alt **verbatim** — and the image itself usually violates 1.4.5 Images of Text anyway (see `media-and-motion.md`). |
| Is it **purely decorative**, or redundant with adjacent text? | `alt=""` (or `role="presentation"`, or a CSS background). |
| Is it a **CAPTCHA**? | Text alternative describing its purpose, plus an alternative modality. See `html-forms.md` and `wcag22-new.md` (3.3.8). |
| Is it `<canvas>`, an icon font, or a `background-image` carrying meaning? | Needs an explicit text alternative — none is supplied automatically. |

**Rules that actually matter:**
- Convey the *function or information*, not the appearance.
- Don't start with "Image of" / "Graphic of" — the role is already announced.
- `alt=""` and a **missing** `alt` are completely different. Missing `alt` causes screen readers to announce the filename.

**How to verify:** `../../scripts/audit_a11y.py --images <url>` lists every image with its computed accessible name and flags missing/filename-like/`alt` starting with "image of". Manually:
`[...document.images].filter(i=>!i.hasAttribute('alt')).map(i=>i.src)` — must be empty. Then read every non-empty alt against the tree above; that part cannot be automated.

---

## 6. Links vs buttons

| Use | When |
|---|---|
| `<a href>` | Navigates somewhere. Appears in the links list. Activated by **Enter**. |
| `<button>` | Performs an action. Activated by **Enter and Space**. |

`<div onclick>` fails **2.1.1** (keyboard), **4.1.2** (name/role/value) and usually **2.4.7** (focus visible) simultaneously. Making it work needs `role`, `tabindex="0"`, and both keydown handlers — three chances to get it wrong. Use the native element.

Link text must make sense out of context (2.4.4 Link Purpose (In Context), A; 2.4.9 Link Purpose (Link Only), AAA):

```html
<!-- Bad -->
Read the annual report <a href="/report.pdf">here</a>.
<!-- Good -->
<a href="/report.pdf">Annual report 2026 (PDF, 2.4 MB)</a>
```

Flagging file type and size on document links is not a WCAG requirement, but it prevents a change-of-context surprise and is required by several public-sector policies.

**How to verify:**
- Clickables that are not links or buttons: `[...document.querySelectorAll('[onclick]')].filter(e=>!/^(A|BUTTON|INPUT|SELECT|TEXTAREA)$/.test(e.tagName))` — must be empty.
- Empty accessible names: `[...document.querySelectorAll('a,button')].filter(e=>!e.textContent.trim() && !e.getAttribute('aria-label') && !e.getAttribute('aria-labelledby') && !e.querySelector('img[alt]:not([alt=""])'))`.
- Ambiguous link text: in NVDA press <kbd>Insert</kbd>+<kbd>F7</kbd> for the links list; in VoiceOver use the rotor → Links. Any entry reading "here", "read more", "click here", or a bare URL fails 2.4.4 in the list context.

---

## 7. Data tables

```html
<table>
  <caption>Permit fees by category, 2026</caption>
  <thead>
    <tr>
      <th scope="col">Permit type</th>
      <th scope="col">Base fee</th>
      <th scope="col">Expedited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">Residential remodel</th>
      <td>$450</td>
      <td>$780</td>
    </tr>
  </tbody>
</table>
```

- `scope="col"` / `scope="row"` on **every** header cell.
- Irregular tables with multi-level headers: `id` on headers and `headers="…"` on data cells.
- `<caption>` gives the table an accessible name.
- **Never use tables for layout.** If forced to, `role="presentation"`.
- Responsive tables: do **not** `display:block` your way out — that destroys the row/column relationships. Use horizontal scroll with `tabindex="0"` and an accessible name on the scroll container so keyboard users can reach it.

**How to verify:** console —
`[...document.querySelectorAll('table')].map(t=>({caption:!!t.caption, ths:t.querySelectorAll('th').length, scoped:t.querySelectorAll('th[scope]').length}))`
— `ths` must equal `scoped`. Then read one row with a screen reader (NVDA: <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+arrows) and confirm each cell is announced with its column and row header.

---

## 8. Colour and contrast

| Requirement | Ratio | SC |
|---|---|---|
| Normal text (<18pt / <14pt bold) | **4.5:1** | 1.4.3 Contrast (Minimum), AA |
| Large text (≥18pt / 24px, or ≥14pt / 18.66px bold) | **3:1** | 1.4.3 AA |
| UI component boundaries, states, focus indicators | **3:1** | 1.4.11 Non-text Contrast, AA |
| Meaningful parts of graphics/icons | **3:1** | 1.4.11 AA |
| Enhanced: normal text / large text | **7:1 / 4.5:1** | 1.4.6 Contrast (Enhanced), AAA |

Exempt: disabled controls, pure decoration, logos, incidental text in photographs.

Test against the **actual rendered background**, including gradients and image overlays — the computed `background-color` of a parent is not necessarily what is behind the glyphs.

**Contrast must be verified in every theme the site can render.** These ratios are not a property of the site; they are a property of a *rendered colour state*. A site with a dark theme has two or more such states, and 1.4.3 / 1.4.11 apply to each independently — a dark theme failing 4.5:1 on body text is the same AA failure the light one would be. Auditing whichever theme happened to load at your OS setting tests one state and reports on all of them. Enumerate the states, force each, and re-run this whole section per state; **every contrast finding must name its theme** or it is not reproducible. Detection, the forcing recipes, the `color-scheme` widget defects, and the images-that-assume-a-light-page failures are in `../code-quality.md` §10. Windows Contrast Themes are a **third** rendering, not a darker second one — §10.6 there, and `testing.md` for the keyboard procedure.

**Colour is never the sole carrier of meaning (1.4.1 Use of Color, A):** red text for errors needs an icon or the word "Error"; a red/green status dot needs a label; a line chart with three coloured lines needs distinct dash patterns or direct labels. Links inside body text must be underlined, or have **≥3:1 contrast against the surrounding text** *plus* a non-colour cue on hover and focus.

**How to verify:** `../../scripts/audit_a11y.py --contrast <url>` for the automatable sweep, then DevTools → Elements → pick the element → the colour swatch in Styles shows the computed contrast ratio and AA/AAA badges. For text over images or gradients, use the **Colour Contrast Analyser** (TPGi) eyedropper on a screenshot — the DevTools number is wrong there. For 1.4.1, grayscale the page (DevTools → Rendering → Emulate vision deficiencies → Achromatopsia) and confirm every state, error and chart series is still distinguishable.

---

## 9. Focus management

```css
/* Never do this: */
:focus { outline: none; }

/* Do this: */
:focus-visible {
  outline: 3px solid #005fcc;
  outline-offset: 2px;         /* helps 1.4.11 and 2.4.13 */
}
/* Fallback for older browsers */
:focus:not(:focus-visible) { outline: none; }
```

| Rule | Criterion |
|---|---|
| Every interactive element has a visible focus indicator | 2.4.7 Focus Visible, AA |
| Tab order preserves meaning and operability; DOM order matches visual order | 2.4.3 Focus Order, A |
| `tabindex` > 0 breaks tab order. **Only ever use `0` and `-1`.** | 2.4.3 |
| Focus can always be moved away by keyboard | 2.1.2 No Keyboard Trap, A |
| Receiving focus never triggers a change of context | 3.2.1 On Focus, A |
| Focused element is never entirely covered by sticky headers, cookie bars or chat bubbles | 2.4.11 Focus Not Obscured (Minimum), AA — see `wcag22-new.md` |

- **Client-side routing:** after a route change, focus stays on the old (now removed) element and lands on `<body>`, so the next Tab restarts at the top of the page and the screen reader announces nothing. **Symptom:** activate an in-app link, press Tab — if focus goes to the browser chrome or the first link on the page rather than into the new content, this is broken. **Fix:** move focus to the new `<h1>` or to `<main tabindex="-1">` and announce the new page title.
- **Modals:** move focus into the dialog on open, trap focus *within* it, mark the rest of the page `inert`, and return focus to the trigger on close. Native `<dialog>` with `showModal()` handles most of this. A component-library modal that renders into a portal at the end of `<body>` without `inert` will let Tab escape into the page behind it — check before trusting it.
- **Reading order vs visual order (1.3.2 Meaningful Sequence, A):** beware `flex-direction: row-reverse`, `order`, and `grid-area` reordering. They change what you see without changing what a screen reader reads.

**How to verify:** the keyboard walkthrough in `testing.md` (Layer 2) is the authoritative procedure. Quick checks:
- Removed outlines: `[...document.styleSheets].flatMap(s=>{try{return [...s.cssRules]}catch(e){return []}}).filter(r=>r.style && /outline/.test(r.cssText) && /none|0px/.test(r.style.outline)).map(r=>r.selectorText)`
- Positive tabindex: `[...document.querySelectorAll('[tabindex]')].filter(e=>+e.tabIndex>0)` — must be empty.
- Live focus tracing: paste `document.addEventListener('focusin',e=>console.log(e.target))` and Tab through; the console log is your focus order.
- DOM vs visual order: DevTools → Rendering → **Show source order overlay** in Firefox, or read the DOM in the Elements panel while watching the page.

---

## 10. Zoom, reflow and text spacing

| Test | Criterion | Pass condition |
|---|---|---|
| Browser zoom to **400% on a 1280×1024 viewport** (equivalent to a **320 CSS px** viewport width / 256 px height) | 1.4.10 Reflow, AA | No two-dimensional scrolling for vertical content. Exceptions: data tables, maps, code, toolbars. |
| **200% text-only zoom** | 1.4.4 Resize Text, AA | No loss of content or function. |
| **Text-spacing bookmarklet**: line-height **1.5×**, paragraph spacing **2×**, letter-spacing **0.12×**, word-spacing **0.16×** font size | 1.4.12 Text Spacing, AA | Nothing clipped, nothing overlapping. Fixed-height containers with `overflow:hidden` are the usual culprit. |
| Rotate the device | 1.3.4 Orientation, AA | No orientation lock in CSS or the web app manifest. |

Use relative units (`rem`, `em`, `%`, `ch`) for anything containing text.

**How to verify:** set the browser window to 1280×1024, then <kbd>Ctrl</kbd>/<kbd>Cmd</kbd>+<kbd>+</kbd> to 400% and scroll top to bottom looking for a horizontal scrollbar. For 1.4.12, run Steve Faulkner's text-spacing bookmarklet or paste:
```js
document.querySelectorAll('*').forEach(e=>{e.style.lineHeight='1.5';e.style.letterSpacing='.12em';e.style.wordSpacing='.16em'});
document.querySelectorAll('p').forEach(e=>e.style.marginBottom='2em');
```
then screenshot and compare. For 1.3.4, grep the CSS for `@media (orientation:` locks and check `manifest.json` for `"orientation"`.

---

## 11. ARIA: the rules

**The first rule of ARIA is don't use ARIA.** If a native HTML element with the semantics you need exists, use it.

| Rule | Why |
|---|---|
| ARIA changes only how AT **perceives** an element — it adds **no behaviour**. | `role="button"` on a `<div>` gets you the announcement but none of the keyboard handling, focusability, or activation. |
| Never change native semantics without cause. | `<h2 role="tab">` is a red flag. |
| Every interactive ARIA control must be keyboard operable. | 2.1.1 |
| Never put `aria-hidden="true"` on a focusable element. | Creates an element that is reachable by Tab but invisible to AT. |
| All interactive elements need an accessible name. | 4.1.2 Name, Role, Value, A |
| `aria-label` is **ignored** on a `<div>`/`<span>` with no role. | Common false-confidence failure — the label is silently discarded. |
| Follow the **ARIA Authoring Practices Guide** patterns for composite widgets (tabs, comboboxes, treegrids, menus). Do not improvise keyboard interaction models. | https://www.w3.org/WAI/ARIA/apg/ |
| The accessible name must contain the visible label text. | 2.5.3 Label in Name, A |

### Live regions — 4.1.3 Status Messages (AA)

```html
<div role="status" aria-live="polite" aria-atomic="true">
  3 results found
</div>
<div role="alert">Your session will expire in 2 minutes.</div>
```

The live region container **must exist in the DOM before the content is injected**, or nothing is announced. This is the single most common reason a screen reader user never learns that a search returned zero results.

**How to verify:**
- Duplicate IDs break `for` and `aria-labelledby`: `const ids={};[...document.querySelectorAll('[id]')].forEach(e=>ids[e.id]=(ids[e.id]||0)+1);Object.entries(ids).filter(([,n])=>n>1)`
- Focusable elements hidden from AT: `[...document.querySelectorAll('[aria-hidden="true"]')].filter(e=>e.querySelector('a[href],button,input,select,textarea,[tabindex]') || e.matches('a[href],button,input,select,textarea,[tabindex]'))`
- Orphan `aria-label`: `[...document.querySelectorAll('[aria-label]')].filter(e=>/^(DIV|SPAN)$/.test(e.tagName) && !e.hasAttribute('role'))`
- Live regions: they must be in the initial HTML. `curl -s <url> | grep -E 'aria-live|role="(status|alert)"'` — if the region only appears after JS injects it, it will not announce. Confirm by triggering the update with NVDA running.
- Names and roles generally: DevTools → Accessibility pane on each custom widget; every control must show a non-empty Name and the role you expect.

---

## 12. Criterion map — structure, navigation, colour, keyboard, robustness

Forms rows are in `html-forms.md`; media and motion rows in `media-and-motion.md`; the 2.2 additions in `wcag22-new.md`.

| SC | Level | HTML/CSS implementation |
|---|---|---|
| 1.1.1 Non-text Content | A | `alt` on all `<img>`; `alt=""` for decorative; `<title>`/`aria-label` on `<svg role="img">`; text alternatives for `<canvas>`, icon fonts, CAPTCHAs |
| 1.3.1 Info and Relationships | A | Semantic elements; `<th scope>`; `<label for>`; `<fieldset>/<legend>`; heading hierarchy; real lists; ARIA only to fill genuine gaps |
| 1.3.2 Meaningful Sequence | A | DOM order matches visual order; beware `flex-direction: row-reverse`, `order`, `grid-area` |
| 1.3.3 Sensory Characteristics | A | Instructions reference the label text, not shape/position/colour |
| 1.3.4 Orientation | AA | No orientation lock in CSS or manifest |
| 1.4.1 Use of Color | A | Icon/text/underline in addition to colour; body links underlined or ≥3:1 against surrounding text *plus* a non-colour cue on hover/focus |
| 1.4.3 Contrast (Minimum) | AA | 4.5:1 / 3:1 — test against actual rendered backgrounds including gradients and image overlays |
| 1.4.4 Resize Text | AA | Relative units; no `user-scalable=no` |
| 1.4.10 Reflow | AA | Responsive layout, no fixed widths, no 2-D scroll at 320 CSS px |
| 1.4.11 Non-text Contrast | AA | 3:1 for borders, focus rings, toggle states, chart elements |
| 1.4.12 Text Spacing | AA | No fixed heights on text containers; `overflow: visible` where possible |
| 1.4.13 Content on Hover or Focus | AA | Tooltips **dismissable** with Esc, **hoverable**, **persistent** until dismissed or invalid |
| 2.1.1 Keyboard | A | Native interactive elements; no mouse-only handlers |
| 2.1.2 No Keyboard Trap | A | Escapable modals, embeds and custom widgets; test with Tab **and** Shift+Tab |
| 2.1.4 Character Key Shortcuts | A | Gate single-key shortcuts behind a modifier or a setting; or make them active only on focus |
| 2.4.1 Bypass Blocks | A | Skip link + landmarks |
| 2.4.2 Page Titled | A | Unique descriptive `<title>`, specific-to-general |
| 2.4.3 Focus Order | A | Logical DOM order; no positive `tabindex`; managed focus on dynamic changes |
| 2.4.4 Link Purpose (In Context) | A | Descriptive link text or `aria-labelledby` / `aria-describedby` context |
| 2.4.5 Multiple Ways | AA | Nav + search + sitemap (any two). Exempt if the page is a step in a process. |
| 2.4.6 Headings and Labels | AA | Descriptive headings and labels |
| 2.4.7 Focus Visible | AA | Visible `:focus-visible` styling |
| 2.5.1 Pointer Gestures | A | Buttons/alternatives for pinch, swipe, path-based gestures |
| 2.5.2 Pointer Cancellation | A | Act on `click`/`pointerup`, not `pointerdown`; or make abortable/reversible |
| 2.5.3 Label in Name | A | Accessible name starts with (or contains) the visible label |
| 2.5.4 Motion Actuation | A | Non-motion alternative for shake/tilt actions, and a way to disable |
| 3.1.1 Language of Page | A | `<html lang>` |
| 3.1.2 Language of Parts | AA | `lang` on foreign-language passages |
| 3.2.1 On Focus | A | No context change on focus |
| 3.2.3 Consistent Navigation | AA | Same nav order across pages |
| 3.2.4 Consistent Identification | AA | Same names/icons for the same functions |
| 4.1.2 Name, Role, Value | A | Native elements, or complete ARIA name/role/value/state |
| 4.1.3 Status Messages | AA | `role="status"` / `role="alert"` / `aria-live` regions present **before** the update |
