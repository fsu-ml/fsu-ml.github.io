# Testing procedure

**Covers:** the four-layer testing model, honest automated coverage figures, the keyboard-only walkthrough, screen reader testing and which reader/browser pairs matter, sampling strategy for a whole-site audit, the pre-launch checklist, and tooling.
**Load when:** every audit, before you record a finding.
**Siblings:** what to look for → `failure-patterns.md` · what each criterion requires → `html-core.md`, `html-forms.md`, `media-and-motion.md`, `wcag22-new.md` · document formats → `documents-pdf.md`, `documents-office.md`, `documents-latex.md`.

---

## 1. The coverage reality

**Automated tools detect roughly 30–40% of WCAG failures.**

That figure is the widely-cited estimate of coverage *by criterion*. Deque reports that axe-core catches around **57% of issues by volume**, which is a different and more flattering measurement — it counts instances of the same detectable rule, not breadth across the criteria set. Cite whichever you use, and say which one it is. Neither number licenses "we ran axe and it passed."

What automation cannot see: whether the reading order makes sense, whether alt text is *right*, whether focus order is *logical*, whether an error message is actually helpful, whether a custom widget's keyboard model is coherent.

For the WCAG 2.2 additions specifically, only **2.5.8 Target Size** is reliably automatable. **Focus Appearance, Dragging Movements, Accessible Authentication, Consistent Help and Redundant Entry all require manual review** (`wcag22-new.md`).

> **Automated testing is a regression net, not an audit.**

---

## 2. The four-layer model

### Layer 1 — Automated, in CI (every commit)

```bash
# axe-core via Playwright
npm i -D @axe-core/playwright

# or Pa11y CI across a sitemap
npx pa11y-ci --sitemap https://example.com/sitemap.xml
```

Gate pull requests on zero critical/serious violations. Run `eslint-plugin-jsx-a11y` or the axe VS Code linter at author time so issues never reach CI. For a one-off audit sweep, `../../scripts/audit_a11y.py <url>` wraps the same engine and emits findings in the report schema.

### Layer 2 — Manual keyboard pass (every feature)

Unplug the mouse. Literally — trackpad off, hands away. Then, in order:

| # | Step | Criterion | Fail signal |
|---|---|---|---|
| 1 | <kbd>Tab</kbd> once from the address bar | 2.4.1 | Skip link absent, or present but invisible on focus |
| 2 | <kbd>Tab</kbd> through the entire page, top to bottom | 2.1.1 | Anything interactive you cannot reach |
| 3 | Watch the focus indicator at every stop | 2.4.7, 1.4.11 | Indicator missing, or <3:1 against its background |
| 4 | Scroll to mid-page and keep tabbing, **with the cookie banner still present** | 2.4.11 | Focus ring disappears under a sticky header/footer/chat bubble |
| 5 | Compare the tab sequence against the visual layout | 2.4.3, 1.3.2 | Order jumps around, or reads right-column-before-left |
| 6 | <kbd>Shift</kbd>+<kbd>Tab</kbd> back out of every component | 2.1.2 | You get stuck — traps often break in one direction only |
| 7 | <kbd>Enter</kbd> on links; <kbd>Enter</kbd> **and** <kbd>Space</kbd> on buttons | 2.1.1, 4.1.2 | Space scrolls the page instead of activating |
| 8 | Arrow keys inside composite widgets (tabs, menus, comboboxes, grids) | 2.1.1, 4.1.2 | Nothing happens; the widget is a `<div>` pretending |
| 9 | Open a modal: focus moves in, wraps inside, <kbd>Esc</kbd> closes, focus returns to the trigger | 2.1.2, 2.4.3 | Tab escapes to the page behind; focus lands on `<body>` after close |
| 10 | Activate an in-app navigation link, then press <kbd>Tab</kbd> immediately | 2.4.3 | Focus restarts at the top of the document — client-side routing never moved it |
| 11 | Submit a form empty, then with one bad field | 3.3.1, 3.3.3, 4.1.3 | No text error, no association, no announcement, or no suggestion |

Trace focus while you do it: `document.addEventListener('focusin',e=>console.log(e.target))`.

### Layer 3 — Screen reader pass (every release)

| Combination | Priority | Notes |
|---|---|---|
| **NVDA + Firefox or Chrome** (Windows) | Highest | Free; the most common testing baseline |
| **JAWS + Chrome** (Windows) | High | Dominant among employed screen reader users |
| **VoiceOver + Safari** (macOS/iOS) | High | The only realistic iOS option |
| **TalkBack + Chrome** (Android) | Medium | |
| **Narrator + Edge** | Low | |

Pair the reader with the browser it is actually used with. VoiceOver + Chrome behaves differently from VoiceOver + Safari and is not a substitute.

Minimum screen reader procedure per page: read the page top to bottom with the browse cursor; then navigate by **headings** (<kbd>H</kbd>), **landmarks** (<kbd>D</kbd> in NVDA, rotor in VoiceOver), **links** (<kbd>Insert</kbd>+<kbd>F7</kbd> list in NVDA), and **form fields** (<kbd>F</kbd>). Each of those four lists should be a usable table of contents on its own. If the links list is full of "here" and "read more", 2.4.4 fails regardless of the surrounding prose.

Also test:
- Browser zoomed to **400%** (1.4.10)
- The **text-spacing bookmarklet** (1.4.12)
- **Windows High Contrast / forced-colors mode** — borders and focus rings drawn with `background-image` or `box-shadow` vanish here
- **`prefers-reduced-motion` enabled** (DevTools → Rendering)

### Layer 4 — Usability testing with disabled users (periodically)

The only layer that tells you whether the product is actually usable rather than merely conformant. **Pay participants.** Nothing else substitutes for this.

---

## 3. Sampling for a site audit

You cannot test every page. Build a representative sample:

| Include | Why |
|---|---|
| Home page, and **one of each template type** | Template failures are the high-reach ones |
| **Every page of every complete process** — checkout, application, registration | **Required by conformance requirement 3** (`targets.md`). A process is only as conformant as its worst step. |
| Highest-traffic pages, pulled from analytics | Reach |
| Forms, search results, **error states, empty states** | The states nobody designs and nobody tests |
| Login and authentication flows | 3.3.8 lives here |
| Any page with data tables, charts, maps, video, or custom widgets | The hard criteria cluster on these |
| PDFs and documents linked from the top 50 pages | Sets `content.has_pdfs` etc. in the profile (`00-map.md`) |

**30–50 pages is a typical audit sample for a mid-size site.** State the sample in the report; a finding count without a denominator is meaningless.

---

## 4. Document-specific testing

| Format | Checks |
|---|---|
| **PDF** | PAC 2024 + veraPDF + Acrobat checker; manual tag-tree review; screen-reader read-through; Reflow view; check the title shows in the window bar → `documents-pdf.md` |
| **Word / PowerPoint / Excel** | Built-in Accessibility Checker; manual reading-order review (Selection Pane for slides); then validate the exported PDF → `documents-office.md` |
| **LaTeX** | veraPDF in CI; ngPDF structure inspection; PAC; check the tagging-status page for every package → `documents-latex.md` |
| **EPUB** | EPUBCheck + Ace by DAISY + read in a real reading system → `documents-office.md` |

---

## 5. HTML page — pre-launch checklist

The consolidated manual pass. Every line is a thing you *check*, not a thing you assume.

- [ ] `<html lang>` set; `lang` on foreign passages
- [ ] Unique, descriptive `<title>`
- [ ] One `<h1>`; heading levels in order, none skipped
- [ ] Landmarks: header/nav/main/footer; one `<main>`; multiple navs labelled
- [ ] Skip link, visible on focus
- [ ] Every image has appropriate `alt` (or `alt=""` if decorative)
- [ ] Every form input has a visible, associated `<label>`; `autocomplete` set
- [ ] Errors identified in text, associated with fields, summarized and announced
- [ ] Every interactive element reachable and operable by keyboard
- [ ] Visible focus indicator everywhere; nothing obscures it
- [ ] Contrast 4.5:1 text / 3:1 large text and UI
- [ ] Colour is never the only cue
- [ ] Works at 400% zoom / 320 CSS px with no 2-D scrolling
- [ ] Survives the text-spacing bookmarklet
- [ ] Targets ≥24×24 CSS px or adequately spaced
- [ ] Drag operations have a single-pointer alternative
- [ ] Password fields allow paste; passkeys or an equivalent offered
- [ ] Help mechanism in a consistent position
- [ ] Nothing already entered has to be re-entered in the same process
- [ ] Video captioned; audio described or text-equivalent; player keyboard-accessible
- [ ] Live regions for dynamic status changes
- [ ] `prefers-reduced-motion` honoured
- [ ] No autoplaying audio; carousels pausable
- [ ] Tested with NVDA or VoiceOver end to end

---

## 6. Tooling

### Web

| Tool | Cost | Use |
|---|---|---|
| **axe DevTools** (Deque) | Free ext / paid Pro | The standard. axe-core powers Lighthouse, Pa11y, Accessibility Insights. Zero-false-positive policy. |
| **WAVE** (WebAIM) | Free | Visual overlay; best for designers and content owners; runs locally |
| **Lighthouse** | Free, in Chrome | Convenient, limited rule set. **Do not treat the score as compliance.** |
| **Pa11y / Pa11y CI** | Free, OSS | CLI/CI gating |
| **Accessibility Insights** (Microsoft) | Free | Guided manual assessment — fills the automation gap well |
| **ARC Toolkit** (TPGi) | Free | Deep DOM/ARIA inspection |
| **IBM Equal Access Checker** | Free | Alternative engine, useful second opinion |
| **Colour Contrast Analyser** (TPGi) | Free | Eyedropper contrast checking — the only reliable option over images and gradients |
| **Text-spacing bookmarklet** (Steve Faulkner) | Free | 1.4.12 testing |
| **Browser DevTools** | Free | Accessibility tree inspection, contrast ratios, emulate `prefers-reduced-motion` and forced-colors |

### Screen readers
NVDA (free, Windows) · JAWS (paid, Windows) · VoiceOver (built in, macOS/iOS) · TalkBack (built in, Android) · Orca (free, Linux)

### PDF
PAC 2024 (free) · veraPDF (free, OSS, CI) · Adobe Acrobat Pro (paid) · axesPDF (paid) · CommonLook (paid) · PDFix (paid) · ngPDF (free web)

### Documents / publishing
Microsoft Accessibility Checker (built in) · Grackle (Google Workspace) · Ace by DAISY + EPUBCheck (EPUB) · Adobe InDesign export tags

### LaTeX
LaTeX 2026-06-01 kernel tagging · `tagpdf` · `luamml` · tagging-status page · Overleaf Rolling TeX Live · `make4ht` / LaTeXML / Pandoc for HTML output

**Sources:** veraPDF https://verapdf.org/ · PAC https://pac.pdf-accessibility.org/ · axe / Deque https://www.deque.com/axe/ · WAVE https://wave.webaim.org/ · WebAIM https://webaim.org/

---

## How to verify your own audit

| Question | Check |
|---|---|
| Did you actually run the manual layers, or just the scan? | The report should contain findings against criteria no scanner can detect — 2.4.3, 3.3.3, 1.3.2, 2.5.7, 3.3.7, 3.2.6. If every finding maps to an axe rule ID, you ran Layer 1 only. |
| Is the sample defensible? | List the sampled URLs in the report with the template each represents, and confirm every step of every process is present. |
| Are findings reproducible? | Each finding needs: URL, selector or screenshot, the criterion, the observed behaviour, and the **exact steps to reproduce**. "Contrast is low on the homepage" is not a finding. |
| Did you double-check the automated results? | Run a second engine (IBM Equal Access or WAVE alongside axe). Engines disagree; a violation only one reports is worth a manual look before you publish it. |
