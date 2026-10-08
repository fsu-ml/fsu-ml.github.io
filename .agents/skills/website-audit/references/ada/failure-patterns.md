# Failure patterns — symptom-indexed triage

**Covers:** the failures that show up in every audit, ordered by what to look at first, with what each symptom usually indicates and how to confirm it.
**Load when:** always, at the start of the manual pass. Work down the list; it is ordered by (harm × frequency) ÷ (time to check).
**Siblings:** implementation detail → `html-core.md`, `html-forms.md`, `media-and-motion.md` · procedure → `testing.md` · fix ordering → `program.md`.

**Baseline:** the WebAIM Million analysis published in early 2026 found detectable WCAG failures on **95.9%** of the top one million home pages, averaging **56.1 errors per page** — and those are only the **automatically detectable** ones. Assume failures exist; your job is to find the ones that matter, not to establish whether any exist.

---

## Pass 1 — three minutes, no tools

| # | Symptom | Usually indicates | Confirm it |
|---|---|---|---|
| 1 | **Tab and you cannot see where you are** | `outline: none` applied globally in a reset stylesheet, or a custom focus style with insufficient contrast | 2.4.7 / 1.4.11. Tab through the header. Then DevTools → Sources → <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>F</kbd> and search all files for `outline:none` and `outline: 0`. The stylesheet sweep is in `html-core.md` §9. |
| 2 | **Tab lands somewhere illogical, or skips visible controls** | Positive `tabindex`, DOM order diverging from visual order via flex/grid `order`, or controls built from non-focusable elements | 2.4.3 / 1.3.2. `[...document.querySelectorAll('[tabindex]')].filter(e=>+e.tabIndex>0)`. Trace order with `document.addEventListener('focusin',e=>console.log(e.target))` |
| 3 | **Tab gets stuck — you cannot leave a widget** | Keyboard trap in a modal, embedded player, map, or third-party iframe | 2.1.2. Try both Tab and **Shift+Tab**. A trap that only breaks in one direction is common. This is a **non-interference** failure: it invalidates the whole page's conformance. |
| 4 | **Grey-on-white text you have to squint at** | Secondary text, placeholder text, disabled-looking-but-enabled controls, button labels on brand-colour backgrounds | 1.4.3. DevTools → Elements → click the colour swatch; it prints the ratio and AA badge. Needs 4.5:1 normal, 3:1 large. |
| 5 | **A clickable thing that a screen reader says nothing about** | `<div onclick>` or an icon-only `<button>` with no accessible name | 2.1.1 + 4.1.2. `[...document.querySelectorAll('[onclick]')].filter(e=>!['A','BUTTON','INPUT','SELECT','TEXTAREA'].includes(e.tagName))` plus the empty-name check in `html-core.md` §6 |

---

## Pass 2 — the fifteen recurring failures

Ordered by how often they cause real harm, not by how often a scanner reports them.

| # | Failure | Symptom you will notice | What it indicates | How to confirm |
|---|---|---|---|---|
| 1 | **Inaccessible custom dropdowns / comboboxes** — *the single richest source of real user harm* | Looks like a select, but arrow keys do nothing, or the screen reader announces only the trigger text and never the options | A `<div>`-based widget built for visual fidelity with no ARIA combobox pattern and no keyboard model | Open it with the keyboard only. Arrow keys must move the active option; Enter selects; Escape closes; the trigger must expose `aria-expanded` and `aria-controls`. Compare against the ARIA APG combobox pattern. Check DevTools → Accessibility pane: role should be `combobox`, not `generic`. |
| 2 | **Modals that don't trap focus or don't return it** | Tab from inside the dialog reaches links on the page behind; closing the dialog dumps focus on `<body>` | Dialog rendered into a portal at the end of `<body>` without `inert` on the rest, and no focus restore on close | Open the modal, Tab past its last control — focus must wrap inside. Close with Escape — focus must return to the trigger. `document.activeElement` after close tells you. |
| 3 | **Missing or meaningless alt text** — *the most common failure in WebAIM's annual survey* | Screen reader announces `IMG_4471.jpg`, or "bar chart" with no data | Missing `alt` (not `alt=""`), CMS-generated alt from the filename, or alt describing appearance rather than information | `[...document.images].filter(i=>!i.hasAttribute('alt')).map(i=>i.src)` must be empty. Then read every non-empty alt against the decision tree in `html-core.md` §5 — that part is human-only. |
| 4 | **Missing form input labels** | Placeholder-only fields; the label vanishes as soon as you type | Designer removed labels for a "clean" look; `placeholder` used as a label | The label sweep in `html-forms.md` §2. Any field whose accessible name comes only from `placeholder` is a 3.3.2 failure. |
| 5 | **Insufficient text contrast** | Light grey secondary text, low-contrast button labels, white text on a mid-tone brand colour | A palette signed off on aesthetics without a contrast check | `../../scripts/audit_a11y.py --contrast <url>`. For text over images/gradients the DevTools number lies — use the Colour Contrast Analyser eyedropper. |
| 6 | **Empty links and buttons** | Screen reader announces "link" or "button" with nothing else | Icon-only controls where the icon is an inline SVG with no `<title>`, a background image, or an icon font | `[...document.querySelectorAll('a,button')].filter(e=>!e.textContent.trim()&&!e.getAttribute('aria-label')&&!e.getAttribute('aria-labelledby'))` |
| 7 | **Focus outline removed globally** | See Pass 1 #1 | A CSS reset (`*:focus{outline:none}`) that nobody replaced | As above. The fix is `:focus-visible` styling, not restoring the default. |
| 8 | **`div`s and `span`s wired up as buttons** | Mouse works, keyboard does nothing; no focus ring; screen reader silent | Component built without a native element, often to avoid resetting button styles | Pass 1 #5. Also check for `role="button"` **without** `tabindex="0"` and without a keydown handler — ARIA adds no behaviour. |
| 9 | **Skipped heading levels / no `<h1>` / headings used for styling** | The heading list reads h1, h3, h3, h6, or contains "Read more" | Headings chosen for font size; CMS templates that hard-code an h2 for the page title | `[...document.querySelectorAll('h1,h2,h3,h4,h5,h6')].map(h=>h.tagName+': '+h.textContent.trim().slice(0,40))` — read it as a table of contents. |
| 10 | **Content injected without a live region** | A search returns zero results and the screen reader says nothing | The status element is created by JS at the same moment the text is inserted, so the live region was never observed | Live regions must exist in the initial DOM. `curl -s <url> > raw.html` then `grep -c -e aria-live -e 'role="status"' -e 'role="alert"' raw.html`. Confirm by triggering the update with NVDA running. |
| 11 | **Missing `lang` attribute** | Screen reader reads English content with a French voice, or vice versa | Template omission | `document.documentElement.lang` |
| 12 | **Positive `tabindex`** | Tab order jumps to one control first, then restarts | Someone "fixed" a tab order problem with `tabindex="1"` | Pass 1 #2. Only `0` and `-1` are ever correct. |
| 13 | **`aria-label` on a non-interactive element** | The label appears in the source but is never announced | Author assumed `aria-label` works anywhere; it is **ignored** on `<div>`/`<span>` with no role | `[...document.querySelectorAll('[aria-label]')].filter(e=>['DIV','SPAN'].includes(e.tagName)&&!e.hasAttribute('role'))` |
| 14 | **Duplicate `id`s breaking `for` / `aria-labelledby`** | A field with a visible label that the screen reader does not announce | A repeated component that hard-codes an `id`, or a template rendered twice | `const seen=new Set(),dupes=[];document.querySelectorAll('[id]').forEach(e=>{seen.has(e.id)?dupes.push(e.id):seen.add(e.id)});dupes` |
| 15 | **Auto-generated captions accepted as-is** | Captions with no punctuation, no speaker IDs, and visible transcription errors | YouTube/Zoom auto-captions published without human correction | Play 60 seconds against the audio and count errors. 5–15% word error rate is typical and does not meet 1.2.2. See `media-and-motion.md`. |

---

## Pass 3 — what only shows up if you look for it

These produce no automated signal at all and are the reason a clean scan means nothing.

| Symptom | Indicates | Confirm it |
|---|---|---|
| Focus ring disappears under the sticky header as you tab down the page | 2.4.11 Focus Not Obscured — see `wcag22-new.md` | Tab slowly at several scroll positions **with the cookie banner still up** |
| Clicking an in-app link, then pressing Tab, restarts at the top of the page | Client-side routing that never moves focus. Screen reader users are not told the page changed at all. | Activate a nav link, immediately press Tab, watch `document.activeElement` |
| A slider, kanban board, crop tool or file drop zone that only responds to dragging | 2.5.7 Dragging Movements | Try the same task with single clicks only |
| A password field that refuses paste | 3.3.8 Accessible Authentication | Copy a string, press Ctrl/Cmd+V in the field |
| The same value typed twice in one checkout | 3.3.7 Redundant Entry | Complete the process with realistic data, counting repeats |
| The help/chat widget in a different position on each template | 3.2.6 Consistent Help | Record its DOM index relative to page content across the sample |
| Error message says "Invalid input" | 3.3.3 Error Suggestion — identification without suggestion | Submit a malformed value and read what you are told to do about it |
| Everything is reachable but nothing makes sense in order | 1.3.2 Meaningful Sequence, or a reading order broken by CSS `order` | Read the page with a screen reader from top to bottom with your eyes closed |
| Text clipped at 400% zoom or under the text-spacing bookmarklet | 1.4.10 / 1.4.12 — fixed-height containers with `overflow:hidden` | The zoom and spacing procedures in `html-core.md` §10 |
| An accessibility overlay widget in the corner | Not a fix. Frequently *introduces* failures and is itself litigated. | `curl -s <url> > raw.html` then `grep -i -e accessibe -e userway -e audioeye -e equalweb raw.html` — record as a finding, see `targets.md` §5 |

---

## Reading a finding correctly

- A failure in a **global component** (header, footer, nav, form field, card) is **one finding with site-wide reach**, not N findings. Report it once and state the reach. Fix ordering in `program.md`.
- A failure in a **step of a process** invalidates the whole process under conformance requirement 3 (`targets.md`).
- A **non-interference** failure — 1.4.2, 2.1.2, 2.2.2, 2.3.1 — invalidates the page regardless of everything else being right, even in content you do not rely on.
- Anything in Pass 1 blocks access entirely and outranks everything in Pass 2 regardless of traffic.
