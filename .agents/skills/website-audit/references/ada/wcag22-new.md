# WCAG 2.2: the six new A/AA criteria

**Covers:** 2.4.11 Focus Not Obscured, 2.5.7 Dragging Movements, 2.5.8 Target Size (Minimum), 3.2.6 Consistent Help, 3.3.7 Redundant Entry, 3.3.8 Accessible Authentication — plus the three new AAA criteria and the removal of 4.1.1 Parsing.
**Load when:** `compliance.target: wcag22-aa`, or the site was built to a WCAG 2.1 mental model and you need to know what it now misses.
**Why:** these are the criteria a site described as "WCAG 2.1 compliant" most reliably fails. WCAG 2.2 was published 5 October 2023 and added 9 SC while removing 1; the totals moved from 78 SC to **86 SC (31 A, 24 AA, 31 AAA)**.

**Automation reality:** of the six, only **2.5.8 Target Size** is reliably automatable. Focus Not Obscured, Dragging Movements, Accessible Authentication, Consistent Help and Redundant Entry all require manual review. An automated scan reporting "no WCAG 2.2 issues" has checked one of six.

---

## 2.4.11 Focus Not Obscured (Minimum) — AA

The element receiving keyboard focus must not be **entirely** hidden by author-created content.

Sticky headers, sticky footers, cookie consent bars and chat bubbles routinely cover the element that just received focus. The user tabs, hears the screen reader announce a control, and cannot see where it is — or, for a sighted keyboard user, appears to lose focus entirely.

```css
:target, [id] { scroll-margin-top: 6rem; }  /* ≥ sticky header height */
html { scroll-padding-top: 6rem; }
```

Implementation: `scroll-margin-top` ≥ the sticky header height, and make sure cookie banners and chat widgets do not cover the focused element. A banner that is `position: fixed; bottom: 0` will cover the last few tab stops of a long form.

**Companion:** **2.4.12 Focus Not Obscured (Enhanced)** — AAA — the focused element is not *partially* hidden either.

**How to verify:** Tab through the page **slowly, at several scroll positions, with the cookie banner still present** (do not dismiss it first — most auditors do, and that is exactly why this gets missed). Watch for the focus ring disappearing under the header as you tab from a link mid-page. Repeat at the bottom of the page with any sticky footer/chat bubble in place. Automated tooling cannot see this.

---

## 2.5.7 Dragging Movements — AA

Any function operated by **dragging** must have a **single-pointer alternative** that does not require dragging, unless dragging is essential or the function is determined by the user agent.

Applies to: range sliders, drag-to-reorder lists, kanban boards, signature pads, drag-and-drop file upload, map panning, image croppers, split panes.

| Fix | Example |
|---|---|
| Pair every drag with buttons | `Move up` / `Move down` on a reorderable list |
| Numeric input | A text field beside a slider |
| Click-to-select then click-to-place | Kanban card: "Move to…" menu |
| A conventional control | "Browse files" button alongside the drop zone |

Note: native `<input type="range">` already satisfies this via arrow keys. **A custom slider does not** — this is the most common way the criterion is failed, because the custom slider was built to match a design system and nobody added keyboard handling.

**How to verify:** list every drag interaction on the page. For each, try to complete the same task with single clicks and with the keyboard only. If you cannot, it fails. Note that 2.5.7 is about **pointer** alternatives specifically — keyboard operability satisfies 2.1.1 but you still need a single-pointer path for users who can click but not drag.

---

## 2.5.8 Target Size (Minimum) — AA

Targets are at least **24 × 24 CSS px**, *or* spaced such that a **24 px-diameter circle centred on the target does not intersect the circle of any other target**.

```css
.icon-button {
  min-inline-size: 24px;
  min-block-size: 24px;
  /* Better: 44px to also meet 2.5.5 AAA and general mobile usability */
}
```

**Exceptions:**
- Links inline within a sentence.
- Targets whose size is determined by the browser and not modified by the author.
- Targets where a size-equivalent alternative exists on the same page.
- Targets where the presentation is essential — e.g. a map pin at a geographic location.
- Where the spacing offset is met.

**Usual offenders:** close "×" buttons, table row action icons, pagination numbers, social icons in footers, checkboxes styled smaller than default.

**Companion:** **2.5.5 Target Size (Enhanced)** — AAA, since 2.1, renamed in 2.2 — targets ≥ **44 × 44** CSS px.

**How to verify:** this one is automatable. `../../scripts/audit_a11y.py --target-size <url>`, or in console:
```js
[...document.querySelectorAll('a,button,input,select,[role=button],[role=link],[role=checkbox]')]
  .map(e=>({e, r:e.getBoundingClientRect()}))
  .filter(({r})=>r.width>0 && (r.width<24 || r.height<24))
```
Then manually clear the exceptions — inline links in prose will show up in that list and are exempt. Measure spacing for anything that fails on size alone.

---

## 3.2.6 Consistent Help — A

If a page offers a **help mechanism** — human contact details, a contact form, live chat, a self-help link, an automated assistant — it must appear in the **same relative order** on each page in the set where it exists.

It does **not** require you to *have* help. Only to be consistent if you do. Easiest satisfied by putting it in a persistent header or footer component.

The failure mode is a chat widget that appears bottom-right on marketing pages, top-nav on the help centre, and inside a hamburger menu on the checkout — same mechanism, three positions in the reading order.

**How to verify:** across your page sample (`testing.md`), record the position of each help mechanism in **DOM order relative to the other page content**, not its visual position. A footer "Contact us" link that is the 3rd-from-last item on every page passes even if the footer visually differs. Console per page:
```js
[...document.querySelectorAll('a,button')].map((e,i)=>[i,e.textContent.trim()]).filter(([,t])=>/contact|help|support|chat/i.test(t))
```
Compare the relative index across pages.

---

## 3.3.7 Redundant Entry — A

Within a **single process**, information previously entered by the user (or provided to the user) is either **auto-populated** or **available for the user to select** — it does not have to be re-entered.

Applies to: multi-step checkouts, multi-page applications, "confirm your email" fields, wizards, renewal flows that already know last year's answers.

**Exceptions:**
- Re-entry is **essential** — password confirmation is the canonical case.
- The information is **no longer valid**.
- Re-entry is required for **security** — re-authentication before a sensitive action.

**How to verify:** complete the whole process with realistic data and count every value you type more than once. Anything on the list that is not covered by one of the three exceptions is a failure. Note this is a **Level A** criterion — it belongs in the blocking section of the report, not the recommendations.

---

## 3.3.8 Accessible Authentication (Minimum) — AA

No **cognitive function test** — remembering a password, solving a puzzle, transcribing a code, doing arithmetic — may be required for **any step** of an authentication process, unless one of these is provided:

| # | Permitted route | Detail |
|---|---|---|
| 1 | **Alternative** | Another authentication method that is not a cognitive test: passkeys/WebAuthn, magic link, biometrics, OAuth. |
| 2 | **Mechanism to assist** | **The critical one.** The password field must **allow paste** and work with password managers. `onpaste="return false"` is now an accessibility failure. |
| 3 | **Object recognition** | "Select all the images containing a bus" is permitted **at AA** (but not AAA). |
| 4 | **Personal content** | Identifying a picture the user themselves provided is permitted at AA. |

**Things that fail:**
- Blocking paste.
- Puzzle or distorted-text CAPTCHA as the **only** option.
- Requiring transcription of a code from an authenticator app *without* allowing paste or autofill — including six single-character OTP boxes that reject a pasted string.
- Memory-based security questions where the answer must be recalled without assistance.

Implementation markup and the paste test are in `html-forms.md` §6.

**Companion:** **3.3.9 Accessible Authentication (Enhanced)** — AAA — as above, **without** the object-recognition and personal-content exceptions. Worth adopting individually for consumer login flows.

**Why this one is expensive to retrofit:** the fix often lives in the identity provider, not the site. Budget for it early; it is the main reason to target 2.2 during a rebuild rather than after one.

---

## 2.4.13 Focus Appearance — AAA

The focus indicator has an area at least the size of a **2 CSS px** perimeter of the unfocused component, and a contrast ratio of at least **3:1** between the focused and unfocused states of that area. Worth adopting individually for high-keyboard-use applications.

---

## The removal of 4.1.1 Parsing

WCAG 2.2 **removed 4.1.1 Parsing** (Level A, from 2.0). It required: no duplicate `id`s, complete start and end tags, correct nesting, no duplicate attributes.

| Situation | What to do |
|---|---|
| Claiming **WCAG 2.2** | 4.1.1 does not apply. It was removed precisely because browsers error-correct malformed markup consistently and the criterion no longer benefited anyone. |
| Claiming **WCAG 2.0 or 2.1** — which is what §508 and ADA Title II reference | 4.1.1 **still applies** and must be addressed in the claim. |
| Either way | **Duplicate `id`s still break things** — they silently break `<label for>`, `aria-labelledby` and `aria-describedby` associations, which fails 1.3.1 and 4.1.2 on their own terms. Keep checking for them. |

For non-web documents, WCAG2ICT limited 4.1.1 to markup-based formats (see `documents-pdf.md`).

**How to verify:** `curl -s <url> | npx html-validate --stdin`, or the W3C validator at https://validator.w3.org/nu/?doc=<url>. For duplicate IDs specifically, the console one-liner in `html-core.md` §11.

---

## Quick 2.2 gap check for a "2.1 compliant" site

| # | Check | Time | Fails if |
|---|---|---|---|
| 1 | Paste into the password field | 10 s | Nothing pastes → 3.3.8 |
| 2 | Tab through a long page with the cookie banner up | 2 min | Focus ring goes under the sticky header → 2.4.11 |
| 3 | Measure the smallest icon button | 1 min | <24×24 CSS px and not adequately spaced → 2.5.8 |
| 4 | Find every drag interaction | 5 min | No single-pointer alternative → 2.5.7 |
| 5 | Complete a multi-step form | 10 min | Any value typed twice without an exception → 3.3.7 |
| 6 | Locate the help mechanism on five pages | 5 min | Different relative position → 3.2.6 |
