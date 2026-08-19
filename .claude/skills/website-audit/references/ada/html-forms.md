# HTML forms

**Covers:** labels, grouping, error identification and suggestion, `autocomplete`, required-field marking, validation timing, authentication fields, and the full form criterion map.
**Load when:** `content.has_forms`, `content.has_auth` or `content.has_search` is true. A search box is a form.
**Siblings:** general structure/focus/ARIA → `html-core.md` · criterion definitions for 3.3.7 / 3.3.8 → `wcag22-new.md`.

Missing form labels are consistently a top-three failure in every automated survey, and error handling is the most-failed thing that automation cannot see.

---

## 1. The reference field

```html
<div class="field">
  <label for="email">Email address</label>
  <p id="email-hint" class="hint">We'll only use this to send your receipt.</p>
  <input type="email" id="email" name="email"
         autocomplete="email"                 <!-- 1.3.5 -->
         aria-describedby="email-hint email-error"
         aria-invalid="true"                  <!-- set only when in error -->
         required>
  <p id="email-error" class="error">
    <span class="visually-hidden">Error:</span>
    Enter an email address in the format name@example.com   <!-- 3.3.1, 3.3.3 -->
  </p>
</div>

<fieldset>
  <legend>Delivery method</legend>            <!-- groups radios: 1.3.1 -->
  <input type="radio" id="std" name="ship" value="std">
  <label for="std">Standard (3–5 days)</label>
  …
</fieldset>
```

---

## 2. Labels and instructions — 3.3.2 (A), 1.3.1 (A), 2.4.6 (AA)

| Rule | Detail |
|---|---|
| Every input has a **programmatically associated label** | `<label for>` matching the input `id`, or the input nested inside the `<label>`. |
| `placeholder` is **not** a label | It disappears on input and typically fails contrast. `aria-label` is acceptable only where a visible label is genuinely impossible (e.g. a search field with an adjacent magnifier button) — and even then it must contain the visible text if any exists (2.5.3). |
| Labels are **persistent and visible** | Floating labels that collapse to nothing on focus fail. |
| Related controls are **grouped** | Radio groups and checkbox groups get `<fieldset>` + `<legend>`. Without it, a screen reader announces "Standard, radio button" with no idea what question it answers. |
| Instructions come **before** the input | And are associated with `aria-describedby`, not left floating as an adjacent `<p>`. |
| Labels describe **purpose**, not just type | "Name" on three separate fields fails 2.4.6; "First name", "Last name", "Name on card" pass. |

**How to verify:** console —
```js
[...document.querySelectorAll('input:not([type=hidden]),select,textarea')]
  .filter(e => !e.labels?.length && !e.getAttribute('aria-label') && !e.getAttribute('aria-labelledby'))
```
must be empty. Then check for placeholder-as-label: `[...document.querySelectorAll('[placeholder]')].filter(e=>!e.labels?.length)`. For grouping: every `input[type=radio]` and grouped checkbox should have a `closest('fieldset')` with a `<legend>`. DevTools → Accessibility pane on each field shows the computed Name — read it and ask whether it would make sense heard alone.

---

## 3. Required fields — 3.3.2 (A), 1.4.1 (A)

- Use the **`required` attribute** (or `aria-required="true"` on custom controls). A visual asterisk alone is not programmatic.
- Mark it visually with **text**, not colour and not an unexplained asterisk. If you use an asterisk, define it in text at the top of the form *and* keep the `required` attribute.
- Do not mark required fields with colour only (1.4.1).
- Inverse pattern — marking only the *optional* fields — is usually clearer and equally conformant, provided `required` is still set on the rest.

**How to verify:** `[...document.querySelectorAll('[required],[aria-required=true]')].length` compared against the count of visually-marked fields. Turn the page greyscale (DevTools → Rendering → Emulate vision deficiencies) and confirm you can still tell which fields are required.

---

## 4. `autocomplete` — 1.3.5 Identify Input Purpose (AA)

A **hard AA requirement** for any field collecting information *about the user*. Not optional, not a nicety.

| Category | Tokens |
|---|---|
| Identity | `name`, `given-name`, `family-name`, `username`, `bday` |
| Contact | `email`, `tel`, `street-address`, `postal-code` |
| Payment | `cc-number` |
| Credentials | `current-password`, `new-password`, `one-time-code` |

Fields not about the user (a search query, a message body, a quantity) do not need a token and should not be given a wrong one.

**How to verify:**
```js
[...document.querySelectorAll('input')].map(i=>[i.name||i.id, i.type, i.autocomplete||'—'])
```
Cross-check every personal-data field against the token list. Also test in the browser: with a saved profile, does the field autofill? If it does not, the token is missing or wrong.

---

## 5. Errors — 3.3.1 (A), 3.3.3 (AA), 3.3.4 (AA), 4.1.3 (AA)

| Requirement | Implementation |
|---|---|
| **3.3.1 Error Identification** (A) — errors identified and described in text | Message adjacent to the field, associated via `aria-describedby`, `aria-invalid="true"` on the field. Text, not just a red border. |
| **3.3.3 Error Suggestion** (AA) — correction suggested where known | "Enter an email address in the format name@example.com", not "Invalid input". "Date must be in DD/MM/YYYY format", not "Bad date". |
| **3.3.4 Error Prevention (Legal, Financial, Data)** (AA) | Submissions that are legally binding, financial, or that modify/delete user-controlled data must be **reversible**, **checked** for input errors with a chance to correct, or **confirmed** on a review step. |
| **4.1.3 Status Messages** (AA) | The error summary is announced without moving focus — `role="alert"` or a live region that already exists in the DOM. |
| **1.4.1 Use of Color** (A) | Never signal error state by colour alone. Icon **and** text. |

**Error summary pattern:** a summary block at the top of the form, focus moved to it on submit, listing every error with each entry linking to its field. This is what makes a 30-field form recoverable.

**Validation timing:** validate on submit, or on blur *after* the user has left the field. Do **not** validate on every keystroke — announcing "invalid email" after the first character is hostile, and a live region that re-announces on each keypress is worse than no announcement. Do not move focus on validation (3.2.1 On Focus, 3.2.2 On Input).

**No auto-submit (3.2.2 On Input, A):** changing a `<select>` must not navigate or submit. Provide an explicit submit button.

**How to verify:** submit the form empty, then with one deliberately malformed field.
1. Does an error appear **in text**? Screenshot it in greyscale — is it still identifiable?
2. Is the message associated? `document.querySelector('#field').getAttribute('aria-describedby')` should resolve to the error node; `aria-invalid` should be `true` on the field and absent/`false` otherwise.
3. With NVDA running, submit — the summary should be spoken without you pressing anything else. If silence, 4.1.3 fails.
4. Tab from the summary — does each entry take you to its field?
5. Does the message tell you **how to fix it**, not just that it is wrong? If not, 3.3.3 fails and no tool will catch it.
6. For 3.3.4: attempt a binding submission and look for a review step, an undo, or an inline check. Note whether data is preserved when you go back.

---

## 6. Authentication fields — 3.3.8 Accessible Authentication (Minimum), AA

Full criterion definition and the permitted exceptions are in `wcag22-new.md`. The implementation side:

```html
<input type="text"     name="username" autocomplete="username">
<input type="password" name="password" autocomplete="current-password">
<input type="text"     name="otp"      autocomplete="one-time-code" inputmode="numeric">
```

| Must | Must not |
|---|---|
| Allow paste into every credential field | `onpaste="return false"` — now an accessibility failure |
| Work with password managers (correct `autocomplete` tokens, a real `<form>`, stable field names) | Split OTP entry into six single-character boxes that reject a pasted code |
| Support WebAuthn/passkeys or another non-cognitive alternative where possible | Offer a puzzle/distorted-text CAPTCHA as the only route |
| Preserve entered data across a re-authentication (2.2.5 is AAA, but data loss here is a common complaint) | Require the user to recall a memorised security answer with no assistance |

**How to verify:** open the login page, copy a string, and press <kbd>Ctrl</kbd>/<kbd>Cmd</kbd>+<kbd>V</kbd> in the password field. If nothing pastes, 3.3.8 fails. Check for the handler directly: `document.querySelector('input[type=password]').onpaste` and `getEventListeners($0)` in Chrome DevTools. Paste a multi-digit code into a segmented OTP input — it must distribute across the boxes.

---

## 7. Multi-step processes — 3.3.7 Redundant Entry (A)

Within a single process, information already entered must be auto-populated or offered for selection rather than re-typed. Full definition in `wcag22-new.md`.

Practical form patterns that satisfy it: a "same as billing address" checkbox; carrying the email from step 1 into the confirmation screen as read-only text; a review step that shows previously entered values rather than re-asking.

**Conformance note:** by requirement 3 of the conformance model (`targets.md`), **every page of a multi-step process must conform**. Auditing the first page of a checkout and stopping is not a valid sample.

**How to verify:** walk the process end to end with realistic data and note every field you type twice. Password confirmation, information that is no longer valid, and re-authentication for security are the only permitted exceptions.

---

## 8. Form criterion map

| SC | Level | Implementation |
|---|---|---|
| 1.3.1 Info and Relationships | A | `<label for>`, `<fieldset>/<legend>`, associated hints |
| 1.3.5 Identify Input Purpose | AA | `autocomplete` tokens on personal-data fields |
| 2.4.6 Headings and Labels | AA | Labels describe purpose, not just type |
| 2.5.3 Label in Name | A | Accessible name contains the visible label text |
| 3.2.1 On Focus | A | No context change when a field receives focus |
| 3.2.2 On Input | A | No auto-submit on select change; explicit submit button |
| 3.3.1 Error Identification | A | Errors described in text, associated via `aria-describedby` |
| 3.3.2 Labels or Instructions | A | Persistent visible labels + instructions |
| 3.3.3 Error Suggestion | AA | Specific correction suggestions |
| 3.3.4 Error Prevention (Legal, Financial, Data) | AA | Reversible / checked / confirmed |
| 3.3.7 Redundant Entry | A | Prefill or offer previously entered data; "same as billing address" |
| 3.3.8 Accessible Authentication (Minimum) | AA | Allow paste; support password managers and WebAuthn/passkeys; no transcription-only OTP; no puzzle CAPTCHA as the only route |
| 4.1.2 Name, Role, Value | A | Custom controls expose name, role, state (`aria-invalid`, `aria-required`, `aria-expanded`) |
| 4.1.3 Status Messages | AA | Error summary announced via `role="alert"` / live region present before update |

**Custom dropdowns and comboboxes** are the single richest source of real user harm in forms. A `<div>` combobox that looks right and announces nothing fails 4.1.2, usually 2.1.1, and often 2.4.7. Follow the ARIA Authoring Practices Guide combobox pattern exactly or use a native `<select>`. See `failure-patterns.md`.

**How to verify (whole form):** run `../../scripts/audit_a11y.py --forms <url>` for the label/autocomplete/required sweep, then do the manual pass: fill the form using only the keyboard with a screen reader on, submit it empty, submit it wrong, and submit it right. Every one of those four states must be comprehensible without looking at the screen.
