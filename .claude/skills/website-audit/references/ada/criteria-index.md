# Criteria index — all 86 success criteria

**Covers:** every WCAG 2.0/2.1/2.2 success criterion with number, name, level, version introduced, and what it actually requires; the A/AA/AAA totals per version; and the legacy WCAG 1.0 checkpoint set with its mapping to 2.x.
**Load when:** you need to check a specific SC number, confirm a level or version, or produce a conformance claim / VPAT row list.
**Do not load for background reading.** This is a lookup table. Implementation guidance lives in `html-core.md`, `html-forms.md`, `media-and-motion.md` and `wcag22-new.md`.

**Legend:** *Since* = version in which the criterion first appeared. **Bold rows are new in WCAG 2.2.**

---

# Principle 1 — Perceivable

### Guideline 1.1 Text Alternatives

| SC | Name | Level | Since | In one line |
|---|---|---|---|---|
| 1.1.1 | Non-text Content | A | 2.0 | Every image, icon, chart, control, CAPTCHA and media has a text alternative serving an equivalent purpose; purely decorative items are hidden from AT. |

### Guideline 1.2 Time-based Media

| SC | Name | Level | Since | In one line |
|---|---|---|---|---|
| 1.2.1 | Audio-only and Video-only (Prerecorded) | A | 2.0 | Transcript for audio-only; transcript or audio track for video-only. |
| 1.2.2 | Captions (Prerecorded) | A | 2.0 | Synchronized captions for all prerecorded audio in video. |
| 1.2.3 | Audio Description or Media Alternative (Prerecorded) | A | 2.0 | Audio description *or* a full text alternative for prerecorded video. |
| 1.2.4 | Captions (Live) | AA | 2.0 | Real-time captions for live audio. |
| 1.2.5 | Audio Description (Prerecorded) | AA | 2.0 | Audio description specifically (text alternative no longer sufficient). |
| 1.2.6 | Sign Language (Prerecorded) | AAA | 2.0 | Sign language interpretation. |
| 1.2.7 | Extended Audio Description | AAA | 2.0 | Pause video where needed to fit description. |
| 1.2.8 | Media Alternative (Prerecorded) | AAA | 2.0 | Full text alternative for all prerecorded synchronized media. |
| 1.2.9 | Audio-only (Live) | AAA | 2.0 | Text alternative for live audio-only. |

### Guideline 1.3 Adaptable

| SC | Name | Level | Since | In one line |
|---|---|---|---|---|
| 1.3.1 | Info and Relationships | A | 2.0 | Structure conveyed visually is also conveyed in markup: headings, lists, tables, labels, groups. **The single most-failed criterion in documents.** |
| 1.3.2 | Meaningful Sequence | A | 2.0 | Reading order is programmatically correct. |
| 1.3.3 | Sensory Characteristics | A | 2.0 | Instructions don't depend solely on shape, size, position, sound ("click the round button on the right"). |
| 1.3.4 | Orientation | AA | 2.1 | Don't lock to portrait or landscape unless essential. |
| 1.3.5 | Identify Input Purpose | AA | 2.1 | Fields collecting user data carry `autocomplete` tokens. |
| 1.3.6 | Identify Purpose | AAA | 2.1 | Icons, regions and controls are programmatically identifiable for personalization. |

### Guideline 1.4 Distinguishable

| SC | Name | Level | Since | In one line |
|---|---|---|---|---|
| 1.4.1 | Use of Color | A | 2.0 | Colour is never the only way information is conveyed. |
| 1.4.2 | Audio Control | A | 2.0 | Audio playing >3 s can be paused/stopped or volume-controlled independently. |
| 1.4.3 | Contrast (Minimum) | AA | 2.0 | **4.5:1** for normal text, **3:1** for large text (≥18 pt / ≥14 pt bold ≈ 24 px / 18.66 px bold). |
| 1.4.4 | Resize Text | AA | 2.0 | Text scales to 200% without loss of content or function. |
| 1.4.5 | Images of Text | AA | 2.0 | Use real text, not pictures of text (logos excepted). |
| 1.4.6 | Contrast (Enhanced) | AAA | 2.0 | 7:1 / 4.5:1. |
| 1.4.7 | Low or No Background Audio | AAA | 2.0 | Background sound ≥20 dB below speech, or mutable. |
| 1.4.8 | Visual Presentation | AAA | 2.0 | User-selectable colours, ≤80 chars/line, no justification, 1.5 line spacing, no horizontal scroll at 200%. |
| 1.4.9 | Images of Text (No Exception) | AAA | 2.0 | No images of text at all except decoration/logos. |
| 1.4.10 | Reflow | AA | 2.1 | No two-dimensional scrolling at **320 CSS px** width / **256 px** height — i.e. 400% zoom on a 1280 px viewport. |
| 1.4.11 | Non-text Contrast | AA | 2.1 | **3:1** for UI component boundaries, focus indicators, and meaningful graphical objects. |
| 1.4.12 | Text Spacing | AA | 2.1 | No loss of content when users set line-height 1.5×, paragraph spacing 2×, letter-spacing 0.12×, word-spacing 0.16× font size. |
| 1.4.13 | Content on Hover or Focus | AA | 2.1 | Tooltips/popovers are **dismissable**, **hoverable**, and **persistent**. |

---

# Principle 2 — Operable

### Guideline 2.1 Keyboard Accessible

| SC | Name | Level | Since | In one line |
|---|---|---|---|---|
| 2.1.1 | Keyboard | A | 2.0 | All functionality available from a keyboard, without timing-dependent keystrokes. |
| 2.1.2 | No Keyboard Trap | A | 2.0 | Focus can always be moved away using the keyboard. |
| 2.1.3 | Keyboard (No Exception) | AAA | 2.0 | 2.1.1 without the path-dependent-input exception. |
| 2.1.4 | Character Key Shortcuts | A | 2.1 | Single-character shortcuts can be turned off, remapped, or are active only on focus. |

### Guideline 2.2 Enough Time

| SC | Name | Level | Since | In one line |
|---|---|---|---|---|
| 2.2.1 | Timing Adjustable | A | 2.0 | Time limits can be turned off, adjusted (10×), or extended (20 s warning, 10 extensions). |
| 2.2.2 | Pause, Stop, Hide | A | 2.0 | Moving/blinking/scrolling >5 s and auto-updating content can be paused or hidden. |
| 2.2.3 | No Timing | AAA | 2.0 | No time limits except real-time events. |
| 2.2.4 | Interruptions | AAA | 2.0 | Interruptions can be postponed or suppressed. |
| 2.2.5 | Re-authenticating | AAA | 2.0 | Data preserved across re-authentication. |
| 2.2.6 | Timeouts | AAA | 2.1 | Users warned about data-loss timeouts unless data is preserved 20+ hours. |

### Guideline 2.3 Seizures and Physical Reactions

| SC | Name | Level | Since | In one line |
|---|---|---|---|---|
| 2.3.1 | Three Flashes or Below Threshold | A | 2.0 | Nothing flashes more than 3× per second above the general/red flash thresholds. |
| 2.3.2 | Three Flashes | AAA | 2.0 | Nothing flashes more than 3× per second, full stop. |
| 2.3.3 | Animation from Interactions | AAA | 2.1 | Motion animation from interactions can be disabled (honour `prefers-reduced-motion`). |

### Guideline 2.4 Navigable

| SC | Name | Level | Since | In one line |
|---|---|---|---|---|
| 2.4.1 | Bypass Blocks | A | 2.0 | Skip link or landmarks to bypass repeated content. |
| 2.4.2 | Page Titled | A | 2.0 | Descriptive, unique page/document title. |
| 2.4.3 | Focus Order | A | 2.0 | Tab order preserves meaning and operability. |
| 2.4.4 | Link Purpose (In Context) | A | 2.0 | Link purpose clear from the text or its programmatic context. |
| 2.4.5 | Multiple Ways | AA | 2.0 | More than one route to each page (nav + search + sitemap), unless it's a step in a process. |
| 2.4.6 | Headings and Labels | AA | 2.0 | Headings and labels describe topic or purpose. |
| 2.4.7 | Focus Visible | AA | 2.0 | Keyboard focus indicator is visible. |
| 2.4.8 | Location | AAA | 2.0 | User's location within a set is indicated (breadcrumbs). |
| 2.4.9 | Link Purpose (Link Only) | AAA | 2.0 | Link text alone identifies purpose. |
| 2.4.10 | Section Headings | AAA | 2.0 | Section headings organize content. |
| **2.4.11** | **Focus Not Obscured (Minimum)** | **AA** | **2.2** | The focused element is not *entirely* hidden by author-created content (sticky headers, cookie bars). |
| **2.4.12** | **Focus Not Obscured (Enhanced)** | **AAA** | **2.2** | The focused element is not *partially* hidden either. |
| **2.4.13** | **Focus Appearance** | **AAA** | **2.2** | Focus indicator ≥2 CSS px perimeter, ≥3:1 contrast against unfocused state. |

### Guideline 2.5 Input Modalities

| SC | Name | Level | Since | In one line |
|---|---|---|---|---|
| 2.5.1 | Pointer Gestures | A | 2.1 | Multipoint/path-based gestures have a single-pointer alternative. |
| 2.5.2 | Pointer Cancellation | A | 2.1 | Action fires on up-event, or is abortable/reversible. |
| 2.5.3 | Label in Name | A | 2.1 | The accessible name contains the visible label text. |
| 2.5.4 | Motion Actuation | A | 2.1 | Device-motion-triggered functions have a UI alternative and can be disabled. |
| 2.5.5 | Target Size (Enhanced) | AAA | 2.1 | Targets ≥ **44×44** CSS px. *(Renamed in 2.2.)* |
| 2.5.6 | Concurrent Input Mechanisms | AAA | 2.1 | Don't restrict which input modality can be used. |
| **2.5.7** | **Dragging Movements** | **AA** | **2.2** | Anything requiring a drag has a single-pointer alternative (tap, buttons, form field). |
| **2.5.8** | **Target Size (Minimum)** | **AA** | **2.2** | Targets ≥ **24×24** CSS px, or spaced so a 24 px circle centred on each doesn't overlap another. Exceptions: inline links in text, browser-default controls, essential sizing. |

---

# Principle 3 — Understandable

### Guideline 3.1 Readable

| SC | Name | Level | Since | In one line |
|---|---|---|---|---|
| 3.1.1 | Language of Page | A | 2.0 | Default human language is programmatically set. |
| 3.1.2 | Language of Parts | AA | 2.0 | Language changes within content are marked up. |
| 3.1.3 | Unusual Words | AAA | 2.0 | Definitions available for jargon and idiom. |
| 3.1.4 | Abbreviations | AAA | 2.0 | Expansions available. |
| 3.1.5 | Reading Level | AAA | 2.0 | Lower-secondary reading level, or a supplement provided. |
| 3.1.6 | Pronunciation | AAA | 2.0 | Pronunciation available where meaning is ambiguous without it. |

### Guideline 3.2 Predictable

| SC | Name | Level | Since | In one line |
|---|---|---|---|---|
| 3.2.1 | On Focus | A | 2.0 | Receiving focus doesn't trigger a change of context. |
| 3.2.2 | On Input | A | 2.0 | Changing a setting doesn't auto-trigger a change of context without warning. |
| 3.2.3 | Consistent Navigation | AA | 2.0 | Repeated navigation appears in the same relative order across pages. |
| 3.2.4 | Consistent Identification | AA | 2.0 | Same-function components are identified consistently. |
| 3.2.5 | Change on Request | AAA | 2.0 | Changes of context only on user request. |
| **3.2.6** | **Consistent Help** | **A** | **2.2** | Help mechanisms (contact details, chat, help link) appear in the same relative order on every page that has them. |

### Guideline 3.3 Input Assistance

| SC | Name | Level | Since | In one line |
|---|---|---|---|---|
| 3.3.1 | Error Identification | A | 2.0 | Errors are identified and described in text. |
| 3.3.2 | Labels or Instructions | A | 2.0 | Inputs have labels/instructions. |
| 3.3.3 | Error Suggestion | AA | 2.0 | Correction suggestions offered where known. |
| 3.3.4 | Error Prevention (Legal, Financial, Data) | AA | 2.0 | Reversible, checked, or confirmed submissions. |
| 3.3.5 | Help | AAA | 2.0 | Context-sensitive help available. |
| 3.3.6 | Error Prevention (All) | AAA | 2.0 | 3.3.4 extended to all submissions. |
| **3.3.7** | **Redundant Entry** | **A** | **2.2** | Information already entered in the same process is auto-populated or selectable, not re-typed. |
| **3.3.8** | **Accessible Authentication (Minimum)** | **AA** | **2.2** | No cognitive function test (memorizing, transcribing, puzzles) unless there's an alternative, a mechanism to assist (password manager / paste), object recognition, or personal-content identification. |
| **3.3.9** | **Accessible Authentication (Enhanced)** | **AAA** | **2.2** | As above, without the object-recognition and personal-content exceptions. |

---

# Principle 4 — Robust

| SC | Name | Level | Since | In one line |
|---|---|---|---|---|
| ~~4.1.1~~ | ~~Parsing~~ | ~~A~~ | 2.0 | **REMOVED in WCAG 2.2.** Still applies for 2.0/2.1 claims: no duplicate IDs, complete start/end tags, proper nesting. |
| 4.1.2 | Name, Role, Value | A | 2.0 | Every UI component exposes a name, role, states, properties and values to AT. |
| 4.1.3 | Status Messages | AA | 2.1 | Status changes are announced without moving focus (`aria-live`, `role="status"`, `role="alert"`). |

---

## Totals

| | Level A | Level AA | Level AAA | Total |
|---|---|---|---|---|
| WCAG 2.0 | 25 | 13 | 23 | **61** |
| WCAG 2.1 | 30 | 20 | 28 | **78** |
| WCAG 2.2 | 31 | 24 | 31 | **86** |

**A + AA is what you must satisfy for a Level AA claim: 55 criteria under 2.2** (31 + 24).

**How to verify a claim is complete:** enumerate the 55 A+AA rows above and require an explicit disposition for each — *Supports*, *Partially Supports*, *Does Not Support*, *Not Applicable* — with remarks. A row with no disposition is an incomplete claim, not a passing one. See `program.md` for VPAT/ACR mechanics.

---

# Appendix — WCAG 1.0 (obsolete, for legacy claims only)

**Status: obsolete since December 2008.** Included for auditing legacy conformance claims and because a handful of old contracts and international standards still reference it. **Do not build against WCAG 1.0.** If you find a policy citing it, treat WCAG 2.2 AA as satisfying and exceeding it.

### Structure

- **14 guidelines**, containing **65 checkpoints** in total.
- Each checkpoint carries a priority:

| Priority | Count | Meaning | Maps to |
|---|---|---|---|
| **Priority 1** | 16 | Developer *must* satisfy, or one or more groups find it **impossible** to access the content | Level **A** |
| **Priority 2** | 30 | *Should* satisfy; otherwise groups find it **difficult** | Level **AA** (Double-A) |
| **Priority 3** | 19 | *May* satisfy; otherwise some groups find it **somewhat difficult** | Level **AAA** (Triple-A) |

### The 14 guidelines

| # | Guideline |
|---|---|
| 1 | Provide equivalent alternatives to auditory and visual content |
| 2 | Don't rely on color alone |
| 3 | Use markup and style sheets, and do so properly |
| 4 | Clarify natural language usage |
| 5 | Create tables that transform gracefully |
| 6 | Ensure that pages featuring new technologies transform gracefully |
| 7 | Ensure user control of time-sensitive content changes |
| 8 | Ensure direct accessibility of embedded user interfaces |
| 9 | Design for device-independence |
| 10 | Use interim solutions |
| 11 | Use W3C technologies and guidelines |
| 12 | Provide context and orientation information |
| 13 | Provide clear navigation mechanisms |
| 14 | Ensure that documents are clear and simple |

### The 16 Priority 1 checkpoints

| Checkpoint | Requirement |
|---|---|
| 1.1 | Provide a text equivalent for every non-text element (`alt`, `longdesc`, element content) |
| 1.2 | Provide redundant text links for each active region of a server-side image map |
| 1.3 | Provide an auditory description of the visual track of a multimedia presentation |
| 1.4 | Synchronize equivalent alternatives (captions, auditory descriptions) with time-based presentations |
| 2.1 | Ensure all information conveyed with color is also available without color |
| 4.1 | Clearly identify changes in the natural language of the text and text equivalents |
| 5.1 | For data tables, identify row and column headers |
| 5.2 | For tables with two or more logical levels of headers, use markup to associate data cells with header cells |
| 6.1 | Organize documents so they may be read without style sheets |
| 6.2 | Ensure that equivalents for dynamic content are updated when the dynamic content changes |
| 6.3 | Ensure pages are usable when scripts, applets or other programmatic objects are turned off |
| 7.1 | Avoid causing the screen to flicker |
| 8.1 | Make programmatic elements (scripts, applets) directly accessible or AT-compatible *(P1 if the functionality is important and not presented elsewhere, else P2)* |
| 9.1 | Provide client-side image maps instead of server-side, except where regions cannot be defined with a geometric shape |
| 11.4 | If you cannot make a page accessible after best efforts, provide a link to an accessible alternative page with equivalent information, updated as often |
| 14.1 | Use the clearest and simplest language appropriate for the content |

**Priority 2 (30 checkpoints)**, thematically: sufficient colour contrast; proper header/list/quotation markup rather than visual fakery; relative rather than absolute units; style sheets rather than presentational markup; avoiding blinking/moving/auto-refreshing content; keyboard operability of interactive elements and logical tab order; `title` on frames; validating to published grammars; avoiding deprecated features; dividing large information blocks into manageable groups.

**Priority 3 (19 checkpoints)**, thematically: expansions of abbreviations and acronyms; identifying document language; summaries for tables; a site map or table of contents; keyboard shortcuts; distinguishing adjacent links with more than whitespace; specifying logical tab order; supplementing text with graphics or audio where it aids comprehension; consistent presentation across pages.

*Authoritative full list: W3C "Checklist of Checkpoints for WCAG 1.0" — https://www.w3.org/TR/WCAG10/full-checklist.html · WCAG 1.0 — https://www.w3.org/TR/WCAG10/*

### Mapping WCAG 1.0 → WCAG 2.x

| WCAG 1.0 | Nearest WCAG 2.x |
|---|---|
| 1.1 text equivalents | 1.1.1 Non-text Content |
| 1.3, 1.4 multimedia alternatives | 1.2.1–1.2.5 |
| 2.1 colour | 1.4.1 Use of Color |
| 2.2 contrast | 1.4.3 / 1.4.6 |
| 3.x structural markup | 1.3.1 Info and Relationships |
| 4.1, 4.3 language | 3.1.1 / 3.1.2 |
| 5.1, 5.2 table headers | 1.3.1 |
| 6.3, 8.1 script independence | 2.1.1 Keyboard, 4.1.2 Name Role Value |
| 7.x blinking, movement, auto-refresh | 2.2.1, 2.2.2, 2.3.1 |
| 9.x device independence | 2.1.1 Keyboard, 2.4.3 Focus Order, 2.4.7 Focus Visible |
| 12.3 grouping | 1.3.1 |
| 13.x navigation | 2.4.x |
| 14.1 clear language | 3.1.5 Reading Level (AAA) |
| 11.4 alternative page | "Conforming alternate version" in the 2.x conformance model |

**Concepts with no WCAG 2.x equivalent**, deliberately dropped as obsolete: "use W3C technologies" (11.1), "avoid deprecated features" (11.2), the server-side image map rules, and most of Guideline 10's "interim solutions until user agents…" checkpoints.
