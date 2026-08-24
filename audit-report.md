# Website audit — FSU SC Artificial Intelligence Seminar

**Target:** https://sc-ai.net/ (audited against `http://localhost:4173/`, commit `c6c9de5`)
**Date:** 2026-08-19 · **Profile:** [`audit-profile.yaml`](audit-profile.yaml)
**Method:** [`.claude/skills/website-audit`](.claude/skills/website-audit) — five parallel agents covering
accessibility, responsive/mobile, performance + motion safety, SEO + hygiene, and code quality + fit.
**Pages:** `/`, `/schedule/`, `/speakers/`, `/archive/`, `/trichotemy.html`

> **All performance numbers here are lab-only.** `seo.data_access` is empty — there is no
> Search Console, CrUX or RUM access for this domain. Nothing below tells you whether real
> visitors are slow. INP is a field-only metric and is not reported.

---

## Standing constraint

The motion design shipped immediately before this audit and is **intentional and out of scope**.
`motion.css`, `reveal.js` and `seminar-card.css` were treated as read-only. The only motion
findings acted on were correctness ones: fail-open behaviour and `prefers-reduced-motion`
conformance. Both were verified **already correct** and needed no change.

---

## What was fixed (commit `4a05ec5`)

| # | Severity | Finding | Fix |
|---|---|---|---|
| 1 | blocker | No skip link on any page (SC 2.4.1) | Skip link + `#main` target on all 5 pages |
| 2 | blocker | Footer Zoom icon 404s on every subdirectory page | Root-relative path in `static/js/ui/icons.js` |
| 3 | major | `/archive/` and `/trichotemy.html` had zero static inbound links | Static footer links in all 5 shells |
| 4 | major | `backup/` (143MB) and `data/past-html/` publicly served and indexable | `robots.txt` disallows |
| 5 | major | Google Fonts stylesheet render-blocking on all 5 pages | `media="print"` + `onload` swap |
| 6 | major | Both CSVs fetched 4× each per homepage load | Memoized fetch, 8 requests → 2 |
| 7 | major | Prompt textarea at 13.44px triggers iOS Safari auto-zoom | Raised to 16px |
| 8 | minor | Heading outline skipped h1→h3 on 3 pages | `sr-only` h2; h3→h2 on `/trichotemy.html` |
| 9 | minor | `aria-label` on generic `<div>` discarded, full date lost to AT | `.sr-only` pattern at 3 call sites |
| 10 | minor | Nav disclosure did not close on Escape or outside click | Both handlers added |
| 11 | minor | 2 templates fetched every load, never rendered | Deleted |
| 12 | minor | Six text runs between 10.24px and 12.16px | Sub-12px raised |
| 13 | minor | LCP element had no priority hint; no intrinsic size | `fetchpriority="high"` + true ratio |
| 14 | advisory | `og:image:alt` missing; `/trichotemy.html` title 70 chars | Added / shortened to 49 |
| 15 | advisory | No `.gitignore` at repo root | Added |

### Verified after the fixes

- **0 4xx/5xx** across all 5 pages, fresh browser context each.
- Skip link is the first tab stop, becomes visible, and Enter moves focus to `<main id="main">`.
- Escape closes the nav and returns focus to the toggle; outside click closes it.
- No heading-level skips on any page; no `div[aria-label]` anywhere.
- CSV requests per homepage load: **8 → 2**.
- Google Fonts reports `renderBlockingStatus: "non-blocking"`.
- LCP 2612ms → 2520ms.
- **Reduced motion still behaves identically** (0.24 mid-fade vs 1.0 instant). Animations unchanged.

---

## Open findings — your decision

### O-1 · No security headers at all (grade E) — *cannot be fixed from this repo*

No CSP, `X-Content-Type-Options`, `Referrer-Policy`, `Permissions-Policy`, COOP or CORP.
GitHub Pages cannot set custom response headers — there is no `_headers` equivalent on stock
Pages. The only real fix is an edge layer (e.g. Cloudflare proxying the existing `CNAME`) with
transform rules. **This is an infrastructure decision, not a code change.**

### O-2 · Content is invisible without JavaScript (architectural)

Raw HTML carries 2–6% of each page's rendered text. With JS disabled, `/speakers/` drops from
192 visible blocks to 5. The static footer links added in fix #3 mitigate discovery and give a
no-JS visitor the Zoom room and a contact address, but the schedule, speakers and archive still
require JS. Googlebot does execute JS, so this is not necessarily an indexing emergency — but
it is unverifiable without Search Console, and `site-categories.md` flags it as the single most
consequential mistake for an academic/lab site, where citability matters over a decade.

Fixing properly means pre-rendering the shell at commit time (a GitHub Action reading the same
CSVs). That is a real project, not an audit fix.

### O-3 · CLS 0.123 on `/` (budget 0.1) — skeleton/card height mismatch

Measured directly: the skeleton is a fixed **545px**. The real card is **541px at 1280px wide**
(−3.6px, effectively perfect) but **627px at 390px wide** — an 82px shift on mobile.
A fixed `min-height` would be a guess that is wrong for any talk with a longer title or bio,
and the skeleton is part of the loading design you just shipped, so **I did not change it**.
The durable fix is O-2 (render real content, no skeleton swap).

### O-4 · Reveal activation is coupled to unrelated network fetches

`observeReveals(document)` runs only at the end of `init()`, after both CSV fetches. Six static
`[data-reveal]` elements in `index.html` don't depend on that data, so on a slow connection they
sit at `opacity: 0` until the fetch resolves or the 6s failsafe trips. **This sits inside your
motion system, so I left it alone.** The fix would not change how anything looks or how long any
animation runs — only when observation starts. Say the word and it's a small patch.

### O-5 · `data/speaker-profiles.csv` exposes 25 unused email addresses

Including two personal Gmail accounts. `speakers.js` maps the `email` column into the record and
**no renderer ever outputs it** — it is dead data, publicly fetchable because the CSV must be.
Cheapest fix is dropping the column. Left alone because you had just edited that file.

### O-6 · Scroll jank exceeds the dropped-frame budget

`/` 5.96% and `/archive/` 17.64%, against a 5% threshold — **headless, which under-reports**, so
treat these as a floor. Both worst frames cluster at `scrollY: 0`, i.e. entrance, not mid-scroll.
Reported, not fixed, per the motion constraint. One part is separable from the animation design:
`static/archive.js` shows 87ms of *forced synchronous layout*, a DOM read/write ordering problem
that could be fixed without altering how anything looks.

### O-7 · Smaller items

- `.next-seminar` uses `border-radius: 16px` while `USAGE.md` documents an 8px ceiling and 13
  other cards use `var(--radius)`. **Code and its own style guide disagree** — tell me which is right.
- 35 `:hover` rules have no `@media (hover: hover)` guard, so they stick after tap on touch.
  Every one pairs with a `:focus-visible` equivalent, so nothing is unreachable. Touches motion selectors.
- Date-formatting helpers are byte-for-byte duplicated in `sections.js` and `archive.js`.
- Dead CSS for `.schedule-item` / `.speaker-card` remains after deleting their templates.
- Five near-identical page-entry scripts could share a `bootPage()` helper.
- `/trichotemy.html` misspells "trichotomy". **Recommend not renaming** — the URL is ~1 year old
  across two semesters of archived content and GitHub Pages has no server-side redirects.

---

## Tooling false positives — do not "fix" these

Recorded because a future run will surface them again.

1. **6 blockers + 8 majors from the automated a11y scan were all false positives.** Verified individually.
2. **Contrast failures on `/archive/`** (2.92:1, 3.75:1, 4.29:1) — the scanner sampled `[data-reveal]`
   elements mid-fade. True ratios are **8.91:1, 15.82:1, 18.46:1**, all passing.
3. **"Image is the content of a link but has empty alt"** — the heuristic inspects only the image's
   immediate parent. The links compute proper names ("Join via Zoom", "Zoom Room").
4. **`.sr-only` "Toggle navigation" flagged as 13.3px text** — it is clipped to 1px for screen
   readers. Resizing it would be a mistake.
5. **Render-blocking third-party still flagged after the fix** — the script reads post-load DOM,
   where `onload` has already restored `media="all"`. The browser reports `non-blocking`.
6. **Canonical "mismatch"** — an artifact of auditing localhost against production canonicals.

---

## Not verified

- Field CWV data, and whether Google indexes the client-rendered content (no Search Console).
- Screen reader passes (NVDA/JAWS/VoiceOver) — no AT available.
- Real-device iOS auto-zoom, safe-area insets, sticky `:hover` — emulation cannot confirm these.
- True vsync scroll jank — headless under-reports.
- TLS chain and ciphers; structured-data eligibility (needs Rich Results Test).
- Speaker-supplied PDFs under `materials/` — out of scope per the profile.

---

## Re-running

```bash
python3 -m http.server 4173          # or: preview_start {name: "site"}
pip install -r .claude/skills/website-audit/scripts/requirements.txt
playwright install chromium

python3 .claude/skills/website-audit/scripts/audit_a11y.py http://localhost:4173/ --all --standard wcag22aa
python3 .claude/skills/website-audit/scripts/audit_responsive.py http://localhost:4173/
python3 .claude/skills/website-audit/scripts/audit_performance.py http://localhost:4173/
python3 .claude/skills/website-audit/scripts/audit_motion.py http://localhost:4173/ --headful
python3 .claude/skills/website-audit/scripts/audit_seo.py http://localhost:4173/
python3 .claude/skills/website-audit/scripts/check_headers.py https://sc-ai.net/
```

`check_headers.py` is standard library only and needs no install.
