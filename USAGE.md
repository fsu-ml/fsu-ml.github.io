# Editing the FSU SC Artificial Intelligence Seminar Website

This is a static site (GitHub Pages, no build step). Most routine updates — adding a
talk, adding a speaker, swapping a link — are edits to a CSV or to one JavaScript data
file. You should not need to touch HTML for normal content changes.

- [Quick reference: "I want to…"](#quick-reference-i-want-to)
- [Running the site locally](#running-the-site-locally)
- [Adding or editing a talk](#adding-or-editing-a-talk)
- [Adding or editing a speaker](#adding-or-editing-a-speaker)
- [Images](#images)
- [Site text, links, and navigation](#site-text-links-and-navigation)
- [How the schedule decides what to show](#how-the-schedule-decides-what-to-show)
- [File and directory reference](#file-and-directory-reference)
- [Conventions to follow when changing code](#conventions-to-follow-when-changing-code)

---

## Quick reference: "I want to…"

| Task | Edit this |
| --- | --- |
| Add / reschedule / remove a talk | `data/speakers.csv` |
| Add a new speaker's bio, photo, or links | `data/speaker-profiles.csv` (+ `data/speaker-images/`) |
| Add a holiday or break row to the schedule | `data/speakers.csv` (see [Break rows](#break-and-no-seminar-rows)) |
| Add slides / video links to a past talk | `materials` column in `data/speakers.csv` |
| Change the Discord, Zoom, mailing list, or room link | `static/js/data/page-data.js` |
| Change hero text, nav items, or homepage cards | `static/js/data/page-data.js` |
| Change colors, spacing, or layout | `static/css/` |
| Change how something is rendered | `static/js/render/` and `templates/` |

---

## Running the site locally

The page fetches its templates and CSV files with `fetch()`, so **opening
`index.html` directly from the filesystem will not render correctly**. Serve the
directory over HTTP instead:

```bash
cd /path/to/fsu-ml.github.io
python3 -m http.server 8000
```

Then open <http://localhost:8000/>. Check the pages you touched: `/`, `/schedule/`,
`/speakers/`, `/archive/`.

If content is missing, open the browser console — the data loaders log a warning and
fall back to empty content when a CSV fails to load or parse (usually an unquoted
comma).

Publishing is just a commit to `main`; GitHub Pages serves the repository as-is.

---

## Adding or editing a talk

All talks — upcoming, past, and holiday placeholders — live in one file:
**`data/speakers.csv`**.

Header row:

```csv
season,name,talk_title,talk_date,description,materials,event_image
```

| Column | Required | Meaning |
| --- | --- | --- |
| `season` | yes | Semester label, `YYYY-Spring` or `YYYY-Fall` (for example `2026-Fall`). |
| `name` | yes | Speaker name, matching a `name` in `speaker-profiles.csv`. Separate co-speakers with `;` (`Tommie Juzek;Lan Li`). Use `Speaker TBA` if unknown. |
| `talk_title` | yes | Title shown in the schedule and archive. |
| `talk_date` | yes | ISO date, `YYYY-MM-DD`. |
| `description` | no | One- or two-sentence summary shown in the schedule's Description column. |
| `materials` | no | Semicolon-separated links (video, slides). Rendered as link chips on `/archive/`. |
| `event_image` | no | Filename only, from `data/event-images/` — artwork shown in place of a speaker on break rows. |

Rules and gotchas:

- **Quote any field containing a comma:** `"Dr. Xiuwen Liu, Professor of Computer Science, on agentic systems."` An unquoted comma silently shifts every later column.
- Rows may be listed in any order; the site sorts by `talk_date` within a season.
- Keep past rows in the file. They are what populates `/archive/`.
- Season labels follow the calendar: Spring covers Dec 21 – Jun 20, Fall covers Jun 21 – Dec 20. A talk on Dec 28, 2026 belongs to `2027-Spring`.
- An optional `featured` column is supported: a row whose `featured` is `false` is hidden everywhere. Omit the column entirely for normal talks.

### A typical new talk

```csv
2026-Fall,Jane Doe,Scaling Retrieval for Scientific Corpora,2026-10-16,"Dr. Jane Doe on retrieval pipelines for large scientific document collections.",
```

Then confirm `Jane Doe` exists in `data/speaker-profiles.csv` (below). If she does
not, the talk still renders, but her name will not link anywhere and she will not
appear on `/speakers/`.

### Break and "no seminar" rows

A row is treated as a break — greyed out, no speaker link, never marked "Next up" —
when its `talk_title` contains **no classes**, **holiday**, **break**, or **recess**.
Give it artwork with `event_image`; `name` is decorative on these rows (emoji are
used today):

```csv
2026-Spring,☀️🍹☀️,Spring Break,2026-03-20,,
2026-Fall,🦃,Thanksgiving Holiday,2026-11-26,,,thanksgiving.png
```

---

## Adding or editing a speaker

**`data/speaker-profiles.csv`** is the source of truth for who a speaker is. Talks
reference it by `name`.

Header row:

```csv
name,title,department,affiliation,specialties,email,website,profile_url,image
```

| Column | Meaning |
| --- | --- |
| `name` | Canonical display name. Use this exact spelling in `speakers.csv`. |
| `title` | Role or academic title (`Assistant Professor`, `PhD Candidate`). |
| `department` | Department, lab, school, or unit. |
| `affiliation` | University, company, or organization. |
| `specialties` | Semicolon-separated topics shown as chips (`LLMs; retrieval augmented generation`). Only the first three display. |
| `email` | Optional, public addresses only. |
| `website` | Optional personal or lab site; used first for name links. |
| `profile_url` | Optional institutional profile; used when `website` is empty. |
| `image` | **Filename only**, resolved as `data/speaker-images/{image}`. Leave empty for a placeholder. |

Notes:

- Matching ignores a leading `Dr. ` and is case-insensitive, so `Dr. Jane Doe` in `speakers.csv` matches `Jane Doe` here.
- A few historical spelling variants are mapped in `static/js/data/speakers.js` (`profileAliases`). Prefer fixing the CSV over adding an alias.
- Any name containing `TBA` is treated as a placeholder: it renders as "To be announced" and is excluded from `/speakers/`.
- `/speakers/` lists everyone in this file, ordered by number of talks, then alphabetically.

---

## Images

| Directory | Used for | Referenced from |
| --- | --- | --- |
| `data/speaker-images/` | Speaker headshots | `image` column of `speaker-profiles.csv` |
| `data/event-images/` | Holiday and break artwork | `event_image` column of `speakers.csv` |
| `images/` | General page artwork (hero and footer use `images/banner-wide-half-dark.png`) | CSS and `page-data.js` |

Use lowercase, underscore-separated filenames (`jane_doe.jpg`). Headshots look best
roughly square; crop before committing rather than resizing in CSS. Reference them by
filename only in the CSVs — the loaders build the full path.

---

## Site text, links, and navigation

Everything that is not talk or speaker data lives in **`static/js/data/page-data.js`**:
page title and meta description, navigation items, hero headline and buttons, the
"Next Seminar" card, homepage sections and cards, community links, and the footer.

Current contact links:

- Discord invite: `https://discord.com/invite/raTxTXmM5B`
- Mailing list: `gerlebacher@fsu.edu`
- Zoom room: `https://fsu.zoom.us/j/9038137210`
- Room location: `https://goo.gl/maps/BJLxE3Q7H1MTBqMu6`

Change a link in `page-data.js` and it updates everywhere it appears.

---

## How the schedule decides what to show

The homepage and `/schedule/` show **upcoming talks for one semester only**:

1. Take the current calendar semester and list its talks dated today or later.
2. If that semester has no upcoming talks, roll forward to the next one (Spring → Fall, Fall → next Spring).
3. If neither has anything scheduled, show a break message — "Taking a Break for the Summer…" before Fall, "…for the Winter…" before Spring.

The next chronological non-break talk is highlighted and tagged **Next up**. Talks
already past within the displayed semester render greyed out rather than disappearing.

`/archive/` shows everything dated before today, grouped by season with the newest
semester first, and turns the `materials` column into link chips.

So: to make a talk appear, add a future-dated row in the right season. To make the
break message appear, simply have no future-dated rows.

---

## File and directory reference

```
index.html              Front-page shell: semantic regions plus empty data-* mount points
schedule/ speakers/ archive/ trichotemy.html
                        Subpage shells, initialized by their own entry scripts
templates/              Reusable HTML fragments fetched and filled at runtime
static/styles.css       CSS entrypoint — only imports from static/css/
static/css/base.css     Variables, resets, typography
static/css/components/  Per-component styling (header, hero, content-sections, footer, …)
static/css/responsive.css  Breakpoint overrides
static/app.js           JS entrypoint — only orchestrates initialization
static/js/data/         page-data.js, speakers.js, semester-schedule.js, archive-schedule.js, templates.js
static/js/render/       Functions that turn data into markup
static/js/ui/           Navigation, icons, scroll behavior
static/js/utils/        CSV parsing, DOM helpers, HTML escaping, materials links
data/                   speakers.csv, speaker-profiles.csv, images, archived HTML
images/                 Banner and general artwork
```

---

## Conventions to follow when changing code

- Keep structure (`index.html`), markup fragments (`templates/`), styling (`static/css/`), behavior (`static/js/`), and content data (`data/`, `page-data.js`) separated.
- No inline `<style>` or `<script>` in `index.html` or the templates. Add a file under `static/css/` or `static/js/` and import it from `static/styles.css` or `static/app.js`.
- Do not duplicate card or button markup inside JavaScript when a template in `templates/` can be used.
- Escape any user-facing string interpolated into markup with `escapeHtml` from `static/js/utils/html.js`.
- Use relative or root-relative URLs (`/schedule/`, `data/speakers.csv`) — never absolute paths to a local machine.

### Design rules

- FSU garnet and gold remain the primary brand colors.
- The AI-energy banner artwork is used for the immersive hero and footer sections.
- Cards stay at or below `8px` border radius.
- Use semantic headings and landmark sections.
- Avoid decorative markup that neither communicates content nor supports layout.
