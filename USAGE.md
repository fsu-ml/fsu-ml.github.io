# Editing the FSU SC Artificial Intelligence Seminar Website

This is a static site (GitHub Pages, no build step). Most routine updates — adding a
talk, adding a speaker, swapping a link — are edits to a CSV or to one JavaScript data
file. You should not need to touch HTML for normal content changes.

- [Quick reference: "I want to…"](#quick-reference-i-want-to)
- [Running the site locally](#running-the-site-locally)
- [Adding or editing a talk](#adding-or-editing-a-talk)
- [Adding or editing a speaker](#adding-or-editing-a-speaker)
  - [What to collect before adding a speaker](#what-to-collect-before-adding-a-speaker)
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
| Find out what to ask a new speaker for | [Intake checklist](#what-to-collect-before-adding-a-speaker) |
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
2026-Fall,🦃,Thanksgiving Holiday,2026-11-26,,,thanksgiving.webp
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

### What to collect before adding a speaker

Most of the effort in adding a speaker is *finding* the facts, not typing them. Gather
these up front — from the speaker, or from whoever invited them — and the CSV row takes
about a minute.

**Ask for a link first.** One good URL usually yields `title`, `department`,
`affiliation`, `specialties`, and often `email` in a single pass. "Send me a link to
your faculty page" is far more efficient than asking eight separate questions, and it
gives you a source to check against later.

| # | Ask for | Fills | Needed? |
| --- | --- | --- | --- |
| 1 | Name, spelled as they want it displayed | `name` | Required |
| 2 | **A link to their profile** — faculty page, lab site, or personal site | `website`, `profile_url`, plus most rows below | **Ask for this first** |
| 3 | Current title and rank | `title` | Required |
| 4 | Department, school, or lab | `department` | Required |
| 5 | University, company, or organization | `affiliation` | Required |
| 6 | Three research topics, in their own words | `specialties` | Required |
| 7 | A public email they are happy to have published | `email` | Optional |
| 8 | A headshot, roughly square, at least 400 px | `image` | Strongly preferred |
| 9 | Talk title, date, and a one- or two-sentence description | `speakers.csv` | Required |
| 10 | Slides and video links | `materials` in `speakers.csv` | After the talk |

#### A message you can paste

Send this to a confirmed speaker; the numbered replies map straight onto the columns.

```text
A few details for the seminar website (https://fsu-ml.github.io), whenever you get a moment:

1. Your name, spelled as you'd like it shown
2. A link to your faculty / lab / personal page
3. Your current title
4. Your department and institution
5. Three research areas, in your own words
6. A public email to list (optional — say "skip" to leave it off)
7. A headshot, roughly square and at least 400 px (JPEG or PNG is fine)
8. Your talk title, plus one or two sentences describing it
```

If you are handing this to an assistant or filling it in yourself, give the profile URL
along with the name — that alone is usually enough to complete the row.

#### If you have to look it up yourself

Work down this list and stop at the first source that answers the question:

1. **Institutional directory page** — the most reliable source for title, department,
   and affiliation, and the one to trust when sources disagree. FSU pages render emails
   through JavaScript, so the address will not appear in the visible text; view source
   and search for `@fsu.edu`.
2. **Personal or lab site** — the best source for `specialties` in the speaker's own
   words, and usually the most current for title changes. Many carry a `mailto:` link.
3. **Google Scholar** — good for confirming research areas. No contact details.
4. **LinkedIn** — last resort. It blocks automated fetching, so open it in a browser.

Three rules when filling gaps from the web:

- **Only publish an email that is already public** on the speaker's own page or their
  institution's directory. Do not reconstruct one from a username pattern.
- **Confirm it is the right person** before copying anything. Match on institution and
  research area, not on name alone.
- **Leave a field blank rather than guess.** Every column except `name` degrades
  gracefully: a missing `image` renders a placeholder, and a missing `website` falls
  back to `profile_url`. A wrong title is worse than no title.

#### Keeping profiles current

Titles and affiliations go stale — people are promoted, and postdocs move on. Worth a
pass at the start of each semester over that semester's speakers, checking `title` and
`affiliation` against the institutional page, and confirming that `website` and
`profile_url` still resolve. Personal sites move more often than departmental ones.

---

## Images

| Directory | Used for | Referenced from |
| --- | --- | --- |
| `data/speaker-images/` | Speaker headshots | `image` column of `speaker-profiles.csv` |
| `data/event-images/` | Holiday and break artwork | `event_image` column of `speakers.csv` |
| `images/` | General page artwork (hero and footer use `images/banner-wide-half-dark.webp`) | CSS and `page-data.js` |

Use lowercase, underscore-separated filenames (`jane_doe.webp`). Headshots look best
roughly square; crop before committing rather than resizing in CSS. Reference them by
filename only in the CSVs — the loaders build the full path.

### Two formats for every raster image

Every raster image is committed **twice** — a `.webp` for the website and a `.jpg` of the
same basename and the same pixel dimensions beside it:

```
data/speaker-images/olmo_zavala.webp    <- what the site loads
data/speaker-images/olmo_zavala.jpg     <- what you paste into an email
```

The website uses WebP because it is markedly smaller. Announcement and newsletter emails
go out as **HTML email**, and several major clients — Outlook on Windows above all — still
will not render WebP, so anything reused in an email needs a JPEG twin. Keeping the pair
side by side means nobody has to go hunting for a converter at send time.

Three rules:

- **Same basename, same dimensions, both formats.** No orphans in either direction.
- **The site references `.webp` only.** Nothing in the markup, CSS, or CSVs should point at
  a `.jpg` — the JPEGs exist purely for email. (One exception: `og:image`, below.)
- **Hard ceiling of 500 KB per file**, in either format. Over that, resize or re-encode
  before committing.

### Optimize before committing

Convert with ImageMagick, WebP first, then the JPEG twin from the same source:

```bash
magick input.jpg -resize 400x400\> -quality 82 data/speaker-images/jane_doe.webp
magick data/speaker-images/jane_doe.webp -background white -alpha remove -alpha off \
       -colorspace sRGB -strip -quality 82 -interlace Plane data/speaker-images/jane_doe.jpg
```

The `\>` means "shrink only, never upscale". `-interlace Plane` makes the JPEG
progressive, which renders more gracefully on a slow mail connection. Both formats use the
same quality so the pair looks identical. Target sizes:

| Kind | Max longest edge | Quality | Why |
| --- | --- | --- | --- |
| Speaker headshot | 400 px | 82 | Renders as a 78 px circle; 400 px covers high-DPI screens |
| Event / holiday art | 256 px | 85 | Renders at 64x52 in the schedule table |
| Page artwork (`images/`) | Its rendered width | 82 | E.g. the hero background stays 1920 px wide |

For the WebP that the site actually serves, keep headshots under roughly 40 KB and page
artwork under roughly 250 KB. The 500 KB ceiling above is the hard limit for any single
file; these are the performance budgets for what visitors download.

**Transparency does not survive the JPEG.** JPEG has no alpha channel, so
`-alpha remove -alpha off` flattens it onto a background — white above, matching a typical
email body. Four committed images carry real transparency and are therefore flattened in
their `.jpg` form only:

`images/trichotomy-illustration`, `data/speaker-images/philippe_miron`,
`data/event-images/fsu-homecoming`, `data/event-images/thanksgiving`

The WebP originals keep their alpha and the site is unaffected. If one of these goes into
an email whose background is not white, regenerate that JPEG with a matching
`-background '#rrggbb'` rather than using the committed copy.

**SVGs are not covered by this rule.** `images/*.svg` (the FSU wordmark, Zoom, Discord and
calendar icons, the favicon) have no raster twin. They are interface chrome rather than
content, and a JPEG would lose their transparency. If one is ever needed in an email,
export a **PNG** at the size required — not a JPEG.

**One exception to "the site references WebP only":** `images/banner-wide.jpg` is the
social-preview card behind the `og:image` / `twitter:image` tags in every page's `<head>`.
Those tags must keep pointing at the **`.jpg`**, because LinkedIn, Slack, and some other
link unfurlers still do not render WebP previews reliably. It has a `.webp` twin like
everything else, but nothing references it. Keep it 1200 px wide.

### Auditing the pairs

Before committing new artwork, check nothing is orphaned, mismatched, or oversized:

```bash
for d in images data/speaker-images data/event-images; do
  for f in "$d"/*.webp "$d"/*.jpg; do
    b="${f%.*}"
    [ -e "$b.webp" ] && [ -e "$b.jpg" ] || echo "UNPAIRED: $b"
    [ "$(magick identify -format '%wx%h' "$b.webp")" = "$(magick identify -format '%wx%h' "$b.jpg")" ] \
      || echo "DIMENSION MISMATCH: $b"
  done
done | sort -u
find images data/speaker-images data/event-images -type f \( -name '*.jpg' -o -name '*.webp' \) -size +500k
```

Silence from both commands means every image is paired, matched, and within budget.

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
static/css/components/motion.css  Motion tokens, scroll-reveal states, skeletons, reduced-motion
static/css/responsive.css  Breakpoint overrides
static/app.js           JS entrypoint — only orchestrates initialization
static/js/data/         page-data.js, speakers.js, semester-schedule.js, archive-schedule.js, templates.js
static/js/render/       Functions that turn data into markup
static/js/ui/           Navigation, icons, scroll behavior, reveal.js, chrome.js
static/js/utils/        CSV parsing, DOM helpers, HTML escaping, materials links
data/                   speakers.csv, speaker-profiles.csv, images, archived HTML
images/                 Banner and general artwork
```

---

## Conventions to follow when changing code

- Keep structure (`index.html`), markup fragments (`templates/`), styling (`static/css/`), behavior (`static/js/`), and content data (`data/`, `page-data.js`) separated.
- No inline `<style>` or `<script>` in `index.html` or the templates. Add a file under `static/css/` or `static/js/` and import it from `static/styles.css` or `static/app.js`.
  - The one sanctioned exception is the four-line motion boot script in each page `<head>`. It must run before the stylesheet is applied, and an external file would add a round-trip during which the entrance states are unstyled — the exact flash it exists to prevent. Do not add a second exception.
- Do not duplicate card or button markup inside JavaScript when a template in `templates/` can be used.
- Escape any user-facing string interpolated into markup with `escapeHtml` from `static/js/utils/html.js`.
- Use relative or root-relative URLs (`/schedule/`, `data/speakers.csv`) — never absolute paths to a local machine.

### Design rules

- FSU garnet and gold remain the primary brand colors.
- The AI-energy banner artwork is used for the immersive hero and footer sections.
- Cards stay at or below `8px` border radius.
- Use semantic headings and landmark sections.
- Avoid decorative markup that neither communicates content nor supports layout.

### Motion rules

Timings, easings and travel distances are tokens in `static/css/components/motion.css`. Use them; do not write a new `0.15s ease` by hand.

- To make something animate in on scroll, add `data-reveal="up|fade|row|left|right|scale"` to the element — in a template, in `index.html`, or in a renderer's markup string. Nothing else is required; `activateMotion()` picks it up after each render pass.
- Never nest one `[data-reveal]` inside another. The outer entrance would run first and the inner one would animate against a moving parent.
- Reveals animate `opacity` plus the independent `translate`/`scale` properties, never `transform`. `transform` is left free for component hover lifts. A reveal that used `transform` would silently disable the hover on every card it touched.
- If a renderer emits a container that starts empty, give it `data-skeleton` and a `--skeleton-height` matching the filled height, so the layout does not jump when data lands.
- Portraits added by a renderer take `data-fade` to fade in as they decode.
- Hover vocabulary is split on purpose. Cards *about a thing* (talks, features, community, guide) lift on hover. Cards *about a person* (`.speaker-directory-card`) do not move at all — a brand rule wipes across the top edge and a gold halo blooms around the portrait. Do not give presenter cards a lift, and never scale someone's photograph.
