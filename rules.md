# FSU SC Artificial Intelligence Seminar Front Page Rules

This directory is a static site. Keep structure, reusable markup, styling, behavior, and content data separated so the front page can be maintained without editing every file.

## Directory Structure

- `index.html` is the front-page shell. It should contain semantic page regions, landmark elements, and empty `data-*` mount points for dynamic content.
- `static/` contains browser assets that control behavior and presentation.
- `static/styles.css` is the CSS entrypoint and should only import responsibility-based files from `static/css/`.
- `static/css/` contains visual styling split by base rules, components, and responsive overrides.
- `static/app.js` is the JavaScript entrypoint and should only orchestrate initialization through modules from `static/js/`.
- `static/js/` contains page data, template loading, rendering, UI behavior, and utility modules split by responsibility.
- `templates/` contains reusable HTML fragments rendered by `static/app.js`.
- `data/` contains content data files.
- `data/speaker-profiles.csv` is the source of truth for speaker biographical details (title, affiliation, image, links).
- `data/speakers.csv` lists seminar talks and references speakers by `name`; profile fields are joined at load time.
- `data/speaker-images/` contains speaker headshots referenced by filename from `data/speaker-profiles.csv`.
- `images/` contains general page artwork. The current hero and footer image is `images/banner-wide-half-dark.png`.

## Rendering Rules

- Serve `web/` with a local static server during development. The page loads templates and CSV data with `fetch()`, so opening `index.html` directly from the filesystem will not reliably render all content.
- Keep reusable HTML fragments in `templates/`. Do not duplicate card or button markup inside JavaScript when a template can be used.
- Keep CSS in responsibility-based files under `static/css/`, then import them from `static/styles.css`. Do not add inline styles to `index.html` or the templates.
- Keep JavaScript in responsibility-based modules under `static/js/`, then import them from `static/app.js`. Do not add inline scripts to `index.html`.
- Use relative URLs from the front-page root, such as `templates/speaker-card.html`, `data/speakers.csv`, and `data/speaker-images/example.jpg`.
- Speaker cards and schedules load from `data/speakers.csv` joined with `data/speaker-profiles.csv`. If the CSV files cannot be loaded, the speaker data module may fall back to minimal hard-coded placeholder speakers.

## Speaker profile CSV schema

`data/speaker-profiles.csv` must use this header row:

```csv
name,title,department,affiliation,specialties,email,website,profile_url,image
```

Field rules:

- `name`: Canonical display name for the speaker. Schedule rows in `speakers.csv` should use this exact name when possible.
- `title`: Role or academic title.
- `department`: Department, school, lab, institute, or unit.
- `affiliation`: University, company, or organization.
- `specialties`: Semicolon-separated topics shown on speaker cards (for example `LLMs; retrieval augmented generation`).
- `email`: Optional public speaker email.
- `website`: Optional personal or lab website used for speaker card links.
- `profile_url`: Optional institutional profile URL (used when `website` is empty).
- `image`: Filename only, not a full path. The renderer resolves it as `data/speaker-images/{image}`.

## Speaker schedule CSV schema

`data/speakers.csv` must use this header row:

```csv
season,name,talk_title,talk_date,featured,description,materials
```

Field rules:

- `season`: Semester label in `YYYY-Spring` or `YYYY-Fall` format (for example `2026-Spring`, `2026-Fall`). Calendar semesters run Spring from Dec 21 through Jun 20 and Fall from Jun 21 through Dec 20. Spring label year is the calendar year of Jan–Jun; if the calendar month is December (from Dec 21), use the next calendar year.
- `name`: References one or more rows in `speaker-profiles.csv` by `name`. Separate multiple co-speakers with `;` (for example `Tommie Juzek;Lan Li`).
- `talk_title`: Talk title for schedule and detail pages.
- `talk_date`: ISO date in `YYYY-MM-DD` format.
- `featured`: Use `true` to show the talk on the front-page featured schedule. Use `false` for archive or placeholder rows.
- `description`: Short summary of what the talk covered or will cover, shown on schedule pages.
- `materials`: Optional semicolon-separated links to videos, slides, or other resources from past seminars.

The home page and `/schedule/` show **upcoming** talks for the active semester only (`talk_date` on or after today, matching `season`). If the current semester has no upcoming talks, the site rolls forward to the next semester (Spring → Fall, Fall → next Spring). If neither has scheduled talks, a seasonal break message is shown (Summer before Fall, Winter before Spring). Past semesters remain in the CSV for archive use but are not listed on the live schedule pages.

## Archive page (`/archive/`)

- `archive/index.html` is the archive subpage shell; `static/archive.js` initializes it.
- Past talks load from `data/speakers.csv` (joined with profiles) where `talk_date` is before today.
- Talks are grouped by `season` (`YYYY-Spring` / `YYYY-Fall`), newest semester first, chronological within each semester.
- Materials in the `materials` column are split on `;` and rendered as link chips (absolute URLs or site-relative paths).
- The archive includes all past CSV rows; `/speakers/` still lists only names present in `speaker-profiles.csv`.

## Current Contact Links

- Discord invite: `https://discord.com/invite/raTxTXmM5B`
- Mailing list email: `gerlebacher@fsu.edu`
- Zoom room: `https://fsu.zoom.us/j/9038137210`
- Building location: `https://goo.gl/maps/BJLxE3Q7H1MTBqMu6`

## Design Rules

- Preserve FSU garnet and gold as the primary brand colors.
- Use the existing AI-energy banner artwork for immersive hero and footer sections.
- Keep cards at or below `8px` border radius.
- Use semantic headings and landmark sections.
- Avoid adding decorative markup that does not communicate content or support layout.
