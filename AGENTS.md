# Rules for coding agents

This is a static GitHub Pages site for the FSU SC Artificial Intelligence Seminar.
[USAGE.md](USAGE.md) is the full guide to content edits, code layout and design and
motion rules. Read the section relevant to your task before changing anything. This
file holds the rules that are easy to break and expensive to get wrong.

## The CSVs are a contract with Otto

`data/speakers.csv` and `data/speaker-profiles.csv` are also read by **Otto**, the
seminar's announcement bot. Otto fetches them raw from GitHub every five minutes and
sends Discord and email announcements from them. A change here reaches Otto within
about ten minutes of being pushed to `main`, with no deploy.

- **Columns may be added, never renamed, reordered or removed.** Both readers match by header name, so adding an optional column at the end is safe. Anything else breaks Otto.
- **Any change to the CSV format must be mirrored in Otto's `docs/upstream-csv-contract.md`.** Tell the user when a change needs that, and do not mark the work finished until they know.
- **Keep the file valid CSV.** Quote any field containing a comma. Otto rejects a malformed file outright and keeps announcing from the last good copy. Edit by hand, not through a spreadsheet round-trip.

## Cancelling or postponing a talk

When the user says a talk is cancelled, postponed, called off or "not happening",
whether for weather, illness or anything else:

1. Set `status` to `cancelled` or `postponed` on **every row with that talk's date**, and set `status_reason` to a short noun phrase (`Hurricane Isaias`, `Speaker illness`).
2. **Never delete the row.** Otto treats a vanished row differently from a cancelled one, and the site loses the history.
3. **Never change `talk_date`** to postpone. Otto keys events on season and date. Leave the old row `postponed`, and add a new row if there is a new date.
4. **Never put the reason in `talk_title`.** Titles containing *break*, *holiday*, *recess* or *no classes* turn into break rows.
5. To un-cancel, clear both cells.
6. The allowed status values are blank, `cancelled` and `postponed`. Do not invent new ones (`delayed`, `moved`, `canceled`) without the user agreeing to add them to Otto first. The site and Otto both treat an unknown value as cancelled.
7. Verify (below) and tell the user what the site now shows. Remind them that Otto stops announcing only if its status support is deployed.

Everything visual follows from those two cells: the table, cards, hero, front-page
notice, storm layer, archive, flyers and JSON-LD. Do not add a separate announcement
file or a hardcoded banner for a cancellation.

## Seasonal themes are never paused

`static/js/seasonal/` runs date-driven themes (Halloween, Día de Muertos,
Thanksgiving, Winter, Lunar New Year). Other features, including notices and weather
visuals, **stack on top** of the active theme. They never pause, replace or reorder
it. Do not change `bindSeasons()` selection to make room for something else.

## Verifying a change

- Serve with the `site` configuration in `.claude/launch.json` (port 4173, no-store caching). Do not open `index.html` from disk.
- Check every page the change touches: `/`, `/schedule/`, `/speakers/`, `/archive/`, `/flyers/`. Check the console for errors at desktop and at 375px width.
- For a status change, confirm that:
  - the row is struck through and tagged;
  - **Next up** moved to the next talk going ahead;
  - the hero card still shows the talk, in its cancelled state, pointing at the next seminar;
  - the notice appears if the talk is within 7 days.
- Leave no test edits in `data/`. If you flag a row only to test, restore it before finishing.

## Code conventions

These are summarised from USAGE.md, which is authoritative.

- Keep the layers separate: structure (`index.html`, page shells), fragments (`templates/`), style (`static/css/`), behaviour (`static/js/`), data (`data/`, `static/js/data/page-data.js`).
- No new inline `<style>` or `<script>` in HTML. The motion boot script in each `<head>` and the static JSON-LD block are the only existing ones.
- Escape every interpolated string with `escapeHtml`.
- Use the motion tokens in `static/css/components/motion.css`. Respect `prefers-reduced-motion`. Reveals animate `opacity`, `translate` and `scale`, never `transform`.
- Cards stay at or below 8px radius. FSU garnet and gold are the brand. Status and warning states use the `--storm-*` tokens.
- Do not commit or push unless the user asks.
