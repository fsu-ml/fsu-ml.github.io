# Otto: reading cancelled and postponed talks

**For:** Otto's coding agent, working in the `seminar_bot` repository.
**From:** the seminar website (`fsu-ml.github.io`), which has already shipped its side.
**Your task:** read this, check it against Otto's current code, and write an
implementation plan. Follow the repo's own rules for tests and docs. Don't implement
until the plan is approved.

---

## 1. What changed upstream

`data/speakers.csv` has **two new optional columns** at the end of the header:

```csv
season,name,talk_title,talk_date,description,materials,event_image,location_note,start_time,location,registration_url,flyers,status,status_reason
```

| Column | Values | Meaning |
|---|---|---|
| `status` | blank, `cancelled` or `postponed` | Blank means the talk goes ahead as listed, which is nearly every row. Any other value means it does **not** go ahead on that date. |
| `status_reason` | short free text, e.g. `Hurricane Isaias`, `Speaker illness` | Why. Readers see it. Usually blank when `status` is blank. |

Example of a flagged row:

```csv
2026-Fall,Xiuwen Liu,Anatomy of an Outlier: …,2026-10-09,"…",,,,,,,2026-10-09_xiuwen_liu.webp|Wide; …,cancelled,Hurricane Isaias
```

Facts about the data:

- Columns are matched by header name, and most rows stop early. A row missing the trailing cells has a blank status. Otto's parser already copes with this.
- **Values are compared case-insensitively after trimming whitespace.**
- **The allowed values are a fixed list:** blank, `cancelled`, `postponed`. The website treats **any other non-blank value as cancelled** and logs a warning. Otto must do the same, so a typo like `canceled` or `cancel` can never let an announcement through.
- **Every row of a date carries the status.** One event, meaning one `(season, talk_date)` group, may span several rows when co-speakers were entered on separate lines. Editors are told to set the status on all of them. Be defensive anyway: if **any** row in the group has a status, the event is not going ahead.
- **A postponement never moves the row's date.** Editors mark the original row `postponed` and add a **new row** on the new date. Otto sees the new row as a separate, ordinary new event. That is intended, because `event_id` comes from `season|talk_date`.
- **Rows are never deleted to cancel.** The existing "vanished upstream" path still exists for genuinely removed rows. Cancellation is now explicit and the row stays.
- **To un-cancel, editors clear both cells.**

## 2. How Otto should read it

**Source of truth.** The raw CSV, which Otto already fetches. Don't scrape the
rendered website.

The site also publishes schema.org JSON-LD and `data-event-status` attributes, but
they're derived from the same cell and exist for search engines and other readers. If
Otto ever needs a cross-check, the JSON-LD `eventStatus` values are:
- `https://schema.org/EventScheduled`
- `https://schema.org/EventCancelled`
- `https://schema.org/EventPostponed`

**Parsing.** Wherever Otto builds an `Event` from a group of rows:
- Read `status` and `status_reason` from the group: the first non-empty value, with any non-empty status winning.
- Normalise the status as described in §1.
- Store both on the event.

Both fields are **reader-visible**, so they belong in whatever drives `content_hash`
(`HASHED_FIELDS` / `content_hash_for`). That way, flipping either one is detected as a
change.

**Announceability.** A cancelled or postponed event is **not announceable**. Give it a
`skip_reason` that names the status and reason, e.g. `cancelled: Hurricane Isaias`, so
`/draft` and the CLI explain why nothing is going out.

## 3. Required behaviour

The decisions below are already made on the website side. Plan to them:

| Situation | Otto must |
|---|---|
| Event flagged before anything was sent | Schedule nothing new. Mark every pending outbox row for the event `cancelled`. `db.cancel_unsent_for_event` exists for this. Send no notice. |
| Event flagged after an advance announcement was sent | Cancel all remaining pending rows (day-of, previews, …). **Post a cancellation notice** to every guild and target that received the earlier announcement. It needs a dedicated template that includes the date, title, speaker and `status_reason`. |
| Flag lands inside the send debounce window (≈15 min before a send) | **A cancellation overrides the debounce.** The debounce protects content edits from racing a send. It must not let a cancelled talk go out. |
| `postponed` | Same as `cancelled` for suppressing sends. The notice wording says "postponed" and doesn't promise a new date. If a new row appears, it is announced normally as its own event. |
| Status cleared (un-cancel) | The event becomes announceable again and normal scheduling resumes. **Never fire an announcement whose window has already passed:** an advance notice due yesterday must not go out late. If a cancellation notice had already been sent, decide in the plan whether to post a "back on" correction. My recommendation is yes, using a short template. |
| `status_reason` edited while cancelled, after the notice went out | Treat it like any other content change on a sent item. Either post a correction through the existing correction machinery, or deliberately ignore it. Say which in the plan. |
| Unknown status value | Treat as cancelled and log a warning that includes the raw value. |
| Malformed CSV | Unchanged: reject the update and keep the last good state. |

Notice tone: plain and calm. The first real use is a hurricane, so don't make it
playful. Suggested shape:

> **Cancelled:** *Anatomy of an Outlier …* with Xiuwen Liu, Fri Oct 9 — Hurricane Isaias. Stay safe; we'll post updates here.

## 4. Things to check in Otto's code before planning

- How `_cancel_vanished`, `TERMINAL_UNSENT` and the `cancelled` outbox state interact.
  - Explicit cancellation should reuse the `cancelled` state where it can.
  - Make sure un-cancel can move rows out of it. The vanish path already relies on `cancelled` not being terminal.
  - Check that a row cancelled by the vanish path and one cancelled by status don't get confused.
- Where the debounce check lives, and whether the tick or send job re-checks event status right before sending. **A final pre-send check is the safety net.** Plan one if it doesn't exist.
- How `_emit_corrections` decides who received an announcement. The cancellation notice must reach exactly those recipients: no one who wasn't announced to, and nobody twice.
- Whether `/draft`, the CLI and any admin views show `skip_reason`.

## 5. Tests the plan should include

1. Parser: blank, `cancelled`, `postponed`, mixed case and whitespace, an unknown value (cancelled plus a warning), and a row missing the trailing columns (blank).
2. Group rule: two rows on one date with only one flagged gives a cancelled event.
3. `content_hash` changes when `status` or `status_reason` changes.
4. Reconcile: flagged before any send leaves all rows cancelled and sends no notice.
5. Reconcile: flagged after the advance announcement leaves the remaining rows cancelled and sends one notice per prior recipient.
6. Flagged inside the debounce window: nothing sends.
7. Pre-send guard: the status flips between reconcile and tick, and nothing sends.
8. Un-cancel: rows reschedule, but no stale past-due announcement fires. Also test the "back on" notice if adopted.
9. Postpone with a new row: the old event is suppressed and the new event is announced normally.
10. Vanished-row behaviour is unchanged (regression).

## 6. Docs and rules to update in Otto's repo

- **`docs/upstream-csv-contract.md`:** add both columns, the fixed list of values, the unknown-value-means-cancelled rule, the any-row-flags-the-group rule, and "postpone means a new row, never a moved date".
- **Otto's agent rules** (CLAUDE.md, AGENTS.md or equivalent):
  - a status means not announceable
  - a cancellation overrides the debounce
  - every send path re-checks status
  - new status values need a contract update on both sides first
  - every status value has a test
- **Operator docs:** how to read `skip_reason`, and the manual stopgap below.

## 7. Rollout

1. **Ship and deploy Otto's status support before any row is flagged in production.** The site already understands the columns, so a row flagged before Otto is deployed shows "Cancelled" on the website while Otto keeps announcing.
2. **Stopgap until then:** cancel the event's pending outbox rows by hand with `cancel_unsent_for_event`, through the CLI or the DB. Post any notice manually. Don't delete the CSV row.
3. After deploy, flag a row and confirm within about 10 minutes that:
   - `/draft` shows the skip reason
   - no pending rows remain for that event
   - the notice went out to the right recipients, if the event was already announced

## 8. Out of scope here

- The website is done. Its behaviour is documented in the site repo's `USAGE.md` ("Cancelling or postponing a talk") and its `AGENTS.md`.
- Changing how `event_id` is derived.
- Statuses beyond `cancelled` and `postponed`. Propose any new value to the site maintainers first, so both sides adopt it together.
