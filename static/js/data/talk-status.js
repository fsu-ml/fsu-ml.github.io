/**
 * Whether a talk is still going ahead.
 *
 * speakers.csv carries two optional columns, `status` and `status_reason`. A
 * blank status is the normal case: the talk happens as listed. Anything else
 * means it does not, and the same cell is read by the Otto announcement bot to
 * stop announcing it - so the vocabulary here must stay in step with Otto's.
 *
 * The recognised values are a closed list. An unrecognised value still means
 * "not going ahead" (shown as cancelled, with a console warning), because a
 * typo in the cell must never make a called-off talk look scheduled. Otto
 * applies the same rule and refuses to announce it.
 */

// Holiday and break rows are placeholders, not talks. They are matched on the
// title, which is why a cancellation reason must never be written into it.
export const isBreakEntry = (talk = {}) =>
  /\b(no classes|holiday|break|recess)\b/i.test(talk.talkTitle || "");

export const TALK_STATUSES = {
  cancelled: { label: "Cancelled", schema: "https://schema.org/EventCancelled" },
  postponed: { label: "Postponed", schema: "https://schema.org/EventPostponed" }
};

const SCHEDULED_SCHEMA = "https://schema.org/EventScheduled";

const warned = new Set();

export const normalizeStatus = (value = "") => {
  const status = String(value).trim().toLowerCase();
  if (!status || TALK_STATUSES[status]) {
    return status;
  }
  if (!warned.has(status)) {
    warned.add(status);
    console.warn(`speakers.csv: unknown status "${value}", treating the talk as cancelled.`);
  }
  return "cancelled";
};

export const talkStatus = (talk = {}) => normalizeStatus(talk.status);

export const isCalledOff = (talk = {}) => Boolean(talkStatus(talk));

export const statusLabel = (talk = {}) => TALK_STATUSES[talkStatus(talk)]?.label || "";

export const statusReason = (talk = {}) => String(talk.statusReason || "").trim();

export const statusSchema = (talk = {}) =>
  TALK_STATUSES[talkStatus(talk)]?.schema || SCHEDULED_SCHEMA;

// Only the icon and the hero's storm layer depend on this, so a reason that
// slips past the pattern just renders with the plain calendar mark.
const WEATHER_PATTERN = /\b(hurricane|tropical storm|tropical depression|storm|flood|tornado|weather)\b/i;

export const isWeatherReason = (talk = {}) => WEATHER_PATTERN.test(statusReason(talk));
