import {
  isCalledOff,
  isWeatherReason,
  statusLabel,
  statusReason,
  talkStatus
} from "../data/talk-status.js";
import { icon } from "../ui/icons.js";
import { escapeHtml } from "../utils/html.js";
import { readableDate } from "../utils/dates.js";

/**
 * The markup every renderer uses for a talk that is not going ahead, so the
 * schedule table, talk cards, hero, archive and flyers all say it the same way.
 * Each helper renders nothing for a talk that is going ahead, so callers can
 * interpolate them unconditionally.
 */

export const statusIconName = (talk = {}) => (isWeatherReason(talk) ? "storm" : "calendar-off");

/** `data-event-status` / `data-status-reason`, for anything reading the DOM. */
export const statusDataAttrs = (talk = {}) => {
  if (!isCalledOff(talk)) {
    return ' data-event-status="scheduled"';
  }
  const reason = statusReason(talk);
  return ` data-event-status="${escapeHtml(talkStatus(talk))}"${
    reason ? ` data-status-reason="${escapeHtml(reason)}"` : ""
  }`;
};

export const statusClasses = (talk = {}) =>
  isCalledOff(talk) ? ["is-called-off", `is-${talkStatus(talk)}`] : [];

/**
 * The title, struck through. The strike is visual only, so the status is also
 * spoken ahead of the title rather than left to the <s> element, which most
 * screen readers do not announce.
 */
export const statusTitleMarkup = (talk = {}, title = talk.talkTitle || "") => {
  if (!isCalledOff(talk)) {
    return escapeHtml(title);
  }
  return `<span class="sr-only">${escapeHtml(statusLabel(talk))}: </span><s>${escapeHtml(title)}</s>`;
};

export const statusTagMarkup = (talk = {}) =>
  isCalledOff(talk)
    ? `<span class="status-tag">${escapeHtml(statusLabel(talk))}</span>`
    : "";

export const statusReasonMarkup = (talk = {}, tag = "span") => {
  const reason = statusReason(talk);
  if (!isCalledOff(talk) || !reason) {
    return "";
  }
  return `<${tag} class="status-reason${isWeatherReason(talk) ? " is-weather" : ""}">${icon(
    statusIconName(talk)
  )}<span>${escapeHtml(reason)}</span></${tag}>`;
};

export const shortDate = (talkDate = "") => readableDate(talkDate).replace(/,\s*\d{4}$/, "");

const joinWords = (items = []) => {
  if (items.length < 2) {
    return items.join("");
  }
  return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
};

/**
 * One sentence covering a run of called-off talks: "October 9 seminar
 * cancelled due to Hurricane Isaias", "October 9 and October 16 seminars
 * postponed".
 */
export const calledOffHeadline = (talks = []) => {
  const statuses = new Set(talks.map(talkStatus));
  const reasons = new Set(talks.map(statusReason));
  const verb = statuses.size === 1 ? statusLabel(talks[0]).toLowerCase() : "called off";
  const noun = talks.length > 1 ? "seminars" : "seminar";
  const dates = joinWords(talks.map((talk) => shortDate(talk.talkDate)));
  const reason = reasons.size === 1 ? [...reasons][0] : "";
  return `${dates} ${noun} ${verb}${reason ? ` due to ${reason}` : ""}`;
};

