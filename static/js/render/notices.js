import { isUpcoming } from "../data/semester-schedule.js";
import { loadSpeakersFromCsv } from "../data/speakers.js";
import { isBreakEntry, isCalledOff, isWeatherReason, statusReason } from "../data/talk-status.js";
import { icon } from "../ui/icons.js";
import { qs } from "../utils/dom.js";
import { escapeHtml } from "../utils/html.js";
import { readableDate } from "../utils/dates.js";
import {
  calledOffHeadline,
  statusDataAttrs,
  statusIconName,
  statusTagMarkup
} from "./talk-status-markup.js";

/**
 * The front-page announcement for talks that are not going ahead.
 *
 * There is no separate announcement file: the notice is derived from the
 * `status` column of speakers.csv, the same cell Otto reads, so the site and
 * the bot cannot disagree about what was called off. It appears a week before
 * the talk and stays up for two days after it would have ended, so someone who
 * missed the news still finds out why nothing happened.
 *
 * Weather reasons add a rain layer to the hero. It is stacked on top of the
 * seasonal theme rather than replacing it - the seasonal layer is never paused.
 */

const DAY_MS = 24 * 60 * 60 * 1000;
export const NOTICE_LEAD_DAYS = 7;
export const NOTICE_TAIL_HOURS = 48;

export const noticeTalks = (talks = [], today = new Date()) => {
  const opens = new Date(today.getTime() + NOTICE_LEAD_DAYS * DAY_MS);
  const closes = new Date(today.getTime() - NOTICE_TAIL_HOURS * 60 * 60 * 1000);
  return talks
    .filter((talk) => talk.talkDate && isCalledOff(talk) && !isBreakEntry(talk))
    .filter((talk) => !isUpcoming(talk.talkDate, opens) && isUpcoming(talk.talkDate, closes))
    .sort((left, right) => left.talkDate.localeCompare(right.talkDate));
};

const noticeItemMarkup = (talk) => `
  <li class="site-notice-item"${statusDataAttrs(talk)}>
    <span class="site-notice-date">${escapeHtml(readableDate(talk.talkDate))}</span>
    <span class="site-notice-talk">
      <s>${escapeHtml(talk.talkTitle)}</s>${talk.name ? ` <span class="site-notice-speaker">&middot; ${escapeHtml(talk.name)}</span>` : ""}
    </span>
    ${statusTagMarkup(talk)}
  </li>
`;

// The homepage version is deliberately short - a headline and one line - since
// the hero card beside it already shows the talk, struck through. /schedule/
// asks for the detailed version, which names each talk that is off.
const noticeMarkup = (talks, { headingLevel, detailed }) => {
  const weather = talks.some(isWeatherReason);
  const reasons = [...new Set(talks.map(statusReason).filter(Boolean))];
  const heading = `h${headingLevel}`;
  const classes = ["site-notice", weather ? "is-weather" : "", detailed ? "is-detailed" : ""]
    .filter(Boolean)
    .join(" ");
  const detailMarkup = detailed
    ? `
        <ul class="site-notice-list">
          ${talks.map(noticeItemMarkup).join("")}
        </ul>`
    : "";
  return `
    <aside class="${classes}" aria-labelledby="site-notice-title"${
      talks.length === 1 ? statusDataAttrs(talks[0]) : ""
    } data-reveal="fade">
      <div class="site-notice-visual" aria-hidden="true">
        ${icon(weather ? "storm" : statusIconName(talks[0]))}
      </div>
      <div class="site-notice-copy">
        ${
          detailed
            ? `<p class="site-notice-kicker">Seminar update${
                reasons.length === 1 ? ` &middot; ${escapeHtml(reasons[0])}` : ""
              }</p>`
            : ""
        }
        <${heading} class="site-notice-title" id="site-notice-title">${escapeHtml(calledOffHeadline(talks))}</${heading}>${detailMarkup}
        <p class="site-notice-foot">Everything else is on schedule. Stay safe &mdash; we will post updates here and on Discord.</p>
      </div>
    </aside>
  `;
};

// Rain and a darkened sky behind the hero copy. Absolutely positioned and
// inserted just before .hero-inner, so it paints over the seasonal scene and
// under the content. Purely decorative.
const stormLayerMarkup = () => `
  <div class="hero-storm" aria-hidden="true" data-hero-storm>
    <span class="hero-storm-rain"></span>
  </div>
`;

export const renderSiteNotice = async ({ headingLevel = 2, detailed = false } = {}) => {
  const mount = qs("[data-site-notice]");
  if (!mount) {
    return;
  }

  const talks = noticeTalks(await loadSpeakersFromCsv({ featuredOnly: false }));
  if (!talks.length) {
    mount.hidden = true;
    mount.innerHTML = "";
    return;
  }

  mount.innerHTML = noticeMarkup(talks, { headingLevel, detailed });
  mount.hidden = false;

  const hero = mount.closest(".hero");
  if (hero && talks.some(isWeatherReason) && !hero.querySelector("[data-hero-storm]")) {
    hero.querySelector(".hero-inner")?.insertAdjacentHTML("beforebegin", stormLayerMarkup());
    hero.classList.add("has-storm");
  }
};
