import { groupPastTalksBySeason } from "../data/archive-schedule.js";
import { loadSpeakersFromCsv } from "../data/speakers.js";
import { renderArchiveSpeakerLine } from "./speaker-links.js";
import { qs } from "../utils/dom.js";
import { escapeHtml } from "../utils/html.js";
import { dateBadge, readableDate } from "../utils/dates.js";
import { parseMaterialLinks } from "../utils/materials.js";
import { flyerTriggerAttrs, parseFlyerList } from "../utils/flyers.js";

const formatTalkCount = (count = 0) => {
  if (count === 1) {
    return "1 talk";
  }
  return `${count} talks`;
};

// Flyers ride in the same chip row as slides and video: from a reader's point
// of view a poster is one more thing the talk left behind.
const renderMaterials = (talk = {}) => {
  const links = parseMaterialLinks(talk.materials);
  const flyers = parseFlyerList(talk.flyers, talk.talkTitle);
  if (!links.length && !flyers.length) {
    return "";
  }
  const flyerItem = flyers.length
    ? `<li><button class="flyer-chip" type="button" data-flyer-set="${escapeHtml(
        flyerTriggerAttrs(flyers, talk.talkTitle)
      )}">${escapeHtml(flyers.length > 1 ? `Flyers (${flyers.length})` : "Flyer")}</button></li>`
    : "";
  return `
    <ul class="archive-materials">
      ${flyerItem}${links
        .map(
          (link) =>
            `<li><a class="archive-material-link" href="${escapeHtml(link.href)}" target="_blank" rel="noopener noreferrer">${escapeHtml(link.label)}</a></li>`
        )
        .join("")}
    </ul>
  `;
};

const renderArchiveTalk = (talk) => {
  const badge = dateBadge(talk.talkDate);
  const description = talk.description
    ? `<p class="archive-talk-description">${escapeHtml(talk.description)}</p>`
    : "";

  return `
    <article class="archive-talk" data-reveal="up">
      <div class="archive-talk-date">
        <span class="sr-only">${escapeHtml(readableDate(talk.talkDate))}</span>
        <span class="archive-date-month" aria-hidden="true">${escapeHtml(badge.month)}</span>
        <span class="archive-date-day" aria-hidden="true">${escapeHtml(badge.day)}</span>
      </div>
      <div class="archive-talk-main">
        <h3 class="archive-talk-title">${escapeHtml(talk.talkTitle || "Talk TBA")}</h3>
        ${description}
      </div>
      <div class="archive-talk-meta">
        <p class="archive-talk-speaker-line">${renderArchiveSpeakerLine(talk)}</p>
        ${renderMaterials(talk)}
      </div>
    </article>
  `;
};

const renderSemesterSection = (semester) => `
  <section id="${escapeHtml(semester.anchorId)}" class="archive-semester" aria-labelledby="${escapeHtml(semester.anchorId)}-title">
    <header class="archive-semester-header" data-reveal="up">
      <h2 id="${escapeHtml(semester.anchorId)}-title">${escapeHtml(semester.heading)}</h2>
      <p class="archive-semester-count">${escapeHtml(formatTalkCount(semester.talks.length))}</p>
    </header>
    <div class="archive-talk-list">
      ${semester.talks.map(renderArchiveTalk).join("")}
    </div>
  </section>
`;

export const renderArchive = async () => {
  const intro = qs("[data-archive-intro]");
  const nav = qs("[data-archive-nav]");
  const content = qs("[data-archive-content]");
  if (!content) {
    return;
  }

  if (intro) {
    intro.innerHTML = `
      <p class="section-kicker">Past Semesters</p>
      <h1 id="archive-title">Seminar Archive</h1>
      <p class="archive-intro-copy">Browse past Artificial Intelligence Seminar talks by semester, with speakers and materials.</p>
    `;
  }

  const talks = await loadSpeakersFromCsv({ featuredOnly: false });
  const semesters = groupPastTalksBySeason(talks);

  if (!semesters.length) {
    if (nav) {
      nav.innerHTML = "";
    }
    content.innerHTML = `<p class="archive-empty">No past talks yet.</p>`;
    return;
  }

  if (nav) {
    nav.innerHTML = `
      <nav class="archive-semester-nav" aria-label="Jump to semester">
        ${semesters
          .map(
            (semester) =>
              `<a class="archive-semester-pill" href="#${escapeHtml(semester.anchorId)}">${escapeHtml(semester.heading)}</a>`
          )
          .join("")}
      </nav>
    `;
  }

  content.innerHTML = semesters.map(renderSemesterSection).join("");
};
