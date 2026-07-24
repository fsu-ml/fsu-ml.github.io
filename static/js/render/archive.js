import { groupPastTalksBySeason } from "../data/archive-schedule.js";
import { loadSpeakersFromCsv } from "../data/speakers.js";
import { renderArchiveSpeakerLine } from "./speaker-links.js";
import { qs } from "../utils/dom.js";
import { escapeHtml } from "../utils/html.js";
import { parseMaterialLinks } from "../utils/materials.js";

const monthLabels = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];
const monthNames = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December"
];

const parseIsoDate = (value = "") => {
  const [year, month, day] = value.split("-").map(Number);
  if (!year || !month || !day) {
    return null;
  }
  return { year, month, day };
};

const dateBadge = (value = "") => {
  const parsed = parseIsoDate(value);
  if (!parsed) {
    return { month: "TBA", day: "" };
  }
  return {
    month: monthLabels[parsed.month - 1],
    day: String(parsed.day).padStart(2, "0")
  };
};

const readableDate = (value = "") => {
  const parsed = parseIsoDate(value);
  if (!parsed) {
    return "Date TBA";
  }
  return `${monthNames[parsed.month - 1]} ${parsed.day}, ${parsed.year}`;
};

const formatTalkCount = (count = 0) => {
  if (count === 1) {
    return "1 talk";
  }
  return `${count} talks`;
};

const renderMaterials = (materials = "") => {
  const links = parseMaterialLinks(materials);
  if (!links.length) {
    return "";
  }
  return `
    <ul class="archive-materials">
      ${links
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
    <article class="archive-talk">
      <div class="archive-talk-date" aria-label="${escapeHtml(readableDate(talk.talkDate))}">
        <span class="archive-date-month">${escapeHtml(badge.month)}</span>
        <span class="archive-date-day">${escapeHtml(badge.day)}</span>
      </div>
      <div class="archive-talk-main">
        <h3 class="archive-talk-title">${escapeHtml(talk.talkTitle || "Talk TBA")}</h3>
        ${description}
      </div>
      <div class="archive-talk-meta">
        <p class="archive-talk-speaker-line">${renderArchiveSpeakerLine(talk)}</p>
        ${renderMaterials(talk.materials)}
      </div>
    </article>
  `;
};

const renderSemesterSection = (semester) => `
  <section id="${escapeHtml(semester.anchorId)}" class="archive-semester" aria-labelledby="${escapeHtml(semester.anchorId)}-title">
    <header class="archive-semester-header">
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
      <p class="section-kicker">Past Seminars</p>
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
