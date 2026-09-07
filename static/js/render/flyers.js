import { groupFlyerTalksBySeason } from "../data/flyer-schedule.js";
import { loadSpeakersFromCsv } from "../data/speakers.js";
import { qs } from "../utils/dom.js";
import { readableDate } from "../utils/dates.js";
import { escapeHtml } from "../utils/html.js";
import { flyerTriggerAttrs } from "../utils/flyers.js";

const formatFlyerCount = (count = 0) => (count === 1 ? "1 flyer" : `${count} flyers`);

/**
 * The tile is an <article> with the trigger stretched across it rather than one
 * big <button>, because a button may only contain phrasing content - the talk
 * title would have had to stop being a heading. The stretched `::after` in
 * flyers.css keeps the whole tile clickable anyway.
 */
const renderFlyerCard = (talk) => {
  const [primary] = talk.flyerList;
  const many = talk.flyerList.length > 1;
  const speaker = (talk.name || "").trim();

  return `
    <article class="flyer-card" data-reveal="up">
      <div class="flyer-card-art">
        <img
          src="${escapeHtml(primary.href)}"
          alt=""
          loading="lazy"
          decoding="async"
        >
        ${many ? `<span class="flyer-card-count">${escapeHtml(formatFlyerCount(talk.flyerList.length))}</span>` : ""}
      </div>
      <div class="flyer-card-body">
        <h3 class="flyer-card-title">
          <button
            class="flyer-card-trigger"
            type="button"
            data-flyer-set="${escapeHtml(flyerTriggerAttrs(talk.flyerList, talk.talkTitle))}"
          >${escapeHtml(talk.talkTitle || "Talk")}<span class="sr-only"> — open flyer</span></button>
        </h3>
        ${speaker ? `<p class="flyer-card-speaker">${escapeHtml(speaker)}</p>` : ""}
        <p class="flyer-card-meta">${escapeHtml(readableDate(talk.talkDate))}</p>
      </div>
    </article>
  `;
};

const renderSemesterSection = (semester) => {
  const flyerCount = semester.talks.reduce((total, talk) => total + talk.flyerList.length, 0);
  return `
    <section id="${escapeHtml(semester.anchorId)}" class="archive-semester" aria-labelledby="${escapeHtml(
      semester.anchorId
    )}-title">
      <header class="archive-semester-header" data-reveal="up">
        <h2 id="${escapeHtml(semester.anchorId)}-title">${escapeHtml(semester.heading)}</h2>
        <p class="archive-semester-count">${escapeHtml(formatFlyerCount(flyerCount))}</p>
      </header>
      <div class="flyer-grid">
        ${semester.talks.map(renderFlyerCard).join("")}
      </div>
    </section>
  `;
};

export const renderFlyers = async () => {
  const intro = qs("[data-flyers-intro]");
  const nav = qs("[data-flyers-nav]");
  const content = qs("[data-flyers-content]");
  if (!content) {
    return;
  }

  if (intro) {
    intro.innerHTML = `
      <p class="section-kicker">Seminar Artwork</p>
      <h1 id="flyers-title">Talk Flyers</h1>
      <p class="archive-intro-copy">Posters and announcement artwork for Artificial Intelligence Seminar talks, by semester. Select one to view it full size, or to download a copy for sharing.</p>
    `;
  }

  // featuredOnly:false so a hidden row still contributes its artwork, matching
  // how /archive/ loads.
  const talks = await loadSpeakersFromCsv({ featuredOnly: false });
  const semesters = groupFlyerTalksBySeason(talks);

  if (!semesters.length) {
    if (nav) {
      nav.innerHTML = "";
    }
    content.innerHTML = `<p class="flyers-empty">No flyers have been posted yet.</p>`;
    return;
  }

  if (nav) {
    nav.innerHTML = `
      <nav class="archive-semester-nav" aria-label="Jump to semester">
        ${semesters
          .map(
            (semester) =>
              `<a class="archive-semester-pill" href="#${escapeHtml(semester.anchorId)}">${escapeHtml(
                semester.heading
              )}</a>`
          )
          .join("")}
      </nav>
    `;
  }

  content.innerHTML = semesters.map(renderSemesterSection).join("");
};
