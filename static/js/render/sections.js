import { pageData } from "../data/page-data.js";
import {
  formatSeasonHeading,
  getBreakMessage,
  getCalendarSemester,
  isUpcoming,
  resolveDisplaySemester
} from "../data/semester-schedule.js";
import {
  getTalkSpeakers,
  isTbaSpeaker,
  loadSpeakersFromCsv,
  loadUniqueSpeakersFromCsv,
  speakerProfileHref
} from "../data/speakers.js";
import { renderSpeakerNameLinks } from "./speaker-links.js";
import { icon } from "../ui/icons.js";
import { activateMotion } from "../ui/reveal.js";
import { qs } from "../utils/dom.js";
import { escapeHtml, renderTemplate } from "../utils/html.js";
import { renderButton } from "./buttons.js";

const footerLogoUrl = new URL("../../../images/FSU-Scientific-Computing.svg", import.meta.url).href;
const findSection = (id) => pageData.sections.find((section) => section.id === id);
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

const speakerInitials = (name = "") => {
  if (/\bTBA\b/i.test(name)) {
    return "?";
  }
  return name
    .replace(/^Dr\.\s+/i, "")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0])
    .join("");
};

const speakerKey = (name = "") => name.replace(/^Dr\.\s+/i, "").trim().toLowerCase();

const titleCase = (text = "") =>
  text
    .trim()
    .split(/\s+/)
    .map((word) => word.replace(/^\p{L}/u, (letter) => letter.toUpperCase()))
    .join(" ");

const formatSpecialties = (topic = "") =>
  topic
    .split(";")
    .map(titleCase)
    .filter(Boolean)
    .join(", ");

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

const topicTag = (topic = "") => titleCase(topic.split(";").find(Boolean) || "AI Seminar");

const speakerWebsite = (speaker = {}, fallback = "/speakers/") => {
  const website = speaker.website || fallback;
  return website.startsWith("#") ? fallback : website;
};

const nextSeminarLabel = (talkDate = "", fallbackDateTime = "") => {
  const dateLabel = talkDate ? readableDate(talkDate) : "";
  if (dateLabel && dateLabel !== "Date TBA") {
    return `Next Seminar: ${dateLabel}`;
  }
  const fallbackDate = fallbackDateTime.split(/\s+-\s+/)?.[0]?.trim();
  if (fallbackDate) {
    return `Next Seminar: ${fallbackDate}`;
  }
  return "Next Seminar: TBA";
};

// Speakers whose title already carries their affiliation (lab directors, and
// anyone without a home department) skip the department half of the line.
const formatDirectoryRole = (speaker = {}) =>
  [speaker.title || "Speaker", speaker.department].filter(Boolean).join(" · ");

const formatDirectoryAffiliation = (speaker = {}) => speaker.affiliation || "Affiliation TBA";

const formatTalkCountLabel = (count = 0) => {
  if (count === 1) {
    return "1 Talk Given";
  }
  return `${count} Talks Given`;
};

const pageScheduleItems = () => findSection("schedule").items || [];

const pageScheduleItemsAsTalks = () =>
  pageScheduleItems().map((item) => ({
    name: item.speaker,
    title: "Speaker",
    department: "Department TBA",
    affiliation: item.affiliation,
    topic: item.tag,
    talkTitle: item.title,
    talkDate: "",
    season: "",
    description: item.description || "",
    image: "",
    website: "/speakers/"
  }));

const loadResolvedSchedule = async () => {
  const csvSpeakers = await loadSpeakersFromCsv({ featuredOnly: false });
  const eligible = csvSpeakers.filter((speaker) => speaker.talkTitle && speaker.talkDate);
  if (!eligible.length) {
    return null;
  }
  const resolved = resolveDisplaySemester(eligible);
  const talksForSeason = (key) =>
    eligible
      .filter((talk) => talk.season === key)
      .sort((left, right) => left.talkDate.localeCompare(right.talkDate));

  // The roster stays pinned to the calendar semester so speakers remain listed for
  // the whole term, even once their individual talks have already happened.
  const calendarSeasonKey = getCalendarSemester().seasonKey;
  const calendarSeasonTalks = talksForSeason(calendarSeasonKey);
  const rosterSeasonKey = calendarSeasonTalks.length ? calendarSeasonKey : resolved.seasonKey;

  return {
    ...resolved,
    seasonTalks: talksForSeason(resolved.seasonKey),
    rosterSeasonKey,
    rosterTalks: calendarSeasonTalks.length ? calendarSeasonTalks : talksForSeason(resolved.seasonKey)
  };
};

const isBreakEntry = (talk = {}) =>
  /\b(no classes|holiday|break|recess)\b/i.test(talk.talkTitle || "");

// Talks inherit one standing time and room from page-data, so a session that
// moves needs its own note. Where a row sets one it replaces the default rather
// than sitting beside it, since showing both would contradict.
const locationNote = (talk = {}) => (talk.locationNote || "").trim();

const locationNoteMarkup = (talk, tag = "span") => {
  const note = locationNote(talk);
  return note
    ? `<${tag} class="location-note">${icon("map-pin")}<span>${escapeHtml(note)}</span></${tag}>`
    : "";
};

const scheduleRowSpeakerCell = (talk) => {
  if (isBreakEntry(talk)) {
    if (talk.eventImage) {
      return `<img class="schedule-table-event-image" src="${escapeHtml(talk.eventImage)}" alt="">`;
    }
    return '<span class="schedule-table-speaker-none" aria-label="No seminar">&mdash;</span>';
  }
  const talkSpeakers = getTalkSpeakers(talk).filter((speaker) => !isTbaSpeaker(speaker.name));
  if (!talkSpeakers.length) {
    return '<span class="schedule-table-speaker-tba">To be announced</span>';
  }
  return renderSpeakerNameLinks({ ...talk, speakers: talkSpeakers }, "schedule-table-speaker-link");
};

const scheduleRowClasses = (talk, nextTalkDate) => {
  const classes = ["schedule-table-row"];
  if (isBreakEntry(talk)) {
    classes.push("is-break");
  }
  if (!isUpcoming(talk.talkDate)) {
    classes.push("is-past");
  } else if (talk.talkDate === nextTalkDate) {
    classes.push("is-next");
  }
  return classes.join(" ");
};

const renderScheduleRow = (talk, nextTalkDate) => {
  const badge = dateBadge(talk.talkDate);
  const isNext = talk.talkDate === nextTalkDate && !isBreakEntry(talk);
  return `
    <tr class="${scheduleRowClasses(talk, nextTalkDate)}" data-reveal="row">
      <th class="schedule-table-date" scope="row">
        <span class="schedule-table-date-badge" aria-hidden="true">
          <span class="date-month">${escapeHtml(badge.month)}</span>
          <span class="date-day">${escapeHtml(badge.day)}</span>
        </span>
        <span class="sr-only">${escapeHtml(readableDate(talk.talkDate))}</span>
      </th>
      <td class="schedule-table-topic">
        <span class="schedule-table-title">${escapeHtml(talk.talkTitle)}</span>
        ${isNext ? '<span class="schedule-table-next-tag">Next up</span>' : ""}
        ${locationNoteMarkup(talk)}
      </td>
      <td class="schedule-table-description">${
        talk.description
          ? escapeHtml(talk.description)
          : '<span class="schedule-table-speaker-none" aria-hidden="true">&mdash;</span>'
      }</td>
      <td class="schedule-table-speaker">${scheduleRowSpeakerCell(talk)}</td>
    </tr>
  `;
};

const renderScheduleTable = (talks) => {
  const nextTalk = talks.find((talk) => isUpcoming(talk.talkDate) && !isBreakEntry(talk));
  const nextTalkDate = nextTalk?.talkDate || "";
  return `
    <div class="schedule-table-wrap">
      <table class="schedule-table">
        <caption class="sr-only">Seminar schedule by date, topic, description, and speaker</caption>
        <thead>
          <tr>
            <th scope="col">Date</th>
            <th scope="col">Topic</th>
            <th scope="col">Description</th>
            <th scope="col">Speaker</th>
          </tr>
        </thead>
        <tbody>
          ${talks.map((talk) => renderScheduleRow(talk, nextTalkDate)).join("")}
        </tbody>
      </table>
    </div>
  `;
};

const renderBreakMarkup = (breakKind) =>
  `<p class="schedule-break-message">${escapeHtml(getBreakMessage(breakKind))}</p>`;

const renderTalkCard = (speaker, details = {}) => {
  const badge = dateBadge(speaker.talkDate);
  const description = speaker.description || details.description || "Talk details coming soon.";

  if (isBreakEntry(speaker)) {
    const eventImageMarkup = speaker.eventImage
      ? `<img src="${escapeHtml(speaker.eventImage)}" alt="${escapeHtml(`${speaker.talkTitle} illustration`)}">`
      : `<span class="talk-event-fallback" aria-hidden="true">${escapeHtml(speaker.name || "")}</span>`;

    return `
      <article class="talk-card talk-card-event" data-reveal="up">
        <div class="talk-card-header">
          <div class="date-badge">
            <span class="sr-only">${escapeHtml(readableDate(speaker.talkDate))}</span>
            <span class="date-month" aria-hidden="true">${escapeHtml(badge.month)}</span>
            <span class="date-day" aria-hidden="true">${escapeHtml(badge.day)}</span>
          </div>
          <h3>${escapeHtml(speaker.talkTitle)}</h3>
        </div>
        ${speaker.description || details.description ? `<p class="talk-description">${escapeHtml(description)}</p>` : ""}
        <div class="talk-card-spacer" aria-hidden="true"></div>
        <div class="talk-event-art">${eventImageMarkup}</div>
      </article>
    `;
  }

  const talkSpeakers = getTalkSpeakers(speaker);
  const primarySpeaker = talkSpeakers[0] || speaker;
  const multiSpeaker = talkSpeakers.length > 1;
  const name = speaker.name || "Speaker TBA";
  const showProfileDetails = !multiSpeaker && (speaker.hasProfile ?? true);
  const initials = escapeHtml(speakerInitials(primarySpeaker.name || name));
  const imageMarkup = primarySpeaker.image
    ? `<img data-fade src="${escapeHtml(primarySpeaker.image)}" alt="${escapeHtml(primarySpeaker.name || name)}" onerror="this.remove(); this.nextElementSibling.hidden = false;">`
    : "";
  const tag = details.tag || topicTag(speaker.topic);
  const titleMarkup =
    showProfileDetails && primarySpeaker.title
      ? ` <span>&middot; ${escapeHtml(primarySpeaker.title)}</span>`
      : "";
  const affiliationMarkup = showProfileDetails
    ? `<span>${escapeHtml(
        [primarySpeaker.department, primarySpeaker.affiliation || "Affiliation TBA"]
          .filter(Boolean)
          .join(" · ")
      )}</span>`
    : "";
  const specialtiesMarkup =
    showProfileDetails && formatSpecialties(speaker.topic || tag)
      ? `<span class="talk-specialties">${escapeHtml(formatSpecialties(speaker.topic || tag))}</span>`
      : "";
  const nameMarkup = multiSpeaker
    ? renderSpeakerNameLinks(speaker)
    : `${escapeHtml(primarySpeaker.name || name)}${titleMarkup}`;
  const speakerRowTag = multiSpeaker ? "div" : "a";
  const speakerRowAttrs = multiSpeaker
    ? ""
    : ` href="${escapeHtml(speakerWebsite(primarySpeaker))}"`;

  return `
    <article class="talk-card" data-reveal="up">
      <div class="talk-card-header">
        <div class="date-badge">
          <span class="sr-only">${escapeHtml(readableDate(speaker.talkDate))}</span>
          <span class="date-month" aria-hidden="true">${escapeHtml(badge.month)}</span>
          <span class="date-day" aria-hidden="true">${escapeHtml(badge.day)}</span>
        </div>
        <h3>${escapeHtml(speaker.talkTitle)}</h3>
      </div>
      ${locationNoteMarkup(speaker, "p")}
      <p class="talk-description">${escapeHtml(description)}</p>
      <div class="talk-card-spacer" aria-hidden="true"></div>
      <${speakerRowTag} class="talk-speaker-row"${speakerRowAttrs}>
        <span class="speaker-media" aria-hidden="true">
          ${imageMarkup}
          <span ${primarySpeaker.image ? "hidden" : ""}>${initials}</span>
        </span>
        <span class="talk-speaker-copy">
          <strong>${nameMarkup}</strong>
          ${affiliationMarkup}
          ${specialtiesMarkup}
        </span>
      </${speakerRowTag}>
    </article>
  `;
};

export const renderNavigation = (activeId = "home") => {
  qs("[data-nav-links]").innerHTML = pageData.navigation.items
    .map(
      (item) =>
        `<a class="nav-link${item.id === activeId ? " is-active" : ""}" href="${escapeHtml(item.href)}">${escapeHtml(
          item.label
        )}</a>`
    )
    .join("");
};

// How long the placeholder takes to fade before the real card replaces it.
// Matches the transition on .seminar-skeleton in seminar-card.css.
const SKELETON_FADE_MS = 240;

// Hands the hero card over from placeholder to content. The placeholder holds
// the card's height while it fades, so the hero does not lurch mid-swap.
const swapSeminarCard = (html) => {
  const mount = qs("[data-next-seminar]");
  if (!mount) {
    return;
  }
  if (!mount.querySelector("[data-seminar-skeleton]")) {
    mount.innerHTML = html;
    activateMotion(mount);
    return;
  }
  mount.classList.add("is-swapping");
  window.setTimeout(() => {
    mount.innerHTML = html;
    mount.classList.remove("is-swapping");
    // This write lands after init()'s activateMotion() pass has already run, so
    // the new subtree needs its own activation - otherwise the speaker portrait
    // never receives .is-loaded and stays fully transparent.
    activateMotion(mount);
  }, SKELETON_FADE_MS);
};

// Wraps each word so the title can arrive a word at a time. Each word is
// escaped individually; splitting on whitespace after escaping would be safe
// too, since no entity contains a space, but this keeps the intent obvious.
// The index is capped so a long title does not animate for several seconds.
const typedTitle = (text = "") => {
  const words = String(text).trim().split(/\s+/).filter(Boolean);
  if (!words.length) {
    return escapeHtml(text);
  }
  return words
    .map(
      (word, index) =>
        `<span class="typed-word" style="--word-index:${Math.min(index, 14)}">${escapeHtml(word)}</span>`
    )
    .join(" ");
};

export const renderHero = async (templates) => {
  qs("[data-hero-actions]").innerHTML = pageData.hero.content.buttons
    .map((button) => renderButton(templates.button, button))
    .join("");

  const seminar = pageData.hero.nextSeminarCard;
  const resolved = await loadResolvedSchedule();

  if (resolved?.status === "break") {
    swapSeminarCard(`
      <div class="seminar-card-body">
        <p class="seminar-label">${icon("calendar")}<span>${escapeHtml(nextSeminarLabel())}</span></p>
        <p class="seminar-description schedule-break-message">${escapeHtml(getBreakMessage(resolved.breakKind))}</p>
      </div>
    `);
    return;
  }

  const nextTalk = resolved?.talks?.[0];
  const scheduleSection = findSection("schedule");
  const scheduleItem =
    scheduleSection.items?.find((item) => item.title === nextTalk?.talkTitle) || {};
  const useCsv = Boolean(nextTalk);
  const speaker = nextTalk || {};
  const talkSpeakers = getTalkSpeakers(speaker);
  const primarySpeaker = talkSpeakers[0] || speaker;
  const multiSpeaker = talkSpeakers.length > 1;
  const speakerName = speaker.name || seminar.speaker;
  const showProfileDetails = !multiSpeaker && (!useCsv || speaker.hasProfile);
  const speakerDepartment = primarySpeaker.department || "";
  const speakerAffiliation = primarySpeaker.affiliation || seminar.affiliation;
  const speakerSpecialties = formatSpecialties(speaker.topic || scheduleItem.tag || "");
  const talkTitle = speaker.talkTitle || seminar.talkTitle;
  const talkDescription =
    speaker.description || scheduleItem.description || "Talk description coming soon.";
  const dateTime = useCsv
    ? `${readableDate(speaker.talkDate)} - 12:00 PM ET`
    : seminar.dateTime;
  const seminarLabel = nextSeminarLabel(speaker.talkDate, seminar.dateTime);
  // A talk carrying its own note keeps neither the standing time nor the
  // standing room, because both would be wrong for it. The date still shows in
  // the card label above, so nothing is lost by dropping the clock row.
  const talkLocationNote = locationNote(speaker);
  const metaRowsMarkup = talkLocationNote
    ? `<div class="meta-row meta-row-note"><span class="meta-icon">${icon(
        "map-pin"
      )}</span><span>${escapeHtml(talkLocationNote)}</span></div>`
    : `<div class="meta-row"><span class="meta-icon">${icon("clock")}</span><span>${escapeHtml(
        dateTime
      )}</span></div>
        <div class="meta-row"><span class="meta-icon">${icon("map-pin")}</span><span>${(
          seminar.locationLinks || []
        )
          .map((link) => `<a href="${escapeHtml(link.href)}">${escapeHtml(link.label)}</a>`)
          .join(" + ")}</span></div>`;
  const speakerInitialsMarkup = escapeHtml(speakerInitials(primarySpeaker.name || speakerName));
  const speakerImageSrc = primarySpeaker.image || (!useCsv ? seminar.speakerImage : "");
  const speakerImageMarkup = speakerImageSrc
    ? `<img data-fade src="${escapeHtml(speakerImageSrc)}" alt="${escapeHtml(primarySpeaker.name || speakerName)}" onerror="this.remove(); this.nextElementSibling.hidden = false;">`
    : "";
  const speakerTitleMarkup =
    showProfileDetails && primarySpeaker.title
      ? ` <span>&middot; ${escapeHtml(primarySpeaker.title)}</span>`
      : "";
  const speakerNameMarkup = multiSpeaker
    ? renderSpeakerNameLinks(speaker, "seminar-speaker-name-link")
    : `${escapeHtml(speakerName)}${speakerTitleMarkup}`;
  const speakerAffiliationMarkup =
    showProfileDetails && (speakerDepartment || speakerAffiliation)
      ? `<p class="seminar-affiliation">${escapeHtml(
          [speakerDepartment, speakerAffiliation].filter(Boolean).join(" · ")
        )}</p>`
      : "";
  const speakerSpecialtiesMarkup =
    showProfileDetails && speakerSpecialties
      ? `<p class="seminar-specialties">${escapeHtml(speakerSpecialties)}</p>`
      : "";

  swapSeminarCard(`
    <div class="seminar-card-body">
      <p class="seminar-label">${icon("calendar")}<span>${escapeHtml(seminarLabel)}</span></p>
      <h2 class="seminar-title">${typedTitle(talkTitle)}</h2>
      <p class="seminar-description">${escapeHtml(talkDescription)}</p>
      <div class="seminar-card-spacer" aria-hidden="true"></div>
      <div class="seminar-speaker-row">
        <div class="seminar-speaker-photo" aria-hidden="true">
          ${speakerImageMarkup}
          <span ${speakerImageSrc ? "hidden" : ""}>${speakerInitialsMarkup}</span>
        </div>
        <div class="seminar-speaker-copy">
          <p class="seminar-speaker-name">${speakerNameMarkup}</p>
          ${speakerAffiliationMarkup}
        </div>
        ${speakerSpecialtiesMarkup}
      </div>
      <div class="seminar-divider" aria-hidden="true"></div>
      <div class="seminar-meta seminar-meta-secondary">
        ${metaRowsMarkup}
      </div>
    </div>
  `);
};

export const renderOverview = (templates) => {
  const section = findSection("why-attend");
  qs("[data-overview-intro]").textContent = section.intro;

  const sister = qs("[data-sister-seminar]");
  if (sister && section.sisterSeminar) {
    const info = section.sisterSeminar;
    sister.innerHTML = `
      <span class="sister-seminar-icon">${icon("presentation")}</span>
      <div class="sister-seminar-copy">
        <p class="sister-seminar-kicker">${escapeHtml(info.kicker)}</p>
        <p class="sister-seminar-text">${escapeHtml(info.text)}</p>
      </div>
      <a class="sister-seminar-link" href="${escapeHtml(info.link.href)}" target="_blank" rel="noopener">${escapeHtml(
        info.link.label
      )} &rarr;</a>
    `;
  }

  const guideLink = qs("[data-overview-link]");
  if (guideLink && section.guideLink) {
    guideLink.innerHTML = `<a href="${escapeHtml(section.guideLink.href)}">${escapeHtml(
      section.guideLink.label
    )} &rarr;</a>`;
  }
  qs("[data-feature-grid]").innerHTML = section.cards
    .map((card) =>
      renderTemplate(templates.featureCard, {
        icon: icon(card.icon),
        title: escapeHtml(card.title),
        description: escapeHtml(card.description)
      })
    )
    .join("");
};

export const renderSchedule = async () => {
  const section = findSection("schedule");
  const speakerSection = findSection("speakers");
  const action = qs("[data-schedule-action]");
  action.href = section.action.href;
  action.textContent = section.action.label;

  const speakerAction = qs("[data-speakers-action]");
  speakerAction.href = speakerSection.action.href;
  speakerAction.textContent = speakerSection.action.label;

  const list = qs("[data-schedule-list]");
  if (!list) {
    return;
  }

  const resolved = await loadResolvedSchedule();
  const kicker = qs("[data-schedule-season]");

  if (resolved?.status === "break") {
    if (kicker) {
      kicker.textContent = "Seminar on Break";
    }
    list.innerHTML = renderBreakMarkup(resolved.breakKind);
    return;
  }

  const talks = resolved?.seasonTalks?.length ? resolved.seasonTalks : pageScheduleItemsAsTalks();

  if (kicker) {
    kicker.textContent = resolved ? `${formatSeasonHeading(resolved.seasonKey)} Schedule` : "This Semester";
  }

  list.innerHTML = renderScheduleTable(talks);
};

const collectSemesterSpeakers = (talks = []) => {
  const byKey = new Map();

  talks.forEach((talk) => {
    if (isBreakEntry(talk)) {
      return;
    }
    getTalkSpeakers(talk).forEach((speaker) => {
      if (!speaker.name || isTbaSpeaker(speaker.name)) {
        return;
      }
      const key = speakerKey(speaker.name);
      if (!byKey.has(key)) {
        byKey.set(key, { ...speaker, talks: [] });
      }
      const entry = byKey.get(key);
      if (talk.talkTitle && !entry.talks.some((item) => item.title === talk.talkTitle)) {
        entry.talks.push({ title: talk.talkTitle, date: talk.talkDate });
      }
    });
  });

  // Talks arrive date-sorted, so Map insertion order lists speakers in the
  // order they present during the semester.
  return Array.from(byKey.values());
};

export const renderSpeakers = async (templates) => {
  const section = findSection("speakers");
  const grid = qs("[data-speaker-grid]");
  if (!grid) {
    return;
  }

  const action = qs("[data-speakers-action]");
  if (action) {
    action.href = section.action.href;
    action.textContent = section.action.label;
  }

  const resolved = await loadResolvedSchedule();
  const seasonLabel = resolved ? formatSeasonHeading(resolved.rosterSeasonKey) : "";
  const kicker = qs("[data-speakers-season]");
  if (kicker) {
    kicker.textContent = seasonLabel ? `${seasonLabel} Speakers` : "Seminar Speakers";
  }

  const speakers = collectSemesterSpeakers(resolved?.rosterTalks || []);

  if (!speakers.length) {
    grid.innerHTML = `<p class="speaker-roster-empty">Speakers for ${escapeHtml(
      seasonLabel || "this semester"
    )} are still being confirmed. Check back soon &mdash; the schedule above lists the topics already locked in.</p>`;
    return;
  }

  grid.innerHTML = speakers
    .map((speaker) =>
      renderTemplate(templates.speakerDirectoryCard, {
        initials: escapeHtml(speakerInitials(speaker.name)),
        name: escapeHtml(speaker.name),
        // Speakers without a profile row show just their name and topic rather
        // than a card full of "TBA" placeholders.
        roleLine: speaker.hasProfile ? escapeHtml(formatDirectoryRole(speaker)) : "",
        affiliationLine: speaker.hasProfile ? escapeHtml(formatDirectoryAffiliation(speaker)) : "",
        talkCountLabel: "",
        topicsMarkup: speaker.talks
          .map(
            (talk) =>
              `<li><span class="speaker-directory-topic-title">${escapeHtml(
                talk.title
              )}</span><span class="speaker-directory-topic-date">${escapeHtml(
                readableDate(talk.date)
              )}</span></li>`
          )
          .join(""),
        website: escapeHtml(speakerProfileHref(speaker, "/speakers/")),
        imageMarkup: speaker.image
          ? `<img data-fade src="${escapeHtml(speaker.image)}" alt="${escapeHtml(speaker.name)}" onerror="this.remove(); this.nextElementSibling.hidden = false;">`
          : "",
        initialsHidden: speaker.image ? "hidden" : ""
      })
    )
    .join("");
};

export const renderCommunity = (templates) => {
  const section = findSection("community");
  qs("[data-community-grid]").innerHTML = section.items
    .map((item) =>
      renderTemplate(templates.communityCard, {
        icon: icon(item.icon),
        title: escapeHtml(item.title),
        description: escapeHtml(item.description),
        actionHref: escapeHtml(item.action.href),
        actionLabel: escapeHtml(item.action.label)
      })
    )
    .join("");
};

const renderFooterLink = (item) => `
  <li>
    <a class="footer-link" href="${escapeHtml(item.href)}">
      <span class="footer-link-icon">${icon(item.icon)}</span>
      <span>${escapeHtml(item.label)}</span>
    </a>
  </li>
`;

export const renderFooter = () => {
  const footer = pageData.footer;
  qs("[data-footer]").innerHTML = `
    <div class="footer-block" data-reveal="up">
      <div class="footer-brand">
        <!-- Doubles as the seasonal-theme easter egg: the panel is mounted by
             the seasonal layer, which listens for [data-season-trigger]. -->
        <button
          class="footer-logo-button"
          type="button"
          data-season-trigger
          aria-haspopup="dialog"
          aria-expanded="false"
          aria-controls="season-switcher-panel"
        >
          <img class="footer-logo" src="${escapeHtml(footerLogoUrl)}" alt="Florida State University Scientific Computing">
        </button>
      </div>
      <p class="footer-description">${escapeHtml(footer.description)}</p>
    </div>
    <div class="footer-block" data-reveal="up">
      <h3>${escapeHtml(footer.contact.title)}</h3>
      <ul class="footer-list">
        ${footer.contact.items.map(renderFooterLink).join("")}
      </ul>
    </div>
    <div class="footer-block" data-reveal="up">
      <h3>${escapeHtml(footer.social.title)}</h3>
      <ul class="footer-list">
        ${footer.social.items.map(renderFooterLink).join("")}
      </ul>
    </div>
  `;
};

export const renderFullSchedule = async () => {
  const list = qs("[data-full-schedule-list]");
  if (!list) {
    return;
  }

  const kicker = qs("[data-schedule-kicker]");
  const resolved = await loadResolvedSchedule();
  const scheduleDetails = new Map(pageScheduleItems().map((item) => [item.title, item]));

  if (resolved?.status === "break") {
    if (kicker) {
      kicker.textContent = "Seminar on Break";
    }
    list.innerHTML = renderBreakMarkup(resolved.breakKind);
    return;
  }

  const talks = resolved?.talks?.length ? resolved.talks : pageScheduleItemsAsTalks();

  if (kicker) {
    kicker.textContent = resolved ? formatSeasonHeading(resolved.seasonKey) : "";
  }

  list.innerHTML = talks
    .map((speaker) => renderTalkCard(speaker, scheduleDetails.get(speaker.talkTitle) || {}))
    .join("");
};

export const renderSpeakerDirectory = async (templates) => {
  const grid = qs("[data-speaker-directory]");
  if (!grid) {
    return;
  }

  const section = findSection("speakers");
  const uniqueSpeakers = await loadUniqueSpeakersFromCsv({ featuredOnly: false });
  const speakers = uniqueSpeakers.length ? uniqueSpeakers : section.speakers.map((speaker) => ({
    ...speaker,
    talkCount: 1,
    topics: (speaker.topic || "")
      .split(";")
      .map((item) => item.trim())
      .filter(Boolean)
      .slice(0, 3)
  }));

  grid.innerHTML = speakers
    .map((speaker) =>
      renderTemplate(templates.speakerDirectoryCard, {
        initials: escapeHtml(speakerInitials(speaker.name)),
        name: escapeHtml(speaker.name),
        roleLine: escapeHtml(formatDirectoryRole(speaker)),
        affiliationLine: escapeHtml(formatDirectoryAffiliation(speaker)),
        talkCountLabel: escapeHtml(formatTalkCountLabel(speaker.talkCount || 0)),
        topicsMarkup: (speaker.topics?.length ? speaker.topics : [speaker.topic || "Specialty TBA"])
          .slice(0, 3)
          .map((topic) => `<li>${escapeHtml(titleCase(topic))}</li>`)
          .join(""),
        website: escapeHtml(speakerWebsite(speaker, "#speaker-directory")),
        imageMarkup: speaker.image
          ? `<img data-fade src="${escapeHtml(speaker.image)}" alt="${escapeHtml(speaker.name)}" onerror="this.remove(); this.nextElementSibling.hidden = false;">`
          : "",
        initialsHidden: speaker.image ? "hidden" : ""
      })
    )
    .join("");
};
