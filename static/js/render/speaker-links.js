import { getTalkSpeakers, speakerProfileHref } from "../data/speakers.js";
import { escapeHtml } from "../utils/html.js";

const joinFragments = (fragments = []) => {
  if (fragments.length <= 1) {
    return fragments[0] || "";
  }
  if (fragments.length === 2) {
    return `${fragments[0]} and ${fragments[1]}`;
  }
  return `${fragments.slice(0, -1).join(", ")}, and ${fragments[fragments.length - 1]}`;
};

export const renderSpeakerNameLink = (speaker = {}, className = "speaker-name-link") => {
  const name = speaker.name || "Speaker TBA";
  return `<a class="${className}" href="${escapeHtml(speakerProfileHref(speaker))}">${escapeHtml(name)}</a>`;
};

export const renderSpeakerNameLinks = (talk = {}, className = "speaker-name-link") => {
  const speakers = getTalkSpeakers(talk);
  if (!speakers.length) {
    return escapeHtml("Speaker TBA");
  }
  return joinFragments(speakers.map((speaker) => renderSpeakerNameLink(speaker, className)));
};

export const renderArchiveSpeakerLine = (talk = {}) => {
  const speakers = getTalkSpeakers(talk);
  if (!speakers.length) {
    return `<a class="archive-talk-speaker" href="${escapeHtml(speakerProfileHref(talk))}">${escapeHtml(
      talk.name || "Speaker TBA"
    )}</a>`;
  }

  return joinFragments(
    speakers.map((speaker) => {
      const label =
        speakers.length === 1 && speaker.title && speaker.hasProfile
          ? `${speaker.name} · ${speaker.title}`
          : speaker.name || "Speaker TBA";
      return `<a class="archive-talk-speaker" href="${escapeHtml(speakerProfileHref(speaker))}">${escapeHtml(
        label
      )}</a>`;
    })
  );
};
