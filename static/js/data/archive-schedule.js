import { formatSeasonHeading, isUpcoming, parseSeason } from "./semester-schedule.js";

const semesterSortKey = (seasonKey = "") => {
  const parsed = parseSeason(seasonKey);
  if (!parsed) {
    return -1;
  }
  return parsed.year * 10 + (parsed.term === "Spring" ? 1 : 2);
};

export const seasonAnchorId = (seasonKey = "") =>
  `season-${seasonKey.trim().toLowerCase()}`;

export const groupPastTalksBySeason = (talks = [], today = new Date()) => {
  const bySeason = new Map();

  talks.forEach((talk) => {
    if (!talk.talkDate || !talk.season || isUpcoming(talk.talkDate, today)) {
      return;
    }
    if (!parseSeason(talk.season)) {
      return;
    }
    if (!bySeason.has(talk.season)) {
      bySeason.set(talk.season, []);
    }
    bySeason.get(talk.season).push(talk);
  });

  return Array.from(bySeason.entries())
    .map(([seasonKey, seasonTalks]) => ({
      seasonKey,
      heading: formatSeasonHeading(seasonKey),
      anchorId: seasonAnchorId(seasonKey),
      talks: [...seasonTalks].sort((left, right) => right.talkDate.localeCompare(left.talkDate))
    }))
    .sort((left, right) => semesterSortKey(right.seasonKey) - semesterSortKey(left.seasonKey));
};
