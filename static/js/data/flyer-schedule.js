import { formatSeasonHeading, parseSeason } from "./semester-schedule.js";
import { parseFlyerList } from "../utils/flyers.js";

const semesterSortKey = (seasonKey = "") => {
  const parsed = parseSeason(seasonKey);
  if (!parsed) {
    return -1;
  }
  return parsed.year * 10 + (parsed.term === "Spring" ? 1 : 2);
};

export const flyerSeasonAnchorId = (seasonKey = "") =>
  `flyers-${seasonKey.trim().toLowerCase()}`;

/**
 * Every talk carrying at least one flyer, grouped by semester.
 *
 * Unlike the archive this deliberately includes upcoming talks: a flyer for a
 * talk that has not happened yet is the one people most want to find and
 * share. Semesters run newest first, but talks run in calendar order *within* a
 * semester, so a term reads as the run of posters in the order they go up.
 */
export const groupFlyerTalksBySeason = (talks = []) => {
  const bySeason = new Map();

  talks.forEach((talk) => {
    const flyers = parseFlyerList(talk.flyers, talk.talkTitle);
    if (!flyers.length || !talk.season || !parseSeason(talk.season)) {
      return;
    }
    if (!bySeason.has(talk.season)) {
      bySeason.set(talk.season, []);
    }
    bySeason.get(talk.season).push({ ...talk, flyerList: flyers });
  });

  return Array.from(bySeason.entries())
    .map(([seasonKey, seasonTalks]) => ({
      seasonKey,
      heading: formatSeasonHeading(seasonKey),
      anchorId: flyerSeasonAnchorId(seasonKey),
      talks: [...seasonTalks].sort((left, right) =>
        (left.talkDate || "").localeCompare(right.talkDate || "")
      )
    }))
    .sort((left, right) => semesterSortKey(right.seasonKey) - semesterSortKey(left.seasonKey));
};
