const parseIsoDateParts = (value = "") => {
  const [year, month, day] = value.split("-").map(Number);
  if (!year || !month || !day) {
    return null;
  }
  return { year, month, day };
};

export const parseSeason = (season = "") => {
  const match = /^(\d{4})-(Spring|Fall)$/i.exec(String(season).trim());
  if (!match) {
    return null;
  }
  return { year: Number(match[1]), term: match[2] };
};

export const seasonKey = ({ year, term }) => `${year}-${term}`;

export const getCalendarSemester = (date = new Date()) => {
  const month = date.getMonth() + 1;
  const day = date.getDate();
  const calendarYear = date.getFullYear();

  if (month === 12 && day >= 21) {
    return { year: calendarYear + 1, term: "Spring", seasonKey: `${calendarYear + 1}-Spring` };
  }
  if (month >= 1 && month <= 5) {
    return { year: calendarYear, term: "Spring", seasonKey: `${calendarYear}-Spring` };
  }
  if (month === 6 && day <= 20) {
    return { year: calendarYear, term: "Spring", seasonKey: `${calendarYear}-Spring` };
  }
  return { year: calendarYear, term: "Fall", seasonKey: `${calendarYear}-Fall` };
};

export const getNextSemester = ({ year, term }) => {
  if (term === "Spring") {
    return { year, term: "Fall", seasonKey: `${year}-Fall` };
  }
  return { year: year + 1, term: "Spring", seasonKey: `${year + 1}-Spring` };
};

// Seminars run 12:00-1:00 PM Eastern, so a talk stays "upcoming" through its
// own hour and only drops off the hero card and the schedule once it has
// actually ended - at 1 PM Eastern, not 1 PM wherever the visitor happens to be.
export const TALK_TIME_ZONE = "America/New_York";
export const TALK_END_HOUR = 13;

// How far the given zone sits from UTC at that instant, in milliseconds. Read
// back through Intl rather than hardcoded, so DST is handled for us.
const zoneOffsetMs = (date, timeZone) => {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone,
    hourCycle: "h23",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  })
    .formatToParts(date)
    .reduce((acc, part) => {
      if (part.type !== "literal") {
        acc[part.type] = Number(part.value);
      }
      return acc;
    }, {});
  const asUtc = Date.UTC(
    parts.year,
    parts.month - 1,
    parts.day,
    parts.hour,
    parts.minute,
    parts.second
  );
  return asUtc - Math.floor(date.getTime() / 1000) * 1000;
};

// The instant matching a wall-clock time in the given zone. The offset depends
// on the instant we are solving for, so the first guess is corrected once -
// enough for every case except a wall time inside a DST gap, which 1 PM is not.
const zonedTime = ({ year, month, day }, hour, timeZone) => {
  const guess = Date.UTC(year, month - 1, day, hour);
  const corrected = guess - zoneOffsetMs(new Date(guess), timeZone);
  const offset = zoneOffsetMs(new Date(corrected), timeZone);
  return guess - offset;
};

export const isUpcoming = (talkDate = "", today = new Date()) => {
  const parsed = parseIsoDateParts(talkDate);
  if (!parsed) {
    return false;
  }
  return zonedTime(parsed, TALK_END_HOUR, TALK_TIME_ZONE) > today.getTime();
};

export const filterUpcomingBySeason = (talks = [], targetSeasonKey = "", today = new Date()) => {
  if (!targetSeasonKey) {
    return [];
  }
  return talks
    .filter((talk) => talk.season === targetSeasonKey && talk.talkTitle && talk.talkDate)
    .filter((talk) => isUpcoming(talk.talkDate, today))
    .sort((left, right) => left.talkDate.localeCompare(right.talkDate));
};

export const getBreakMessage = (breakKind = "summer") => {
  if (breakKind === "winter") {
    return "Taking a Break for the Winter. Revisit the Website for Updates for Spring!";
  }
  return "Taking a Break for the Summer. Revisit the Website for Updates for Fall!";
};

export const formatSeasonHeading = (seasonKey = "") => {
  const parsed = parseSeason(seasonKey);
  if (!parsed) {
    return "";
  }
  return `${parsed.term} ${parsed.year}`;
};

export const resolveDisplaySemester = (talks = [], today = new Date()) => {
  const calendar = getCalendarSemester(today);
  const currentKey = calendar.seasonKey;
  const next = getNextSemester(calendar);

  const currentTalks = filterUpcomingBySeason(talks, currentKey, today);
  if (currentTalks.length) {
    return {
      seasonKey: currentKey,
      talks: currentTalks,
      status: "active",
      breakKind: null
    };
  }

  const nextTalks = filterUpcomingBySeason(talks, next.seasonKey, today);
  if (nextTalks.length) {
    return {
      seasonKey: next.seasonKey,
      talks: nextTalks,
      status: "active",
      breakKind: null
    };
  }

  const breakKind = calendar.term === "Spring" ? "summer" : "winter";
  return {
    seasonKey: currentKey,
    talks: [],
    status: "break",
    breakKind
  };
};
