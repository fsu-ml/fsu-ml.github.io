import { parseCsv } from "../utils/csv.js";

const speakerImagesUrl = new URL("../../../data/speaker-images/", import.meta.url);
const eventImagesUrl = new URL("../../../data/event-images/", import.meta.url);
const speakersCsvUrl = new URL("../../../data/speakers.csv", import.meta.url);
const speakerProfilesCsvUrl = new URL("../../../data/speaker-profiles.csv", import.meta.url);

// Both CSVs are fetched by more than one renderer per page (renderSchedule and
// renderSpeakers on the homepage, for instance), which previously meant the same
// two files were pulled over the network four times each on a single load. The
// response text is cached per URL for the lifetime of the page; the parse still
// runs per caller, so every caller keeps its own independent parsed result and
// no shared mutable state is introduced. A failed fetch is not cached, so a
// transient error can still be retried by the next caller.
const csvTextCache = new Map();

const fetchCsvText = (url) => {
  const key = url.href;
  if (!csvTextCache.has(key)) {
    csvTextCache.set(
      key,
      fetch(url).then((response) => {
        if (!response.ok) {
          throw new Error(`Unable to load ${key}`);
        }
        return response.text();
      })
    );
    csvTextCache.get(key).catch(() => csvTextCache.delete(key));
  }
  return csvTextCache.get(key);
};

export const speakerKey = (name = "") => name.replace(/^Dr\.\s+/i, "").trim().toLowerCase();

export const splitSpeakerNames = (name = "") =>
  String(name)
    .split(";")
    .map((part) => part.trim())
    .filter(Boolean);

export const formatSpeakerNameList = (names = []) => {
  if (!names.length) {
    return "";
  }
  if (names.length === 1) {
    return names[0];
  }
  if (names.length === 2) {
    return `${names[0]} and ${names[1]}`;
  }
  return `${names.slice(0, -1).join(", ")}, and ${names[names.length - 1]}`;
};

export const speakerProfileHref = (speaker = {}, fallback = "/speakers/") => {
  const website = speaker.website || speaker.profileUrl || fallback;
  return website.startsWith("#") ? fallback : website;
};

export const getTalkSpeakers = (talk = {}) => {
  if (talk.speakers?.length) {
    return talk.speakers;
  }
  if (talk.name) {
    return [talk];
  }
  return [];
};

const profileAliases = new Map([
  [speakerKey("Tom Juzek"), speakerKey("Tommie Juzek")],
  [speakerKey("Gorden Erlebacher"), speakerKey("Gordon Erlebacher")],
  [speakerKey("Olmo Zavala"), speakerKey("Olmo Zavala Romero")]
]);

const resolveProfileKey = (name = "") => {
  const key = speakerKey(name);
  return profileAliases.get(key) || key;
};

const emptyProfile = (name = "") => ({
  name,
  title: "",
  department: "",
  affiliation: "",
  specialties: "",
  email: "",
  website: "",
  profile_url: "",
  image: ""
});

export const loadSpeakerProfilesFromCsv = async () => {
  try {
    const profilesCsvText = await fetchCsvText(speakerProfilesCsvUrl);

    const profilesByKey = new Map();

    parseCsv(profilesCsvText).forEach((profile) => {
      const key = speakerKey(profile.name);
      if (!key) {
        return;
      }
      profilesByKey.set(key, profile);
    });

    return profilesByKey;
  } catch (error) {
    console.warn(error);
    return new Map();
  }
};

const profileForName = (profilesByKey, name = "") => {
  const key = resolveProfileKey(name);
  return profilesByKey.get(key) || emptyProfile(name);
};

// A placeholder like "Speaker TBA" must never render as a real profile, even
// though the profiles CSV carries a generic row for it.
export const isTbaSpeaker = (name = "") => /\bTBA\b/i.test(name);

const hasProfile = (profilesByKey, name = "") =>
  !isTbaSpeaker(name) && profilesByKey.has(resolveProfileKey(name));

const mapSpeakerRecord = (profile, schedule = {}, { hasProfile: profileListed = false } = {}) => ({
  name: profile.name || schedule.name,
  title: profile.title,
  department: profile.department,
  affiliation: profile.affiliation,
  topic: profile.specialties || schedule.talk_title,
  talkTitle: schedule.talk_title,
  talkDate: schedule.talk_date,
  email: profile.email,
  website: profile.website || profile.profile_url || "#speakers",
  profileUrl: profile.profile_url,
  featured: schedule.featured !== "false",
  season: schedule.season || "",
  description: schedule.description || "",
  materials: schedule.materials || "",
  locationNote: schedule.location_note || "",
  startTime: schedule.start_time || "",
  location: schedule.location || "",
  registrationUrl: schedule.registration_url || "",
  eventImage: schedule.event_image ? new URL(schedule.event_image, eventImagesUrl).href : "",
  image: profile.image ? new URL(profile.image, speakerImagesUrl).href : "",
  hasProfile: profileListed
});

const compareSpeakersForDirectory = (left, right) => {
  const talkDiff = (right.talkCount || 0) - (left.talkCount || 0);
  if (talkDiff !== 0) {
    return talkDiff;
  }
  return (left.name || "").localeCompare(right.name || "", undefined, { sensitivity: "base" });
};

const buildTalkFromRow = (row, profilesByKey) => {
  const names = splitSpeakerNames(row.name);
  if (!names.length) {
    return null;
  }

  const speakers = names.map((name) =>
    mapSpeakerRecord(profileForName(profilesByKey, name), { ...row, name }, {
      hasProfile: hasProfile(profilesByKey, name)
    })
  );

  const primary = speakers[0];

  return {
    ...primary,
    name: formatSpeakerNameList(speakers.map((speaker) => speaker.name)),
    speakers,
    hasProfile: speakers.some((speaker) => speaker.hasProfile)
  };
};

export const loadSpeakersFromCsv = async ({ featuredOnly = true } = {}) => {
  try {
    const [profilesByKey, scheduleCsvText] = await Promise.all([
      loadSpeakerProfilesFromCsv(),
      fetchCsvText(speakersCsvUrl)
    ]);

    return parseCsv(scheduleCsvText)
      .filter((row) => !featuredOnly || row.featured !== "false")
      .map((row) => buildTalkFromRow(row, profilesByKey))
      .filter(Boolean)
      .filter((talk) => talk.name);
  } catch (error) {
    console.warn(error);
    return [];
  }
};

export const loadUniqueSpeakersFromCsv = async ({ featuredOnly = true } = {}) => {
  const profilesByKey = await loadSpeakerProfilesFromCsv();
  let scheduleRows = [];

  try {
    scheduleRows = parseCsv(await fetchCsvText(speakersCsvUrl));
  } catch (error) {
    console.warn(error);
  }

  const speakersByKey = new Map();

  const upsertSpeaker = (profile, schedule = {}) => {
    const canonicalName = profile.name || schedule.name;
    const key = resolveProfileKey(canonicalName);
    if (!key) {
      return;
    }

    const merged = mapSpeakerRecord(profile, schedule, {
      hasProfile: hasProfile(profilesByKey, canonicalName)
    });
    const hasTalk = Boolean(merged.talkTitle || merged.talkDate);
    const specialties = (merged.topic || "")
      .split(";")
      .map((item) => item.trim())
      .filter(Boolean);

    if (!speakersByKey.has(key)) {
      speakersByKey.set(key, {
        ...merged,
        talkCount: hasTalk ? 1 : 0,
        topics: [...specialties]
      });
      return;
    }

    const speaker = speakersByKey.get(key);
    speaker.talkCount += hasTalk ? 1 : 0;

    specialties.forEach((topic) => {
      if (!speaker.topics.includes(topic)) {
        speaker.topics.push(topic);
      }
    });

    if (merged.image && !speaker.image) {
      speaker.image = merged.image;
    }
    if (merged.website && merged.website !== "#speakers") {
      speaker.website = merged.website;
    }
    if (merged.featured) {
      speaker.featured = true;
    }
  };

  profilesByKey.forEach((profile) => {
    upsertSpeaker(profile);
  });

  scheduleRows.forEach((row) => {
    splitSpeakerNames(row.name).forEach((name) => {
      if (!hasProfile(profilesByKey, name)) {
        return;
      }
      upsertSpeaker(profileForName(profilesByKey, name), { ...row, name });
    });
  });

  return Array.from(speakersByKey.values())
    .filter((speaker) => speakerKey(speaker.name) !== speakerKey("Speaker TBA"))
    .filter((speaker) => !featuredOnly || speaker.featured)
    .map((speaker) => ({
      ...speaker,
      topics: speaker.topics.slice(0, 3)
    }))
    .sort(compareSpeakersForDirectory);
};
