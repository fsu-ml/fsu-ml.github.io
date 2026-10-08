import { resolveDisplaySemester } from "../data/semester-schedule.js";
import { loadSpeakersFromCsv } from "../data/speakers.js";
import {
  isBreakEntry,
  isCalledOff,
  statusLabel,
  statusReason,
  statusSchema
} from "../data/talk-status.js";

/**
 * schema.org Event entries for the semester on screen, each carrying an
 * `eventStatus` (EventScheduled / EventCancelled / EventPostponed).
 *
 * This is the machine-readable mirror of the schedule for search engines and
 * any agent that reads the rendered page. Otto does not need it - it reads
 * speakers.csv directly - but both are built from the same `status` cell.
 *
 * Location and organizer are borrowed from the page's static EventSeries
 * block, so the standing room is written down in exactly one place.
 */

const SCRIPT_MARKER = "data-talk-events";

const seriesFromPage = () => {
  const block = document.querySelector('script[type="application/ld+json"]:not([data-talk-events])');
  try {
    return block ? JSON.parse(block.textContent) : {};
  } catch {
    return {};
  }
};

const talkEvent = (talk, series) => {
  const event = {
    "@type": "Event",
    name: talk.talkTitle,
    startDate: talk.talkDate,
    eventStatus: statusSchema(talk),
    eventAttendanceMode: "https://schema.org/MixedEventAttendanceMode",
    description: talk.description || undefined,
    performer: (talk.speakers || [])
      .filter((speaker) => speaker.name && !/\bTBA\b/i.test(speaker.name))
      .map((speaker) => ({ "@type": "Person", name: speaker.name })),
    location: series.location,
    organizer: series.organizer,
    superEvent: series.name ? { "@type": "EventSeries", name: series.name, url: series.url } : undefined
  };
  if (isCalledOff(talk)) {
    // schema.org has no field for why an event was called off, so the reason
    // rides in the one free-text slot meant for telling events apart.
    event.disambiguatingDescription = [statusLabel(talk), statusReason(talk)].filter(Boolean).join(": ");
  }
  return event;
};

export const renderTalkStructuredData = async () => {
  const talks = (await loadSpeakersFromCsv({ featuredOnly: false })).filter(
    (talk) => talk.talkTitle && talk.talkDate && !isBreakEntry(talk)
  );
  const resolved = resolveDisplaySemester(talks);
  document.head.querySelector(`script[${SCRIPT_MARKER}]`)?.remove();
  if (!resolved.talks.length) {
    return;
  }

  const series = seriesFromPage();
  const script = document.createElement("script");
  script.type = "application/ld+json";
  script.setAttribute(SCRIPT_MARKER, "");
  script.textContent = JSON.stringify({
    "@context": "https://schema.org",
    "@graph": resolved.talks.map((talk) => talkEvent(talk, series))
  });
  document.head.append(script);
};
