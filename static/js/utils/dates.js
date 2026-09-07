/**
 * Talk-date formatting.
 *
 * Talk dates are plain ISO `YYYY-MM-DD` strings with no time component, so they
 * are split by hand rather than passed to `new Date()`: parsing a bare ISO date
 * treats it as UTC midnight, which renders as the previous day for anyone west
 * of Greenwich - the seminar's own timezone included.
 */

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

export const parseIsoDate = (value = "") => {
  const [year, month, day] = value.split("-").map(Number);
  if (!year || !month || !day) {
    return null;
  }
  return { year, month, day };
};

/** The two-line block used by the date badges: `{ month: "SEP", day: "18" }`. */
export const dateBadge = (value = "") => {
  const parsed = parseIsoDate(value);
  if (!parsed) {
    return { month: "TBA", day: "" };
  }
  return {
    month: monthLabels[parsed.month - 1],
    day: String(parsed.day).padStart(2, "0")
  };
};

/** `September 18, 2026` - the spoken form, used for screen readers and captions. */
export const readableDate = (value = "") => {
  const parsed = parseIsoDate(value);
  if (!parsed) {
    return "Date TBA";
  }
  return `${monthNames[parsed.month - 1]} ${parsed.day}, ${parsed.year}`;
};
