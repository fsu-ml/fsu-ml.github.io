/**
 * Lunar New Year dates, 2027–2036.
 *
 * Its own module because two callers need it and neither should own it:
 * `season.js` reads it to decide whether the season is on at all, and
 * `lunar.js` reads it to know which animal to draw. Importing one from the
 * other would make a cycle out of what is really just a table.
 *
 * The date moves against the Gregorian calendar, so unlike every other season
 * this one cannot be derived from a month. There is no arithmetic that would
 * be shorter than the list, so it is simply listed.
 */

/** ISO date of each new year, to the animal that year belongs to. */
export const LUNAR_NEW_YEAR = {
  "2027-02-06": "goat",
  "2028-01-26": "monkey",
  "2029-02-13": "rooster",
  "2030-02-03": "dog",
  "2031-01-23": "pig",
  "2032-02-11": "rat",
  "2033-01-31": "ox",
  "2034-02-19": "tiger",
  "2035-02-08": "rabbit",
  "2036-01-28": "dragon"
};

/* Eve, the day, and the day after — three days, which is how the holiday is
   actually kept, and long enough that a midweek new year is still on the site
   for someone who visits either side of it.

   Keyed by `toDateString()` so the lookup is a local-midnight day identity and
   never a timezone comparison: a visitor in Taipei and one in Tallahassee each
   see the season on their own three days. */
const WINDOW = new Map();
for (const [iso, animal] of Object.entries(LUNAR_NEW_YEAR)) {
  const [y, m, d] = iso.split("-").map(Number);
  for (let offset = -1; offset <= 1; offset += 1) {
    /* Constructed from parts rather than parsed: `new Date("2032-02-11")` is
       UTC midnight, which is the previous day for everyone west of Greenwich
       and would shift the whole window by a day for most of the US. */
    WINDOW.set(new Date(y, m - 1, d + offset).toDateString(), animal);
  }
}

/**
 * The animal for this date if it falls in a new year window, else null.
 *
 * Past 2036 the table runs out and this returns null forever, which is the
 * correct failure: no season is better than a season on the wrong day.
 */
export const lunarAnimalForDate = (date = new Date()) =>
  WINDOW.get(date.toDateString()) ?? null;
