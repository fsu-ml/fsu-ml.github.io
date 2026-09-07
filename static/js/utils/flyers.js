/**
 * Flyer list parsing.
 *
 * The `flyers` column of speakers.csv is a semicolon-separated list of
 * filenames from data/flyers/, each optionally carrying a display label after
 * a pipe:
 *
 *   2026-09-18_jackie_ye.webp|Wide; 2026-09-18_jackie_ye_vertical.webp|Vertical
 *
 * Same shape as the `materials` column, so a maintainer who has written one has
 * already learned the other. The label is only for humans - it names the
 * variant when a talk has several ("Wide", "Vertical", "Printed"). Without one
 * the flyer is numbered by position, which is what a single-flyer talk wants.
 */

const flyerBaseUrl = new URL("../../../data/flyers/", import.meta.url);

/** `filename.webp|Label` -> `{ file, label }`, label optional. */
const splitEntry = (entry = "") => {
  const [file, ...labelParts] = entry.split("|");
  return { file: file.trim(), label: labelParts.join("|").trim() };
};

/**
 * `[{ href, label, alt }]` for one talk, in CSV order. The first flyer is the
 * one a thumbnail should show.
 *
 * `title` is the talk title, used to build alt text - a flyer is a poster for a
 * specific talk, so "Flyer for <title>" says more to a screen reader than any
 * generic label would.
 */
export const parseFlyerList = (flyers = "", title = "") =>
  String(flyers)
    .split(";")
    .map((entry) => entry.trim())
    .filter(Boolean)
    .map(splitEntry)
    .filter((entry) => entry.file)
    .map(({ file, label }, index, all) => {
      const position = all.length > 1 ? ` ${index + 1} of ${all.length}` : "";
      const named = label ? ` (${label})` : "";
      return {
        href: new URL(file, flyerBaseUrl).href,
        label: label || (all.length > 1 ? `Flyer ${index + 1}` : "Flyer"),
        alt: title ? `Flyer${position} for ${title}${named}` : `Flyer${position}${named}`
      };
    });

/**
 * The attribute payload a lightbox trigger carries. JSON rather than a
 * delimited string so a label containing the delimiter cannot break the parse;
 * callers must still run it through escapeHtml before writing it into markup.
 */
export const flyerTriggerAttrs = (flyerList = [], title = "") =>
  JSON.stringify({ title, items: flyerList });
