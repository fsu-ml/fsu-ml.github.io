/**
 * Card hover: the vine.
 *
 * On hover a vine draws itself along the card's own edges, just inside the
 * border, starting at the top-left corner and running clockwise. Leaves bud
 * as the tip passes them, lying along the edge and leaning into the card, and
 * a small pumpkin sets in the top-right corner when the vine comes round.
 *
 * Everything stays inside the card: the artwork is a `.season-card-art`,
 * clipped to the card's rounded corners, so nothing hangs off its outline.
 * It is decoration only: it never changes a text colour, a background a text
 * sits on, or a focus ring, and it carries `pointer-events: none` so it can
 * never take a click meant for a link inside the card.
 *
 * Presenter cards follow the site's own rule: they never lift or scale. The
 * vine runs round their edge like any other card's, well clear of the
 * portrait in the middle.
 */

import { decorate, seeded } from "../engine.js";
import { leafSvg, pickOf } from "./art.js";

const CARDS = ".speaker-directory-card, .feature-card, .talk-card, .community-card";

const VINE_LEAVES = 12;

/* How far inside the card's edge the vine runs, in px. Close enough to read
   as the card's own border dressed up, far enough that the stroke is not cut
   by the clip. */
const INSET = 4;

const VINE_GREENS = ["#6f7d42", "#8a9a52", "#a0782e", "#c8641f", "#b4562c"];

/* A point at fraction `f` of the way round a w x h rectangle, clockwise from
   the top-left corner, with the angle pointing into the rectangle. */
const onPerimeter = (f, w, h) => {
  const P = 2 * (w + h);
  let d = f * P;
  if (d < w) {
    return [d, 0, 90];
  }
  d -= w;
  if (d < h) {
    return [w, d, 180];
  }
  d -= h;
  if (d < w) {
    return [w - d, h, -90];
  }
  d -= w;
  return [0, h - d, 0];
};

const vineArt = (seed) => {
  const rand = seeded(seed);
  let leaves = "";
  for (let i = 0; i < VINE_LEAVES; i += 1) {
    leaves += `<span class="tg-vine-leaf">${leafSvg(
      pickOf(rand, ["birch", "elm", "maple"]),
      pickOf(rand, VINE_GREENS),
      (12 + rand() * 6).toFixed(0)
    )}</span>`;
  }
  return `<svg class="tg-vine-line" aria-hidden="true" focusable="false"><path pathLength="1"></path></svg>${leaves}
    <span class="tg-vine-fruit"><svg viewBox="0 0 32 28" width="20" height="18" aria-hidden="true" focusable="false">
      <path d="M16 6c-2-4 0-6 2-6" stroke="#5a3a1a" stroke-width="2" fill="none" stroke-linecap="round"/>
      <ellipse cx="9" cy="16" rx="8" ry="10" fill="#c8641f"/><ellipse cx="23" cy="16" rx="8" ry="10" fill="#c8641f"/>
      <ellipse cx="16" cy="16" rx="8" ry="11" fill="#e07a2f"/><path d="M16 6v20" stroke="rgba(90,30,10,.35)"/></svg></span>`;
};

/* The vine is drawn to the card's real size, so a corner stays a corner
   instead of stretching with `preserveAspectRatio="none"`. Measured once per
   size, on mount and again on hover if the card has changed size since. */
const fit = (card) => {
  const art = card.querySelector(":scope > .tg-vine");
  if (!art) {
    return;
  }
  const W = art.clientWidth;
  const H = art.clientHeight;
  if (!W || !H || art.dataset.size === `${W}x${H}`) {
    return;
  }
  art.dataset.size = `${W}x${H}`;
  const w = W - INSET * 2;
  const h = H - INSET * 2;
  const rand = seeded(W * 31 + H);
  const svg = art.querySelector("svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  /* A gentle wander either side of the inset line, never outward past the
     card's edge. */
  let d = "";
  const steps = Math.round((2 * (w + h)) / 12);
  for (let i = 0; i <= steps; i += 1) {
    const [x, y, a] = onPerimeter(Math.min(i / steps, 0.9999), w, h);
    const wob = (Math.sin(i * 0.55) + 1) * 0.9;
    const rad = (a * Math.PI) / 180;
    d += `${i ? "L" : "M"}${(INSET + x + Math.cos(rad) * wob).toFixed(1)} ${(INSET + y + Math.sin(rad) * wob).toFixed(1)}`;
  }
  svg.querySelector("path").setAttribute("d", d);

  /* Leaves lie along the edge, leaning into the card by alternating angles,
     the way leaves sit on a vine trained along a fence. */
  art.querySelectorAll(".tg-vine-leaf").forEach((leaf, i) => {
    const f = (i + 0.35 + rand() * 0.3) / VINE_LEAVES;
    const [x, y, inward] = onPerimeter(f, w, h);
    const lean = i % 2 ? 58 : -58;
    leaf.style.left = `${INSET + x}px`;
    leaf.style.top = `${INSET + y}px`;
    /* A leaf's tip points up at rotate(0); +90 turns "up" into the inward
       direction. */
    leaf.style.setProperty("--rot", `${inward + 90 + lean}deg`);
    leaf.style.setProperty("--wait", `${Math.round(f * 900)}ms`);
  });
};

export const mountCards = ({ disposer }) => {
  const arts = decorate(disposer, CARDS, "season-card-art tg-vine", vineArt(131));
  arts.forEach((art) => fit(art.parentElement));
  disposer.listen(document, "pointerover", (event) => {
    const card = event.target.closest?.(CARDS);
    if (card && !(event.relatedTarget && card.contains(event.relatedTarget))) {
      fit(card);
    }
  });
};
