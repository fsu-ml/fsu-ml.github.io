/**
 * Thanksgiving: shared artwork.
 *
 * Leaf silhouettes, palette and the small SVG builders the theme shares.
 * Leaves are drawn in a 100x100 box with the stem base at (50, 98), so one
 * path serves the canvas field and the inline SVGs alike.
 */

/* Amber, burnt orange, rust, ochre, brown, and one garnet to keep the brand
   in the mix. */
export const LEAF_COLORS = ["#d98e3f", "#c8641f", "#b4562c", "#e0a83c", "#9a3e24", "#8e1f31", "#c9952f"];

export const LEAF = {
  maple:
    "M50 98V80L31 88l3-13-21-4 10-10L7 47l16-2-6-16 16 6 3-17 8 13 6-25 6 25 8-13 3 17 16-6-6 16 16 2-15 14 10 10-21 4 3 13-19-8z",
  oak: "M50 98c-1-5-1-9 0-13-10 2-16-3-12-9-10 2-14-6-6-10-10-2-10-12 0-12-8-6-4-16 5-12-5-10 3-18 9-12-2-10 2-20 4-24 2 4 6 14 4 24 6-6 14 2 9 12 9-4 13 6 5 12 10 0 10 10 0 12 8 4 4 12-6 10 4 6-2 11-12 9z",
  birch: "M50 98V86C30 78 22 50 50 6c28 44 20 72 0 80z",
  ginkgo: "M50 98V60C38 58 14 48 10 22c14 6 26 4 40 0 14 4 26 6 40 0-4 26-28 36-40 38z",
  elm: "M50 98V84C26 74 24 40 50 8c26 32 24 66 0 76z"
};

export const LEAF_KINDS = Object.keys(LEAF);

/* Midrib plus a few side veins. Drawn over the fill at low alpha, it is what
   turns a flat silhouette into a leaf at anything above 20px. */
export const VEINS = "M50 96V14M50 70L30 56M50 70L70 56M50 50L34 36M50 50L66 36";

export const pickOf = (rand, list) => list[Math.min(list.length - 1, Math.floor(rand() * list.length))];

export const leafSvg = (kind, color, size, { veins = true, extra = "" } = {}) =>
  `<svg viewBox="0 0 100 100" width="${size}" height="${size}" aria-hidden="true" focusable="false" ${extra}>
     <path d="${LEAF[kind] || LEAF.elm}" fill="${color}"></path>
     ${veins ? `<path d="${VEINS}" fill="none" stroke="rgba(42,17,10,.32)" stroke-width="2.6" stroke-linecap="round"></path>` : ""}
   </svg>`;

/* Leaf shapes as Path2D, built once per page. */
let paths2d = null;
export const leafPaths = () => {
  if (!paths2d) {
    paths2d = { veins: new Path2D(VEINS) };
    for (const kind of LEAF_KINDS) {
      paths2d[kind] = new Path2D(LEAF[kind]);
    }
  }
  return paths2d;
};

/**
 * Draws one leaf centred at (x, y), `size` px tall. `flip` scales the leaf's
 * height by the cosine of its tumble, turning it edge-on and back: a tumble
 * in three dimensions from a 2D context.
 */
export const drawLeaf = (ctx, kind, x, y, size, rot, color, alpha = 1, flip = 1) => {
  const p = leafPaths();
  const s = size / 100;
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rot);
  ctx.scale(s, s * flip);
  ctx.translate(-50, -52);
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.fill(p[kind] || p.elm);
  ctx.strokeStyle = "rgba(42,17,10,.34)";
  ctx.lineWidth = 3;
  ctx.stroke(p.veins);
  ctx.restore();
};

/* ---------------------------------------------------------------------------
   Small helpers
   -------------------------------------------------------------------------- */

export const f1 = (n) => n.toFixed(1);
