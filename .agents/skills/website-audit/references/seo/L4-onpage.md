# Layer 4 — On-page optimisation

Covers: per-page elements, and the page types worth having at all.
Load: `seo.priority: basic|full`. For `seo.priority: none`, the MVP pass in `00-map.md` covers the non-negotiable subset.
Depends on: `L1-foundations.md` (indexable) and `L3-architecture.md` (linked). Optimising a page nothing links to is decoration.

Run these per template, not per page. Fixing a template fixes every page it renders.

---

## 4.1 Per-page elements

- [ ] **Title tag — unique, descriptive, primary topic near the front, ~50–60 characters, brand at the end** — Screaming Frog → Page Titles tab flags Missing / Duplicate / Over 60 Characters. Pass: zero duplicates, zero missing, length in range. Write it for a human deciding whether to click, not for a keyword count.
- [ ] **Meta description — unique, ~140–160 characters** — Screaming Frog → Meta Description tab. Pass: unique per page, in range. **Not a ranking factor, but it is the ad copy for your listing.** Google rewrites it often; that's fine, a written one still wins more often than an absent one.
- [ ] **Exactly one `<h1>` per page, matching the page's actual subject** — `curl -s URL | grep -c '<h1'`. Pass: `1`. Fail: `0` (common when a logo replaced the h1) or `3+` (CSS-driven heading choices).
- [ ] **Logical H2/H3 hierarchy mapping the content's real structure** — `curl -s URL | grep -oE '<h[1-6][^>]*>' | sed 's/[^h1-6]//g'` and read the sequence. Pass: no skipped levels (h2 → h4), and the outline reads as a genuine table of contents. Don't pick heading levels for their font size.
- [ ] **Content answers the query in the opening lines, before any preamble** — read the first two sentences aloud. Pass: they *answer*; fail: they *introduce* ("In today's fast-moving landscape..."). Both featured snippets and AI extraction favour front-loaded answers — see the passage test in `L9-ai-search.md`.
- [ ] **Sufficient depth for the intent** — compare against the pages currently ranking for the target query. Pass: every sub-question a searcher has is addressed. **Length matters only as a consequence of covering the topic** — word-count targets are a cargo cult.
- [ ] **Images have descriptive filenames and meaningful alt text** — `curl -s URL | grep -oE '<img [^>]*>'`. Pass: filenames like `roof-repair-flashing-detail.webp` not `IMG_4471.jpg`; alt text describing the image's content and function. Decorative images get `alt=""` — explicitly empty, not missing. See `../ada/`.
- [ ] **Videos have transcripts on-page** — Pass: full text transcript rendered in the HTML (not only in a player's caption track). *AI systems cannot watch video; they read transcripts.* This is also how video content enters text retrieval at all.
- [ ] **Outbound links to credible sources where claims need support** — Pass: factual claims cite something. Outbound links to good sources are a trust signal, not a leak.
- [ ] **Author byline linking to a real author bio page** — Pass: the byline is a link, the destination is a real bio with credentials, and the same person exists in the `Article` schema as a `Person` entity — see `L6-structured-data.md`.
- [ ] **Visible published date and, where relevant, a genuine last-updated date** — Pass: both visible and true. Fail: a `dateModified` bumped by a build script with no content change. Recency-weighted engines penalise undated content; they also learn to distrust fake freshness.
- [ ] **A clear next action** — Pass: you can state in one sentence what the page wants the reader to do, and that action is visible without hunting.
- [ ] **No keyword stuffing, doorway pages, or hidden text** — read the copy; check for `display:none` text blocks and near-identical pages differing only by a city or keyword. Pass: none. All three are explicit Google spam policy violations.

---

## 4.2 Page-type templates worth having

Presence audit. Mark each Present / Missing / N/A against the business from `business-context.md`.

| Page type | The bar it must clear | Why it earns its place |
|---|---|---|
| **Homepage** | States what you do, for whom, in the first screen | The single most-linked page; often the only one a model reads to decide what the entity is |
| **Service / product pages** | One per offering, not one page listing all of them | One page per intent (`L3-architecture.md`); a combined page ranks for none of them |
| **Location pages** | Only if you genuinely serve those locations, with unique content each | Templated pages with a swapped city name are **doorway pages** and a spam violation. See `L8-local.md` |
| **Comparison / alternatives** | "X vs Y" with a decision table and an explicit recommendation per use case | Heavily surfaced in AI answers — see `L9-ai-search.md` §answer-shaped formats |
| **Pricing** | Actual numbers or at minimum a range | "Contact us for pricing" makes you uncitable for every pricing query in your category |
| **About** | Real people, real credentials, real history | Core E-E-A-T and entity-clarity asset (`L5-content.md`, `L9-ai-search.md`) |
| **Contact** | Physical address, phone, and email where applicable | Trust signal; NAP source of truth for local |
| **Blog / resources** | Organised by topic cluster, not chronology | Chronological archives orphan good content within a year |
| **Legal** | Privacy policy, terms | Expected trust signals; their absence is noticed by quality raters and by users |

---

## 4.3 Social preview metadata — Open Graph and Twitter cards

Not a ranking factor. It is the **rendered appearance of every link to this site pasted into a message, a post, a Slack channel or a chat client** — the one on-page element whose failure is visible to the client's whole audience the first time someone shares a page. A missing or broken `og:image` is a routine post-launch finding precisely because nobody looks at it before launch.

Run per template. Verify per page for the templates that actually get shared.

### The minimum viable set

The Open Graph protocol (`ogp.me`) defines **four required properties**. Everything else is recommended or optional, and the table says which.

| Tag | Status | What it does | Severity if absent |
|---|---|---|---|
| `og:title` | **OGP-required** | Headline of the card. Not the `<title>` — no brand suffix, no "\| Home" | minor |
| `og:type` | **OGP-required** | `website` for most pages, `article` for editorial. Determines which extra properties apply | minor |
| `og:image` | **OGP-required** | The preview image. Constraints below | minor — **major** if present-but-unfetchable, see below |
| `og:url` | **OGP-required** | The canonical, absolute URL. This is the object's permanent ID; a wrong value points the share at the wrong page | minor — **major** when it points at a different page |
| `og:description` | OGP-recommended | 1–2 sentences. Independent of the meta description; the meta description is ad copy for a SERP, this is ad copy for a feed | minor |
| `twitter:card` | Required for an X card | `summary_large_image` (2:1 hero) or `summary` (small square thumbnail). Absent, X may render a plain link with no card | minor |
| `og:image:alt` | OGP-recommended | "If the page specifies an `og:image` it should specify `og:image:alt`" — ogp.me. Accessibility of the shared card, which is not your page and not your DOM | minor |
| `og:image:width`, `og:image:height` | Recommended | Lets a crawler render the card immediately instead of asynchronously downloading and measuring the file | advisory |
| `og:site_name`, `og:locale`, `twitter:site` | Optional | Polish | advisory |

**Escalation, per `../reporting.md` §2.** These baselines are for a site where sharing is incidental. Where distribution is part of the site's stated job — publisher, marketing, campaign, launch, anything with `seo.priority: full` or an editorial output — a missing or broken `og:image` **materially undermines the site's stated job** and is **major**. A broken share card on an internal tool is minor. Say which rule you applied.

**Always major, regardless of site type**, because the share renders visibly broken rather than plainly:

- `og:image` returns a non-200, or is `robots.txt`-blocked, auth-gated, or behind an IP allowlist the platform crawlers cannot reach.
- `og:image` or `og:url` is a **relative** URL. Both must be absolute; relative values are silently dropped.
- `og:url` resolves to a different page than the one serving the tag (a stale template default, or a value that disagrees with `link[rel=canonical]`). The share then points somewhere the sharer did not intend.
- The image is served over `http://` on an `https://` page (use `og:image:secure_url`, or just make `og:image` https).

### Image constraints — as currently published, verified 19 August 2026

Date these. Platforms change them, and a report that does not say when it was written cannot be trusted a year later (`../reporting.md` §6).

| | **Facebook / Open Graph** | **X — `summary_large_image`** |
|---|---|---|
| Recommended size | **1200 × 630** for high-resolution displays | — |
| Minimum | **200 × 200** allowed; **600 × 315** to get the large-image layout at all. Below 600 × 315 the post still renders, "but the size will be much smaller" | **300 × 157** |
| Maximum | — | **4096 × 4096** |
| Aspect ratio | "as close to **1.91:1** as possible" to avoid cropping | **2:1** |
| File size | **must not exceed 8 MB** | **less than 5 MB** |
| Formats | — | **JPG, PNG, WEBP, GIF** (first frame only). **SVG is not supported** |
| Alt text | `og:image:alt` | `twitter:image:alt`, **max 420 characters** |
| Other | The Facebook crawler "only accepts gzip and deflate encodings" — a `br`/`zstd`-only origin is a real failure mode | — |

`summary` (the small square card) is the other X option; if the site uses it, verify against X's `summary` requirements rather than the 2:1 column above.

**The 1.91:1-vs-2:1 gap is why one image cannot be pixel-perfect everywhere.** 1200 × 630 is the pragmatic single answer: it satisfies Facebook's recommendation and sits inside X's bounds, at the cost of a few pixels cropped off X's 2:1 rendering. **Keep anything load-bearing — text, logos, faces — out of the outer ~5% on every edge.** Do not put small type in a share image at all; it is displayed at thumbnail size in most feeds.

**Caching is per-URL, and it bites.** Facebook caches by image URL: replacing the file at the same URL does not update the preview. **Ship a new URL, keep the old file reachable** (existing shares still reference it), and force a re-scrape. The first person to share a URL sees no image at all unless it was pre-cached — so pre-caching is part of a launch checklist, not a nicety.

### Per-page vs site-wide, and the fallback chain

- **Site-wide default is a legitimate baseline, not a target.** A single brand image in the base template beats nothing, and beats a per-page image that 404s. It is a finding only when the page types that get shared — articles, products, case studies, launches — all render the same generic card, because then every share of a content site looks identical and none of them describes what was shared.
- **Per-page tags are required for `og:title`, `og:description` and `og:url` on any template with more than one instance.** A site-wide `og:title` means every shared link reads as the homepage.
- **Duplicate-detection is the same check as titles and descriptions:** crawl, group by `og:title` + `og:image`, and report the count of pages sharing each value. Identical values across distinct content is the finding.
- **Fallback chain, in the order parsers apply it:**
  1. The platform-specific tag (`twitter:title`, `twitter:image`, …) — X's own documentation shows its parser reading `og:description` where the `twitter:` equivalent is absent. `[S]` The broader `twitter:` → `og:` fallback is well-established practice; **verify with the validator rather than assuming it**, and do not tell a client to delete `twitter:` tags on the strength of it.
  2. The Open Graph tag.
  3. The parser's own guess — `<title>`, the meta description, and the first sufficiently large image in the document. **This is the state a "missing tags" finding actually describes**: not "no preview", but "a preview nobody chose", frequently a logo, a sprite, or a tracking pixel.
- **Absent `twitter:card`, an X post may render as a bare link.** The other tags do not rescue it; `twitter:card` is the switch.
- **`og:image` accepts multiple values** — repeat the tag. "The first tag (from top to bottom) is given preference during conflicts" (ogp.me), so a template that appends a default *after* a page-specific one is harmless, and one that prepends it silently wins. Check the order, not just the presence.

### How to verify

**The platform debuggers are authoritative** — they show you what the platform's own parser resolved, including a stale cache, and they are the only way to force a re-scrape.

| Tool | Use it for |
|---|---|
| **Facebook Sharing Debugger** (`developers.facebook.com/tools/debug/`) | The resolved card, scrape errors, the cached image, and the **"Scrape Again"** button that refreshes it |
| **LinkedIn Post Inspector** (`linkedin.com/post-inspector/`) | LinkedIn reads OG tags and caches them separately; a card correct on Facebook can be stale here |
| **X Card Validator** | X's card documentation still points at it. **Verify it is live before putting it in a report** — this tooling has been unstable; if it is unavailable, the curl check below plus the constraint table is the defensible substitute |

**Debuggers require a publicly reachable URL.** For staging, auth-gated, pre-launch, IP-allowlisted or internal pages they are useless, and that is exactly when this check gets skipped. Use curl:

```bash
URL="https://example.com/page"

# 1. Extract every og:/twitter: tag from the RAW HTML.
#    Raw, not rendered: tags injected client-side are invisible to every
#    social crawler, which do not execute JavaScript. See L1-foundations.md §1.4.
curl -sL "$URL" \
  | grep -oiE '<meta[^>]+(property|name)="(og|twitter):[^"]+"[^>]*>' \
  | sed -E 's/.*(property|name)="([^"]+)".*content="([^"]*)".*/\2\t\3/I'

# 2. Fetch the og:image and check it is reachable, its type, and its size.
IMG=$(curl -sL "$URL" | grep -oiE '<meta[^>]+property="og:image"[^>]*>' \
      | head -1 | sed -E 's/.*content="([^"]*)".*/\1/I')
echo "og:image = $IMG"
case "$IMG" in https://*) ;; *) echo "FAIL: og:image is not an absolute https URL" ;; esac
curl -sIL "$IMG" | grep -iE '^HTTP/|^content-type:|^content-length:'

# 3. Confirm the platform crawlers are allowed to fetch it.
curl -s https://example.com/robots.txt | grep -iE 'facebookexternalhit|twitterbot|linkedinbot|Disallow'
```

Read step 2 against the constraint table: `HTTP/2 200`, an image `content-type` (not `text/html` — a soft-404 image endpoint is a classic), `content-length` under 5 MB to satisfy both platforms, and dimensions matching `og:image:width`/`height` if declared. Declared dimensions that disagree with the actual file are their own finding.

**Also check what a headless render adds.** If `og:` tags appear in the rendered DOM but not in `curl` output, they are client-side injected and **no social crawler will ever see them** — that is the same class of failure as a client-side `<title>`, and it is a `L1-foundations.md` rendering finding, not a cosmetic one. `../../scripts/audit_seo.py` collects both halves and can answer this directly.

---

## Questions to ask

1. Read the first two sentences of my key page. Do they answer the question, or introduce it?
2. Does every page have a job, and can I state it in one sentence?
3. If I saw only my title tag and meta description in a list of ten results, would I click mine?
4. Are my headings a genuine outline, or styling decisions?
5. Would a knowledgeable reader learn anything on this page they couldn't get from the top three results?

---

Next: `L5-content.md` — on-page elements are the packaging; the next layer is whether there's anything inside.
