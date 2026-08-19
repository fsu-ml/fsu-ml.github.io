# Layer 3 — Information architecture, internal linking & URLs

Covers: site structure, internal link equity, URL design, pagination, faceting, hreflang, cannibalisation.
Load: `seo.priority: full`, or `site.url_count > ~30`, or `seo.ecommerce: true`, or `site.multilingual: true`.
Depends on: `L1-foundations.md` (a page that can't be crawled can't be linked into anything meaningful).

Search engines infer importance from structure. **A page nobody links to is a page nobody ranks.** Requires a crawl — most checks here cannot be done by inspecting single URLs.

🔒 = requires first-party data or a full-site crawl.

---

## 3.1 Reachability

- [ ] **Every important page reachable within 3 clicks of the homepage** — Screaming Frog → Site Structure tab, or the `Crawl Depth` column. Pass: all commercially important URLs at depth ≤ 3. Fail: money pages at depth 5+, typically buried under a blog archive.
- [ ] **No orphan pages** 🔒 — crawl the site *and* import the XML sitemap (Screaming Frog: Configuration → Spider → Crawl Linked XML Sitemaps), then Reports → Orphan Pages. Pass: empty. Orphans are in the sitemap with zero internal links pointing at them — usually pages that were built and forgotten, or a broken template.
- [ ] **Navigation is HTML-crawlable and reflects real topical structure** — `curl -s https://DOMAIN/ | grep -oE '<a [^>]*href="[^"]*"[^>]*>[^<]*'` on the nav region. Pass: real `<a href>` links (see the rendering checks in `L1-foundations.md`) whose grouping matches how the business actually thinks about its offerings.
- [ ] **Footer/sidebar links aren't diluting into hundreds of low-value links** — count outbound internal links on a typical page: Screaming Frog → Outlinks. Pass: a defensible number, mostly contextual. Fail signature: 200+ identical footer links on every page, which flattens the priority signal the structure is supposed to carry.

---

## 3.2 URL structure

- [ ] **URLs short, readable, lowercase, hyphenated, stable** — sample 20 URLs from the sitemap. Pass: `/services/roof-repair/` not `/Services/Roof_Repair.aspx?id=8842`. Uppercase in URLs creates case-sensitivity duplicates on most servers.
- [ ] **URLs describe hierarchy sensibly** — Pass: the path reads as a breadcrumb. A URL should let a human guess the page's parent.
- [ ] **URLs don't change casually — every change requires a 301** — check the redirect map in version control or the CMS. Pass: a documented process exists. Fail: URL edits ship without redirects (see `triage.md` — this is in the "actively hurts you" list).
- [ ] **Trailing-slash and index-file consistency enforced** — `curl -sI https://DOMAIN/about` vs `/about/` vs `/about/index.html`. Pass: one canonical form serving 200, the rest 301ing to it.

---

## 3.3 Internal linking

- [ ] **Internal links use descriptive anchor text** — Screaming Frog → Bulk Export → Links → All Anchor Text; then count occurrences of `click here`, `read more`, `learn more`. Pass: anchors describe the destination. `read more` on every card is a wasted signal repeated site-wide.
- [ ] **Internal links reflect commercial priority** — Screaming Frog → Bulk Export → Inlinks, then count inlinks per URL and compare that ranking to the business's own priority list from `business-context.md`. Pass: the two orderings roughly agree. Fail signature: the blog has 400 inlinks and the pricing page has 3.
- [ ] **Topic clusters: a strong pillar page linked to and from its supporting pages** — pick a topic, map the links. Pass: pillar → each supporting page, and each supporting page → pillar. Fail: a set of posts on the same topic with no links between them.
- [ ] **Breadcrumbs implemented — visible *and* `BreadcrumbList` schema** — check the rendered page for a visible trail, and `curl -s URL | grep -A5 BreadcrumbList`. Pass: both present and matching. Markup that disagrees with the visible trail is a policy violation — see `L6-structured-data.md`.

---

## 3.4 Scale problems

- [ ] **Pagination handled with real crawlable links and self-referencing canonicals** — inspect `/page/2/`. Pass: `<a href>` links to prev/next, and page 2's canonical points at page 2 (**not** at page 1 — that de-indexes the deeper pages and orphans everything on them).
- [ ] **Faceted navigation controlled** — `seo.ecommerce: true` especially. Crawl and count URLs containing `?`; check which facet combinations are linkable. Pass: a documented rule — which facets are indexable (usually single-facet, high-demand ones), which are `noindex`, which are robots-blocked or `nofollow`ed. **This is the #1 source of index bloat on ecommerce**: 12 filters combinatorially generate more URLs than the site has products.
- [ ] **Crawl budget spent on pages that matter** 🔒 — GSC → Settings → Crawl stats → By file type / By purpose. Pass: crawl requests concentrated on content URLs, not parameter noise.

---

## 3.5 Multilingual / multiregional

Only if `site.multilingual: true`.

- [ ] **`hreflang` correct, reciprocal, with `x-default`** — `curl -s URL | grep -i hreflang`, then fetch each alternate and confirm it points back. Pass: every language pair references the other in both directions, plus one `x-default`. Validate with Screaming Frog → Hreflang tab, or GSC's International Targeting report 🔒. Non-reciprocal hreflang is silently ignored.
- [ ] **hreflang consistent with canonicals** — Pass: each page's canonical is itself, and hreflang points at the *canonical* URL of each alternate. A hreflang pointing at a URL that canonicalises elsewhere invalidates the cluster.
- [ ] **Language codes valid** — ISO 639-1 language, optional ISO 3166-1 Alpha-2 region: `en`, `en-GB`, `es-MX`. Pass: no invented codes (`en-UK` is invalid; `uk` is Ukrainian).

---

## 3.6 Cannibalisation

- [ ] **One page per intent — no two pages competing for the same query** 🔒 — GSC → Performance → filter by query → Pages tab. If more than one URL earns impressions for the same query, and the winner alternates over time, that's cannibalisation. Also test with `site:DOMAIN "target phrase"`. Pass: one clear owner per intent.
  - Resolution: **keep** the strongest, **merge** the content of the others into it, **301** the merged URLs. Do not simply delete — see `triage.md`.
  - The `Cannibalization` prompt in `tools.md` automates the triage once you have the URL/query list.

---

## Questions to ask

1. If I drew my site as a tree from memory, would it match what a crawler finds?
2. Which pages are my most important commercially — and do my internal links reflect that?
3. Are there pages that exist purely because someone made them once and nobody deleted them?
4. Do two or more of my pages target the same query? Which one should win, and can I merge or redirect the rest? 🔒
5. When I add a new page, what's the process for linking it into the existing structure?

---

Next: `L4-onpage.md`. Related: `L6-structured-data.md` for `BreadcrumbList`; `L5-content.md` for what to do with the pages cannibalisation says to merge.
