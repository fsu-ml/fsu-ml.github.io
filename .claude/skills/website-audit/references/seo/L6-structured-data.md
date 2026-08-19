# Layer 6 — Structured data

Covers: what schema types still produce results in 2026, the implementation checklist, and validation.
Load: `seo.priority: basic|full`, or `seo.ecommerce: true`, or `seo.local_business: true`, or the site already ships JSON-LD.
Depends on: `L5-content.md` — schema describes content that must already exist and must match it exactly.

Structured data (JSON-LD, in the page's `<head>`) tells machines what a page *is*, not just what it says. **Two things changed materially in 2025–26 and most checklists haven't caught up.**

---

## 6.1 What changed — support status as of July 2026

| Type | Status |
|---|---|
| **FAQPage** | ❌ FAQ rich results stopped appearing in Google Search on **May 7, 2026**. Search Console reporting and Rich Results Test support dropped **June 2026**; Search Console API support ends **August 2026**. The schema type itself remains valid schema.org vocabulary and does not need removing — Google states unused structured data causes no problems, but it has no visible effect in Search. |
| **HowTo** | ❌ Rich results retired in **2023**. Valid markup, zero SERP effect. |
| **Seven further types** | ❌ Retired from search appearance in **June 2025**. |
| **Article / BlogPosting** | ✅ Supported |
| **Organization** | ✅ Supported — connects your site to your brand entity |
| **Product / Merchant listing / Review snippet** | ✅ Supported, increasingly important for shopping surfaces |
| **BreadcrumbList** | ✅ Supported |
| **LocalBusiness** | ✅ Supported |
| **Event, Job posting, Recipe, Video, Q&A, Profile page, Software app, Dataset, Course** | ✅ Supported |

**Do not rip out existing FAQPage markup just because of the deprecation.** Leave it if removal costs engineering time. Just stop building new implementations expecting a SERP payoff, and keep the FAQ *content* if it answers real questions (see `L9-ai-search.md` §answer-shaped formats).

---

## 6.2 Implementation checklist

- [ ] **JSON-LD used, not microdata** — `curl -s URL | grep -c 'application/ld+json'`. Pass: ≥1, and no `itemprop=` scattered through the body. JSON-LD is easier to maintain and validate, and it's what Google recommends.
- [ ] **`Organization` schema sitewide** — must carry `name`, `url`, `logo`, and `sameAs` links to every official profile you control. Pass: present on at least the homepage, with a populated `sameAs` array. This is the entity anchor — see `L9-ai-search.md` §entity clarity.
- [ ] **`Article`/`BlogPosting` on content pages** — with `headline`, `author`, `datePublished`, `dateModified`, `publisher`. Pass: all five present. **`author` must be a `Person` entity object, not a bare text string:**
  ```json
  "author": {
    "@type": "Person",
    "name": "Jane Doe",
    "url": "https://domain.com/about/jane-doe",
    "sameAs": ["https://www.linkedin.com/in/janedoe"]
  }
  ```
- [ ] **`BreadcrumbList` matching the visible breadcrumbs** — compare the JSON-LD `itemListElement` names to the on-page trail. Pass: identical order and labels. See `L3-architecture.md`.
- [ ] **`Product` with price, availability, and review data on commerce pages** — `seo.ecommerce: true`. Pass: `offers.price`, `offers.priceCurrency`, `offers.availability` present and matching what the page displays right now.
- [ ] **`LocalBusiness` with `address`, `geo`, `openingHoursSpecification`** — `seo.local_business: true`. Pass: present and byte-identical to the Google Business Profile — see `L8-local.md`.
- [ ] **Markup matches visible on-page content exactly** — read the JSON-LD and the page side by side. Pass: every marked-up value appears on the page. **Mismatched markup is a policy violation**, not a technicality: marking up a 4.9 rating that isn't displayed, or a price you don't charge, risks a manual action.
- [ ] **Validated in Google's Rich Results Test *and* the schema.org validator** — `search.google.com/test/rich-results` (tells you what Google will *do* with it) and `validator.schema.org` (tells you whether it's *valid*). Pass: zero errors in both; warnings triaged. You need both — Google's tool only reports on types it supports, so it silently ignores valid-but-unsupported markup.
- [ ] **Schema versioned in the same commit as the content it describes** — check whether markup lives in the template alongside the content or in a separate CMS field/plugin nobody edits. Pass: they move together. **Drift between markup and page is the most common long-term failure** — the price changes, the schema doesn't.
- [ ] **Structured data monitored in Search Console's Enhancements reports** 🔒 — GSC → Enhancements. Pass: reviewed on the monthly cadence in `L10-measurement.md`, errors at zero.
- [ ] **No markup maintained for a rich result that no longer exists** — cross-check every type present against the table in §6.1. Pass: nothing new is being built for FAQPage or HowTo. Existing markup can stay.

---

## 6.3 The AI-citation caveat — keep this distinction visible

| Claim | Epistemic status |
|---|---|
| Schema earns rich results and improves machine comprehension | **Google says** — documented, and directly observable in the Rich Results Test |
| Schema improves the odds of being cited in AI Overviews / AI Mode | **Unproven.** Google explicitly states structured data is **not required** for AI Overviews or AI Mode, and there is **no special schema.org markup to add for them.** |
| "Schema markup increases AI citation likelihood by X%" | **Hype.** Third-party correlational claims, unverified by Google or any AI vendor. |

**Implement schema because it earns rich results and improves machine comprehension — not because someone promised you AI citations.** If a vendor's pitch rests on the third row, that is the whole pitch.

---

## Questions to ask

1. Does my schema describe reality, or the reality I wish I had?
2. Is my author markup a real person with a real bio page?
3. If a machine read only my JSON-LD and nothing else, what would it think this site is?
4. Am I maintaining markup for a rich result that no longer exists?

---

Next: `L7-authority.md`. Related: the schema review prompt in `tools.md` extracts and validates all JSON-LD from a URL in one pass.
