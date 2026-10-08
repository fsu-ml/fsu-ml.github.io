# Layer 5 — Content strategy, intent & quality

Covers: keyword/topic research, intent matching, E-E-A-T, and the 2026 content quality bar.
Load: `seo.priority: full`, or whenever the site publishes content (blog, docs, guides, resources).
Depends on: `business-context.md` (who and what for) and `L3-architecture.md` (cannibalisation findings feed the prune/merge decisions here).

**This is where SEO is actually won or lost.** Technical work removes obstacles; content creates value. A perfectly-crawled site with commodity content ranks nowhere.

🔒 = requires keyword tools, analytics, or Search Console.

---

## 5.1 Research

- [ ] **Keyword/topic research done against real search demand, not guesses** 🔒 — Ahrefs/Semrush/Moz for volume, or GSC → Performance → Queries for what already earns impressions, or Google Trends for direction. Pass: a written list of target queries with a demand figure attached to each.
- [ ] **Every target query classified by intent** — informational / commercial investigation / transactional / navigational. Pass: every query in the list carries a label. Method: search the query and read what Google chose to rank — the SERP *is* the intent classification.
- [ ] **Page type matched to intent** — Pass: transactional queries point at product/service pages; informational queries point at guides. **A blog post will not rank for a transactional query, and vice versa.** This mismatch is the most common reason good content never ranks.
- [ ] **SERP for each target query actually examined** — search it, incognito, in the target locale. Record the format Google is rewarding: lists, videos, forums, local pack, shopping, a fat AI Overview. Pass: the planned page format matches the rewarded format. If the SERP is all Reddit threads, publishing a corporate landing page is a plan to lose.
- [ ] **Realistic difficulty assessment** — look at who ranks in the top 5 and their domain strength. Pass: the target set is winnable given the site's authority. Don't target queries owned by Wikipedia and Amazon on a new domain — go long-tail first and earn the right to compete upward.
- [ ] **Long-tail and question-shaped queries mapped** — "People also ask", Google autocomplete, `answerthepublic`-style expansion, GSC queries at positions 5–20. Pass: a mapped set. These are also the queries AI answer engines fan out into — see `L9-ai-search.md`.
- [ ] **Competitor content gap analysis done** 🔒 — Ahrefs/Semrush Content Gap, or manually: for the top 3 search competitors from `business-context.md`, list what they cover that you don't. Pass: a written gap list, filtered for what's actually worth having.
- [ ] **Existing content inventoried** 🔒 — one row per URL: target query, organic traffic, conversions, last updated, current position. Source: crawl + GA4 + GSC. Pass: the sheet exists and covers every indexable content URL. **You cannot prune what you haven't counted.**

---

## 5.2 Quality — E-E-A-T

Experience, Expertise, Authoritativeness, Trustworthiness.

**Epistemic note:** E-E-A-T is **not a score Google assigns.** It is a description of signals that correlate with ranking, and the framework quality raters are asked to assess when producing training data for the ranking systems. Treat it as a design target, not a metric — anyone selling you an "E-E-A-T score" is selling a proxy of their own invention.

- [ ] **Experience** — content demonstrates the author actually did the thing. Pass: first-hand accounts, original photos (not stock), real numbers from real work. Verify by asking: could this have been written by someone who has never done it?
- [ ] **Expertise** — credentials stated and verifiable where relevant. Pass: named author, stated qualification, and a way to check it (professional profile, publication record, licence number).
- [ ] **Authoritativeness** — the site is cited or referenced elsewhere for this topic. Pass: search `"BRAND" TOPIC -site:DOMAIN` and find independent references. This is largely earned in `L7-authority.md`.
- [ ] **Trustworthiness** — accurate, current, transparent about who's responsible. Pass: contact details, a real About page, visible dates, and a corrections policy if the site publishes claims.
- [ ] **YMYL topics held to a much higher standard** — health, finance, safety, legal, civic. Pass: named, credentialled authors with verifiable expertise, and sourcing for every substantive claim. **Anonymous content in these categories will not rank.**

---

## 5.3 The 2026 content bar

Google's own guidance is unusually direct: create **non-commodity** content. Their published example contrasts:

| Commodity | Non-commodity |
|---|---|
| "7 Tips for First-Time Homebuyers" — common knowledge, could have been written by anyone, or generated wholesale | "Why We Waived the Inspection & Saved Money: A Look Inside the Sewer Line" — specific, experienced, unrepeatable |

- [ ] **Content offers a point of view a model couldn't generate from general knowledge** — the test: paste the page's title into a chat model and ask it to write the article. Pass: the model's version is visibly worse and missing the specifics. Fail: it's equivalent or better, in which case the page has no reason to exist.
- [ ] **Original data, case studies, before/after results, or proprietary experience included where possible** — Pass: at least one number, screenshot, or account that exists nowhere else on the internet.
- [ ] **No mass-produced pages targeting every query variation** — check for programmatic page sets differing only by a substituted term. Pass: none. This is explicitly the **scaled content abuse** spam policy, *and* Google notes it is ineffective anyway because their systems understand relevance without exact-match pages.
- [ ] **AI-assisted content, if used, has genuine human expertise, verification, and editorial oversight layered on** — Pass: a named human reviewed and added something. **The test is unique value, not method of production.** Google's stated position is that AI assistance is not itself a violation; unhelpful mass-produced output is.
- [ ] **Claims are accurate and checkable** — spot-check five statistics against their cited sources. Pass: all five resolve to a real source saying that thing. Hallucinated statistics are a reputational risk and, once noticed, a ranking one.
- [ ] **Thin, outdated, or redundant pages pruned, merged, or improved** 🔒 — using the inventory from §5.1, filter for zero traffic + zero conversions + not updated in 24 months. Pass: each has a decision — improve, merge (with a 301), or remove (with a `410` or a 301 to the nearest relevant page). **Never delete without redirecting.**
- [ ] **Update cadence defined for pages where freshness genuinely matters** — pricing, comparisons, "best X in [year]", anything version-dependent. Pass: a named owner and an interval. Not every page needs this; forcing freshness on evergreen content produces fake `dateModified` bumps, which is worse than nothing.

---

## Questions to ask

1. What can I write that literally nobody else can, because of what I've done or what I have access to?
2. For each existing page: does this deserve to exist? If the honest answer is no — delete, merge, or rewrite.
3. Am I answering the question the searcher has, or the question I want them to have?
4. Who is the human being responsible for this page, and is that visible on it?
5. If a competitor copied this page word for word, would anyone notice the difference?
6. What percentage of my content is restating things available elsewhere?

---

Next: `L6-structured-data.md`. Related: the content-quality audit prompt in `tools.md` scores pages against this layer's bar; `L9-ai-search.md` §12.3 covers which *formats* of good content get pulled into AI answers.
