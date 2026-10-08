# Triage — what to fix first

Covers: the P0–P4 severity ladder, the things that actively hurt you, a 90-day sequence, and the seven-line short version.
Load: always, at the end of the audit. This is what you read when there are 40 findings and limited time.
Depends on: the findings from L1–L10. Merges the source's Appendices D, E, F, and G.

---

## 1. The severity ladder

Assign every finding a priority. Work top to bottom. **Resist skipping ahead to the interesting parts** — the interesting parts are almost always P3.

*This P0–P4 ladder is local to the SEO track and does not ship. Translate to the canonical `blocker | major | minor | advisory` before writing a finding; crosswalk in `../reporting.md` §2. Note that `P2` here means High, whereas `P2` in `../animation-and-motion.md` and `../dynamic-loading.md` means advisory.*

| Priority | Trigger | Why it comes first |
|---|---|---|
| **P0 — Emergency** | Not indexed at all · `Disallow: /` live · sitewide `noindex` · manual action · security warning · site down | Everything else is worthless until this is fixed. **A P0 halts the audit** — report it, get it fixed, re-crawl, continue |
| **P1 — Critical** | Key pages unindexed · content invisible without JS · HTTPS broken · mobile unusable · 404s on money pages · retrieval bots blocked | Blocks visibility entirely for the affected pages and surfaces |
| **P2 — High** | Core Web Vitals in the "poor" band · missing or duplicate titles · keyword cannibalization · broken internal links · no Search Console access · not in Bing's index | Significant drag on potential you already have |
| **P3 — Medium** | Missing schema · weak internal linking · thin content · unoptimised GBP · no measurement baseline · passages that don't survive extraction | Meaningful upside, not urgent |
| **P4 — Ongoing** | New content · earning corroboration · refreshes · AI visibility monitoring · original data | Compounding returns; never "finished" |

**Tie-breaker within a priority band:** rank by impact ÷ effort, then by whether the fix is at template level. A template fix multiplies across every page it renders; a page fix does not.

**Severity is relative to `business-context.md`.** A missing `LocalBusiness` schema is P2 for a dentist and N/A for a portfolio. Never assign severity without the profile.

---

## 2. Things that actively hurt you

From Google's spam policies and common real-world failure patterns. Presence of any of these is at least P1, regardless of what else the audit found.

| Failure | Why it's here |
|---|---|
| Blanket `Disallow: /` shipped from staging | **Still the single most common catastrophe** |
| Sitewide `noindex` left on after a redesign | Same class, same cause, harder to spot — often header-level, invisible in view-source |
| Blocking retrieval bots while paying for AI-visibility services | Paying to be found in engines you've locked out |
| Cloaking — serving different content to crawlers than to users, **including to AI crawlers** | Confirmed policy violation |
| Scaled content abuse — mass-generated pages targeting every query variation | Explicit spam policy, and ineffective anyway |
| Buying or exchanging links for ranking purposes | Explicit link spam policy |
| Doorway pages — near-identical pages differing only by city or keyword | Explicit spam policy. See `L8-local.md` |
| Keyword stuffing and hidden text | Explicit spam policy |
| Expired-domain abuse and site-reputation abuse ("parasite SEO") | Explicit spam policy |
| Intrusive interstitials on entry | Explicit demotion |
| Redesigning or migrating without a URL redirect map | The most expensive self-inflicted traffic loss there is |
| Publishing unverified AI output with fabricated facts or invented citations | Reputational and ranking risk; the fabricated-citation variant is checkable by anyone |
| Deleting pages without redirecting them | Throws away every link and signal the page had earned |
| Buying reviews or astroturfing forum mentions | Spam risk, and per Google's 2026 guidance less effective than marketed |
| Building for SERP features that no longer exist | FAQ and HowTo rich results are dead. See `L6-structured-data.md` |

---

## 3. A sane 90-day sequence

If the full framework is overwhelming, this is the order that produces results fastest.

**Days 1–7 — Instrument and diagnose**
- [ ] Verify Search Console and Bing Webmaster Tools
- [ ] Read robots.txt line by line; check for `noindex` and blocked crawlers (`L1-foundations.md`, `L9-ai-search.md` §9.9)
- [ ] Crawl the site with Screaming Frog
- [ ] Pull Core Web Vitals **field** data for mobile
- [ ] Run the reality-check prompts from `question-list.md` §AI and log the results — **this is your AI baseline**
- [ ] Write down your `business-context.md` answers
- [ ] Take the dated baseline snapshot (`L10-measurement.md` §10.2)

**Days 8–30 — Fix what's blocking**
- [ ] Everything at P0 and P1
- [ ] Fix indexation gaps
- [ ] Fix titles and meta descriptions across key pages
- [ ] Resolve cannibalization
- [ ] Add or correct `Organization` and `Article` schema
- [ ] Decide your AI crawler policy and implement it

**Days 31–60 — Strengthen the core**
- [ ] Improve the 10 pages that already earn impressions but few clicks — **fastest wins available**
- [ ] Apply the passage test to key pages and restructure where it fails (`L9-ai-search.md` §9.4)
- [ ] Fill the missing answer-shaped formats: comparison, alternatives, pricing, criteria
- [ ] Sort out internal linking around your commercial priorities
- [ ] Fix Core Web Vitals on your worst **template**
- [ ] Claim and complete Google Business Profile if relevant

**Days 61–90 — Build things that compound**
- [ ] Publish one genuinely original asset — data, research, a tool, a case study with real numbers
- [ ] Pursue presence on the third-party sources that dominate AI answers in your category
- [ ] Establish the maintenance cadence from `L10-measurement.md` §10.3
- [ ] Re-run the AI visibility prompt set and compare to your day-1 baseline
- [ ] Write down what changed, what you did, and when — so next quarter's decisions have evidence

**Then repeat.** SEO is not a project with an end date; it's a maintenance discipline with occasional bursts of construction.

---

## 4. The short version

If you remember nothing else:

1. **Be findable.** Indexed, crawlable, renders without JavaScript, fast on a mid-range phone. Nothing else matters until this is true.
2. **Be clear.** One page per intent, honest titles, real headings, answers before elaboration, schema that matches reality.
3. **Be specific.** Numbers, dates, names, prices, first-hand experience. Vagueness is uncitable and uncompetitive.
4. **Be worth citing.** Publish something that couldn't have been generated without you. The only durable advantage.
5. **Be corroborated.** Presence off your own site increasingly determines whether you get named in an answer.
6. **Be measured.** Baseline first, changes logged, decisions made on data rather than vibes.
7. **Be patient.** Months, not weeks. And be suspicious of anyone who promises otherwise.

Everything labelled "AI SEO", "AEO", or "GEO" that's genuinely effective reduces to doing items 1–6 with more rigour. **Google has said this in writing.** The tactics that genuinely *differ* — passage-level extractability, entity clarity, crawler governance, off-site corroboration, and monitoring what the engines actually say about you — are in `L9-ai-search.md`, and they're additive, not a replacement.

---

## 5. Currency

This reference set reflects publicly available guidance as of **July 2026**. Key dated facts relied on:

- **Google's official generative AI optimization guide**, published 2026-05-15 and updated since — source for the mythbusting section, eligibility requirements, the query fan-out and RAG description, and the commodity vs. non-commodity content framing
- **FAQ rich results ended 2026-05-07**; Rich Results Test and Search Console reporting support dropped June 2026; Search Console API support ends August 2026. FAQPage remains valid schema.org vocabulary
- **HowTo rich results retired in 2023**; seven further structured data types retired June 2025
- **INP replaced FID** as a Core Web Vital in March 2024
- **Core Web Vitals thresholds** — LCP ≤ 2.5 s, INP ≤ 200 ms, CLS ≤ 0.1, at the 75th percentile of field data over a 28-day window
- **AI crawler user-agent names** as documented by their operators as of mid-2026

**Search changes constantly, and this space changes faster than the rest of it.** Re-verify anything version-dependent — especially structured data support, crawler names, and AI-surface behaviour — against Google Search Central and each vendor's own documentation before acting on it. **Treat any third-party claim about AI ranking factors as unproven unless the vendor themselves has confirmed it.**
