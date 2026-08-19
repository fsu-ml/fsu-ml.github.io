# Tools & audit prompts

Covers: the tooling for every verification method in these files, plus ready-to-use AI audit prompts.
Load: whenever you need the instrument for a check, or want a ready-made prompt. This is lookup, not reading.
Depends on: nothing.

---

## 1. Tools

### Free and essential

| Tool | What it's the ground truth for | Used by |
|---|---|---|
| **Google Search Console** | Everything Google. Including the **Generative AI performance report** — the only first-party AI-surface data that exists | L1, L2, L6, L9, L10 |
| **Bing Webmaster Tools** | Bing index coverage. Matters for Bing directly **and for ChatGPT search indirectly** | L1, L9, L10 |
| **PageSpeed Insights** | Both field (CrUX) and lab data. **Only the field data counts for ranking** — the top panel, not the Lighthouse score below it | L2 |
| **Google Rich Results Test** (`search.google.com/test/rich-results`) | What Google will actually *do* with your markup. Silently ignores unsupported types | L6 |
| **schema.org validator** (`validator.schema.org`) | Whether the markup is *valid*. Needed alongside the Rich Results Test, not instead of it | L6 |
| **Google Business Profile** | Local presence | L8 |
| **Google Trends** | Demand direction and seasonality | L5 |
| **Chrome DevTools / Lighthouse** | **Lab only.** Useful for diagnosis, useless for judging whether you pass Core Web Vitals | L1, L2 |
| **`curl`** | Status codes, headers, raw HTML, per-user-agent fetch tests. The workhorse of L1 and L9 | L1, L2, L6, L9 |

### Crawlers

| Tool | Notes |
|---|---|
| **Screaming Frog** | Free up to 500 URLs. Orphan pages, redirect chains, duplicate titles, crawl depth, hreflang, anchor text export |
| **Sitebulb** | Paid alternative with stronger reporting and hint prioritisation |

### Paid, optional

- **Ahrefs / Semrush / Moz** — keyword and backlink data. There is no free substitute for backlink data at any depth.
- **AI visibility trackers** — evaluate **only after** you've established manual baselines, and only if they save real time. **No third-party tool has access to internal Google or AI-vendor ranking systems, whatever the marketing says.**

### The free AI-visibility method that costs nothing

A spreadsheet, the prompt set from `question-list.md` Part 2, and 30 minutes a month across four chat interfaces. **This beats most paid tools for a single site.**

### Primary sources worth reading directly

| Topic | URL |
|---|---|
| Google Search Essentials | `developers.google.com/search/docs/essentials` |
| Optimizing for generative AI features | `developers.google.com/search/docs/fundamentals/ai-optimization-guide` |
| AI features and your website | `developers.google.com/search/docs/appearance/ai-features` |
| Creating helpful, reliable, people-first content | `developers.google.com/search/docs/fundamentals/creating-helpful-content` |
| Spam policies | `developers.google.com/search/docs/essentials/spam-policies` |
| Structured data gallery | `developers.google.com/search/docs/appearance/structured-data/search-gallery` |
| Evaluating third-party SEO advice | `developers.google.com/search/docs/fundamentals/third-party-seo` |
| Agent-friendly website best practices | `web.dev/articles/ai-agent-site-ux` |
| Core Web Vitals | `web.dev` |

**Local automation:** `../../scripts/audit_seo.py` collects the mechanical checks (robots.txt, headers, canonicals, titles, status codes, per-user-agent fetch tests) in one pass.

---

## 2. Audit prompts

Paste with the relevant layer file(s) attached. Substitute the bracketed values.

### Full audit
> Using the attached SEO framework, audit `[URL]`. Go layer by layer, L1 through L10. For each checklist item output: item, status (PASS / FAIL / CAN'T VERIFY / N/A), the evidence you used, severity, and a specific fix. Don't guess — mark CAN'T VERIFY where you'd need Search Console, CrUX field data, server logs, or backlink data. Finish with the 10 highest-impact fixes ranked by impact ÷ effort.

### Technical only
> Audit `[URL]` against L1–L3. Check robots.txt, sitemap, canonicals, meta robots, status codes, rendering without JavaScript, internal link structure, and mobile parity. Show me the raw robots.txt and sitemap contents you found, and quote the exact tags you're basing each conclusion on.

### Content quality against Google's 2026 bar
> For each of these URLs `[list]`, assess against L5. Specifically: is this commodity or non-commodity content by Google's own definition? What unique experience, data, or perspective does it contain? Is there a named author with demonstrable expertise? Does it answer the target query in the opening lines? Score each 1–5 with reasoning, and tell me which to delete, merge, or rewrite.

### Passage-extraction test *(AI-specific)*
> Take `[URL]`. Split it into ~150-word passages the way a retrieval system would. For each passage, tell me: (a) what question it would answer if retrieved alone, (b) whether it makes sense without the surrounding page, (c) what context is missing, (d) how to fix it without making the page worse for a human reader.

### AI citation gap analysis *(AI-specific)*
> Here are 10 questions my customers ask: `[list]`. For each: what would a well-informed AI assistant answer today, which sources would it likely draw on, and what would my site need to contain — and what would need to be true off-site — for me to be one of those sources? Separate "content I can write" from "authority I have to earn."

### Entity clarity check *(AI-specific)*
> Based only on what's publicly findable, write a one-paragraph description of `[my company]`. Then tell me: what was ambiguous, what you couldn't determine, what conflicting information you found, and where the conflicts came from.

### AI crawler governance review *(AI-specific)*
> Fetch and analyze `https://[domain]/robots.txt`. List every user-agent rule, identify which AI crawlers are allowed vs. blocked and what each bot's actual role is (training / retrieval / user-fetch), and flag anything that would make me invisible in AI answers. Then propose a robots.txt reflecting this policy: `[state your intent]`.

### Competitor gap
> Compare `[my URL]` to `[competitor URLs]` for the query `[query]`. What do they cover that I don't? What's their content structure? What signals of expertise do they show? What would I need to do to be a genuinely better result — not just a longer one?

### Cannibalization
> Given this list of my URLs and their target queries `[list]`, identify pages competing for the same intent. For each conflict recommend: keep which, merge which, redirect which — with reasoning.

### Schema review
> Extract all JSON-LD from `[URL]`. Validate the syntax, check whether each type still produces rich results as of 2026, verify every property matches visible page content, and flag anything I'm maintaining for a deprecated feature.

### Migration safety check
> I'm changing `[describe the change]`. Build me a pre-launch SEO checklist: URL mapping, redirect plan, what commonly breaks, and a post-launch verification list for the first 48 hours.

### Monthly standing check
> Re-run the AI visibility prompt set from `question-list.md` Part 2 against `[list of engines]`. Compare to last month's results in `[paste previous log]`. Report changes in citation status, new competitors appearing, and any factual errors about my business.

---

**Caveat on all of the above:** an AI cannot see your Search Console data, your CrUX field data, your backlink profile, or your revenue. Anything marked 🔒 in these files requires you to pull real data. **Don't let an audit that skips those pretend to be complete.**
