# Layer 9 — AI search visibility (AEO / GEO)

Covers: what Google says vs. what practitioners believe vs. hype; the retrieval pipeline; the passage test; answer-shaped formats; entity clarity; corroboration; engine differences; crawler governance; AI-visibility measurement; risk.
Load: `seo.ai_visibility: true`, or `seo.priority: full`, or the client is asking about (or being sold) AEO/GEO/LLM-visibility services.
Depends on: L1–L6. This layer is **additive, not a replacement.** Content that already ranks well is disproportionately likely to be cited; nothing here substitutes for being indexed.

**Read this section skeptically.** There is more marketing than evidence in this space, and Google has published an explicit mythbusting position on it. §9.9 is the most valuable part of this file — read it before spending money.

---

## 9.1 The three epistemic tiers

Keep these separate in every recommendation you make.

| Tier | Meaning | Example |
|---|---|---|
| **Google says** | Documented in Google's published guidance; citable | "Structured data is not required for AI Overviews or AI Mode" |
| **Practitioners believe** | Correlational, observed, plausible, unconfirmed by any vendor | "Comparison tables get quoted more often than prose" |
| **Hype** | Marketed, unevidenced, sometimes contradicted by the vendor | "Publish llms.txt to get cited" |

Do not flatten these into confident advice. An audit that presents tier 3 as tier 1 is worse than no audit.

---

## 9.2 What Google says — official, May 2026

Google published its first formal guide to generative AI features on **2026-05-15**. Its stated position: AI Overviews and AI Mode are rooted in the core Search ranking and quality systems, using retrieval-augmented generation and query fan-out over the same index. **Optimising for AI search *is* SEO.** There is no separate AI algorithm to game.

**Required to be eligible for AI Overviews / AI Mode:**
- [ ] **Page is indexed and eligible to appear in Google Search with a snippet** — check for `nosnippet`, `max-snippet:0`, or `data-nosnippet` on key pages: `curl -s URL | grep -iE 'nosnippet|max-snippet'`. Pass: none present on pages you want cited. A `nosnippet` directive removes you from AI surfaces.
- [ ] **Site included in Search generative AI features in Search Console settings** 🔒 — GSC → Settings. Pass: enabled.

**Explicitly stated as unnecessary for Google:**

| Thing | Google's position |
|---|---|
| `llms.txt` and other "special" AI markup | ❌ Google Search doesn't use them. Harmless to publish; will neither help nor hurt Google rankings |
| "Chunking" content into tiny pieces | ❌ There is no ideal page length |
| Rewriting content specifically for AI systems | ❌ Models understand synonyms and intent |
| Special schema markup for AI features | ❌ None exists |
| Chasing inauthentic mentions across the web | ❌ Called out as less useful than it appears |

**What Google says actually works:** the fundamentals — unique non-commodity content, clear technical structure, crawlability, good page experience, high-quality images and video, and for commercial sites, Merchant Center feeds and a Google Business Profile.

---

## 9.3 How an AI answer actually gets built

Understanding the pipeline tells you where you can intervene.

```
USER PROMPT
   ↓
1. INTENT PARSING      → the model interprets what's really being asked
   ↓
2. QUERY FAN-OUT       → the prompt is decomposed into several sub-queries
                          "best CRM for a 5-person agency" becomes
                          → "CRM pricing small business"
                          → "CRM features for agencies"
                          → "CRM comparison 2026"
   ↓
3. RETRIEVAL           → each sub-query hits an index (Google's, Bing's, or the
                          vendor's own crawl) and returns candidate documents
   ↓
4. CHUNKING & RANKING  → documents are split into passages; passages are scored
                          for relevance to the sub-query
   ↓
5. GROUNDING           → the model reads the selected passages
   ↓
6. SYNTHESIS           → an answer is composed from multiple passages
   ↓
7. CITATION SELECTION  → a subset of sources gets named and linked
   ↓
ANSWER
```

| Pipeline stage | What determines whether you're in it | What you control |
|---|---|---|
| Retrieval | Index presence and classic ranking strength | `L1-foundations.md` – `L5-content.md`. Not in the index → nothing else applies |
| Chunking | Whether your passages stand alone when severed from the page | Content structure — headings, self-contained paragraphs |
| Passage ranking | Whether a single passage answers a sub-query cleanly | Front-loaded answers, direct phrasing |
| Grounding | Whether your claims are specific and verifiable | Data, dates, named sources, attribution |
| Citation | Whether the model trusts you enough to name you | Authority, entity clarity, corroboration elsewhere |

**The last row is the one most people miss. A model can use your content without citing you.** Getting retrieved is a content problem; getting *named* is a trust problem.

---

## 9.4 The passage test

Because retrieval operates on chunks, not pages, this is the most useful single test in this reference:

> **Take any 150-word passage from your page, out of context, with no heading and no surrounding paragraphs. Does it answer a real question completely and attributably on its own?**

If it doesn't, it won't survive extraction.

- [ ] **Each section begins by answering, then elaborates — never the reverse** — read the first sentence under each H2. Pass: it is an answer.
- [ ] **Pronouns and vague back-references minimised** — search the page for "this approach", "as mentioned above", "the former", "as we saw". Pass: near-zero. **A retrieved chunk has no "above".**
- [ ] **Entities named explicitly rather than implied** — Pass: "Shopify's Basic plan costs $X" not "their entry tier costs $X".
- [ ] **Numbers, dates, versions, and units stated inline rather than assumed from context** — Pass: "as of July 2026, 2.5 seconds" not "currently, under three".
- [ ] **Each H2/H3 phrased close to how a person would actually ask the question** — read the headings alone. Pass: they read as a question list.
- [ ] **Definitions given in a single clean sentence near the top of the relevant section** — Pass: one extractable sentence per defined term.
- [ ] **Comparisons rendered as tables** — Pass: any A-vs-B content is a table. Tables extract cleanly and are quoted often *(practitioners believe; not vendor-confirmed)*.
- [ ] **Lists used for genuinely enumerable things** — steps, criteria, options. Pass: not used as a formatting habit on prose that isn't a list.
- [ ] **Key claims are self-sourcing** — Pass: "According to [source, date]…" *inside* the passage itself, not in a footnote the chunk won't carry.

**Anti-pattern:** don't shred your page into disconnected fragments in pursuit of this. Google's guidance is explicit that chunking is unnecessary and there is no ideal page length. The goal is **coherent prose that also happens to survive being cut** — write for the human, then check the passages hold up.

---

## 9.5 Answer-shaped content formats

Certain formats get pulled into AI answers far out of proportion to their share of the web *(practitioners believe, consistently observed; not vendor-confirmed)*. Audit which you have:

- [ ] **Direct definition pages** — "What is X" with a one-sentence answer up top
- [ ] **Comparison pages** — "X vs Y", with a decision table and an explicit recommendation per use case
- [ ] **Alternatives pages** — "Alternatives to X", including honest treatment of competitors
- [ ] **Pricing transparency** — actual numbers or ranges. **"Contact us for pricing" makes you uncitable for every pricing query in your category**
- [ ] **Specification and criteria pages** — "How to choose an X", with named criteria
- [ ] **Original data** — surveys, benchmarks, internal statistics. **The highest-value asset in AI search, because it's the one thing that cannot be synthesised from other sources**
- [ ] **Case studies with numbers** — before/after with real figures and context
- [ ] **Troubleshooting / error-message pages** — extremely high-intent, chronically under-served
- [ ] **Glossary of your domain's terms** — cheap to build, feeds definitional queries
- [ ] **Genuinely useful Q&A sections** — keep the ones answering real questions; delete the ones that existed only to chase the now-dead FAQ rich result (`L6-structured-data.md`)

**Questions:** If someone asks an AI "what's the best [my category] for [my customer type]", is there any page on my site that could serve as the source? Do I publish any number that exists nowhere else on the internet? Would I be comfortable if an AI quoted my pricing page verbatim — and if not, why is it vague? Am I honest about who my product isn't for? *(Models reward and cite balanced treatment; pure sales copy reads as low-trust.)*

---

## 9.6 Entity clarity

Retrieval systems reason about **entities**, not just strings. If a model isn't confident about what your organisation is, it won't put your name in an answer.

- [ ] **Brand name used consistently everywhere** — one spelling, one capitalisation, no drifting variants. Verify: search the brand and count spelling variants across your own properties.
- [ ] **One-sentence organisation description identical across site, GBP, LinkedIn, Crunchbase, and every directory** — open them side by side and diff. Pass: the same sentence.
- [ ] **`Organization` schema with `sameAs` pointing to every official profile you control** — `L6-structured-data.md`. Pass: array covers LinkedIn, X, GitHub, Crunchbase, YouTube, industry profiles.
- [ ] **About page states what the company is, when founded, where it operates, who runs it** — Pass: all four, in plain text, not implied by a photo carousel.
- [ ] **Named authors with real bio pages, credentials, and `sameAs` links to professional profiles** — Pass: every byline resolves to a real person.
- [ ] **Authors have a footprint beyond your own domain** — search the author's name minus your domain. Pass: independent results. **This is what separates "a name on a byline" from a recognised entity.**
- [ ] **Consistent NAP if you're a physical business** — `L8-local.md`.
- [ ] **Product and feature names used consistently, not renamed casually across pages** — Pass: one name per thing, site-wide.
- [ ] **Wikipedia/Wikidata presence if you legitimately qualify** — **never fabricate notability.** Failing Wikipedia's notability bar and pushing anyway produces a deletion log entry, which is worse than absence.
- [ ] **Disambiguation handled if your brand name collides with a common word or another company** — search the brand alone. Pass: the About page and schema make the distinction unambiguous.

**Questions:** Ask three different AI assistants "What is [my company]?" — do they get it right, wrong, or not know? If wrong, where did the wrong information come from? *(That's your fix target.)* Does my description differ between my homepage and my LinkedIn — why? Is there a single page on the internet that authoritatively states what we do, that we control?

---

## 9.7 Corroboration — the part that isn't on your website

**The biggest strategic difference between classic SEO and AI visibility.** Ranked search asks "which page is best?" Answer engines synthesise across sources and lean toward claims that appear in **more than one independent place.**

Consequence: a large share of what determines whether you're recommended happens on sites you don't own.

- [ ] **Presence audited on the third-party sources that dominate your category's AI answers** 🔒 — run five category prompts, list every domain cited, then check whether you appear on any of them. Typically some mix of:
  - Review platforms (G2, Capterra, Trustpilot, industry-specific equivalents)
  - Reddit and specialist forums
  - YouTube — **with transcripts; this is how video enters text retrieval**
  - Industry publications and newsletters
  - Comparison and "best of" roundups
  - Wikipedia / Wikidata where legitimate
- [ ] **Reviews present and current on the platforms that matter in your vertical** — see `L7-authority.md`
- [ ] **Roundup and "best X" articles in your category identified — are you in them?** 🔒 — search `best [category] 2026`, list the top 10 articles, check each. Pass: a worked list with outreach targets.
- [ ] **Independent coverage earned through genuine newsworthiness, data, or expert commentary**
- [ ] **Community participation is authentic** — answering real questions, not seeding mentions

**A hard boundary:** manufacturing mentions is both a spam risk and, per Google's 2026 guidance, **less effective than it appears.** Astroturfing Reddit, buying reviews, and paying for "brand mention" packages are the 2026 equivalent of buying links. The strategy is to **be genuinely worth mentioning and then be findable** — not to fake the mentions.

**Questions:** When an AI recommends products in my category, which *domains* does it cite? (Check five prompts; the pattern is usually obvious within minutes.) Am I present on those domains at all? What's said about us in places we don't control — is it accurate? Which of those sources could we legitimately earn a place on in the next 90 days?

---

## 9.8 Engine-by-engine differences

The engines are not interchangeable. Where your effort pays off depends on which ones your buyers actually use.

| Engine | Index it draws on | What it tends to reward | Practical implication |
|---|---|---|---|
| **Google AI Overviews / AI Mode** | Google's own index | The same things classic Google rewards; established sources; structured, helpful content | Existing SEO investment transfers directly. Requires being indexed, snippet-eligible, and enabled for generative AI features in Search Console |
| **ChatGPT (search)** | Substantially Bing-derived, plus its own crawlers | Semantic authority, established presence, clarity | **Bing Webmaster Tools is not optional.** If Bing hasn't indexed you, ChatGPT search can't find you |
| **Perplexity** | Live web retrieval, citations-first | Recency, explicit dates, clean technical health, direct answers | Fastest feedback loop of the major engines — good place to test whether changes move anything. Robots.txt changes typically honoured within ~a day |
| **Gemini** | Google's index | Same as Google, plus multimodal content | Overlaps heavily with AI Overviews work |
| **Claude** | Web search where enabled, plus training data | Cites more sparingly and conservatively | Hardest to influence directly; benefits from the same authority and clarity signals |
| **Copilot** | Bing | Same as Bing/ChatGPT | Covered by Bing work |

- [ ] **Bing index coverage verified** 🔒 — Bing Webmaster Tools, or `site:DOMAIN` on bing.com. Pass: page count roughly matches your real page count.

**Strategy: don't spread across all of them at once.** Pick the one where your buyers actually are, get results, then expand. Already rank in Bing → start with ChatGPT. Strong E-E-A-T and video → start with Google's AI surfaces. Want fast signal to learn from → start with Perplexity.

---

## 9.9 AI crawler governance

`robots.txt` in 2026 is a governance document, not just a traffic-control file. **Training access, retrieval access, and user-initiated fetches are separate decisions.**

| Role | What it does | Blocking it means |
|---|---|---|
| **Training bots** | Fetch content to train future models | Your content won't inform model weights. No direct effect on being cited in live answers |
| **Retrieval / search bots** | Build the index used to answer live queries | **You become invisible in that engine's answers.** This is the consequential one |
| **User-initiated fetchers** | Fetch a page because a specific user asked about it | Users can't get your page pulled into their chat, even deliberately |

Named agents to make a decision about *(verify against each vendor's current docs — these change)*:

```
# OpenAI
User-agent: GPTBot          # training
User-agent: OAI-SearchBot   # ChatGPT search index
User-agent: ChatGPT-User    # user-initiated fetch

# Anthropic
User-agent: ClaudeBot       # training
User-agent: Claude-SearchBot
User-agent: Claude-User

# Perplexity
User-agent: PerplexityBot

# Google
User-agent: Googlebot       # Search + AI Overviews + AI Mode
User-agent: Google-Extended # Gemini/Vertex training — separate from Search
```

- [ ] **Current robots.txt read line by line, every AI-related rule traced to a deliberate decision** 🔒 — `curl -s https://DOMAIN/robots.txt`. Pass: each rule has a named owner and a reason.
- [ ] **Retrieval bots allowed if AI visibility is a goal** — Pass: `OAI-SearchBot`, `Claude-SearchBot`, `PerplexityBot`, `Googlebot` not disallowed. *Blocking these while chasing GEO is a genuinely common self-own.*
- [ ] **Training-bot policy decided on its own merits** — Pass: a stated position. It's a legitimate content-rights decision, **not an SEO lever.** Blocking `GPTBot` does not remove you from ChatGPT search.
- [ ] **`Google-Extended` correctly understood** — Pass: the team knows it controls Gemini/Vertex *training* use and does **not** control AI Overviews or AI Mode. Blocking it does not remove you from Google's AI search features. Blocking `Googlebot` removes you from everything.
- [ ] **Live fetch test per user-agent** — `curl -sI -A "OAI-SearchBot" https://DOMAIN/ | head -1`, repeat for each agent above. Pass: `200`. A `403` or JS challenge means something below robots.txt is blocking regardless of what the file says.
- [ ] **Server logs checked for which AI bots are actually hitting the site and how often** 🔒 — grep the access log for each user-agent string. Pass: the bots you allow are actually arriving. Absence over weeks means a block you haven't found.
- [ ] **CDN/WAF rules checked** — Cloudflare bot-management / AI-scraper toggles, or equivalent. Pass: reviewed. **Many block AI user-agents by default without anyone deciding to** — this overrides robots.txt silently.
- [ ] **No cloaking** — serving different content to AI crawlers than to users is a confirmed black-hat tactic and a spam policy violation. Verify by diffing a Googlebot-UA fetch against a browser-UA fetch.
- [ ] **Terms/licensing position documented if content rights matter to the business** — Pass: written down somewhere other than robots.txt.

---

## 9.10 Hype versus evidence

**Be ruthless here. This is where budgets get wasted.**

| Claim | Status | What to do |
|---|---|---|
| "Publish llms.txt to get cited" | Google states it doesn't use them. Adoption elsewhere is partial and unmeasured | Optional. Costs ~nothing. **Don't pay anyone for it, don't expect results** |
| "Chunk your content for AI" | Google explicitly says unnecessary | Ignore as a mandate; write clear structured prose anyway because it helps humans |
| "Rewrite pages in AI-friendly language" | Google explicitly says unnecessary | Ignore. Write well for people |
| "FAQ schema boosts AI citations" | Unconfirmed by Google or any AI vendor. FAQ rich results dead as of May 7, 2026 | Keep FAQ *content* that answers real questions. Stop building markup for a dead SERP feature |
| "Schema markup increases citation likelihood by X%" | Third-party correlational claims; Google says structured data isn't required for AI features | Implement schema for rich results and machine comprehension. **Treat AI-citation claims as unproven** |
| "Buy brand mentions across the web" | Called out by Google as less useful than it appears; overlaps with link/reputation spam | Don't. Earn genuine coverage instead |
| "AI search killed SEO" | Google's own guidance: optimising for generative AI search *is* SEO | Do the fundamentals better, not differently |
| "This tool shows your AI ranking factors" | **No third party has access to internal Google or vendor ranking systems** | Use tools for workflow and directional tracking only |
| **"Be in the index, be clear, be corroborated, be specific"** | **Consistently supported by both Google's guidance and observed behaviour** | **This is the actual work** |

If a proposal's value rests on any of the first eight rows, say so plainly in the audit. This section is the reason clients keep money.

---

## 9.11 Measuring AI visibility

**There is no Search Console for ChatGPT or Perplexity.** Current practice is prompt-based monitoring. Build your own baseline.

**Set up (once):**

- [ ] **Write 15–25 prompts your actual buyers would use, spanning the funnel** — category-definition ("what is X"), solution-seeking ("how do I solve Y"), comparison ("X vs Y", "alternatives to Z"), recommendation ("best X for [my customer type]"), branded ("is [my company] any good", "what does [my company] do"). Pass: written down, fixed, and not edited between runs — a changing prompt set produces uncomparable data.
- [ ] **Run across ChatGPT, Perplexity, Gemini / AI Overviews, and Claude on a fixed cadence**
- [ ] **Log in a sheet: date, platform, prompt, cited yes/no, position in the answer, which competitors appeared, which domains were cited**
- [ ] **Track *which sources* the engines cite, not just whether you appeared** — that's your roadmap for `L9-ai-search.md` §9.7
- [ ] **Run branded prompts specifically to catch misinformation about you**
- [ ] **Baseline taken and dated before making any changes** 🔒

**Ongoing:**

- [ ] **Re-run the full set monthly, same prompts, logged with dates** 🔒
- [ ] **GSC Generative AI performance report reviewed monthly** 🔒 — **the only first-party data source for AI Overviews / AI Mode performance**
- [ ] **GSC impressions vs. clicks trend watched for the zero-click pattern** — impressions flat/up with clicks down = answers satisfied without a visit. A very different diagnosis from falling impressions, and it demands a different response
- [ ] **Referral traffic from AI domains segmented in analytics** 🔒 — `chat.openai.com`, `chatgpt.com`, `perplexity.ai`, `gemini.google.com`, `claude.ai`. These visitors typically convert differently, often better, because they arrive pre-qualified
- [ ] **Direct/branded search volume tracked as a proxy** for AI-driven awareness that never shows as a referral
- [ ] **Server logs monitored for AI crawler activity as a leading indicator** 🔒
- [ ] **Paid AI-visibility tools evaluated only *after* manual baselines exist**, and only if they save real time. **Be wary of any tool claiming access to "internal" Google metrics. None have it.**

**Expect volatility.** Citation sets turn over substantially month to month. Judge trends over quarters, not weeks, and don't rebuild your strategy off a single bad month.

---

## 9.12 Risk management

New surface, new failure modes.

- [ ] **Branded prompts run regularly to catch AI systems stating wrong things about you** 🔒 — pricing, features, availability, ownership. Pass: a run in the last 30 days with errors logged.
- [ ] **Outdated content on your own site corrected** — that's often the source of the wrong answer.
- [ ] **Stale third-party listings and old press releases updated or corrected where possible.**
- [ ] **Discontinued products clearly marked as discontinued rather than silently deleted** — a deleted page leaves the old claim alive in every other source; a page saying "discontinued in 2025" corrects it.
- [ ] **Dates visible and accurate** — undated content gets treated as unreliable by recency-weighted engines.
- [ ] **Business model reviewed for zero-click exposure** — if 100% of revenue depends on ad impressions from informational traffic, that model is under structural pressure **regardless of what you do.** Say this out loud in the audit; it's a business finding, not an SEO one.
- [ ] **Conversion paths reviewed for a lower-volume, higher-intent traffic mix.**
- [ ] **Content that's purely a summary of other people's work identified** — the category most exposed to being replaced by the answer itself.

**Questions:** What do AI assistants say about my company today — and is any of it wrong? If half my informational traffic disappeared, which pages would I still need? Am I building assets AI can't replicate, or content AI can replace? Where does the *money* come from — and does that path run through a clickable link?

---

## 9.13 Content extractability — the short list

Consolidated from the checks above, for when you only have time for one pass:

- [ ] Clear headings, direct answers up front, self-contained paragraphs, comparison tables, definition blocks
- [ ] Video content has on-page text transcripts — AI systems can't watch video, they read transcripts
- [ ] Facts are attributable, dated, and specific — retrieval systems favour content they can verify
- [ ] `llms.txt` — optional. Near-zero cost, unproven benefit, ignored by Google. Ship it if you like; don't build strategy on it
- [ ] Agentic access considered — browser agents parse the DOM, screenshots, and the accessibility tree, so semantic HTML and accessibility work double duty here. See `../ada/` and `web.dev/articles/ai-agent-site-ux`

---

Next: `L10-measurement.md`. The AI-visibility question set lives in `question-list.md`; the AI-specific audit prompts live in `tools.md`.
