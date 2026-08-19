# Question list — the interview instrument

Covers: the master question list (34 questions across all layers) and the AI visibility question set (30 questions).
Load: interview mode; when the client is available to answer; when the audit must cover strategy, not just crawl output.
Depends on: nothing. Answering these *before* a crawl produces far better output than a raw crawl, because most SEO failure is strategic, not technical.

🔒 = requires real data only the site owner can pull. An auditor cannot infer these from outside. Mark unanswered items explicitly rather than guessing.

---

## Part 1 — The master question list

### Strategy
1. What does this site sell, to whom, and where?
2. What is the one action a visitor should take?
3. Who are my three search competitors?
4. What's my success metric, and by when?
5. What do I know that my competitors don't?

→ Feeds `business-context.md`. Do not proceed to technical work with these unanswered.

### Discoverability
6. Is the site verified in Google Search Console and Bing Webmaster Tools? 🔒
7. How many pages are indexed vs. how many exist — and what explains the gap? 🔒
8. Is anything blocked in robots.txt that shouldn't be?
9. Does the content render without JavaScript?
10. Are there orphan pages or crawl traps?

→ `L1-foundations.md`, `L3-architecture.md`

### Performance
11. What are my 75th-percentile *mobile* LCP, INP, and CLS from field data? 🔒
12. What's the slowest template on the site?
13. Which third-party scripts could I remove today?

→ `L2-technical-performance.md`, `../performance.md`

### Structure
14. Is every key page within three clicks of home?
15. Do any two pages target the same query?
16. Do my internal links reflect commercial priority?

→ `L3-architecture.md`

### Content
17. Does each page answer its query in the first two sentences?
18. What percentage of my content is commodity content?
19. Who is the named human behind each page?
20. Which pages should be deleted outright?

→ `L4-onpage.md`, `L5-content.md`

### Machine comprehension
21. Is `Organization` and `Article` schema present and accurate?
22. Does my schema match my visible content exactly?
23. Am I maintaining markup for a rich result that no longer exists?

→ `L6-structured-data.md`

### Authority
24. Why would anyone link to this site? Name a specific reason.
25. What does my brand SERP look like? 🔒
26. Have I ever bought links? 🔒

→ `L7-authority.md`

### Local *(if `seo.local_business: true`)*
27. Is my NAP identical everywhere it appears?
28. When did I last post to GBP or earn a review?

→ `L8-local.md`

### AI search *(expanded set in Part 2)*
29. Which AI crawlers can access my site — by decision, or by accident?
30. Am I in Bing's index at all? 🔒
31. Who gets cited when I ask AI assistants my customers' questions? 🔒

→ `L9-ai-search.md`

### Measurement
32. What's my baseline, dated? 🔒
33. Which queries rank 5–15 with low CTR? 🔒
34. What's my maintenance cadence, and who owns it?

→ `L10-measurement.md`

---

## Part 2 — The AI visibility question set

Run these deliberately. **Most take under an hour and produce a clearer picture than any tool.**

### Reality check — do this first
1. Ask ChatGPT, Perplexity, Gemini, and Claude: *"What is [my company]?"* — is the answer right, wrong, or absent?
2. Ask each: *"What's the best [my category] for [my ideal customer]?"* — am I named? Who is?
3. For every answer above, list the **domains cited**. Am I present on any of them?
4. Ask: *"What are the alternatives to [my main competitor]?"* — am I in that list?
5. Ask: *"How much does [my product] cost?"* — is the answer correct, or invented?

### Access
6. Does my robots.txt allow `OAI-SearchBot`, `PerplexityBot`, `Claude-SearchBot`, and `Googlebot`?
7. Was that a decision someone made, or a default nobody reviewed?
8. Is my CDN, WAF, or bot-protection layer blocking AI user-agents without my knowledge? 🔒
9. Do my server logs show AI crawlers actually reaching the site? 🔒
10. Am I indexed in Bing? 🔒
11. Is my site enabled for Search generative AI features in Search Console? 🔒

### Extractability
12. Take a random 150-word passage from a key page. Out of context, does it answer a question completely?
13. Do my H2s read like questions a person would ask?
14. Does any page state a price, a number, or a specification a model could quote?
15. Do my videos have on-page transcripts?
16. Do I have comparison tables, or only prose?

### Trust and entity
17. Is my company described identically on my site, LinkedIn, and every directory?
18. Do my authors exist as real, verifiable people outside my own domain?
19. Does my `Organization` schema link to all my official profiles via `sameAs`?
20. Is there anything on the internet that authoritatively describes what we do, that we control?

### Corroboration
21. Which third-party sources dominate AI answers in my category?
22. What do those sources currently say about us — anything?
23. Which of them could we legitimately earn a place on in 90 days?
24. Are there "best of" roundups in my category that omit us?

### Substance
25. Do I publish any number, dataset, or finding that exists nowhere else?
26. What could I write that a model genuinely couldn't generate without me?
27. Am I honest anywhere about who my product *isn't* for?

### Risk
28. Is any AI assistant currently stating something false about my business? 🔒
29. If half my informational traffic vanished, which pages would still earn their keep?
30. Does my revenue depend on clicks that AI answers might absorb?

---

## How to run the interview

| Mode | Method |
|---|---|
| **Interview first, audit second** | The client answers Part 1 themselves; the auditor works from real constraints. Produces far better output than a raw crawl |
| **Audit fills what it can** | The auditor answers what's externally verifiable, and returns the rest flagged as owner-only. Never guess a 🔒 item |
| **AI baseline** | Part 2 §Reality check is the day-1 AI baseline in `triage.md` §3. Log the results with a date — the whole point is comparability next month |

Answers to Part 1 §Strategy get recorded in the audit profile — see `business-context.md` for the field names.
