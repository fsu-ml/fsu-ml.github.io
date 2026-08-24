# Layer 0 — Business context

Covers: the questions to answer before any technical work, and which answers get recorded in the audit profile.
Load: always, first within the SEO track. Skipping this is why most audits produce generic advice.
Depends on: **`../scoping.md`, which produces the audit profile.** That is a hard gate for the whole audit — this file does not create the profile, it *augments* an existing one.
Output: the SEO-specific fields written back into the profile — `seo.priority`, `seo.local_business`, `seo.ecommerce`, `seo.ai_visibility`, `seo.data_access`, `site.url_count`, `site.multilingual`, plus the claimed `stack.rendering` and the `audience.success_metric` / `engagement.constraints` entries this file's questions sharpen. `00-map.md`'s load-decision table reads them.

---

## Why this comes first

Most SEO failure is strategic, not technical. A flawless crawl of a site targeting the wrong queries for the wrong audience produces a flawless report and zero result. Answer these before touching robots.txt.

These answers are also what makes the difference between an audit and a checklist run: severity is meaningless without knowing what the site is for. A broken `LocalBusiness` schema is P2 for a dentist and N/A for a portfolio. (`triage.md` §1 owns the ladder; read the priority off that table, not off this sentence.)

---

## The intake checklist

Each item states how to verify it is genuinely answered — not just gestured at.

- [ ] **Primary business goal written down** — one of: leads / sales / signups / ad revenue / credibility / none. Pass: a single word, not "growth" or "visibility".
- [ ] **One-sentence definition of what the site sells or offers** — Pass: a sentence a stranger could repeat back accurately. If it takes a paragraph, the positioning problem is upstream of SEO.
- [ ] **Target geography defined** — global / national / regional / a radius around a physical location. Pass: an explicit answer, because "radius around a location" is what triggers `L8-local.md`.
- [ ] **Target audience defined, including their level of expertise** — Pass: names a role or situation plus a knowledge level (e.g. "in-house counsel at 50–500 person firms, legally expert, not technical").
- [ ] **Top 3 competitors *in search results* identified** — Pass: three domains, found by searching 2–3 target queries and reading who actually ranks. These are frequently *not* the business competitors the client names from memory.
- [ ] **Definition of "success" with a number and a deadline** — Pass: e.g. "40 qualified demo requests/month from organic by Q2". Fail: "more traffic".
- [ ] **Realistic budget and time commitment stated** — Pass: hours/month and £/$ per month. SEO on 2 hrs/month is a different plan than 20; the audit's recommendations must fit the actual capacity.
- [ ] **Known constraints listed** — can't change CMS, no dev resources, legal review required, no ability to publish, procurement lead times. Pass: an explicit list, including "none known".
- [ ] **Honest timeline expectation stated and agreed** — Pass: the client has heard and accepted that meaningful movement takes 3–6 months minimum, and competitive terms 12+ months. Recording this prevents the audit being judged against a fantasy.

---

## The seven questions

Ask verbatim. The answers shape which layers matter and how findings are prioritised.

| # | Question | What the answer changes |
|---|---|---|
| 1 | If someone finds this site and does exactly one thing, what should that thing be? | The conversion definition in `L10-measurement.md`; the "clear next action" check in `L4-onpage.md` |
| 2 | What would someone type — or ask an AI assistant — at the exact moment they need what we offer? | The query set for `L5-content.md` research and the prompt set for `L9-ai-search.md` |
| 3 | Do we sell to people who search for a *solution*, or people who already know our *category*? | Changes everything about content strategy — solution-aware vs category-aware demand are different keyword universes |
| 4 | Are we trying to be found by humans, by AI assistants, or both — and does the buying journey actually run through search at all? | Sets `seo.ai_visibility`; may reveal that SEO is the wrong channel entirely, which is a legitimate finding |
| 5 | What do we know that our competitors don't? | The only durable SEO asset. Feeds the non-commodity content bar in `L5-content.md` |
| 6 | Is there any reason a search engine should prefer us over an established incumbent? | Realistic difficulty assessment. If there is no answer, the plan is long-tail and authority-building, not head terms |
| 7 | What's the honest timeline expectation? | Whether the engagement is set up to succeed or to be judged a failure at week 6 |

---

## Profile fields to record

**The profile already exists.** `../scoping.md` writes it from `../../templates/audit-profile.yaml` before any track starts. This section *augments* that profile with the SEO-specific fields below; it does not create it. Field names and value sets are canonical in the template — do not invent new keys here.

`00-map.md` reads these to decide which layer files load.

| Field | Values | Sourced from |
|---|---|---|
| `seo.priority` | `none` \| `basic` \| `full` | Business goal + budget/time. `none` = don't ship broken; `basic` = be findable and clear; `full` = compete |
| `seo.local_business` | `true` \| `false` | Target geography. `true` only if there is a physical location or a defined service area |
| `seo.ecommerce` | `true` \| `false` | What the site sells. `true` if transactions complete on-site |
| `seo.ai_visibility` | bool (`true` \| `false`) | Question 4. Also set `true` if the client is being sold GEO/AEO services — they need `L9-ai-search.md` §hype-vs-evidence. If genuinely undetermined, leave `false` and say so in the report; the schema has no `unknown` for this field |
| `stack.rendering` | `ssg` \| `ssr` \| `csr` \| `hybrid` \| `unknown` | **Claimed** here, **verified** in `L1-foundations.md`. Never trust the claim. A prerendered/"static" site is `ssg` — there is no separate `static` value |
| `site.url_count` | integer or `unknown` | Sitemap count or crawl. Drives whether `L3-architecture.md` and crawl-budget checks apply |
| `site.multilingual` | `true` \| `false` | Target geography. `true` triggers the hreflang checks in `L3-architecture.md` |
| `seo.data_access` | list of `search-console` \| `bing-webmaster` \| `analytics` \| `crux` \| `server-logs` \| `backlink-tool` \| `none` | Which instruments the auditor actually has. Determines how many 🔒 items resolve to `CAN'T VERIFY` |

Two fields these questions sharpen live **outside** the `seo.` block — write them where the schema puts them, not under `seo.`:

| Field | Values | Sourced from |
|---|---|---|
| `engagement.out_of_scope` | list | The known-constraints checklist above — uneditable CMS, no dev resource, no publishing ability, procurement lead times. Recommendations that violate these are wasted output. (The schema has no dedicated `constraints` key; constraints that make work impossible are recorded here, and `engagement.fix_budget` carries the capacity answer) |
| `audience.success_metric` | free text + date | The number and deadline. Every finding's severity is argued against this |

If a field cannot be answered, record `unknown` and state it in the report. An `unknown` in `stack.rendering` means the rendering deep dive in `L1-foundations.md` is **mandatory**, not optional.

---

Next: `L1-foundations.md`. Nothing technical happens before the profile exists.
