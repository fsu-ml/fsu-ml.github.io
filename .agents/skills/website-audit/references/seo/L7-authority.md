# Layer 7 — Authority & off-site signals

Covers: backlink profile, link spam boundaries, linkable assets, brand SERP, reputation.
Load: `seo.priority: full`, or whenever the site competes commercially for search traffic. Skip for internal tools, portfolios, lab sites.
Depends on: `L5-content.md` — nobody links to a site with nothing worth linking to.

**You cannot fully control this layer, which is exactly why it carries weight.** Almost every check here is 🔒 — it needs a backlink tool, Search Console, or manual searching. Do not estimate a backlink profile from the outside.

---

## 7.1 Audit what exists

- [ ] **Current backlink profile audited: volume, quality, relevance, anchor text distribution** 🔒 — Ahrefs / Semrush / Moz, or GSC → Links → External links (free, less complete). Pass: you can state referring-domain count, roughly how many are topically relevant, and whether anchor text looks natural. Fail signature: a large share of exact-match commercial anchors, which is the fingerprint of bought links.
- [ ] **Toxic/spammy links identified; disavow used sparingly and only for a genuine problem** 🔒 — Pass: either no action needed (the normal outcome — Google ignores most junk automatically), or a disavow file addressing links you actually caused. **Reflexive disavowing is more likely to cause harm than the links were.** Reserve it for a manual action or a known bought-link history.
- [ ] **No purchased links, link exchanges, or paid guest posts passing ranking credit** — ask directly; check whether any agency invoices mention placements. Pass: none, or historical ones cleaned up. All three are explicitly **link spam** under Google's policies. Paid placements are fine if the link carries `rel="sponsored"` or `rel="nofollow"`.
- [ ] **Brand searchable by name — does the brand SERP look like a credible business?** 🔒 — search the brand name incognito. Pass: page one is your site, your profiles, and neutral-to-positive third-party coverage. Fail: competitor ads dominating, a dead LinkedIn, an old complaint thread at position 3, or nothing at all.

---

## 7.2 Build what earns links

- [ ] **Genuinely linkable assets exist** — original research, free tools, definitive guides, datasets. Pass: at least one asset you can name where the reason to link is obvious. Verify by asking a journalist-shaped question: would someone cite this to support a point they were already making?
- [ ] **Unlinked brand mentions found and converted where possible** — search `"BRAND" -site:DOMAIN`, or use a mention-monitoring tool. Pass: a worked list; outreach sent for the ones worth asking about.
- [ ] **Digital PR, expert commentary, podcast and industry participation happening** — Pass: a named cadence and a named person doing it. This is the only reliable link-earning motion for most businesses.
- [ ] **Business listed in relevant, legitimate industry directories** — Pass: present in the directories that a buyer in this category actually uses. Not link-farm directories; the test is whether a human would use it to find a supplier.
- [ ] **Reviews and third-party reputation actively managed** — Google, Trustpilot, G2, Capterra, industry-specific equivalents. Pass: profiles claimed, reviews current, negative reviews responded to. This overlaps directly with corroboration in `L9-ai-search.md` §12.5 — the same third-party presence that supports rankings is what AI answer engines synthesise across.
- [ ] **No "inauthentic mentions" campaigns** — no bought reviews, no astroturfed forum threads, no paid "brand mention" packages. Pass: none. **Google's 2026 AI guidance calls this out specifically as less useful than it appears**, and it overlaps with link and reputation spam policy. The 2026 version of buying links.

---

## 7.3 The boundary, stated plainly

| Tactic | Status |
|---|---|
| Earning coverage through genuinely newsworthy work, data, or expert commentary | The actual work |
| Paid placements marked `rel="sponsored"` / `rel="nofollow"` | Legitimate advertising |
| Buying or exchanging links for ranking credit | **Link spam.** Explicit policy violation |
| Paid guest posts passing credit | **Link spam.** Same policy |
| Buying reviews, astroturfing forums, "brand mention" packages | Spam risk **and** per Google's 2026 guidance, less effective than marketed |

---

## Questions to ask

1. Why would anyone link to this site? Name a specific reason.
2. What do we have that a journalist, blogger, or industry newsletter would find genuinely useful?
3. If I search my brand name, does page one look trustworthy? 🔒
4. Are we participating in our industry's conversation, or publishing into a void?
5. Have we ever paid for links? Be honest — it determines the recovery plan. 🔒

---

Next: `L8-local.md` if `seo.local_business: true`, otherwise `L9-ai-search.md`.
