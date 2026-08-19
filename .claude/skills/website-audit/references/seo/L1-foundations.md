# Layer 1 — Foundations: can it be found, rendered, and indexed at all?

Covers: domain/hosting, crawlability, indexability, and rendering verification.
Load: always, including `seo.priority: none`. This is a gate, not an optimisation layer.
Depends on: `business-context.md` for the profile. Nothing below this line matters if this layer fails.
This is Google's "Search Essentials → Technical Requirements": Googlebot isn't blocked, the page returns a working HTTP status, the content is indexable and non-violating.

**Stop rule:** a P0 finding here halts the audit. Report it, get it fixed, re-crawl, then continue. Do not deliver a content critique for a site serving `Disallow: /`.

🔒 = requires first-party data (Search Console, Bing WMT, server logs). Mark `CAN'T VERIFY` if unavailable; do not infer.

---

## 1.1 Domain & hosting

- [ ] **One canonical hostname enforced** — `curl -sI https://www.DOMAIN/ | head -1` and `curl -sI https://DOMAIN/ | head -1`. Pass: exactly one returns `200`; the other returns `301` pointing at it. Fail: both `200` (duplicate site) or a redirect chain between them.
- [ ] **HTTPS sitewide with a valid, unexpired certificate** — `curl -sI https://DOMAIN/` succeeds without `-k`; or `echo | openssl s_client -connect DOMAIN:443 -servername DOMAIN 2>/dev/null | openssl x509 -noout -dates`. Pass: no TLS error, `notAfter` in the future with comfortable margin.
- [ ] **All HTTP requests 301 to HTTPS** — `curl -sI http://DOMAIN/some/deep/path | head -3`. Pass: `301` (or `308`) with a `Location:` on `https://` preserving the path. Fail: redirects everything to the homepage — that's a soft-404 pattern.
- [ ] **No mixed content** — DevTools Console on an HTTPS page, or `curl -s https://DOMAIN/ | grep -oE '(src|href)="http://[^"]*"'`. Pass: no output.
- [ ] **Domain not on a spam blocklist; no unresolved manual actions** 🔒 — Search Console → Security & Manual Actions. Pass: both panels show "No issues detected".
- [ ] **If the domain was bought used: prior history checked** — Wayback Machine (`web.archive.org/web/*/DOMAIN`), plus a `site:DOMAIN` search for unrelated content. Pass: no evidence of a prior unrelated site being repurposed. Expired-domain abuse is an explicit Google spam policy violation.
- [ ] **Hosting geographically sensible and reliable; uptime monitored** — check TTFB from the target market and confirm an uptime monitor exists with alerting. Pass: a monitor is named and someone receives its alerts.
- [ ] **No bot protection / WAF / Cloudflare rule silently blocking legitimate crawlers** 🔒 — `curl -sI -A "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)" https://DOMAIN/ | head -1`, then repeat with `OAI-SearchBot`, `PerplexityBot`, `ClaudeBot`, `bingbot`. Pass: `200` for every UA you intend to allow. Fail: `403`, `429`, or a JS challenge page. Cross-check against server logs and the CDN's managed bot rules — **many CDNs block AI user-agents by default without anyone deciding to.**

---

## 1.2 Crawlability

- [ ] **`robots.txt` exists and returns 200** — `curl -sI https://DOMAIN/robots.txt | head -1`. Pass: `200` with `content-type: text/plain`. A `404` is survivable (everything is allowed); a `500` is not — crawlers may back off entirely.
- [ ] **No blanket `Disallow: /`** — `curl -s https://DOMAIN/robots.txt`. Pass: no `Disallow: /` on its own line under `User-agent: *` or `User-agent: Googlebot`. **This is the single most common catastrophic error, and it is almost always a staging config that shipped.** P0.
- [ ] **CSS and JS not blocked** — check robots.txt for `Disallow: /assets/`, `/static/`, `/_next/`, `*.js`, `*.css`. Pass: none present. Confirm in Search Console → URL Inspection → Test Live URL → "More info" → page resources: no blocked resources 🔒. Google needs these to render.
- [ ] **Sitemap declared in robots.txt** — `curl -s https://DOMAIN/robots.txt | grep -i sitemap`. Pass: at least one absolute `Sitemap:` URL that itself returns 200.
- [ ] **Crawl traps identified and handled** — crawl with Screaming Frog and sort by URL count per directory; look for infinite calendars (`?date=`, `/2031/07/`), faceted-navigation parameter combinations, session IDs, endless pagination. Pass: no directory generating unbounded URLs. Fail signature: crawl doesn't terminate, or one path holds more URLs than the site has content.
- [ ] **Crawl budget reviewed if the site is large (roughly >10k URLs) or updated very frequently** 🔒 — Search Console → Settings → Crawl stats. Pass: crawl requests roughly track content volume; average response time stable; no rising 5xx share.
- [ ] **Server handles crawl load without 5xx or rate-limiting crawlers** 🔒 — GSC Crawl stats "By response" breakdown, plus server logs. Pass: 5xx is a negligible fraction and not correlated with crawl spikes.
- [ ] **Login walls / paywalls handled deliberately** — either robots-blocked, or implemented with flexible sampling / paywalled-content structured data. Pass: the choice is documented. Fail: content silently invisible to crawlers with nobody aware.

---

## 1.3 Indexability

- [ ] **Site verified in Google Search Console** 🔒 — non-negotiable. This is ground truth. Pass: property verified and someone on the team has access.
- [ ] **Site verified in Bing Webmaster Tools** 🔒 — matters more than people think; **ChatGPT search leans on Bing's index.** Pass: property verified.
- [ ] **No accidental `noindex` in meta robots** — `curl -s URL | grep -iE '<meta[^>]+name=["'"'"']robots'`. Pass: absent, or present without `noindex`. Run against the homepage plus a page of each template.
- [ ] **No accidental `noindex` in `X-Robots-Tag` HTTP headers** — `curl -sI URL | grep -i x-robots-tag`. Pass: no output, or no `noindex`. **Header-level noindex is invisible in view-source and is a classic missed cause of a de-indexed site.**
- [ ] **Important pages appear as "Indexed" in GSC's Pages report** 🔒 — Pages → Indexed. Spot-check by URL Inspection on your top 10 commercial URLs. Pass: "URL is on Google".
- [ ] **"Discovered – currently not indexed" and "Crawled – currently not indexed" counts reviewed** 🔒 — GSC Pages → the exclusion reasons. Pass: counts understood and explained. *These are usually quality or duplication signals, not bugs* — treat them as a content finding for `L5-content.md`, not a technical one.
- [ ] **Canonical tags present and self-referencing on canonical pages** — `curl -s URL | grep -i 'rel="canonical"'`. Pass: the `href` equals the page's own absolute URL (correct protocol, host, trailing slash).
- [ ] **No canonical pointing to the homepage from every page** — run the above across a sample of 10 URLs. Pass: 10 distinct canonicals. Fail: all 10 point at `/` — a common CMS/theme misconfiguration that de-indexes the whole site.
- [ ] **XML sitemap contains only canonical, 200-status, indexable URLs** — fetch the sitemap, then batch-check statuses: `curl -s SITEMAP_URL | grep -oP '(?<=<loc>)[^<]+' | while read u; do echo "$(curl -so /dev/null -w '%{http_code}' "$u") $u"; done`. Pass: every line `200`. Any `301`, `404`, or `noindex` URL in a sitemap is a wasted crawl signal.
- [ ] **Sitemap submitted** 🔒 — GSC → Sitemaps. Pass: status "Success", discovered-URL count close to the file's count.
- [ ] **Sitemap under 50,000 URLs and 50MB uncompressed; sitemap index used if larger** — count `<loc>` entries and check file size. Pass: within both limits, or a valid `<sitemapindex>` splitting them.
- [ ] **`lastmod` dates accurate, not bulk-updated on every deploy** — compare `lastmod` values across the sitemap. Fail signature: every URL sharing today's date. That teaches crawlers to ignore the field.
- [ ] **Staging/dev environments noindexed *and* password-protected** — try `staging.DOMAIN`, `dev.DOMAIN`, `DOMAIN.netlify.app`, and a `site:` search for the staging host. Pass: HTTP auth or a 403/404. **`noindex` alone leaks** — staging URLs get linked and shared, and a noindexed page still gets crawled and can be discovered.
- [ ] **Parameter/filter URLs handled via canonicals, robots rules, or noindex** — crawl and count URLs containing `?`. Pass: parameter variants canonicalise to the clean URL. Fail: they multiply unchecked (see also faceted navigation in `L3-architecture.md`).

---

## 1.4 Rendering — the critical section

**This is where framework sites fail, and it is the check most audits get wrong by reasoning from the framework name.**

A framework name tells you what the project *can* do, not what this deployment *does*. Next.js can be fully static, fully server-rendered, fully client-rendered, or all three on different routes in the same app. React can be pre-rendered. A "static site generator" can ship a client-only route for search or filtering. Astro islands hydrate. Vue in SPA mode ships an empty `<div id="app">`.

**Rule: test the rendered output. Never infer rendering strategy from `package.json`.**

### 1.4.1 Detect what the site actually does

Run all four. They disagree in informative ways.

| # | Test | Command / method | What the result means |
|---|---|---|---|
| 1 | **Raw HTML content test** | `curl -s URL \| sed 's/<[^>]*>//g' \| tr -s '[:space:]' ' ' \| head -c 2000` | The main body copy is present → server-rendered or pre-rendered. Only nav/footer boilerplate, or nothing but an empty root div → client-rendered. |
| 2 | **Raw HTML byte size** | `curl -s URL \| wc -c` vs the DOM size in DevTools | Raw ≈ rendered → SSR/SSG. Raw a small fraction of rendered → CSR. |
| 3 | **JS disabled** | DevTools → Settings → Debugger → Disable JavaScript, then hard-reload | Content still readable → safe. Blank page or spinner → the content does not exist for any non-executing consumer. |
| 4 | **Google's own renderer** 🔒 | GSC → URL Inspection → Test Live URL → View Tested Page → **HTML** tab | The definitive answer for Google. The rendered HTML must contain the real content. This is the only test that reflects Googlebot's actual JS execution. |

Record the outcome as `stack.rendering` in the profile, **overwriting whatever was claimed.** Per-route if routes differ — a marketing homepage that is static and a `/products` route that is CSR is a single-site, two-answer situation and only the CSR route needs the deep dive.

### 1.4.2 Rendering checks

- [ ] **Core content present in initial HTML, or reliably server-rendered / pre-rendered** — tests 1–4 above. Pass: test 1 or test 4 shows the real body copy. If only test 4 passes, note the risk: non-Google consumers (Bing, ChatGPT's crawlers, Perplexity, in-chat fetchers, link previews) execute far less JavaScript than Googlebot does. CSR that "works for Google" can still be invisible in AI answers.
- [ ] **Internal links are real `<a href>` elements** — `curl -s URL | grep -oE '<a [^>]*href="[^"]*"' | head -50`. Pass: real hrefs to real URLs. Fail: `<div onclick>`, `<button>` with a router binding, or `href="#"` with JS navigation. Crawlers follow `href`; they do not click.
- [ ] **Lazy-loaded content loads without user interaction** — load the page, do not scroll, and inspect the DOM; or compare `curl` output to a headless fetch. Pass: content present in the DOM at load. **Crawlers don't scroll and don't click.**
- [ ] **Infinite scroll has a paginated, crawlable fallback** — look for `/page/2/` or `?page=2` URLs reachable via `<a href>`. Pass: every item reachable through a link chain without JS interaction.
- [ ] **No client-side-only redirects for anything important** — `curl -sI URL | head -1` for URLs you believe redirect. Pass: real `301`/`302` at the HTTP layer. Fail: `200` plus a `window.location` or router redirect in the body.
- [ ] **Error pages return real status codes** — `curl -sI https://DOMAIN/definitely-not-real-xyz | head -1`. Pass: `404` (or `410` for intentionally removed). Fail: `200` with a "not found" message = soft 404. *Google has tightened rendering behaviour around non-200 responses — content injected client-side onto error pages may never be seen at all.*
- [ ] **`<title>`, meta description, canonical, and JSON-LD exist in the raw HTML, not injected by JS** — `curl -s URL | grep -iE '<title|canonical|application/ld\+json'`. Pass: present in the raw response. Head tags injected client-side are the most common partial-CSR failure: the page renders for users, the metadata never reaches non-JS consumers.

### 1.4.3 If rendering fails

Ranked by durability, not by ease:

| Fix | When it fits | Caveat |
|---|---|---|
| Static generation (SSG) at build time | Content changes on a deploy cadence | Best outcome. Raw HTML is the real HTML for every consumer. |
| Server-side rendering (SSR) | Content is personalised or changes per request | Adds server cost and a TTFB risk — see `../performance.md` |
| Pre-rendering / hybrid per-route | Only some routes are dynamic | Verify per route; the audit must test each route type separately |
| Dynamic rendering (serve pre-rendered HTML to bots) | Legacy stacks with no migration path | Google treats this as a workaround, not a recommendation. **Serving materially *different* content to crawlers is cloaking and a spam policy violation** — the pre-rendered output must match what users see. |

---

## Questions to ask

1. Can I load a key page with JS disabled and still read the content? If not, what's the rendering strategy — verified, not assumed?
2. How many pages does GSC say are indexed, versus how many pages do I think I have? Explain the gap. 🔒
3. Is anything in robots.txt there for a reason nobody remembers?
4. If I lost Search Console access tomorrow, would I notice a traffic problem before revenue dropped?
5. Which URLs are in the sitemap but not indexed — and is that a bug or a quality signal? 🔒

---

Next: `L2-technical-performance.md`. Related: `L9-ai-search.md` for AI crawler governance in robots.txt; `../../scripts/audit_seo.py` automates most of §1.1–1.3.
