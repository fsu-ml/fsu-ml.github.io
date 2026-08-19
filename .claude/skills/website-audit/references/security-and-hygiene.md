# Security headers & hygiene — cheap evidence of a well-built site

**Scope: quality signalling, not a penetration test.** Nothing here proves an application is secure. What it does prove is that somebody was paying attention: headers are a handful of lines of config, they cost nothing at runtime, and their absence is the fastest way to tell that a site was shipped rather than engineered. A site that *is* clean has these; a site that merely *looks* clean does not.

Load: every audit — this is a 60-second baseline — and in full for any security or trust question.
Companions: `code-quality.md` (build/deploy hygiene, source maps, exposed `.env`/`.git`, console/network hygiene, and §10 for theming the consent banner), `performance.md` §3.8 (caching and `no-store` vs bfcache — the caching header decisions interact) and §4.4 (the CMP as a performance risk), `ada/wcag22-new.md` (2.4.11, the banner-obscures-focus criterion), `site-categories.md` (the dark-pattern measurement recipes §6.2 reuses), `seo/L2-technical-performance.md` (HTTPS/redirect hygiene as a crawl signal).
Scripts: `../scripts/check_headers.py`.

**Reliability key.** `[P]` primary · `[S]` secondary · `[?]` contested.

**The one interpretive rule for this whole file:** grade the **policy**, not the letter. **An A+ on `securityheaders.com` with `unsafe-inline` in the CSP is worse than a B with a strict nonce policy.** `[S]` Scanners score presence; you score effect.

---

## 1. The 60-second baseline

```bash
curl -sI https://example.com
```

Read it against this table. Every row has a "why it is also a quality signal" column because that is how the finding gets funded.

| Header | Recommended value | Why (security) | Why (quality signal) |
|---|---|---|---|
| `Strict-Transport-Security` | `max-age=31536000; includeSubDomains; preload` | HTTPS enforcement; kills SSL-strip | Somebody owns the TLS config |
| `Content-Security-Policy` | nonce-based + `strict-dynamic`; `object-src 'none'; base-uri 'none'` | XSS mitigation | Requires knowing every script on the page — the single strongest signal in this table |
| `X-Content-Type-Options` | `nosniff` | MIME sniffing | One line; its absence means nobody ever looked |
| `Referrer-Policy` | `strict-origin-when-cross-origin` | Leaky URLs (tokens/IDs in paths) | Privacy posture |
| `Permissions-Policy` | `camera=(), microphone=(), geolocation=(), payment=(), usb=()` — **deny by default** | Feature abuse, especially via iframes | Deliberate third-party containment |
| `X-Frame-Options` / CSP `frame-ancestors` | `frame-ancestors 'none'` (**CSP supersedes XFO**) | Clickjacking | |
| `Cross-Origin-Opener-Policy` | `same-origin` | Cross-window attacks; also enables `crossOriginIsolated` | Enables `SharedArrayBuffer`-class features |
| `Cross-Origin-Resource-Policy` | `same-origin` (or `cross-origin` for public CDN assets) | Side-channel / resource theft | |

**Also check for what should *not* be there:**
```bash
curl -sI https://example.com | grep -iE 'server:|x-powered-by|x-aspnet-version|x-generator'
```
A version-disclosing `Server` or any `X-Powered-By` is a finding — see `code-quality.md` §9.4.

---

## 2. HTTPS and HSTS

| Check | Command | Pass |
|---|---|---|
| HTTP redirects to HTTPS with a **301** | `curl -sIL http://example.com \| grep -iE '^HTTP\|^location'` | First response `301`, `Location:` is `https://`, then `200`. A `302` here is a finding |
| `www` / apex canonicalisation | Same command against both hosts | One canonical host; at most **one hop** |
| HSTS present and long | `curl -sI https://example.com \| grep -i strict-transport` | `max-age` ≥ **31536000** |
| `includeSubDomains` | same | Present — **verify every subdomain is HTTPS-capable first**, this is irreversible for the `max-age` duration |
| `preload` | same, plus check the [HSTS preload list](https://hstspreload.org/) | Present **only** if the team has consciously accepted preload-list semantics. Do not recommend it casually |
| Certificate chain, protocol versions, cipher suites | `testssl.sh https://example.com` (or `nmap --script ssl-enum-ciphers -p443 example.com`) | TLS 1.2 + 1.3 only; no TLS 1.0/1.1; complete chain; > 30 days to expiry |
| **Mixed content** | DevTools console on every audited template; `curl -s URL \| grep -oE 'src="http://[^"]*"\|href="http://[^"]*"'` | **Zero.** Any `http://` subresource on an HTTPS page is blocked or upgraded, and is always a defect |

**HSTS caveat to write into the report:** HSTS applies only **after** the first successful HTTPS visit unless the domain is on the preload list. It is not a substitute for the redirect.

---

## 3. Content-Security-Policy — the one that takes real work

### 3.1 Why `unsafe-inline` defeats the policy

CSP's entire XSS value comes from the browser refusing to execute script the server did not sanction. `script-src 'unsafe-inline'` tells the browser to execute **any** inline `<script>` and any inline event-handler attribute it finds in the document — which is precisely the payload an injected-HTML XSS delivers. **A CSP with `script-src 'unsafe-inline'` provides essentially zero XSS mitigation while still scoring points on header scanners.** `unsafe-eval` is a smaller but related hole (it re-enables `eval`, `new Function`, string `setTimeout`).

A host allowlist alone is also weak: one JSONP endpoint, one hosted-library CDN with arbitrary path serving, or one open redirect on an allowlisted origin, and the policy is bypassed. **This is why the modern recommendation is nonce + `strict-dynamic`, not a domain list.**

**Check:**
```bash
curl -sI https://example.com | grep -i '^content-security-policy' | tr ';' '\n' | sed 's/^ *//'
```
Grade it:

| Observation | Grade |
|---|---|
| No CSP at all | Fail — but honest |
| CSP present with `script-src ... 'unsafe-inline'` | **Worse than honest.** Report as "CSP present but non-functional for XSS" |
| CSP present with `'unsafe-eval'` | Note; find out which dependency demands it |
| Host-allowlist CSP, no `unsafe-inline` | Partial — check the allowlist for JSONP/library CDNs |
| `'nonce-…' 'strict-dynamic'` + `object-src 'none'` + `base-uri 'none'` | Pass |
| `Content-Security-Policy-Report-Only` only | In-progress — acceptable **if** dated; see §3.3 |

### 3.2 A realistic CSP for a site that has third-party scripts

The common objection is "we have analytics, a tag manager, a chat widget, and a payment iframe, so we can't have a real CSP." Nonce + `strict-dynamic` is exactly the answer to that: you nonce the loader tags you control, and `strict-dynamic` **propagates trust to the scripts they inject**, so you do not have to enumerate every downstream domain a tag manager pulls in.

```
Content-Security-Policy:
  default-src 'self';
  script-src 'nonce-{RANDOM_PER_RESPONSE}' 'strict-dynamic' https: 'unsafe-inline';
  style-src 'self' 'unsafe-inline';
  img-src 'self' data: https:;
  font-src 'self';
  connect-src 'self' https://analytics.example.net https://api.example.com;
  frame-src https://checkout.example-payments.com;
  frame-ancestors 'none';
  form-action 'self';
  base-uri 'none';
  object-src 'none';
  upgrade-insecure-requests;
  report-uri /csp-report; report-to csp
```

Read that carefully — three things in it are deliberate and are the things auditors get wrong:

| Line | Why |
|---|---|
| `'unsafe-inline'` **inside** a nonce'd `script-src` | It is a **backwards-compatibility fallback only**. Browsers that understand nonces **ignore `'unsafe-inline'` entirely** when a nonce or hash is present. Browsers that don't understand `strict-dynamic` fall back to it rather than breaking the site. **Do not flag this as a failure** — flag `'unsafe-inline'` **without** an accompanying nonce/hash |
| `https:` after `'strict-dynamic'` | Same fallback logic — ignored by browsers that honour `strict-dynamic` |
| `style-src 'unsafe-inline'` | Pragmatic. Inline **styles** are a far weaker vector than inline scripts, and almost every component framework and CSS-in-JS runtime emits them. Nonce your styles if you can; do not block the CSP rollout on it |
| `{RANDOM_PER_RESPONSE}` | The nonce must be **cryptographically random and regenerated on every response**. A static nonce, or one cached at the CDN with the HTML, is **equivalent to `unsafe-inline`**. Check by requesting the page twice and diffing the nonce: `for i in 1 2; do curl -sI URL \| grep -io "nonce-[A-Za-z0-9+/=_-]*"; done` — identical values are a finding |
| `connect-src` enumerated | This is the line that actually constrains exfiltration. A `connect-src *` or missing `connect-src` (falling back to `default-src 'self'` is fine; falling back to a permissive `default-src` is not) undoes much of the value |
| `frame-ancestors 'none'` | Supersedes `X-Frame-Options`. Keep XFO too for very old agents; it costs nothing |
| `form-action 'self'` | Frequently omitted; blocks form-hijacking injections |

**Static-hosting caveat:** per-response nonces require the HTML to be generated or edge-transformed per request. On a pure static host that serves identical bytes, use **hashes** (`'sha256-…'`) for the small set of inline scripts you actually ship, or move all inline script into files. `[S]`

### 3.3 Rolling out a CSP without breaking the site

1. Ship **`Content-Security-Policy-Report-Only`** with the intended policy plus a `report-uri`/`report-to` endpoint.
2. Run it for **2–4 weeks**. `[P]`
3. Triage every violation: legitimate resource → add it; unexpected resource → you just found an unowned third-party tag (see `performance.md` §4.4).
4. Only then switch the header name to enforcing.

**Check whether a site is mid-rollout:** `curl -sI URL | grep -i 'content-security-policy-report-only'`. Report-Only alone, with no enforcing policy and no stated date, is a stalled rollout — a common finding.
**Check for violations at runtime:** DevTools console, or `document.addEventListener('securitypolicyviolation', e => console.log(e.violatedDirective, e.blockedURI))`.

---

## 4. Subresource Integrity

**SRI (`integrity` + `crossorigin`) on every third-party `<script>` / `<link rel=stylesheet>` loaded from a CDN.** Without it, a compromised or hijacked CDN executes arbitrary code in your origin.

```bash
# every cross-origin script/stylesheet, and whether it carries integrity=
curl -s https://example.com \
  | grep -oE '<(script|link)[^>]*(src|href)="https?://[^"]+"[^>]*>' \
  | grep -v 'integrity='
```
Any output is a finding. Generate a hash with:
```bash
curl -s https://cdn.example.com/lib.js | openssl dgst -sha384 -binary | openssl base64 -A
```

**Caveats to state honestly:** SRI is incompatible with resources that legitimately change per request (most tag managers, most analytics loaders, "latest" CDN URLs). For those the correct finding is not "add SRI" but **"pin the version, self-host, or accept and document the risk."** SRI also requires `crossorigin="anonymous"` and correct CORS headers on the CDN.

---

## 5. Cookies

Requirements: **`Secure; HttpOnly; SameSite=Lax`** (or `Strict`). `SameSite=None` **requires** `Secure`. **Session cookies must not be readable by JS.**

```bash
curl -sI https://example.com | grep -i '^set-cookie'
```
Then in the browser: **DevTools → Application → Cookies** — sort by the `HttpOnly` and `Secure` columns and look for any session/auth cookie missing either. Cross-check `document.cookie` in the console: **anything session-related that appears there is a finding.**

| Cookie kind | Required flags |
|---|---|
| Session / auth | `Secure`, `HttpOnly`, `SameSite=Lax` or `Strict`, scoped `Path`, no over-broad `Domain` |
| CSRF token | `Secure`, `SameSite=Strict`; readable by JS only if the double-submit pattern requires it |
| Third-party / cross-site | `Secure`, `SameSite=None` — and justify it |
| Analytics | `Secure`, `SameSite=Lax`; **check it is actually gated on consent — §6.1, and that check is the whole point of §6** |

Also note **cookie count and size** — cookies are sent on every same-origin request, including static assets on the same host. A 4 KB cookie header on 60 asset requests is a performance finding too (`performance.md`).

---

## 6. Cookie consent as a compliance object

**Not legal advice.** This section describes what an auditor can *observe and prove* about a consent banner, and summarises regulation as publicly published. Confirm current obligations and deadlines with counsel. Source guide prepared 19 August 2026. (Same disclaimer, same reason, as `ada/00-map.md`.)

Elsewhere in this skill the consent banner appears three times, always as somebody else's problem: a CLS source (`performance.md` §2.1, §4.4), a focus obstruction (`ada/wcag22-new.md` — 2.4.11), and a tracker-exposure risk on health and statutory pages (`site-categories.md`). **None of those ask the only question the banner exists to answer: does it actually gate anything?** A banner that fires the trackers before you touch it is not a compliance control. It is a decoration that creates exposure while advertising awareness of the rule it breaks.

Every check below is observable from the outside with DevTools and a clean browser profile. Nothing here requires the client's CMP config.

### 6.1 Do the trackers fire before consent? — the check that matters

**Procedure. Run it exactly, and record each step; the ordering is what makes it evidence.**

1. **Fresh profile.** New incognito/private window, or a Playwright context with no storage state. A returning-visitor profile carries a prior consent decision and will show you a compliant-looking load that proves nothing.
2. **Open DevTools → Network *before* navigating.** Tick **Preserve log** and **Disable cache**.
3. **Navigate. Do not touch the banner.** Let the page settle, then scroll once — some tags are scroll-triggered.
4. **Filter to third-party origins.** Sort by Domain, or paste the console snippet below. Screenshot the filtered list with the banner still visible in the viewport — that single screenshot is the finding.
5. **Read `Application → Cookies`, `Local Storage` and `Session Storage`** for the same page state. Storage written before consent is the same violation as a network request; a CMP that blocks the script but not the cookie is a common half-fix.
6. **Now click "Reject all"** (or the equivalent). Re-check network and storage. **Requests that appear only after Reject are the strongest finding in this whole section.**
7. **Reload with the reject decision persisted.** Re-check. Trackers that return on the second page view mean the decision is not being enforced, only recorded.
8. **Repeat for Accept**, to confirm the banner does something at all — a banner where accept and reject produce identical network traffic is not a consent mechanism in either direction.

```js
// Paste at step 4. Inventories third-party requests already made, by origin.
(() => {
  const here = location.hostname.split('.').slice(-2).join('.');
  const byOrigin = {};
  for (const e of performance.getEntriesByType('resource')) {
    let h; try { h = new URL(e.name).hostname; } catch { continue; }
    if (h.endsWith(here)) continue;                       // first-party (crude but adequate)
    const rec = byOrigin[h] || (byOrigin[h] = { count: 0, bytes: 0, types: new Set(), first: null });
    rec.count++; rec.bytes += e.transferSize || 0; rec.types.add(e.initiatorType);
    if (rec.first === null) rec.first = Math.round(e.startTime);
  }
  return Object.entries(byOrigin)
    .map(([host, r]) => ({ host, requests: r.count, kb: +(r.bytes / 1024).toFixed(1),
                           types: [...r.types].join(','), firstAtMs: r.first }))
    .sort((a, b) => a.firstAtMs - b.firstAtMs);
})()
```

```js
// Storage written before any interaction.
({ cookies: document.cookie.split(';').map(s => s.trim().split('=')[0]).filter(Boolean),
   localStorage: Object.keys(localStorage), sessionStorage: Object.keys(sessionStorage) })
```

Record per origin: **host · what it is · first request time · fired before consent (y/n) · fired after reject (y/n).** "Analytics loads before consent" is a complaint; that table is a finding.

**What counts as a tracker for this purpose** is broader than the analytics tag people expect: product analytics and session replay, ad pixels and conversion tags, tag managers (the container itself, which then loads everything else), A/B testing and personalisation SDKs, chat and support widgets, embedded video in its non-cookieless mode, map and font CDNs that set identifiers, and CAPTCHA/anti-fraud scripts. **Strictly-necessary is a narrow category, and it is the site's job to demonstrate it.** The EDPB's Cookie Banner Taskforce recorded that some controllers classify as "essential" or "strictly necessary" cookies serving purposes that are not, that no stable list of essential cookies exists because cookie behaviour changes, and that the website owner carries the responsibility to maintain the list and demonstrate the essentiality of what is on it. `[P]` **Ask for that list. Its absence is itself a finding.**

The tooling caveat, stated in the report: the same taskforce noted that available scanners can enumerate the cookies a site places but **cannot determine their nature**. `[P]` So an auditor can prove *what fired and when*; classifying each as essential or not requires the client's documentation.

### 6.2 Reject must be as easy as accept

The single most-cited banner defect, and it is measurable rather than aesthetic. Positions below are the common denominator the EU supervisory authorities agreed for handling the NOYB complaints (EDPB Cookie Banner Taskforce report, **adopted 17 January 2023**) `[P]` — the report is explicit that it reflects a minimum threshold and not a certification, and that national implementing law applies on top.

| Practice | What the taskforce recorded | What to measure |
|---|---|---|
| **No reject option on any layer carrying a consent button** | "a vast majority of authorities considered that the absence of refuse/reject/not consent options on any layer with a consent button of the cookie consent banner is not in line with the requirements for a valid consent and thus constitutes an infringement" `[P]` — a few authorities disagreed, since Art. 5(3) ePrivacy does not name a reject option explicitly | Is there a reject control on the **first layer**? If not, count the clicks to refuse everything vs the one click to accept |
| **Pre-ticked boxes** on the settings layer | Do not produce valid consent, per GDPR recital 32 ("Silence, pre-ticked boxes or inactivity should not therefore constitute consent") and Art. 5(3) `[P]` | `$$('input[type=checkbox]')` inside the banner on a **fresh profile** — read `.checked` and `.defaultChecked` before touching anything |
| **Deceptive "link design"** — refusal offered as body text rather than a control | Not valid where the alternative is "embedded in a paragraph of text… in the absence of sufficient visual support", or placed **outside** the banner frame `[P]` | Is reject a `<button>`/`<a>` of comparable prominence, or an underlined word in a sentence? Is it inside the banner? |
| **Deceptive colours / contrast** | No universal colour standard can be imposed; assessment is **case by case**. The taskforce did identify one manifestly misleading case: an alternative action rendered as a button "where the contrast between the text and the button background is so minimal that the text is unreadable to virtually any user" `[P]` | Measure both buttons: computed size in px, contrast ratio of label against button background, and position. Report the numbers and state that the standard is case-by-case, **not** that a colour difference is per se unlawful |
| **Legitimate interest claimed for the trackers themselves** | The legal basis for placing/reading cookies under Art. 5(3) **cannot** be the controller's legitimate interest; and where Art. 5(3) is not complied with, the subsequent GDPR processing cannot be compliant either `[P]` | Read the second layer for "legitimate interest" toggles, especially ones defaulted on, and for a design that makes the user refuse twice |
| **No way to withdraw** | Three cumulative conditions on top of valid consent: withdrawal must be **possible**, **at any time**, and **as easy as giving consent**. A specific mechanism (e.g. a hovering icon) cannot be mandated; a persistent link or icon in a standardised place is the expected shape, assessed case by case `[P]` | Find the control on a page that is **not** the homepage, after consent has been given. Count clicks to withdraw vs clicks to accept |

**Dark-pattern overlap:** `site-categories.md` already carries the measurement recipes for unequal accept/reject prominence and pre-ticked boxes as DSA Art. 25 items. Use those recipes; cite them once, in whichever section the finding lands, and do not report the same observation twice.

### 6.3 Consent persistence and enforcement

| Check | How | Fail signal |
|---|---|---|
| The decision survives a reload | Reject, reload, re-read network + storage | Trackers return, or the banner re-prompts as if nothing was chosen |
| The decision survives navigation | Reject, then visit three other templates | Banner re-appears on inner pages, or trackers fire there |
| The decision survives the session | Reject, close the tab, return | A consent record with a very short lifetime is a re-prompt loop; note the observed `Expires`/`Max-Age` on the consent cookie |
| The record is scoped correctly | `Application → Cookies`, read the consent cookie's `Domain` and `SameSite` | A consent cookie without `Secure`/`SameSite` is also a §5 finding |
| **Accept and reject actually differ** | Diff the third-party origin inventory from §6.1 step 6 vs step 8 | Identical lists in both directions — the banner records a preference it never enforces |
| The CMP is not itself the leak | Check where the CMP's own script and its config are hosted, and what it sends | Also a performance finding: `performance.md` §4.4 flags CMPs as a top INP/LCP offender and recommends self-hosting |

**Third-party CMPs do not transfer responsibility.** California's enforcement head, announcing a **$345,178** fine against Todd Snyder, Inc. (**6 May 2025**) for — among other things — a privacy portal misconfigured such that opt-out requests went unprocessed for 40 days: *"Businesses should scrutinize their privacy management solutions to ensure they comply with the law and work as intended, because the buck stops with the businesses that use them… Using a consent management platform doesn't get you off the hook for compliance."* `[P]` **That is the sentence to quote when a client says the CMP vendor handles it.**

### 6.4 The banner's own accessibility

The banner is the first interactive thing on the page and routinely the least tested. It is author content, so every criterion applies to it.

| Check | Method | Criterion |
|---|---|---|
| Reachable and operable by keyboard alone | Load, press <kbd>Tab</kbd> **without touching the mouse**. Can you reach accept, reject, and settings, and activate each with <kbd>Enter</kbd>/<kbd>Space</kbd>? | **2.1.1 Keyboard** |
| No keyboard trap | Tab past the last banner control | **2.1.2 No Keyboard Trap** |
| Focus is *managed*, not merely possible | If the banner is modal, focus should move into it on appear and return to a sensible place on dismiss. If it is **not** modal, it must not steal focus mid-task | **2.4.3 Focus Order**, **3.2.1 On Focus** |
| Focus trap matches the visual claim | A banner that visually blocks the page but leaves the page behind it tabbable is a trap in the wrong direction — the user tabs into content they cannot see or use. Prefer `<dialog>` + `showModal()` (`code-quality.md` §2.1) or `inert` on the rest of the document | **4.1.2**, `ada/html-core.md` §9 |
| **Does not obscure the focused element** | Tab through a long page **with the banner still up** — a bottom-anchored banner covers the last tab stops of a long form. This is the procedure `ada/wcag22-new.md` already insists on, and the reason it insists on it | **2.4.11 Focus Not Obscured (Minimum), AA** |
| Announced to a screen reader | If it takes focus, it needs an accessible name and a role. If it does not, it needs to be discoverable in reading order at the top of the document, not appended at the end of `<body>` | **1.3.1 / 4.1.2** |
| Contrast, in every theme | Banners are frequently the one surface the site's dark theme never reached | **1.4.3 / 1.4.11**, `code-quality.md` §10.7 |
| Target size | Accept, reject and every settings toggle | **2.5.8 (AA)**, `mobile.md` §1.4 |
| Screen budget on mobile | A 120 px banner plus header plus tab bar can leave under half an iPhone SE screen | `mobile.md` §2.6 |

**An inaccessible reject button is a compliance finding twice over:** it fails WCAG, and it means refusal is *not* as easy as acceptance for the users it excludes. Report it under both headings and say so.

### 6.5 Privacy policy — presence, reachability, and linkage

Cheap to check, routinely missing or orphaned. `seo/L4-onpage.md` §4.2 already lists privacy policy and terms as expected trust signals; this is the compliance-side version.

| Check | Command / method |
|---|---|
| A privacy policy exists and returns 200 | `curl -s https://example.com \| grep -oiE '<a[^>]+href="[^"]*(privacy\|datenschutz\|confidentialite\|cookie)[^"]*"' \| sort -u`, then status-check each |
| It is linked from **every** page, not just the homepage | Check the footer template on three different page types |
| The **banner itself** links to it, before consent | Read the banner's own markup; a policy reachable only after accepting is not notice |
| A cookie/tracker list or table exists and matches what §6.1 observed | Compare the published list against the measured origin inventory. **Mismatches are the most defensible finding in this section** — a policy that omits a tracker you watched fire is documented and dated |
| A withdrawal route is described and works | §6.2, last row |
| A US-facing site offers the opt-out link its state laws expect | Look for a "Do Not Sell or Share My Personal Information" link (or a preference centre) in the footer, and confirm it does something |
| **Opt-out preference signals are honoured** | Load with a browser or extension sending **Global Privacy Control**, then re-run §6.1. `navigator.globalPrivacyControl` should read `true`; the site's behaviour should change. A site that reads the signal and ignores it is worse than one that never looked |
| The policy is not a template with the placeholders left in | Read it. `[COMPANY NAME]` in production is a real finding and a fast one |

### 6.6 The 2026 landscape — what an auditor should flag

Not legal advice; see the disclaimer at the top of this section. The point of this table is to tell the client *which* observations carry exposure, so the finding gets funded.

| Regime | Mechanism | What the auditor flags |
|---|---|---|
| **ePrivacy Directive Art. 5(3)** (as transposed nationally) | Governs the **storage of, and access to, information on the user's device** — cookies, `localStorage`, fingerprinting. Consent-first, opt-in | Anything non-essential written or read before an affirmative action. The EDPB taskforce confirmed the applicable framework for the *placement* of cookies is the national ePrivacy transposition, **not** the GDPR `[P]` |
| **GDPR** | Governs the **subsequent processing** of what those cookies collect, and supplies the definition and conditions of valid consent that ePrivacy borrows `[P]` | Invalid consent (pre-ticked, bundled, no refusal, not withdrawable) contaminates the processing downstream — the taskforce's position is that non-compliance with Art. 5(3) means the subsequent processing cannot be GDPR-compliant `[P]` |
| **GDPR one-stop-shop** | Does **not** apply to ePrivacy matters `[P]` | A multinational client cannot assume a single lead authority for cookie complaints. Worth saying out loud; clients routinely assume otherwise |
| **CCPA / CPRA (California)** | **Opt-out**, not opt-in. Sale/sharing of personal information, plus a right to limit use of sensitive PI | A US-only site does not need an EU-style consent wall — but it does need a working opt-out link, honoured opt-out preference signals, and a mechanism that is not itself a dark pattern. Auditing a Californian site against an EU model produces findings the client will correctly dismiss |
| **DSA Art. 25** (EU online platforms) | Prohibits interface designs that deceive or manipulate | Unequal accept/reject prominence, pre-ticked defaults. Recipes in `site-categories.md` |

**Enforcement trend to name, with dates.** California's regulator has been the most legible source of concrete, citable actions: a **$632,500** order against American Honda Motor Co. (**12 March 2025**), the **$345,178** Todd Snyder order (**6 May 2025**) quoted above, and a **$1.35M** CCPA resolution with a national retailer (**30 September 2025**) `[P]`. It issued an enforcement advisory specifically on **avoiding dark patterns** (**4 September 2024**) `[P]`, and one warning businesses against demanding excessive information from consumers exercising privacy rights `[P]`. It now operates a **Delete Request and Opt-out Platform (DROP)**, and as of **26 January 2026** publishes as **CalPrivacy** at `privacy.ca.gov` `[P]`. The through-line for an audit: **regulators are testing whether the mechanism works, not whether it exists.** A privacy portal that silently fails, a reject button wired to nothing, and an opt-out link behind an identity check are the shapes that have actually drawn fines.

**What this section deliberately does not do:** assert which regime applies to the client. That is a scoping question (`scoping.md`) and, past a certain point, a question for counsel. Record the observable facts, note the regime the client says applies, and let the severity follow from `reporting.md` §2.

### 6.7 Testable checklist for the banner

- [ ] Fresh profile used, network log preserved, **screenshot taken with the banner still on screen**
- [ ] Third-party origin inventory recorded **before** any interaction — §6.1 snippet
- [ ] Storage inventory recorded before any interaction — cookies, `localStorage`, `sessionStorage`
- [ ] Inventory re-taken after **Reject**, and after **Accept**; the two differ
- [ ] Rejection persists across reload, across navigation, and across a new session
- [ ] Reject control present on the **first layer**, as a control of comparable prominence, inside the banner frame
- [ ] Zero pre-ticked non-essential boxes on a fresh profile — read `.defaultChecked`
- [ ] Accept/reject button sizes, contrast ratios and positions measured and reported as numbers
- [ ] No legitimate-interest basis claimed for the placement of trackers
- [ ] Withdrawal control findable on a non-homepage after consent; click count recorded vs accept
- [ ] Banner is keyboard-reachable and operable; no trap; focus managed
- [ ] Focused elements never entirely obscured by the banner — tab a long page with it up (**2.4.11**)
- [ ] Banner contrast and target sizes pass, in **every** theme
- [ ] Privacy policy exists, returns 200, is linked sitewide and from the banner itself
- [ ] Published cookie/tracker list reconciled against the measured origin inventory; mismatches listed
- [ ] Opt-out preference signal (GPC) sent and the site's behaviour re-measured
- [ ] Client asked for the essential-cookie justification list; its absence recorded as a finding

---

## 7. Copy-pasteable baselines

### 7.1 Static site (no per-request HTML generation)

Nginx form; translate directly to `_headers`, `netlify.toml`, `staticwebapp.config.json`, CloudFront function, or Caddy.

```nginx
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
add_header X-Content-Type-Options "nosniff" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Permissions-Policy "camera=(), microphone=(), geolocation=(), payment=(), usb=(), interest-cohort=()" always;
add_header Cross-Origin-Opener-Policy "same-origin" always;
add_header Cross-Origin-Resource-Policy "same-origin" always;
add_header X-Frame-Options "DENY" always;
add_header Content-Security-Policy "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; form-action 'self'; base-uri 'none'; object-src 'none'; upgrade-insecure-requests" always;
```

Pair with the caching rules from `performance.md` §3.8:
```nginx
location ~* \.[0-9a-f]{8,}\.(js|css|woff2|avif|webp|png|jpg|svg)$ {
  add_header Cache-Control "public, max-age=31536000, immutable" always;
}
location = /index.html { add_header Cache-Control "no-cache" always; }   # never no-store — it kills bfcache
```

### 7.2 What changes for a dynamic site

| Change | Reason |
|---|---|
| `script-src 'nonce-{PER_RESPONSE}' 'strict-dynamic' https: 'unsafe-inline'` | You can now generate a per-request nonce; take it. §3.2 |
| Enumerate `connect-src`, `frame-src`, `img-src` for real third parties | A dynamic site has APIs, embeds, payment iframes |
| Add `form-action` explicitly | Real forms exist |
| Cookie headers appear — apply §5 in full | Sessions exist |
| `Cache-Control: private, no-cache` + `ETag` on personalised responses; **never `no-store` on HTML you want bfcached** | Firefox/Safari treat `no-store` as a hard bfcache blocker; **23% of sites do it** (`performance.md` §2.3) |
| `Vary: Accept-Encoding` (+ `Accept` if you content-negotiate images) | Prevents cache poisoning across encodings |
| CSP nonce must **not** be cached with the HTML at the CDN | Cache the page without the nonce, or mark it uncacheable, or inject the nonce at the edge |
| `report-to` / `report-uri` endpoint | You now have somewhere to send violations |

---

## 8. Scanners — what each is good for

| Tool | Use it for | Limit |
|---|---|---|
| **`curl -sI`** | Ground truth. Always run this first and read the raw headers yourself | Doesn't evaluate policy quality |
| **`securityheaders.com`** | Fast third-party-verifiable grade to put in a report | **Scores presence, not effect.** A+ with `unsafe-inline` is a false positive — §0 rule |
| **Mozilla Observatory** | Broader: CSP quality, cookies, SRI, redirection, referrer policy. Better CSP grading than the above | Opinionated; some deductions are debatable |
| **`testssl.sh`** | TLS version/cipher/chain/expiry detail | Nothing above the TLS layer |
| **CSP Evaluator** (Google) | Paste a CSP, get bypass analysis (JSONP endpoints, permissive allowlists) | Static analysis only |
| **DevTools → Security panel** | Per-resource certificate and mixed-content view for the *rendered* page | One page at a time |
| **DevTools → Application → Cookies** | The only reliable place to read `HttpOnly` | Manual |
| **`../scripts/check_headers.py`** | Batch all of the above across every template, plus the exposed-path probes from `code-quality.md` §9.2 | Automate, then eyeball the CSP by hand |

**Run headers on more than the homepage.** Header config often differs between the CDN-cached marketing pages and the origin-served application routes — check at least one of each.

---

## 9. Testable checklist

`[AUTO]` = automatable via `curl` / `../scripts/check_headers.py` / Playwright.

### A. Transport
- [ ] `[AUTO]` HTTPS everywhere; HTTP redirects to HTTPS with a **301**, at most one hop
- [ ] `[AUTO]` HSTS present with `max-age` ≥ **31536000**; `includeSubDomains` present (and every subdomain verified HTTPS-capable first)
- [ ] `[AUTO]` `preload` present only if consciously accepted
- [ ] `[AUTO]` TLS 1.2/1.3 only; complete chain; > 30 days to expiry (`testssl.sh`)
- [ ] `[AUTO]` **Zero mixed content** on every audited template

### B. Headers
- [ ] `[AUTO]` `X-Content-Type-Options: nosniff`
- [ ] `[AUTO]` `Referrer-Policy` set (`strict-origin-when-cross-origin` or stricter)
- [ ] `[AUTO]` `Permissions-Policy` present and **deny-by-default**
- [ ] `[AUTO]` `frame-ancestors` in CSP (and/or `X-Frame-Options`)
- [ ] `[AUTO]` `Cross-Origin-Opener-Policy: same-origin`
- [ ] `[AUTO]` `Cross-Origin-Resource-Policy` set appropriately for the asset class
- [ ] `[AUTO]` No `X-Powered-By`; no version-disclosing `Server`
- [ ] `[AUTO]` Headers checked on **≥ 2 templates** (one CDN-cached, one origin-served)

### C. CSP — graded on policy, not presence
- [ ] `[AUTO]` CSP present (enforcing, not only Report-Only — or Report-Only with a stated end date)
- [ ] `[AUTO]` `script-src` does **not** contain `'unsafe-inline'` **without** an accompanying nonce or hash
- [ ] `[AUTO]` `script-src` does not contain `'unsafe-eval'` (or the reason is documented)
- [ ] `[AUTO]` Nonce **differs between two consecutive requests**
- [ ] `[AUTO]` `object-src 'none'`, `base-uri 'none'`, `form-action` set
- [ ] `[AUTO]` `connect-src` is enumerated, not wildcard
- [ ] Host allowlist (if used) contains no JSONP endpoints or arbitrary-path library CDNs — check with **CSP Evaluator**
- [ ] `[AUTO]` **Zero CSP violations** in the console across load and the primary user flow
- [ ] `report-uri`/`report-to` endpoint exists and is monitored

### D. Subresources & cookies
- [ ] `[AUTO]` SRI `integrity` (+ `crossorigin`) on all CDN-hosted `<script>` / `<link rel=stylesheet>`, or the exception is documented (tag managers, versionless URLs)
- [ ] `[AUTO]` Session/auth cookies have `Secure` **and** `HttpOnly`; `SameSite` set on every cookie
- [ ] `[AUTO]` `SameSite=None` never appears without `Secure`
- [ ] `[AUTO]` No session/auth cookie readable from `document.cookie`
- [ ] `[AUTO]` Cookie header size is not inflating every asset request
- [ ] **Consent banner audited as a compliance object, not just a CLS source — the full list is §6.7**, and its headline check is that no non-essential tracker fires before an affirmative action, evidenced by a preserved network log screenshotted with the banner still on screen

### E. Exposure (shared with `code-quality.md` §9)
- [ ] `[AUTO]` `/.git/config`, `/.git/HEAD`, `/.env`, `/.env.production`, `/.DS_Store`, `/backup.zip` all return 404
- [ ] `[AUTO]` No reachable `.map` files
- [ ] `[AUTO]` No secrets (`sk_live`, `AKIA`, `-----BEGIN`, bearer tokens) in the built bundle
- [ ] `[AUTO]` Error pages return no stack traces
- [ ] `[AUTO]` Third-party-verifiable grade recorded (securityheaders.com **and** Mozilla Observatory), **with the policy-quality caveat stated in the report**
