# Legal regimes and conformance targets

**Covers:** which law applies, its deadlines and thresholds, how to pick a conformance target, what a formal conformance claim requires, and what is not compliance.
**Load when:** `compliance.regime` is anything other than `none`, the client asks what they are actually obligated to do, or the report must state a target.
**Prerequisite:** the three-layer model (law → standard → techniques) in `00-map.md`.

---

## 1. Which regime applies

### 1.1 The ADA itself

| Title | Covers | Technical standard |
|---|---|---|
| **Title I** | Employment (15+ employees) | None specified. Reasonable-accommodation duty covers internal systems, HR portals, application forms. |
| **Title II** | State and local government — public universities, K-12 districts, transit, courts | **WCAG 2.1 Level AA**, via the DOJ 2024 rule (28 CFR Part 35) |
| **Title III** | Places of public accommodation — private businesses open to the public | **None specified.** Courts and DOJ settlements overwhelmingly apply WCAG 2.0/2.1 AA as the de facto benchmark. |

### 1.2 ADA Title II — the DOJ web rule

The only place the ADA carries an actual binding technical standard for digital content.

| Fact | Value |
|---|---|
| Published | **24 April 2024** |
| Effective | **24 June 2024** |
| Standard | **WCAG 2.1 Level A and AA** |
| Scope | Web content **and** mobile apps provided by or on behalf of a public entity, including content built and maintained by third-party vendors. If a contractor runs your permits portal, it is still your obligation. |
| Deadline — entities serving **50,000+** people | **26 April 2027** (extended from 24 April 2026) |
| Deadline — entities serving **under 50,000**, and **special district governments** | **26 April 2028** (extended from 26 April 2027) |
| Source of the extension | DOJ Interim Final Rule, **20 April 2026** |

The extension moved **only the dates**. The standard and scope did not change. Do not read it as a softening.

**Exceptions in the rule** — all construed narrowly; do not build a strategy on them:

| # | Exception | The catch |
|---|---|---|
| 1 | **Archived web content** | Must be kept only for reference/research, unaltered since archiving, **and** stored in a designated archive area. All three. |
| 2 | **Preexisting conventional electronic documents** (PDF, Word, Excel, PowerPoint posted before the compliance date) | Not exempt if *currently used* to apply for, access, or participate in a service. A 2019 tax form people still file is not exempt. |
| 3 | **Third-party content** | Only where posted by someone with no contractual or licensing arrangement with the entity. |
| 4 | **Individualized password-protected documents** (e.g. one student's transcript) | Note "individualized" — a password-protected document sent to everyone is not exempt. |
| 5 | **Preexisting social media posts** | |
| 6 | **Undue burden / fundamental alteration** | Requires a **written determination by a high-level official with budgetary authority**, documented reasoning, and provision of an alternative means of access. Paperwork-heavy defence, not a shrug. See `program.md`. |

### 1.3 Section 508 — federal agencies and their vendors

- Rehabilitation Act §508, revised January 2017, **effective 18 January 2018**.
- **Standard: WCAG 2.0 Level AA**, incorporated by reference for both web and non-web electronic content.
- Explicitly covers **electronic documents** — the Revised 508 Standards apply WCAG 2.0 A/AA to PDFs and Office files.
- Drives the **VPAT/ACR** procurement ecosystem (`program.md`). Selling to the U.S. federal government means you will be asked for one.
- Note the version lag: 508 points at 2.0 while Title II points at 2.1. **Building to 2.2 AA satisfies both.**

### 1.4 Section 504 — HHS rule

HHS finalized a Section 504 rule in **May 2024** applying **WCAG 2.1 Level AA** to web content and mobile apps of recipients of HHS funding — hospitals, health systems, many social service providers, state health agencies. Compliance dates run **May 2026** for larger recipients and **May 2027** for smaller ones. This rule has **not** received the DOJ Title II extension. If the client is healthcare or HHS-funded, verify their date independently.

### 1.5 Air Carrier Access Act

DOT rules require airline websites marketing to the U.S. public to meet **WCAG 2.0 Level AA**, and require accessible kiosks. Separate enforcement regime from the ADA.

### 1.6 Outside the U.S.

| Jurisdiction | Instrument | Standard | Status |
|---|---|---|---|
| **EU** | European Accessibility Act (Dir. 2019/882) | EN 301 549 (currently v3.2.1 → WCAG 2.1 AA) | Applies since **28 June 2025**. Contracts concluded before that date get until **28 June 2027**. |
| **EU** | Web Accessibility Directive (2016/2102) | EN 301 549 | Public sector; in force since 2020/2021 |
| **EU** | **EN 301 549 v4.1.1** | **WCAG 2.2 AA** | Expected to publish in 2026; becomes the presumption-of-conformity standard once cited in the Official Journal |
| **UK** | Public Sector Bodies Accessibility Regs 2018; Equality Act 2010 | WCAG 2.2 AA | Public sector regs updated to 2.2 |
| **Canada** | Accessible Canada Act; **AODA** (Ontario) | WCAG 2.0 AA | AODA deadline passed 1 Jan 2021 |
| **Australia** | Disability Discrimination Act; DTA standard | WCAG 2.1 AA | |
| **Israel** | IS 5568 | WCAG 2.0 AA + national additions | |
| **India** | RPwD Act; GIGW 3.0 | WCAG 2.1 AA | |

**EN 301 549 is broader than WCAG.** Separate clauses for web (ch. 9), **non-web documents (ch. 10)**, non-web software (ch. 11), documentation and support services (ch. 12), and hardware. Selling into Europe with downloadable documents? Chapter 10 governs your PDFs.

---

## 2. The standards layer

| Version | Published | Structure | Count | Status |
|---|---|---|---|---|
| **WCAG 1.0** | 5 May 1999 | 14 guidelines, checkpoints at Priority 1/2/3 | 65 checkpoints | **Obsolete.** Superseded Dec 2008. Legacy claims only — mapping in `criteria-index.md`. |
| **WCAG 2.0** | 11 Dec 2008 | 4 principles → 12 guidelines → SC at A/AA/AAA | 61 SC (25 A, 13 AA, 23 AAA) | W3C Rec. Also **ISO/IEC 40500:2012**. Referenced by Section 508. |
| **WCAG 2.1** | 5 June 2018 | Same, + mobile, low vision, cognitive | 78 SC (30 A, 20 AA, 28 AAA) | W3C Rec. Referenced by ADA Title II, EN 301 549 v3.2.1. |
| **WCAG 2.2** | 5 Oct 2023 | Same, + 9 new SC, − 1 removed | **86 SC (31 A, 24 AA, 31 AAA)** | **Current recommended target.** |
| **WCAG 3.0** | Draft | Outcomes + scoring model, not A/AA/AAA | — | Working Draft. Years from Recommendation. **Do not plan against it.** |

**Backward compatibility:** 2.1 and 2.2 are strictly additive. Anything conforming to 2.2 conforms to 2.1 and 2.0 — with one asterisk: WCAG 2.2 **removed 4.1.1 Parsing**, so a page can conform to 2.2 while technically failing 2.0/2.1. In practice this never matters (browsers error-correct malformed markup consistently, which is why it was removed), but a formal claim against 2.0 or 2.1 must still address it. See `wcag22-new.md`.

**POUR** — the four principles, introduced in 2.0 and unchanged since:

| Principle | Means |
|---|---|
| **Perceivable** | Users can perceive the information — it isn't invisible to all their senses. |
| **Operable** | Users can operate the interface — it doesn't require interaction they can't perform. |
| **Understandable** | Users can understand both the information and the operation. |
| **Robust** | Content works reliably with current and future user agents, including assistive technology. |

**Why 1.0 → 2.0 mattered:** 1.0 was written against 1999 technology and named specific HTML elements and features ("until user agents allow users to turn off spawned windows, do not cause pop-ups"). 2.0 replaced technique-specific rules with **technology-neutral, testable success criteria** and moved implementation advice into a separate, continuously updated *Techniques* document. That is why WCAG 2.x can be applied to PDF, EPUB and native apps, and it is the structural reason WCAG2ICT exists at all.

### Companion document standards

| Standard | What it covers |
|---|---|
| **PDF/UA-1** — ISO 14289-1:2014 | Accessible PDF, based on PDF 1.7. The practical production target today. |
| **PDF/UA-2** — ISO 14289-2:2024 | Accessible PDF, based on PDF 2.0 (ISO 32000-2). Adds MathML, better annotations, richer metadata. Derived from the PDF Association's WTPDF spec. Tooling still catching up. |
| **Matterhorn Protocol 1.1** | PDF/UA-1 translated into **31 checkpoints / 136 failure conditions** — 89 machine-checkable, 45 requiring human judgement, 2 undetermined. What PAC and veraPDF implement. |
| **PDF/A** — ISO 19005 | Archiving. Orthogonal to accessibility but frequently required alongside. PDF/A-4 pairs with PDF 2.0. |
| **ISO 32005** | Tagged PDF structure element mapping (PDF 1.7 tags ↔ PDF 2.0 namespaces). |
| **EPUB Accessibility 1.1** | ISO/IEC 23761. References WCAG 2.x. Required by the EAA for ebooks sold in the EU. |
| **ARIA 1.2 / ARIA in HTML** | Normative rules for using ARIA. Relevant to WCAG 4.1.2. |

**PDF/UA does not replace WCAG.** The PDF Association is explicit that conformity to PDF/UA does not by itself ensure the content is accessible. PDF/UA guarantees the *machinery* (tags exist, nest correctly, reading order defined, metadata present); WCAG governs the *content decisions* (is the alt text meaningful, is contrast sufficient, is the language plain). Both, always.

---

## 3. Choosing the target

**Short answer: WCAG 2.2 Level AA, plus PDF/UA for any PDFs.**

1. 2.2 AA is a superset of every currently-referenced legal standard — 2.0 AA for §508, 2.1 AA for Title II / §504 / EN 301 549 v3.2.1.
2. The six new A/AA criteria in 2.2 are cheap to satisfy if designed in and expensive to retrofit — particularly **3.3.8 Accessible Authentication**, which can require changes to your identity provider.
3. EN 301 549 v4.1.1 is expected to move Europe to 2.2 AA in 2026 anyway.

**On Level AAA:** do not adopt AAA as a blanket target. W3C explicitly states it is not possible to satisfy all AAA criteria for some content, and full AAA conformance is vanishingly rare. Adopt **individual** AAA criteria where the audience warrants it, and name them individually in the claim:

| AAA criterion | Adopt when |
|---|---|
| 1.4.6 Contrast (Enhanced) 7:1 | Older or low-vision-heavy audience |
| 1.4.8 Visual Presentation | Long-form reading content |
| 2.2.6 Timeouts | Forms with data-loss risk |
| 2.4.13 Focus Appearance | High-keyboard-use applications |
| 3.1.5 Reading Level | Public-facing government information |
| 3.3.9 Accessible Authentication (Enhanced) | Consumer login flows |

---

## 4. What a conformance claim actually requires

Five formal requirements. **All five must hold.** "We fixed most of the issues" is not conformance.

| # | Requirement | What it means in an audit |
|---|---|---|
| 1 | **Conformance level** | All SC at the claimed level *and below* are satisfied, or a conforming alternate version is provided. For 2.2 AA that is **55 criteria** (31 A + 24 AA). |
| 2 | **Full pages** | Conformance is a property of a *complete page*. You cannot exclude part of a page. A failing third-party ad widget fails the page. |
| 3 | **Complete processes** | If a page is part of a multi-step process (checkout, application, enrollment), **every page in that process must conform**. A conformant product page plus a non-conformant payment step = the process does not conform. This drives sampling — see `testing.md`. |
| 4 | **Accessibility-supported technologies only** | Any technology relied on to satisfy a criterion must be supported by users' AT. Content in a non-accessibility-supported technology needs a conformant alternative. |
| 5 | **Non-interference** | Technologies you *don't* rely on must not block access. Even in non-relied-upon parts you must satisfy **1.4.2 Audio Control, 2.1.2 No Keyboard Trap, 2.2.2 Pause Stop Hide, 2.3.1 Three Flashes**. A decorative autoplaying background video with a keyboard trap breaks conformance for the whole page. |

**Conforming alternate versions.** A separate page/version that conforms, contains the same information and functionality, is as up to date, and is reachable from the non-conforming page (or vice versa) by an accessible mechanism. Legitimate, but a maintenance liability — two versions drift. A bridge, not a destination.

**Statement of partial conformance.** Two forms: (a) third-party content outside your control that you cannot fix but monitor and repair **within two business days**; (b) language — content in a language whose accessibility support is unknown. Neither is a general-purpose excuse.

---

## 5. What is not compliance

| Claim | Reality |
|---|---|
| **Overlay widget / accessibility toolbar** (accessiBe, UserWay and similar) | Does not produce conformance. Subject of a large volume of litigation itself. Publicly opposed by most of the accessibility profession — several hundred practitioners signed the Overlay Fact Sheet. **Flag its presence as a finding, not as a mitigation.** |
| **"We ran axe and it passed" / a Lighthouse score of 100** | Automated tools detect roughly **30–40%** of WCAG failures. A perfect automated score is consistent with a completely unusable site. See `testing.md`. |
| **An "ADA compliance certificate"** | No accrediting body issues one. The vendor is selling something that does not exist. |
| **A single audit, three years ago** | Conformance is a property of your *current* content, which changes weekly. |

## How to verify

| Question | Concrete check |
|---|---|
| Which regime applies? | Identify the entity type from the site itself: `.gov`/`.us` domain, "City of", "County of", "State of", public university → Title II. Federal agency `.gov` or a federal contract page → §508. Hospital / HHS grantee → §504. EU-facing commerce → EAA. Record in `compliance.regime`. |
| Which Title II deadline? | Find the population served — U.S. Census QuickFacts for the jurisdiction named in the site footer. ≥50,000 → **26 April 2027**; <50,000 or a special district → **26 April 2028**. |
| Is an overlay installed? | `curl -s <url> > raw.html` then `grep -i -e accessibe -e userway -e audioeye -e equalweb -e accessiway -e maxaccess raw.html`, or open DevTools → Network and filter for those hostnames. Also look for a floating accessibility icon in a page corner. |
| Is the target claim honest? | Search the site for its accessibility statement (`/accessibility`, footer link). Compare the level it claims against your findings. A claim of "fully conformant" alongside any Level A failure is a misrepresentation worth naming in the report. |
| Does a multi-step process conform? | List every URL in the process from entry to confirmation. If you have not tested all of them, you cannot claim the process conforms — requirement 3 above. |

## Primary sources

- WCAG 2.2 — https://www.w3.org/TR/WCAG22/ · 2.1 — https://www.w3.org/TR/WCAG21/ · 2.0 — https://www.w3.org/TR/WCAG20/
- How to Meet WCAG (Quick Reference) — https://www.w3.org/WAI/WCAG22/quickref/
- Understanding WCAG 2.2 — https://www.w3.org/WAI/WCAG22/Understanding/
- Techniques for WCAG 2 — https://www.w3.org/WAI/WCAG22/Techniques/
- ISO/IEC 40500:2012 (= WCAG 2.0) — https://www.iso.org/standard/58625.html
- PDF/UA (ISO 14289) — https://pdfa.org/resource/iso-14289-pdfua/ · PDF/UA-2 — https://pdfa.org/iso-14289-2-pdfua-2/
- Matterhorn Protocol — https://pdfa.org/resource/matterhorn-protocol/
- ARIA Authoring Practices Guide — https://www.w3.org/WAI/ARIA/apg/
- DOJ ADA Title II web rule — https://www.ada.gov/resources/2024-03-08-web-rule/
- Section 508 — https://www.section508.gov/
- EN 301 549 — https://www.etsi.org/deliver/etsi_en/301500_301599/301549/
- European Accessibility Act — https://ec.europa.eu/social/main.jsp?catId=1202
