# Running the program: deliverables and remediation

**Covers:** VPAT/ACR, accessibility statements, documenting a Title II exception, remediation prioritisation, the document backlog, and a realistic 12-month sequence.
**Load when:** `compliance.deliverable` is `vpat`, `statement`, `exception` or `roadmap` — i.e. the engagement produces something beyond the audit findings.
**Prerequisite:** the conformance model and the Title II exceptions in `targets.md`.

---

## 1. VPAT / ACR

- A **VPAT®** (Voluntary Product Accessibility Template, from ITI) is the **blank form**. A completed one is an **Accessibility Conformance Report (ACR)**. Using the terms interchangeably marks you as not having done one.
- **Editions:** WCAG, Revised Section 508, EN 301 549, and **INT** (all three combined). **Choose INT if you sell internationally.**
- Each criterion gets a conformance level — **Supports**, **Partially Supports**, **Does Not Support**, **Not Applicable** — **with explanatory remarks**. A VPAT that says "Supports" on all 55 A/AA criteria with no remarks is a **red flag** to any experienced procurement reviewer.
- Have it prepared or reviewed by someone qualified. **An inaccurate ACR is a misrepresentation with contractual consequences.**

**How to verify an ACR is complete and honest:** enumerate the 55 Level A + AA criteria from `criteria-index.md` and require an explicit disposition for each. Then cross-check the ACR against the audit findings: every "Does Not Support" and "Partially Supports" in the report must appear in the ACR, and every ACR "Supports" must have been actually tested, not inferred. Any criterion where the only evidence is an automated scan is at best "Partially Supports" — automation covers 30–40% (`testing.md`).

---

## 2. Accessibility statement

Required by law in the EU and UK public sectors, and expected practice everywhere.

| # | Must include | Failure mode if omitted |
|---|---|---|
| 1 | Your **conformance target and actual status** — e.g. "partially conformant with WCAG 2.2 AA" | Vague "we are committed to accessibility" boilerplate, which is worthless |
| 2 | **Known limitations**, specifically listed, with the reason and a **target fix date** | A statement claiming full conformance beside a site with Level A failures — a misrepresentation |
| 3 | **Feedback mechanism** — an email and/or phone number **a human monitors**, with a stated response time | An unmonitored form; the most-complained-about pattern in public-sector accessibility |
| 4 | **Alternative access arrangements** for anything not yet accessible | |
| 5 | **Enforcement/escalation route** where legally required | Non-compliant statement under the UK/EU public sector regs |
| 6 | **Date prepared and date last reviewed** — and actually review it | A statement dated three years ago describing a site that has been rebuilt since |

**How to verify:** fetch the statement (`/accessibility` is the conventional path) and check the six items against the page. Then test the feedback channel: send a message and time the response. An unanswered feedback address is itself a finding.

---

## 3. Documenting a Title II exception

If a public entity is relying on the **undue burden / fundamental alteration** exception (exception 6 in `targets.md`), the rule expects:

- A **written determination** by a high-level official with budgetary authority, or their designee
- A statement of the **reasons and supporting analysis**
- A description of **how you will otherwise provide access** to the affected information or service
- Ideally, a **review date**

Keep the written determination on file. **"It was too expensive," said verbally by a project manager, is not the exception.**

**How to verify:** ask for the document. If it does not exist as a signed artefact naming an official with budgetary authority, the exception is not established and the content is in scope. Record that as a finding, not as an exemption.

---

## 4. Remediation: prioritise by harm, then by reach

### Tier 1 — blocks access entirely (fix first)

Keyboard traps · unlabelled form fields in critical flows · missing alt on functional images · inaccessible authentication · unreachable interactive content · untagged PDFs required to access a service · flashing content

### Tier 2 — severe barriers

Contrast failures · missing headings/structure · unlabelled buttons · poor focus indication · missing captions · broken reading order in documents

### Tier 3 — friction

Vague link text · missing landmarks · missing skip links · inconsistent navigation · missing `lang` on passages

**Cross-cut this with reach.** The header component that appears on 4,000 pages is **one fix with 4,000 pages of benefit**.

> **Fix design systems and templates before individual pages.**

**How to verify prioritisation is right:** for each finding, record (tier, page count affected). Sort by tier, then by page count. If your top ten items are not dominated by global components, you are remediating pages instead of causes.

---

## 5. Stop the bleeding first

The highest-leverage move in almost every organisation is **not** remediating the backlog — it is preventing new inaccessible content.

| Control | Effect |
|---|---|
| Accessible **component library** with the patterns solved once | Removes the whole class of custom-widget failures |
| **Author training** for anyone who publishes — headings, alt text, link text, tables, document export | **Ninety minutes of training prevents years of remediation** |
| **Acceptance criteria** that include accessibility, and definition-of-done gates | Stops issues at the ticket, not the audit |
| **Procurement language** requiring conformance and an ACR from every vendor and SaaS tool | Third-party content still fails your pages (conformance requirement 2, `targets.md`) |
| **CI gates** on automated checks | Regression net — see `testing.md` Layer 1 |
| **Design review** for contrast, target size, focus states and text-spacing resilience *before* build | These four are near-free at design time and expensive after |

---

## 6. The document backlog

Public entities typically discover they have **tens of thousands** of PDFs.

| # | Step | Detail |
|---|---|---|
| 1 | **Inventory** | Crawl the site, list every document, capture last-accessed analytics. |
| 2 | **Triage** | Is this needed to access a service? Is it current? Is it used? |
| 3 | **Delete ruthlessly** | **The cheapest accessible PDF is the one you take down.** A large fraction of most backlogs is obsolete. |
| 4 | **Convert to HTML** where the content suits it | Usually better for users *and* cheaper than remediation. **Forms especially** — an accessible HTML form beats an accessible PDF form on every dimension. |
| 5 | **Archive** what qualifies | Under the archived-content exception, in a designated archive area, unmodified — all three conditions (`targets.md`). |
| 6 | **Remediate** what's left | Highest-use first, fixing the source where it exists (`documents-pdf.md` §6). Budget a full day or more for a complex 40-page report. |
| 7 | **Fix the pipeline** | So new documents are born accessible (`documents-office.md`). |

**How to verify the backlog is under control:** the count of documents published in the last 90 days that fail `verapdf --flavour ua1` should be **zero**. If new untagged PDFs are still appearing, step 7 has not happened and steps 1–6 are a treadmill.

---

## 7. A realistic 12-month sequence

| Months | Focus |
|---|---|
| **1–2** | Inventory content and documents. Baseline automated scan. Commission a manual audit of a representative sample. Pick the conformance target (2.2 AA). |
| **2–3** | Publish the accessibility statement and feedback channel. Train content authors. Update procurement templates. |
| **3–6** | Fix the design system / templates / global components. Add CI gates. Fix Tier 1 issues across the board. |
| **6–9** | Page-level and flow-level remediation, prioritized by traffic. Document backlog triage and deletion. |
| **9–12** | Document remediation. Screen reader testing pass. Usability testing with disabled users. Re-audit. Update the statement and VPAT. |
| **Ongoing** | Regression testing, author training refreshes, annual re-audit. |

Note the ordering: the **statement and training come before most of the fixing**. The statement is what tells users how to get help while the fixing is in progress, and training is what stops the backlog growing faster than you can clear it.

**How to verify progress is real:** re-run the same sample with the same procedure and compare finding counts by tier, not in aggregate. A drop in Tier 3 findings with Tier 1 unchanged means the easy things got fixed and the blocking things did not.
