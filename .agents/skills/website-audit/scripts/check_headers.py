#!/usr/bin/env python3
"""Grade security, caching and hygiene headers against the audit baseline.

Standard library only — no pip install, no browser. This is the 60-second
baseline from ``references/security-and-hygiene.md`` §1, automated.

The interpretive rule the whole script is built around
(``security-and-hygiene.md`` §0):

    Grade the **policy**, not the letter. An A+ on securityheaders.com with
    ``unsafe-inline`` in the CSP is worse than a B with a strict nonce policy.

Concretely that means:

* ``script-src 'unsafe-inline'`` **with** a nonce or hash present is *not*
  flagged — browsers that understand nonces ignore ``'unsafe-inline'``
  entirely, so it is a backwards-compatibility fallback (§3.2).
* ``script-src 'unsafe-inline'`` **without** a nonce or hash is a real finding:
  "CSP present but non-functional for XSS" (§3.1).
* A nonce that does not change between two consecutive responses is equivalent
  to ``'unsafe-inline'`` and is reported as such (§3.2).

Usage:
    ./check_headers.py https://example.com
    ./check_headers.py https://example.com --path /pricing --path /app/dashboard
    ./check_headers.py https://example.com --json out/headers.json --fail-on major
"""

from __future__ import annotations

import re
import ssl
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.client import HTTPResponse
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse, urlunparse

from _common import (
    Finding,
    Report,
    TargetUnreachable,
    base_parser,
    finish,
    normalise_url,
    run_cli,
)

# --------------------------------------------------------------------------- #
# Thresholds — every number below is cited to its source
# --------------------------------------------------------------------------- #

# security-and-hygiene.md §2: "HSTS present and long ... max-age >= 31536000"
HSTS_MIN_MAX_AGE = 31_536_000

# security-and-hygiene.md §1: recommended Referrer-Policy, plus the values that
# are strictly *more* private than it. Anything else is a finding.
ACCEPTABLE_REFERRER_POLICIES = {
    "strict-origin-when-cross-origin",
    "strict-origin",
    "same-origin",
    "no-referrer",
}

# security-and-hygiene.md §1: "camera=(), microphone=(), geolocation=(),
# payment=(), usb=() — deny by default"
EXPECTED_PERMISSIONS_FEATURES = ("camera", "microphone", "geolocation", "payment", "usb")

# security-and-hygiene.md §9 E, shared with code-quality.md §9.2
EXPOSURE_PROBE_PATHS = (
    "/.env",
    "/.env.production",
    "/.git/HEAD",
    "/.git/config",
    "/.DS_Store",
    "/backup.zip",
)

# security-and-hygiene.md §1: version-disclosing Server / any X-Powered-By
VERSION_DISCLOSING_SERVER = re.compile(r"\d+\.\d+")

# security-and-hygiene.md §5: cookie names that are almost certainly a session
SESSION_COOKIE_HINT = re.compile(
    r"sess|sid|auth|token|login|jwt|csrf|xsrf|remember", re.IGNORECASE
)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; website-audit/check_headers.py; +local audit tooling)"
)


# --------------------------------------------------------------------------- #
# Tiny HTTP layer (stdlib only)
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Response:
    """A single HTTP response, flattened into the bits the checks need."""

    url: str
    status: int
    headers: dict[str, str]          # lower-cased names, last value wins
    multi: dict[str, list[str]]      # lower-cased names, every value
    body: bytes

    def get(self, name: str, default: str = "") -> str:
        """Case-insensitive single-value header lookup."""
        return self.headers.get(name.lower(), default)

    def all(self, name: str) -> list[str]:
        """Case-insensitive multi-value header lookup (Set-Cookie, CSP)."""
        return self.multi.get(name.lower(), [])

    @property
    def text(self) -> str:
        """Body decoded leniently — we only ever grep it."""
        return self.body.decode("utf-8", errors="replace")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Redirect handler that stops instead of following, so we can count hops."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


def fetch(
    url: str,
    *,
    timeout: float,
    follow: bool = True,
    body_limit: int = 512_000,
    user_agent: str = DEFAULT_USER_AGENT,
    verify_tls: bool = True,
) -> Response:
    """Fetch *url* and return a :class:`Response`.

    Non-2xx statuses are returned rather than raised — a 404 on ``/.env`` is
    the *pass* condition, not an error.

    Raises:
        TargetUnreachable: DNS, TLS or connection-level failure.
    """
    handlers: list[Any] = []
    if not follow:
        handlers.append(_NoRedirect())
    if not verify_tls:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        handlers.append(urllib.request.HTTPSHandler(context=context))
    opener = urllib.request.build_opener(*handlers)

    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        raw: HTTPResponse = opener.open(request, timeout=timeout)
        return _to_response(url, raw, body_limit)
    except urllib.error.HTTPError as exc:
        # HTTPError *is* a response object; redirects land here when follow=False.
        return _to_response(url, exc, body_limit)
    except urllib.error.URLError as exc:
        raise TargetUnreachable(url, str(exc.reason)) from exc
    except (TimeoutError, OSError, ssl.SSLError) as exc:
        raise TargetUnreachable(url, str(exc)) from exc


def _to_response(url: str, raw: Any, body_limit: int) -> Response:
    """Normalise anything urllib hands back into a :class:`Response`."""
    headers: dict[str, str] = {}
    multi: dict[str, list[str]] = {}
    for name, value in raw.headers.items():
        key = name.lower()
        headers[key] = value
        multi.setdefault(key, []).append(value)
    try:
        body = raw.read(body_limit)
    except Exception:  # noqa: BLE001 - a truncated body is still useful
        body = b""
    finally:
        raw.close()
    return Response(url=getattr(raw, "url", url) or url,
                    status=raw.status if hasattr(raw, "status") else raw.code,
                    headers=headers, multi=multi, body=body)


# --------------------------------------------------------------------------- #
# CSP parsing
# --------------------------------------------------------------------------- #


def parse_csp(value: str) -> dict[str, list[str]]:
    """Parse a CSP header into ``{directive: [source, ...]}``.

    Directive names are lower-cased; source expressions keep their case because
    nonces and hashes are case-sensitive.
    """
    policy: dict[str, list[str]] = {}
    for chunk in value.split(";"):
        parts = chunk.strip().split()
        if not parts:
            continue
        policy[parts[0].lower()] = parts[1:]
    return policy


def effective_sources(policy: dict[str, list[str]], directive: str) -> list[str] | None:
    """Return the sources for *directive*, falling back to ``default-src``.

    ``None`` means neither the directive nor ``default-src`` is present, i.e.
    the directive is entirely unconstrained.
    """
    if directive in policy:
        return policy[directive]
    return policy.get("default-src")


def has_nonce_or_hash(sources: Iterable[str]) -> bool:
    """True when the source list carries a nonce or a hash.

    This is the check that stops the script from false-flagging the realistic
    policy in ``security-and-hygiene.md`` §3.2, where ``'unsafe-inline'`` sits
    *inside* a nonce'd ``script-src`` purely as an old-browser fallback.
    """
    return any(
        source.strip("'").lower().startswith(("nonce-", "sha256-", "sha384-", "sha512-"))
        for source in sources
    )


NONCE_RE = re.compile(r"nonce-([A-Za-z0-9+/=_-]+)")


def extract_nonces(value: str) -> list[str]:
    """Pull every nonce value out of a raw CSP header string."""
    return NONCE_RE.findall(value)


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def check_https_redirect(report: Report, origin: str, timeout: float,
                         verify_tls: bool) -> None:
    """HTTP must 301 to HTTPS in at most one hop (§2)."""
    parsed = urlparse(origin)
    http_url = urlunparse(("http", parsed.netloc, parsed.path or "/", "", "", ""))
    try:
        first = fetch(http_url, timeout=timeout, follow=False, verify_tls=verify_tls)
    except TargetUnreachable as exc:
        report.note(f"HTTP redirect check skipped: {exc.reason}")
        return

    location = first.get("location")
    if first.status not in (301, 302, 307, 308):
        report.add(Finding(
            id="security.transport.no-https-redirect",
            severity="blocker", rule="https-redirect", wcag_sc=None, url=http_url,
            viewport=None, selector=None, section="transport",
            message=(f"Plain HTTP returned {first.status} instead of redirecting to "
                     f"HTTPS. Traffic can be served, and stripped, over cleartext."),
            evidence={"status": first.status, "location": location or None,
                      "expected": "301 to https://"},
            how_to_fix="Return a 301 from every http:// URL to the identical https:// "
                       "URL, preserving the path, then add HSTS.",
        ))
        return

    if not location.lower().startswith("https://"):
        report.add(Finding(
            id="security.transport.redirect-not-https",
            severity="blocker", rule="https-redirect", wcag_sc=None, url=http_url,
            viewport=None, selector=None, section="transport",
            message="The HTTP redirect does not point at an https:// URL.",
            evidence={"status": first.status, "location": location},
            how_to_fix="Point the redirect at the https:// form of the same path.",
        ))
        return

    if first.status != 301:
        # §2: "A 302 here is a finding"
        report.add(Finding(
            id="security.transport.redirect-not-permanent",
            severity="minor", rule="https-redirect", wcag_sc=None, url=http_url,
            viewport=None, selector=None, section="transport",
            message=f"HTTP redirects to HTTPS with a {first.status}, not a 301.",
            evidence={"status": first.status, "expected": 301},
            how_to_fix="Use 301 (or 308) so the redirect is cached and the signal "
                       "consolidates.",
        ))

    # §2: "at most one hop"
    try:
        second = fetch(location, timeout=timeout, follow=False, verify_tls=verify_tls)
    except TargetUnreachable:
        return
    if second.status in (301, 302, 307, 308):
        report.add(Finding(
            id="security.transport.redirect-chain",
            severity="minor", rule="redirect-chain", wcag_sc=None, url=http_url,
            viewport=None, selector=None, section="transport",
            message="HTTP to HTTPS takes more than one hop.",
            evidence={"hop1": location, "hop2": second.get("location"),
                      "threshold": "at most one hop"},
            how_to_fix="Collapse the protocol and host canonicalisation into a single "
                       "redirect rule.",
        ))


def check_hsts(report: Report, response: Response) -> None:
    """HSTS present, long, and scoped (§2)."""
    value = response.get("strict-transport-security")
    if not value:
        report.add(Finding(
            id="security.headers.hsts-missing",
            severity="major", rule="hsts", wcag_sc=None, url=response.url,
            viewport=None, selector=None, section="transport",
            message="No Strict-Transport-Security header. HTTPS is not enforced for "
                    "return visits.",
            evidence={"header": None,
                      "recommended": "max-age=31536000; includeSubDomains"},
            how_to_fix="Add `Strict-Transport-Security: max-age=31536000; "
                       "includeSubDomains`. Verify every subdomain is HTTPS-capable "
                       "first — this is irreversible for the max-age duration.",
        ))
        return

    match = re.search(r"max-age\s*=\s*(\d+)", value, re.IGNORECASE)
    max_age = int(match.group(1)) if match else 0
    if max_age < HSTS_MIN_MAX_AGE:
        report.add(Finding(
            id="security.headers.hsts-short",
            severity="minor", rule="hsts", wcag_sc=None, url=response.url,
            viewport=None, selector=None, section="transport",
            message=f"HSTS max-age is {max_age}s, below the one-year baseline.",
            # security-and-hygiene.md §2: "max-age >= 31536000"
            evidence={"max_age": max_age, "threshold": HSTS_MIN_MAX_AGE},
            how_to_fix="Raise max-age to 31536000 once you are confident in the TLS "
                       "setup for every host it covers.",
        ))
    if "includesubdomains" not in value.lower():
        report.add(Finding(
            id="security.headers.hsts-no-subdomains",
            severity="advisory", rule="hsts", wcag_sc=None, url=response.url,
            viewport=None, selector=None, section="transport",
            message="HSTS does not include subdomains.",
            evidence={"header": value},
            how_to_fix="Add `includeSubDomains` — but only after verifying every "
                       "subdomain, including internal ones, serves HTTPS.",
        ))


def check_csp(report: Report, response: Response, second: Response | None) -> str:
    """Grade the Content-Security-Policy on effect, not presence (§3).

    Returns:
        A short label used in the overall grade: ``"none"``, ``"report-only"``,
        ``"unsafe-inline"``, ``"allowlist"`` or ``"strict"``.
    """
    enforcing = response.get("content-security-policy")
    report_only = response.get("content-security-policy-report-only")

    if not enforcing and not report_only:
        report.add(Finding(
            id="security.csp.absent",
            severity="major", rule="csp", wcag_sc=None, url=response.url,
            viewport=None, selector=None, section="csp",
            message="No Content-Security-Policy. Nothing constrains which scripts the "
                    "browser will execute.",
            evidence={"header": None},
            how_to_fix="Start with Content-Security-Policy-Report-Only carrying a "
                       "nonce + strict-dynamic policy and a report endpoint; run it "
                       "2-4 weeks, triage, then switch to enforcing.",
        ))
        return "none"

    if not enforcing and report_only:
        # §3.3: Report-Only alone with no stated end date is a stalled rollout.
        report.add(Finding(
            id="security.csp.report-only-only",
            severity="minor", rule="csp", wcag_sc=None, url=response.url,
            viewport=None, selector=None, section="csp",
            message="CSP exists only in Report-Only mode, so nothing is actually "
                    "blocked.",
            evidence={"content-security-policy-report-only": report_only[:300]},
            how_to_fix="Triage the collected violations and switch the header name to "
                       "the enforcing form, or record a dated plan to do so.",
        ))
        return "report-only"

    policy = parse_csp(enforcing)
    script_src = effective_sources(policy, "script-src") or []
    nonced = has_nonce_or_hash(script_src)
    grade = "strict" if nonced else "allowlist"

    # §3.1 — the single most important grading rule in this script.
    if any(s.strip("'").lower() == "unsafe-inline" for s in script_src) and not nonced:
        grade = "unsafe-inline"
        report.add(Finding(
            id="security.csp.unsafe-inline-without-nonce",
            severity="major", rule="csp-unsafe-inline", wcag_sc=None, url=response.url,
            viewport=None, selector=None, section="csp",
            message="script-src contains 'unsafe-inline' with no nonce or hash. The "
                    "browser will execute any injected inline script, so the policy "
                    "provides essentially zero XSS mitigation while still scoring on "
                    "header scanners.",
            evidence={"script-src": " ".join(script_src)},
            how_to_fix="Move to `script-src 'nonce-{PER_RESPONSE}' 'strict-dynamic' "
                       "https: 'unsafe-inline'`. Once a nonce is present browsers "
                       "ignore 'unsafe-inline', so it becomes a harmless fallback.",
        ))

    if any(s.strip("'").lower() == "unsafe-eval" for s in script_src):
        report.add(Finding(
            id="security.csp.unsafe-eval",
            severity="minor", rule="csp-unsafe-eval", wcag_sc=None, url=response.url,
            viewport=None, selector=None, section="csp",
            message="script-src allows 'unsafe-eval', re-enabling eval(), "
                    "new Function() and string setTimeout().",
            evidence={"script-src": " ".join(script_src)},
            how_to_fix="Find which dependency demands it and replace or configure it; "
                       "if it must stay, document the reason.",
        ))

    for directive, expected in (("object-src", "'none'"), ("base-uri", "'none'")):
        sources = policy.get(directive)
        if sources is None or expected not in [s.lower() for s in sources]:
            report.add(Finding(
                id=f"security.csp.{directive}",
                severity="minor", rule="csp-directive", wcag_sc=None, url=response.url,
                viewport=None, selector=None, section="csp",
                message=f"CSP does not set `{directive} {expected}`.",
                evidence={"present": " ".join(sources) if sources else None,
                          "expected": expected},
                how_to_fix=f"Add `{directive} {expected}` — it costs nothing and "
                           f"closes a standard bypass.",
            ))

    if "form-action" not in policy:
        report.add(Finding(
            id="security.csp.form-action",
            severity="minor", rule="csp-directive", wcag_sc=None, url=response.url,
            viewport=None, selector=None, section="csp",
            message="CSP has no `form-action`, so an injected form can post anywhere.",
            evidence={"expected": "form-action 'self'"},
            how_to_fix="Add `form-action 'self'` (plus any genuine external post "
                       "targets).",
        ))

    connect = effective_sources(policy, "connect-src")
    if connect is None or "*" in connect:
        report.add(Finding(
            id="security.csp.connect-src-wildcard",
            severity="minor", rule="csp-directive", wcag_sc=None, url=response.url,
            viewport=None, selector=None, section="csp",
            message="connect-src is missing or wildcarded, so exfiltration to any "
                    "origin is permitted.",
            evidence={"connect-src": " ".join(connect) if connect else None},
            how_to_fix="Enumerate the API and analytics origins the site genuinely "
                       "talks to.",
        ))

    # §3.2: a nonce cached with the HTML is equivalent to 'unsafe-inline'.
    if nonced and second is not None:
        first_nonces = extract_nonces(enforcing)
        second_nonces = extract_nonces(second.get("content-security-policy"))
        if first_nonces and first_nonces == second_nonces:
            report.add(Finding(
                id="security.csp.static-nonce",
                severity="major", rule="csp-nonce-reuse", wcag_sc=None,
                url=response.url, viewport=None, selector=None, section="csp",
                message="The CSP nonce is identical across two consecutive requests. "
                        "A static or CDN-cached nonce is equivalent to "
                        "'unsafe-inline'.",
                evidence={"nonce_request_1": first_nonces,
                          "nonce_request_2": second_nonces},
                how_to_fix="Generate the nonce per response and either mark the HTML "
                           "uncacheable or inject the nonce at the edge.",
            ))
            grade = "unsafe-inline"

    if "frame-ancestors" not in policy and not response.get("x-frame-options"):
        report.add(Finding(
            id="security.headers.framing",
            severity="major", rule="framing", wcag_sc=None, url=response.url,
            viewport=None, selector=None, section="csp",
            message="Neither CSP frame-ancestors nor X-Frame-Options is set; the page "
                    "can be framed for clickjacking.",
            evidence={"frame-ancestors": None, "x-frame-options": None},
            how_to_fix="Add `frame-ancestors 'none'` to the CSP (it supersedes XFO) "
                       "and keep `X-Frame-Options: DENY` for very old agents.",
        ))

    return grade


def check_simple_headers(report: Report, response: Response) -> None:
    """The one-line headers from the §1 baseline table."""
    if response.get("x-content-type-options").strip().lower() != "nosniff":
        report.add(Finding(
            id="security.headers.nosniff",
            severity="minor", rule="x-content-type-options", wcag_sc=None,
            url=response.url, viewport=None, selector=None, section="headers",
            message="X-Content-Type-Options is not `nosniff`.",
            evidence={"header": response.get("x-content-type-options") or None},
            how_to_fix="Add `X-Content-Type-Options: nosniff`. One line; its absence "
                       "means nobody ever looked.",
        ))

    referrer = response.get("referrer-policy").strip().lower()
    if not referrer:
        report.add(Finding(
            id="security.headers.referrer-policy-missing",
            severity="minor", rule="referrer-policy", wcag_sc=None, url=response.url,
            viewport=None, selector=None, section="headers",
            message="No Referrer-Policy, so full URLs leak to third parties.",
            evidence={"recommended": "strict-origin-when-cross-origin"},
            how_to_fix="Add `Referrer-Policy: strict-origin-when-cross-origin`.",
        ))
    elif not any(p.strip() in ACCEPTABLE_REFERRER_POLICIES for p in referrer.split(",")):
        report.add(Finding(
            id="security.headers.referrer-policy-weak",
            severity="advisory", rule="referrer-policy", wcag_sc=None, url=response.url,
            viewport=None, selector=None, section="headers",
            message=f"Referrer-Policy is `{referrer}`, weaker than the baseline.",
            evidence={"header": referrer,
                      "acceptable": sorted(ACCEPTABLE_REFERRER_POLICIES)},
            how_to_fix="Use `strict-origin-when-cross-origin` or stricter.",
        ))

    permissions = response.get("permissions-policy")
    if not permissions:
        report.add(Finding(
            id="security.headers.permissions-policy-missing",
            severity="minor", rule="permissions-policy", wcag_sc=None, url=response.url,
            viewport=None, selector=None, section="headers",
            message="No Permissions-Policy, so embedded third parties inherit every "
                    "powerful feature.",
            evidence={"recommended":
                      "camera=(), microphone=(), geolocation=(), payment=(), usb=()"},
            how_to_fix="Add a deny-by-default Permissions-Policy and re-enable only "
                       "what the site uses.",
        ))
    else:
        undenied = [
            feature for feature in EXPECTED_PERMISSIONS_FEATURES
            if not re.search(rf"\b{feature}\s*=\s*\(\s*\)", permissions)
        ]
        if undenied:
            report.add(Finding(
                id="security.headers.permissions-policy-not-deny-by-default",
                severity="advisory", rule="permissions-policy", wcag_sc=None,
                url=response.url, viewport=None, selector=None, section="headers",
                message="Permissions-Policy is present but does not deny "
                        + ", ".join(undenied) + ".",
                evidence={"header": permissions, "not_denied": undenied},
                how_to_fix="Deny every feature the site does not use, using the "
                           "empty allowlist form `feature=()`.",
            ))

    if response.get("cross-origin-opener-policy").strip().lower() != "same-origin":
        report.add(Finding(
            id="security.headers.coop",
            severity="advisory", rule="coop", wcag_sc=None, url=response.url,
            viewport=None, selector=None, section="headers",
            message="Cross-Origin-Opener-Policy is not `same-origin`.",
            evidence={"header": response.get("cross-origin-opener-policy") or None},
            how_to_fix="Add `Cross-Origin-Opener-Policy: same-origin`; it also unlocks "
                       "crossOriginIsolated features.",
        ))

    if not response.get("cross-origin-resource-policy"):
        report.add(Finding(
            id="security.headers.corp",
            severity="advisory", rule="corp", wcag_sc=None, url=response.url,
            viewport=None, selector=None, section="headers",
            message="No Cross-Origin-Resource-Policy.",
            evidence={"recommended": "same-origin (cross-origin for public CDN assets)"},
            how_to_fix="Set `Cross-Origin-Resource-Policy` appropriately for the asset "
                       "class.",
        ))

    if response.get("x-powered-by"):
        report.add(Finding(
            id="security.headers.x-powered-by",
            severity="minor", rule="version-disclosure", wcag_sc=None, url=response.url,
            viewport=None, selector=None, section="headers",
            message="X-Powered-By discloses the server stack.",
            evidence={"x-powered-by": response.get("x-powered-by")},
            how_to_fix="Remove the header at the app or proxy layer.",
        ))

    server = response.get("server")
    if server and VERSION_DISCLOSING_SERVER.search(server):
        report.add(Finding(
            id="security.headers.server-version",
            severity="minor", rule="version-disclosure", wcag_sc=None, url=response.url,
            viewport=None, selector=None, section="headers",
            message=f"Server header discloses a version: `{server}`.",
            evidence={"server": server},
            how_to_fix="Suppress the version token (`server_tokens off` in nginx, or "
                       "equivalent).",
        ))


def check_caching(report: Report, response: Response) -> None:
    """HTML caching rules from performance.md §3.8, referenced by §6.1."""
    cache_control = response.get("cache-control").lower()
    if "no-store" in cache_control:
        report.add(Finding(
            id="security.caching.html-no-store",
            severity="minor", rule="cache-control", wcag_sc=None, url=response.url,
            viewport=None, selector=None, section="caching",
            message="HTML is served with `no-store`, which Firefox and Safari treat "
                    "as a hard back/forward-cache blocker.",
            # performance.md §2.3 / §3.8, quoted in security-and-hygiene.md §7.2
            evidence={"cache-control": response.get("cache-control"),
                      "recommended": "no-cache (revalidate), never no-store"},
            how_to_fix="Use `Cache-Control: no-cache` (or `private, no-cache` + ETag "
                       "for personalised responses) so bfcache stays available.",
        ))
    elif not cache_control:
        report.add(Finding(
            id="security.caching.html-unset",
            severity="advisory", rule="cache-control", wcag_sc=None, url=response.url,
            viewport=None, selector=None, section="caching",
            message="No Cache-Control on the HTML response; caching is left to "
                    "heuristics.",
            evidence={"recommended": "no-cache, or a short max-age + "
                                     "stale-while-revalidate"},
            how_to_fix="Set an explicit HTML caching policy.",
        ))


def check_cookies(report: Report, response: Response) -> None:
    """Cookie flags from §5."""
    for raw in response.all("set-cookie"):
        name = raw.split("=", 1)[0].strip()
        attributes = {part.strip().lower() for part in raw.split(";")[1:]}
        flat = " ".join(attributes)
        secure = "secure" in attributes
        http_only = "httponly" in attributes
        same_site = next((a.split("=", 1)[1].strip()
                          for a in attributes if a.startswith("samesite=")), None)
        looks_like_session = bool(SESSION_COOKIE_HINT.search(name))

        if "samesite=none" in flat and not secure:
            report.add(Finding(
                id=f"security.cookies.samesite-none-insecure.{name}",
                severity="blocker", rule="cookie-flags", wcag_sc=None, url=response.url,
                viewport=None, selector=None, section="cookies",
                message=f"Cookie `{name}` sets SameSite=None without Secure; browsers "
                        f"reject it and the intent is cross-site anyway.",
                evidence={"set-cookie": raw[:200]},
                how_to_fix="Add `Secure`. SameSite=None requires it.",
            ))
        if looks_like_session and not http_only:
            report.add(Finding(
                id=f"security.cookies.no-httponly.{name}",
                severity="major", rule="cookie-flags", wcag_sc=None, url=response.url,
                viewport=None, selector=None, section="cookies",
                message=f"Session-looking cookie `{name}` has no HttpOnly flag, so any "
                        f"injected script can read it.",
                evidence={"set-cookie": raw[:200], "name_matched": name},
                how_to_fix="Add `HttpOnly` to every session/auth cookie. Confirm in "
                           "DevTools > Application > Cookies, which is the only "
                           "reliable place to read the flag.",
            ))
        if looks_like_session and not secure:
            report.add(Finding(
                id=f"security.cookies.no-secure.{name}",
                severity="major", rule="cookie-flags", wcag_sc=None, url=response.url,
                viewport=None, selector=None, section="cookies",
                message=f"Session-looking cookie `{name}` has no Secure flag.",
                evidence={"set-cookie": raw[:200]},
                how_to_fix="Add `Secure` so the cookie is never sent over cleartext.",
            ))
        if same_site is None:
            report.add(Finding(
                id=f"security.cookies.no-samesite.{name}",
                severity="advisory", rule="cookie-flags", wcag_sc=None, url=response.url,
                viewport=None, selector=None, section="cookies",
                message=f"Cookie `{name}` has no SameSite attribute.",
                evidence={"set-cookie": raw[:200]},
                how_to_fix="Set `SameSite=Lax` (or `Strict` for CSRF tokens).",
            ))


MIXED_CONTENT_RE = re.compile(r'(?:src|href)\s*=\s*["\'](http://[^"\']+)', re.IGNORECASE)
SUBRESOURCE_RE = re.compile(
    r"<(script|link)\b[^>]*\b(?:src|href)\s*=\s*[\"'](https?://[^\"']+)[\"'][^>]*>",
    re.IGNORECASE,
)
SOURCEMAP_RE = re.compile(r"//[#@]\s*sourceMappingURL=([^\s*]+)")


def check_markup(report: Report, response: Response, origin_host: str) -> None:
    """Mixed content and Subresource Integrity, read out of the served HTML (§2, §4)."""
    if response.url.startswith("https://"):
        mixed = sorted({m for m in MIXED_CONTENT_RE.findall(response.text)})[:10]
        if mixed:
            report.add(Finding(
                id="security.transport.mixed-content",
                severity="blocker", rule="mixed-content", wcag_sc=None, url=response.url,
                viewport=None, selector=None, section="transport",
                message=f"{len(mixed)} subresource reference(s) use http:// on an "
                        f"HTTPS page.",
                evidence={"examples": mixed},
                how_to_fix="Serve every subresource over HTTPS and add "
                           "`upgrade-insecure-requests` to the CSP as a backstop.",
            ))

    missing_sri: list[str] = []
    for tag, url in SUBRESOURCE_RE.findall(response.text):
        if urlparse(url).hostname in (None, origin_host):
            continue
        full_tag_match = re.search(
            rf"<{tag}\b[^>]*{re.escape(url)}[^>]*>", response.text, re.IGNORECASE
        )
        if full_tag_match and "integrity=" not in full_tag_match.group(0).lower():
            missing_sri.append(url)
    if missing_sri:
        report.add(Finding(
            id="security.subresources.no-sri",
            severity="minor", rule="subresource-integrity", wcag_sc=None,
            url=response.url, viewport=None, selector=None, section="subresources",
            message=f"{len(missing_sri)} cross-origin script/stylesheet(s) load without "
                    f"an `integrity` attribute.",
            evidence={"examples": sorted(set(missing_sri))[:10]},
            how_to_fix="Add `integrity` + `crossorigin=\"anonymous\"`. Where the "
                       "resource legitimately changes per request (tag managers, "
                       "versionless CDN URLs), the correct fix is to pin the version "
                       "or self-host, and to document the accepted risk.",
        ))


def probe_exposed_paths(report: Report, origin: str, timeout: float,
                        verify_tls: bool) -> None:
    """Every path in §8 E must return 404."""
    for path in EXPOSURE_PROBE_PATHS:
        url = urljoin(origin, path)
        try:
            response = fetch(url, timeout=timeout, follow=False, body_limit=4096,
                             verify_tls=verify_tls)
        except TargetUnreachable:
            continue
        if response.status != 200 or not response.body.strip():
            continue
        # A SPA that returns its index.html for everything is a false positive.
        if b"<html" in response.body[:400].lower():
            continue
        report.add(Finding(
            id=f"security.exposure{path.replace('/', '.')}",
            severity="blocker", rule="exposed-path", wcag_sc=None, url=url,
            viewport=None, selector=None, section="exposure",
            message=f"`{path}` is publicly readable and returned content.",
            evidence={"status": response.status,
                      "first_bytes": response.body[:120].decode("utf-8", "replace")},
            how_to_fix="Block dot-files and archive files at the web server or CDN, "
                       "and rotate anything that was exposed.",
        ))


def probe_source_maps(report: Report, response: Response, timeout: float,
                      verify_tls: bool, limit: int = 6) -> None:
    """Follow the first few script URLs and see whether their .map files are live."""
    scripts = [
        url for tag, url in SUBRESOURCE_RE.findall(response.text) if tag.lower() == "script"
    ][:limit]
    reachable: list[str] = []
    for script_url in scripts:
        try:
            script = fetch(script_url, timeout=timeout, body_limit=200_000,
                           verify_tls=verify_tls)
        except TargetUnreachable:
            continue
        match = SOURCEMAP_RE.search(script.text)
        if not match:
            continue
        map_url = urljoin(script_url, match.group(1))
        if map_url.startswith("data:"):
            continue
        try:
            map_response = fetch(map_url, timeout=timeout, body_limit=4096,
                                 verify_tls=verify_tls)
        except TargetUnreachable:
            continue
        if map_response.status == 200 and b'"sources"' in map_response.body:
            reachable.append(map_url)
    if reachable:
        report.add(Finding(
            id="security.exposure.source-maps",
            severity="minor", rule="source-maps", wcag_sc=None, url=response.url,
            viewport=None, selector=None, section="exposure",
            message=f"{len(reachable)} source map(s) are publicly reachable, exposing "
                    f"original source.",
            evidence={"maps": reachable},
            how_to_fix="Stop publishing .map files, or upload them to the error "
                       "tracker and block them at the CDN.",
        ))


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #


def overall_grade(report: Report, csp_grade: str) -> str:
    """Derive a letter grade weighted towards policy effect, not header count.

    The CSP is the dominant term deliberately: it is the only header in the
    baseline whose presence requires knowing every script on the page.
    """
    counts = report.counts()
    if counts["blocker"]:
        return "F"
    base = {"strict": 95, "allowlist": 80, "unsafe-inline": 55,
            "report-only": 50, "none": 45}[csp_grade]
    score = base - 6 * counts["major"] - 3 * counts["minor"] - 1 * counts["advisory"]
    for threshold, letter in ((90, "A"), (80, "B"), (70, "C"), (60, "D"), (0, "E")):
        if score >= threshold:
            return letter
    return "F"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def build_parser():
    """Construct the argument parser."""
    parser = base_parser(
        "check_headers.py",
        "Grade security, caching and hygiene headers. Standard library only.",
        epilog=(
            "Thresholds come from references/security-and-hygiene.md.\n"
            "Header configuration often differs between CDN-cached marketing pages "
            "and origin-served application routes — pass --path at least once so at "
            "least two templates are checked."
        ),
    )
    parser.add_argument(
        "--path", action="append", default=[], metavar="PATH",
        help="Additional path on the same origin to check (repeatable). "
             "Check at least one CDN-cached and one origin-served template.",
    )
    parser.add_argument(
        "--no-probe", action="store_true",
        help="Skip the exposed-path and source-map probes (they issue extra requests)",
    )
    parser.add_argument(
        "--insecure", action="store_true",
        help="Do not verify TLS certificates (for staging hosts with self-signed certs)",
    )
    return parser


def main() -> int:
    """Run the header audit and return the process exit code."""
    args = build_parser().parse_args()
    origin = normalise_url(args.url)
    host = urlparse(origin).hostname or ""

    report = Report(tool="check_headers", target=origin)
    report.meta["dependencies"] = "none (standard library only)"

    check_https_redirect(report, origin, args.timeout, not args.insecure)

    targets = [origin] + [urljoin(origin, path) for path in args.path]
    for index, url in enumerate(targets):
        response = fetch(url, timeout=args.timeout, verify_tls=not args.insecure)
        # Two consecutive fetches of the first template so the nonce can be diffed.
        second = None
        if index == 0 and response.get("content-security-policy"):
            try:
                second = fetch(url, timeout=args.timeout, body_limit=1,
                               verify_tls=not args.insecure)
            except TargetUnreachable:
                second = None

        csp_grade = check_csp(report, response, second)
        check_hsts(report, response)
        check_simple_headers(report, response)
        check_caching(report, response)
        check_cookies(report, response)
        check_markup(report, response, host)
        if index == 0:
            report.meta["csp_policy_grade"] = csp_grade
            if not args.no_probe:
                probe_exposed_paths(report, origin, args.timeout, not args.insecure)
                probe_source_maps(report, response, args.timeout, not args.insecure)

    if len(targets) < 2:
        report.note(
            "Only one template was checked. Header configuration commonly differs "
            "between CDN-cached and origin-served routes; re-run with --path to cover "
            "at least one of each."
        )
    report.note(
        "This script grades headers only. It does not evaluate TLS versions, ciphers "
        "or certificate chains (use testssl.sh), does not read HttpOnly cookies set "
        "by JavaScript, and does not detect CSP violations at runtime (watch the "
        "console, or listen for securitypolicyviolation events)."
    )
    report.note(
        "A host-allowlist CSP still needs a manual pass through Google's CSP "
        "Evaluator for JSONP endpoints and arbitrary-path library CDNs; this script "
        "cannot see those."
    )

    report.meta["grade"] = overall_grade(report, report.meta.get("csp_policy_grade", "none"))
    report.meta["templates_checked"] = len(targets)
    return finish(report, args)


if __name__ == "__main__":
    run_cli(main)
