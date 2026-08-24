"""Shared plumbing for the website-audit scripts.

Everything in this module is **standard library only** so that
``check_headers.py`` — which must run on a machine with nothing installed —
can import it safely.

The three things every audit script shares:

1. :class:`Finding` — the single report schema. Every script emits findings in
   exactly this shape so that a downstream report generator never has to
   special-case a tool.
2. :class:`Report` — collects findings, prints a human summary, writes machine
   JSON, and derives the process exit code.
3. Graceful failure — :class:`MissingDependency` / :class:`TargetUnreachable`
   plus :func:`run_cli`, which turn "playwright isn't installed" into a one
   paragraph explanation and a distinct exit code rather than a traceback.

Exit codes (identical across all six scripts):

===== =========================================================
Code  Meaning
===== =========================================================
0     Ran cleanly; nothing at or above the ``--fail-on`` severity
1     Ran cleanly; findings at or above ``--fail-on`` (or a budget breach)
2     Bad command line (argparse's own convention)
3     A required dependency is missing — stdout says which and how to install
4     The target could not be reached / a required network resource failed
5     An unexpected internal error, reported without a traceback
===== =========================================================
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

# --------------------------------------------------------------------------- #
# Exit codes
# --------------------------------------------------------------------------- #

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_USAGE = 2
EXIT_MISSING_DEPENDENCY = 3
EXIT_TARGET_UNREACHABLE = 4
EXIT_INTERNAL_ERROR = 5


# --------------------------------------------------------------------------- #
# Severity
# --------------------------------------------------------------------------- #

#: The only four severities any script may emit, most severe first.
SEVERITIES: tuple[str, ...] = ("blocker", "major", "minor", "advisory")

_SEVERITY_RANK = {name: index for index, name in enumerate(SEVERITIES)}


def severity_rank(severity: str) -> int:
    """Return a sortable rank for *severity* (0 = most severe)."""
    try:
        return _SEVERITY_RANK[severity]
    except KeyError:  # pragma: no cover - guards against typos in new rules
        raise ValueError(
            f"unknown severity {severity!r}; expected one of {', '.join(SEVERITIES)}"
        ) from None


def at_or_above(severity: str, threshold: str) -> bool:
    """True when *severity* is as severe as, or more severe than, *threshold*."""
    return severity_rank(severity) <= severity_rank(threshold)


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Finding:
    """One audit finding.

    The field list is the report schema shared by every script in this
    directory. ``wcag_sc``, ``viewport``, ``selector`` and ``section`` are
    nullable because not every check has them, but they are never omitted from
    the JSON — a consumer can rely on the keys existing.

    Args:
        id: Stable, machine-readable identifier for this *instance*, e.g.
            ``responsive.overflow.section-pricing.v1``. Used for dedup.
        severity: One of :data:`SEVERITIES`.
        rule: The check that produced it, e.g. ``horizontal-overflow`` or the
            upstream engine's rule id (``color-contrast``).
        wcag_sc: Dotted WCAG Success Criterion number with level, e.g.
            ``"1.4.10 (AA)"``. ``None`` when the finding maps to no criterion —
            an honest ``None`` is better than a stretched citation.
        url: The page the finding was observed on.
        viewport: Emulation context, e.g. ``"320x512 @2x isMobile"``.
        selector: A CSS-ish path to the offending element.
        section: The nearest identifiable section/landmark. A page-level
            finding nobody can locate is not actionable.
        message: One sentence stating what was measured and why it fails.
        evidence: Measured values, thresholds, screenshots — the numbers that
            turn an opinion into a violation.
        how_to_fix: Concrete remediation, not "improve accessibility".
    """

    id: str
    severity: str
    rule: str
    wcag_sc: str | None
    url: str
    viewport: str | None
    selector: str | None
    section: str | None
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)
    how_to_fix: str = ""

    def __post_init__(self) -> None:
        severity_rank(self.severity)  # validate eagerly, fail at the call site

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the wire format (all keys always present)."""
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Report:
    """Accumulates findings and renders them.

    ``notes`` carries the honesty layer: coverage caveats, checks that were
    skipped, and — critically for :mod:`audit_a11y` — the count of things the
    automated engine could not test. A report that omits this implies the scan
    was complete.
    """

    tool: str
    target: str
    findings: list[Finding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    # -- collection ------------------------------------------------------- #

    def add(self, finding: Finding) -> None:
        """Record one finding."""
        self.findings.append(finding)

    def extend(self, findings: Iterable[Finding]) -> None:
        """Record many findings."""
        for finding in findings:
            self.add(finding)

    def note(self, text: str) -> None:
        """Record a coverage caveat that belongs in the written report."""
        self.notes.append(text)

    # -- derived ---------------------------------------------------------- #

    def counts(self) -> dict[str, int]:
        """Finding count per severity, including zeroes."""
        counts = {name: 0 for name in SEVERITIES}
        for finding in self.findings:
            counts[finding.severity] += 1
        return counts

    def worst_severity(self) -> str | None:
        """The most severe severity present, or ``None`` for a clean run."""
        if not self.findings:
            return None
        return min((f.severity for f in self.findings), key=severity_rank)

    def exit_code(self, fail_on: str) -> int:
        """``EXIT_FINDINGS`` if anything reaches *fail_on*, else ``EXIT_OK``."""
        if any(at_or_above(f.severity, fail_on) for f in self.findings):
            return EXIT_FINDINGS
        return EXIT_OK

    # -- output ----------------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        """The full machine-readable document written by ``--json``."""
        return {
            "tool": self.tool,
            "target": self.target,
            "started_at": self.started_at,
            "schema": "website-audit/finding@1",
            "meta": self.meta,
            "counts": self.counts(),
            "notes": self.notes,
            "findings": [f.to_dict() for f in self.findings],
        }

    def write_json(self, path: str | Path) -> None:
        """Write the JSON document to *path*, creating parent directories."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )

    def print_summary(self, stream: Any = None, show_all: bool = False) -> None:
        """Print a readable summary, most severe first.

        Args:
            stream: Where to write. Defaults to ``sys.stdout``.
            show_all: Print every finding. By default advisory findings are
                collapsed to a count so the terminal output stays scannable.
        """
        out = stream or sys.stdout
        counts = self.counts()
        print(f"\n{self.tool} — {self.target}", file=out)
        print("=" * min(78, len(self.tool) + len(self.target) + 3), file=out)

        for key, value in self.meta.items():
            print(f"  {key}: {value}", file=out)
        if self.meta:
            print(file=out)

        summary = "  ".join(f"{name}: {counts[name]}" for name in SEVERITIES)
        print(f"Findings — {summary}", file=out)

        for severity in SEVERITIES:
            bucket = [f for f in self.findings if f.severity == severity]
            if not bucket:
                continue
            if severity == "advisory" and not show_all:
                print(f"\n[advisory] {len(bucket)} advisory finding(s) "
                      f"— use --show-all or --json to read them", file=out)
                continue
            print(f"\n--- {severity.upper()} ({len(bucket)}) ---", file=out)
            for finding in bucket:
                _print_finding(finding, out)

        if self.notes:
            print("\n--- COVERAGE NOTES ---", file=out)
            for note in self.notes:
                for line in textwrap.wrap(note, width=78,
                                          initial_indent="  * ",
                                          subsequent_indent="    "):
                    print(line, file=out)
        print(file=out)


def _print_finding(finding: Finding, out: Any) -> None:
    """Render a single finding as an indented block."""
    location = finding.section or finding.selector or "page"
    header = f"  [{finding.rule}] {location}"
    if finding.viewport:
        header += f"  @ {finding.viewport}"
    print(header, file=out)
    for line in textwrap.wrap(finding.message, width=76,
                              initial_indent="      ", subsequent_indent="      "):
        print(line, file=out)
    if finding.wcag_sc:
        print(f"      WCAG SC {finding.wcag_sc}", file=out)
    if finding.selector and finding.section:
        print(f"      selector: {finding.selector}", file=out)
    if finding.evidence:
        rendered = ", ".join(f"{k}={v!r}" for k, v in list(finding.evidence.items())[:6])
        for line in textwrap.wrap(rendered, width=76,
                                  initial_indent="      evidence: ",
                                  subsequent_indent="        "):
            print(line, file=out)
    if finding.how_to_fix:
        for line in textwrap.wrap(finding.how_to_fix, width=76,
                                  initial_indent="      fix: ",
                                  subsequent_indent="        "):
            print(line, file=out)


# --------------------------------------------------------------------------- #
# Graceful failure
# --------------------------------------------------------------------------- #


class AuditError(Exception):
    """Base class for errors that must not surface as a traceback."""

    exit_code = EXIT_INTERNAL_ERROR


class MissingDependency(AuditError):
    """A required package or browser binary is not available."""

    exit_code = EXIT_MISSING_DEPENDENCY

    def __init__(self, what: str, install: str) -> None:
        super().__init__(what)
        self.what = what
        self.install = install

    def explain(self) -> str:
        return (
            f"Missing dependency: {self.what}\n\n"
            f"Install it with:\n\n"
            + textwrap.indent(self.install.strip(), "    ")
            + "\n"
        )


class TargetUnreachable(AuditError):
    """The site under audit could not be fetched."""

    exit_code = EXIT_TARGET_UNREACHABLE

    def __init__(self, url: str, reason: str) -> None:
        super().__init__(f"{url}: {reason}")
        self.url = url
        self.reason = reason

    def explain(self) -> str:
        return f"Could not reach {self.url}\n  {self.reason}\n"


def import_playwright() -> Any:
    """Return ``playwright.sync_api.sync_playwright``.

    Raises:
        MissingDependency: If the package is not installed. The message names
            both the pip install and the ``playwright install chromium`` step,
            because forgetting the second one is the usual failure.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise MissingDependency(
            "playwright (Python package)",
            "pip install -r scripts/requirements.txt\nplaywright install chromium",
        ) from exc
    return sync_playwright


def launch_chromium(playwright: Any, headless: bool = True,
                    extra_args: Sequence[str] = ()) -> Any:
    """Launch Chromium, converting a missing browser binary into advice.

    Args:
        playwright: An active ``sync_playwright()`` context.
        headless: Run headless. Frame-timing work should pass ``False`` —
            a headful browser has a real vsync-driven compositor
            (``references/animation-and-motion.md`` §8.7).
        extra_args: Additional Chromium command-line switches.
    """
    try:
        return playwright.chromium.launch(headless=headless, args=list(extra_args))
    except Exception as exc:  # playwright raises its own Error subclass
        message = str(exc)
        if "Executable doesn't exist" in message or "playwright install" in message:
            raise MissingDependency(
                "the Chromium browser binary Playwright drives",
                "playwright install chromium",
            ) from exc
        raise


# --------------------------------------------------------------------------- #
# CLI helpers
# --------------------------------------------------------------------------- #


def base_parser(prog: str, description: str, epilog: str = "") -> argparse.ArgumentParser:
    """Build an ``ArgumentParser`` carrying the conventions all six scripts share.

    Every script accepts ``url``, ``--json``, ``--fail-on``, ``--show-all``,
    ``--quiet`` and ``--timeout``.
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("url", help="URL to audit, including the scheme (https://...)")
    parser.add_argument(
        "--json", metavar="PATH", dest="json_path",
        help="Write the machine-readable report to PATH (JSON, shared finding schema)",
    )
    parser.add_argument(
        "--fail-on", choices=SEVERITIES, default="blocker",
        help="Exit 1 when a finding of this severity or worse is present (default: blocker)",
    )
    parser.add_argument(
        "--show-all", action="store_true",
        help="Print advisory findings in the terminal summary as well",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress the human summary; useful with --json in CI",
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0, metavar="SECONDS",
        help="Per-navigation timeout in seconds (default: 30)",
    )
    return parser


def normalise_url(raw: str) -> str:
    """Add a scheme if the user omitted it, and reject obvious nonsense."""
    url = raw.strip()
    if not url:
        raise AuditError("empty URL")
    if "://" not in url:
        url = "https://" + url
    if not url.startswith(("http://", "https://")):
        raise AuditError(f"unsupported URL scheme in {raw!r}; use http:// or https://")
    return url


def run_cli(main: Callable[[], int]) -> None:
    """Run *main*, converting known failures into messages and exit codes.

    This is the only place any of these scripts is allowed to terminate the
    process, which keeps the "never print a traceback" rule enforceable.
    """
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(EXIT_INTERNAL_ERROR)
    except AuditError as exc:
        explain = getattr(exc, "explain", None)
        print(explain() if explain else str(exc), file=sys.stderr)
        raise SystemExit(exc.exit_code)
    except Exception as exc:  # noqa: BLE001 - deliberate: no tracebacks for users
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print("This is a bug in the audit script, not a finding about the site.",
              file=sys.stderr)
        raise SystemExit(EXIT_INTERNAL_ERROR)


def finish(report: Report, args: argparse.Namespace) -> int:
    """Emit output according to *args* and return the process exit code."""
    if args.json_path:
        report.write_json(args.json_path)
    if not args.quiet:
        report.print_summary(show_all=args.show_all)
    return report.exit_code(args.fail_on)
