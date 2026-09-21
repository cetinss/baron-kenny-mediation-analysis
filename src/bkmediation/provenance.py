"""Computational provenance: versions, seed, git commit, platform, timing.

Reviewer 3 asked that the exact Python and package versions, the random seed
and the GitHub commit be reported so that a reader can reconstruct the
computational environment. `collect_provenance()` gathers them, and the
analysis entry points write the result next to their outputs.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import ALPHA, N_BOOTSTRAP, PROJECT_DIR, RANDOM_SEED

__all__ = ["collect_provenance", "provenance_markdown", "write_provenance",
           "TRACKED_PACKAGES"]

TRACKED_PACKAGES = (
    "numpy", "pandas", "scipy", "statsmodels", "matplotlib", "seaborn", "pytest",
    "pingouin",
)


def _package_versions() -> dict:
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = "not installed"
    return versions


def _git_info(repo: Path = PROJECT_DIR) -> dict:
    """Commit hash, branch and dirty flag; empty strings outside a checkout."""
    def run(args):
        try:
            out = subprocess.run(
                ["git", *args], cwd=str(repo), capture_output=True, text=True, timeout=10
            )
            return out.stdout.strip() if out.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            return ""

    commit = run(["rev-parse", "HEAD"])
    return {
        "commit": commit,
        "commit_short": commit[:8],
        "branch": run(["rev-parse", "--abbrev-ref", "HEAD"]),
        "remote": run(["config", "--get", "remote.origin.url"]),
        "dirty": bool(run(["status", "--porcelain"])),
        "last_commit_date": run(["log", "-1", "--format=%cI"]),
    }


def collect_provenance(extra: dict | None = None) -> dict:
    from . import __version__

    info = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bkmediation_version": __version__,
        "python": sys.version.split()[0],
        "python_full": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "analysis_settings": {
            "random_seed": RANDOM_SEED,
            "n_bootstrap": N_BOOTSTRAP,
            "alpha": ALPHA,
        },
        "git": _git_info(),
    }
    if extra:
        info.update(extra)
    return info


def provenance_markdown(info: dict | None = None) -> str:
    info = info or collect_provenance()
    git = info["git"]
    lines = [
        "# Computational Provenance",
        "",
        f"Generated: {info['generated_utc']} (UTC)",
        "",
        "## Environment",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| bkmediation version | {info['bkmediation_version']} |",
        f"| Python | {info['python']} |",
        f"| Platform | {info['platform']} |",
    ]
    for name, version in info["packages"].items():
        lines.append(f"| {name} | {version} |")
    lines += [
        "",
        "## Analysis settings",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Random seed | {info['analysis_settings']['random_seed']} |",
        f"| Bootstrap resamples | {info['analysis_settings']['n_bootstrap']:,} |",
        f"| Alpha | {info['analysis_settings']['alpha']} |",
        "",
        "## Repository state",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Commit | `{git['commit'] or 'n/a'}` |",
        f"| Branch | {git['branch'] or 'n/a'} |",
        f"| Remote | {git['remote'] or 'n/a'} |",
        f"| Working tree | {'modified (uncommitted changes present)' if git['dirty'] else 'clean'} |",
        f"| Last commit date | {git['last_commit_date'] or 'n/a'} |",
        "",
        ("Re-running `python -m bkmediation all` at this commit, with this seed "
         "and these package versions, reproduces every number in the reports and "
         "figures."
         if not git["dirty"] else
         "The working tree held uncommitted changes when this run was made, so "
         "the commit above identifies the last committed state rather than the "
         "exact code that produced these numbers. Re-run after committing to "
         "obtain a fully pinned provenance record."),
        "",
    ]
    return "\n".join(lines)


def write_provenance(out_dir, extra: dict | None = None) -> dict:
    """Write provenance.json and provenance.md into `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = collect_provenance(extra)
    (out_dir / "provenance.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (out_dir / "provenance.md").write_text(provenance_markdown(info), encoding="utf-8")
    return info
