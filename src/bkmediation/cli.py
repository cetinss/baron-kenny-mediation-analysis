"""Command-line interface: ``python -m bkmediation <command>``.

Commands
--------
    analyze      cleaning + Baron-Kenny estimation + all report tables
    hypotheses   print the H1/H2/H3 decision table
    samples      re-estimate on the full raw n=10,000 and each cleaned sample
    gender       re-estimate under both gender-coding schemes
    benchmark    runtime, scalability and cross-software comparison
    provenance   versions, seed and git commit
    all          everything above, in that order
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .benchmark import run_benchmarks
from .comparisons import write_gender_coding_report, write_sample_definition_report
from .config import N_BOOTSTRAP, OUTPUTS_DIR, RANDOM_SEED, REPORTS_DIR
from .data import build_analytic_sample
from .diagnostics import run_diagnostics
from .hypotheses import evaluate_hypotheses
from .mediation import run_mediation
from .provenance import write_provenance
from .reporting import write_analysis_outputs


def _out_dir(args):
    """Where the versioned reports go (outputs/ unless --out-dir is given)."""
    return Path(args.out_dir) if args.out_dir else OUTPUTS_DIR


def _build(args):
    return build_analytic_sample(
        path=args.data,
        gender_scheme=args.gender_scheme,
        verbose=not args.quiet,
    )


def _analyze(args):
    sample = _build(args)
    result = run_mediation(sample, n_boot=args.n_boot, seed=args.seed,
                           cov_type=args.cov_type)
    diagnostics = run_diagnostics(sample)
    written = write_analysis_outputs(sample, result, REPORTS_DIR, diagnostics,
                                     markdown_copy_dir=_out_dir(args))
    if not args.quiet:
        print()
        print(result.summary())
        print()
        print("Written:")
        for name, path in written.items():
            print(f"  {path}")
    return sample, result


def _hypotheses(args):
    sample = _build(args)
    result = run_mediation(sample, n_boot=args.n_boot, seed=args.seed,
                           cov_type=args.cov_type)
    table = evaluate_hypotheses(result)
    out_dir = _out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "hypothesis_tests.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(table[["Hypothesis", "Tested quantity", "Estimate (per 100 mg)",
                     "95% CI low (per 100 mg)", "95% CI high (per 100 mg)",
                     "p", "Decision"]].to_string(index=False))
    for _, row in table.iterrows():
        print(f"\n{row['Hypothesis']}: {row['Statement']}\n"
              f"   rule: {row['Decision rule']}\n"
              f"   -> {row['Decision']}")
    return table


def _samples(args):
    table = write_sample_definition_report(_out_dir(args), n_boot=args.n_boot, seed=args.seed)
    print(table.drop(columns=["Definition"]).to_string(index=False))
    return table


def _gender(args):
    table = write_gender_coding_report(_out_dir(args), n_boot=args.n_boot, seed=args.seed)
    print(table.to_string(index=False))
    return table


def _benchmark(args):
    results = run_benchmarks(_out_dir(args), n_boot=args.n_boot, quick=args.quick)
    for name, frame in results.items():
        print(f"\n--- {name} ---")
        print(frame.to_string(index=False))
    return results


def _provenance(args):
    info = write_provenance(_out_dir(args))
    print(f"Python {info['python']} | bkmediation {info['bkmediation_version']} | "
          f"seed {info['analysis_settings']['random_seed']} | "
          f"commit {info['git']['commit_short'] or 'n/a'}")
    for pkg, version in info["packages"].items():
        print(f"  {pkg}: {version}")
    return info


def _all(args):
    _analyze(args)
    print("\n" + "=" * 70 + "\nHYPOTHESIS TESTS\n" + "=" * 70)
    _hypotheses(args)
    print("\n" + "=" * 70 + "\nGENDER-CODING COMPARISON\n" + "=" * 70)
    _gender(args)
    print("\n" + "=" * 70 + "\nPROVENANCE\n" + "=" * 70)
    _provenance(args)
    print("\n" + "=" * 70 + "\nBENCHMARKS\n" + "=" * 70)
    _benchmark(args)


COMMANDS = {
    "analyze": _analyze,
    "hypotheses": _hypotheses,
    "samples": _samples,
    "gender": _gender,
    "benchmark": _benchmark,
    "provenance": _provenance,
    "all": _all,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bkmediation",
        description="Baron-Kenny mediation analysis: caffeine -> stress -> sleep.",
    )
    parser.add_argument("command", choices=sorted(COMMANDS), help="what to run")
    parser.add_argument("--data", default=None, help="path to the input CSV")
    parser.add_argument("--gender-scheme", default="three_level",
                        choices=["three_level", "binary_male"],
                        help="gender covariate coding (default: three_level)")
    parser.add_argument("--n-boot", type=int, default=N_BOOTSTRAP,
                        help=f"bootstrap resamples (default: {N_BOOTSTRAP})")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"random seed (default: {RANDOM_SEED})")
    parser.add_argument("--cov-type", default="HC3", choices=["HC3", "HC0", "HC1", "nonrobust"],
                        help="covariance estimator for reported SEs (default: HC3)")
    parser.add_argument("--quick", action="store_true",
                        help="benchmark: fewer replicates and sizes")
    parser.add_argument("--out-dir", default=None,
                        help="where to write the versioned reports (default: outputs/)")
    parser.add_argument("--quiet", action="store_true", help="less console output")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    COMMANDS[args.command](args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
