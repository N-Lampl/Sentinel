"""Command line interface.

    sentinel scan [PATH] [--format coco|yolo|voc|image-folder] [--config sentinel.yaml] ...
    sentinel baseline create [PATH] -o .sentinel/baseline.json
    sentinel diff [PATH] --baseline .sentinel/baseline.json [--fail-on new-error]
    sentinel init [--format coco] [-o sentinel.yaml]
    sentinel plugins
    sentinel version

Exit codes: 0 pass, 1 policy failure, 2 usage / runtime error.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from . import __version__
from .config import SentinelConfig, config_template, load_config
from .model import Report, Severity
from .registry import adapters, detectors, reporters

log = logging.getLogger("dataset_sentinel")

CONFIG_CANDIDATES = ("sentinel.yaml", "sentinel.yml", ".sentinel.yaml", ".sentinel.yml", "sentinel.json")


def _add_scan_arguments(scan: argparse.ArgumentParser, *, diff_mode: bool = False, baseline_mode: bool = False) -> None:
    scan.add_argument("path", nargs="?", help="dataset root (defaults to dataset.root from the config, else the current directory)")
    scan.add_argument("-c", "--config", help="sentinel.yaml (auto-discovered in PATH or the current directory when omitted)")
    scan.add_argument("-f", "--format", help=f"dataset format: {', '.join(adapters.names())} (auto-detected when omitted)")
    scan.add_argument("--split", action="append", metavar="NAME=SPEC", default=[],
                      help="split definition. COCO: name=annotations.json[:images_dir]; VOC: name=ImageSets/Main/x.txt; image-folder: name=dir. Repeatable.")
    scan.add_argument("--data", help="YOLO data.yaml path")
    scan.add_argument("--metadata", "--manifest", dest="metadata", help="metadata manifest (CSV/TSV/JSON/JSONL/Parquet) with per-sample columns such as patient_id, source, captured_at")
    scan.add_argument("--metadata-key", help="manifest column that identifies the sample (default file_name)")
    scan.add_argument("--group-key", "--group-by", dest="group_key", action="append", default=[], help="metadata key that identifies an entity/source/batch (repeatable)")
    scan.add_argument("--text-field", action="append", default=[], metavar="FIELD",
                      help="text datasets: record field(s) holding the text (repeatable; concatenated)")
    scan.add_argument("--label-field", help="text datasets: record field holding the label")
    scan.add_argument("--id-field", help="text datasets: record field holding the sample id")
    scan.add_argument("--predictions", action="append", default=[], metavar="SPLIT=PATH",
                      help="model predictions for an evaluation split (COCO results JSON, YOLO txt directory or classification CSV); "
                           "the split is scored with and without the flagged samples. Repeatable.")
    if diff_mode:
        scan.add_argument("--baseline", required=True, help="JSON report of the approved previous run")
        scan.add_argument("--fail-on", choices=["new-error", "new-warning", "new-info", "error", "warning", "info", "none"], default="new-error",
                          help="what fails the run (default new-error: only findings absent from the baseline)")
    else:
        scan.add_argument("--baseline", help="JSON report of a previous run; findings already present there are marked as known")
        scan.add_argument("--new-only", action="store_true", help="with --baseline: fail only on new findings / a worse violation rate")
        scan.add_argument("--fail-on", choices=["error", "warning", "info", "none"], help="lowest severity that fails the run (default error)")
    scan.add_argument("--max-violation-rate", type=float, help="fail when the split integrity violation rate exceeds this fraction (default 0)")
    if baseline_mode:
        scan.add_argument("-o", "--output", default=".sentinel/baseline.json", help="where to write the baseline JSON report")
    scan.add_argument("--json", dest="json_out", metavar="FILE", help="write the JSON report here (default sentinel-report.json)")
    scan.add_argument("--html", dest="html_out", metavar="FILE", help="write the HTML report here (default sentinel-report.html)")
    scan.add_argument("--markdown", dest="md_out", metavar="FILE", help="write a Markdown summary (e.g. for $GITHUB_STEP_SUMMARY)")
    scan.add_argument("--sarif", dest="sarif_out", metavar="FILE", help="write a SARIF 2.1.0 file for GitHub code scanning")
    scan.add_argument("--dvc", dest="dvc_out", metavar="FILE", help="write flat metrics (YAML or JSON) for DVC")
    scan.add_argument("--fix-plan", dest="fix_plan_out", metavar="FILE", help="write suggested per-sample remove/move actions (JSON or CSV)")
    scan.add_argument("--report-dir", help="directory for default report files")
    scan.add_argument("--no-report", action="store_true", help="do not write report files")
    scan.add_argument("--github-annotations", action="store_true", help="print GitHub Actions workflow annotations for errors and warnings")
    scan.add_argument("--detectors", help="comma-separated detectors to run (default all)")
    scan.add_argument("--disable", help="comma-separated detectors to skip")
    scan.add_argument("--fast", action="store_true",
                      help="reduced-resolution JPEG decoding without pixel-identity hashing: 3-5x faster on large photos")
    scan.add_argument("--workers", type=int, help="fingerprinting threads (default: cpu count, max 8)")
    scan.add_argument("--cache", help="fingerprint cache directory")
    scan.add_argument("--no-cache", action="store_true")
    scan.add_argument("--no-thumbnails", action="store_true", help="omit image thumbnails from the HTML report")
    scan.add_argument("--max-samples", type=int, help="limit samples per split (debugging)")
    scan.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                      help="override any config key, e.g. --set policy.near_duplicate.threshold=4")
    scan.add_argument("-q", "--quiet", action="store_true", help="only print the final result line")
    scan.add_argument("-v", "--verbose", action="store_true")
    scan.add_argument("--no-color", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentinel",
        description="Dataset Sentinel: validate ML dataset split integrity before training and evaluation.",
    )
    parser.add_argument("--version", action="version", version=f"dataset-sentinel {__version__}")
    sub = parser.add_subparsers(dest="command")

    scan = sub.add_parser("scan", help="scan a dataset and write reports")
    _add_scan_arguments(scan)

    baseline = sub.add_parser("baseline", help="create an approved baseline report")
    bsub = baseline.add_subparsers(dest="baseline_command")
    create = bsub.add_parser("create", help="scan and store the result as a baseline (never fails the run)")
    _add_scan_arguments(create, baseline_mode=True)

    diff = sub.add_parser("diff", help="scan and compare against a baseline; fail only on regressions")
    _add_scan_arguments(diff, diff_mode=True)

    init = sub.add_parser("init", help="write a sentinel.yaml template")
    init.add_argument("-f", "--format", default="coco", choices=["coco", "yolo", "voc", "image-folder", "text"])
    init.add_argument("-o", "--output", default="sentinel.yaml")
    init.add_argument("--force", action="store_true")

    sub.add_parser("plugins", help="list available adapters, detectors and reporters")
    sub.add_parser("version", help="print the version")
    return parser


# --------------------------------------------------------------------------- helpers
def _parse_value(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except Exception:
        return text


def _parse_split_specs(specs: List[str], fmt: Optional[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--split expects NAME=SPEC, got {spec!r}")
        name, value = spec.split("=", 1)
        if fmt == "coco" or value.lower().endswith(".json") or ".json:" in value.lower():
            if ":" in value and not value[1:3] == ":\\":
                ann, img = value.split(":", 1)
                out[name] = {"annotations": ann, "images": img}
            else:
                out[name] = {"annotations": value}
        else:
            out[name] = value
    return out


def _discover_config(path_arg: Optional[str]) -> Optional[Path]:
    dirs = []
    if path_arg:
        dirs.append(Path(path_arg))
    dirs.append(Path.cwd())
    for d in dirs:
        for name in CONFIG_CANDIDATES:
            p = d / name
            if p.is_file():
                return p
    return None


def build_config_from_args(args: argparse.Namespace, mode: str = "scan") -> SentinelConfig:
    cfg_path: Optional[Path] = Path(args.config) if args.config else _discover_config(args.path)
    config = load_config(cfg_path) if cfg_path else SentinelConfig()
    if cfg_path and not args.quiet:
        print(f"using config {cfg_path}", file=sys.stderr)

    if args.path:
        config.set("dataset.root", str(Path(args.path).expanduser().resolve()))
    if args.format:
        config.set("dataset.format", None if args.format == "auto" else args.format)
    fmt = config.get("dataset.format")
    if args.split:
        config.set("dataset.splits", _parse_split_specs(args.split, fmt))
    if args.data:
        config.set("dataset.data", args.data)
        if not fmt:
            config.set("dataset.format", "yolo")
    if args.metadata:
        config.set("dataset.metadata.file", args.metadata)
    if args.metadata_key:
        config.set("dataset.metadata.key", args.metadata_key)
    if args.group_key:
        keys = list(config.get("dataset.groups.keys") or []) + args.group_key
        config.set("dataset.groups.keys", keys)
    if getattr(args, "text_field", None):
        config.set("dataset.text.fields", list(args.text_field))
        if not fmt:
            config.set("dataset.format", "text")
    if getattr(args, "label_field", None):
        config.set("dataset.text.label_field", args.label_field)
    if getattr(args, "id_field", None):
        config.set("dataset.text.id_field", args.id_field)
    for item in getattr(args, "predictions", []) or []:
        if "=" not in item:
            raise SystemExit(f"--predictions expects SPLIT=PATH, got {item!r}")
        split, value = item.split("=", 1)
        preds = dict(config.section("evaluation.predictions"))
        preds[split.strip()] = str(Path(value).expanduser())
        config.set("evaluation.predictions", preds)

    # baseline / fail conditions
    if getattr(args, "baseline", None):
        config.set("baseline.file", str(Path(args.baseline).expanduser()))
    if mode == "diff":
        config.set("fail_on.new_only", True)
        fail_on = args.fail_on
        if fail_on.startswith("new-"):
            fail_on = fail_on[4:]
        config.set("fail_on.severity", fail_on)
    elif mode == "baseline":
        config.set("fail_on.severity", "none")
        config.set("fail_on.max_violation_rate", None)
        config.set("fail_on.detector_errors", False)
    else:
        if getattr(args, "new_only", False):
            config.set("fail_on.new_only", True)
        if args.fail_on:
            config.set("fail_on.severity", args.fail_on)
    if args.max_violation_rate is not None:
        config.set("fail_on.max_violation_rate", args.max_violation_rate)

    if args.detectors:
        config.set("detectors.enabled", [x.strip() for x in args.detectors.split(",") if x.strip()])
    if args.disable:
        config.set("detectors.disabled", list(config.get("detectors.disabled") or []) + [x.strip() for x in args.disable.split(",") if x.strip()])
    if getattr(args, "fast", False):
        config.set("performance.mode", "fast")
    if args.workers is not None:
        config.set("performance.workers", args.workers)
    if args.cache:
        config.set("performance.cache", args.cache)
    if args.no_cache:
        config.set("performance.cache", None)
    if args.no_thumbnails:
        config.set("output.thumbnails", False)
    if args.max_samples:
        config.set("performance.max_samples_per_split", args.max_samples)
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        config.set(key.strip(), _parse_value(value))

    # report destinations
    report_dir = Path(args.report_dir) if args.report_dir else None
    if args.no_report:
        for kind in ("json", "html", "markdown", "sarif", "dvc", "fix_plan"):
            config.set(f"output.{kind}", None)
    else:
        if mode == "baseline":
            config.set("output.json", args.output)
            if args.html_out:
                config.set("output.html", args.html_out)
            else:
                config.set("output.html", None)
        else:
            if args.json_out:
                config.set("output.json", args.json_out)
            elif config.get("output.json") is None:
                config.set("output.json", str((report_dir or Path.cwd()) / "sentinel-report.json"))
            if args.html_out:
                config.set("output.html", args.html_out)
            elif config.get("output.html") is None:
                config.set("output.html", str((report_dir or Path.cwd()) / "sentinel-report.html"))
        if args.md_out:
            config.set("output.markdown", args.md_out)
        if args.sarif_out:
            config.set("output.sarif", args.sarif_out)
        if args.dvc_out:
            config.set("output.dvc", args.dvc_out)
        if args.fix_plan_out:
            config.set("output.fix_plan", args.fix_plan_out)
    return config


def _progress_printer(quiet: bool):
    if quiet or not sys.stderr.isatty():
        return None
    state = {"last": ""}

    def show(stage: str, done: int, total: int) -> None:
        if stage == "fingerprint":
            msg = f"\rfingerprinting images {done}/{total}"
        elif stage.startswith("detector:") and stage != "detector:done":
            msg = f"\rrunning detector {stage.split(':', 1)[1]} ({done + 1}/{total})          "
        else:
            msg = "\r" + " " * len(state["last"]) + "\r"
        state["last"] = msg
        sys.stderr.write(msg)
        sys.stderr.flush()

    return show


def _gh_escape(text: str, data: bool = False) -> str:
    text = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    if not data:
        text = text.replace(":", "%3A").replace(",", "%2C")
    return text


def emit_github_annotations(report: Report, limit_per_level: int = 10, stream=None) -> int:
    """Print ``::error`` / ``::warning`` workflow commands for the top findings."""
    stream = stream or sys.stdout
    workspace = Path(os.environ.get("GITHUB_WORKSPACE") or Path.cwd()).resolve()
    dataset = report.dataset_ref
    counts = {"error": 0, "warning": 0}
    emitted = 0
    for f in report.findings:
        if f.suppressed or f.severity is Severity.INFO:
            continue
        level = "error" if f.severity is Severity.ERROR else "warning"
        if counts[level] >= limit_per_level:
            continue
        file_attr = ""
        if dataset is not None and f.samples:
            s = dataset.get_or_none(f.samples[0].id)
            if s is not None and s.path is not None:
                try:
                    rel = Path(s.path).resolve().relative_to(workspace)
                    file_attr = f"file={_gh_escape(rel.as_posix())},"
                except ValueError:
                    pass
        title = _gh_escape(f"[{f.detector}] {f.title}"[:200])
        message = _gh_escape(f"{f.message} Remediation: {f.remediation}"[:1000], data=True)
        stream.write(f"::{level} {file_attr}title={title}::{message}\n")
        counts[level] += 1
        emitted += 1
    stream.flush()
    return emitted


# --------------------------------------------------------------------------- commands
def run_scan(args: argparse.Namespace, mode: str = "scan") -> int:
    from .engine import scan
    from .fixplan import write_fix_plan
    from .reporters.console import ConsoleReporter

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    config = build_config_from_args(args, mode)
    try:
        report = scan(config, progress=_progress_printer(args.quiet))
    except (FileNotFoundError, ValueError, KeyError) as exc:
        if args.verbose:
            traceback.print_exc()
        print(f"error: {exc}", file=sys.stderr)
        return 2

    written: List[Path] = []
    for kind in ("json", "html", "markdown", "sarif", "dvc"):
        target = config.get(f"output.{kind}")
        if not target:
            continue
        try:
            reporter = reporters.get(kind)(config)
            written.append(reporter.write(report, config.resolve_path(target, Path.cwd())))
        except Exception as exc:  # pragma: no cover - defensive
            if args.verbose:
                traceback.print_exc()
            print(f"warning: failed to write {kind} report: {exc}", file=sys.stderr)
    plan_target = config.get("output.fix_plan")
    if plan_target and report.fix_plan is not None:
        written.append(write_fix_plan(report.fix_plan, config.resolve_path(plan_target, Path.cwd())))

    console = ConsoleReporter(config, color=False if args.no_color else None)
    if args.quiet:
        status = "PASS" if report.passed else "FAIL"
        base = report.stats.get("baseline") or {}
        extra = f" new={base.get('new', 0)}" if base.get("applied") else ""
        print(f"{status} violation_rate={report.metrics.violation_rate:.4f} errors={report.metrics.severity_counts.get('error', 0)}{extra}")
    else:
        sys.stdout.write(console.render(report))
        if written:
            print("Reports: " + ", ".join(str(p) for p in written))
    if getattr(args, "github_annotations", False):
        emit_github_annotations(report)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path and not config.get("output.markdown"):
        try:
            md = reporters.get("markdown")(config).render(report)
            with open(summary_path, "a", encoding="utf-8") as fh:
                fh.write(md)
        except Exception as exc:  # pragma: no cover
            print(f"warning: could not write job summary: {exc}", file=sys.stderr)
    if mode == "baseline":
        print(f"baseline written to {config.get('output.json')}")
        return 0
    return 0 if report.passed else 1


def cmd_init(args: argparse.Namespace) -> int:
    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"{out} already exists (use --force to overwrite)", file=sys.stderr)
        return 2
    out.write_text(config_template(args.format), encoding="utf-8")
    print(f"wrote {out}")
    return 0


def cmd_plugins(_: argparse.Namespace) -> int:
    print("Adapters:")
    for name, cls in adapters.items():
        print(f"  {name:<14} {getattr(cls, 'description', '')}  [modality: {getattr(cls, 'modality', '?')}]")
    print("Detectors:")
    for name, cls in detectors.items():
        mods = ",".join(getattr(cls, "modalities", ()))
        print(f"  {name:<18} {getattr(cls, 'description', '')}  [modalities: {mods}]")
    print("Reporters:")
    for name, _cls in reporters.items():
        print(f"  {name}")
    from .embeddings.base import embeddings

    print("Embedding providers (optional near-duplicate re-ranking):")
    for name, cls in embeddings.items():
        state = "available" if cls.available() else "missing optional dependency"
        print(f"  {name:<14} {getattr(cls, 'description', '')}  [{state}]")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in (None, "version"):
        if args.command is None:
            parser.print_help()
            return 0
        print(f"dataset-sentinel {__version__}")
        return 0
    try:
        if args.command == "scan":
            return run_scan(args, "scan")
        if args.command == "diff":
            return run_scan(args, "diff")
        if args.command == "baseline":
            if args.baseline_command != "create":
                parser.parse_args(["baseline", "--help"])
                return 2
            return run_scan(args, "baseline")
        if args.command == "init":
            return cmd_init(args)
        if args.command == "plugins":
            return cmd_plugins(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # pragma: no cover - top-level guard
        if getattr(args, "verbose", False):
            traceback.print_exc()
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
