"""Command-line entry point.

    python -m src.run_pipeline [--config configs/default.yaml] [-o key=value ...] <command>

Commands (each stage caches its output and is skipped when already done; --force redoes it):
    prep      --split {train,test}   parse records, mine lexicons
    block     --split {train,test}   candidate generation (+ stage-0 ranker on train);
                                     --resume continues an interrupted run
    block-bench                      recall + speed of the blocking settings on 1 chunk/country
    blank --src DIR --country C --out DIR   LB probe: DIR's matches with country C emptied
    shift                            adversarial validation train vs test slices (read-only)
    aug-check                        synthetic records vs real: no giveaway in stage-0 columns
    gates     [--tag B]              offline gates: accepted per S1 by house-number relation, train vs test
    augment                          synthetic decoy twins in train (after train blocking)
    features  --split {train,test}   pairwise + context features
    train-a                          stage-A models, train OOF predictions
    predict-a --split test           stage-A test predictions
    collective --split {train,test}  collective features + orphan model
    train-b / predict-b              stage-B models / test predictions
    decide    [--tag A|B]            calibrate + tune the decision rule on train OOF
    write                            decide on test, write output/*.tsv, run the validator
    all                              everything above, in order
    report                           rebuild work/reports/summary.md
    probe     --tag B --out DIR      cached test decision minus the decoy-twin signature -> DIR
    loco                             leave-one-country-out generalisation check
    dev-slice                        build the ~5% dev dataset into work/dev_data
    neural-export / neural-train --half {0,1} / neural-infer   optional cross-encoder
    clean     --from <stage>         delete cached outputs of a stage and everything after it
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time

from .config import load_config, models_dir, report_dir, split_dir
from .utils import LOG, is_done, mark_done, setup_logging

STAGE_DIRS = {  # stage -> per-split cache directories it produces (for `clean`)
    "prep": ["records_s1", "records_q"],
    "block": ["cand", "union"],
    "features": ["feat", "ctx"],
    "stage-a": ["pred_A"],
    "collective": ["coll"],
    "neural": ["nn"],
    "stage-b": ["pred_B"],
}
ORDER = ["prep", "block", "features", "stage-a", "collective", "neural", "stage-b"]


def _clean(cfg: dict, start: str) -> None:
    stages = ORDER[ORDER.index(start):]
    for split in ("train", "test"):
        for st in stages:
            for d in STAGE_DIRS[st]:
                path = split_dir(cfg, split) / d
                if path.exists():
                    shutil.rmtree(path)
                    LOG.info("removed %s", path)
    if start in ("prep", "block"):
        for f in models_dir(cfg).glob("*"):
            f.unlink()
        LOG.info("removed all models")


def _train_a(cfg: dict, force: bool) -> None:
    from .decide import run_decide_train
    from .train_gbdt import train_stage

    if force or not is_done(split_dir(cfg, "train") / "pred_A"):
        train_stage(cfg, "A")
        run_decide_train(cfg, "A")


def _predict(cfg: dict, split: str, tag: str, force: bool) -> None:
    from .train_gbdt import predict_stage

    if force or not is_done(split_dir(cfg, split) / f"pred_{tag}"):
        predict_stage(cfg, split, tag)


def _train_b(cfg: dict, force: bool) -> None:
    from .train_gbdt import train_stage

    if force or not is_done(split_dir(cfg, "train") / "pred_B"):
        train_stage(cfg, "B")


def run_all(cfg: dict, force: bool) -> None:
    from .blocking import run_blocking
    from .collective import run_collective
    from .context_features import run_context
    from .decide import run_decide_train
    from .features import run_features
    from .prep import run_prep
    from .report import build_summary
    from .write_outputs import run_write

    for split in ("train", "test"):
        run_prep(cfg, split)
    for split in ("train", "test"):
        run_blocking(cfg, split)
    from .augment import run_augment

    run_augment(cfg)
    for split in ("train", "test"):
        run_features(cfg, split)
        run_context(cfg, split)
    _train_a(cfg, force)
    _predict(cfg, "test", "A", force)
    for split in ("train", "test"):
        run_collective(cfg, split)
    if cfg["neural"]["enabled"]:
        from .neural.pipeline import run_neural

        run_neural(cfg)
    _train_b(cfg, force)
    _predict(cfg, "test", "B", force)
    run_decide_train(cfg, "B")
    run_write(cfg, "B")
    build_summary(cfg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command")
    ap.add_argument("--config", default=None)
    ap.add_argument("-o", "--override", action="append", default=[],
                    help="config override key=value (repeatable)")
    ap.add_argument("--split", choices=["train", "test"], default=None)
    ap.add_argument("--tag", default="B")
    ap.add_argument("--half", type=int, default=None)
    ap.add_argument("--from", dest="from_stage", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="block: keep the chunks an interrupted run already wrote")
    ap.add_argument("--out", default=None, help="probe: folder for the probe's TSV files")
    ap.add_argument("--src", default=None, help="blank: submission folder to copy")
    ap.add_argument("--country", default=None, help="blank: country whose rows are emptied")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.override)
    setup_logging(report_dir(cfg) / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log")
    LOG.info("command=%s split=%s work_dir=%s", args.command, args.split,
             cfg["paths"]["work_dir"])
    cmd = args.command

    def need_split() -> str:
        if args.split is None:
            ap.error(f"{cmd} needs --split train|test")
        return args.split

    if cmd == "prep":
        from .prep import run_prep

        run_prep(cfg, need_split())
    elif cmd == "block":
        from .blocking import run_blocking

        run_blocking(cfg, need_split(), resume=args.resume)
    elif cmd == "blank":
        from pathlib import Path

        from .postprocess import blank_country

        if args.out is None or args.src is None or args.country is None:
            ap.error("blank needs --src DIR --country NAME --out DIR")
        blank_country(cfg, Path(args.src), args.country, Path(args.out))
    elif cmd == "gates":
        from .diagnose import run_gates

        run_gates(cfg, args.tag)
    elif cmd == "aug-check":
        from .diagnose import run_aug_check

        run_aug_check(cfg)
    elif cmd == "shift":
        from .diagnose import run_shift

        run_shift(cfg)
    elif cmd == "block-bench":
        from .blocking import run_block_bench

        run_block_bench(cfg)
    elif cmd == "augment":
        from .augment import run_augment

        run_augment(cfg)
    elif cmd == "features":
        from .context_features import run_context
        from .features import run_features

        run_features(cfg, need_split())
        run_context(cfg, need_split())
    elif cmd == "train-a":
        _train_a(cfg, args.force)
    elif cmd == "predict-a":
        _predict(cfg, need_split(), "A", args.force)
    elif cmd == "collective":
        from .collective import run_collective

        run_collective(cfg, need_split())
    elif cmd == "train-b":
        _train_b(cfg, args.force)
    elif cmd == "predict-b":
        _predict(cfg, need_split(), "B", args.force)
    elif cmd == "decide":
        from .decide import run_decide_train

        run_decide_train(cfg, args.tag)
    elif cmd == "write":
        from .write_outputs import run_write

        run_write(cfg, args.tag)
    elif cmd == "all":
        run_all(cfg, args.force)
    elif cmd == "report":
        from .report import build_summary

        build_summary(cfg)
    elif cmd == "probe":
        from pathlib import Path

        from .postprocess import run_probe

        if args.out is None:
            ap.error("probe needs --out <folder> (never output/)")
        run_probe(cfg, args.tag, Path(args.out))
    elif cmd == "loco":
        from .loco import run_loco

        run_loco(cfg)
    elif cmd == "dev-slice":
        from .make_dev_slice import make_dev_slice

        make_dev_slice(cfg)
    elif cmd == "neural-export":
        from .neural.pipeline import export_pairs

        export_pairs(cfg)
    elif cmd == "neural-train":
        from .neural.pipeline import train_half

        if args.half is None:
            ap.error("neural-train needs --half 0|1")
        train_half(cfg, args.half)
    elif cmd == "neural-infer":
        from .neural.pipeline import infer_all

        infer_all(cfg)
    elif cmd == "clean":
        if args.from_stage not in ORDER:
            ap.error(f"--from must be one of {ORDER}")
        _clean(cfg, args.from_stage)
    else:
        ap.error(f"unknown command {cmd!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
