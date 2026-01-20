#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _repo_root() -> Path:
    # This file lives at <repo_root>/contextbench_eval/run_batch_eval_selected500.py
    return Path(__file__).resolve().parent.parent


def _env_or_default(env_key: str, default_path: Path) -> str:
    v = (os.environ.get(env_key) or "").strip()
    return v if v else str(default_path)


REPO_ROOT = _repo_root()
PKG_ROOT = Path(__file__).resolve().parent

DEFAULT_GOLD = _env_or_default("CONTEXTBENCH_GOLD", REPO_ROOT / "ContextBench_HF" / "data" / "full.parquet")
DEFAULT_CACHE = _env_or_default("CONTEXTBENCH_CACHE", REPO_ROOT / "repos")
DEFAULT_SELECTED = _env_or_default("CONTEXTBENCH_SELECTED_CSV", REPO_ROOT / "selected_500_instances.csv")
DEFAULT_RESULTS = _env_or_default("CONTEXTBENCH_RESULTS_ROOT", REPO_ROOT / "results")
DEFAULT_AGENTLESS_TRAJ_ROOT = _env_or_default("CONTEXTBENCH_TRAJ_AGENTLESS", PKG_ROOT / "traj" / "agentless")
DEFAULT_MINISWE_TRAJ_ROOT = _env_or_default("CONTEXTBENCH_TRAJ_MINISWE", PKG_ROOT / "traj" / "miniswe")


@dataclass(frozen=True)
class Row:
    bench: str
    instance_id: str
    original_inst_id: str


@dataclass(frozen=True)
class Job:
    agent: str  # agentless | miniswe
    model: str  # default | claude45 | gemini | gpt5 | mistral
    bench: str
    traj_path: str
    out_jsonl: str
    out_stdout: str
    out_stderr: str


def _load_selected_rows(csv_path: str, benches: Optional[List[str]] = None) -> List[Row]:
    want = set(b.strip() for b in benches) if benches else None
    out: List[Row] = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            bench = (row.get("bench") or "").strip()
            if want is not None and bench not in want:
                continue
            iid = (row.get("instance_id") or "").strip()
            oid = (row.get("original_inst_id") or "").strip()
            if not bench or not iid or not oid:
                continue
            out.append(Row(bench=bench, instance_id=iid, original_inst_id=oid))
    return out


def _agentless_traj_path(base: Path, bench: str, instance_id: str) -> Optional[Path]:
    if bench == "Verified":
        return base / "Verified" / f"{instance_id}_traj.json"
    if bench in ("Multi", "Pro", "Poly"):
        return base / bench / bench / f"{instance_id}_traj.json"
    return None


def _miniswe_traj_path(base: Path, model: str, bench: str, original_inst_id: str) -> Optional[Path]:
    bench_map = {"Multi": "multi", "Pro": "pro", "Poly": "poly", "Verified": "verified"}
    b = bench_map.get(bench)
    if not b:
        return None

    if model == "claude45":
        return base / "claude" / f"traj_{b}_claude45" / original_inst_id / f"{original_inst_id}.traj.json"
    if model == "gemini":
        return base / "gemini" / f"traj_{b}_gemini" / original_inst_id / f"{original_inst_id}.traj.json"
    if model == "gpt5":
        return base / "gpt" / f"traj_{b}_gpt5" / original_inst_id / f"{original_inst_id}.traj.json"
    if model == "mistral":
        return base / "mistral" / f"traj_{b}_mistral" / original_inst_id / f"{original_inst_id}.traj.json"
    return None


def _build_jobs(
    rows: List[Row],
    agents: List[str],
    miniswe_models: List[str],
    results_root: Path,
    agentless_root: Path,
    miniswe_root: Path,
) -> Tuple[List[Job], List[dict]]:
    jobs: List[Job] = []
    missing: List[dict] = []

    for r in rows:
        if "agentless" in agents:
            tp = _agentless_traj_path(agentless_root, r.bench, r.instance_id)
            if tp is None or not tp.is_file():
                missing.append(
                    {
                        "agent": "agentless",
                        "model": "default",
                        "bench": r.bench,
                        "instance_id": r.instance_id,
                        "original_inst_id": r.original_inst_id,
                        "expected_traj": str(tp) if tp is not None else None,
                    }
                )
            else:
                out_dir = results_root / "agentless" / "default" / r.bench
                out_dir.mkdir(parents=True, exist_ok=True)
                jobs.append(
                    Job(
                        agent="agentless",
                        model="default",
                        bench=r.bench,
                        traj_path=str(tp),
                        out_jsonl=str(out_dir / f"{r.instance_id}.jsonl"),
                        out_stdout=str(out_dir / f"{r.instance_id}.stdout.log"),
                        out_stderr=str(out_dir / f"{r.instance_id}.stderr.log"),
                    )
                )

        if "miniswe" in agents:
            for m in miniswe_models:
                tp = _miniswe_traj_path(miniswe_root, m, r.bench, r.original_inst_id)
                if tp is None or not tp.is_file():
                    missing.append(
                        {
                            "agent": "miniswe",
                            "model": m,
                            "bench": r.bench,
                            "instance_id": r.instance_id,
                            "original_inst_id": r.original_inst_id,
                            "expected_traj": str(tp) if tp is not None else None,
                        }
                    )
                    continue

                out_dir = results_root / "miniswe" / m / r.bench
                out_dir.mkdir(parents=True, exist_ok=True)
                jobs.append(
                    Job(
                        agent="miniswe",
                        model=m,
                        bench=r.bench,
                        traj_path=str(tp),
                        out_jsonl=str(out_dir / f"{r.original_inst_id}.jsonl"),
                        out_stdout=str(out_dir / f"{r.original_inst_id}.stdout.log"),
                        out_stderr=str(out_dir / f"{r.original_inst_id}.stderr.log"),
                    )
                )

    return jobs, missing


def _run_one(job: Job, gold: str, cache: str) -> Tuple[Job, int, float]:
    t0 = time.time()

    cmd = [
        "python",
        "-m",
        "contextbench_eval.evaluate",
        "--gold",
        gold,
        "--pred",
        job.traj_path,
        "--cache",
        cache,
        "--out",
        job.out_jsonl,
    ]

    with open(job.out_stdout, "w", encoding="utf-8") as out, open(job.out_stderr, "w", encoding="utf-8") as err:
        p = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=out, stderr=err)

    return job, int(p.returncode), time.time() - t0


def _summarize(
    jobs: List[Job],
    missing: List[dict],
    results: List[Tuple[Job, int, float]],
    out_path: Path,
    gold: str,
    cache: str,
    workers: int,
) -> None:
    by_run: Dict[Tuple[str, str], dict] = {}

    def key(j: Job) -> Tuple[str, str]:
        return (j.agent, j.model)

    for agent, model in sorted({(j.agent, j.model) for j in jobs}):
        by_run[(agent, model)] = {
            "agent": agent,
            "model": model,
            "gold": gold,
            "cache": cache,
            "workers": workers,
            "planned_rows": None,
            "found_traj": 0,
            "missing_traj": 0,
            "succeeded": 0,
            "failed": 0,
            "avg_seconds": None,
            "total_seconds": None,
        }

    for j in jobs:
        by_run[key(j)]["found_traj"] += 1

    for m in missing:
        by_run.setdefault((m["agent"], m["model"]), {
            "agent": m["agent"],
            "model": m["model"],
            "gold": gold,
            "cache": cache,
            "workers": workers,
            "planned_rows": None,
            "found_traj": 0,
            "missing_traj": 0,
            "succeeded": 0,
            "failed": 0,
            "avg_seconds": None,
            "total_seconds": None,
        })
        by_run[(m["agent"], m["model"])]["missing_traj"] += 1

    times_by_run: Dict[Tuple[str, str], List[float]] = {}
    for j, code, secs in results:
        if code == 0:
            by_run[key(j)]["succeeded"] += 1
        else:
            by_run[key(j)]["failed"] += 1
        times_by_run.setdefault(key(j), []).append(secs)

    for k, vals in times_by_run.items():
        by_run[k]["avg_seconds"] = sum(vals) / len(vals) if vals else None
        by_run[k]["total_seconds"] = sum(vals) if vals else None

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gold": gold,
        "cache": cache,
        "workers": workers,
        "runs": [by_run[k] for k in sorted(by_run.keys())],
        "missing_examples": missing[:50],
        "note": "Per-instance logs are stored alongside each output jsonl as .stdout.log and .stderr.log.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected_csv", default=DEFAULT_SELECTED)
    ap.add_argument("--gold", default=DEFAULT_GOLD)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--results_root", default=DEFAULT_RESULTS)
    ap.add_argument("--traj_root_agentless", default=DEFAULT_AGENTLESS_TRAJ_ROOT)
    ap.add_argument("--traj_root_miniswe", default=DEFAULT_MINISWE_TRAJ_ROOT)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--agents", default="agentless,miniswe", help="Comma-separated: agentless,miniswe")
    ap.add_argument("--miniswe_models", default="claude45,gemini,gpt5,mistral", help="Comma-separated")
    ap.add_argument("--benches", default="Multi,Pro,Poly,Verified", help="Comma-separated")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    models = [m.strip() for m in args.miniswe_models.split(",") if m.strip()]
    benches = [b.strip() for b in args.benches.split(",") if b.strip()]

    rows = _load_selected_rows(args.selected_csv, benches=benches)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    jobs, missing = _build_jobs(
        rows,
        agents,
        models,
        results_root,
        agentless_root=Path(args.traj_root_agentless),
        miniswe_root=Path(args.traj_root_miniswe),
    )

    if args.dry_run:
        print(f"rows={len(rows)} jobs={len(jobs)} missing={len(missing)}")
        for j in jobs[:10]:
            print(j.agent, j.model, j.bench, j.traj_path, "->", j.out_jsonl)
        return 0

    results: List[Tuple[Job, int, float]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        futs = [ex.submit(_run_one, j, args.gold, args.cache) for j in jobs]
        for fut in as_completed(futs):
            job, code, secs = fut.result()
            results.append((job, code, secs))

    _summarize(
        jobs=jobs,
        missing=missing,
        results=results,
        out_path=results_root / "summary.json",
        gold=args.gold,
        cache=args.cache,
        workers=int(args.workers),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



