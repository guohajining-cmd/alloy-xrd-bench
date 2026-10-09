"""Run a bounded corrected-label training batch in a fresh isolated workspace."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alloy_xrd_bench.sro_revision import file_sha256


def build_jobs(config, models, seeds, splits):
    allowed = {"peak-xgboost", "resnet1d", "transformer", "physics-transformer"}
    if not set(models).issubset(allowed) or not set(seeds).issubset(set(config["seeds"])) or not set(splits).issubset(set(config["split_columns"])):
        raise ValueError("Unsupported training cases")
    jobs = [(model, seed, split) for split in splits for seed in seeds for model in models]
    if not jobs or len(jobs) != len(set(jobs)):
        raise ValueError("Empty or duplicate training cases")
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--revision", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["peak-xgboost", "resnet1d", "transformer", "physics-transformer"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2028, 2029, 2030, 2031])
    parser.add_argument("--splits", nargs="+", default=["id_split"])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available() or len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 or gpu >= torch.cuda.device_count() for gpu in args.gpus):
        raise ValueError("Requested exclusive GPU workers are not available")
    out = args.output.resolve()
    if out.is_relative_to(args.dataset.resolve()) or out.is_relative_to(ROOT / "data"):
        raise ValueError("Output must not be inside source data")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a new run directory; do not overwrite old experiments")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    jobs = build_jobs(config, args.models, args.seeds, args.splits)
    revision = json.loads((args.revision / "revision.json").read_text(encoding="utf-8"))
    if revision["status"] != "verified" or revision["source_manifest_sha256"] != file_sha256(args.dataset / "manifests/task_samples.parquet"):
        raise ValueError("Corrected labels are not bound to this source dataset")
    if revision["labels_sha256"] != file_sha256(args.revision / "labels.parquet") or revision["sro_targets"] != config["sro_targets"]:
        raise ValueError("Corrected label content or target order mismatch")
    config["sro_label_revision"] = str(args.revision.resolve())
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out / "logs").mkdir()
    protocol = dict(status="running", runs=jobs, source_manifest_sha256=revision["source_manifest_sha256"],
                    labels_sha256=revision["labels_sha256"], revision_sha256=file_sha256(args.revision / "revision.json"),
                    config_sha256=file_sha256(out / "config.json"),
                    source_hashes={str(p.relative_to(ROOT)): file_sha256(p) for p in
                                   sorted((ROOT / "alloy_xrd_bench").glob("*.py"))},
                    python=sys.version, packages={name: importlib.metadata.version(name) for name in
                                                 ["numpy", "pandas", "torch", "scikit-learn", "xgboost", "h5py"]},
                    gpu_worker_count=len(args.gpus), sample_limit=None,
                    qualification="Corrected-label retraining with original model budgets and frozen splits; historical exact environment equivalence is not asserted.")
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    lock, stop = threading.Lock(), threading.Event()
    results = []
    def worker(gpu, queue):
        for model, seed, split in queue:
            if stop.is_set():
                return
            name = f"{model}_{split}_seed{seed}"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                       OPENBLAS_NUM_THREADS="4", CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1")
            command = [sys.executable, "-m", "alloy_xrd_bench.hea_validation", "train", "--model", model,
                       "--dataset", str(args.dataset.resolve()), "--config", str(out / "config.json"),
                       "--output", str(out / "predictions"), "--seed", str(seed), "--split-column", split]
            started = time.monotonic()
            print(f"START gpu={gpu} {name}", flush=True)
            with (out / "logs" / f"{name}.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            result = dict(model=model, seed=seed, split_column=split, exit_code=completed.returncode,
                          seconds=time.monotonic()-started)
            with lock:
                results.append(result)
                (out / "progress.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"DONE rc={completed.returncode} {name} {result['seconds']:.1f}s", flush=True)
            if completed.returncode:
                stop.set()
                return
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(worker, gpu, jobs[index::len(args.gpus)]) for index, gpu in enumerate(args.gpus)]
        for future in futures:
            future.result()
    unchanged = all(file_sha256(ROOT / path) == digest for path, digest in protocol["source_hashes"].items())
    unchanged &= file_sha256(args.dataset / "manifests/task_samples.parquet") == protocol["source_manifest_sha256"]
    unchanged &= file_sha256(args.revision / "labels.parquet") == protocol["labels_sha256"]
    protocol.update(status="complete" if not stop.is_set() and len(results) == len(jobs) and unchanged else "failed",
                    completed_run_count=len(results), input_hashes_unchanged=unchanged)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    if protocol["status"] != "complete":
        raise SystemExit("Training batch failed; preserve logs and do not publish incomplete results")
    print(f"COMPLETE {len(jobs)} corrected training runs", flush=True)


if __name__ == "__main__":
    main()
