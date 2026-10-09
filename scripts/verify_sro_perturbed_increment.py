"""Re-score matched held-out perturbations and compare to composition controls."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alloy_xrd_bench.hea_validation_data import vectorize_sro
from alloy_xrd_bench.sro_revision import apply_label_revision, file_sha256
from scripts.rescore_sro_matrix import stem, verify_downloads
from scripts.verify_sro_signal_increment import bootstrap_contrast, require_spectral_matrix


def observation_parent_errors(truth, mask, prediction, frame):
    if prediction.shape != truth.shape or truth.shape != mask.shape or len(frame) != len(truth):
        raise ValueError("Unaligned observation arrays")
    if not mask.any(axis=1).all() or not np.isfinite(prediction).all():
        raise ValueError("Invalid observation prediction or mask")
    errors = np.where(mask, np.abs(prediction-truth), 0).sum(axis=1) / mask.sum(axis=1)
    table = frame[["parent_group_id", "structure_state_id"]].copy()
    table["mae"] = errors
    return table.groupby(["parent_group_id", "structure_state_id"]).mae.mean().groupby(level=0).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True, help="Frozen xrd_samples.parquet")
    parser.add_argument("--revision", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--perturbed-runs", type=Path, nargs="+", required=True)
    parser.add_argument("--ridge", type=Path, required=True)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--require-full-matrix", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() or out.is_relative_to(ROOT/"data") or out.is_relative_to(args.dataset.resolve()):
        raise ValueError("Fresh non-data output required")
    config = json.loads((args.runs[0]/"config.json").read_text())
    manifest = args.dataset/"manifests/task_samples.parquet"
    inputs = {}

    def bind(path):
        inputs[str(path.resolve())] = file_sha256(path)
        return path

    for path in [Path(__file__),ROOT/"scripts/verify_sro_signal_increment.py"]:
        bind(path)

    tasks = apply_label_revision(pd.read_parquet(bind(manifest)), manifest, args.revision, config["sro_targets"])
    bind(args.revision/"labels.parquet")
    observations = pd.read_parquet(bind(args.observations))
    if tasks.structure_state_id.duplicated().any() or observations.xrd_sample_id.duplicated().any():
        raise ValueError("Duplicate identity")
    source = tasks[["structure_state_id", "parent_group_id", "xrd_sample_id", "sro_shell1_json", "sro_shell2_json", *config["split_columns"]]]
    source = source.rename(columns={"xrd_sample_id":"clean_xrd_sample_id"})
    panel = observations[["structure_state_id", "parent_group_id", "xrd_sample_id", "source_xrd_sample_id"]].merge(
        source, on="structure_state_id", how="left", validate="many_to_one", suffixes=("_obs", ""), indicator=True)
    if not panel._merge.eq("both").all() or not panel.parent_group_id.eq(panel.parent_group_id_obs).all() or not panel.source_xrd_sample_id.eq(panel.clean_xrd_sample_id).all():
        raise ValueError("Observation provenance mismatch")
    if any(tasks.groupby("parent_group_id")[s].nunique().max()>1 for s in config["split_columns"]):
        raise ValueError("Parent leakage")
    clean_paths = {}
    for directory in args.runs:
        verify_downloads(directory)
        meta = json.loads(bind(directory/"protocol.json").read_text())
        if meta["status"]!="complete" or not meta["input_hashes_unchanged"] or meta["source_manifest_sha256"]!=file_sha256(manifest) or meta["labels_sha256"]!=file_sha256(args.revision/"labels.parquet"):
            raise ValueError("Clean training contract mismatch")
        for model,seed,split in meta["runs"]:
            clean_paths[(split,model,seed)] = directory/"predictions"/f"{stem(model,seed,split)}.parquet"
    if args.controls:
        control=json.loads(bind(args.controls/"protocol.json").read_text())
        if control["status"]!="complete" or not control["input_hashes_unchanged"] or control["labels_sha256"]!=file_sha256(args.revision/"labels.parquet") or control["source_manifest_sha256"]!=file_sha256(manifest):
            raise ValueError("Incomplete or mismatched controls")
        if control["config"]["sro_targets"]!=config["sro_targets"]:
            raise ValueError("Control target order differs")
        for split,feature in control["planned_runs"]:
            path=args.controls/"predictions"/f"{split}_{feature}.parquet"
            if file_sha256(path)!=control["output_hashes"][f"predictions/{path.name}"]:
                raise ValueError("Control prediction hash mismatch")
            clean_paths[(split,f"control-xgboost-{feature}",control["seed"])]=path
    scores, errors, cache = [], {}, {}
    for directory in args.perturbed_runs:
        verify_downloads(directory)
        meta = json.loads(bind(directory/"protocol.json").read_text())
        if meta["status"]!="verified":
            raise ValueError("Unverified perturbation run")
        for filename, path in [("task_samples.parquet", manifest), ("labels.parquet", args.revision/"labels.parquet"), ("xrd_samples.parquet", args.observations)]:
            found = [v for k,v in meta["source_hashes"].items() if Path(k).name==filename]
            if found != [file_sha256(path)]:
                raise ValueError("Perturbation source contract mismatch")
        for model,seed,split in meta["runs"]:
            key=(split,model,seed)
            if key in errors:
                raise ValueError("Duplicate case")
            if split not in cache:
                part=panel[panel[split].eq("test")].copy().reset_index(drop=True)
                pairs=[vectorize_sro(r.sro_shell1_json,r.sro_shell2_json,config["sro_targets"]) for r in part.itertuples(index=False)]
                y,m=np.stack([p[0] for p in pairs]).astype(float),np.stack([p[1] for p in pairs])
                eligible=m.any(axis=1)
                cache[split]=(part[eligible].reset_index(drop=True), y[eligible],m[eligible])
            part,y,m=cache[split]
            name=stem(model,seed,split)
            frame=pd.read_parquet(bind(directory/f"{name}.parquet"))
            if frame.xrd_sample_id.duplicated().any() or set(frame.xrd_sample_id)!=set(part.xrd_sample_id):
                raise ValueError("Perturbation test membership mismatch")
            frame=frame.set_index("xrd_sample_id").loc[part.xrd_sample_id]
            if not np.array_equal(frame.parent_group_id,part.parent_group_id) or not np.array_equal(frame.structure_state_id,part.structure_state_id):
                raise ValueError("Perturbation identity mismatch")
            native_dirty=np.stack(frame.sro_predictions)
            dirty_values=native_dirty.astype(float)
            clean_frame=pd.read_parquet(bind(clean_paths[key])).set_index("xrd_sample_id")
            if not clean_frame.index.is_unique or not np.array_equal(clean_frame.loc[part.clean_xrd_sample_id].parent_group_id,part.parent_group_id):
                raise ValueError("Clean identity mismatch")
            clean_values=np.stack(clean_frame.loc[part.clean_xrd_sample_id].sro_predictions).astype(float)
            result=pd.DataFrame({"clean":observation_parent_errors(y,m,clean_values,part),
                                 "perturbed":observation_parent_errors(y,m,dirty_values,part)})
            saved=pd.read_csv(bind(directory/f"{name}.parents.csv")).set_index("parent_group_id")
            if set(result.index)!=set(saved.index):
                raise ValueError("Independent parent membership disagrees")
            max_parent_diff=float(np.max(np.abs(result.sort_index().to_numpy()-saved.loc[result.sort_index().index,["clean","perturbed"]].to_numpy())))
            # Reproduce the producer's native arithmetic before reporting float64 scores.
            native_clean=clean_values.astype(native_dirty.dtype)
            native_y=y.astype(np.float32)
            reproduced=pd.DataFrame({"clean":observation_parent_errors(native_y,m,native_clean,part),
                                     "perturbed":observation_parent_errors(native_y,m,native_dirty,part)})
            native_diff=float(np.max(np.abs(reproduced.sort_index().to_numpy()-saved.loc[reproduced.sort_index().index,["clean","perturbed"]].to_numpy())))
            if native_diff>1e-10:
                raise ValueError(f"Native-precision parent scores disagree: {name} maxdiff={native_diff:.12g}")
            errors[key]=result
            scores.append(dict(split=split,model=model,seed=seed,clean_mae=float(result.clean.mean()),
                               perturbed_mae=float(result.perturbed.mean()),parents=len(result),states=part.structure_state_id.nunique(),observations=len(part),
                               native_precision_max_parent_difference=native_diff,float64_vs_native_max_parent_difference=max_parent_diff))
    if args.require_full_matrix:
        require_spectral_matrix([key for key in errors if not key[1].startswith("control-xgboost-")],config)
    for split,(part,y,m) in cache.items():
        baselines={"composition-ridge": args.ridge/"C"/f"{split}.parquet"}
        if args.controls:
            control=json.loads(bind(args.controls/"protocol.json").read_text())
            if control["status"]!="complete" or control["labels_sha256"]!=file_sha256(args.revision/"labels.parquet"):
                raise ValueError("Incomplete composition control")
            control_path=args.controls/"predictions"/f"{split}_C.parquet"
            if file_sha256(control_path)!=control["output_hashes"][f"predictions/{split}_C.parquet"]:
                raise ValueError("Composition control hash mismatch")
            baselines["control-xgboost-C"]=control_path
        for model,path in baselines.items():
            if (split,model,2027) in errors:
                existing=errors[(split,model,2027)]
                if not np.array_equal(existing.clean,existing.perturbed):
                    raise ValueError("Composition errors changed under spectral perturbation")
                continue
            frame=pd.read_parquet(bind(path)).set_index("xrd_sample_id")
            if not frame.index.is_unique or not np.array_equal(frame.loc[part.clean_xrd_sample_id].parent_group_id,part.parent_group_id):
                raise ValueError("Composition baseline identity mismatch")
            prediction=np.stack(frame.loc[part.clean_xrd_sample_id].sro_predictions)
            parent=observation_parent_errors(y,m,prediction,part)
            errors[(split,model,2027)]=pd.DataFrame({"clean":parent,"perturbed":parent})
            scores.append(dict(split=split,model=model,seed=2027,clean_mae=float(parent.mean()),
                               perturbed_mae=float(parent.mean()),parents=len(parent),states=part.structure_state_id.nunique(),observations=len(part)))
    out.mkdir(parents=True)
    table=pd.DataFrame(scores)
    table.to_csv(out/"scores.csv",index=False)
    matrices={}
    for (split,model),part in table.groupby(["split","model"]):
        seeds=sorted(part.seed)
        if model not in ["composition-ridge","control-xgboost-C","control-xgboost-X","control-xgboost-CX"] and seeds!=sorted(config["seeds"]):
            raise ValueError("Incomplete perturbation seeds")
        for phase in ["clean","perturbed"]:
            matrix=pd.DataFrame({s:errors[(split,model,s)][phase] for s in seeds})
            matrices[(split,model,phase)]=matrix
            matrix.to_csv(out/f"{split}_{model}_{phase}.parents.csv",index_label="parent_group_id")
    contrasts=[]
    for split in cache:
        models=sorted({model for case,model,_ in matrices if case==split})
        for model in models:
            if model in ["composition-ridge","control-xgboost-C"]:
                continue
            contrast=bootstrap_contrast(matrices[(split,model,"perturbed")],matrices[(split,model,"clean")],args.bootstrap)
            contrasts.append(dict(split=split,candidate=model,reference=model,phase="perturbed-minus-clean",**contrast))
            for reference in [m for m in ["composition-ridge","control-xgboost-C"] if m in models]:
                for phase in ["clean","perturbed"]:
                    contrast=bootstrap_contrast(matrices[(split,model,phase)],matrices[(split,reference,phase)],args.bootstrap)
                    contrasts.append(dict(split=split,candidate=model,reference=reference,phase=phase,**contrast))
        if "physics-transformer" in models and "transformer" in models:
            for phase in ["clean","perturbed"]:
                contrast=bootstrap_contrast(matrices[(split,"physics-transformer",phase)],matrices[(split,"transformer",phase)],args.bootstrap)
                contrasts.append(dict(split=split,candidate="physics-transformer",reference="transformer",phase=phase,**contrast))
    (out/"comparisons.json").write_text(json.dumps(contrasts,indent=2,allow_nan=False))
    if any(file_sha256(p)!=digest for p,digest in inputs.items()):
        raise ValueError("Inputs changed during verification")
    (out/"protocol.json").write_text(json.dumps(dict(status="verified",input_hashes=inputs,cases=len(scores),
        full_spectral_matrix_required=args.require_full_matrix,
        environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
        qualification="Frozen held-out perturbation panel only. Same state masks and observation->state->parent aggregation. Composition predictions are unchanged because perturbations change only spectra. All seeds retained; no real-noise calibration.",
        arithmetic="Producer-native float32/float64 parent errors reproduced within 1e-10; reported independent scores and contrasts recomputed in float64, with native-rounding differences retained per run.",
        interval_policy="Parent-only and crossed seed/parent percentile intervals; exploratory comparisons."),indent=2))
    print(table.groupby(["split","model"])[["clean_mae","perturbed_mae"]].mean().to_string())


if __name__=="__main__":
    main()
