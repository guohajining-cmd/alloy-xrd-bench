"""Frozen held-out perturbation inference for matched-budget XGBoost controls."""

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from alloy_xrd_bench.hea_validation_data import vectorize_sro
from alloy_xrd_bench.sro_revision import file_sha256,apply_label_revision
from scripts.evaluate_sro_corrected_perturbations import matched_test_observations,load_raw,paired_errors
from scripts.run_sro_signal_controls import feature_sets


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True)
    parser.add_argument("--observation-dataset",type=Path,required=True)
    parser.add_argument("--revision",type=Path,required=True)
    parser.add_argument("--controls",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve()
    if out.exists() or out.is_relative_to(args.dataset.resolve()) or out.is_relative_to(ROOT/"data"):
        raise ValueError("Fresh output outside source data required")
    from alloy_xrd_bench.hea_validation_models import peak_features
    trained=json.loads((args.controls/"protocol.json").read_text())
    if trained["status"]!="complete" or not trained["input_hashes_unchanged"]:
        raise ValueError("Completed controls required")
    config=trained["config"]
    manifest=args.dataset/"manifests/task_samples.parquet"
    composition_path=args.dataset/"manifests/composition_only_features.parquet"
    observations_path=args.observation_dataset/"xrd/runs/hea_observation_v1_1/xrd_samples.parquet"
    if trained["source_manifest_sha256"]!=file_sha256(manifest) or trained["labels_sha256"]!=file_sha256(args.revision/"labels.parquet"):
        raise ValueError("Control data contract mismatch")
    inputs={str(p.resolve()):file_sha256(p) for p in [manifest,composition_path,observations_path,
            args.revision/"labels.parquet",args.controls/"protocol.json",Path(__file__),
            ROOT/"scripts/evaluate_sro_corrected_perturbations.py",ROOT/"scripts/run_sro_signal_controls.py"]}
    tasks=apply_label_revision(pd.read_parquet(manifest),manifest,args.revision,config["sro_targets"])
    fractions=[f"fraction_{e}" for e in config["elements"]]
    tasks=tasks.merge(pd.read_parquet(composition_path)[["structure_state_id","parent_group_id",*fractions]],
                      on=["structure_state_id","parent_group_id"],how="left",validate="one_to_one")
    if not np.isfinite(tasks[fractions].to_numpy()).all():
        raise ValueError("Missing composition features")
    observations=pd.read_parquet(observations_path)
    out.mkdir(parents=True)
    records=[]
    for split in config["split_columns"]:
        panel=matched_test_observations(observations,tasks,split)
        pairs=[vectorize_sro(r.sro_shell1_json,r.sro_shell2_json,config["sro_targets"]) for r in panel.itertuples()]
        truth,mask=np.stack([p[0] for p in pairs]),np.stack([p[1] for p in pairs])
        keep=mask.any(axis=1)
        panel,truth,mask=panel[keep].reset_index(drop=True),truth[keep],mask[keep]
        clean=panel.drop_duplicates("structure_state_id").reset_index(drop=True)
        clean_raw,axis=load_raw(args.dataset,clean,"clean_",inputs)
        pert_raw,pert_axis=load_raw(args.observation_dataset,panel,"",inputs)
        if not np.allclose(axis,pert_axis,atol=1e-6,rtol=0):
            raise ValueError("Axes disagree")
        with threadpool_limits(limits=4):
            max_peaks=config["model_budgets"]["peak_xgboost"]["max_peaks"]
            clean_peaks=np.stack([peak_features(row,axis,max_peaks=max_peaks) for row in clean_raw])
            pert_peaks=np.stack([peak_features(row,axis,max_peaks=max_peaks) for row in pert_raw])
        clean_features=feature_sets(clean[fractions].to_numpy(dtype=np.float32),clean_peaks)
        pert_features=feature_sets(panel[fractions].to_numpy(dtype=np.float32),pert_peaks)
        indexes=pd.Index(clean.structure_state_id).get_indexer(panel.structure_state_id)
        panel[["xrd_sample_id","structure_state_id","parent_group_id","clean_xrd_sample_id"]].to_csv(out/f"{split}.selection.csv",index=False)
        for feature in ["C","X","CX"]:
            name=f"{split}_{feature}"
            checkpoint=args.controls/"predictions"/f"{name}.joblib"
            saved_path=args.controls/"predictions"/f"{name}.parquet"
            for path in [checkpoint,saved_path]:
                digest=file_sha256(path)
                if digest!=trained["output_hashes"][f"predictions/{path.name}"]:
                    raise ValueError("Control artifact changed")
                inputs[str(path.resolve())]=digest
            with threadpool_limits(limits=4):
                estimator=joblib.load(checkpoint)
                clean_prediction=estimator.predict(clean_features[feature])
                pert_prediction=estimator.predict(pert_features[feature])
            saved=pd.read_parquet(saved_path).set_index("xrd_sample_id").loc[clean.clean_xrd_sample_id]
            if not np.array_equal(saved.parent_group_id,clean.parent_group_id) or not np.array_equal(np.stack(saved.sro_predictions),clean_prediction):
                raise ValueError(f"Clean control replay differs: {name}")
            if feature=="C" and not np.array_equal(clean_prediction[indexes],pert_prediction):
                raise ValueError("Composition-only predictions changed under spectral perturbation")
            model=f"control-xgboost-{feature}"
            filename=f"{model}_seed{trained['seed']}_{split}_test"
            parents=paired_errors(truth,mask,clean_prediction[indexes],pert_prediction,panel)
            parents.to_csv(out/f"{filename}.parents.csv")
            panel[["xrd_sample_id","structure_state_id","parent_group_id"]].assign(sro_predictions=list(pert_prediction)).to_parquet(out/f"{filename}.parquet",index=False)
            record=dict(model=model,seed=trained["seed"],split_column=split,clean_mae=float(parents.clean.mean()),
                        perturbed_mae=float(parents.perturbed.mean()),parent_count=len(parents),structure_count=panel.structure_state_id.nunique(),
                        observation_count=len(panel),clean_prediction_max_difference=0.)
            records.append(record)
            print(json.dumps(record),flush=True)
            del estimator
    pd.DataFrame(records).to_csv(out/"scores.csv",index=False)
    if any(file_sha256(path)!=digest for path,digest in inputs.items()):
        raise ValueError("Input changed during control perturbation evaluation")
    protocol=dict(status="verified",source_hashes=inputs,runs=[[r["model"],r["seed"],r["split_column"]] for r in records],
                  completed_run_count=len(records),qualification="Matched frozen test observation panel; clean control replay exactly equal; synthetic perturbations only, no retraining or noise tuning.")
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2))
    print(f"VERIFIED {len(records)} control perturbation cases",flush=True)


if __name__=="__main__":
    main()
