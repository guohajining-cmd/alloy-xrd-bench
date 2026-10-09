"""Render three distinct, verified scopes of the spectral incremental-value audit."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from alloy_xrd_bench.sro_revision import file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full",type=Path,required=True)
    parser.add_argument("--fixed-cell",type=Path,required=True)
    parser.add_argument("--perturbed",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(ROOT/"data"):
        raise ValueError("Fresh non-data output required")
    inputs={}
    for directory in [args.full,args.fixed_cell,args.perturbed]:
        if json.loads((directory/"protocol.json").read_text())["status"]!="verified":
            raise ValueError("Verified analyses required")
        for path in directory.glob("*.json"):
            inputs[str(path.resolve())]=file_sha256(path)
    full=json.loads((args.full/"comparisons.json").read_text())
    fixed=json.loads((args.fixed_cell/"comparisons.json").read_text())
    noisy=pd.read_csv(args.perturbed/"scores.csv")
    inputs[str((args.perturbed/"scores.csv").resolve())]=file_sha256(args.perturbed/"scores.csv")
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                         "pdf.fonttype":42,"font.family":"DejaVu Sans"})
    fig,axes=plt.subplots(1,3,figsize=(16,4.8),gridspec_kw={"width_ratios":[1.05,1.1,1.1]})
    fig.subplots_adjust(left=.13,right=.985,bottom=.28,top=.79,wspace=.62)
    split_names={"id_split":"ID","composition_ood_split":"Composition OOD",
                 "chemistry_ood_split":"Chemistry OOD","sro_ood_split":"Frozen SRO OOD"}
    sources=[]
    ax=axes[0]
    for i,(split,label) in enumerate(split_names.items()):
        row=next(r for r in full if r["split"]==split and r["candidate"]=="control-xgboost-CX" and r["reference"]=="control-xgboost-C")
        v=row["relative_reduction_percent"]
        lo,hi=row["relative_ci95_parent"]
        ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt="o",color="#147D70",capsize=3)
        ax.text(29.3,i,f"{v:.1f}%",va="center",ha="right",fontsize=8)
        sources.append(dict(panel="full",group=split,estimate=v,low=lo,high=hi))
    ax.set(yticks=range(4),yticklabels=list(split_names.values()),xlim=(0,30),xlabel="Relative MAE reduction (%)")
    ax.invert_yaxis()
    ax.set_title("A  Adding peaks to composition\nFull frozen test splits; fixed-budget XGBoost",loc="left",fontsize=10,pad=15)
    models=["peak-xgboost","resnet1d","transformer","physics-transformer"]
    names=["Peak-XGB","ResNet","Transformer","Physics-T"]
    ax=axes[1]
    pair_models=["control-xgboost-CX",*models]
    for i,model in enumerate(pair_models):
        row=next(r for r in fixed if r["split"]=="id_split" and r["model"]==model)
        v=row["delta_mae"]
        lo,hi=row["delta_ci95_seed_parent"]
        ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt="o",color="#52616B",capsize=3)
        sources.append(dict(panel="fixed_cell",group=model,estimate=v,low=lo,high=hi))
    ax.axvline(0,color="#999999",linestyle="--",linewidth=1)
    ax.set(yticks=range(5),yticklabels=["C+X XGB",*names],xticks=[-.015,0,.015],
           xlabel="SRO-change MAE minus zero-change baseline")
    ax.invert_yaxis()
    ax.set_title("B  Fixed-cell occupancy changes\nID held out: 150 parents, 600 pairs",loc="left",fontsize=10,pad=15)
    ax=axes[2]
    for i,model in enumerate(pair_models):
        part=noisy[(noisy.split=="id_split") & (noisy.model==model)].sort_values("seed")
        offsets=np.linspace(-.15,.15,len(part))
        ax.scatter(i+offsets,part.perturbed_mae,color="#BF503C",s=24,zorder=3)
        for row in part.itertuples(index=False):
            sources.append(dict(panel="perturbed",group=model,seed=row.seed,estimate=row.perturbed_mae))
    reference=float(noisy[(noisy.split=="id_split") & (noisy.model=="composition-ridge")].perturbed_mae.iloc[0])
    ax.axhline(reference,color="#147D70",linestyle="--",label=f"Matched Ridge: {reference:.3f}")
    ax.set(xticks=range(5),xticklabels=["C+X XGB",*names],yscale="log",ylabel="Perturbed parent-macro MAE (log scale)")
    ax.tick_params(axis="x",rotation=25)
    ax.legend(loc="upper right",fontsize=8,frameon=False)
    ax.set_title("C  Frozen synthetic perturbations\nSame 150 held-out parents; fitted seeds",loc="left",fontsize=10,pad=15)
    fig.suptitle("Alloy-XRD-Bench: spectral gains depend on the evaluation scope",fontsize=14,y=.96)
    fig.text(.13,.065,"A: 95% paired-parent intervals, one fitted seed.  B: 95% crossed seed/parent intervals, five seeds (C+X XGB: one).\n"
                        "C: all five seeds shown (C+X XGB: one); 750 states / 3,750 views. No experimental SRO validation or real-noise calibration.",fontsize=9)
    args.output.mkdir(parents=True)
    pd.DataFrame(sources).to_csv(args.output/"figure_source.csv",index=False)
    fig.savefig(args.output/"signal_evidence.png",dpi=180,facecolor="white")
    fig.savefig(args.output/"signal_evidence.pdf",facecolor="white")
    plt.close(fig)
    if any(file_sha256(p)!=h for p,h in inputs.items()):
        raise ValueError("Inputs changed while plotting")
    (args.output/"metadata.json").write_text(json.dumps(dict(inputs=inputs,scope="Three different estimands; not a cross-track composite score"),indent=2))


if __name__=="__main__":
    main()
