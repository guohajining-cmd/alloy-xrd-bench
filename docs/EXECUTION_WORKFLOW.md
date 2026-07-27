# Execution Workflow

This workflow keeps MLIP storage, generated data, and provenance aligned.

## 1. Register A Potential

Create or update:

```text
data/registry/potentials_catalog.yml
```

Each row must include:

- `potential_id`
- `family`
- `engine`
- `elements`
- `phases`
- `source_url`
- `doi`
- `license`
- `file_sha256`
- `redistribution_policy`
- `citation_keys`
- `potential_dir`
- `generated_data_dir`

The storage fields must be:

```text
potential_dir: data/potentials/{potential_id}/potential
generated_data_dir: data/potentials/{potential_id}/generated
```

Use the registration CLI so the catalog, metadata, checksum file, and directory
layout stay synchronized.

Do not manually create placeholder directories under `data/potentials/`. Every
directory there must be registered in `data/registry/potentials_catalog.yml`;
the validator reports unregistered potential directories as release-blocking
issues.

For a redistributable local potential file:

```powershell
python -m alloy_xrd_bench.potentials register `
  --root . `
  --potential-id pot_example `
  --family MACE `
  --engine mace `
  --elements Cr Co Ni `
  --phases fcc `
  --source-url https://example.org/potential `
  --doi 10.0000/example `
  --license MIT `
  --redistribution-policy redistributable `
  --citation-key example_mlip `
  --file path\to\model.pt
```

For a non-redistributable or externally hosted potential:

```powershell
python -m alloy_xrd_bench.potentials register `
  --root . `
  --potential-id pot_external `
  --family CHGNet `
  --engine chgnet `
  --elements Ti Ta V W `
  --phases bcc `
  --source-url https://example.org/external `
  --doi 10.0000/external `
  --license custom `
  --redistribution-policy metadata_only `
  --citation-key external_mlip `
  --file-sha256 bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
```

Repeated registration of the same `potential_id` updates that catalog row rather
than creating duplicates.

## 1a. Audit MLIP Candidates Before Admission

Pre-admission candidates live in:

```text
data/registry/mlip_candidates.yml
```

This file records general baselines, stronger general models, and HEA-specific
candidate families before they become accepted benchmark potentials. Candidate
rows must not create folders under `data/potentials/`; only accepted
`potentials_catalog.yml` rows may do that.

Audit the candidate registry:

```powershell
python -m alloy_xrd_bench.potentials candidates audit `
  --root . `
  --json
```

Report a candidate download/install action without staging it as benchmark data:

```powershell
python -m alloy_xrd_bench.potentials download `
  --root . `
  --candidate-id pot_crconi_mtp_csro `
  --json
```

External candidate model files are cached under:

```text
local_potential_cache/{candidate_id}/
  downloads/
  download_audit.json
  checksums.sha256
```

For large archive-hosted datasets, candidate rows may declare `zip_members`.
The downloader extracts only those member files into `downloads/` and does not
stage the full archive.

This cache is ignored by git and is not part of release artifacts. A candidate
still remains out of `data/potentials/` until model files, license, engine, hash,
and smoke/reference inference are accepted. Current target-specific candidate
states are:

- Round-35 MTP audit: 17 downloaded `.mtp`/`.mtp_` files were header-parsed and
  SHA256-recorded, 16 showed trained coefficients, and the DaRUS Ni3Al
  `24.mtp` file appears header/template-like while `Trained.mtp_` is the
  coefficient-bearing file. MLIP-3 and LAMMPS/MLIP interface sources are
  cached. At round-35 no `mlp` executable or local C++ toolchain was on PATH;
  after the round-48 WSL MLIP-3 build, `pot_crconi_mtp_csro` and
  `pot_ni3al_mtp_darus_yield_anomaly_2024` have since been admitted through
  `mtp_mlp`, while other MTP candidates still need mapping/license/runtime
  closure before acceptance.
- `pot_crconi_mtp_csro`: CrCoNi MTP files downloaded and round-35 integrity-checked. Round-46 found bundled LAMMPS, but it exposes `mliap`, not Shapeev MTP/user-mlip. Round-48 built audit-only MLIP-3 `mlp` and completed reference EFS inference, round-49 confirmed the source type map and smoke relaxation, and round-50 accepted the model through the reproducible `mtp_mlp` runner with 5 formal CrCoNi relaxed structures.
- `pot_crconi_nep_wu_2026`: CrCoNi NEP `nep.txt` found in the MIT-licensed `mushroomfire/NEP-CrCoNi` repository, downloaded, hash-recorded, registered as an accepted target-specific potential, verified through `mdapy.NEP4ASE` reference inference plus CrCoNi single-point smoke, and round-45 generated 5 formal converged/pass CrCoNi relaxed structures. The NEP adapter now ignores `nep.in` input files when selecting the model file.
- Round-37 gamma/gamma-prime checkpoint search: `audits/mlip_gamma_prime_public_checkpoint_search_round37_2026-07-05.json` records fresh GitHub, Hugging Face, Zenodo, Crossref, ScienceDirect, materialsmodeling.org NEP-table, arXiv, and SMC-X supplementary-repository checks. No public Ni-Al-Ti, Ni-Al-Ta, Ni-Al-Ti-Ta, Ni-Co-Al, or broader Ni-based gamma/gamma-prime checkpoint with license/checksum/engine recipe was found, and no formal generated manifest changed.
- `pot_rhea9_nep_byggmastar_2026`: RHEA nine-element refractory-alloy NEP `nep-RHEA-large.txt` found in Zenodo record `10.5281/zenodo.18863415`, redistributed under CC-BY-4.0, copied into the formal potential workspace, hash-recorded, registered as an accepted refractory-alloy-specific potential, verified through `mdapy.NEP4ASE` reference/smoke checks on WTaCrV, MoNbTaVW, and TiTaVW public initial structures, and round-45 generated 15 formal converged/pass refractory relaxed structures. Round-38 evidence is in `audits/mlip_rhea9_nep_byggmastar_2026_round38_2026-07-05.json`.
- `pot_heaml_pointff_wu_2024_lead`: CrFeCoNiPd HEA PointFF/TorchScript model file and custom LAMMPS pair-style source were downloaded from the public `lianpingWu/HEAML.github.io` repository and hash-recorded. `torch.jit.load` succeeds on `jit_epoch.pth`, but the candidate stays blocked because there is no explicit repository license/README, no PointFF-enabled LAMMPS/LibTorch runtime is available, the atom-type mapping still needs confirmation, and it is not an exact current benchmark target. Round-39 evidence is in `audits/mlip_heaml_pointff_wu_2024_round39_2026-07-05.json`.
- Round-40 gamma/gamma-prime checkpoint search: `audits/mlip_gamma_prime_public_checkpoint_search_round40_2026-07-05.json` records fresh ScienceDirect/Crossref, Elsevier supplement-path, GitHub, Hugging Face, Zenodo, `brucefan1983/nep-data`, and Ni-Al MTP repository checks. No new admittable Ni-Al-Ti, Ni-Al-Ta, Ni-Al-Ti-Ta, or Ni-Co-Al checkpoint was found; the Ni-Co-Al NEP remains request-only, and the Ni-Al MTP remains blocked by missing license plus MTP runtime.
- Round-41 MTP/HDNNP runtime probe: `audits/mlip_mtp_hdnnp_runtime_resolution_round41_2026-07-05.json` confirms that the downloaded Ni-Al/Ni3Al `.mtp` files and Al-Ni HDNNP package are still blocked by runtime, not by missing model files. At that point no `mlp`, `lmp`, or `lammps` executable was on PATH; Python `lammps` could not load `liblammps.dll`; PyPI `mlip` is not the Shapeev MTP runtime; MAML wraps MTP/LAMMPS but still calls external `mlp`/LAMMPS; and `n2p2`/`pynnp` have no PyPI package in this environment. No accepted potential or generated manifest changed.
- Round-42 gamma/gamma-prime checkpoint search: `audits/mlip_gamma_prime_public_checkpoint_search_round42_2026-07-05.json` records fresh public web, GitHub repository, Hugging Face model, Zenodo, and DataCite searches for Ni-Al-Ti, Ni-Al-Ta, Ni-Al-Ti-Ta, and Ni-Co-Al targets. GitHub/Hugging Face exact queries returned zero target checkpoint hits; Zenodo/DataCite hits were unrelated, classical/literature/data-only, or non-MLIP; GitHub code search requires authentication and returned HTTP 401. No accepted potential or generated manifest changed.
- Round-43 author/repository probe: `audits/mlip_gamma_prime_author_repo_probe_round43_2026-07-05.json` records checks of Marium Mou's public homepage/GitHub profile, known NequIP/Allegro support discussions, LAM-DPA FCC+L12, SMC-X supplementary, and LightPFP. The new public Mou repository is a high-entropy carbide MACE-OMAT workflow, not a Ni-based gamma/gamma-prime checkpoint. LAM-DPA still has only `input_dpa3ft.json` in FCC+L12, LightPFP still has datasets/configs/results without a standalone checkpoint, and SMC-X still has no direct model file. No accepted potential or generated manifest changed.
- Round-44 strong-lead follow-up: `audits/mlip_nialti_nicoal_strong_lead_followup_round44_2026-07-05.json` confirms the TMS 2026 Ni-Al-Ti MACE/FitSNAP abstract and the Materials & Design/GPUMD Ni-Co-Al NEP publication are both target-relevant, but neither source exposes a downloadable checkpoint, license, checksum, or runnable engine recipe. Fresh GitHub repository, Hugging Face, Zenodo, DataCite, GitLab `brucefan1983/nep-data`, and GPUMD tree checks found no public Ni-Al-Ti MACE/FitSNAP model or Ni-Co-Al `nep.txt`. No accepted potential or generated manifest changed.
- Round-45 accepted-NEP formal relaxation: `audits/mlip_accepted_nep_formal_relaxation_round45_2026-07-05.json` records the first formal target-specific NEP relaxed structures. The ASE-backed MLIP runner now skips unsupported initial structures before writing formal manifests. `pot_crconi_nep_wu_2026` generated 5 converged/pass CrCoNi relaxed JSON/CIF pairs; `pot_rhea9_nep_byggmastar_2026` generated 15 converged/pass TiTaVW, MoNbTaVW, and WTaCrV relaxed JSON/CIF pairs. No new XRD samples were generated.
- Round-46 LAMMPS/DeepMD runtime resolution: `audits/mlip_lammps_deepmd_runtime_resolution_round46_2026-07-05.json` records that bundled/user Python `lmp.exe` and `dp.exe` are discoverable. LAMMPS exposes `snap`, `mliap`, `meam`, and `uf3`, but not `deepmd`, `pace`, `hdnnp`, `mlip`, `quip`, or `tabgap`; `dp --pt show` on the downloaded MoNbTaVW DPA2 model still fails with a DeepMD/PyTorch `CXX11_ABI_FLAG` mismatch. No accepted potential or generated manifest changed.
- Round-47 gamma/gamma-prime and Cantor MTP follow-up: `audits/mlip_gamma_prime_and_cantor_mtp_followup_round47_2026-07-05.json` records fresh GitHub, Hugging Face, Zenodo, GPUMD `nep-data`, and NEP model-table checks. No public Ni-Al-Ti, Ni-Al-Ta, Ni-Al-Ti-Ta, or Ni-Co-Al gamma/gamma-prime checkpoint was found. The CoCrFeMnNi Cantor MTP repository was rechecked and confirmed to expose `Trained_MTP_models` plus GPL-3.0 license text, so that candidate remains found-but-MTP-runtime-blocked rather than missing-model.
- Round-48 MLIP-3/MTP runtime follow-up: `audits/mlip3_mtp_runtime_reference_round48_2026-07-05.json` records a local WSL serial MLIP-3 `mlp` build in `local_potential_cache/source_documents/mlip-3-wsl-build-round48/bin/mlp`. Because WSL has no `gfortran` and sudo requires a password, the build copy uses a minimal CBLAS shim and is audit-only. The resulting `mlp` passed upstream `calculate_efs` and completed reference EFS inference for `pot_crconi_mtp_csro` on CrCoNi and `pot_cocrfemnni_mtp_cantor_alloy_mlip` on CoCrFeMnNi.
- Round-49 MTP mapping/smoke follow-up: `audits/mlip_mtp_mapping_and_smoke_round49_2026-07-05.json` confirms the CrCoNi cSRO MTP type map from the source README as `{0: Cr, 1: Co, 2: Ni}` and records an audit-only `mlp relax` smoke pass for `init_CrCoNi_fcc_Co1Cr1Ni1_seed000007` with relaxed max force 0.04208459415273005 eV/A. `pot_crconi_mtp_csro` is now blocked only on reproducible `mtp_mlp` adapter/config integration and catalog admission. The CoCrFeMnNi Cantor MTP remains blocked because the source README, validation LAMMPS scripts, and reference POSCAR expose conflicting element orders.
- Round-50 formal CrCoNi MTP admission: `audits/mlip_crconi_mtp_csro_formal_admission_round50_2026-07-05.json` records the new `mtp_mlp` runner, `pot_crconi_mtp_csro` catalog admission, packaged MIT `.mtp` model file, verification pass, and 5 formal converged/pass CrCoNi relaxed JSON/CIF structures. This accepted potential writes only under `data/potentials/pot_crconi_mtp_csro/`; no XRD was generated in this phase.
- Round-51 formal Ni3Al MTP admission: `audits/mlip_ni3al_mtp_darus_yield_anomaly_formal_admission_round51_2026-07-05.json` records `pot_ni3al_mtp_darus_yield_anomaly_2024` catalog admission through the `mtp_mlp` runner, packaged CC-BY-4.0 DaRUS `Ni3Al.mtp`, an inferred `{0: Al, 1: Ni}` type map from DaRUS `train.cfg` Ni3Al stoichiometry, verification pass, phase-scope skip behavior for fcc smoke structures, and 2 formal converged/pass Ni-Al L12 relaxed JSON/CIF structures. This accepted potential writes only under `data/potentials/pot_ni3al_mtp_darus_yield_anomaly_2024/`; no XRD was generated in this phase.
- `pot_crconi_ace_sheriff_motif_2025`: CrCoNi motif-based ACE/PACE `.yace` file downloaded; blocked on LAMMPS/PACE or ASE/PyACE.
- `pot_cocrfemnni_mtp_cantor_alloy_mlip`: CoCrFeMnNi MTP files downloaded and round-35 integrity-checked; round-47 rechecked the public `Trained_MTP_models` directory and GPL-3.0 license. Round-48 built audit-only MLIP-3 `mlp` and completed reference EFS inference with an assumed type map; round-49 found conflicting source mapping evidence, so it is blocked until the intended MTP type-index mapping is clarified.
- `pot_cocrfeni_cocrni_mtp_nitol_2026`: CoCrFeNi/CoCrNi fcc MTP files downloaded from the public GitLab repository and hash-recorded in the registry; useful for CrCoNi/CoCrNi-style fcc structures, but blocked on missing explicit repository license and LAMMPS/user-mlip or MLIP-3 runtime.
- `pot_titavw_ace_sheriff_motif_2025`: TiTaVW motif-based ACE/PACE `.yace` file downloaded; blocked on LAMMPS/PACE or ASE/PyACE.
- `pot_titavw_ace_sheriff_chemical_mixing_2025`: TiTaVW chemical-mixing ACE/PACE `.yace` file downloaded; blocked on LAMMPS/PACE or ASE/PyACE.
- `pot_titavw_milady_wrobel_2025`: Ti-V-Ta-W linear and kernel ML potentials from Zenodo downloaded and hash-recorded; blocked on LAMMPS `pair_style milady` / MILADY adapter setup.
- `pot_rhea9_nep_byggmastar_2026`: RHEA nine-element refractory-alloy NEP model and README extracted from Zenodo `RHEA-data.zip`; round-33 verified `mdapy.NEP4ASE` reference inference, round-38 formally accepted the model after WTaCrV, MoNbTaVW, and TiTaVW smoke checks, and round-45 generated 15 formal converged/pass refractory relaxed structures.
- `pot_rhea9_tabgap_byggmastar_2026`: RHEA nine-element refractory-alloy tabGAP EAM companion and README extracted; primary HDF5 tabGAP member is resolved but skipped until explicit staged transfer and LAMMPS/tabGAP setup.
- `pot_unep_v1_16metals_without_zbl` and `pot_unep_v1_16metals_with_zbl`: UNEP-v1 16-metal NEP models downloaded from Zenodo; round-33 verifies `mdapy.NEP4ASE` reference inference for Ni-Al-Ti-Ta smoke structures. They are useful metal-domain baselines, but not gamma/gamma-prime-specific MLIPs.
- `pot_alloy53_dpa1_vasp_dai_2026_aissquare`: Alloy53/DPA-1 53-metal AISSquare model-list entry found from the 2026 npj Computational Materials Alloy53 article; anonymous detail/download currently returns HTTP 401, so no model hash is available yet.
- `pot_alloy53_dpa1_2_multitask_wen_2024_aissquare`: AISSquare `DPA-1&2-53-alloy-multitask-400w` model-list entries found; anonymous detail/download currently returns HTTP 401, so no model hash is available yet.
- `pot_nep89_gpumd_2025`: GPUMD NEP89 89-element model downloaded and hash-recorded; round-33 verifies `mdapy.NEP4ASE` reference inference for a Cr-Co-Ni smoke structure. It covers all current target elements, but is a broad general baseline and not gamma/gamma-prime-specific.
- `pot_nequip_oam_l_0p1` and `pot_allegro_oam_l_0p1`: NequIP/Allegro OAM-L foundation `.nequip.zip` artifacts downloaded from Zenodo record `10.5281/zenodo.18775904` and hash-recorded; they cover the current HEA and Ni-Al-Ti-Ta element sets as broad inorganic foundation baselines, but are not target-trained HEA or gamma/gamma-prime MLIPs and remain blocked on NequIP/Allegro load/reference inference and smoke relaxation.
- `pot_wtacrv_tabgap_byggmastar_2025`: WTaCrV HDF5 LAMMPS tabGAP package and GAP XML zip downloaded; blocked because round-46 found no tabGAP custom pair style in the bundled LAMMPS.
- `pot_monbtavw_tabgap_byggmastar`: Fairdata authorized paths resolved; primary 842948145-byte `.tabgap`, companion EAM/README files, and pair-style source files downloaded and hash-recorded; blocked because round-46 found no tabGAP custom pair style in the bundled LAMMPS.
- `pot_monbtavw_mtp_wang_2024`: GitHub MoNbTaVW `fitted.mtp`, `mlip.ini`, and `quinary.json` downloaded, hash-recorded, and round-35 integrity-checked; this is an exact MoNbTaVW target-specific MTP candidate, blocked on missing repository license and LAMMPS/user-mlip or MLIP-3 smoke.
- `pot_monbtaw_mtp_ufsf_2025`, `pot_monbtaw_mtp_mvl_2021`, and `pot_nbmotaw_snap_mvl_2021`: MoNbTaW MTP/SNAP model files downloaded and hash-recorded as refractory supporting candidates; they lack V and therefore do not replace exact MoNbTaVW runs. Round-32 adds `snap_lammps` adapter support and verifies that the NbMoTaW SNAP model runs through installed LAMMPS ML-SNAP after a temporary compatibility transform removing the obsolete `diagonalstyle` parameter.
- `pot_monbtavw_dpa2_lam_jiang_2026`: Ta-Nb-W-Mo-V DPA2/DeepMD distilled and finetune checkpoint/config/ZBL files downloaded from the LAM-DPA GitHub repository and hash-recorded; blocked on missing repository license and DeePMD/LAMMPS `deepmd` engine setup.
- `pot_monbtavw_dpa2_lam_all_jiang_2026`: Ta-Nb-W-Mo-V DPA2-All/DeepMD distilled and finetune checkpoint/config/ZBL files downloaded from the same repository and hash-recorded; blocked on missing repository license and DeePMD/LAMMPS `deepmd` engine setup.
- `pot_monbtavw_dp_lam_jiang_2026` and `pot_monbtavw_dp_lam_all_jiang_2026`: Ta-Nb-W-Mo-V DP-5E/DP-All checkpoints downloaded from the LAM-DPA GitHub repository and hash-recorded; blocked on missing repository license and DeePMD/LAMMPS `deepmd` engine setup.
- Round-30 adds a `deepmd_lammps` readiness adapter so DPA/DP candidates now fail with structured `missing_engine`/`missing_model` audit records rather than an unsupported-engine path. This does not accept any DeepMD candidate yet because `deepmd`, `lammps`, `lmp`, and `dp` are still absent locally and license/smoke gates remain open.
- Round-31 installs `deepmd-kit==3.1.3` and `lammps==2025.7.22.4.0` into the user Python environment. `deepmd`, `lammps`, `dp.exe`, and `lmp.exe` are present, but the installed LAMMPS wheel does not recognize `pair_style deepmd`, and `dp --pt show` on the downloaded MoNbTaVW DPA/DP checkpoints fails with a PyTorch CXX11 ABI mismatch. Round-46 confirms the executables are discoverable by the adapter helper, but the `deepmd` pair style and ABI blockers remain. These candidates therefore remain blocked until a DeePMD-enabled LAMMPS build or compatible DeepMD/PyTorch runtime passes reference inference or smoke relaxation.
- The same LAMMPS probe found `snap`, `mliap`, `meam`, and `uf3` reachable, but `pace`, `hdnnp`, and `quip` are not enabled. This means `.yace` ACE/PACE and n2p2 HDNNP candidates still need a different engine build, while SNAP-family supporting candidates are now the most practical next engine-smoke target.
- `pot_nicofecraltita_dpa3_lam_jiang_2026_lead`: Ni-Co-Fe-Cr-Al-Ti-Ta FCC+L12 DPA3 `input_dpa3ft.json` training config found, downloaded, and hash-recorded. Round-29 rechecked the current GitHub main tree, branch list, and releases: the FCC+L12 path still contains only the training config, the repository has no release and no explicit license, and no trained checkpoint/model file is present, so it is tracked as `research_needed`.
- `pot_mncofenicu_nep_unimi_2026`: UNIMI Dataverse CC0 Mn-Co-Fe-Ni-Cu NEP/GPUMD archive set found, downloaded, and hash-recorded; three `nep.txt` files were extracted for audit, and round-33 verifies `mdapy.NEP4ASE` reference inference. This is useful HEA-related NEP coverage, but it lacks Cr/Ti/Ta/Mo/Nb/V/W and therefore cannot replace exact CrCoNi, CoCrFeMnNi, refractory bcc, or gamma/gamma-prime target-specific potentials.
- `pot_nialti_mace_fitsnap_song_2026_lead`: 2026 TMS ProgramMaster abstract reports Ni-Al-Ti gamma/gamma-prime ML-IAPs trained with MACE/FitSNAP NN approaches; exact author/title, GitHub, Zenodo, Materials Cloud, Hugging Face, and AISSquare searches found no public checkpoint/repository/license. Round-5 author/index checks confirm the ProgramMaster page describes MACE/FitSNAP potentials trained from Quantum Espresso AIMD data, but exposes no model artifact. Round-6 code-host/profile checks found no GitLab, Codeberg, LANL, ORCID, NOMAD, or Materials Cloud artifact. Round-7 preprint/supplement checks found no OSTI, NSF-PAR, ResearchSquare, or ChemRxiv/Crossref artifact. Round-8 file-pattern searches found no Hugging Face, Mendeley Data, Dryad, CORE, or OpenAIRE model artifact. Round-9 primary-source follow-up confirms ProgramMaster has only planned proceedings status and no artifact/download/license/checksum links. Round-21 and round-22 rechecked the ProgramMaster primary page plus public model indexes. Round-26 re-read the primary ProgramMaster page, confirmed Huajing Song/Pratt and Whitney evidence from the TMS 2026 program, and again found no repository, checkpoint, training archive, license, or checksum. Round-44 and round-47 repeated public model-index checks; the lead is real and target-relevant but still has no public model file, so it remains `research_needed`.
- `pot_nibased_gamma_prime_mou_2023_lead`: BearWorks thesis metadata reports Ni-based superalloy Gamma/Gamma Prime MLIPs using DeePMD, NequIP/Allegro, and MTP; no public checkpoint/archive/license is exposed, the PDF request returned an AWS WAF challenge/403 HTML in this environment, and the author's public GitHub repositories do not expose a Ni-based gamma/gamma-prime model. Round-12 follow-up confirmed the APS 2022 and 2023 abstracts, author homepage, and NC State profile as real literature evidence, then re-checked GitHub, Hugging Face, and Zenodo with zero target checkpoint hits. Round-43 found and checked the author's current public GitHub repos; the only new relevant-looking repo is a high-entropy carbide MACE-OMAT workflow, not a target checkpoint.
- `pot_nial_mtp_wang_2025_candidate`: Ni-Al `potential.mtp` downloaded and hash-recorded in the registry. Round-51 audit-only MLIP-3 checks pass for Ni-Al fcc/L12 single-point and smoke relaxation using the README type map `0=Al, 1=Ni`; it remains blocked from accepted/formal generation because the source repository has no explicit license or repository license metadata.
- `pot_nial_hdnnp_sandberg_2024`: Materials Cloud Al-Ni HDNNP/n2p2 potential package downloaded and hash-recorded; blocked on n2p2/LAMMPS HDNNP engine setup and does not cover Ti/Ta substitutions. Round-41 confirmed no PyPI `n2p2`/`pynnp` fallback and no loadable local LAMMPS runtime.
- `pot_ni3al_mtp_darus_yield_anomaly_2024`: DaRUS Ni3Al L12 `Ni3Al.mtp`, `train.cfg`, and small thermodynamic/planar-defect tables are downloaded and hash-validated. Round-51 accepted it through `mtp_mlp`, packaged the model in its own potential workspace, verified load/reference/smoke inference, and generated 2 formal Ni-Al L12 relaxed JSON/CIF structures. It is gamma-prime-specific and does not cover fcc gamma or Ti/Ta substitutions.
- `pot_ni3al_mtp_darus_spin_fluctuations_2023`: second DaRUS Ni3Al L12 MTP source identified; round-13 partially transferred the large `MTP-training-set.tar.gz` archive, listed the early tar members, and extracted/hash-recorded `MTP-training-set/mtp-training/24.mtp` and `MTP-training-set/mtp-training/Trained.mtp_`. Round-35 classifies `24.mtp` as header/template-like and `Trained.mtp_` as coefficient-bearing. Round-41 confirmed the model-file gate is effectively closed for `Trained.mtp_`, but LAMMPS/user-mlip or MLIP-3 smoke is still not configured.
- `pot_ni3al_lightpfp_research_lead_2025`: LightPFP Ni3Al MTP lead recorded as `research_needed`; examples/data are public, but the engine depends on Matlantis commercial components and no standalone public checkpoint was found. Round-43 rechecked the repository tree and releases: still datasets, configs, notebooks, and results, not a redistributable trained checkpoint.
- `pot_alcocrfeni_lightpfp_2025_lead`: LightPFP HEA reproduction-data lead recorded as `research_needed`; the public repository has Al-Co-Cr-Fe-Ni HEA HDF5 datasets, CIF inputs, training JSON, and property notebooks, but no standalone trained checkpoint or redistributable runtime. It is a useful HEA lead, not an accepted CrCoNi/CoCrFeMnNi potential.
- `pot_nicoal_nep_qin_2025_lead`: Ni-Co-Al NEP gamma/gamma-prime lead recorded as `research_needed`; the article states data are available on request, and no public `nep.txt` model file was found. Round-26 checked Crossref relations and common Elsevier supplementary-file routes for the primary article and found no public model archive, checksum, or license. Round-42 exact public web, GitHub repository, Hugging Face, Zenodo, and DataCite searches still found no public `nep.txt` or checkpoint. Round-44 rechecked ScienceDirect, GPUMD publications, GitHub, Hugging Face, Zenodo, DataCite, GitLab `nep-data`, and the GPUMD source tree; the publication is confirmed but the checkpoint remains request-only.
- `pot_nicrfealp_mtp_ning_2026_lead`: Ni-Cr-Fe-Al-P nickel-polycrystal MTP lead recorded as `research_needed`; Supplementary Data 1 was downloaded and hash-recorded, confirms Ni3Al/Ni3P and Ni54Cr20Fe20Al5P1 training coverage, but exposes no runnable `.mtp` model file and does not cover Ti/Ta gamma-prime substitutions.
- `pot_nialti_meam_kim_2017_classical_fallback`: NIST Ni-Al-Ti MEAM `library.meam` and `NiAlTi.meam` files downloaded and hash-recorded; useful only as a classical gamma-prime fallback, not as a dedicated MLIP.
- `pot_nialcoti_meam_huang_2024_classical_fallback`: NIST Ni-Al-Co-Ti MEAM `library.meam` and `NiAlCoTi.meam` files downloaded and hash-recorded; useful only as a classical L12/gamma-prime fallback, not as a dedicated MLIP.
- `pot_nialti_meam_sharifi_2025_classical_fallback`: NIST Ni-Ti-Al MEAM `library.meam` and `NiTiAl.meam` files downloaded and hash-recorded; useful as a second Ni-Al-Ti classical fallback only, not as a dedicated MLIP.
- `pot_nicoal_meam_sharifi_2025_classical_fallback`: NIST Ni-Co-Al MEAM `library.meam` and `NiCoAl.meam` files downloaded and hash-recorded; useful as a Ni-Co-Al classical fallback only, not as the missing Ni-Co-Al NEP/MLIP checkpoint.
- `pot_nialta_eam_fan_2016_classical_lead`: Ni-Al-Ta gamma-prime EAM publication lead recorded from the 2016 Computational Materials Science article; no public potential file was found in the checked NIST/OpenKIM/GitHub/Zenodo endpoints, and EAM remains classical rather than MLIP.
- `pot_altiv_mtp_nitol_2025`: Al-Ti-V MTP files downloaded from NIST; useful as related Ti/Al MLIP evidence, but not a Ni-Al-Ti gamma/gamma-prime potential because it lacks Ni.
- `pot_tial_polymlp_seko_2022p3_openkim`: Ti-Al OpenKIM PolyMLP files downloaded; useful as related Ti/Al MLIP evidence, but not a Ni-Al-Ti gamma/gamma-prime potential because it lacks Ni.
- `pot_tial_mtp_mvl_2023`: Ti-Al Materials Virtual Lab MTP files downloaded; useful as related Ti/Al MLIP evidence, but not a Ni-Al-Ti gamma/gamma-prime potential because it lacks Ni.
- `pot_nialti_mlip_candidate`, `pot_nialta_mlip_candidate`, and `pot_nialtita_mlip_candidate`: no public MLIP model/checkpoint found yet; classical MEAM/EAM leads are fallback-only, UNEP-v1, NEP89, NequIP-OAM-L, Allegro-OAM-L, and OpenKIM TorchML MACE are baseline-only, the seven-element Fe-Co-Ni-Al-Ti-Ta-V arXiv lead exposes no usable checkpoint, round-4 Figshare/OSF/Dataverse/DaRUS/DataCite checks found no new admittable model, round-5 GitHub/Hugging Face exact searches returned zero target repositories/models, round-6 GitLab/Codeberg/profile/NOMAD checks found no target artifact, round-7 preprint/supplement indexes found no target checkpoint, round-8 explicit file-pattern searches found no target model artifact, round-9 ProgramMaster follow-up found no source-page artifact link, round-10 GitHub/Hugging Face/Zenodo exact searches still found no gamma/gamma-prime checkpoint, round-11 separated Ni-based polycrystal/CrCoNi/NiCoCr article or structure-data leads from actual model files, round-12 re-checked the Mou/Sakidja author/repository trail without finding a public checkpoint, round-14 found no target checkpoint across GitHub, Hugging Face, Zenodo/DataCite, OSF, AISSquare, Materials Cloud, or the public LAM-DPA FCC+L12 DPA3 tree, round-15 found new refractory/Ti-Al model files but still no dedicated Ni-Al-Ti, Ni-Al-Ta, Ni-Al-Ti-Ta, or Ni-Co-Al gamma/gamma-prime checkpoint, round-16 again found no public Ni-based gamma/gamma-prime checkpoint while adding more MoNbTaVW LAM-DPA model files, round-17 rechecked public repositories/model hubs/data indexes/DOI metadata with no new target checkpoint, round-18 found a Ni-Cr-Fe-Al-P MTP supplement but still no runnable Ni-Al-Ti/Ta gamma/gamma-prime model, round-19 completed an existing DaRUS Ni3Al MTP candidate, round-20 downloaded NequIP/Allegro foundation baselines, round-21 rechecked primary Ni-Al-Ti/Ni-Co-Al literature leads and the NEP model table, round-22 rechecked Zenodo, Hugging Face, GitHub repositories, DataCite, and AISSquare, round-23 added NIST Ni-Ti-Al and Ni-Co-Al MEAM fallback files, round-24 added a Ni-Al-Ta EAM publication-only lead, round-25 rechecked GitHub/Hugging Face/Zenodo/DataCite plus the Ni-Co-Al primary page, round-26 deep-probed primary-source/supplement routes for Ni-Al-Ti and Ni-Co-Al while rechecking AISSquare Alloy53 download access, round-27 added an HEA-related Mn-Co-Fe-Ni-Cu NEP while rechecking exact target indexes, round-29 rechecked LAM-DPA FCC+L12 plus local GPUMD/DeepMD engine readiness, and round-31 rechecked Hugging Face, DataCite, and Zenodo keyword indexes while testing DeePMD runtime; no public dedicated multicomponent gamma/gamma-prime checkpoint was found.
- Specialized-search evidence is frozen in `audits/mlip_specialized_search_2026-07-04.json`, `audits/mlip_specialized_search_lam_dpa_2026-07-04.json`, `audits/mlip_openkim_target_species_query_2026-07-04.json`, `audits/mlip_aissquare_alloy53_dpa_2026_audit.json`, `audits/mlip_tivta_w_milady_wrobel_2025_audit.json`, `audits/mlip_specialized_search_round10_2026-07-04.json`, `audits/mlip_specialized_search_round11_2026-07-04.json`, `audits/mlip_gamma_prime_public_index_queries_2026-07-04.json`, `audits/mlip_gamma_prime_public_search_round3_2026-07-04.json`, `audits/mlip_gamma_prime_public_repository_search_round4_2026-07-04.json`, `audits/mlip_gamma_prime_author_and_index_search_round5_2026-07-04.json`, `audits/mlip_gamma_prime_code_host_and_profile_search_round6_2026-07-04.json`, `audits/mlip_gamma_prime_preprint_and_supplement_search_round7_2026-07-04.json`, `audits/mlip_gamma_prime_code_and_data_api_search_round8_2026-07-04.json`, `audits/mlip_gamma_prime_programmaster_followup_round9_2026-07-04.json`, `audits/mlip_gamma_prime_author_repository_followup_round12_2026-07-04.json`, `audits/mlip_ni3al_mtp_darus_spin_fluctuations_2023_round13_audit.json`, `audits/mlip_gamma_prime_public_checkpoint_search_round14_2026-07-04.json`, `audits/mlip_specialized_search_round15_2026-07-04.json`, `audits/mlip_specialized_search_round16_2026-07-04.json`, `audits/mlip_gamma_prime_public_checkpoint_search_round17_2026-07-04.json`, `audits/mlip_gamma_prime_author_supplement_followup_round18_2026-07-04.json`, `audits/mlip_gamma_prime_checkpoint_and_ni3al_completion_round19_2026-07-04.json`, `audits/mlip_gamma_prime_and_nequip_allegro_followup_round20_2026-07-04.json`, `audits/mlip_gamma_prime_public_checkpoint_search_round21_2026-07-04.json`, `audits/mlip_gamma_prime_public_checkpoint_search_round22_2026-07-04.json`, `audits/mlip_gamma_prime_nist_meam_fallback_round23_2026-07-04.json`, `audits/mlip_gamma_prime_public_checkpoint_search_round24_2026-07-04.json`, `audits/mlip_gamma_prime_public_checkpoint_search_round25_2026-07-04.json`, `audits/mlip_gamma_prime_primary_source_deep_probe_round26_2026-07-04.json`, `audits/mlip_hea_nep_unimi_and_target_gap_round27_2026-07-04.json`, `audits/mlip_lightpfp_hea_repository_probe_round28_2026-07-04.json`, `audits/mlip_lam_dpa_gamma_prime_checkpoint_and_engine_probe_round29_2026-07-04.json`, `audits/mlip_deepmd_lammps_adapter_readiness_round30_2026-07-04.json`, `audits/mlip_gamma_prime_search_and_deepmd_runtime_round31_2026-07-04.json`, `audits/mlip_snap_lammps_supporting_candidate_round32_2026-07-04.json`, `audits/mlip_nep_mdapy_runtime_round33_2026-07-04.json`, `audits/mlip_gamma_prime_public_checkpoint_search_round34_2026-07-04.json`, `audits/mlip_mtp_engine_and_file_integrity_round35_2026-07-04.json`, `audits/mlip_crconi_nep_wu_2026_round36_2026-07-04.json`, `audits/mlip_gamma_prime_public_checkpoint_search_round37_2026-07-05.json`, `audits/mlip_rhea9_nep_byggmastar_2026_round38_2026-07-05.json`, `audits/mlip_nialti_meam_kim_2017_classical_fallback_audit.json`, and `audits/mlip_nialcoti_meam_huang_2024_classical_fallback_audit.json` so later work can distinguish found model files from no-checkpoint/commercial/classical/baseline-only leads.

Accepted potentials can be verified without changing formal benchmark
manifests:

```powershell
python -m alloy_xrd_bench.potentials verify `
  --root . `
  --potential-id pot_chgnet_0p3p0 `
  --reference-initial-id init_CrCoNi_fcc_Co1Cr1Ni1_seed000007 `
  --smoke-initial-id init_CrCoNi_fcc_Co1Cr1Ni1_seed000007 `
  --smoke-initial-id init_MoNbTaVW_bcc_Mo1Nb1Ta1V1W1_seed000017 `
  --smoke-initial-id init_Ni-Al_l12_Al1Ni3_seed000023 `
  --json
```

Verification writes:

```text
data/potentials/{potential_id}/audits/potential_verification.json
audits/potential_verification_summary.parquet
```

MLIP smoke checks write only audit files, not `generated/manifests` rows:

```powershell
python -m alloy_xrd_bench.relax smoke-mlip `
  --root . `
  --potential-id pot_chgnet_0p3p0 `
  --initial-id init_CrCoNi_fcc_Co1Cr1Ni1_seed000007 `
  --json
```

Per-structure smoke outputs live under:

```text
data/potentials/{potential_id}/audits/smoke_relaxations/
```

## 2. Store The Potential

Use one folder per potential:

```text
data/potentials/{potential_id}/potential/
  metadata.yml
  checksums.sha256
  files/
```

If the license allows redistribution, put the potential files in `files/` and
record their hashes in `checksums.sha256`.

If redistribution is not allowed, do not put the model file in the repository or
release artifact. Keep only metadata, source URL, DOI, checksum, and install
instructions.

## 3. Generate Shared Initial Structures

Shared initial structures go under:

```text
data/shared/structures_initial/
data/shared/manifests/initial_structures.parquet
```

These are inputs reused by all potentials. Do not copy them into every
potential folder.

Generate the v0.1 starter structures:

```powershell
python -m alloy_xrd_bench.structures generate-initial `
  --root . `
  --recipes configs/alloy_recipes_v0_1.yml
```

The v0.1 recipe catalog includes:

- `CrCoNi` fcc
- `CoCrFeMnNi` fcc
- `TiTaVW` bcc
- `MoNbTaVW` bcc
- `WTaCrV` bcc
- `Ni-Al`, `Ni-Al-Ti`, `Ni-Al-Ta`, and `Ni-Al-Ti-Ta` gamma/gamma-prime pairs

The current v0.1 generator uses `pair_balance_occupancy` for fcc/bcc solid
solutions and gamma matrix structures. It samples multiple deterministic
occupancy candidates per recipe, scores short-range pair balance against the
requested composition, and records `generation_score` plus `candidate_count` in
`initial_structures.parquet`.

Gamma-prime L12 structures use `l12_sublattice_occupancy`: corner sites are
occupied by Al/Ti/Ta according to the recipe, while face-center sites are mostly
Ni. Do not use ordinary random occupancy as a gamma-prime surrogate.

Gamma/gamma-prime initial pairs are written to:

```text
data/shared/manifests/gamma_prime_pairs.parquet
```

These generated files are initial structures only. They are not relaxed
structures and are not XRD labels.

Hash fields are intentionally split:

- `structure_sha256`: canonical hash of sorted pymatgen JSON content.
- `artifact_file_sha256`: byte hash of the actual JSON artifact.

## 4. Run Relaxation For One Potential

For each `potential_id`, write generated artifacts under:

```text
data/potentials/{potential_id}/generated/
  relaxation_logs/
  structures_relaxed/
  manifests/
```

The per-potential `relaxations.parquet` rows must reference:

- the shared `initial_structure_id`
- the `potential_id`
- the `relax_config_id`
- the relaxation log path under the same potential folder

The per-potential `relaxed_structures.parquet` rows must reference the
relaxation run and store JSON/CIF artifacts under the same potential folder.

For relaxed structures, `structure_sha256` remains the canonical structure hash.
Use `artifact_json_file_sha256` and `artifact_cif_file_sha256` for byte-level
artifact hashes.

Engineering smoke runner:

```powershell
python -m alloy_xrd_bench.relax smoke `
  --root . `
  --potential-id pot_example `
  --quality-tier smoke_only
```

The smoke runner copies initial structures into the relaxed-structure layout and
writes valid manifests/provenance. It exists only to verify the file layout,
manifest contracts, and downstream XRD simulation. It is not an MLIP relaxation
and should not be used for official benchmark labels.

For local end-to-end smoke tests only, `--quality-tier pass` can be used before
running the XRD simulator. Official releases should use real MLIP relaxation
outputs and record the actual engine, potential, logs, convergence status, and
quality checks.

CHGNet baseline runner:

```powershell
python -m alloy_xrd_bench.potentials register `
  --root . `
  --potential-id pot_chgnet_0p3p0 `
  --family CHGNet `
  --engine chgnet `
  --elements Cr Co Ni Fe Mn Ti Ta V W Mo Nb Al `
  --phases fcc bcc l12 `
  --source-url https://github.com/CederGroupHub/chgnet `
  --doi 10.1038/s42256-023-00716-3 `
  --license BSD-3-Clause `
  --redistribution-policy metadata_only `
  --citation-key chgnet `
  --file-sha256 d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1
```

Then run:

```powershell
python -m alloy_xrd_bench.relax chgnet `
  --root . `
  --potential-id pot_chgnet_0p3p0 `
  --model-name 0.3.0 `
  --use-device cpu `
  --steps 1000 `
  --fmax 0.05
```

The CHGNet package loads the pretrained model through its own API. The catalog
entry remains `metadata_only` unless the model file license and distribution
path are explicitly audited for packaging.

ASE-backed MLIP baseline runner:

```powershell
python -m alloy_xrd_bench.relax ase-mlip `
  --root . `
  --potential-id pot_mace_mp_0b3_medium `
  --steps 1000 `
  --fmax 0.05 `
  --no-relax-cell
```

Use the same command for `pot_matgl_m3gnet_matpes_pbe_2025_2`. The current
runner performs fixed-cell position relaxation through the potential's ASE
calculator. It writes the same per-potential `relaxations.parquet`,
`relaxed_structures.parquet`, and `provenance_edges.parquet` layout as CHGNet.

After relaxation, each potential with gamma/gamma-prime structures also writes:

```text
data/potentials/{potential_id}/generated/manifests/gamma_prime_pairs.parquet
```

This resolves shared initial gamma/gamma-prime pairs to that potential's relaxed
structure IDs.

## 5. Simulate XRD For One Potential

For each passing relaxed structure, write:

```text
data/potentials/{potential_id}/generated/xrd/grid/
data/potentials/{potential_id}/generated/xrd/peaks/
```

The `xrd_samples.parquet` rows must point to these paths and to the
`relaxed_structure_id`.

Run:

```powershell
python -m alloy_xrd_bench.xrd simulate `
  --root . `
  --potential-id pot_example `
  --split train
```

The simulator reads:

```text
data/potentials/{potential_id}/generated/manifests/relaxed_structures.parquet
```

It writes:

```text
data/potentials/{potential_id}/generated/xrd/grid/{xrd_sample_id}.parquet
data/potentials/{potential_id}/generated/xrd/peaks/{xrd_sample_id}.parquet
data/potentials/{potential_id}/generated/manifests/xrd_samples.parquet
```

It also appends:

```text
relaxed_structure -> xrd_sample
xrd_config -> xrd_sample
```

to the per-potential `provenance_edges.parquet`.

The simulator uses the Phase 1 default XRD parameters: Cu Kalpha, 20-100
degrees two-theta, 0.02 degree step, Gaussian FWHM 0.1 degree, and max
intensity normalization to 100.

The XRD line only consumes `relaxed_structure_id` rows. Do not use this command
on initial structures for official benchmark labels.

## 6. Generate Averaged XRD

Generate per-composition averaged XRD after individual XRD samples and splits
exist:

```powershell
python -m alloy_xrd_bench.xrd average `
  --root . `
  --potential-id pot_chgnet_0p3p0
```

This writes:

```text
data/potentials/{potential_id}/generated/xrd/averages/grid/
data/potentials/{potential_id}/generated/xrd/averages/peaks/
data/potentials/{potential_id}/generated/manifests/xrd_averages.parquet
```

Average rows are grouped by `potential_id`, `alloy_system`, `phase`,
`composition`, and `split`. They record `source_xrd_sample_ids` instead of a
single `relaxed_structure_id`, so averaged spectra remain traceable without
pretending to be a direct simulation from one structure.

## 7. Write Provenance

For each XRD sample, the per-potential `provenance_edges.parquet` must include:

```text
initial_structure -> relaxation_run
potential_file -> relaxation_run
relax_config -> relaxation_run
relaxation_run -> relaxed_structure
relaxed_structure -> xrd_sample
xrd_config -> xrd_sample
```

The public statement remains: XRD labels are directly derived from
`relaxed_structure_id`. MLIP and seed data are provenance.

Average XRD rows additionally add:

```text
xrd_sample -> xrd_average
```

with `relation_type=averaged_into`.

## 8. Assign Splits

Apply split labels after XRD samples exist:

```powershell
python -m alloy_xrd_bench.splits apply `
  --root . `
  --config configs/splits_v0_1.yml
```

The v0.1 split config covers `id`, `composition_ood`, `chemistry_ood`,
`phase_ood`, and `private_test`. It updates per-potential
`xrd_samples.parquet` files in place.

If splits are changed after averages are generated, rerun the average command so
`xrd_averages.parquet` uses the current split labels.

## 9. Validate

Run:

```powershell
python -m alloy_xrd_bench.validate . --json
```

For the generated example workspace:

```powershell
python scripts/make_minimal_workspace.py --root data/examples/minimal_workspace
python -m alloy_xrd_bench.validate data/examples/minimal_workspace --json
```

For a compact local inventory of validation status, registered potential
directories, manifest counts, installed package versions, and detected CHGNet
model files:

```powershell
python -m alloy_xrd_bench.status . `
  --json `
  --output audits/workspace_status.json
```

The validator checks required fields, ID uniqueness, provenance reachability,
per-potential artifact scoping, structure hashes, artifact existence,
relaxation thresholds, XRD geometry, split isolation, potential storage layout,
and unregistered potential directories.

## 10. Generate Release Audits

After validation, generate audit CSVs:

```powershell
python -m alloy_xrd_bench.audit . `
  --output-dir audits `
  --release-version v0.1
```

This writes:

```text
audits/license_audit.csv
audits/leakage_audit.csv
```

`license_audit.csv` records whether each potential can be packaged or must
remain metadata-only. `leakage_audit.csv` records the private-test relaxed
structure isolation check.

Run physics sanity audits:

```powershell
python -m alloy_xrd_bench.quality . `
  --output-dir audits
```

This writes:

```text
audits/physics_sanity.csv
audits/quality_summary.json
```

The audit checks structure nearest-neighbor distances, element conservation
across relaxation, relaxation thresholds, XRD grid non-emptiness, peak ranges,
and intensity normalization.

## 11. Export Release View

Export a public/private-controlled release view:

```powershell
python -m alloy_xrd_bench.release . `
  --output-dir data/releases/v0.1-working `
  --release-version v0.1-working
```

Public splits include relaxed structures, individual XRD, averaged XRD, and
provenance. `private_test` entries are exported only through
`manifests/private_inputs.parquet`, which contains recipe/composition/phase
metadata and does not include private relaxed structures or XRD labels.

## 12. Package Release

Refresh release-specific Croissant metadata before packaging:

```powershell
python -m alloy_xrd_bench.metadata croissant data/releases/v0.1-working
```

Generate file-level checksums, a file manifest, and a zip archive:

```powershell
python -m alloy_xrd_bench.package data/releases/v0.1-working `
  --archive data/releases/v0.1-working.zip
```

Run the publication handoff readiness gate after packaging:

```powershell
python -m alloy_xrd_bench.release_readiness data/releases/v0.1-working `
  --archive data/releases/v0.1-working.zip `
  --json `
  --output audits/release_readiness_v0.1-working.json
```

After external upload, copy `templates/publication_record.yml` into the release
directory, fill the archive name, DOI, dataset URL, release date, code
repository URL, and `release_archive_sha256`, then run:

```powershell
python -m alloy_xrd_bench.release_readiness data/releases/v0.1-working `
  --archive data/releases/v0.1-working.zip `
  --publication-record data/releases/v0.1-working/publication_record.yml `
  --require-publication `
  --json
```

This writes:

```text
data/releases/v0.1-working/checksums.sha256
data/releases/v0.1-working/file_manifest.parquet
data/releases/v0.1-working.zip
data/releases/v0.1-working/publication_record.yml
audits/release_readiness_v0.1-working.json
```

Use these artifacts as the local handoff package before uploading to Zenodo,
Hugging Face Dataset, Materials Cloud, or NOMAD.
