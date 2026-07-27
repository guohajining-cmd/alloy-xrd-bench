# Remaining Work

This file tracks work that is outside the verified local v0.1 working release.

## External Publication

- Upload `data/releases/v0.1-working.zip` to the chosen public archive.
- Fill in final DOI, repository URL, and release date after Zenodo, Hugging Face
  Dataset, Materials Cloud, or NOMAD publication in `publication_record.yml`.
- Re-run packaging and `alloy_xrd_bench.release_readiness` if archive-specific
  files are added before upload.
- Run `alloy_xrd_bench.release_readiness --require-publication` after filling
  `publication_record.yml`.

## Additional Potentials

- Current verified accepted MLIP set contains three general baselines
  (`pot_chgnet_0p3p0`, `pot_mace_mp_0b3_medium`, and
  `pot_matgl_m3gnet_matpes_pbe_2025_2`) plus four target/refractory/gamma-prime
  models (`pot_crconi_mtp_csro`, `pot_crconi_nep_wu_2026`,
  `pot_rhea9_nep_byggmastar_2026`, and
  `pot_ni3al_mtp_darus_yield_anomaly_2024`).
- `data/registry/mlip_candidates.yml` now records the three-layer admission
  queue: CHGNet/MACE/MatGL accepted general baselines; target-specific
  CrCoNi, CoCrFeMnNi, TiTaVW, MoNbTaVW, WTaCrV, and Ni-Al candidates;
  refractory-alloy-specific RHEA9 NEP/tabGAP candidates; and MTP/GAP/SNAP/
  NequIP-Allegro plus gamma/gamma-prime multicomponent research-needed
  candidates.
- The accepted potentials each have local verification audits under
  `data/potentials/{potential_id}/audits/potential_verification.json` and
  formal potential storage under `data/potentials/{potential_id}/potential/`.
  The three general baselines, the accepted CrCoNi MTP, the accepted Ni3Al MTP,
  and both accepted NEP potentials now have formal relaxed-structure manifests under
  `data/potentials/{potential_id}/generated/manifests/`. The CrCoNi NEP
  generated 5 CrCoNi relaxed structures; the RHEA9 NEP generated 15 refractory
  TiTaVW/MoNbTaVW/WTaCrV relaxed structures; the CrCoNi cSRO MTP generated 5
  formal CrCoNi relaxed structures; the DaRUS Ni3Al yield-anomaly MTP generated
  2 formal Ni-Al L12 gamma-prime relaxed structures.
- Round-37 rechecked the dedicated gamma/gamma-prime public checkpoint leads on
  2026-07-05. GitHub repository search, Hugging Face model search, Zenodo,
  Crossref, the Ni-Co-Al ScienceDirect article, the published NEP model table,
  arXiv 2603.21207, and the `xianglil/JCTC-SMCX-Supplementary` repository still
  expose no downloadable Ni-Al-Ti, Ni-Al-Ta, Ni-Al-Ti-Ta, Ni-Co-Al, or broader
  Ni-based gamma/gamma-prime checkpoint with license, checksum, and runnable
  engine recipe. The audit is recorded in
  `audits/mlip_gamma_prime_public_checkpoint_search_round37_2026-07-05.json`.
- Round-39 added an HEA-related PointFF/TorchScript lead from the 2024
  CrFeCoNiPd HEA MLIP article and the public `lianpingWu/HEAML.github.io`
  repository. The `jit_epoch.pth` model, custom LAMMPS `PointFF` source files,
  input files, and CMake file are hash-recorded under `local_potential_cache/`;
  `torch.jit.load` succeeds. It remains blocked because the repository has no
  explicit LICENSE/README, the atom-type mapping needs confirmation, it is not
  an exact current benchmark target, and no PointFF-enabled LAMMPS/LibTorch
  runtime has passed smoke.
- Round-40 rechecked exact Ni-Al-Ti, Ni-Al-Ta, Ni-Al-Ti-Ta, and Ni-Co-Al
  gamma/gamma-prime checkpoint routes. ScienceDirect/Crossref for the Ni-Co-Al
  NEP paper still exposes request-only data availability, common Elsevier
  supplement paths return 404, Hugging Face and GitHub repository searches
  expose no target checkpoint, and the public `brucefan1983/nep-data` tree has
  no target path. A TiAlNb Zenodo record was observed but kept out of registry
  because it lacks Ni and does not close the benchmark gamma/gamma-prime gaps.
- Round-41 resolved the current MTP/HDNNP runtime question. Several specialized
  model files are already downloaded, especially Ni-Al/Ni3Al `.mtp` files and
  the Al-Ni HDNNP/n2p2 package, but this Windows workspace still has no working
  `mlp`, `lmp`, or `lammps` executable on PATH. The installed Python `lammps`
  package cannot load `liblammps.dll`; PyPI `mlip` is a different JAX package;
  MAML contains MTP/LAMMPS wrappers but still shells out to external
  `mlp`/LAMMPS; and `n2p2`/`pynnp` are not available from PyPI here. The audit
  is recorded in
  `audits/mlip_mtp_hdnnp_runtime_resolution_round41_2026-07-05.json`.
- Round-42 repeated public checkpoint searches for the missing Ni-Al-Ti,
  Ni-Al-Ta, Ni-Al-Ti-Ta, and Ni-Co-Al gamma/gamma-prime targets. Public web,
  GitHub repository, Hugging Face model, Zenodo, and DataCite searches found no
  admittable checkpoint/model file with license and engine recipe. GitHub code
  search was attempted for `nep.txt`, `.mtp`, `.pth`, and `.model` patterns but
  returned HTTP 401 without authentication in this environment. The audit is
  recorded in
  `audits/mlip_gamma_prime_public_checkpoint_search_round42_2026-07-05.json`.
- Round-43 followed the author/repository trail for the strongest remaining
  gamma/gamma-prime leads. Marium Mou's public GitHub profile was found and
  checked; it contains a MIT-licensed high-entropy carbide MACE-OMAT workflow
  and a forked MAML framework, but no Ni-based gamma/gamma-prime model file.
  The LAM-DPA FCC+L12 path still contains only `input_dpa3ft.json`; LightPFP
  still exposes datasets/configs/results but no standalone checkpoint; and the
  SMC-X supplementary repository still exposes no direct model file. The audit
  is recorded in
  `audits/mlip_gamma_prime_author_repo_probe_round43_2026-07-05.json`.
- Round-44 rechecked the two strongest remaining target-specific leads:
  ProgramMaster/TMS 2026 confirms a Ni-Al-Ti gamma/gamma-prime ML-IAP abstract
  using MACE and FitSNAP, and Materials & Design/GPUMD confirms the Ni-Co-Al
  NEP publication. Neither source exposes a downloadable checkpoint, license,
  checksum, or runnable engine recipe. Fresh GitHub repository, Hugging Face,
  Zenodo, DataCite, GitLab `brucefan1983/nep-data`, and GPUMD tree checks
  found no public Ni-Al-Ti MACE/FitSNAP or Ni-Co-Al `nep.txt` model. The audit
  is recorded in
  `audits/mlip_nialti_nicoal_strong_lead_followup_round44_2026-07-05.json`.
- Round-45 promoted the two accepted target/refractory NEP models from
  verification/smoke-only status to formal relaxed-structure data. The
  ASE-backed MLIP runner now skips unsupported initial structures before
  writing manifests. `pot_crconi_nep_wu_2026` produced 5 converged/pass CrCoNi
  relaxed JSON/CIF pairs, and `pot_rhea9_nep_byggmastar_2026` produced 15
  converged/pass TiTaVW, MoNbTaVW, and WTaCrV relaxed JSON/CIF pairs. No formal
  XRD samples were generated in this phase. The audit is recorded in
  `audits/mlip_accepted_nep_formal_relaxation_round45_2026-07-05.json`.
- Round-35 audited the downloaded MTP files before admission:
  17 `.mtp`/`.mtp_` files were header-parsed, SHA256-recorded, and classified;
  16 showed trained coefficients and one Ni3Al `24.mtp` file appears to be a
  header/template-like file. MLIP-3 and LAMMPS/MLIP interface sources are cached
  under `local_potential_cache/source_documents/`.
- Round-48 built an audit-only WSL serial MLIP-3 `mlp` executable in
  `local_potential_cache/source_documents/mlip-3-wsl-build-round48/bin/mlp`.
  The build copy normalizes CRLF line endings and uses a minimal local CBLAS
  shim because WSL has no `gfortran` and sudo requires a password. This `mlp`
  passed upstream `calculate_efs` and completed reference EFS inference for
  `pot_crconi_mtp_csro` and `pot_cocrfemnni_mtp_cantor_alloy_mlip`; at that
  point no MTP candidate was accepted because element-to-type mappings were
  still assumed, not primary-source-confirmed, and smoke relaxation/formal
  adapter integration was not done.
- Round-49 resolved the nearest MTP admission split. The CrCoNi cSRO MTP source
  README explicitly gives the type map `{0: Cr, 1: Co, 2: Ni}` and the local
  audit-only `mlp relax` smoke run on
  `init_CrCoNi_fcc_Co1Cr1Ni1_seed000007` succeeded with relaxed max force
  0.04208459415273005 eV/A and zero volume change because cell relaxation was
  disabled. `pot_crconi_mtp_csro` is now blocked only on a reproducible
  `mtp_mlp` adapter/config path and formal catalog admission. The
  CoCrFeMnNi/Cantor MTP remains blocked because the source README, validation
  LAMMPS scripts, and reference POSCAR expose conflicting element orders. The
  audit is recorded in
  `audits/mlip_mtp_mapping_and_smoke_round49_2026-07-05.json`.
- Round-50 promoted `pot_crconi_mtp_csro` to an accepted potential using the new
  reproducible `mtp_mlp` runner. The MIT-licensed TS-3 Levmax 20 `.mtp` file is
  packaged under `data/potentials/pot_crconi_mtp_csro/potential/files/`, the
  catalog records the source README type map `{0: Cr, 1: Co, 2: Ni}`, and
  `potentials verify` passes load/reference/smoke with unsupported refractory
  and Ni-Al smoke structures correctly skipped. Formal `mtp-mlp` relaxation
  generated 5 converged/pass CrCoNi relaxed JSON/CIF structures and 20
  provenance edges under the potential's own `generated/` folder. The audit is
  recorded in
  `audits/mlip_crconi_mtp_csro_formal_admission_round50_2026-07-05.json`.
- Round-51 promoted `pot_ni3al_mtp_darus_yield_anomaly_2024` to an accepted
  gamma-prime-specific potential using the same `mtp_mlp` runner. The CC-BY-4.0
  DaRUS `Ni3Al.mtp` file is packaged under the potential's own `potential/files/`
  folder, the catalog records an Al/Ni type map inferred from the DaRUS
  `train.cfg` Ni3Al stoichiometry, verification passes load/reference/smoke,
  fcc smoke structures are skipped by phase scope, and formal relaxation
  generated 2 converged/pass Ni-Al L12 relaxed JSON/CIF structures. The audit is
  recorded in
  `audits/mlip_ni3al_mtp_darus_yield_anomaly_formal_admission_round51_2026-07-05.json`.
- Specialized Tier A status:
  - `pot_crconi_mtp_csro`: recommended CrCoNi `.mtp` files are downloaded under
    `local_potential_cache/`; round-35 confirms the MTP headers and trained
    coefficients. Round-48 completed audit-only `mlp calculate_efs` reference
    inference. Round-49 confirmed the source README type map and completed an
    audit-only `mlp relax` smoke check below the default fmax gate. Round-50
    added formal `mtp_mlp` runner support, admitted the model in
    `potentials_catalog.yml`, verified it, and generated 5 formal CrCoNi
    relaxed structures.
  - `pot_crconi_nep_wu_2026`: round-36 found the public MIT-licensed
    `mushroomfire/NEP-CrCoNi` repository, downloaded `model/nep.txt`,
    `model/nep.in`, README, and LICENSE, registered the model as an accepted
    target-specific CrCoNi NEP, and verified mdapy.NEP4ASE load plus CrCoNi
    reference/single-point smoke. Round-45 generated 5 formal converged/pass
    CrCoNi relaxed JSON/CIF structures and per-potential provenance manifests.
  - `pot_heaml_pointff_wu_2024_lead`: HEA-specific CrFeCoNiPd
    PointFF/TorchScript model files are downloaded and hash-recorded under
    `local_potential_cache/`. This is a useful HEA-specific lead, but not an
    exact CrCoNi/CoCrFeMnNi benchmark potential; it remains blocked on missing
    repository license, atom-type mapping confirmation, and a PointFF-enabled
    LAMMPS/LibTorch runtime.
  - `pot_crconi_ace_sheriff_motif_2025`: CrCoNi ACE/PACE `.yace` file and
    README are downloaded under `local_potential_cache/`, but LAMMPS/PACE or
    ASE/PyACE is not installed/configured.
  - `pot_cocrfemnni_mtp_cantor_alloy_mlip`: CoCrFeMnNi Cantor alloy `.mtp`
    files are downloaded under `local_potential_cache/`; round-35 confirms the
    MTP headers and trained coefficients. Round-47 confirmed the public
    `Trained_MTP_models` directory and GPL-3.0 license. Round-48 completed
    audit-only `mlp calculate_efs` reference inference with an assumed type
    map. Round-49 found conflicting primary-source element-order evidence, so
    acceptance waits on an unambiguous author/source mapping before smoke or
    formal adapter integration.
  - `pot_cocrfeni_cocrni_mtp_nitol_2026`: CoCrFeNi/CoCrNi fcc MTP files are
    downloaded under `local_potential_cache/`. This is a useful target-specific
    fcc HEA lead for CrCoNi/CoCrNi-style structures, but it is not accepted
    because the repository exposes no explicit LICENSE file and LAMMPS/user-mlip
    is not installed/configured.
  - `pot_titavw_ace_sheriff_motif_2025` and
    `pot_titavw_ace_sheriff_chemical_mixing_2025`: TiTaVW ACE/PACE `.yace`
    files are downloaded under `local_potential_cache/`, but LAMMPS/PACE or
    ASE/PyACE is not installed/configured.
  - `pot_titavw_milady_wrobel_2025`: Zenodo-hosted Ti-V-Ta-W linear and kernel
    ML potential files for LAMMPS `pair_style milady` are downloaded and
    hash-recorded under `local_potential_cache/`. This is a real TiTaVW
    target-specific MLIP candidate, but remains blocked until LAMMPS/MILADY and
    a `milady_lammps` adapter are installed/configured.
  - `pot_rhea9_nep_byggmastar_2026`: RHEA Cr-Hf-Mo-Nb-Ta-Ti-V-W-Zr
    `nep-RHEA-large.txt` and README are extracted from the Zenodo `RHEA-data.zip`
    archive with HTTP range requests and hash-validated under
    `local_potential_cache/`. Round-33 verified the model with
    `mdapy.NEP4ASE` reference inference. Round-38 formally registered it as an
    accepted redistributable-with-attribution potential, copied the model into
    `data/potentials/pot_rhea9_nep_byggmastar_2026/potential/`, and passed
    reference/smoke checks on WTaCrV, MoNbTaVW, and TiTaVW public initial
    structures. Round-45 generated 15 formal converged/pass TiTaVW, MoNbTaVW,
    and WTaCrV relaxed JSON/CIF structures and per-potential provenance
    manifests.
  - `pot_rhea9_tabgap_byggmastar_2026`: RHEA Cr-Hf-Mo-Nb-Ta-Ti-V-W-Zr EAM
    companion file and README are extracted and hash-validated under
    `local_potential_cache/`; the primary 773142992-byte HDF5 tabGAP member is
    resolved in the archive but intentionally skipped until an explicit staged
    transfer is requested and LAMMPS/tabGAP is available.
  - `pot_unep_v1_16metals_without_zbl` and `pot_unep_v1_16metals_with_zbl`:
    Zenodo-hosted UNEP-v1 NEP text models are downloaded and hash-recorded under
    `local_potential_cache/`. UNEP-v1 covers Ag, Al, Au, Cr, Cu, Mg, Mo, Ni, Pb,
    Pd, Pt, Ta, Ti, V, W, and Zr, so it covers the Ni-Al-Ti-Ta element set and
    is useful as a metal-domain baseline. It is not a gamma/gamma-prime-specific
    MLIP. Round-33 verifies both text models with `mdapy.NEP4ASE` reference
    inference; they remain blocked only as not-yet-admitted baselines.
  - `pot_alloy53_dpa1_vasp_dai_2026_aissquare`: the npj Computational Materials
    2026 Alloy53/DPA-1 article reports a 53-metal DeePMD/DPA-1 model and states
    that the potential was uploaded to AISSquare. The AISSquare public search API
    resolves the `Alloy53_VASP` entry, but anonymous detail/download requests
    currently return HTTP 401, so model filenames, license metadata, and hashes
    are not yet available. This is a metal-domain baseline lead, not a dedicated
    gamma/gamma-prime MLIP.
  - `pot_alloy53_dpa1_2_multitask_wen_2024_aissquare`: the AISSquare public
    search API resolves English and Chinese `DPA-1&2-53-alloy-multitask-400w`
    model entries. The same anonymous detail/download blocker prevents hash
    verification. This is also a metal-domain baseline lead, not a dedicated
    gamma/gamma-prime MLIP.
  - `pot_nialti_mace_fitsnap_song_2026_lead`: a 2026 TMS ProgramMaster abstract
    reports Ni-Al-Ti gamma/gamma-prime ML-IAPs trained with MACE and FitSNAP NN
    approaches. This is a high-priority dedicated lead, but no public checkpoint,
    repository, or license has been found yet after exact author/title,
    GitHub, Zenodo, Materials Cloud, Hugging Face, and AISSquare searches. A
    round-5 author/index search confirms the ProgramMaster abstract itself
    describes MACE and FitSNAP potentials trained from Quantum Espresso AIMD
    data, but the page exposes no model artifact link. A round-6 code-host and
    author-profile search found no target GitLab, Codeberg, LANL, ORCID, NOMAD,
    or Materials Cloud artifact. A round-7 preprint/supplement search found no
    target checkpoint in OSTI, NSF-PAR, ResearchSquare, or ChemRxiv/Crossref.
    A round-8 file-pattern search found no matching Hugging Face, Mendeley Data,
    Dryad, CORE, or OpenAIRE model artifact. A round-9 primary-source follow-up
    re-read the ProgramMaster page, confirmed the proceedings inclusion is only
    planned, and found no artifact/download/license link on the page.
  - `pot_nibased_gamma_prime_mou_2023_lead`: a BearWorks thesis metadata page
    reports DeepMD, NequIP/Allegro, and MTP potentials for multiphase,
    multicomponent Ni-based superalloys including Gamma and Gamma Prime. The page
    exposes no public checkpoint/archive, and the PDF download is protected by an
    AWS WAF challenge in this environment. The author's public GitHub account was
    scanned and exposes no Ni-based gamma/gamma-prime checkpoint, so this remains
    a literature lead. Round-43 found a new public high-entropy carbide
    MACE-OMAT workflow in the same author's GitHub account, but it is non-target
    and asks users to provide a pretrained MACE model file. Round-12 follow-up re-checked the thesis page, APS 2022
    and 2023 abstracts, author homepage, NC State profile, GitHub repository
    search, Hugging Face, and Zenodo. These sources reinforce that the work is a
    dedicated lead, but still expose no downloadable DeePMD, NequIP/Allegro,
    MTP, MACE, FitSNAP, or NEP checkpoint.
  - `pot_nep89_gpumd_2025`: GPUMD NEP89 89-element `nep89_20250409.txt` and
    `nep.in` are downloaded and hash-recorded under `local_potential_cache/`.
    NEP89 covers Ni, Al, Ti, Ta, and all current HEA target elements, making it
    a useful broad NEP baseline. It is not a gamma/gamma-prime-specific MLIP and
    remains blocked as a not-yet-admitted baseline, although round-33 verifies
    Cr-Co-Ni reference inference through `mdapy.NEP4ASE`.
  - `pot_nequip_oam_l_0p1` and `pot_allegro_oam_l_0p1`: NequIP/Allegro OAM-L
    foundation `.nequip.zip` artifacts are downloaded from Zenodo record
    `10.5281/zenodo.18775904` and hash-recorded under `local_potential_cache/`.
    They cover the current HEA and Ni-Al-Ti-Ta element sets as broad inorganic
    foundation baselines. They are not target-trained HEA or gamma/gamma-prime
    MLIPs and remain blocked until NequIP/Allegro load/reference inference and
    smoke relaxation are configured.
  - `pot_wtacrv_tabgap_byggmastar_2025`: Zenodo HDF5 LAMMPS tabGAP and GAP XML
    packages are downloaded and zip-validated under `local_potential_cache/`,
    but LAMMPS/tabGAP is not installed/configured.
  - `pot_monbtavw_tabgap_byggmastar`: Fairdata/Metax file metadata and
    authorized-download paths are resolved. The 842948145-byte primary
    `.tabgap`, companion `.eam.fs`, README, and pair-style source files are
    downloaded under `local_potential_cache/` and hash-validated. The remaining
    blocker is LAMMPS/tabGAP installation and smoke relaxation.
  - `pot_monbtavw_mtp_wang_2024`: GitHub-hosted MoNbTaVW `fitted.mtp`,
    `mlip.ini`, and `quinary.json` are downloaded and hash-recorded under
    `local_potential_cache/`. This is now the exact MoNbTaVW target-specific
    MTP candidate; round-35 confirms the MTP header and trained coefficients.
    It remains blocked because the repository has no explicit license and
    LAMMPS/user-mlip or MLIP-3 smoke is not configured.
  - `pot_monbtaw_mtp_ufsf_2025`, `pot_monbtaw_mtp_mvl_2021`, and
    `pot_nbmotaw_snap_mvl_2021`: MoNbTaW MTP/SNAP model files are downloaded
    and hash-recorded under `local_potential_cache/`. They are real refractory
    HEA/MPEA MLIPs, but they lack V, so they are supporting candidates rather
    than replacements for exact MoNbTaVW runs.
  - `pot_monbtavw_dpa2_lam_jiang_2026`: the npj Computational Materials 2026
    LAM/DPA repository provides Ta-Nb-W-Mo-V DPA2-5E distilled and finetune
    DeepMD checkpoints plus config/ZBL/LAMMPS files. These are downloaded and
    hash-recorded under `local_potential_cache/`. This is a real MoNbTaVW
    refractory-alloy MLIP candidate, but it remains blocked until the repository
    license is clarified and DeePMD-kit/LAMMPS `deepmd` smoke is configured.
  - `pot_monbtavw_dpa2_lam_all_jiang_2026`: the same repository also provides
    Ta-Nb-W-Mo-V DPA2-All distilled and finetune DeepMD checkpoints. The
    model/config/ZBL files are downloaded and hash-recorded under
    `local_potential_cache/`; blockers are the same missing repository license
    and missing DeePMD/LAMMPS `deepmd` smoke path.
  - `pot_monbtavw_dp_lam_jiang_2026` and
    `pot_monbtavw_dp_lam_all_jiang_2026`: the LAM/DPA repository's DP-5E and
    DP-All Ta-Nb-W-Mo-V checkpoints are now downloaded and hash-recorded as
    additional MoNbTaVW DeepMD candidates. They remain blocked on the same
    license and DeePMD/LAMMPS smoke requirements.
  - `pot_nicofecraltita_dpa3_lam_jiang_2026_lead`: the same repository exposes
    a DPA3 FCC+L12 training config with Ti-Ta-Al-Cr-Fe-Ni-Co type map, which is
    directly relevant to Ni-Al-Ti-Ta gamma/gamma-prime chemistry.
    `input_dpa3ft.json` is downloaded and hash-recorded under
    `local_potential_cache/`. Round-29 rechecked the current public GitHub main
    tree, branch list, and releases. The FCC+L12 path still contains only
    `input_dpa3ft.json`; the repository has no releases and no explicit license.
    It remains only a `research_needed` lead until a trained checkpoint/model
    artifact is published.
  - `pot_nial_mtp_wang_2025_candidate`: Ni-Al `potential.mtp` is downloaded
    and hash-validated. Round-51 audit-only MLIP-3 checks now pass for Ni-Al
    fcc/L12 single-point and smoke relaxation using the README type map
    `0=Al, 1=Ni`. It is still not accepted because the source repository has no
    explicit license or repository license metadata.
  - `pot_nial_hdnnp_sandberg_2024`: Materials Cloud hosts a compact
    `nnp-data.AlNi.tgz` HDNNP/n2p2 potential package for Al-Ni under CC-BY-4.0.
    The package and file description are downloaded and hash-recorded under
    `local_potential_cache/`. This is a real Ni-Al-specific MLIP candidate, but
    it does not cover Ti/Ta gamma-prime substitutions and remains blocked until
    n2p2/LAMMPS HDNNP smoke is configured. Round-41 confirmed there is no PyPI
    `n2p2`/`pynnp` package available in this environment and the Python LAMMPS
    package is not currently loadable.
  - `pot_ni3al_mtp_darus_yield_anomaly_2024`: DaRUS-hosted Ni3Al L12
    `Ni3Al.mtp`, `train.cfg`, and small thermodynamic/planar-defect tables are
    downloaded and hash-validated under `local_potential_cache/`. This is a true
    gamma-prime Ni3Al MTP and is now accepted through the `mtp_mlp` runner.
    Round-51 packaged `Ni3Al.mtp`, recorded the inferred `{0: Al, 1: Ni}` type
    map, verified reference/smoke inference, and generated 2 formal Ni-Al L12
    relaxed structures. It does not cover Ti/Ta substitutions or fcc gamma
    matrix structures.
  - `pot_ni3al_mtp_darus_spin_fluctuations_2023`: a second DaRUS Ni3Al L12 MTP
    source is identified. Its metadata states that the archive contains utilized
    MTPs. Round-13 follow-up partially transferred the 580894549-byte
    `MTP-training-set.tar.gz`, listed the early tar members, and extracted
    `MTP-training-set/mtp-training/24.mtp` and
    `MTP-training-set/mtp-training/Trained.mtp_`. Both files are now
    hash-recorded under `local_potential_cache/`; round-35 finds `24.mtp`
    header/template-like and `Trained.mtp_` trained-coefficient-bearing. The
    full archive transfer still hits a Stuttgart S3 403/timeout path, and MTP
    engine smoke is still missing. Round-41 confirmed that the model-file gate
    is effectively closed for `Trained.mtp_`, but the runtime gate is still open.
  - `pot_ni3al_lightpfp_research_lead_2025`: LightPFP demonstrates fine-tuned
    MTP students on Ni3Al and provides a public examples/data repository, but
    the implementation depends on Matlantis commercial PFP/LightPFP components
    and no standalone redistributable Ni3Al checkpoint was found. Round-43
    rechecked the repository tree and releases; it still contains datasets,
    configs, notebooks, and results rather than a standalone checkpoint. It is
    tracked only as a `research_needed` lead.
  - `pot_alcocrfeni_lightpfp_2025_lead`: round-28 inspected the public
    LightPFP repository as an HEA source. The repository has Al-Co-Cr-Fe-Ni HEA
    HDF5 datasets, CIF inputs, training configs, and property notebooks, plus
    Ni3Al pretrained-student datasets, but no standalone trained checkpoint or
    redistributable runtime. It is tracked only as a `research_needed`
    HEA-related lead and does not create an accepted potential directory.
  - `pot_nicoal_nep_qin_2025_lead`: a Materials & Design 2025 paper reports a
    Ni-Co-Al NEP for gamma/gamma-prime Ni-based single-crystal superalloys, but
    ScienceDirect lists data availability as on request and no public `nep.txt`
    model file was found in GitHub, Zenodo, or the published NEP model list.
    Round-37 rechecked the paper, public model indexes, and the NEP model table;
    this remains request-only and is tracked only as a `research_needed` lead.
    Round-42 exact public web, GitHub repository, Hugging Face, Zenodo, and
    DataCite searches still found no public `nep.txt` or checkpoint. Round-44
    rechecked ScienceDirect, GPUMD publications, GitHub, Hugging Face, Zenodo,
    DataCite, GitLab `nep-data`, and the GPUMD source tree; the publication is
    real and target-relevant, but the model remains request-only.
  - `pot_nicrfealp_mtp_ning_2026_lead`: a Materials & Design 2026 paper and
    Supplementary Data 1 describe a Ni-Cr-Fe-Al-P moment tensor potential for
    nickel-based polycrystals. The supplementary PDF is downloaded and
    hash-recorded under
    `local_potential_cache/pot_nicrfealp_mtp_ning_2026_lead/`; it confirms
    Ni3Al/Ni3P ordered phases and Ni54Cr20Fe20Al5P1 training data, but exposes
    no runnable `.mtp` model file. This is a useful Ni-based MTP author-contact
    lead, not an accepted potential and not a substitute for Ni-Al-Ti/Ta
    gamma/gamma-prime MLIPs.
  - `pot_nialti_meam_kim_2017_classical_fallback`: NIST-hosted Ni-Al-Ti
    `library.meam` and `NiAlTi.meam` files are downloaded and hash-recorded
    under `local_potential_cache/`. This is directly relevant to Ni-Al-Ti
    gamma-prime classical fallback checks, but it is MEAM, not an MLIP, and
    cannot satisfy the dedicated gamma/gamma-prime MLIP requirement.
  - `pot_nialcoti_meam_huang_2024_classical_fallback`: NIST-hosted Ni-Al-Co-Ti
    `library.meam` and `NiAlCoTi.meam` files are downloaded and hash-recorded
    under `local_potential_cache/`. This potential is relevant to Ni-Al-Ti
    L12/gamma-prime classical fallback checks, but it is MEAM, not an MLIP, and
    cannot satisfy the dedicated gamma/gamma-prime MLIP requirement.
  - `pot_nialti_meam_sharifi_2025_classical_fallback`: NIST-hosted Ni-Ti-Al
    `library.meam` and `NiTiAl.meam` files are downloaded and hash-recorded
    under `local_potential_cache/`. This is another Ni-Al-Ti chemistry fallback
    from the Sharifi/Wick CCA MEAM family; it is MEAM, not an MLIP.
  - `pot_nicoal_meam_sharifi_2025_classical_fallback`: NIST-hosted Ni-Co-Al
    `library.meam` and `NiCoAl.meam` files are downloaded and hash-recorded
    under `local_potential_cache/`. This is useful for classical Ni-Co-Al
    comparison only and cannot replace the missing Ni-Co-Al NEP/MLIP checkpoint.
  - `pot_altiv_mtp_nitol_2025`: NIST-hosted Al-Ti-V `AlTiV.mtp`, `mlip.ini`,
    and `potential.mod` are downloaded and hash-recorded under
    `local_potential_cache/`. This is a real related MTP, but it lacks Ni, so it
    cannot replace the missing Ni-Al-Ti gamma/gamma-prime MLIP.
  - `pot_tial_polymlp_seko_2022p3_openkim`: OpenKIM Ti-Al PolyMLP model files
    are downloaded and hash-recorded under `local_potential_cache/`. This is a
    real Ti-Al MLIP relevant to the gamma-prime Ti/Al subspace, but it lacks Ni,
    so it cannot replace the missing Ni-Al-Ti gamma/gamma-prime MLIP.
  - `pot_nialti_mlip_candidate`, `pot_nialta_mlip_candidate`, and
    `pot_nialtita_mlip_candidate`: no public MLIP model file/checkpoint was
    found. Classical MEAM/EAM fallbacks, partial Al-Ti/Al-Ti-V MLIPs, UNEP-v1
    metal-domain baselines, OpenKIM's broad TorchML MACE return for the target
    species queries, NequIP/Allegro foundation baselines, the Ni-Al-Ti
    MACE/FitSNAP conference lead, the Ni-based gamma/gamma-prime thesis lead,
    and the seven-element Fe-Co-Ni-Al-Ti-Ta-V arXiv ML-surrogate lead are noted
    only as leads/baselines. Round-4
    Figshare/OSF/Dataverse/DaRUS/DataCite checks also found no downloadable
    gamma/gamma-prime-specialized checkpoint or model file for them. Round-10
    GitHub/Hugging Face/Zenodo exact searches still found no Ni-Al-Ti,
    Ni-Al-Ta, Ni-Al-Ti-Ta, or Ni3Al DeepMD/NequIP gamma-prime checkpoint.
    Round-11 author/repository/data-archive follow-up separated new Ni-based
    polycrystal, NiCoCr structure-dump, and CrCoNi simulation-article leads from
    actual downloadable MLIP model files. Round-12 author/repository follow-up
    confirmed the Mou/Sakidja thesis and APS abstracts as real dedicated
    literature leads, but found no public model file on GitHub, Hugging Face, or
    Zenodo. Round-14 exact checkpoint checks again found no target Ni-Al-Ti,
    Ni-Al-Ta, Ni-Al-Ti-Ta, Ni-Co-Al NEP, or FCC+L12 DPA3 checkpoint across
    GitHub, Hugging Face, Zenodo/DataCite, OSF, AISSquare, Materials Cloud, or
    the public LAM-DPA repository tree. Round-15 found new refractory/Ti-Al
    model files, including an exact MoNbTaVW MTP, but still found no public
    dedicated Ni-Al-Ti, Ni-Al-Ta, Ni-Al-Ti-Ta, or Ni-Co-Al gamma/gamma-prime
    checkpoint/model file. Round-20 downloaded NequIP/Allegro foundation
    baselines and rechecked public model/data indexes, but still found no
    dedicated multicomponent gamma/gamma-prime checkpoint. Round-21 rechecked
    the primary Ni-Al-Ti ProgramMaster lead, the Ni-Co-Al NEP article, the
    published NEP model table, BearWorks, and AISSquare. Round-22 rechecked
    Zenodo, Hugging Face models/datasets, GitHub repositories, DataCite, and
    AISSquare with exact checkpoint/model queries. No public Ni-Al-Ti,
    Ni-Al-Ta, Ni-Al-Ti-Ta, or Ni-Co-Al gamma/gamma-prime checkpoint was found.
    Round-23 added NIST Ni-Ti-Al and Ni-Co-Al MEAM fallback files. Round-24
    added a Ni-Al-Ta EAM publication-only lead from Fan/Wang/Yu 2016, but no
    public model file or dedicated MLIP checkpoint. Round-25 rechecked GitHub
    repositories, Hugging Face models/datasets, Zenodo, DataCite, and the
    Ni-Co-Al primary page; it still found no public dedicated checkpoint.
    Round-26 deep-probed primary-source routes for the Ni-Al-Ti ProgramMaster
    lead and the Ni-Co-Al NEP article, including Crossref relations and common
    Elsevier supplementary-file URL patterns, and rechecked AISSquare Alloy53
    detail/object-download access. It found no new downloadable dedicated
    gamma/gamma-prime checkpoint. Round-27 found and downloaded a CC0
    Mn-Co-Fe-Ni-Cu NEP/GPUMD model set from UNIMI Dataverse, but this is
    HEA-related rather than benchmark-target-specific because it lacks Cr, Ti,
    Ta, Mo, Nb, V, and W; exact Ni-Al-Ti, Ni-Al-Ta, Ni-Al-Ti-Ta, and Ni-Co-Al
    gamma/gamma-prime checkpoint searches still returned zero public hits.
    Round-37 rechecked GitHub repositories, Hugging Face, Zenodo, Crossref,
    arXiv 2603.21207, the SMC-X supplementary repository trail, the Ni-Co-Al
    primary article, and the public NEP model table on 2026-07-05; no public
    target checkpoint, license, checksum, or installable artifact was found.
    Round-42 repeated public web, GitHub repository, Hugging Face, Zenodo, and
    DataCite searches and still found no public Ni-Al-Ti, Ni-Al-Ta,
    Ni-Al-Ti-Ta, or Ni-Co-Al gamma/gamma-prime MLIP checkpoint. Round-44
    rechecked the Ni-Al-Ti TMS 2026 MACE/FitSNAP abstract and Ni-Co-Al NEP
    paper/publication-list paths; both remain strong leads but no model file is
    public yet.
- Specialized-potential search audits are recorded in
  `audits/mlip_specialized_search_2026-07-04.json` and
  `audits/mlip_specialized_search_lam_dpa_2026-07-04.json`; round-10 evidence
  is recorded in `audits/mlip_specialized_search_round10_2026-07-04.json`. They separate
  positive model-file findings from non-admissible classical/commercial/
  no-checkpoint leads, including the hash-recorded LAM/DPA DPA3 FCC+L12 config
  that lacks a trained checkpoint.
- The NEP89 baseline audit is recorded in
  `audits/mlip_nep89_gpumd_2025_audit.json`; it explicitly marks NEP89 as a
  broad GPUMD baseline, not a dedicated gamma/gamma-prime potential.
- The Ni-Al HDNNP Materials Cloud audit is recorded in
  `audits/mlip_nial_hdnnp_sandberg_2024_audit.json`; it explicitly marks the
  model as Ni-Al-specific but not Ti/Ta gamma-prime-capable.
- The OpenKIM target-species audit is recorded in
  `audits/mlip_openkim_target_species_query_2026-07-04.json`; it records exact
  Ni-Al-Ti, Ni-Al-Ta, and Ni-Al-Ti-Ta species-query returns and classifies
  OpenKIM's TorchML MACE hit as a broad baseline rather than a dedicated
  gamma/gamma-prime MLIP.
- The AISSquare Alloy53/DPA audit is recorded in
  `audits/mlip_aissquare_alloy53_dpa_2026_audit.json`; it tracks public
  Alloy53/DPA model-list entries and the current authenticated-detail/download
  blocker.
- The Ti-V-Ta-W MILADY audit is recorded in
  `audits/mlip_tivta_w_milady_wrobel_2025_audit.json`; it records downloaded
  Zenodo model files, SHA256/MD5 hashes, and the current missing LAMMPS/MILADY
  engine blocker.
- The gamma/gamma-prime public-index audit is recorded in
  `audits/mlip_gamma_prime_public_index_queries_2026-07-04.json`; it records
  exact GitHub, Hugging Face, Zenodo, and AISSquare queries for Ni-Al-Ti,
  Ni-Al-Ta, and Ni-Al-Ti-Ta MLIP model files and separates true misses from
  useful nonmatching baseline/TiTaVW leads.
- The third gamma/gamma-prime public-search audit is recorded in
  `audits/mlip_gamma_prime_public_search_round3_2026-07-04.json`; it records
  direct GitHub repository-search zeros for Ni-Al-Ta/Ni-Al-Ti-Ta, AISSquare
  public-list filtering over 131 entries with no target match, and the Mou author
  GitHub scan.
- The fourth gamma/gamma-prime public-repository audit is recorded in
  `audits/mlip_gamma_prime_public_repository_search_round4_2026-07-04.json`; it
  records Figshare API 403 access blockers, OSF title-query zero hits,
  low-precision Harvard Dataverse misses, DaRUS confirmation of only the two
  already-known Ni3Al MTP sources, and DataCite hits that are related but not
  admittable model checkpoints.
- The fifth gamma/gamma-prime author/index audit is recorded in
  `audits/mlip_gamma_prime_author_and_index_search_round5_2026-07-04.json`; it
  records exact GitHub repository and Hugging Face model-search zeros for
  Song/Ni-Al-Ti, Ni-Al-Ta, Ni-Al-Ti-Ta, and Ni3Al checkpoint queries, plus
  Zenodo/Crossref checks that separate generic FitSNAP software and classical
  Ni-Al-Ti 2NN-MEAM from dedicated MLIP checkpoints.
- The sixth gamma/gamma-prime code-host/profile audit is recorded in
  `audits/mlip_gamma_prime_code_host_and_profile_search_round6_2026-07-04.json`;
  it records zero GitLab public-project and Codeberg repository hits, Bitbucket
  and Materials Cloud endpoint blockers, NOMAD perovskite-archive noise, and no
  target-like Huajing Song ORCID/LANL model artifacts.
- The seventh gamma/gamma-prime preprint/supplement audit is recorded in
  `audits/mlip_gamma_prime_preprint_and_supplement_search_round7_2026-07-04.json`;
  it records OSTI/DOE-style record misses, NSF-PAR pages with no target terms,
  ResearchSquare search endpoint blockers, and ChemRxiv/Crossref nonmatching
  method/different-chemistry hits.
- The eighth gamma/gamma-prime code/data API audit is recorded in
  `audits/mlip_gamma_prime_code_and_data_api_search_round8_2026-07-04.json`; it
  records explicit file-pattern searches for `model.pt`, `potential.mtp`,
  `nep.txt`, `graph.pb`, and related terms across Hugging Face, Mendeley Data,
  Dryad, CORE, and OpenAIRE, plus grep.app/GitHub code-search access blockers.
- The ninth gamma/gamma-prime primary-source follow-up is recorded in
  `audits/mlip_gamma_prime_programmaster_followup_round9_2026-07-04.json`; it
  confirms the TMS ProgramMaster abstract is a strong Ni-Al-Ti dedicated MLIP
  lead, records the planned proceedings status, and verifies that the page does
  not expose a model artifact, checkpoint, license, or checksum link.
- The tenth specialized MLIP search audit is recorded in
  `audits/mlip_specialized_search_round10_2026-07-04.json`; it records the new
  public CoCrFeNi/CoCrNi MTP GitLab source, downloaded file hashes, and the
  remaining license/engine blockers, while also confirming that the new
  Ni-based polycrystal paper and TMS program evidence do not expose a
  gamma/gamma-prime checkpoint.
- The eleventh specialized MLIP search audit is recorded in
  `audits/mlip_specialized_search_round11_2026-07-04.json`; it confirms the
  Ni-Al MTP GitHub model is already staged and hash-recorded, records missing
  license evidence for the Ni-Al and CoCrFeNi/CoCrNi MTPs, and classifies the
  NiCoCr Zenodo structure dataset, CrCoNi RSC simulation article, QuantumATK
  commercial/classical entries, and Alloy53/AISSquare follow-up as non-admitted
  evidence rather than new gamma/gamma-prime model files.
- The twelfth gamma/gamma-prime author/repository follow-up is recorded in
  `audits/mlip_gamma_prime_author_repository_followup_round12_2026-07-04.json`;
  it records BearWorks/APS/author-page confirmation of the Mou/Sakidja
  dedicated Ni-based gamma/gamma-prime MLIP lead, repeats the blocked PDF probe,
  scans the author's current GitHub repositories, and records zero target
  Hugging Face or Zenodo checkpoint hits.
- The thirteenth Ni3Al DaRUS extraction audit is recorded in
  `audits/mlip_ni3al_mtp_darus_spin_fluctuations_2023_round13_audit.json`; it
  records the DaRUS metadata, Stuttgart S3 403/partial-transfer behavior, tar
  member listing, and SHA256 hashes for the extracted `24.mtp` and
  `Trained.mtp_` Ni3Al MTP files.
- The fourteenth gamma/gamma-prime public checkpoint search is recorded in
  `audits/mlip_gamma_prime_public_checkpoint_search_round14_2026-07-04.json`;
  it records zero target checkpoint hits across GitHub, Hugging Face,
  Zenodo/DataCite, OSF, AISSquare, Materials Cloud, and the public LAM-DPA
  FCC+L12 DPA3 repository tree.
- The fifteenth specialized-potential search is recorded in
  `audits/mlip_specialized_search_round15_2026-07-04.json`; it records the
  newly downloaded exact MoNbTaVW MTP candidate, MoNbTaW MTP/SNAP supporting
  candidates, TiAl MTP supporting candidate, and negative checks for GPUMD
  `nep-data`, the Sheriff ACE repository, and the new Ni-based polycrystal
  literature lead.
- The sixteenth specialized-potential search is recorded in
  `audits/mlip_specialized_search_round16_2026-07-04.json`; it records newly
  downloaded LAM-DPA DP-5E/DP-All MoNbTaVW checkpoints, supplemental DPA2
  finetune checkpoints, and another negative public-checkpoint pass for
  Ni-Al-Ti, Ni-Al-Ta, Ni-Al-Ti-Ta, Ni-Co-Al, and FCC+L12 DPA3 gamma/gamma-prime
  targets.
- The seventeenth gamma/gamma-prime public checkpoint search is recorded in
  `audits/mlip_gamma_prime_public_checkpoint_search_round17_2026-07-04.json`;
  it re-checks GitHub repositories, Hugging Face models/datasets, DataCite,
  GitLab projects, AISSquare's public model list, Materials Cloud, DOI metadata,
  and GitHub code-search availability. It found no public dedicated Ni-Al-Ti,
  Ni-Al-Ta, Ni-Al-Ti-Ta, Ni-Co-Al NEP, or FCC+L12 DPA3 checkpoint; GitHub code
  search was unavailable without authentication and is not used as proof of
  absence.
- The eighteenth author/supplement follow-up is recorded in
  `audits/mlip_gamma_prime_author_supplement_followup_round18_2026-07-04.json`;
  it downloads Supplementary Data 1 for the Ni-Cr-Fe-Al-P nickel-polycrystal
  MTP paper, registers `pot_nicrfealp_mtp_ning_2026_lead`, records the
  corresponding author contact and SHA256 hashes, and confirms that no runnable
  `.mtp` model file is exposed.
- The nineteenth gamma/gamma-prime checkpoint follow-up is recorded in
  `audits/mlip_gamma_prime_checkpoint_and_ni3al_completion_round19_2026-07-04.json`;
  it re-checks ProgramMaster, ScienceDirect, BearWorks, materialsmodeling.org,
  Hugging Face, GitHub, GitLab, Zenodo, and DataCite for missing Ni-Al-Ti,
  Ni-Al-Ta, Ni-Al-Ti-Ta, and Ni-Co-Al runnable checkpoints. It finds no new
  public checkpoint, but completes the existing DaRUS Ni3Al yield-anomaly MTP
  cache by downloading `train.cfg` and four small supporting `.dat` tables with
  hash validation.
- The twentieth gamma/gamma-prime and foundation-model follow-up is recorded in
  `audits/mlip_gamma_prime_and_nequip_allegro_followup_round20_2026-07-04.json`;
  it downloads/hash-records NequIP-OAM-L and Allegro-OAM-L foundation artifacts
  from Zenodo, then rechecks GitHub, Hugging Face, Zenodo, DataCite, Materials
  Cloud, AISSquare, BearWorks, ScienceDirect, and the public NEP model registry.
  It still finds no public dedicated Ni-Al-Ti, Ni-Al-Ta, Ni-Al-Ti-Ta, or
  Ni-Co-Al gamma/gamma-prime checkpoint.
- The twenty-first gamma/gamma-prime public checkpoint search is recorded in
  `audits/mlip_gamma_prime_public_checkpoint_search_round21_2026-07-04.json`;
  it rechecks the primary TMS Ni-Al-Ti ProgramMaster abstract, the Materials &
  Design Ni-Co-Al NEP article, the published NEP model table, BearWorks thesis
  metadata, and AISSquare public model searches. It confirms strong literature
  leads but finds no public model artifact, checkpoint, license, or checksum.
- The twenty-second gamma/gamma-prime public checkpoint search is recorded in
  `audits/mlip_gamma_prime_public_checkpoint_search_round22_2026-07-04.json`;
  it rechecks Zenodo, Hugging Face models/datasets, GitHub repository search,
  DataCite, and AISSquare for exact Ni-Al-Ti, Ni-Al-Ta, Ni-Al-Ti-Ta, Ni-Co-Al,
  and Ni3Al checkpoint/model queries. GitHub, Hugging Face, DataCite, and
  AISSquare return no target hits; Zenodo returns only unrelated or already
  tracked broad baseline records.
- The twenty-third NIST fallback follow-up is recorded in
  `audits/mlip_gamma_prime_nist_meam_fallback_round23_2026-07-04.json`; it
  downloads/hash-records NIST Sharifi/Wick 2025 Ni-Ti-Al and Ni-Co-Al MEAM
  files. This is useful fallback coverage, but it explicitly remains outside
  the dedicated MLIP requirement.
- The twenty-fourth gamma/gamma-prime public checkpoint search is recorded in
  `audits/mlip_gamma_prime_public_checkpoint_search_round24_2026-07-04.json`;
  it adds `pot_nialta_eam_fan_2016_classical_lead` as a Ni-Al-Ta gamma-prime
  EAM publication-only lead, records zero OpenKIM/GitHub/Hugging Face/Zenodo
  target checkpoint hits, and confirms that the lead is classical and has no
  public potential file in the checked endpoints.
- The twenty-fifth gamma/gamma-prime public checkpoint search is recorded in
  `audits/mlip_gamma_prime_public_checkpoint_search_round25_2026-07-04.json`;
  it rechecks GitHub repositories, Hugging Face models/datasets, Zenodo,
  DataCite, and the Ni-Co-Al primary article page. It records zero target
  checkpoint/model hits and keeps all formal generated manifests unchanged.
- The twenty-sixth primary-source deep probe is recorded in
  `audits/mlip_gamma_prime_primary_source_deep_probe_round26_2026-07-04.json`;
  it rechecks the Ni-Co-Al primary ScienceDirect/Crossref/Elsevier supplement
  routes, the TMS ProgramMaster Ni-Al-Ti MACE/FitSNAP lead, and AISSquare
  Alloy53 access. It found no public dedicated gamma/gamma-prime checkpoint,
  no new downloaded model files, and no formal manifest changes.
- The twenty-seventh HEA NEP follow-up is recorded in
  `audits/mlip_hea_nep_unimi_and_target_gap_round27_2026-07-04.json`; it adds
  `pot_mncofenicu_nep_unimi_2026`, downloads three UNIMI Dataverse NEP archives
  plus README with matching MD5/SHA256 hashes, extracts three `nep.txt` files,
  and confirms the model's Mn/Fe/Co/Ni/Cu element coverage from `nep.in`.
  This improves NEP-framework coverage but does not close the exact CrCoNi,
  CoCrFeMnNi, refractory bcc, or gamma/gamma-prime target gaps.
- The twenty-ninth LAM-DPA and engine follow-up is recorded in
  `audits/mlip_lam_dpa_gamma_prime_checkpoint_and_engine_probe_round29_2026-07-04.json`;
  it rechecks the public LAM-DPA FCC+L12 DPA3 tree, confirms the Ti-Ta-Al-Cr-Fe-
  Ni-Co training config is still config-only with no trained checkpoint, and
  records local GPUMD/DeepMD blockers (`gpumd`, `nep`, `lmp`, `lammps`, `dp`,
  `cl.exe`, `make`, and `nvcc` are not on `PATH`).
- The thirtieth DeepMD/LAMMPS readiness follow-up is recorded in
  `audits/mlip_deepmd_lammps_adapter_readiness_round30_2026-07-04.json`; it adds
  the `deepmd_lammps` adapter path so DPA/DP candidates report structured
  `missing_engine`/`missing_model` blockers instead of unsupported-engine
  failures, confirms `deepmd` and `lammps` Python modules are still absent, and
  keeps all formal generated manifests unchanged.
- The thirty-first gamma/gamma-prime and DeepMD runtime follow-up is recorded
  in `audits/mlip_gamma_prime_search_and_deepmd_runtime_round31_2026-07-04.json`;
  it rechecks Hugging Face, DataCite, and Zenodo for NiAlTi, Ni-Al-Ti,
  Ni-Al-Ta, Ni-Al-Ti-Ta, Ni3Al, and gamma-prime potential keywords, finds no
  new admittable public gamma/gamma-prime checkpoint, installs `deepmd-kit`
  3.1.3 and `lammps` 2025.7.22.4.0 in the user Python environment, and records
  that `pair_style deepmd` is still unavailable while `dp --pt show` fails with
  a PyTorch CXX11 ABI mismatch.
- The same round-31 LAMMPS style probe found `snap`, `mliap`, `meam`, and
  `uf3` reachable in the installed wheel, but `pace`, `hdnnp`, and `quip` are
  not enabled. This keeps the downloaded `.yace` ACE/PACE models and Al-Ni
  n2p2 HDNNP package blocked, while SNAP/MEAM/UF3 candidates can be probed
  next with less engine setup.
- The thirty-second SNAP follow-up is recorded in
  `audits/mlip_snap_lammps_supporting_candidate_round32_2026-07-04.json`; it
  adds a `snap_lammps` adapter and verifies that the installed LAMMPS ML-SNAP
  engine can evaluate the downloaded NbMoTaW SNAP coeff/param files after a
  temporary smoke-only compatibility transform that removes the obsolete
  `diagonalstyle` keyword. This candidate still lacks V and remains supporting
  coverage only, not an accepted MoNbTaVW potential.
- The thirty-third NEP runtime follow-up is recorded in
  `audits/mlip_nep_mdapy_runtime_round33_2026-07-04.json`; it adds a
  `mdapy.NEP4ASE` path to the NEP adapter and verifies reference inference for
  RHEA9 NEP, UNEP-v1 with/without ZBL, NEP89, and Mn-Co-Fe-Ni-Cu NEP text
  models. This reduces the NEP engine blocker for single-point checks, but it
  does not create accepted potentials or modify formal generated manifests.
- The thirty-fourth gamma-prime public-checkpoint follow-up is recorded in
  `audits/mlip_gamma_prime_public_checkpoint_search_round34_2026-07-04.json`.
  It rechecks Hugging Face, GitHub repositories, DataCite, Zenodo, arXiv, the
  Ni-Co-Al ScienceDirect paper, NIST Al-Ni-Ti, BearWorks thesis metadata, and
  the LAM-DPA FCC+L12 GitHub path. No new public Ni-Al-Ti, Ni-Al-Ta,
  Ni-Al-Ti-Ta, Ni-Co-Al, or Ni-based gamma/gamma-prime MLIP checkpoint was
  found; unresolved leads remain `research_needed`.
- The thirty-seventh gamma-prime public-checkpoint follow-up is recorded in
  `audits/mlip_gamma_prime_public_checkpoint_search_round37_2026-07-05.json`.
  It rechecks GitHub repositories, Hugging Face, Zenodo, Crossref, the Ni-Co-Al
  ScienceDirect article, the public NEP model table, arXiv 2603.21207, and the
  SMC-X supplementary repository trail. No new public target checkpoint,
  license, checksum, or installable artifact was found; formal generated
  manifests remain unchanged.
- The thirty-eighth refractory NEP admission audit is recorded in
  `audits/mlip_rhea9_nep_byggmastar_2026_round38_2026-07-05.json`. It promotes
  `pot_rhea9_nep_byggmastar_2026` from blocked candidate to accepted
  refractory-alloy-specific NEP after Zenodo/CC-BY-4.0 source confirmation,
  formal model placement, hash checks, and mdapy.NEP4ASE reference/smoke checks
  on WTaCrV, MoNbTaVW, and TiTaVW initial structures.
- The thirty-ninth HEA-related PointFF lead audit is recorded in
  `audits/mlip_heaml_pointff_wu_2024_round39_2026-07-05.json`. It records the
  CrFeCoNiPd TorchScript model and custom LAMMPS pair-style source files from
  `lianpingWu/HEAML.github.io`, verifies `torch.jit.load`, and keeps the
  candidate blocked because license, atom-type mapping, and PointFF runtime
  gates are still open.
- The fortieth gamma-prime public-checkpoint follow-up is recorded in
  `audits/mlip_gamma_prime_public_checkpoint_search_round40_2026-07-05.json`.
  It freezes the request-only Ni-Co-Al NEP evidence, exact public-index misses,
  the still-unlicensed Ni-Al MTP source state, and the non-target TiAlNb
  supporting lead classification.
- The Ni-Al-Co-Ti MEAM classical fallback audit is recorded in
  `audits/mlip_nialcoti_meam_huang_2024_classical_fallback_audit.json`; it
  records NIST file hashes and explicitly marks the fallback as not an MLIP.
- The Ni-Al-Ti MEAM classical fallback audit is recorded in
  `audits/mlip_nialti_meam_kim_2017_classical_fallback_audit.json`; it records
  NIST file hashes and explicitly marks the fallback as not an MLIP.
- Each added potential must be registered in `data/registry/potentials_catalog.yml`
  and must write all generated data under its own
  `data/potentials/{potential_id}/generated/` directory.
- Candidate verification and `smoke-mlip` checks write only under
  `data/potentials/{potential_id}/audits/`; they do not add rows to official
  relaxation, relaxed-structure, XRD, or provenance manifests.
- Round-47 rechecked public GitHub, Hugging Face, Zenodo, GPUMD `nep-data`, and
  NEP model-table indexes for Ni-Al-Ti, Ni-Al-Ta, Ni-Al-Ti-Ta, and Ni-Co-Al
  gamma/gamma-prime MLIPs. No public checkpoint/model file was found; the
  remaining gamma/gamma-prime gap is source availability, not benchmark storage
  layout.

## Engine Setup Still Needed

- Local round-29 engine probe found an NVIDIA GeForce RTX 4060 with compute
  capability 8.9, but no `gpumd`, `nep`, `lmp`, `lammps`, or `dp` executable was
  on `PATH`.
- GPUMD remains source/build-gated on this Windows machine because `cl.exe`,
  `make.exe`, and `nvcc` are not on `PATH`. This keeps GPUMD-driven NEP
  relaxation blocked, but round-33 shows that downloaded `nep.txt` files can now
  pass Python reference inference through `mdapy.NEP4ASE`.
- DeePMD/LAMMPS remains unverified: `deepmd-kit` and `lammps` packages are
  visible on PyPI, but no runtime has passed reference inference or smoke
  relaxation here. Do not promote DeepMD candidates to accepted until
  `pair_style deepmd` smoke works.
- A round-30 dry-run resolved `deepmd-kit==3.1.3` and
  `lammps==2025.7.22.4.0` as installable package candidates, but no
  installation was performed and both `import deepmd` and `import lammps`
  currently fail in this workspace.
- Round-31 installed those Python wheels successfully, so `import deepmd`,
  `import lammps`, `dp.exe --version`, and `lmp.exe -h` work from the user
  Python package/script locations. This is still not enough for candidate
  admission: `lmp.exe` reports `ERROR: Unrecognized pair style 'deepmd'`, and
  `dp --pt show` on the downloaded MoNbTaVW DPA/DP checkpoints fails because
  the installed DeepMD wheel was compiled with `CXX11_ABI_FLAG=0` while the
  current PyTorch runtime reports `CXX11_ABI_FLAG=1`.
- Round-46 resolves the executable-discovery layer: the adapter helper now finds
  user/bundled `lmp.exe` and `dp.exe`. LAMMPS exposes `snap`, `mliap`, `meam`,
  and `uf3`, but not `deepmd`, `pace`, `hdnnp`, `mlip`, `quip`, or `tabgap`.
  This narrows the remaining blockers to engine-specific builds: tabGAP LAMMPS
  custom style for `.tabgap`, DeePMD pair style or ABI-compatible
  DeepMD/PyTorch for DPA/DP, and PACE/HDNNP runtimes for `.yace`/n2p2
  candidates. For `.mtp`, Round-48 now provides an audit-only `mlp`
  executable; the remaining MTP blockers are primary-source type mapping,
  formal adapter wiring, and smoke relaxation.

## Structure Generation

- v0.1 now contains 41 shared initial structures: fcc/bcc HEA starters plus
  paired Ni-Al, Ni-Al-Ti, Ni-Al-Ta, and Ni-Al-Ti-Ta gamma/gamma-prime
  structures.
- HEA solid-solution starters use `pair_balance_occupancy` with recorded
  `generation_score` and `candidate_count`.
- Gamma-prime L12 structures use `l12_sublattice_occupancy`, with corner
  sublattice Al/Ti/Ta and face-center sublattice Ni.
- Shared gamma/gamma-prime pair metadata lives in
  `data/shared/manifests/gamma_prime_pairs.parquet`. Per-potential pair rows
  with relaxed IDs live in each accepted potential's
  `generated/manifests/gamma_prime_pairs.parquet`.
- Strict external SQS tooling, such as ATAT/mcsqs, has not been integrated.

## XRD Follow-Up

- This phase refreshed relaxed structures only. Existing CHGNet v0.1 XRD files
  are retained for continuity, but MACE/MatGL XRD and refreshed gamma/gamma-prime
  mixture spectra are intentionally deferred to the next phase.

## Version Control

- `E:\project\xrd-benchmark` now has a local Git repository on branch `main`.
- Configure `user.name` and `user.email`, make the initial commit, and add a
  remote before public code release or collaborative review.
