# External Experimental Sanity Sets

These spectra are external-validity checks, not benchmark ground truth. They
must never be used to report experimental SRO, site occupancy, or exact
Gamma-Decomp component accuracy.

## HEA v1

`hea_v1` contains two experimental AlCoCrFeNi thin-film XRD patterns from
Zenodo DOI `10.5281/zenodo.4623002` (CC BY 4.0). The source Bruker RAW4 files,
converted angle-intensity arrays, SHA-256 values, and treatment metadata are
retained. Conversion used the LGPL-2.1 RAW4 reader from
`eljackobrien/Binary_file_parsers` at the URL recorded in
`hea_v1/sanity_report.json`; parser code is not redistributed here.

Allowed analyses are mother-phase trends, composition consistency, and the
simulation-to-experiment domain gap. Exact SRO and occupancy truth are absent.

## Gamma-Decomp

`gamma_v1` contains two precipitation-heat-treated Ni-based superalloy patterns
for IN718 and AM718R from Apollo DOI `10.17863/CAM.87406` (CC BY-ND 4.0).
The original `XRD Data.zip` and its two Bruker-exported angle-intensity `.xy`
files are retained byte-for-byte; no converted pattern files are distributed.
The associated article DOI is `10.1017/dce.2022.31`.

The article explicitly reports that the gamma matrix peaks are visible while
gamma-prime and gamma-double-prime superlattice reflections are not
distinguishable. These spectra therefore support only phase-level trends,
whole-pattern reconstruction checks, and simulation-to-experiment gap analysis.
They do not provide experimental component, occupancy, or sublattice truth.
