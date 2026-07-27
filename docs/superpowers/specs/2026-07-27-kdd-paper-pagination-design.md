# KDD 2027 Paper Pagination Design

## Official Constraint

The KDD 2027 Research Track requires `acmart` in anonymous review mode, up to
eight main-content pages, followed by references and an optional appendix. The
appendix has no submission page limit but must use the same format as the body.

Official source: https://kdd2027.kdd.org/research-track-call-for-papers/

## Layout Decision

- Preserve the required order: main paper, references, appendix.
- Remove explicit page breaks around the bibliography and appendix.
- Allow appendix figures to use normal top-of-column placement instead of `[H]`.
- Preserve all frozen counts, metrics, captions, and generated assets.
- Recompile and inspect every page; only add more invasive figure grouping if
  normal float placement still leaves incoherent whitespace.

## Acceptance

- Anonymous `sigconf` review format remains intact.
- References precede the appendix.
- No forced blank transition pages remain.
- No clipping, overlap, undefined references, or horizontal overfull boxes.
- Existing paper-contract and repository tests pass.
