# KDD Paper Pagination Implementation Plan

1. Add a paper-contract test for official class options, reference/appendix
   order, and absence of forced transition page breaks.
2. Remove the two transition `clearpage` commands and replace appendix `[H]`
   figures with normal top floats.
3. Compile LaTeX and render all pages to PNG.
4. If whitespace remains severe, group related diagnostics without changing
   their data or captions.
5. Run focused and full tests, then record final hashes.
