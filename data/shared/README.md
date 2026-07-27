# Shared Inputs

Shared benchmark inputs live here. These files are not produced by any specific
MLIP potential.

```text
recipes/
structures_initial/
```

Relaxation runs reference shared initial structures by `initial_structure_id`.

Generate the v0.1 starter set with:

```powershell
python -m alloy_xrd_bench.structures generate-initial --root . --recipes configs/alloy_recipes_v0_1.yml
```
