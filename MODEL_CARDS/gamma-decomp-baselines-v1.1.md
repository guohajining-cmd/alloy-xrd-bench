# Gamma-Decomp v1.1 Baseline Model Card

## Models

The frozen comparison suite includes a no-spectrum prior, peak-feature model,
pseudo-Voigt BIC fitter for windows, ResNet set predictor, Transformer set
predictor, and physics-constrained Transformer. Reconstruction-only,
weak-peak, clean-training, and random-split variants are ablations rather than
separate leaderboard entries.

## Inputs And Outputs

Models receive only the public task identifier, two-theta grid, and observed
intensity. Outputs are unordered components with phase, grouped parent HKL,
center, FWHM, integrated intensity, pseudo-Voigt mixing parameter, and
confidence. Parent structures, clean validation/test spectra, overlap labels,
and generation metadata are not model inputs.

## Evaluation

Primary metrics include matched component F1, phase/HKL accuracy, count
accuracy, and parameter errors. Forward reconstruction is supplementary and
cannot replace correct latent component recovery. Learned baselines use seeds
2027-2029; seed standard deviation and 2,000-replicate parent-pair bootstrap
intervals are reported separately.

## Limitations

The release is simulated. It tests structured recovery under controlled
overlap and perturbation, not exact decomposition of arbitrary experimental
superalloy patterns. Results must be qualified by overlap difficulty and
instrument-shift slices.
