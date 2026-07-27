# Peak XGBoost: HEA v1.1

XGBoost baseline over deterministic top-peak features. Train an independent
model for each split and each seed 2027-2031. The `.joblib` bundle stores the
estimator, seed, split, and peak budget for matched perturbation inference.
Known limitations include sensitivity to peak extraction and loss of diffuse or
weak-pattern information.
