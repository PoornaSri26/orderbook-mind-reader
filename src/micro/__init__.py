"""micro — order-book microstructure modelling for the Mind Reader project.

Submodules:
    load       bookTicker archives -> event frames
    labels     next-mid-move labels (event time and fixed horizon)
    features   per-event features + fold-local normalisation
    baselines  logit / weighted mid / Stoikov micro-price / LightGBM
    model      causal transformer (torch optional)
    train      purged, embargoed walk-forward
    evaluate   log-loss, AUC, calibration, bp translation, block bootstrap
    interp     probes with controls, top-k SAE, activation patching
    style      figure design tokens
"""

__version__ = "0.1.0"
