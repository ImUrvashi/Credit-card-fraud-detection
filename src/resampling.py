"""Wrappers around imbalanced-learn resamplers so train.py can look
strategies up by name."""

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

RESAMPLERS = {
    "baseline": None,
    "class_weight": None,
    "undersample": RandomUnderSampler,
    "oversample": RandomOverSampler,
    "smote": SMOTE,
    "adasyn": ADASYN,
    "smotetomek": SMOTETomek,
}


def resample(X, y, strategy, random_state=42):
    """Return (X, y) resampled per `strategy`. baseline/class_weight
    don't touch the data — class_weight instead changes how the model
    itself is built (see models.py)."""
    resampler_cls = RESAMPLERS[strategy]
    if resampler_cls is None:
        return X, y
    resampler = resampler_cls(random_state=random_state)
    return resampler.fit_resample(X, y)
