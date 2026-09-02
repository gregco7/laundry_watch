"""The classifier: one normalized feature vector in, one phase out.

A HistGradientBoostingClassifier, deliberately. The features already encode
the domain knowledge -- eight bands, a centroid, a tilt, three deltas -- so
what is left is finding thresholds in fifteen dimensions, which is exactly
what a boosted tree ensemble is good at and what a neural net would need
thousands of cycles to learn worse. It trains in under a second on this data,
which matters more than it sounds: a model that retrains in a second gets
retrained after every wash, and one that takes an hour gets retrained never.

This file is thin on purpose. It knows nothing about recordings, labels, or
the HMM; it takes an (n, 15) array and a vector of phase strings. Everything
about WHERE the data came from lives in analysis/train.py, and everything
about smoothing lives in pipeline/hmm.py.

Three decisions that are not the library defaults:

CLASS WEIGHTS ARE ON BY DEFAULT. The recorder runs whenever a wash does, and
a wash is a third idle by window count -- before you count the hours of empty
laundry room a 24/7 deployment will add. Left alone, the model learns that
guessing the majority class is a fine strategy, which scores well and is
useless. balanced weights make one fill window worth as many as thirty wash
windows, so a rare phase can still move a split.

PROBABILITIES, NOT JUST LABELS, ARE THE REAL OUTPUT. predict() exists for
eyeballing, but the thing that goes into production is predict_proba(), because
the HMM needs a distribution per window to run Viterbi over. A classifier that
only emits argmax throws away exactly the information that lets a sequence
model overrule a confidently wrong window.

THE CLASS ORDER IS PINNED AND SAVED. sklearn sorts classes alphabetically:
[done, fill, idle, rinse, spin, wash]. The HMM's transition matrix is written
in cycle order: [idle, fill, wash, rinse, spin]. Those are different orderings
of overlapping sets, and lining them up by position rather than by name is a
bug that produces a working system which is wrong about which phase it is in.
So the classes ride inside the saved model file and every caller looks columns
up by name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "models" / "clf-v1.joblib"

# Small on purpose. With ~1,100 windows per cycle and a handful of cycles, the
# ceiling on what can be learned is set by how many DISTINCT washes exist, not
# by how many windows -- adjacent windows are nearly copies of each other. A
# deeper forest fits the same wash more precisely and generalizes no better.
DEFAULT_PARAMS: dict[str, Any] = {
    "max_iter": 200,
    "learning_rate": 0.08,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    # Reproducibility is not a nicety here: two models trained on the same data
    # that disagree make it impossible to tell a feature change from noise.
    "random_state": 0,
}


def train(X: np.ndarray, y: Sequence[str], *, balanced: bool = True,
          params: Optional[dict[str, Any]] = None) -> HistGradientBoostingClassifier:
    """Fit a classifier on normalized features.

    X is (n_windows, n_features) AFTER pipeline.calibrate.zscore -- raw features
    would train a model pinned to one sensor mount. Nothing here can detect that
    mistake, which is why train.py does the z-scoring in one place.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"{X.shape[0]} feature rows against {y.shape[0]} labels")
    if not np.all(np.isfinite(X)):
        raise ValueError("non-finite feature; check calibrate.MIN_SCALE")
    if len(np.unique(y)) < 2:
        raise ValueError(f"only one class present: {np.unique(y)}")

    clf = HistGradientBoostingClassifier(
        class_weight="balanced" if balanced else None,
        **(params or DEFAULT_PARAMS),
    )
    clf.fit(X, y)
    return clf


def predict(clf: HistGradientBoostingClassifier, X: np.ndarray) -> np.ndarray:
    """Per-window phase. Useful for a confusion matrix; NOT what the server
    should show -- a per-window argmax flickers between phases that sound
    alike, which is what pipeline/hmm.py exists to fix."""
    return clf.predict(np.atleast_2d(np.asarray(X, dtype=float)))


def predict_proba(clf: HistGradientBoostingClassifier, X: np.ndarray) -> np.ndarray:
    """(n_windows, n_classes), columns in clf.classes_ order.

    This is the real interface. The HMM consumes these as emission
    probabilities, and the difference between a window the model is sure about
    and one it barely prefers is exactly what lets the sequence prior win where
    it should and lose where it shouldn't.
    """
    return clf.predict_proba(np.atleast_2d(np.asarray(X, dtype=float)))


def class_columns(clf: HistGradientBoostingClassifier,
                  phases: Sequence[str]) -> np.ndarray:
    """Indices of `phases` within clf.classes_, by name.

    The one function standing between this file and the alignment bug in the
    header. Raises rather than filling in a default, because a phase the model
    was never trained on cannot be a column of zeros -- that silently tells the
    HMM the phase is impossible, and the system confidently reports the wrong
    state forever.
    """
    order = list(clf.classes_)
    missing = [p for p in phases if p not in order]
    if missing:
        raise ValueError(
            f"the model has no class {missing}; it was trained on {order}. "
            "Retrain, or drop these phases from the HMM's state list."
        )
    return np.array([order.index(p) for p in phases])


def save(clf: HistGradientBoostingClassifier, path: Path = DEFAULT_PATH, *,
         meta: Optional[dict[str, Any]] = None) -> Path:
    """Write the model plus the context needed to interpret it.

    The joblib file holds a dict, not a bare estimator, because six months from
    now the question is never "what are its weights" -- it is "which cycles was
    this trained on, with which feature layout, and does that layout still
    match the code". A bare estimator answers none of those, and models/ is
    committed to git, so the file outlives the session that made it.
    """
    from joblib import dump  # imported here so `import pipeline.model` stays cheap
    from pipeline.features import FEATURE_NAMES

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump({
        "clf": clf,
        "features": list(FEATURE_NAMES),
        "classes": [str(c) for c in clf.classes_],
        "meta": meta or {},
    }, path)
    return path


def load(path: Path = DEFAULT_PATH) -> tuple[HistGradientBoostingClassifier, dict[str, Any]]:
    """(classifier, bundle). Raises FileNotFoundError if there is no model yet.

    Checks the saved feature layout against the running features.py and refuses
    a mismatch. A model trained on fifteen features fed sixteen would raise on
    its own; a model trained on fifteen features fed fifteen DIFFERENT ones
    would not, and that is the failure this catches.
    """
    from joblib import load as jload
    from pipeline.features import FEATURE_NAMES

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no model at {path}; run analysis/train.py")

    bundle = jload(path)
    stored = tuple(bundle.get("features", ()))
    if stored != tuple(FEATURE_NAMES):
        raise ValueError(
            "this model was trained on a different feature layout:\n"
            f"  stored:  {list(stored)}\n"
            f"  current: {list(FEATURE_NAMES)}\n"
            "retrain with analysis/train.py"
        )
    return bundle["clf"], bundle


def try_load(path: Path = DEFAULT_PATH) -> Optional[tuple[HistGradientBoostingClassifier, dict[str, Any]]]:
    """load(), but None instead of an exception when there is no model.

    For server/main.py: "no model yet" is a normal state that the status screen
    already renders (predicted=False), and a server that refuses to start
    because it has not been trained is worse than one that reports marks.
    """
    try:
        return load(path)
    except (FileNotFoundError, ValueError):
        return None
