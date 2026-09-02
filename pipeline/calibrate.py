"""Per-machine normalization: what "off" sounds like, and everything relative to it.

A feature vector straight out of features.py is in the units of one sensor
taped to one washer. Move the sensor 10 cm, tighten the tape, swap in a
different unit, put the same model on a second machine -- and every band
energy shifts by an amount that has nothing to do with what the washer is
doing. This file removes that shift by expressing each feature as a distance
from THAT machine's own quiet baseline, measured in that machine's own idle
jitter.

    z = (feature - centre) / scale        centre, scale learned from idle only

For the eight band energies (which are log10 power) this has a physical
reading: z is "how many idle-noise-widths above this machine's own noise
floor". Spin on a tightly-mounted sensor and spin on a loose one land in the
same place, which is the entire reason one model can cover any washer.

WHY THIS MATTERS EVEN WITH ONE MACHINE, TODAY. The classifier is a tree
ensemble and trees are invariant to any monotone rescaling of a feature --
z-scoring cannot improve accuracy on the machine it was trained on, and it is
fair to ask why bother. Two reasons, both about the future: the split points
a tree learns are absolute numbers, so a model trained on raw features is
silently pinned to this exact mount and re-taping the sensor invalidates it
with no error message; and retrofitting normalization after labels, a schema
and a trained model exist means redoing all three. It costs an afternoon now
and a rebuild later.

The traps, in the order they bite:

  A SILENT BAND HAS NO VARIANCE. Band 0 on an idle washer is sensor noise
  measured in the fourth decimal place. Divide by that standard deviation and
  ordinary noise becomes a z-score in the thousands, which then dominates
  every split the tree considers. MIN_SCALE floors it.

  ONE SLAMMED DOOR RUINS AN AVERAGE. The baseline is learned from a laundry
  room, not a lab: someone walks past, the dryer next to it starts, a cat
  jumps on the machine. A mean and a standard deviation are both wrecked by a
  handful of samples like that -- the mean drifts up and the spread inflates,
  so every real phase afterwards looks closer to idle than it is. Median and
  MAD ignore anything under half the samples, which is the right shape for
  this failure.

  A CALIBRATION IS ONLY VALID FOR ONE FEATURE LAYOUT. The stored blob is a
  list of numbers; if features.py ever gains a band or reorders a column, an
  old blob still has the right length and the wrong meaning, and every z-score
  is subtly wrong with nothing to notice. So the feature NAMES ride along in
  the blob and zscore() refuses to run against a mismatch.

  YOU CAN CALIBRATE ON A RUNNING WASHER. Point this at a recording that is
  90% wash and the "quiet baseline" is a wash cycle; afterwards a real wash
  reads as z ~ 0 and idle reads as strongly negative, which looks plausible
  on a plot and is completely wrong. check_baseline() exists to catch it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from pipeline.features import FEATURE_NAMES, N_FEATURES

# 1.4826 makes the MAD an unbiased estimate of the standard deviation for
# normally distributed data, so `scale` stays comparable to a std and the
# floors below can be reasoned about in familiar units.
MAD_TO_SIGMA = 1.4826

# The floor on scale. It exists to catch a DEGENERATE feature -- one that is
# literally constant across every quiet window -- not to clip small-but-real
# jitter, so it is set below anything this washer actually produces. Measured
# idle MADs on the two recorded cycles run from 0.011 (log_rms, the steadiest)
# up to 0.29 (band0), reproducing to within ~20% across two different nights;
# 0.005 sits below all of them and above floating-point noise.
MIN_SCALE = 0.005

# What fraction of an unlabelled recording is assumed to be quiet when there
# are no labels to say. A machine that runs a few hours a day is mostly idle,
# so the lowest fifth by amplitude is a safe bet -- and check_baseline()
# verifies the assumption rather than trusting it.
QUIET_QUANTILE = 0.20

# A baseline built from fewer windows than this is noise pretending to be a
# statistic: MAD over 20 samples has an enormous standard error of its own.
MIN_QUIET_WINDOWS = 60  # ~2.5 minutes at one window every 2.6 s


@dataclass
class Baseline:
    """One machine's idea of silence. Stored on machines.calibration as JSON."""

    features: tuple[str, ...]
    centre: np.ndarray
    scale: np.ndarray
    n: int
    method: str = "median/MAD"
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.centre = np.asarray(self.centre, dtype=float)
        self.scale = np.asarray(self.scale, dtype=float)
        if not (len(self.features) == self.centre.size == self.scale.size):
            raise ValueError("features, centre and scale must be the same length")
        if not np.all(np.isfinite(self.centre)) or not np.all(np.isfinite(self.scale)):
            raise ValueError("a non-finite baseline would poison every later feature")
        if np.any(self.scale < MIN_SCALE):
            raise ValueError(f"scale below MIN_SCALE={MIN_SCALE}; use learn_baseline()")

    def __repr__(self) -> str:
        return f"<Baseline {len(self.features)} features from {self.n} quiet windows>"


def quiet_rows(X: np.ndarray, quantile: float = QUIET_QUANTILE,
               feature_names: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
    """Boolean mask of the quietest rows, for a machine with no labels yet.

    Ranked by log_rms, which is the one feature that means the same thing on
    every machine before calibration: total amplitude. The labelled path --
    "use the windows a human marked idle" -- is better and analysis/train.py
    uses it; this exists so a NEW washer can be calibrated from an hour of
    recording without anyone marking anything, which is what makes adding a
    second machine an insert rather than a project.
    """
    rms = X[:, list(feature_names).index("log_rms")]
    return rms <= np.quantile(rms, quantile)


def learn_baseline(quiet_X: np.ndarray,
                   feature_names: Sequence[str] = FEATURE_NAMES) -> Baseline:
    """Fit centre and scale from windows believed to be quiet.

    Median and MAD rather than mean and std -- see the trap about slammed
    doors. Where the MAD collapses to zero (a feature that is literally
    constant across every quiet window, which happens to the delta features
    when the recording has no gaps and the machine truly is still) the scale
    falls back to MIN_SCALE rather than to zero.
    """
    X = np.atleast_2d(np.asarray(quiet_X, dtype=float))
    if X.shape[1] != len(feature_names):
        raise ValueError(f"expected {len(feature_names)} features, got {X.shape[1]}")
    if not np.all(np.isfinite(X)):
        raise ValueError("non-finite feature in the quiet set; check features.py first")

    centre = np.median(X, axis=0)
    mad = np.median(np.abs(X - centre), axis=0) * MAD_TO_SIGMA
    scale = np.maximum(mad, MIN_SCALE)

    return Baseline(features=tuple(feature_names), centre=centre, scale=scale, n=X.shape[0])


def zscore(X: np.ndarray, baseline: Baseline,
           feature_names: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
    """Normalize one vector or a whole matrix against a machine's baseline.

    The name check is the point of the argument: a stored blob and a running
    features.py that disagree about what column 4 is produce numbers, not
    errors. Shapes match, everything is finite, and every value is wrong.
    """
    if tuple(feature_names) != tuple(baseline.features):
        raise ValueError(
            "this baseline was built for a different feature layout:\n"
            f"  stored:  {list(baseline.features)}\n"
            f"  current: {list(feature_names)}\n"
            "recalibrate the machine rather than trusting these numbers"
        )
    A = np.asarray(X, dtype=float)
    return (A - baseline.centre) / baseline.scale


def check_baseline(baseline: Baseline, X: np.ndarray,
                   feature_names: Sequence[str] = FEATURE_NAMES) -> list[str]:
    """Sanity-check a baseline against the recording it came from.

    Returns a list of human-readable warnings -- empty is good. Called by
    analysis/train.py and printed, rather than raised, because every one of
    these can be legitimate on some machine and the right response is a person
    looking at it, not a crash mid-training.
    """
    warnings: list[str] = []
    z = zscore(X, baseline, feature_names)
    rms_i = list(feature_names).index("log_rms")

    if baseline.n < MIN_QUIET_WINDOWS:
        warnings.append(
            f"only {baseline.n} quiet windows (want >= {MIN_QUIET_WINDOWS}); "
            "the scale estimate is itself noisy"
        )

    # The one that catches calibrating on a running washer: if the baseline is
    # really the quiet end of the recording, most of the recording should sit
    # at or above it. A median z near zero across EVERYTHING means the
    # "baseline" is the whole distribution, i.e. the machine was never off.
    median_z = float(np.median(z[:, rms_i]))
    if median_z < 0.5:
        warnings.append(
            f"median log_rms z-score is {median_z:.2f}: the quiet set may not be quiet "
            "-- was this recorded while the machine was running?"
        )

    floored = int(np.sum(baseline.scale <= MIN_SCALE))
    if floored:
        names = [n for n, s in zip(baseline.features, baseline.scale) if s <= MIN_SCALE]
        warnings.append(f"{floored} feature(s) hit the scale floor: {', '.join(names)}")

    spread = float(np.max(np.abs(z)))
    if spread > 200:
        warnings.append(f"largest |z| is {spread:.0f}; a feature is close to constant when idle")

    return warnings


# --------------------------------------------------------------------------
# Storage -- the machines.calibration JSON blob
# --------------------------------------------------------------------------


def serialize(baseline: Baseline) -> dict[str, Any]:
    """Baseline -> plain JSON. Lists, not arrays: this goes through SQLAlchemy's
    JSON column and numpy types do not survive that trip."""
    return {
        "features": list(baseline.features),
        "centre": [float(v) for v in baseline.centre],
        "scale": [float(v) for v in baseline.scale],
        "n": int(baseline.n),
        "method": baseline.method,
        "created_at": float(baseline.created_at),
    }


def deserialize(blob: Optional[dict[str, Any]]) -> Optional[Baseline]:
    """JSON -> Baseline, or None for an uncalibrated machine.

    None rather than an exception because "no calibration yet" is a normal
    state -- machines.calibration is NULL until a machine has been recorded --
    and the caller has to handle it either way.
    """
    if not blob:
        return None
    return Baseline(
        features=tuple(blob["features"]),
        centre=np.asarray(blob["centre"], dtype=float),
        scale=np.asarray(blob["scale"], dtype=float),
        n=int(blob.get("n", 0)),
        method=blob.get("method", "median/MAD"),
        created_at=float(blob.get("created_at", 0.0)),
    )
