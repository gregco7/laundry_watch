"""Raw window -> ~15-number fingerprint. Pure numpy, no state, no I/O.

Everything downstream rests on this file, and a wrong FFT does not crash --
it produces plausible garbage for days. tests/test_features.py exists for
exactly that reason: a synthetic tone at a known frequency has to land its
energy in the band that contains it, or nothing below is worth running.

Per window (256 samples, ~2.56 s at 100 Hz):

    per axis: subtract mean -> Hann taper -> rfft -> |X|^2
    sum the three axes' power  ->  one spectrum
    reduce to: 8 log-spaced band energies, centroid, tilt, RMS,
               3 deltas vs the previous window, 1 rolling mean

Four choices worth defending:

1. THE THREE AXES ARE SUMMED IN POWER, NOT PICKED FROM.
   Adding |X|^2 + |Y|^2 + |Z|^2 per frequency bin gives the total vibration
   power at that frequency, and that sum is invariant to how the sensor is
   rotated -- it is the trace of the spectral matrix, and a rotation cannot
   change a trace. Picking "the z axis" instead would mean every feature
   silently changes the day the sensor is re-taped at a different angle, and
   the model would quietly get worse with no bug to find. Calibration handles
   a change of scale; only this handles a change of orientation.
   (tests/test_features.py rotates a signal by a random orthogonal matrix and
   asserts the features do not move.)

2. BAND ENERGIES ARE STORED AS log10.
   Idle is ~33 counts RMS and spin ~660: a factor of 20 in amplitude, 400 in
   power, and worse per band. Linear energies are heavy-tailed, so a z-score
   against a quiet baseline turns into a number in the thousands and the
   split points a tree learns are all crammed against zero. Logs make the
   distance from idle to wash and from wash to spin comparable sizes.

3. MEAN REMOVAL IS NOT OPTIONAL, AND IT IS PER AXIS.
   Gravity is a constant ~17000 counts on whichever axis is vertical. Its
   square dwarfs the vibration by four orders of magnitude, and it lands in
   the DC bin where it would leak across the whole low band through the
   window's sidelobes. Removing the mean per axis also makes the features
   immune to a DC offset drifting with temperature.

4. TEMPORAL CONTEXT RESETS ACROSS A GAP.
   The delta and rolling-mean features describe "what changed since the last
   window". Cycle 1 has a 310-second hole in it: the window after that hole
   has a previous window five minutes old, and differencing against it
   invents a transition that never happened -- at the exact moment the wash
   changed phase, which is the worst possible place for a lie. extract_sequence
   resets context whenever the gap exceeds MAX_CONTEXT_GAP.

A limit worth knowing before you read the low bands: at 100 Hz with n=256 the
bin spacing is 0.39 Hz and a Hann window's main lobe is ~4 bins, so anything
narrower than ~1.6 Hz cannot be resolved. The three lowest bands are narrower
than that. They still MEASURE how much energy sits down there -- and band1
turns out to be the single strongest spin/wash separator in the recordings --
but they cannot tell 1.2 Hz from 1.6 Hz, and a tone in one of them leaks into
its neighbours. That is a property of a 2.56 s window, not a bug; the node
would have to send longer windows to fix it. tests/test_features.py asserts
exact band placement only above the lobe width, and states the limit below it.

One deliberate deviation from the plan in CLAUDE.md, which asked for "top-3
band deltas": the three deltas here are of FIXED low/mid/high aggregates, not
of whichever three bands happened to be loudest. A feature column whose
meaning changes from row to row is not learnable -- column 12 would be "the
delta of band 2" in one window and "the delta of band 7" in the next, and the
model has no way to know which.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import numpy as np

# 8 bands, log-spaced. The low edge is 0.5 Hz because at 100 Hz with n=256 the
# bin spacing is 0.39 Hz -- below ~0.5 Hz there is nothing resolvable, only the
# skirt of the DC bin we just removed. The high edge is Nyquist, and there is
# no point pretending otherwise.
N_BANDS = 8
BAND_LO_HZ = 0.5
BAND_HI_HZ = 50.0

# The three aggregates the deltas and the tilt are built from. Chosen from the
# physics, not from the data: drum rotation and agitation live below ~3 Hz,
# the wash's mechanical clatter in the middle, and spin's broadband roar and
# out-of-balance harmonics up top.
LOW_HZ = (0.5, 3.0)
MID_HZ = (3.0, 15.0)
HIGH_HZ = (15.0, 50.0)

# Windows arrive every ~2.6 s. Anything past this is a hole, not a cadence --
# see note 4 above. Deliberately tight: two missed windows in a row is already
# enough for "what changed since last time" to stop meaning anything.
MAX_CONTEXT_GAP = 8.0

# ~13 s of context at one window every 2.6 s. Long enough to smooth the
# second-to-second noise in agitation, short enough that a phase change is not
# smeared across a minute of the recording.
HISTORY_LEN = 5

# Nothing here is ever zero in practice, but a silent band in a synthetic test
# is, and log10(0) is -inf, which propagates into every mean and every z-score
# downstream as NaN. Floored rather than clipped so the value stays monotonic.
EPS = 1e-12

FEATURE_NAMES: tuple[str, ...] = (
    "band0", "band1", "band2", "band3", "band4", "band5", "band6", "band7",
    "centroid_hz",
    "tilt",          # log high energy - log low energy: where the weight sits
    "log_rms",
    "d_low", "d_mid", "d_high",
    "roll_mid",      # mid-band log energy, averaged over the last HISTORY_LEN
)
N_FEATURES = len(FEATURE_NAMES)


def band_edges(n_bands: int = N_BANDS, lo: float = BAND_LO_HZ, hi: float = BAND_HI_HZ) -> np.ndarray:
    """n_bands+1 log-spaced edges. Log-spaced because a fixed-width band is the
    wrong shape for this signal: 0.5-6 Hz is where a washer's rhythm lives and
    deserves several bands, while 40-50 Hz is one undifferentiated hiss."""
    return np.logspace(np.log10(lo), np.log10(hi), n_bands + 1)


def window_signal(samples: Sequence[float], n: int = 256, overlap: float = 0.5) -> np.ndarray:
    """Slice a continuous signal into overlapping frames, shape (frames, n).

    The live path never needs this -- the node already ships exactly one 256-
    sample frame per POST, so a recording is a sequence of frames, not a
    stream. It exists for the tests (which synthesize long signals) and for
    re-framing a recording at a different window length later without having
    to re-record a wash.

    Partial trailing frames are dropped: a short frame's spectrum is computed
    over a different duration, so its bins mean different frequencies.
    """
    x = np.asarray(samples, dtype=float)
    step = max(1, int(round(n * (1.0 - overlap))))
    if x.size < n:
        return np.empty((0, n))
    starts = range(0, x.size - n + 1, step)
    return np.stack([x[s:s + n] for s in starts])


def _as_axes(frame: Any) -> np.ndarray:
    """Accept one axis (n,) or several (axes, n) and always return 2-D."""
    a = np.asarray(frame, dtype=float)
    return a[None, :] if a.ndim == 1 else a


def power_spectrum(frame: Any, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """(freqs, power) for one window -- mean-removed, Hann-tapered, axes summed.

    Scaled so that power.sum() equals the signal's variance (Parseval). That
    normalization is not cosmetic: it makes a band energy readable as "counts
    squared in this band", so band energies and RMS are in the same units and
    a sanity check like "the bands should add up to roughly the total" works.
    Without it the numbers are proportional to the right answer by a factor
    involving the window length -- fine until you compare two recordings taken
    at different n.
    """
    axes = _as_axes(frame)
    n = axes.shape[1]

    # Hann, because a rectangular window is a hard discontinuity at the frame
    # edges and its transform smears one pure tone across every bin in the
    # spectrum -- a 12 Hz agitation would show up as energy at 40 Hz.
    taper = np.hanning(n)
    tapered = (axes - axes.mean(axis=1, keepdims=True)) * taper

    spec = np.fft.rfft(tapered, axis=1)
    power = (np.abs(spec) ** 2).sum(axis=0)

    # Parseval for numpy's unnormalized DFT, corrected for the taper's power
    # loss, then doubled for the negative frequencies rfft folded away. DC and
    # Nyquist have no mirror image and are not doubled.
    power = power * (2.0 / (n * np.sum(taper ** 2)))
    power[0] /= 2.0
    if n % 2 == 0:
        power[-1] /= 2.0

    return np.fft.rfftfreq(n, 1.0 / fs), power


def band_energies(freqs: np.ndarray, power: np.ndarray,
                  edges: Optional[np.ndarray] = None) -> np.ndarray:
    """log10 of the power summed within each band. Half-open [lo, hi) so a bin
    that sits exactly on an edge belongs to one band, not two."""
    if edges is None:
        edges = band_edges()
    out = np.empty(len(edges) - 1)
    for i in range(len(edges) - 1):
        sel = (freqs >= edges[i]) & (freqs < edges[i + 1])
        out[i] = power[sel].sum()
    return np.log10(np.maximum(out, EPS))


def _range_energy(freqs: np.ndarray, power: np.ndarray, span: tuple[float, float]) -> float:
    sel = (freqs >= span[0]) & (freqs < span[1])
    return float(np.log10(max(power[sel].sum(), EPS)))


def spectral_centroid(freqs: np.ndarray, power: np.ndarray,
                      lo: float = BAND_LO_HZ, hi: float = BAND_HI_HZ) -> float:
    """Centre of mass of the spectrum, in Hz -- one number for "is this a low
    rumble or a high hiss", which is most of what separates agitation from spin.

    Restricted to the band of interest so the DC bin (already ~0 after mean
    removal, but not exactly) cannot drag the answer toward zero. Returns 0.0
    on silence, where a centroid has no meaning; a NaN here would propagate
    into the calibration mean and take every feature with it.
    """
    sel = (freqs >= lo) & (freqs < hi)
    total = power[sel].sum()
    if total <= EPS:
        return 0.0
    return float((freqs[sel] * power[sel]).sum() / total)


def rms(frame: Any) -> float:
    """Mean-removed RMS across all axes, in raw counts.

    Deliberately the same quantity tools/strip.py plots and the phase table in
    the notes is written in (idle ~33, wash ~150, spin ~572 on one axis), so a
    feature vector can be sanity-checked against a number measured by hand.
    """
    axes = _as_axes(frame)
    dev = axes - axes.mean(axis=1, keepdims=True)
    return float(np.sqrt((dev ** 2).mean(axis=1).sum()))


def deltas(low: float, mid: float, high: float,
           prev: Optional[tuple[float, float, float]]) -> np.ndarray:
    """Change in the three aggregate log energies since the previous window.

    A phase boundary is a change, not a level: rinse agitation and wash sit at
    similar amplitudes, and what distinguishes the start of a spin is that
    everything moved at once. Zeros when there is no usable previous window --
    the first window of a recording, or the first after a gap -- which reads as
    "nothing changed" rather than as a fabricated jump.
    """
    if prev is None:
        return np.zeros(3)
    return np.array([low - prev[0], mid - prev[1], high - prev[2]])


def rolling_mean(history: Sequence[float]) -> float:
    """Mean of the recent mid-band energies. Empty history returns 0.0."""
    return float(np.mean(history)) if len(history) else 0.0


def extract(frame: Any, fs: float,
            prev: Optional[tuple[float, float, float]] = None,
            history: Optional[Sequence[float]] = None) -> tuple[np.ndarray, tuple[float, float, float]]:
    """One window -> (feature vector, the context the NEXT window needs).

    Returning the context rather than mutating anything keeps this file pure:
    the same call with the same arguments gives the same answer, which is what
    makes it testable and what makes the server's live path and the offline
    trainer provably identical. Two implementations of feature extraction that
    drift apart is the classic way to train a model that works on the bench
    and not on the machine.
    """
    freqs, power = power_spectrum(frame, fs)
    bands = band_energies(freqs, power)

    low = _range_energy(freqs, power, LOW_HZ)
    mid = _range_energy(freqs, power, MID_HZ)
    high = _range_energy(freqs, power, HIGH_HZ)

    vec = np.concatenate([
        bands,
        [spectral_centroid(freqs, power),
         high - low,
         np.log10(max(rms(frame), EPS))],
        deltas(low, mid, high, prev),
        [rolling_mean(history if history is not None else [])],
    ])
    assert vec.shape == (N_FEATURES,), f"{vec.shape} != {(N_FEATURES,)}"
    return vec, (low, mid, high)


def extract_sequence(windows: Iterable[dict[str, Any]],
                     axes: Sequence[str] = ("x", "y", "z")) -> tuple[np.ndarray, np.ndarray]:
    """A recording's windows -> (X of shape (n_windows, N_FEATURES), centre times).

    `windows` are the node's payload dicts as stored by the server, in time
    order: {"t", "hz", "n", "x", "y", "z", ...}. This is the only function in
    pipeline/ that knows that shape, and it is the seam where the temporal
    context is managed -- including the gap reset described at the top.

    Times returned are window CENTRES, matching analysis/label.py, so a feature
    row and a label row refer to the same moment. Two files disagreeing about
    what `t` means is an offset nobody finds later.
    """
    X: list[np.ndarray] = []
    times: list[float] = []

    prev: Optional[tuple[float, float, float]] = None
    prev_t: Optional[float] = None
    history: list[float] = []

    for w in windows:
        fs = float(w["hz"])
        n = int(w["n"])
        t = float(w["t"])

        if prev_t is not None and t - prev_t > MAX_CONTEXT_GAP:
            prev, history = None, []

        frame = np.array([w[a] for a in axes], dtype=float)
        vec, context = extract(frame, fs, prev=prev, history=history)

        X.append(vec)
        times.append(t - n / (2.0 * fs))

        prev, prev_t = context, t
        history.append(context[1])  # mid-band energy
        if len(history) > HISTORY_LEN:
            history.pop(0)

    return (np.array(X) if X else np.empty((0, N_FEATURES))), np.array(times)
