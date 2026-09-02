"""What has to be true about pipeline/features.py before anything trusts it.

A wrong FFT does not raise. It returns an array of the right shape full of
plausible numbers, and the first symptom is a model that is mysteriously
mediocre three days later. So every claim features.py makes is checked here
against a signal whose answer is known analytically -- a sine wave at a
frequency chosen in advance, not a recording whose truth is itself in doubt.

Run:  venv/bin/pytest tests/ -q
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.features import (
    BAND_HI_HZ,
    BAND_LO_HZ,
    FEATURE_NAMES,
    HISTORY_LEN,
    MAX_CONTEXT_GAP,
    N_BANDS,
    N_FEATURES,
    band_edges,
    band_energies,
    extract,
    extract_sequence,
    power_spectrum,
    rms,
    spectral_centroid,
    window_signal,
)

FS = 100.0
N = 256
GRAVITY = 17000.0  # what the vertical axis actually reads on this sensor


def tone(freq: float, amp: float = 3000.0, n: int = N, fs: float = FS,
         offset: float = 0.0, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * freq * t + phase) + offset


def three_axis(freq: float, amp: float = 3000.0) -> np.ndarray:
    """A tone on one axis, gravity on another, noise on the third -- roughly
    what the sensor actually delivers."""
    rng = np.random.default_rng(0)
    return np.array([
        tone(freq, amp) + GRAVITY,
        rng.normal(0, 20, N),
        tone(freq, amp * 0.4, phase=1.1),
    ])


# --------------------------------------------------------------------------
# The band layout
# --------------------------------------------------------------------------


def test_band_edges_span_the_declared_range():
    e = band_edges()
    assert len(e) == N_BANDS + 1
    assert e[0] == pytest.approx(BAND_LO_HZ)
    assert e[-1] == pytest.approx(BAND_HI_HZ)
    assert np.all(np.diff(e) > 0)


def test_band_edges_are_log_spaced():
    """Ratios equal, not differences. A linear layout would give the 0.5-6 Hz
    range -- where a washer's whole rhythm lives -- a single band."""
    ratios = band_edges()[1:] / band_edges()[:-1]
    assert np.allclose(ratios, ratios[0])


# --------------------------------------------------------------------------
# The spectrum itself
# --------------------------------------------------------------------------


# A Hann window's main lobe is 4 bins wide, so nothing narrower than this can
# be resolved -- 1.56 Hz at 100 Hz with n=256. Bands below it are blurred by
# construction; see the note in pipeline/features.py.
LOBE_HZ = 4 * FS / N


@pytest.mark.parametrize("freq", [4.0, 12.0, 25.0, 40.0])
def test_tone_lands_in_the_band_that_contains_it(freq):
    """The one test this file exists for. A wrong FFT still returns numbers of
    the right shape, and this is what tells them apart from right ones."""
    freqs, power = power_spectrum(tone(freq), FS)
    edges = band_edges()
    expected = int(np.searchsorted(edges, freq, side="right") - 1)
    assert edges[expected + 1] - edges[expected] > LOBE_HZ, "test picked an unresolvable band"

    energies = band_energies(freqs, power, edges)
    assert int(np.argmax(energies)) == expected, (
        f"{freq} Hz should be band {expected} "
        f"[{edges[expected]:.2f}, {edges[expected + 1]:.2f}) "
        f"but landed in {int(np.argmax(energies))}"
    )
    # And not marginally: a leaking window would put a close second elsewhere.
    ordered = np.sort(energies)
    assert ordered[-1] - ordered[-2] > 1.0  # >10x the power of any other band


def test_the_lowest_bands_are_blurred_and_that_is_expected():
    """Stated as a test so the limit is recorded rather than discovered.

    The three lowest bands are each narrower than the main lobe, so a 1.2 Hz
    tone spills across them and the widest neighbour can win. They remain
    useful -- they measure how much energy is down there, and band1 is the
    strongest spin separator in the real recordings -- but they are not a
    frequency readout, and no amount of tuning here changes that. Only a
    longer window would, and the node ships 256 samples.
    """
    edges = band_edges()
    assert np.all(np.diff(edges)[:3] < LOBE_HZ)

    freqs, power = power_spectrum(tone(1.2), FS)
    energies = band_energies(freqs, power, edges)
    exact = int(np.searchsorted(edges, 1.2, side="right") - 1)
    assert abs(int(np.argmax(energies)) - exact) <= 1
    # It is still unmistakably low-frequency, which is all the model needs.
    assert int(np.argmax(energies)) < 4


def test_peak_bin_is_the_tone_frequency():
    freqs, power = power_spectrum(tone(12.0), FS)
    assert freqs[int(np.argmax(power))] == pytest.approx(12.0, abs=FS / N)


def test_power_sums_to_the_variance():
    """Parseval. If this drifts, band energies are off by a factor that depends
    on the window length, and two recordings framed differently stop comparing."""
    x = tone(12.0)
    _, power = power_spectrum(x, FS)
    assert power.sum() == pytest.approx(np.var(x), rel=0.02)


def test_centroid_of_a_pure_tone_is_that_tone():
    freqs, power = power_spectrum(tone(12.0), FS)
    assert spectral_centroid(freqs, power) == pytest.approx(12.0, abs=0.5)


def test_centroid_rises_with_frequency():
    """The feature has to be monotonic in the thing it claims to measure --
    it is most of what separates a low agitation rumble from a spin."""
    got = []
    for f in (2.0, 10.0, 30.0):
        freqs, power = power_spectrum(tone(f), FS)
        got.append(spectral_centroid(freqs, power))
    assert got[0] < got[1] < got[2]


def test_rms_matches_the_analytic_value():
    """A sine of amplitude A has RMS A/sqrt(2), regardless of any DC offset.

    12.5 Hz, not 12: 256 samples at 100 Hz is exactly 32 periods of 12.5 Hz, so
    the identity is exact. At 12 Hz the frame ends mid-period and the true RMS
    differs by a few counts -- correct behaviour that would look like a bug.
    """
    assert rms(tone(12.5, amp=3000.0, offset=GRAVITY)) == pytest.approx(3000 / np.sqrt(2), rel=1e-9)


# --------------------------------------------------------------------------
# The invariances -- what must NOT change the answer
# --------------------------------------------------------------------------


def test_gravity_offset_changes_nothing():
    """A 17000-count DC term would otherwise be four orders of magnitude louder
    than the signal, and its leakage would swamp the low bands."""
    quiet, _ = extract(tone(12.0), FS)
    with_g, _ = extract(tone(12.0, offset=GRAVITY), FS)
    assert np.allclose(quiet, with_g)


def test_features_survive_the_sensor_being_rotated():
    """Summing power across axes makes the fingerprint orientation-blind, so a
    remount changes nothing. This is the claim in features.py note 1."""
    frame = three_axis(12.0)
    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))  # a random rotation/reflection

    a, _ = extract(frame, FS)
    b, _ = extract(q @ frame, FS)
    assert np.allclose(a, b, atol=1e-8)


def test_doubling_the_amplitude_moves_every_band_by_log10_of_four():
    """Power goes as amplitude squared, and the bands are log10 -- so the whole
    fingerprint shifts by a constant 0.602 and its SHAPE does not move. That is
    what lets calibration handle a sensor mounted more or less tightly."""
    frame = three_axis(12.0)
    a, _ = extract(frame, FS)
    b, _ = extract(2.0 * frame, FS)  # the same signal, twice as big -- noise included
    bands_a, bands_b = a[:N_BANDS], b[:N_BANDS]
    assert np.allclose(bands_b - bands_a, np.log10(4.0), atol=1e-9)
    assert b[FEATURE_NAMES.index("log_rms")] - a[FEATURE_NAMES.index("log_rms")] == pytest.approx(
        np.log10(2.0), abs=0.02
    )


def test_silence_produces_finite_features():
    """An all-zero window is not hypothetical -- a disconnected sensor reads a
    constant. log10(0) is -inf, and one -inf in a training row becomes NaN in
    the calibration mean and takes every other feature down with it."""
    vec, _ = extract(np.zeros((3, N)), FS)
    assert np.all(np.isfinite(vec))
    assert vec[FEATURE_NAMES.index("centroid_hz")] == 0.0


# --------------------------------------------------------------------------
# Framing and temporal context
# --------------------------------------------------------------------------


def test_window_signal_overlap_and_count():
    x = tone(12.0, n=1024)
    frames = window_signal(x, n=256, overlap=0.5)
    assert frames.shape == (7, 256)  # 1024 samples, hop 128
    assert np.array_equal(frames[0][128:], frames[1][:128])


def test_window_signal_drops_a_partial_tail():
    """A short frame's bins mean different frequencies than a full one's."""
    assert window_signal(tone(12.0, n=300), n=256, overlap=0.5).shape == (1, 256)
    assert window_signal(tone(12.0, n=100), n=256).shape == (0, 256)


def test_feature_vector_is_the_declared_shape():
    vec, _ = extract(three_axis(12.0), FS)
    assert vec.shape == (N_FEATURES,) == (len(FEATURE_NAMES),)


def _fake_windows(freqs, t0=1000.0, dt=2.6):
    out = []
    for i, f in enumerate(freqs):
        frame = three_axis(f)
        out.append({"t": t0 + i * dt, "hz": FS, "n": N,
                    "x": frame[0].tolist(), "y": frame[1].tolist(), "z": frame[2].tolist()})
    return out


def test_first_window_has_no_delta():
    """Zero reads as "nothing changed". Anything else is a transition the
    recording did not contain."""
    X, _ = extract_sequence(_fake_windows([12.0] * 3))
    i = FEATURE_NAMES.index("d_low")
    assert np.allclose(X[0, i:i + 3], 0.0)


def test_a_gap_resets_the_context():
    """Cycle 1 has a 310-second hole. Differencing across it would invent a
    phase transition at exactly the moment the wash actually changed phase."""
    windows = _fake_windows([2.0, 2.0, 30.0, 30.0])
    windows[2]["t"] += MAX_CONTEXT_GAP + 60  # a hole before the frequency change
    for w in windows[3:]:
        w["t"] += MAX_CONTEXT_GAP + 60

    X, _ = extract_sequence(windows)
    d = slice(FEATURE_NAMES.index("d_low"), FEATURE_NAMES.index("d_low") + 3)
    assert np.allclose(X[2, d], 0.0), "the window after a gap must not carry a delta"

    # And the other half of the claim: without the gap, that same frequency
    # change DOES show up as a delta. A reset that swallowed real transitions
    # would pass the assertion above and be useless.
    X2, _ = extract_sequence(_fake_windows([2.0, 2.0, 30.0, 30.0]))
    assert np.abs(X2[2, d]).max() > 0.5


def test_times_are_window_centres():
    """features.py and analysis/label.py must agree on what a window's time is,
    or every label is offset by half a window and nothing says so."""
    windows = _fake_windows([12.0, 12.0])
    _, times = extract_sequence(windows)
    assert times[0] == pytest.approx(windows[0]["t"] - N / (2 * FS))


def test_rolling_mean_only_uses_the_recent_past():
    """Purely a shape check on the context window: the rolling feature of row k
    must not depend on anything after row k, or the model is reading the future
    and evaluation is fiction."""
    windows = _fake_windows([2.0] * 10)
    X_all, _ = extract_sequence(windows)
    X_head, _ = extract_sequence(windows[:6])
    i = FEATURE_NAMES.index("roll_mid")
    assert np.allclose(X_all[:6, i], X_head[:, i])
    assert HISTORY_LEN >= 2


# --------------------------------------------------------------------------
# The real washer, if it is on this machine
# --------------------------------------------------------------------------

RECORDINGS = Path(__file__).resolve().parent.parent / "analysis" / "recordings"
LABELS = Path(__file__).resolve().parent.parent / "analysis" / "labels"


def _labels_with_data() -> list[Path]:
    """Label files whose recording is actually present on this machine.

    analysis/labels/ is tracked and analysis/recordings/ is not, so a fresh
    clone has every label and none of the data. Checking only for labels turns
    that into a FileNotFoundError instead of a skip -- and a suite that fails
    the first time someone clones the repo reads as a broken project, not as
    missing data.
    """
    if not (LABELS.is_dir() and RECORDINGS.is_dir()):
        return []
    return [p for p in sorted(LABELS.glob("*.json"))
            if (RECORDINGS / json.loads(p.read_text())["file"]).exists()]


LABEL_FILES = _labels_with_data()


@pytest.mark.skipif(not LABEL_FILES, reason="no labelled recordings on this machine")
def test_the_phases_actually_separate_on_the_real_washer():
    """The one check a synthetic sine cannot make: that these features actually
    separate the phases on the machine in the room. Skipped for anyone who cloned the
    repo, because analysis/recordings/ is gitignored and cannot be shipped.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.label import label_windows, load_recording

    label = json.loads(LABEL_FILES[-1].read_text())
    windows = load_recording(RECORDINGS / label["file"])
    X, _ = extract_sequence(windows)
    y = np.array([lab or "" for lab in label_windows(windows, label["phases"])])

    rms_i = FEATURE_NAMES.index("log_rms")
    cent_i = FEATURE_NAMES.index("centroid_hz")
    low_i = FEATURE_NAMES.index("band1")

    def med(phase, col):
        return float(np.median(X[y == phase, col]))

    assert med("idle", rms_i) < med("wash", rms_i) < med("spin", rms_i)

    # The low band is the spin separator: an out-of-balance drum rocks the
    # whole chassis at ~1 Hz, and nothing else in the cycle does that.
    assert med("spin", low_i) > med("wash", low_i) + 1.0

    # Only that the centroid MOVES, not which way. It moves down, which is the
    # opposite of the intuition that a spin is a higher-pitched noise: spin
    # 15.6/17.6 Hz against wash 20.2/19.9 across the two recorded cycles, and
    # idle is highest of all at ~23.6 because a near-silent sensor is reading
    # its own white noise floor. Asserting the direction would encode two
    # cycles' worth of one washer as a law.
    assert abs(med("spin", cent_i) - med("wash", cent_i)) > 1.0
