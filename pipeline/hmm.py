"""Temporal smoothing: turn per-window guesses into a sequence that makes sense.

The classifier looks at 2.56 seconds at a time and has no idea what happened
before. That is fine for wash against idle and hopeless for the two things this
washer actually does:

  A DRAIN SPIN IN THE MIDDLE OF RINSE. Four minutes of the loudest noise in the
  whole cycle, louder than the final spin, occurring while the panel says Rinse.
  Per-window, it is a spin. Nothing in those 2.56 seconds says otherwise, and no
  amount of feature engineering will change that.

  WASH AND RINSE AGITATION. The same mechanism at the same level. The only thing
  that separates them is that one comes before the drain and one after.

Both are answered the same way: by position in the sequence. A washer runs
idle -> fill -> wash -> rinse -> spin and then stops, so reaching spin early
would require a spin -> rinse transition afterwards, which the matrix makes
nearly impossible. Viterbi therefore takes the cheaper path -- calling the
mid-cycle burst rinse -- and the drain spin resolves itself from structure that
was never in the audio.

TWO ALGORITHMS, ON PURPOSE:

  viterbi()  is offline. It sees a whole recording and picks the single most
             likely path through it. That is what analysis/evaluate.py scores,
             and it is allowed to revise the past: a window called spin at
             13:04 can become rinse once 13:20 turns out to be quiet.

  Filter     is online, and it is what server/main.py runs. It cannot see the
             future, so it computes P(phase now | everything so far) one window
             at a time. It is strictly worse than Viterbi and it is the only
             honest choice for a live screen -- a dashboard that retroactively
             changed what it said ten minutes ago would be lying about what it
             knew at the time.

The gap between them is worth measuring rather than assuming; evaluate.py
reports both.

WHY NOT hmmlearn: the transition matrix here is not learned, it is asserted
from how a washing machine works, and the emissions come from a classifier
rather than from a Gaussian. What is left is thirty lines of dynamic
programming, and thirty lines that can be read beat a dependency that has to be
argued with.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# Cycle order, which is also the order of the matrix. `done` is not a state:
# it is acoustically identical to idle, so the model would have to guess it
# from nothing. It is DERIVED -- the moment the path returns to idle after
# having been in spin -- which is what done_time() does.
STATES: tuple[str, ...] = ("idle", "fill", "wash", "rinse", "spin")

# Mean phase length in windows (~2.6 s each), measured across the three
# recorded cycles: fill 106-130, wash 284-344, rinse 306-357, spin 173-178.
# idle is not a duration so much as "the rest of the day", and is set long
# enough that the prior does not push the machine into a wash on its own.
MEAN_WINDOWS: dict[str, float] = {
    "idle": 600.0,
    "fill": 120.0,
    "wash": 320.0,
    "rinse": 340.0,
    "spin": 175.0,
}

# Probability mass for transitions the washer does not make -- skipping a
# phase, or running backwards. Not zero: Viterbi needs SOME path to exist for a
# recording that starts mid-cycle or has a hole in it, or it returns whatever
# the tie-break happened to pick.
#
# It was 1e-4, which was far too generous. Live on 2026-09-01 the filter went
# straight from idle to rinse eleven seconds into a wash, on a 0.44-vs-0.41
# emission, and then could never come back because the chain is forward-only.
# A leak is for a phase that was genuinely missed, not a shortcut through two
# of them; starting mid-cycle is what initial_distribution() is for, and a hole
# big enough to swallow a phase resets the filter anyway.
LEAK = 1e-8

# Emissions raised to this power before use. The classifier is trained on
# windows that overlap in content and is therefore overconfident -- it reports
# 0.999 for a window it has essentially memorized, and one such window can
# overpower a transition prior trying to say "you cannot be in spin yet".
# Tempering flattens the emissions without changing their order.
#
# Chosen by sweeping it in analysis/evaluate.py, leave-one-cycle-out over the
# three recorded cycles. Both ends of the range fail, in opposite ways:
#
#   power  viterbi acc   online end error   ends detected
#   1.00      0.830          1800 s              3/3   classifier overrules the
#                                                      prior; the drain spin is
#                                                      called a spin and the
#                                                      cycle "finishes" at the
#                                                      quiet refill after it
#   0.50      0.858            35 s              3/3
#   0.10      0.914            25 s              3/3   <- default
#   0.02      0.941            43 s              2/3   evidence too weak to move
#                                                      the state at all; the run
#                                                      that scores best is also
#                                                      the one that misses an end
#
# 0.02 has the highest per-window accuracy and is unusable: the thing anyone
# actually wants is the finish time, and it loses one of three. A control worth
# knowing about, in evaluate.py: run this matrix on EXACTLY uniform emissions
# and it scores 0.176 -- with no evidence at all the diagonal simply parks in
# idle forever, which is the proof that these numbers come from the audio and
# not from the schedule. Three cycles is thin ground for a hyperparameter;
# re-run the sweep once there are six.
EMISSION_POWER = 0.1

# How many consecutive windows the belief must disagree with what is on screen
# before the screen changes. Output hysteresis, not a constraint on the belief.
#
# It exists because on 2026-09-01 the live filter went idle -> rinse eleven
# seconds into a wash on a 0.44-vs-0.41 emission, and a forward-only chain
# meant it could never come back. A washer's shortest phase is ~106 windows, so
# nothing in this range can hide a real transition.
#
# Swept leave-one-cycle-out over the three recorded cycles. Note that the
# metric that matters and the metric that looks impressive disagree:
#
#   dwell   acc    fill   end error   ends found
#       0   0.909  0.46      25 s        3/3
#       8   0.902  0.33       5 s        3/3     <- default
#      12   0.896  0.33       5 s        2/3
#      30   0.850  0.38      55 s        2/3     first attempt, much worse
#
# 8 costs half a point of accuracy and a third of fill recall, and takes
# cycle-end error from 25 s to 5 s (-5, -6, -5 s). That is the one number
# anybody acts on, so it wins. Be honest about WHY it is 5 s: the ~21 s of lag
# happens to cancel the ~25 s it takes a human to notice and tap Done, so this
# is 5 s against a human's judgement, not against physics.
#
# 12 already misses a cycle -- cycle 1's recording stops shortly after the wash
# does, so any longer lag runs out of windows. That is a property of that
# recording rather than of the method, but it is why 8 and not 12.
MIN_DWELL = 8

# Consecutive windows of raw "this is idle" evidence, after a spin, before the
# wash is called finished.
#
# Done-detection deliberately does NOT read the smoothed phase. Smoothing
# exists to stop the phase LABEL flickering, and it costs lag -- which is
# exactly the wrong trade for the one event anybody acts on. Measured against
# the moment the drum actually stops, reading the smoothed phase puts the
# finish 65 s late; reading the classifier's own per-window opinion of the same
# windows puts it at 28 s, because the cliff from a spin to a still machine is
# the least ambiguous thing in the entire recording and needs no help.
#
# So the two consumers of the same model get what each needs: the phase label
# is slow and stable, the finish is fast and direct.
#
# Swept leave-one-cycle-out, measured against the moment the drum stops (found
# in the recording as the last loud window followed by 20 quiet ones), which is
# a harder and more honest target than the human's Done tap -- a person taps 70
# s after the machine stops, with a spread of only 5 s, so scoring against the
# tap flatters any detector whose lag happens to be similar.
#
#   DONE_QUIET   vs drum stop   vs the tap
#            3       38 s          32 s      <- default: 45, 36, 33 per cycle
#            5       44 s          27 s
#            8       52 s          18 s
#
# 3 is the floor worth having: a single spurious idle window during a spin
# cannot fire it, and three consecutive ones (~8 s) never happened in any of
# the three recorded cycles.
DONE_QUIET = 3

EPS = 1e-12


def build_transitions(states: Sequence[str] = STATES,
                      mean_windows: Optional[dict[str, float]] = None,
                      leak: float = LEAK) -> np.ndarray:
    """The 5x5 matrix, from durations and the order of a wash.

    Each row is: stay with probability 1 - 1/mean_length, move on with the
    rest. That is the geometric distribution whose mean is the measured phase
    length -- crude (it makes a phase most likely to end immediately) but it
    encodes the one fact that matters, that phases last minutes rather than
    windows, and it has no parameters to fit beyond a number measured from the
    labels.
    """
    mean_windows = mean_windows or MEAN_WINDOWS
    n = len(states)
    A = np.full((n, n), leak)

    for i, s in enumerate(states):
        stay = 1.0 - 1.0 / max(mean_windows.get(s, 100.0), 2.0)
        # The cycle wraps: after spin the machine goes quiet, which is idle.
        # That wrap is what makes "done" detectable as a transition rather
        # than as a sound.
        nxt = (i + 1) % n
        A[i, i] = stay
        A[i, nxt] = 1.0 - stay
        A[i] /= A[i].sum()

    return A


def initial_distribution(states: Sequence[str] = STATES,
                         start: str = "idle", confidence: float = 0.9) -> np.ndarray:
    """Where a recording is assumed to begin.

    Weighted towards idle rather than fixed to it: every recording here starts
    with pre-roll from a still machine, but a server restarted mid-wash must
    not be forced to believe the washer is off -- it would take several minutes
    of evidence to climb back out, and the screen would be wrong for all of it.
    """
    p = np.full(len(states), (1.0 - confidence) / max(len(states) - 1, 1))
    p[list(states).index(start)] = confidence
    return p / p.sum()


def _log_emissions(proba: np.ndarray, power: float) -> np.ndarray:
    """Class probabilities -> tempered log emissions, columns already aligned
    to STATES by the caller."""
    P = np.clip(np.asarray(proba, dtype=float), EPS, None)
    P = P / P.sum(axis=1, keepdims=True)
    return power * np.log(P)


def viterbi(proba: np.ndarray, transitions: Optional[np.ndarray] = None,
            initial: Optional[np.ndarray] = None,
            power: float = EMISSION_POWER) -> np.ndarray:
    """Most likely state path over a whole recording. Returns state INDICES.

    All arithmetic in logs. With ~1,300 windows, a product of probabilities
    underflows to exactly zero within about forty windows, and then every path
    ties at zero and the answer is whichever index argmax happens to return --
    a bug that looks like a model that "just isn't very good".
    """
    P = np.atleast_2d(np.asarray(proba, dtype=float))
    n_obs, n_states = P.shape

    A = np.log(np.clip(transitions if transitions is not None else build_transitions(), EPS, None))
    pi = np.log(np.clip(initial if initial is not None else initial_distribution(), EPS, None))
    E = _log_emissions(P, power)

    delta = pi + E[0]
    psi = np.zeros((n_obs, n_states), dtype=int)

    for t in range(1, n_obs):
        # (n_states, n_states): score of arriving at j from every i.
        scores = delta[:, None] + A
        psi[t] = np.argmax(scores, axis=0)
        delta = scores[psi[t], np.arange(n_states)] + E[t]

    path = np.zeros(n_obs, dtype=int)
    path[-1] = int(np.argmax(delta))
    for t in range(n_obs - 2, -1, -1):
        path[t] = psi[t + 1][path[t + 1]]
    return path


def smooth(proba: np.ndarray, states: Sequence[str] = STATES, **kw) -> np.ndarray:
    """viterbi(), but returning phase names."""
    return np.array([states[i] for i in viterbi(proba, **kw)])


def done_time(path_labels: Sequence[str], times: Sequence[float]) -> Optional[float]:
    """When the wash finished: the first return to idle after any spin.

    This is the output the whole project exists to produce, and it is defined
    as a transition rather than as a sound -- a finished washer and an empty
    laundry room are the same silence. None means the recording never got as
    far as a spin, or has not finished yet.
    """
    seen_spin = False
    for label, t in zip(path_labels, times):
        if label == "spin":
            seen_spin = True
        elif seen_spin and label == "idle":
            return float(t)
    return None


class Filter:
    """Online forward filtering -- one window at a time, no lookahead.

    What server/main.py holds per machine. update() returns the posterior over
    phases given everything seen so far; the screen shows its argmax.

    Normalized after every step, for the same underflow reason as viterbi, and
    because a posterior that has decayed to 1e-300 stops responding to new
    evidence at all -- the screen would freeze on whatever it believed an hour
    ago and look, from outside, exactly like a working system.
    """

    def __init__(self, states: Sequence[str] = STATES,
                 transitions: Optional[np.ndarray] = None,
                 initial: Optional[np.ndarray] = None,
                 power: float = EMISSION_POWER,
                 min_dwell: int = MIN_DWELL) -> None:
        self.states = tuple(states)
        self.A = transitions if transitions is not None else build_transitions(states)
        self.alpha = initial if initial is not None else initial_distribution(states)
        self.power = power
        self.min_dwell = min_dwell
        self.seen_spin = False
        self._reported = self.states[int(np.argmax(self.alpha))]
        self.disagree = 0
        self._idle_run = 0
        self._finished = False

    def update(self, proba_row: np.ndarray) -> np.ndarray:
        """Advance one window. `proba_row` must already be in STATES order."""
        raw = np.clip(np.asarray(proba_row, dtype=float), EPS, None)
        raw = raw / raw.sum()

        # The classifier's unsmoothed opinion of THIS window, kept for
        # done-detection only. It flips the moment the drum stops.
        if self.states[int(np.argmax(raw))] == "idle":
            self._idle_run += 1
        else:
            self._idle_run = 0

        e = raw ** self.power

        self.alpha = (self.alpha @ self.A) * e
        total = self.alpha.sum()
        # A total of zero means every state was ruled impossible, which can only
        # happen through arithmetic, not through evidence. Restarting from the
        # prior is wrong but recoverable; leaving NaN in there is neither.
        self.alpha = self.alpha / total if total > EPS else initial_distribution(self.states)

        # Output hysteresis, not a frozen transition matrix. Freezing the
        # matrix does NOT hold the state: the emission multiply redistributes
        # mass between states on its own, so a persistently wrong classifier
        # walks the belief across anyway -- measured, not assumed, after the
        # first attempt at this failed a 20-window test.
        #
        # So the BELIEF updates freely and the REPORTED phase lags it: the
        # argmax has to disagree for min_dwell consecutive windows before the
        # reported phase moves. A washer's shortest phase is ~106 windows, so
        # 30 (~78 s) cannot hide a real transition, and it is long enough that
        # no momentary confusion is visible on the screen at all.
        top = self.states[int(np.argmax(self.alpha))]
        if top == self._reported:
            self.disagree = 0
        else:
            self.disagree += 1
            if self.disagree >= self.min_dwell:
                self._reported, self.disagree = top, 0

        if self._reported == "spin":
            self.seen_spin = True
        if self.seen_spin and self._idle_run >= DONE_QUIET:
            self._finished = True
        return self.alpha

    def correct(self, phase: str, confidence: float = 0.97) -> np.ndarray:
        """A human says the machine is in `phase`. Believe them.

        A person reading the panel is a direct observation and this is an
        inference, so the observation wins outright rather than being blended
        in as one more window of evidence -- at EMISSION_POWER a single window
        moves the belief by almost nothing, and a correction that takes four
        minutes to take effect is not a correction, it is a suggestion.

        The dwell counter restarts, so the correction is also protected from
        being immediately undone by the same confused emissions that caused the
        wrong answer in the first place. And anything downstream of a spin is
        forgotten: if a human says we are in wash, this cycle has not finished,
        whatever the filter thought a moment ago.
        """
        if phase not in self.states:
            return self.alpha
        p = np.full(len(self.states), (1.0 - confidence) / (len(self.states) - 1))
        p[self.states.index(phase)] = confidence
        self.alpha = p
        self._reported, self.disagree = phase, 0
        self.seen_spin = phase == "spin"
        self._idle_run, self._finished = 0, False
        return self.alpha

    @property
    def phase(self) -> str:
        """What the screen shows: the belief, held steady by hysteresis."""
        return self._reported

    @property
    def believed(self) -> str:
        """The raw argmax, before hysteresis. For debugging, not for display."""
        return self.states[int(np.argmax(self.alpha))]

    @property
    def confidence(self) -> float:
        return float(np.max(self.alpha))

    @property
    def finished(self) -> bool:
        """True once the machine has been through a spin and has then been
        quiet for DONE_QUIET consecutive windows. Latched: a wash finishes
        once, and a door slam an hour later must not un-finish it."""
        return self._finished
