"""The dumb baseline. Thresholds and hysteresis, no learning anywhere.

Built BEFORE the model and kept afterwards, because "the classifier gets 84%"
is not a fact until something says what a stopwatch and four if-statements get.
Most of the value in this project is in detecting the END of a cycle, and a
rule that watches for loud-then-quiet is a genuinely strong competitor at that
one job. If the model cannot beat this, the model is not earning its place.

It reads z-scored features (pipeline.calibrate), not raw ones, so the numbers
below are in units of "idle-widths above this machine's own noise floor" and
mean the same thing on any washer. Measured on the three recorded cycles:

    idle  ~0      fill ~22      rinse ~44      wash ~54      spin ~115

Two mechanisms, both of which the HMM later replaces with something principled:

HYSTERESIS. A threshold crossed by a single window is noise; agitation dips
below its own level constantly as the drum reverses. Nothing changes phase
until CONFIRM windows in a row agree, which costs ~8 s of latency at every
boundary and removes essentially all of the flicker.

FORWARD ONLY. A washer goes idle -> fill -> wash -> rinse -> spin -> done and
never backwards, so this machine cannot either. That single constraint is what
lets it survive the awkward fact that the loudest event in the whole cycle --
a drain spin lasting several minutes -- happens in the MIDDLE, inside the span
the panel calls Rinse. A level-only classifier calls that spin and then has to
explain the quiet refill that follows it. This one calls it rinse because rinse
is what comes next, which is the same argument the HMM makes with a transition
matrix instead of an if-statement.

What it cannot do, and neither can any per-window rule: separate wash from
rinse agitation. They are the same sound at the same level. It guesses by
position, and so does everything else in this repo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from pipeline.features import FEATURE_NAMES

# Level boundaries on z-scored log_rms. Set from the medians above, roughly
# halfway between neighbouring phases in log space -- not fitted, on purpose.
# A fitted threshold is a one-parameter model, and then this stops being a
# baseline and starts being a competitor with an unfair advantage.
QUIET_BELOW = 8.0     # idle
FILL_BELOW = 33.0     # filling: audible, not agitating
AGITATE_BELOW = 85.0  # wash or rinse agitation
# ...and anything above AGITATE_BELOW is a spin, drain or final.

# Windows that must agree before a phase changes. ~2.6 s each, so 3 is ~8 s.
CONFIRM = 3

# How long the machine must be NOT loud before another loud stretch counts as
# a new burst rather than a continuation of the last one. A drain spin is not
# smoothly loud -- it surges and dips -- and a single quiet window inside it is
# not the end of it. Measured across the three recorded cycles: the longest dip
# inside a drain spin is 16 windows, and the real quiet gap between the drain
# spin and the final spin is 236-276 windows. 40 sits an order of magnitude
# clear of the first and six times inside the second.
REARM_QUIET = 40

# Quiet windows required to call a cycle finished. Longer than CONFIRM because
# this is the one output anyone acts on -- a false "done" sends someone
# downstairs to an empty machine, while a late one costs a minute.
DONE_QUIET = 5

PHASES = ("idle", "fill", "wash", "rinse", "spin", "done")


def level(z_row: np.ndarray, rms_col: int) -> str:
    """Which band of loudness this window sits in."""
    v = z_row[rms_col]
    if v < QUIET_BELOW:
        return "quiet"
    if v < FILL_BELOW:
        return "fill"
    if v < AGITATE_BELOW:
        return "agitate"
    return "loud"


@dataclass
class RuleState:
    """Where the machine is, and how long it has been trying to leave."""

    phase: str = "idle"
    since: Optional[float] = None
    # The candidate next phase and how many consecutive windows have voted for
    # it. Reset the moment one window disagrees -- that IS the hysteresis.
    pending: Optional[str] = None
    votes: int = 0
    pending_since: Optional[float] = None
    # Consecutive not-loud windows. A new loud burst only counts once this has
    # reached REARM_QUIET; see that constant. Without any such rule the drain
    # spin walks the machine wash -> rinse -> spin inside a single burst,
    # because the burst is still loud three windows after it caused the first
    # transition -- the first version of this file called every cycle finished
    # twenty minutes early. With a rule of only ONE quiet window, cycle 1 still
    # failed by 19 minutes, because a drain spin dips below the threshold on
    # its own several times a minute.
    quiet_run: int = REARM_QUIET
    # Sticky: earned by a long enough quiet stretch, spent by one loud
    # transition. It has to be sticky, because the loud burst that the arming
    # exists to permit resets quiet_run on its own first window -- checking
    # `quiet_run >= REARM_QUIET` at vote time instead means the burst disarms
    # itself before it can be confirmed, and nothing ever transitions at all.
    armed: bool = True
    # Set once the cycle has been seen to finish, so a later noise (someone
    # opening the door, the next load going in) cannot un-finish it.
    done_at: Optional[float] = None
    history: list[tuple[float, str]] = field(default_factory=list)


def _next_phase(current: str, lvl: str, armed: bool) -> Optional[str]:
    """The forward-only transition table, as a function.

    Returns None for "stay where you are". Every rule here is a sentence about
    a washer, and the order of the checks matters: a loud burst means different
    things depending on what has already happened.
    """
    if current == "idle":
        if lvl == "fill":
            return "fill"
        if lvl == "agitate":
            return "wash"      # a late start, or fill was too quiet to see
        if lvl == "loud":
            return "spin"      # recording started mid-spin
    elif current == "fill":
        if lvl == "agitate":
            return "wash"
        if lvl == "loud":
            return "wash"      # some washers agitate hard while still filling
    elif current == "wash":
        # The drain spin. Loud, several minutes, and the panel still says Rinse.
        if lvl == "loud":
            return "rinse"
    elif current == "rinse":
        # The SECOND loud stretch is the real one. Everything between them --
        # the quiet refill and the rinse agitation -- stays rinse by position.
        # `armed` is what makes "second" mean second: the burst that carried
        # wash -> rinse must end before another one can carry rinse -> spin.
        if lvl == "loud" and armed:
            return "spin"
    return None


def step(state: RuleState, z_row: np.ndarray, t: float, rms_col: int) -> RuleState:
    """Advance the state machine by one window. Mutates and returns `state`."""
    if state.since is None:
        state.since = t

    lvl = level(z_row, rms_col)

    # The difference between "a new loud burst" and "still the same loud burst"
    # is only visible in how long the quiet between them lasted; no single
    # window carries it.
    if lvl != "loud":
        state.quiet_run += 1
        if state.quiet_run >= REARM_QUIET:
            state.armed = True
    else:
        state.quiet_run = 0

    # Finishing outranks everything: quiet for long enough after a spin is the
    # end of the wash, and it is the only conclusion this file exists to reach.
    if state.phase == "spin" and state.done_at is None:
        if lvl == "quiet":
            state.pending = "done"
            state.votes += 1
            if state.votes >= DONE_QUIET:
                state.phase, state.since, state.done_at = "done", t, t
                state.history.append((t, "done"))
                state.votes, state.pending = 0, None
            return state
        state.votes, state.pending = 0, None
        return state

    if state.phase == "done":
        return state  # a finished cycle stays finished

    candidate = _next_phase(state.phase, lvl, state.armed)
    if candidate is None or candidate == state.phase:
        state.votes, state.pending, state.pending_since = 0, None, None
        return state

    if candidate == state.pending:
        state.votes += 1
    else:
        state.pending, state.votes, state.pending_since = candidate, 1, t

    if state.votes >= CONFIRM:
        # Dated to the FIRST window that voted, not the one that carried the
        # motion. Otherwise every boundary is reported ~8 s late and the timing
        # error this file exists to measure has a constant bias baked into it.
        started = state.pending_since if state.pending_since is not None else t
        state.phase, state.since = candidate, started
        state.history.append((started, candidate))
        state.pending, state.votes, state.pending_since = None, 0, None
        if lvl == "loud":
            state.armed = False

    return state


def run(Z: np.ndarray, times: Sequence[float],
        feature_names: Sequence[str] = FEATURE_NAMES) -> tuple[np.ndarray, Optional[float]]:
    """Whole recording -> (phase per window, time the cycle was called done).

    Z is z-scored. `times` are window centres, matching features.extract_sequence
    and analysis/label.py -- the returned done time is only comparable to a
    label file's done_at if all three agree on what a window's time is.
    """
    rms_col = list(feature_names).index("log_rms")
    state = RuleState()
    out: list[str] = []
    for row, t in zip(np.atleast_2d(Z), times):
        step(state, row, float(t), rms_col)
        out.append(state.phase)
    return np.array(out), state.done_at
