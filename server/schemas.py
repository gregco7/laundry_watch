"""Pydantic schemas -- the shapes that cross the HTTP boundary.

Two very different clients POST here, and the split matters:

The ESP32 sends ReadingIn, ~1400 times an hour, forever. Its shape is dictated
by firmware/main.py, which is already running on hardware -- if these two ever
disagree, every window 422s and the node logs one line and keeps sampling into
nothing. Change the firmware first, then this file.

Your phone sends CycleStart and MarkIn, a handful of times per wash.

Note what is NOT in ReadingIn: a timestamp. The ESP32 has no RTC, so the server
stamps arrival time and owns the clock. `seq` rides along only so a dropped
window can be told from a slow one.
"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# The label vocabulary, enforced at the API edge and nowhere else. models.Mark
# stores a plain string, so widening or collapsing this tuple costs one line
# here rather than a schema change -- which matters, because whether `fill` is
# separable from `idle` is something the data decides, not you.
PHASES = ("idle", "fill", "agitate", "spin", "done")

# Guards against a malformed body allocating something enormous. The firmware
# sends 256; anything near this ceiling is a bug, not a longer window.
MAX_SAMPLES = 4096


class ReadingIn(BaseModel):
    """One window of raw accelerometer samples, exactly as firmware/main.py
    builds it. Field names are the node's, not ours."""

    # extra="forbid" so a field renamed on the node fails loudly on the first
    # POST rather than being silently dropped and noticed a wash later. The
    # trade-off is real: adding a field to the firmware breaks ingest until
    # this file catches up. That is the intended order of operations.
    model_config = ConfigDict(extra="forbid")

    node: str = Field(min_length=1, max_length=64)
    mode: Literal["raw"] = "raw"
    seq: int = Field(ge=0)

    # The measured rate, not the nominal one -- the sampling loop drifts and
    # main.py computes what it actually achieved. features.py needs the real
    # number or every frequency it reports is wrong by that ratio.
    hz: float = Field(gt=0, le=10_000)

    n: int = Field(gt=0, le=MAX_SAMPLES)
    late: int = Field(ge=0)  # samples the loop couldn't deliver on time

    x: list[int]
    y: list[int]
    z: list[int]

    @model_validator(mode="after")
    def _axes_match_n(self) -> "ReadingIn":
        """A short axis means a truncated body -- a partial window whose tail
        would otherwise be read as real signal."""
        for axis in ("x", "y", "z"):
            got = len(getattr(self, axis))
            if got != self.n:
                raise ValueError(f"axis {axis!r} has {got} samples but n={self.n}")
        return self


class ReadingAck(BaseModel):
    """What the node gets back. It only checks the status code, but a body
    that says whether the window was written is what makes /docs usable as a
    debugging tool from a phone."""

    ok: bool = True
    recorded: bool = False  # True only when an open cycle sent it to disk
    cycle_id: Optional[int] = None


class CycleStart(BaseModel):
    """Sent when you tap Start. Everything but the machine is optional so a
    cycle can always be opened in a hurry."""

    model_config = ConfigDict(extra="forbid")

    machine_id: str = Field(default="washer-01", min_length=1, max_length=64)
    setting: Optional[str] = Field(default=None, max_length=64)
    load_size: Optional[str] = Field(default=None, max_length=32)
    notes: Optional[str] = None


class MarkIn(BaseModel):
    """Sent when you tap a phase button."""

    model_config = ConfigDict(extra="forbid")

    phase: str

    # Optional so the phone doesn't have to have a correct clock -- omit it and
    # the server stamps now, keeping marks on the same clock as the windows
    # they will be joined against. Present only for correcting a late tap.
    t: Optional[float] = None

    @field_validator("phase")
    @classmethod
    def _known_phase(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in PHASES:
            raise ValueError(f"unknown phase {v!r}; expected one of {', '.join(PHASES)}")
        return v


class MarkOut(BaseModel):
    """A mark on its way back out. from_attributes lets FastAPI build this
    straight from a models.Mark instead of a dict."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    cycle_id: int
    t: float
    phase: str


class CycleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    machine_id: str
    started_at: float
    ended_at: Optional[float] = None
    recording: Optional[str] = None
    setting: Optional[str] = None
    load_size: Optional[str] = None
    notes: Optional[str] = None

    # Serializing this triggers a lazy load of Cycle.marks. That is fine inside
    # an endpoint, where get_session() keeps the session open until after the
    # response is built -- but it is why you cannot hand a detached Cycle to
    # this model from outside a request.
    marks: list[MarkOut] = Field(default_factory=list)


class MachineOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    label: str
    installed_at: float
    calibrated: bool = False
    open_cycle_id: Optional[int] = None

    @classmethod
    def from_machine(cls, machine: Any, open_cycle_id: Optional[int] = None) -> "MachineOut":
        """Calibration is a JSON blob that can be large and is never useful to
        the dashboard -- it only needs to know whether one exists."""
        return cls(
            id=machine.id,
            label=machine.label,
            installed_at=machine.installed_at,
            calibrated=machine.calibration is not None,
            open_cycle_id=open_cycle_id,
        )
