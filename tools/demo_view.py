"""A terminal view of what the server currently believes. For screen recording.

Polls /status and tails the prediction log, and prints one line whenever the
phase changes plus a periodic heartbeat with the full posterior. Append-only
rather than a redrawing dashboard: a scrolling log reads better on video, and
it survives a terminal resize mid-recording, which a cursor-addressed screen
does not.

Run:  venv/bin/python -u tools/demo_view.py
      (-u matters: without it Python buffers stdout and the video shows
       nothing for a minute, then everything at once)
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS = REPO_ROOT / "analysis" / "predictions"

BOLD, DIM, OFF = "\033[1m", "\033[2m", "\033[0m"
COLOR = {"idle": "\033[38;5;245m", "fill": "\033[38;5;38m", "wash": "\033[38;5;42m",
         "rinse": "\033[38;5;214m", "spin": "\033[38;5;203m", "done": "\033[38;5;250m"}


def newest_prediction() -> dict | None:
    """Last line of the most recently written prediction log, or None."""
    if not PREDICTIONS.is_dir():
        return None
    files = sorted(PREDICTIONS.glob("*.pred.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    try:
        lines = files[-1].read_text().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue  # the server may be mid-write on the final line
    return None


def bar(p: float, width: int = 22) -> str:
    filled = int(round(p * width))
    return "█" * filled + "·" * (width - filled)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--every", type=float, default=2.0, help="poll seconds")
    ap.add_argument("--heartbeat", type=float, default=30.0, help="posterior every N seconds")
    args = ap.parse_args()

    print(f"{BOLD}LaundryWatch — what the model thinks, live{OFF}")
    print(f"{DIM}polling {args.url}  ·  nothing here is a human tap{OFF}\n")

    last_phase, last_beat, started = None, 0.0, {}
    while True:
        try:
            with urllib.request.urlopen(f"{args.url}/status", timeout=5) as r:
                st = json.load(r)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"{DIM}{time.strftime('%H:%M:%S')}  server unreachable ({exc}){OFF}")
            time.sleep(args.every)
            continue

        now = time.time()
        phase = st.get("phase") or ("done" if st["mode"] == "done" else None)
        src = "model" if st.get("predicted") else "as marked"

        if phase != last_phase and phase is not None:
            c = COLOR.get(phase, "")
            held = ""
            if last_phase in started:
                held = f"  {DIM}(after {(now - started[last_phase]) / 60:.1f} min of {last_phase}){OFF}"
            print(f"{time.strftime('%H:%M:%S')}  {c}{BOLD}{phase.upper():<6}{OFF}  {DIM}{src}{OFF}{held}")
            started[phase] = now
            last_phase = phase
            last_beat = now

        if now - last_beat >= args.heartbeat:
            row = newest_prediction()
            if row and row.get("p"):
                parts = "   ".join(
                    f"{COLOR.get(k,'')}{k:<5}{OFF} {bar(v, 12)} {v:.2f}"
                    for k, v in row["p"].items()
                )
                print(f"{DIM}{time.strftime('%H:%M:%S')}{OFF}  {parts}")
            elif st.get("sensor_ok"):
                print(f"{DIM}{time.strftime('%H:%M:%S')}  {last_phase or st['mode']}, steady{OFF}")
            last_beat = now

        time.sleep(args.every)


if __name__ == "__main__":
    main()
