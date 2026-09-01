#!/usr/bin/env python3
"""Live amplitude strip, in the terminal.

The mounting check, before the dashboard has one. Run it, start a wash, and
watch: SPIN MUST BE OBVIOUSLY LOUDER THAN WASH. If it is not, the sensor is
picking up a resonating panel rather than the drum, and every cycle you record
through it is worthless -- while looking completely normal on a spectrogram.

    python3 tools/strip.py                  # localhost
    python3 tools/strip.py --url http://10.0.0.17:8000

Stdlib only, so it runs anywhere the repo does.
"""

import argparse
import json
import time
import urllib.error
import urllib.request

BARS = " ▁▂▃▄▅▆▇█"


def bar(value: float, ceiling: float, width: int = 46) -> str:
    """One row of a bar chart, scaled linearly against the loudest window seen.

    Linear on purpose. A log scale would be the obvious choice for a range this
    wide, and it is exactly wrong here: it renders a quiet 85 at two-thirds the
    length of a 3000 and hides the one distinction this tool exists to show.
    Spin should dwarf everything else on the screen. If it doesn't, that is the
    finding.
    """
    if value <= 0:
        return ""
    frac = value / max(ceiling, 1.0)
    return "\u2588" * max(1, min(width, round(frac * width)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--every", type=float, default=2.0, help="poll seconds")
    args = ap.parse_args()

    print(f"watching {args.url}/live  --  ctrl-c to stop\n")
    print(f"{'time':>8}  {'seq':>6} {'hz':>6} {'late':>4}  {'z-rms':>8}  amplitude")
    print("-" * 86)

    seen = -1
    ceiling = 500.0  # grows to fit; a spin will blow past this immediately

    while True:
        try:
            with urllib.request.urlopen(f"{args.url}/live", timeout=5) as r:
                live = json.load(r)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            print(f"  server unreachable: {e}")
            time.sleep(args.every)
            continue

        node = next(iter(live.get("nodes", {}).values()), None)
        if node is None:
            print("  no node has ever posted -- is main.py running on the ESP32?")
            time.sleep(args.every)
            continue
        if node.get("silent_for", 0) > 20:
            print(f"  node silent for {node['silent_for']}s")

        for w in live.get("series", []):
            if w["seq"] <= seen:
                continue
            seen = w["seq"]
            z = w["z"]
            ceiling = max(ceiling, z)
            stamp = time.strftime("%H:%M:%S", time.localtime(w["t"]))
            late = w["late"]
            flag = "!" if late > 8 else " "
            print(f"{stamp:>8}  {w['seq']:>6} {w['hz']:>6.1f} {late:>3}{flag}  {z:>8.1f}  {bar(z, ceiling)}")

        time.sleep(args.every)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped")
