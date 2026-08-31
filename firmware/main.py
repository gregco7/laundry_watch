# Lives on Node/ESP32
# Runs on every boot, after boot.py has brought up WiFi and started WebREPL.
#
# One job: sample the accelerometer at ~100 Hz and POST fixed-size windows to
# the laptop. It has no concept of a cycle, a phase, or a recording -- the
# server decides what to keep. That split is deliberate: a node holding no
# state cannot desync, and a "start recording" command lost to a WiFi blip
# cannot cost you a 50-minute wash.
#
# Reflashing means walking to the washer, so this file is written to survive a
# day unattended: every I2C read and every network call is guarded, and
# nothing is allowed to raise out of the loop.

import gc
import os
import time
from array import array

import network
import urequests
from machine import I2C, Pin
from micropython import const

from mpu6050 import MPU6050
from secrets import SERVER_URL

# Drop an empty file named PAUSE on the board and this script exits at boot,
# leaving the REPL free. It survives reboots, which is the point: push a broken
# main.py and the board will otherwise re-run it on every reset. It cannot
# rescue you on its own -- creating the file needs a REPL, so interrupt first
# with tools/wrepl.py.
if "PAUSE" in os.listdir():
    raise SystemExit("main: PAUSE present, not starting")

NODE      = "washer-01"
N         = const(256)    # samples per window -- must match pipeline/features.py
PERIOD_US = const(10000)  # 10 ms -> nominally 100 Hz. The REAL rate is measured below.

# boot.py owns the radio: association, retries, power-save. All this does is
# refuse to start if that didn't work, rather than sampling into the void for
# a day. Trade-off: a router still rebooting when boot.py gives up leaves the
# node down until it's power-cycled.
if not network.WLAN(network.STA_IF).isconnected():
    raise SystemExit("main: no wifi -- see boot.py output")

i2c    = I2C(0, scl=Pin(22), sda=Pin(21), freq=400000)
sensor = MPU6050(i2c)

# Allocated ONCE. Three arrays per window at 100 Hz would churn the heap, and
# the resulting collection pause lands as a hole in the middle of a window.
xs = array("h", [0] * N)
ys = array("h", [0] * N)
zs = array("h", [0] * N)

seq  = 0
down = False   # only used to log the link dropping and coming back, once each

print("main: %s sampling %d-sample windows -> %s" % (NODE, N, SERVER_URL))

while True:
    # Collect here, at the one point in the loop where a pause is free. Left to
    # its own devices the GC would fire mid-window and cost us samples.
    gc.collect()

    # ---- fill one window ------------------------------------------------
    late = 0
    t0 = next_t = time.ticks_us()

    try:
        for i in range(N):
            next_t = time.ticks_add(next_t, PERIOD_US)
            xs[i], ys[i], zs[i] = sensor.read_raw()

            slack = time.ticks_diff(next_t, time.ticks_us())
            if slack > 0:
                time.sleep_us(slack)
            else:
                late += 1     # the loop could not keep up with PERIOD_US
    except OSError as e:
        # An I2C NAK -- electrical noise, a jostled Dupont wire. Discard the
        # partial window rather than shipping a half-filled buffer whose tail
        # is last window's data.
        print("main: sensor read failed: %s" % e)
        time.sleep_ms(500)
        continue

    elapsed = time.ticks_diff(time.ticks_us(), t0)
    hz = round(N * 1000000 / elapsed, 2)

    # ---- ship it --------------------------------------------------------
    # No timestamp: the ESP32 has no RTC, so the server stamps arrival time and
    # owns the clock. `seq` rides along only so the server can tell a dropped
    # window from a slow one.
    seq += 1
    body = {"node": NODE, "mode": "raw", "seq": seq,
            "hz": hz, "n": N, "late": late,
            "x": list(xs), "y": list(ys), "z": list(zs)}

    try:
        r = urequests.post(SERVER_URL, json=body)
        status = r.status_code
        r.close()    # urequests leaks the socket if you skip this, and a few
                     # hundred leaked sockets is a node that stops talking
    except OSError as e:
        # Expected, routinely: laptop asleep, server not running
        # Drop the window and keep sampling. Logged once per outage rather than
        # once per window -- an overnight gap is ~1400 windows.
        if not down:
            print("main: post failed at seq %d: %s" % (seq, e))
            down = True
    else:
        # A 404 or a 500 is a *successful* HTTP transaction, so nothing above
        # raises. Without this check a typo'd path in SERVER_URL is silent on
        # both ends and looks exactly like the node being dead.
        if status != 200:
            if not down:
                print("main: server said HTTP %d at seq %d" % (status, seq))
                down = True
        elif down:
            print("main: server back at seq %d" % seq)
            down = False
