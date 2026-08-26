# Runs on every boot, before main.py.
#
# Brings up WiFi with retries, then starts WebREPL. Order matters, and so does
# the retry: this node lives on a laundry machine, so "failed to associate once
# at boot" must not mean "unreachable until someone brings a USB cable".
#
# NOTE: webrepl_setup appends `import webrepl; webrepl.start()` to THIS file.
# Overwriting boot.py silently removes it and WebREPL stops starting.


import network
import time

from secrets import WIFI_SSID, WIFI_PASSWORD

ATTEMPTS = 4          # rounds of connect
WAIT_PER_ATTEMPT = 15 # seconds each -- worst case 60s, still bounded

wlan = network.WLAN(network.STA_IF)


def _connect():
    for attempt in range(ATTEMPTS):
        if wlan.isconnected():
            return True
        # Full radio reset between rounds. connect() on a half-open link raises
        # OSError: Wifi Internal State Error and the attempt is wasted.
        try:
            wlan.disconnect()
        except OSError:
            pass
        wlan.active(False)
        time.sleep(0.5)
        wlan.active(True)
        try:
            wlan.connect(WIFI_SSID, WIFI_PASSWORD)
        except OSError:
            continue
        for _ in range(WAIT_PER_ATTEMPT * 10):
            if wlan.isconnected():
                return True
            time.sleep(0.1)
        print("wifi: attempt", attempt + 1, "failed, status", wlan.status())
    return False


ok = _connect()

# Default power-save parks the radio between beacons: ~1 s ping latency, which
# would show up as jitter on every reading POSTed from this node.
try:
    wlan.config(pm=network.WLAN.PM_NONE)
except (AttributeError, ValueError, OSError):
    pass

print("wifi:", wlan.ifconfig()[0] if ok else "FAILED")

# Started unconditionally and bound to 0.0.0.0, so it also becomes reachable if
# the link comes up later than boot.
import webrepl
webrepl.start()
