#!/usr/bin/env python3
"""Drive the mounted ESP32 node over WiFi. Three commands: `run`, `put`, `interrupt`.

WHY THIS EXISTS
The DevKit v1 has a single micro-USB port, so laptop power and wall power can
never overlap -- plugging in a cable reboots the board. Once the node is bonded
to the washer, USB is gone as a working channel and everything has to happen
over the network. `mpremote` speaks only serial, so it cannot help here; it is
now only for reflashing MicroPython.

WHAT IT SPEAKS
Three layers, stacked:

1. WebSocket (RFC 6455). A plain HTTP GET with `Upgrade: websocket` and a
   `Sec-WebSocket-Key`, answered with `101 Switching Protocols`. After that the
   stream is length-prefixed frames. Client-to-server frames MUST be masked
   with a random 4-byte key; server-to-client frames must not be.

2. MicroPython WebREPL, which listens on port 8266 and puts an ordinary REPL
   behind that socket. It prompts `Password:` first -- the board's only access
   control, since this is plain ws:// on the LAN.

3. The REPL's *raw* mode, entered with Ctrl-A (0x01) and executed with Ctrl-D
   (0x04). Raw mode has no echo and no `...` continuation prompts, so a
   multi-line file pastes cleanly instead of being mangled. Output comes back
   as `OK<stdout>\\x04<stderr>\\x04>`.

   Ctrl-C (0x03) is the same idea in reverse: the byte raises KeyboardInterrupt
   in whatever is currently running. `interrupt` sends only that and skips raw
   mode entirely -- a board busy inside main.py's loop never answers the
   Ctrl-A handshake, so asking for it first would just time out.

`put` sends files as base64 chunks through raw mode rather than using WebREPL's
binary transfer protocol -- it reuses the machinery already here, and chunking
means board RAM never has to hold the whole file at once.

Address and password come from the repo's `.env` (ESP32_IP,
ESP32_WEBREPL_PASSWORD) so no secret ever lands in shell history.

USAGE
    python3 tools/wrepl.py run firmware/step6.py
    python3 tools/wrepl.py run firmware/step6.py --timeout 90
    python3 tools/wrepl.py put firmware/main.py
    python3 tools/wrepl.py put firmware/sensor.py lib/sensor.py
    python3 tools/wrepl.py put firmware/boot.py --force
    python3 tools/wrepl.py interrupt
"""

import argparse
import base64
import os
import socket
import struct
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"
WEBREPL_PORT = 8266

# Base64 expands 3 bytes to 4, so 768 raw bytes is 1024 encoded -- comfortably
# under the raw REPL's appetite and small enough that the board never holds
# more than a KB of transfer buffer at once.
CHUNK_RAW = 768

def load_env():
    if not ENV_PATH.exists():
        sys.exit("no .env at %s -- copy .env.example and fill it in" % ENV_PATH)
    vals = {}
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        vals[k.strip()] = v.strip().strip('"').strip("'")
    return vals


def config():
    env = load_env()
    host = env.get("ESP32_IP")
    pw = env.get("ESP32_WEBREPL_PASSWORD")
    # Fail loudly rather than defaulting. A hardcoded fallback address would
    # silently probe whatever sits at that IP on someone else's network.
    if not host:
        sys.exit("ESP32_IP not set in .env")
    if not pw:
        sys.exit("ESP32_WEBREPL_PASSWORD not set in .env")
    return host, pw


def mask_frame(payload, opcode=0x01):
    """Build one client frame. RFC 6455 5.3: client frames must be masked."""
    data = payload.encode() if isinstance(payload, str) else payload
    header = bytes([0x80 | opcode])
    n = len(data)
    if n < 126:
        header += bytes([0x80 | n])
    elif n < 65536:
        header += bytes([0x80 | 126]) + struct.pack(">H", n)
    else:
        header += bytes([0x80 | 127]) + struct.pack(">Q", n)
    key = os.urandom(4)
    return header + key + bytes(b ^ key[i % 4] for i, b in enumerate(data))


class WebREPL:
    def __init__(self, host, password, port=WEBREPL_PORT, timeout=10, raw=True):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(timeout)
        self.buf = b""
        self.raw = raw
        self._handshake(host, port)
        self.read_until(b"Password:")
        self.send(password + "\r\n")
        banner = self.read_until(b">>>")
        if b"connected" not in banner:
            raise RuntimeError("login failed: %r" % banner[-200:])
        # webrepl prints "WebREPL connected\n>>> " itself on auth, so the banner
        # arrives even when a script is running and the REPL is not at a prompt.
        # Raw mode is different: it needs the interpreter to be listening, so a
        # busy board would silently time out here. `interrupt` passes raw=False.
        if raw:
            self.send("\x01")                  # enter raw REPL
            self.read_until(b"raw REPL")

    def _handshake(self, host, port):
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((
            f"GET / HTTP/1.1\r\nHost: {host}:{port}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
            "Sec-WebSocket-Protocol: chat\r\n\r\n"
        ).encode())
        # Drain the HTTP headers off the stream before frame parsing starts.
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = self.sock.recv(1)
            if not chunk:
                raise RuntimeError("connection closed during handshake")
            resp += chunk
        if b"101" not in resp.split(b"\r\n")[0]:
            raise RuntimeError("handshake rejected: %r" % resp[:200])

    def send(self, text):
        self.sock.sendall(mask_frame(text))

    def _take(self, n):
        while len(self.buf) < n:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise RuntimeError("connection closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def recv(self):
        _, b1 = self._take(2)
        n = b1 & 0x7F
        if n == 126:
            n = struct.unpack(">H", self._take(2))[0]
        elif n == 127:
            n = struct.unpack(">Q", self._take(8))[0]
        if b1 & 0x80:                          # server shouldn't mask, but be safe
            key = self._take(4)
            return bytes(c ^ key[i % 4] for i, c in enumerate(self._take(n)))
        return self._take(n)

    def read_until(self, marker, timeout=15):
        acc = b""
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.sock.settimeout(max(0.2, deadline - time.time()))
            try:
                acc += self.recv()
            except socket.timeout:
                continue
            if marker in acc:
                return acc
        return acc

    def execute(self, source, timeout=40):
        """Run source in raw REPL. Returns (stdout, stderr).

        Globals persist between calls on one connection, which is what lets
        `put` keep a file handle open across many chunk uploads.
        """
        self.send(source + "\x04")
        out = self.read_until(b"\x04>", timeout=timeout)
        body = out.split(b"OK", 1)[-1]
        parts = body.split(b"\x04")
        stdout = parts[0].decode(errors="replace")
        stderr = parts[1].decode(errors="replace") if len(parts) > 1 else ""
        return stdout, stderr

    def close(self):
        try:
            if self.raw:
                self.send("\x02")              # back to the friendly REPL
            self.sock.close()
        except OSError:
            pass


def cmd_run(ws, args):
    src = Path(args.file).read_text()
    stdout, stderr = ws.execute(src, timeout=args.timeout)
    print(stdout, end="")
    if stderr.strip():
        print("--- traceback on device ---", file=sys.stderr)
        print(stderr, end="", file=sys.stderr)
        return 1
    return 0


def cmd_interrupt(ws, args):
    """Send Ctrl-C to whatever is running, and confirm we got a prompt back.

    This is the rescue hatch for firmware/main.py. KeyboardInterrupt is not an
    OSError, so none of that file's guards catch it -- the loop unwinds and the
    board drops to a REPL.

    Three are sent, spaced out. A Ctrl-C landing while the interpreter sits in a
    blocking socket call (the POST) is recorded but not raised until that call
    returns, so a single byte can look like it did nothing.
    """
    for _ in range(3):
        ws.send("\x03")
        time.sleep(0.4)

    tail = ws.read_until(b">>>", timeout=8).decode(errors="replace").strip()

    if ">>>" in tail:
        print("interrupted -- board is at a prompt")
        if tail:
            print(tail)
        return 0

    # No prompt. Either nothing was running, or the loop never yields to the
    # network stack and cannot be reached this way -- the case the PAUSE file
    # exists for, recoverable only by power-cycling and racing the boot.
    print("no prompt after Ctrl-C. device said:", file=sys.stderr)
    print(tail or "(nothing)", file=sys.stderr)
    return 1


def guard_boot_py(args):
    """Local pre-flight, run before the socket is opened.

    Overwriting boot.py removes the `webrepl.start()` that webrepl_setup
    appended, and the node comes back up on WiFi with no way in. Recovering
    that means physically retrieving a board bonded to a washer, so this
    refuses before doing any network work at all.
    """
    remote = args.remote or Path(args.file).name
    if Path(remote).name == "boot.py" and not args.force:
        sys.exit(
            "refusing to overwrite boot.py without --force.\n"
            "  boot.py must start WebREPL itself; clobbering it locks you out\n"
            "  of the node with no error message. See bullets.md problem #4."
        )


def cmd_put(ws, args):
    local = Path(args.file)
    remote = args.remote or local.name
    data = local.read_bytes()

    _, err = ws.execute("import ubinascii\n_f = open(%r, 'wb')" % remote)
    if err.strip():
        sys.exit("could not open %s on device:\n%s" % (remote, err))

    sent = 0
    for i in range(0, len(data), CHUNK_RAW):
        chunk = base64.b64encode(data[i:i + CHUNK_RAW]).decode()
        _, err = ws.execute("_f.write(ubinascii.a2b_base64(%r))" % chunk)
        if err.strip():
            ws.execute("_f.close()")
            sys.exit("transfer failed at byte %d:\n%s" % (sent, err))
        sent += len(data[i:i + CHUNK_RAW])
        print("\r  %s -> %s  %d/%d bytes" % (local, remote, sent, len(data)),
              end="", file=sys.stderr)
    print(file=sys.stderr)

    ws.execute("_f.close()")
    # Confirm rather than assume: a silent short write is the failure that
    # would otherwise show up much later as a mysterious SyntaxError on boot.
    stdout, err = ws.execute(
        "import os\nprint(os.stat(%r)[6])" % remote)
    if err.strip():
        sys.exit("wrote %s but could not stat it:\n%s" % (remote, err))
    on_device = int(stdout.strip())
    if on_device != len(data):
        sys.exit("SIZE MISMATCH: sent %d bytes, device has %d"
                 % (len(data), on_device))
    print("ok: %s -> %s (%d bytes verified)" % (local, remote, on_device))
    return 0


def main():
    p = argparse.ArgumentParser(
        description="Run code on, or copy files to, the ESP32 node over WiFi.",
        epilog="Address and password are read from the repo's .env.")
    p.add_argument("--timeout", type=int,
                   default=int(os.environ.get("WREPL_TIMEOUT", "40")),
                   help="seconds to wait for device output (default 40)")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="execute a local file on the device")
    r.add_argument("file")

    u = sub.add_parser("put", help="copy a local file onto the device")
    u.add_argument("file")
    u.add_argument("remote", nargs="?",
                   help="path on device (default: same basename)")
    u.add_argument("--force", action="store_true",
                   help="allow overwriting boot.py")

    sub.add_parser("interrupt",
                   help="send Ctrl-C to stop whatever is running on the device")

    args = p.parse_args()
    if args.cmd in ("run", "put"):
        if not Path(args.file).is_file():
            sys.exit("no such file: %s" % args.file)
    if args.cmd == "put":
        guard_boot_py(args)

    host, pw = config()
    try:
        # A board busy in main.py's loop cannot enter raw mode, so `interrupt`
        # must connect without asking for it.
        ws = WebREPL(host, pw, timeout=10, raw=args.cmd != "interrupt")
    except OSError as e:
        sys.exit("could not reach %s:%d -- %s\n"
                 "  node powered? IP drifted? check the Orbi's client list."
                 % (host, WEBREPL_PORT, e))
    except RuntimeError as e:
        sys.exit(str(e))

    handlers = {"run": cmd_run, "put": cmd_put, "interrupt": cmd_interrupt}
    try:
        return handlers[args.cmd](ws, args)
    finally:
        ws.close()


if __name__ == "__main__":
    sys.exit(main())
