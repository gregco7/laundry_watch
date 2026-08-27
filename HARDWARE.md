# Hardware

The physical side of LaundryWatch. One **node** on the washer: a vibration
sensor wired to a WiFi microcontroller, taped to the outer shell, powered from
the wall. The node samples vibration and POSTs readings to the laptop over the
LAN. Total build cost ~$25.

Quantities below are per node. A second set is worth having as a bench spare —
you can develop against it without pulling the deployed node off the washer.

## Bill of materials

| Item | Qty | Notes |
|---|---|---|
| ESP32 DevKit v1 (ESP32-**WROOM-32**) | 1 | The brain + WiFi. WROOM-32, **not C3/S3** — it has a separate USB-serial bridge (CP2102/CH340) so the serial port survives resets. Native-USB boards drop off the bus on every reboot. |
| MPU6050 accelerometer (GY-521) | 1 | The sensor. I²C, 3.3 V, onboard regulator. Ships with header pins loose — **solder them on**. |
| Micro-USB data cable | 1 | Flashes firmware + powers the board during dev. Must be a **data** cable, not charge-only. |
| USB wall brick | 1 | Always-on mains power for the deployed nodes. No batteries. |
| Dupont jumper wires (F-F) | 1 pack | 4 wires needed. |
| 3M VHB tape | 1 | Mounts the sensor. Better mechanical coupling than gaffer tape — coupling quality *is* signal quality. |
| Soldering iron kit | 1 | For the MPU6050 header pins (~6 joints per board). |

## Wiring — 4 wires per node

The sensor and ESP32 talk directly over I²C (SDA data + SCL clock); the other
two wires are power and ground.

| MPU6050 pin | ESP32 pin | Purpose |
|---|---|---|
| VCC | 3V3 | Power (3.3 V) |
| GND | GND | Ground / return path |
| SDA | GPIO21 | I²C data |
| SCL | GPIO22 | I²C clock |

```
MPU6050            ESP32 (WROOM-32)
 VCC ------------- 3V3
 GND ------------- GND
 SDA ------------- GPIO21
 SCL ------------- GPIO22
```

## Sampling

- **100 Hz**, single axis (the one normal to the panel usually wins — verify per machine).
- Nyquist gives 50 Hz of usable band, which covers everything a machine does:
  - Wash agitation: **0.5–3 Hz** (slow rhythmic thump)
  - Fill: near-silent — mostly water noise, may not separate from `idle`
  - Spin cycle: **10–25 Hz** (motor + drum imbalance, the loudest event)
  - Impacts, door slams, leaning: broadband **25–50 Hz+**
- **Measure the real rate** — naive MicroPython loops drift. A loop you believe
  is 100 Hz but is really 87 Hz throws every frequency estimate off by 15%.

## Mounting

- Outer shell, near a **corner where the panel is stiffest**. Avoid the flexible
  center of a lid — it resonates on its own and adds signal unrelated to the drum.
- **Photograph the placement and mark the outline in tape.** Every recording you
  make is only comparable to the others if the sensor goes back in the same
  spot; a re-mount two inches over changes the band energies and quietly
  invalidates everything trained before it.

## Firmware toolchain

- **MicroPython** — the runtime flashed onto each ESP32 (current stable `.bin`).
- **`esptool`** — flashes MicroPython onto a blank board (once per board).
- **`mpremote`** — pushes `.py` files and gives a REPL. Scriptable.
- **MPU6050 driver** — ~40 lines of I²C register reads, written by hand so the
  sample-rate and low-pass config are understood.

## Node modes

- **Raw mode** — streams 256-sample blocks to the laptop. Development only; how
  training data is collected. Lets you change feature extraction without reflashing.
- **Feature mode** — computes features on-device, POSTs a small JSON summary
  every ~2.5 s. Production; two orders of magnitude less bandwidth.
