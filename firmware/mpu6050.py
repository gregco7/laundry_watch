# MPU-6050 driver: wakes the chip, configures the anti-alias filter and ragnge

from micropython import const
from machine import Pin, SoftI2C

ADDR       = const(0x68)
PWR_MGMT_1 = const(0x6B)
CONFIG     = const(0x1A)
ACCEL_CFG  = const(0x1C)
WHO_AM_I   = const(0x75)
ACCEL_X    = const(0x3B)

LSB_PER_G  = 16384.0          # valid only while AFS_SEL == 0


class MPU6050:

    def __init__(self, i2c, addr=ADDR, dlpf=3):
        # Store the bus. Allocate the 6-byte buffer ONCE, here.
        # Verify, then wake, then configure — in that order.
        ...

    def whoami(self):
        # -> int. Raise if it isn't 0x68; a clone answering 0x70
        # or 0x72 is an MPU-6500/9250 with a different register map.
        ...

    def read_raw(self):
        # -> (x, y, z) signed ints. Hot path: no allocation,
        # no floats, no printing. Called ~100x/sec forever.
        ...

    def read_g(self):
        # -> (x, y, z) floats. Bench and calibration only.
        ...


