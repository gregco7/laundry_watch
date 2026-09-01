# MPU-6050 driver: wakes the chip, configures the anti-alias filter and range

from micropython import const
import time,struct

ADDR       = const(0x68) #Sensor Address for MPU 6050, could be 0x69
EXPECTED_ID = const(0x68) #Always 0x68

PWR_MGMT_1 = const(0x6B) #Power Register on Sensor
CONFIG     = const(0x1A) #Filter Register on Sensor
ACCEL_CFG  = const(0x1C) #Readings Amplitude (in g) Range Register
WHO_AM_I   = const(0x75)
ACCEL_X    = const(0x3B) #Register where accelerometer readings start

LSB_PER_G  = 16384.0          # valid only while Accelerometer Range Register at +-2 g


class MPU6050:

    def __init__(self, i2c, addr=ADDR, dlpf=3):

        # Store the bus. Allocate the 6-byte buffer ONCE, here.
        # Verify, then wake, then configure — in that order.
        # EXAMPLE I2C: i2c = SoftI2C(sda=Pin(21),scl=Pin(22),freq=400000)
        
        self.i2c = i2c 
        self.addr = addr
        self.buf = bytearray(6) #allocate memory for sample reading

        self.whoami()
        
        self.i2c.writeto_mem(self.addr,PWR_MGMT_1,b"\x01") #wake sensor
        self.i2c.writeto_mem(self.addr,CONFIG,bytes([dlpf])) #Set the Digital Low Pass Filter (dlpf)
        self.i2c.writeto_mem(self.addr,ACCEL_CFG,b"\x00") #Accelerometer Range Setting: This sets it to +-2 g (default range)

        
        time.sleep_ms(100)


    def whoami(self):
        # -> int. Raise if it isn't 0x68; a clone answering 0x70
        # or 0x72 is an MPU-6500/9250 with a different register map.
        
        who = self.i2c.readfrom_mem(self.addr,WHO_AM_I,1)[0] 
        if who != EXPECTED_ID:
            raise OSError("not an MPU-6050 at 0x%02x: who=0x%02x" %(self.addr,who))
        
        return who

    def read_raw(self):
        # -> (x, y, z) signed ints. Hot path: no allocation,
        # no floats, no printing. Called ~100x/sec forever.

        self.i2c.readfrom_mem_into(self.addr,ACCEL_X,self.buf)
        return struct.unpack(">hhh",self.buf) # -> (x,y,z)

    def read_g(self):
        # -> (x, y, z) floats. Bench and calibration only.
        x,y,z = self.read_raw()
        return x/LSB_PER_G,y/LSB_PER_G,z/LSB_PER_G


