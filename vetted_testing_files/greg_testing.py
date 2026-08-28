#MicroPython slimmed-down copies ~ operate same way as standard CPython stdlib dependencies
import time, struct
from array import array

#MicroPython native
from micropython import const 

#Chip/Microprocessor
from machine import Pin, SoftI2C

#Boot these constants as hexadecimal integer constants to save RAM on the microcontroller

ADDR = const(0x68) #sensor address, decimal 104
PWR_MGMT_1 = const(0x6B) # power register, decimal 107
CONFIG = const(0x1A) #digital low-pass filter setting, decimal 26
WHO_AM_I = const(0x75) #read-only ID register, hexadecimal to decimal 117d

# First of 6 acceleration bytes, hexadecimal to decimal 59
# This is where the accelerometer temporarily will store the g force data. 
# Six registers, starting at 59 out of ~117 registers
#Called ACCEL_X cus the first (and second) address is for X, next four for Y Z
ACCEL_X = const(0x3B) 

#Tuning constants
N = const(500) #samples -> ~ 5 seconds
PERIOD = const(10000) #microseconds (10^-6 seconds) between samples -> 100 hz | (0.01 seconds)

"""
Why this represents a +-2g range. The MPU 6050 uses a 16-bit 
Analog-To-Digital convertor (ADC).

A 16-bit register can hold 2^16 (65,535) possible values for any given axis

The sensor splits these values into negative and positive motions, creating a raw scale
from -32,768 to +32,768

And 1 LSB ubiquitously represents a change of exactly 1 count on that raw scale.

So, a LSB_PER_G of 16384 means you are mapping each gram to approximately 1/4 of the
digital resolution that the chip is offering, meaning you have a span of 4 grams, representing a +-2g range.

Tho this variable is kinda useless, cus the +-2g is already set on boot, would have to explicitly change it
"""
LSB_PER_G = 16384.0 



i2c = SoftI2C(scl=Pin(22),sda = Pin(21),freq = 400000) # (review: what does freq parameter represent)
seen = i2c.scan()

if ADDR not in seen:
    raise SystemExit("nothing at sensor address: 0x68—scan saw %s" % seen)

# (...) : (addr:which device on the bus, memaddr: which register inside the device, nbytes: how many bytes to read)
who = i2c.readfrom_mem(ADDR, WHO_AM_I,1)[0] #shud be 104

#b"\x00" is a bytes literal. the b prefix means these are raw bytes not text
i2c.writeto_mem(ADDR,PWR_MGMT_1, b"\x00") # wake sensor command, value 00
i2c.writeto_mem(ADDR, CONFIG, b"\x03") # DLPF 44 hz, value 03
time.sleep_ms(100) # let the filter settle

# bytearray(b'\x00\x00\x00\x00\x00\x00') in bytes
# bytearray(6) : one byte array with six zero bytes
# bytearray([6]) : one byte array with one six byte: ONE byte, holding the value 6
buf = bytearray(6) #mutable block of bytes. six zero bytes 

xs, ys, zs = array("h", [0] * N), array("h", [0] * N), array("h", [0]*N)

prev = t0 = next_t = time.ticks_us() #binds all three names to one int. Ints are immutable

late = 0
worst = 0

#SAMPLER LOOP

for i in range(N):
    next_t = time.ticks_add(next_t,PERIOD)

    #par. 1: "0x68, wake up" par. 2: "point at register 0x3B" par. 3 "give me 6 bytes"

    #0x3B ACCEL_XOUT_H : top 8 bits of X
    #0x3C ACCEL_XOUT_L : bottom 8 bits of X

    i2c.readfrom_mem_into(ADDR,ACCEL_X,buf)
    
    xs[i],ys[i],zs[i] = struct.unpack(">hhh",buf)
    now = time.ticks_us()
    gap = time.ticks_diff(now,prev) 
    prev = now 
    
    #Statistical Semantics Below

    # 1: Evalutate the gap between now and last sample

    if i and gap > worst: #Syntax here, on first loop i = 0 so there cannot be a worst on the first loop
        worst = gap
    
    slack = time.ticks_diff(next_t,time.ticks_us())

    if slack > 0:
        time.sleep_us(slack)
    else:
        late += 1

elapsed = time.ticks_diff(time.ticks_us(), t0)

#SUMMARY SECTION
#MEAN, RANGE, LOW AND HIGH FOR EACH AXIS

# READING (in g) = READING / LSB_PER_G
# +16,251 / 16384 or whatever ~= 1


def DigitalStepToForceg(reading):
    # -32768 to +32767 is the full digital resolution/reading window. Sensitivity is +-2g, this returns the digital window reading back in grams linear motion metric
    return reading / LSB_PER_G


def axis(name: str,data: array):
    minimum,maximum = min(data),max(data)

    print( "accelerometer readings for axis %s:\n MEAN: %.3fg, RANGE: %.3fg <-> %.3fg" 
    % (name, DigitalStepToForceg(sum(data)/len(data)), DigitalStepToForceg(minimum),DigitalStepToForceg(maximum) ))

READINGS_X_MISSING = not any(xs)
READINGS_Y_MISSING = not any(ys)
READINGS_Z_MISSING = not any(zs)

if READINGS_X_MISSING or READINGS_Y_MISSING or READINGS_Z_MISSING:
    if READINGS_X_MISSING and READINGS_Y_MISSING and READINGS_Z_MISSING:
        print("Complete Failure, no axis could be read, maybe power? Check XCC cord")
    else:
        print("Partial failure, 1 or 2 axis could not be read")
else:
    print("late iterations: %.3f" % late)
    print("worst gap: %.3f seconds" % (worst/1000000))
    axis("x",xs)
    axis("y",ys)
    axis("z",zs)





