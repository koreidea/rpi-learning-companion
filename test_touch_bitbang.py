#!/usr/bin/env python3
"""Bit-bang SPI test for XPT2046 touch controller.
Bypasses kernel spidev to test raw GPIO communication."""

import lgpio
import time

h = lgpio.gpiochip_open(4)

# Use free GPIO pins to bit-bang (SPI pins are owned by kernel overlay)
# We'll use free pins: BCM6=CLK, BCM13=MOSI, BCM22=CS, BCM26=MISO
# BUT we need to use the ACTUAL physical wires to the XPT2046
# So instead, let's use spidev for CLK/MOSI/MISO and only manual CS

# Actually — let's use a free GPIO pin as CS and spidev for the SPI bus
# This tests if the issue is CE1 (BCM7) not reaching the XPT2046

import spidev

# Use BCM6 (free) as manual CS for XPT2046 T_CS
MANUAL_CS = 6
IRQ = 5

lgpio.gpio_claim_output(h, MANUAL_CS, 1)  # Start HIGH (deselected)
lgpio.gpio_claim_input(h, IRQ, lgpio.SET_PULL_UP)

spi = spidev.SpiDev()
spi.open(0, 1)
spi.max_speed_hz = 500000
spi.mode = 0


def bb_xfer_byte(byte_out):
    result = 0
    for bit in range(8):
        lgpio.gpio_write(h, MOSI, (byte_out >> (7 - bit)) & 1)
        time.sleep(0.000005)
        lgpio.gpio_write(h, CLK, 1)
        time.sleep(0.000005)
        result = (result << 1) | lgpio.gpio_read(h, MISO)
        lgpio.gpio_write(h, CLK, 0)
        time.sleep(0.000005)
    return result


def read_xpt(cmd):
    lgpio.gpio_write(h, CS, 0)
    time.sleep(0.0001)
    bb_xfer_byte(cmd)
    b1 = bb_xfer_byte(0x00)
    b2 = bb_xfer_byte(0x00)
    lgpio.gpio_write(h, CS, 1)
    return ((b1 << 8) | b2) >> 3 & 0x0FFF


print("=== Full Bit-Bang SPI XPT2046 Test ===")
print()

# Read without touching
print("Not touching:")
for i in range(3):
    x = read_xpt(0xD0)
    y = read_xpt(0x90)
    irq = lgpio.gpio_read(h, IRQ)
    print(f"  raw_x={x:4d}  raw_y={y:4d}  IRQ={irq}")
    time.sleep(0.1)

print()
print("TOUCH AND HOLD screen! Reading in 3 seconds...")
time.sleep(3)

print("While touching:")
for i in range(10):
    irq = lgpio.gpio_read(h, IRQ)
    x = read_xpt(0xD0)
    y = read_xpt(0x90)
    print(f"  raw_x={x:4d}  raw_y={y:4d}  IRQ={irq}")
    time.sleep(0.2)

# Cleanup
for pin in [CLK, MOSI, CS]:
    lgpio.gpio_free(h, pin)
lgpio.gpio_free(h, MISO)
lgpio.gpio_free(h, IRQ)
lgpio.gpiochip_close(h)
