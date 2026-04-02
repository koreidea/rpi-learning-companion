"""Bluetooth car chassis driver — communicates with Arduino + HC-05.

Connects to the HC-05 Bluetooth module on the car chassis via
Bluetooth Serial Port Profile (SPP). Sends ASCII character commands
that match the Arduino sketch protocol.

NOTE: The bot is mounted FACING BACKWARD on the car chassis.
Direction reversal is handled in the Arduino sketch (motor pins swapped),
so this driver sends logical commands directly:
    'F'  = Forward (toward bot face)  'B'  = Backward (away from bot face)
    'L'  = Spin Left                  'R'  = Spin Right
    'G'  = Forward-Left               'H'  = Forward-Right
    'I'  = Backward-Left              'J'  = Backward-Right
    'S'  = Stop
    '0'-'9' = Speed (0=stop, 9=full)
"""

import asyncio
import random
import threading
import time
from pathlib import Path
from typing import Optional

from loguru import logger

# HC-05 default names to look for during discovery
HC05_NAMES = {"HC-05", "HC05", "HC-06", "HC06", "CAR"}

# Saved MAC address file (so we don't scan every time)
MAC_FILE = Path(__file__).parent / ".car_bt_mac"

# Speed mapping: 0-255 → '0'-'9'
def _speed_char(speed: int) -> str:
    """Map speed 0-255 to ASCII digit '0'-'9'."""
    clamped = max(0, min(255, speed))
    digit = round(clamped * 9 / 255)
    return str(digit)


class CarChassis:
    """Bluetooth SPP driver for the 4WD car chassis."""

    def __init__(self):
        self._serial: Optional[object] = None  # serial.Serial instance
        self._mac: Optional[str] = None
        self._connected = False
        self._rfcomm_proc: Optional[asyncio.subprocess.Process] = None
        self._speaking = False
        self._speak_thread: Optional[threading.Thread] = None

    @property
    def connected(self) -> bool:
        return self._connected

    # ── Connection ────────────────────────────────────────────────

    async def connect(self, mac: Optional[str] = None) -> bool:
        """Connect to the car chassis via Bluetooth SPP.

        Args:
            mac: Optional HC-05 MAC address. If not given, tries saved
                 MAC or scans for HC-05 device.

        Returns:
            True if connected successfully.
        """
        if self._connected:
            return True

        # Determine MAC address
        if mac:
            self._mac = mac
        elif MAC_FILE.exists():
            self._mac = MAC_FILE.read_text().strip()
            logger.info("Using saved car MAC: {}", self._mac)
        else:
            self._mac = await self._scan_for_hc05()

        if not self._mac:
            logger.warning("Car chassis not found via Bluetooth")
            return False

        # Save MAC for next time
        MAC_FILE.write_text(self._mac)

        # Ensure device is paired and trusted
        await self._ensure_paired(self._mac)

        # Bind rfcomm and open serial
        try:
            return await self._open_rfcomm(self._mac)
        except Exception as e:
            logger.error("Failed to connect to car: {}", e)
            return False

    async def disconnect(self):
        """Disconnect from the car chassis."""
        if self._serial:
            try:
                self._send_raw('S')  # Stop motors before disconnecting
                self._serial.close()
            except Exception:
                pass
            self._serial = None

        if self._rfcomm_proc:
            try:
                self._rfcomm_proc.terminate()
                await self._rfcomm_proc.wait()
            except Exception:
                pass
            self._rfcomm_proc = None

        # Release rfcomm binding
        try:
            proc = await asyncio.create_subprocess_exec(
                "rfcomm", "release", "0",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        except Exception:
            pass

        self._connected = False
        logger.info("Car chassis disconnected")

    async def forget(self):
        """Disconnect and delete saved MAC so next connect re-scans."""
        await self.disconnect()
        if MAC_FILE.exists():
            MAC_FILE.unlink()
            logger.info("Cleared saved car MAC address")
        self._mac = None

    # ── Scanning & Pairing ────────────────────────────────────────

    async def _scan_for_hc05(self) -> Optional[str]:
        """Scan for HC-05 Bluetooth device. Returns MAC or None.

        Uses hcitool scan (Bluetooth Classic / BR/EDR) because HC-05
        does not appear in bluetoothctl's default BLE scan on Pi 5.
        """
        logger.info("Scanning for HC-05 via hcitool (Bluetooth Classic)...")

        try:
            # hcitool scan finds Bluetooth Classic devices like HC-05
            proc = await asyncio.create_subprocess_exec(
                "hcitool", "scan", "--flush",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
            output = (stdout or b"").decode()

            # Parse "	XX:XX:XX:XX:XX:XX	DeviceName" lines
            for line in output.splitlines():
                line = line.strip()
                if not line or line.startswith("Scanning"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    mac = parts[0].strip()
                    name = parts[1].strip() if len(parts) > 1 else ""
                    if name.upper() in {n.upper() for n in HC05_NAMES}:
                        logger.info("Found HC-05: {} ({})", name, mac)
                        return mac

            logger.warning("HC-05 not found in hcitool scan results")
            return None

        except asyncio.TimeoutError:
            logger.warning("Bluetooth Classic scan timed out")
            return None
        except Exception as e:
            logger.error("Bluetooth Classic scan error: {}", e)
            return None

    async def scan_for_devices(self) -> list[dict]:
        """Scan and return all nearby Bluetooth devices as list of {mac, name}.

        Used by the settings menu to show available devices.
        """
        logger.info("Scanning for nearby Bluetooth devices...")
        devices = []
        try:
            proc = await asyncio.create_subprocess_exec(
                "bluetoothctl", "--timeout", "8", "scan", "on",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=12)

            proc2 = await asyncio.create_subprocess_exec(
                "bluetoothctl", "devices",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout2, _ = await proc2.communicate()
            output = (stdout2 or b"").decode()

            for line in output.splitlines():
                line = line.strip()
                if "Device" in line:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == "Device" and i + 2 < len(parts):
                            mac = parts[i + 1]
                            name = " ".join(parts[i + 2:])
                            devices.append({"mac": mac, "name": name})
        except Exception as e:
            logger.error("Device scan error: {}", e)

        return devices

    async def _ensure_paired(self, mac: str):
        """Ensure the HC-05 is paired and trusted.

        HC-05 uses legacy PIN pairing (default PIN: 1234).
        We use bluetooth-agent or bluetoothctl with a NoInputNoOutput agent
        and fall back to rfcomm direct binding if pairing fails.
        """
        # First, trust the device so reconnections work
        try:
            proc = await asyncio.create_subprocess_exec(
                "bluetoothctl", "trust", mac,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=10)
        except Exception:
            pass

        # Try pairing with PIN agent script
        # Create a small expect-like script to answer PIN prompt
        pair_script = (
            f'#!/bin/bash\n'
            f'echo -e "agent on\\ndefault-agent\\npair {mac}\\n" | '
            f'bluetoothctl &\n'
            f'sleep 2\n'
            f'echo -e "1234\\n" | bluetoothctl\n'
            f'sleep 2\n'
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                "bash", "-c", pair_script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=15)
            logger.info("Pairing attempt completed for {}", mac)
        except Exception as e:
            logger.debug("Pairing script: {} (may already be paired)", e)

    async def _open_rfcomm(self, mac: str) -> bool:
        """Bind rfcomm0 to the HC-05 and open serial port."""
        import serial  # pyserial

        # Release any existing rfcomm binding
        try:
            proc = await asyncio.create_subprocess_exec(
                "rfcomm", "release", "0",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        except Exception:
            pass

        # Bind rfcomm0 to the HC-05 MAC (channel 1 is default SPP)
        proc = await asyncio.create_subprocess_exec(
            "rfcomm", "bind", "0", mac, "1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            err = (stderr or b"").decode().strip()
            logger.error("rfcomm bind failed: {}", err)
            return False

        # Wait for device to appear
        await asyncio.sleep(1)

        # Open serial connection
        rfcomm_dev = "/dev/rfcomm0"
        try:
            self._serial = serial.Serial(
                rfcomm_dev,
                baudrate=9600,
                timeout=1,
                write_timeout=1,
            )
            self._connected = True
            logger.info("Car chassis connected via {} (MAC: {})", rfcomm_dev, mac)
            return True
        except Exception as e:
            logger.error("Failed to open {}: {}", rfcomm_dev, e)
            return False

    # ── Command Sending ───────────────────────────────────────────

    def _send_raw(self, *chars: str):
        """Send one or more ASCII characters to the Arduino."""
        if not self._serial or not self._connected:
            return
        try:
            data = "".join(chars).encode("ascii")
            self._serial.write(data)
            self._serial.flush()
        except Exception as e:
            logger.error("Car command send failed: {}", e)
            self._connected = False

    async def _send(self, *chars: str):
        """Async wrapper for sending commands."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._send_raw, *chars)

    # ── Public Motor Commands ─────────────────────────────────────

    async def forward(self, speed: int = 200, duration: float = 0):
        """Drive forward (toward bot's face/camera)."""
        logger.debug("Car: forward speed={}", speed)
        await self._send(_speed_char(speed), 'F')
        if duration > 0:
            await asyncio.sleep(duration)
            await self.stop()

    async def backward(self, speed: int = 200, duration: float = 0):
        """Drive backward (away from bot's face/camera)."""
        logger.debug("Car: backward speed={}", speed)
        await self._send(_speed_char(speed), 'B')
        if duration > 0:
            await asyncio.sleep(duration)
            await self.stop()

    async def turn_left(self, speed: int = 180, duration: float = 0):
        """Curve left (bot-perspective)."""
        logger.debug("Car: turn_left speed={}", speed)
        await self._send(_speed_char(speed), 'G')
        if duration > 0:
            await asyncio.sleep(duration)
            await self.stop()

    async def turn_right(self, speed: int = 180, duration: float = 0):
        """Curve right (bot-perspective)."""
        logger.debug("Car: turn_right speed={}", speed)
        await self._send(_speed_char(speed), 'H')
        if duration > 0:
            await asyncio.sleep(duration)
            await self.stop()

    async def spin_left(self, speed: int = 180, duration: float = 0):
        """Spin left in place (bot-perspective)."""
        logger.debug("Car: spin_left speed={}", speed)
        await self._send(_speed_char(speed), 'L')
        if duration > 0:
            await asyncio.sleep(duration)
            await self.stop()

    async def spin_right(self, speed: int = 180, duration: float = 0):
        """Spin right in place (bot-perspective)."""
        logger.debug("Car: spin_right speed={}", speed)
        await self._send(_speed_char(speed), 'R')
        if duration > 0:
            await asyncio.sleep(duration)
            await self.stop()

    async def forward_left(self, speed: int = 200, duration: float = 0):
        """Forward + curve left (bot-perspective)."""
        logger.debug("Car: forward_left speed={}", speed)
        await self._send(_speed_char(speed), 'G')
        if duration > 0:
            await asyncio.sleep(duration)
            await self.stop()

    async def forward_right(self, speed: int = 200, duration: float = 0):
        """Forward + curve right (bot-perspective)."""
        logger.debug("Car: forward_right speed={}", speed)
        await self._send(_speed_char(speed), 'H')
        if duration > 0:
            await asyncio.sleep(duration)
            await self.stop()

    async def stop(self):
        """Stop all motors immediately."""
        logger.debug("Car: stop")
        await self._send('S')

    async def dance(self):
        """Fun dance sequence for kids — longer, synced with display."""
        logger.info("Car: dance!")
        # Wiggle left-right
        await self.spin_left(200, duration=0.4)
        await self.spin_right(200, duration=0.4)
        await self.spin_left(200, duration=0.4)
        await self.spin_right(200, duration=0.4)
        # Forward-backward rock
        await self.forward(180, duration=0.3)
        await self.backward(180, duration=0.3)
        await self.forward(180, duration=0.3)
        await self.backward(180, duration=0.3)
        # Fast spin
        await self.spin_right(255, duration=0.8)
        # More wiggles
        await self.spin_left(220, duration=0.3)
        await self.spin_right(220, duration=0.3)
        await self.spin_left(220, duration=0.3)
        await self.spin_right(220, duration=0.3)
        # Big finale spin
        await self.spin_left(255, duration=1.2)
        await self.stop()

    # ── Speaking Gestures (subtle car movements while bot speaks) ──

    def start_speaking_gestures(self):
        """Start gentle car movements while the bot is speaking."""
        if not self._connected:
            return
        self._speaking = True
        self._speak_thread = threading.Thread(
            target=self._speaking_loop, daemon=True
        )
        self._speak_thread.start()
        logger.info("Car speaking gestures started")

    def stop_speaking_gestures(self):
        """Stop car speaking gestures."""
        self._speaking = False
        if self._speak_thread:
            self._speak_thread.join(timeout=2)
            self._speak_thread = None
        # Stop motors
        if self._connected:
            self._send_raw('S')
        logger.info("Car speaking gestures stopped")

    def _speaking_loop(self):
        """Background loop — subtle car movements while speaking.

        Mostly pauses with occasional short wiggles/rocks to stay in place.
        """
        gestures = [
            self._gesture_wiggle,
            self._gesture_rock,
            self._gesture_pause,
            self._gesture_pause,
            self._gesture_pause,
        ]
        while self._speaking and self._connected:
            gesture = random.choice(gestures)
            try:
                gesture()
            except Exception:
                pass
            if not self._speaking:
                break

    def _gesture_wiggle(self):
        """Quick left-right wiggle in place — stays stationary.
        Bot faces backward: swap L↔R on wire."""
        self._send_raw('7', 'R')
        time.sleep(0.1)
        self._send_raw('7', 'L')
        time.sleep(0.1)
        self._send_raw('S')
        time.sleep(random.uniform(0.6, 1.0))

    def _gesture_rock(self):
        """Tiny forward-backward rock — like a nod.
        Bot faces backward: swap F↔B on wire."""
        self._send_raw('7', 'B')
        time.sleep(0.1)
        self._send_raw('S')
        time.sleep(0.1)
        self._send_raw('7', 'F')
        time.sleep(0.1)
        self._send_raw('S')
        time.sleep(random.uniform(0.6, 1.0))

    def _gesture_pause(self):
        """Just pause — keeps the car still most of the time."""
        time.sleep(random.uniform(0.8, 1.5))

    # ── Cleanup ───────────────────────────────────────────────────

    def close(self):
        """Synchronous cleanup."""
        if self._serial:
            try:
                self._send_raw('S')
                self._serial.close()
            except Exception:
                pass
            self._serial = None
        self._connected = False

    def __del__(self):
        self.close()
