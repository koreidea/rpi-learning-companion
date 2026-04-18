#!/usr/bin/env python3
"""
WiFi AP Fallback Manager for RPi Learning Companion.

On boot:
1. Waits up to 30s for WiFi connection
2. If no connection, starts a hotspot "HairoBot-Setup"
3. Runs a captive portal on port 80 for WiFi configuration
4. Once user picks a network, connects and exits

Uses NetworkManager (nmcli) — default on Raspberry Pi OS Bookworm.
"""

import http.server
import json
import os
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

AP_SSID = "HairoBot-Setup"
AP_PASSWORD = "hairobot1"  # Min 8 chars for WPA
AP_IFACE = "wlan0"
PORTAL_PORT = 80
HOTSPOT_CON_NAME = "hairobot-hotspot"
PORTAL_HTML = Path(__file__).parent / "wifi_portal.html"
CHECK_TIMEOUT = 30  # seconds to wait for existing WiFi
AP_IP = "10.42.0.1"  # nmcli hotspot default

# --- Connectivity checks ---

def is_wifi_connected():
    """Check if wlan0 has an IP and can reach the gateway."""
    try:
        result = subprocess.run(
            ["nmcli", "-t", "-f", "DEVICE,STATE", "device"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.strip().split("\n"):
            parts = line.split(":")
            if len(parts) >= 2 and parts[0] == AP_IFACE and parts[1] == "connected":
                return True
    except Exception:
        pass
    return False


def wait_for_wifi(timeout=CHECK_TIMEOUT):
    """Wait up to timeout seconds for WiFi to connect."""
    print(f"[wifi] Waiting up to {timeout}s for WiFi connection...")
    start = time.time()
    while time.time() - start < timeout:
        if is_wifi_connected():
            print("[wifi] WiFi connected!")
            return True
        time.sleep(2)
    print("[wifi] No WiFi connection found.")
    return False


# --- Network scanning ---

def scan_networks():
    """Scan for available WiFi networks. Returns list of dicts."""
    try:
        # Rescan first
        subprocess.run(
            ["nmcli", "device", "wifi", "rescan", "ifname", AP_IFACE],
            capture_output=True, timeout=10
        )
        time.sleep(2)
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY", "device", "wifi", "list", "--rescan", "no"],
            capture_output=True, text=True, timeout=10
        )
        networks = []
        seen = set()
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            # nmcli -t uses : as separator, but SSID can contain :
            # Format: SSID:SIGNAL:SECURITY
            # Parse from the right since SIGNAL and SECURITY don't contain :
            parts = line.rsplit(":", 2)
            if len(parts) < 3:
                continue
            ssid, signal, security = parts[0], parts[1], parts[2]
            ssid = ssid.strip()
            if not ssid or ssid == AP_SSID or ssid in seen:
                continue
            seen.add(ssid)
            networks.append({
                "ssid": ssid,
                "signal": int(signal) if signal.isdigit() else 0,
                "security": security.strip()
            })
        # Sort by signal strength descending
        networks.sort(key=lambda n: n["signal"], reverse=True)
        return networks
    except Exception as e:
        print(f"[wifi] Scan error: {e}")
        return []


# --- AP mode (hotspot) ---

def start_hotspot():
    """Start WiFi hotspot using nmcli."""
    print(f"[wifi] Starting hotspot '{AP_SSID}'...")
    # Remove any existing hotspot connection
    subprocess.run(
        ["nmcli", "connection", "delete", HOTSPOT_CON_NAME],
        capture_output=True, timeout=10
    )
    time.sleep(1)

    result = subprocess.run(
        [
            "nmcli", "device", "wifi", "hotspot",
            "ifname", AP_IFACE,
            "con-name", HOTSPOT_CON_NAME,
            "ssid", AP_SSID,
            "password", AP_PASSWORD,
        ],
        capture_output=True, text=True, timeout=15
    )
    if result.returncode != 0:
        print(f"[wifi] Hotspot failed: {result.stderr}")
        return False

    print(f"[wifi] Hotspot active: SSID='{AP_SSID}', Password='{AP_PASSWORD}'")
    print(f"[wifi] Portal at http://{AP_IP}")
    return True


def stop_hotspot():
    """Stop the hotspot and remove the connection."""
    print("[wifi] Stopping hotspot...")
    subprocess.run(
        ["nmcli", "connection", "down", HOTSPOT_CON_NAME],
        capture_output=True, timeout=10
    )
    subprocess.run(
        ["nmcli", "connection", "delete", HOTSPOT_CON_NAME],
        capture_output=True, timeout=10
    )


# --- Connect to selected WiFi ---

def connect_wifi(ssid, password):
    """Connect to a WiFi network. Returns (success, message)."""
    print(f"[wifi] Connecting to '{ssid}'...")

    # Stop hotspot first
    stop_hotspot()
    time.sleep(2)

    # Try to connect
    cmd = ["nmcli", "device", "wifi", "connect", ssid, "ifname", AP_IFACE]
    if password:
        cmd += ["password", password]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    if result.returncode == 0:
        # Verify connectivity
        time.sleep(3)
        if is_wifi_connected():
            ip = get_ip()
            print(f"[wifi] Connected to '{ssid}', IP: {ip}")
            return True, f"Connected! Bot is now at http://{ip}:8080"
        else:
            return False, "Connected but no IP. Check the network."
    else:
        err = result.stderr.strip() or "Connection failed"
        print(f"[wifi] Connection failed: {err}")
        # Restart hotspot so user can try again
        start_hotspot()
        return False, f"Failed: {err}"


def get_ip():
    """Get current wlan0 IP address."""
    try:
        result = subprocess.run(
            ["nmcli", "-t", "-f", "IP4.ADDRESS", "device", "show", AP_IFACE],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.strip().split("\n"):
            if "IP4.ADDRESS" in line:
                # Format: IP4.ADDRESS[1]:192.168.0.174/24
                addr = line.split(":", 1)[1].split("/")[0]
                return addr
    except Exception:
        pass
    return "unknown"


# --- DNS redirect (captive portal detection) ---

class DNSRedirectHandler:
    """Simple UDP DNS server that redirects all queries to AP_IP."""

    def __init__(self, bind_ip=AP_IP, port=53):
        self.bind_ip = bind_ip
        self.port = port
        self.sock = None
        self._running = False

    def start(self):
        self._running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.bind_ip, self.port))
        self.sock.settimeout(1)
        print(f"[wifi] DNS redirect active on {self.bind_ip}:{self.port}")
        while self._running:
            try:
                data, addr = self.sock.recvfrom(512)
                response = self._build_response(data)
                self.sock.sendto(response, addr)
            except socket.timeout:
                continue
            except Exception:
                continue

    def stop(self):
        self._running = False
        if self.sock:
            self.sock.close()

    def _build_response(self, query):
        """Build a DNS response redirecting to AP_IP."""
        # DNS header: copy ID, set flags (response, authoritative)
        response = bytearray(query[:2])  # Transaction ID
        response += b'\x81\x80'  # Flags: response, authoritative, no error
        response += query[4:6]   # Questions count
        response += query[4:6]   # Answers count (same as questions)
        response += b'\x00\x00'  # Authority RRs
        response += b'\x00\x00'  # Additional RRs

        # Copy question section
        pos = 12
        while pos < len(query):
            if query[pos] == 0:
                pos += 5  # null byte + QTYPE(2) + QCLASS(2)
                break
            pos += query[pos] + 1
        response += query[12:pos]

        # Answer section: pointer to name in question, type A, class IN, TTL 60, IP
        response += b'\xc0\x0c'  # Pointer to name in question
        response += b'\x00\x01'  # Type A
        response += b'\x00\x01'  # Class IN
        response += b'\x00\x00\x00\x3c'  # TTL 60s
        response += b'\x00\x04'  # Data length 4
        response += bytes(int(x) for x in AP_IP.split('.'))  # IP address

        return bytes(response)


# --- Captive Portal HTTP Server ---

class PortalHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for the captive portal."""

    def log_message(self, format, *args):
        # Suppress default logging
        pass

    def do_GET(self):
        path = self.path.split("?")[0]

        # Captive portal detection endpoints — redirect to portal
        captive_paths = [
            "/generate_204", "/gen_204",           # Android
            "/hotspot-detect.html", "/library/test/success.html",  # Apple
            "/connecttest.txt", "/ncsi.txt",       # Windows
            "/canonical.html", "/success.txt",     # Others
        ]

        if path == "/" or path in captive_paths:
            self._serve_portal()
        elif path == "/api/scan":
            self._serve_scan()
        elif path == "/api/status":
            self._serve_status()
        else:
            # Redirect everything else to portal
            self.send_response(302)
            self.send_header("Location", f"http://{AP_IP}/")
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/connect":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len).decode()
            data = json.loads(body)
            ssid = data.get("ssid", "")
            password = data.get("password", "")

            # Run connection in background thread so we can respond first
            def do_connect():
                time.sleep(1)  # Let response go out
                success, msg = connect_wifi(ssid, password)
                if success:
                    # Signal main thread to exit
                    time.sleep(2)
                    os.kill(os.getpid(), signal.SIGUSR1)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "connecting",
                "message": f"Connecting to {ssid}... Please wait 15 seconds, then check if the bot's eyes turn green."
            }).encode())

            threading.Thread(target=do_connect, daemon=True).start()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_portal(self):
        """Serve the captive portal HTML page."""
        try:
            html = PORTAL_HTML.read_text()
        except Exception:
            html = "<h1>HairoBot WiFi Setup</h1><p>Portal page not found.</p>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html.encode())))
        # Prevent caching
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(html.encode())

    def _serve_scan(self):
        """Return scanned WiFi networks as JSON."""
        networks = scan_networks()
        data = json.dumps(networks)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data.encode())

    def _serve_status(self):
        """Return current connection status."""
        connected = is_wifi_connected()
        ip = get_ip() if connected else None
        data = json.dumps({"connected": connected, "ip": ip})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data.encode())


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


# --- Main ---

def main():
    # Step 1: Wait for existing WiFi
    if wait_for_wifi(CHECK_TIMEOUT):
        ip = get_ip()
        print(f"[wifi] Already connected. IP: {ip}. Exiting.")
        sys.exit(0)

    # Step 2: Start hotspot
    if not start_hotspot():
        print("[wifi] Failed to start hotspot. Exiting.")
        sys.exit(1)

    time.sleep(2)

    # Step 3: Start DNS redirect (so captive portal auto-opens on phones)
    dns = DNSRedirectHandler(AP_IP, 53)
    dns_thread = threading.Thread(target=dns.start, daemon=True)
    dns_thread.start()

    # Step 4: Start captive portal web server
    server = ReusableTCPServer((AP_IP, PORTAL_PORT), PortalHandler)
    print(f"[wifi] Captive portal running on http://{AP_IP}:{PORTAL_PORT}")

    # Handle SIGUSR1 (sent after successful WiFi connection)
    def handle_connected(signum, frame):
        print("[wifi] WiFi connected via portal. Shutting down...")
        dns.stop()
        server.shutdown()

    signal.signal(signal.SIGUSR1, handle_connected)

    # Run server (blocks until shutdown)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        dns.stop()
        stop_hotspot()
        print("[wifi] WiFi manager exiting.")


if __name__ == "__main__":
    main()
