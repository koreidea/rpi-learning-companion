"""Shared SPI bus lock for display + touch on SPI0.

Both the ILI9341 display (CE0) and XPT2046 touch (CE1) share the same
SPI0 bus (MOSI, MISO, SCK). They must not talk simultaneously or data
will be corrupted. This module provides a single threading.Lock that
both drivers acquire before any SPI transaction.
"""

import threading

# Single global lock for SPI0 bus access
spi_bus_lock = threading.Lock()
