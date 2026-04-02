import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'car_chassis.dart';

/// Manages Bluetooth discovery, pairing, and connection to HC-05.
class BluetoothManager {
  final CarChassis car = CarChassis();
  static const _macKey = 'car_bt_mac';

  /// Known HC-05 MAC address (fallback if scanning fails).
  static const _knownHC05Mac = '20:15:01:30:07:17';

  /// Get saved MAC address.
  Future<String?> getSavedMac() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_macKey);
  }

  /// Save MAC address.
  Future<void> saveMac(String mac) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_macKey, mac);
  }

  /// Clear saved MAC address.
  Future<void> forgetMac() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_macKey);
  }

  /// Try to connect to saved MAC, or scan and connect to first HC-05.
  Future<bool> autoConnect() async {
    debugPrint('[BtManager] autoConnect() started');

    // Initialize permissions first
    debugPrint('[BtManager] Step 1: initPermissions');
    await car.initPermissions();

    // Try saved MAC first
    debugPrint('[BtManager] Step 2: getSavedMac');
    final savedMac = await getSavedMac();
    debugPrint('[BtManager] Saved MAC: $savedMac');
    if (savedMac != null) {
      debugPrint('[BtManager] Trying saved MAC: $savedMac');
      final ok = await car.connect(savedMac);
      debugPrint('[BtManager] Saved MAC connect result: $ok');
      if (ok) return true;
    }

    // Scan bonded devices for HC-05
    debugPrint('[BtManager] Step 3: scanForHC05');
    final devices = await car.scanForHC05();
    debugPrint('[BtManager] Found ${devices.length} HC-05 devices');
    for (final device in devices) {
      debugPrint('[BtManager] Trying device: ${device.name} (${device.address})');
      if (device.address.isNotEmpty) {
        final ok = await car.connect(device.address);
        debugPrint('[BtManager] Connect result: $ok');
        if (ok) {
          await saveMac(device.address);
          return true;
        }
      }
    }

    // Fallback: try known HC-05 MAC directly
    debugPrint('[BtManager] Step 4: Fallback — trying known MAC $_knownHC05Mac');
    final ok = await car.connect(_knownHC05Mac);
    debugPrint('[BtManager] Known MAC connect result: $ok');
    if (ok) {
      await saveMac(_knownHC05Mac);
      return true;
    }

    debugPrint('[BtManager] autoConnect() FAILED — no connection');
    return false;
  }

  /// Disconnect car.
  Future<void> disconnect() async {
    await car.disconnect();
  }
}
