import 'dart:async';

import 'package:bluetooth_classic/bluetooth_classic.dart';
import 'package:bluetooth_classic/models/device.dart';
import 'package:flutter/foundation.dart';
import 'car_protocol.dart';

/// Bluetooth car chassis controller — port of rpi/modules/car.py.
/// Sends ASCII commands to Arduino via HC-05 Bluetooth SPP.
class CarChassis {
  final BluetoothClassic _bt = BluetoothClassic();
  bool _connected = false;
  bool _permissionsInitialized = false;
  String? _macAddress;

  bool get connected => _connected;
  String? get macAddress => _macAddress;

  /// Initialize Bluetooth permissions (required on Android 12+).
  /// Only calls native once — subsequent calls are no-ops.
  Future<bool> initPermissions() async {
    if (_permissionsInitialized) {
      debugPrint('[CarChassis] initPermissions() — already done, skipping');
      return true;
    }
    try {
      debugPrint('[CarChassis] initPermissions() called (first time)');
      final result = await _bt.initPermissions();
      debugPrint('[CarChassis] initPermissions() → $result');
      _permissionsInitialized = result;
      return result;
    } catch (e) {
      debugPrint('[CarChassis] initPermissions() ERROR: $e');
      return false;
    }
  }

  /// Connect to HC-05 by MAC address.
  /// Assumes initPermissions() was already called.
  Future<bool> connect(String mac) async {
    try {
      debugPrint('[CarChassis] connect($mac) called');
      debugPrint('[CarChassis] Calling _bt.connect($mac, ${CarProtocol.sppUuid})...');
      final result = await _bt.connect(mac, CarProtocol.sppUuid);
      debugPrint('[CarChassis] _bt.connect() → $result');
      _connected = result;
      if (result) _macAddress = mac;
      return result;
    } catch (e) {
      debugPrint('[CarChassis] connect() ERROR: $e');
      _connected = false;
      return false;
    }
  }

  /// Disconnect from HC-05.
  Future<void> disconnect() async {
    try {
      debugPrint('[CarChassis] disconnect() called');
      await _bt.disconnect();
    } catch (e) {
      debugPrint('[CarChassis] disconnect() ERROR: $e');
    }
    _connected = false;
  }

  /// Send raw ASCII characters to the Arduino.
  void _sendRaw(String data) {
    if (!_connected) return;
    try {
      _bt.write(data);
    } catch (e) {
      debugPrint('[CarChassis] write() ERROR: $e');
      _connected = false;
    }
  }

  // ── Movement commands (same API as Pi's car.py) ──

  Future<void> forward({int speed = 200, Duration? duration}) async {
    _sendRaw('${CarProtocol.speedChar(speed)}${CarProtocol.forward}');
    if (duration != null) {
      await Future.delayed(duration);
      await stop();
    }
  }

  Future<void> backward({int speed = 200, Duration? duration}) async {
    _sendRaw('${CarProtocol.speedChar(speed)}${CarProtocol.backward}');
    if (duration != null) {
      await Future.delayed(duration);
      await stop();
    }
  }

  Future<void> spinLeft({int speed = 180, Duration? duration}) async {
    _sendRaw('${CarProtocol.speedChar(speed)}${CarProtocol.spinLeft}');
    if (duration != null) {
      await Future.delayed(duration);
      await stop();
    }
  }

  Future<void> spinRight({int speed = 180, Duration? duration}) async {
    _sendRaw('${CarProtocol.speedChar(speed)}${CarProtocol.spinRight}');
    if (duration != null) {
      await Future.delayed(duration);
      await stop();
    }
  }

  Future<void> forwardLeft({int speed = 200, Duration? duration}) async {
    _sendRaw('${CarProtocol.speedChar(speed)}${CarProtocol.forwardLeft}');
    if (duration != null) {
      await Future.delayed(duration);
      await stop();
    }
  }

  Future<void> forwardRight({int speed = 200, Duration? duration}) async {
    _sendRaw('${CarProtocol.speedChar(speed)}${CarProtocol.forwardRight}');
    if (duration != null) {
      await Future.delayed(duration);
      await stop();
    }
  }

  Future<void> stop() async {
    _sendRaw(CarProtocol.stop);
  }

  /// Dance sequence — same as Pi.
  Future<void> dance() async {
    const speed = 220;
    const d = Duration(milliseconds: 400);
    await spinLeft(speed: speed, duration: d);
    await spinRight(speed: speed, duration: d);
    await spinLeft(speed: speed, duration: d);
    await spinRight(speed: speed, duration: d);
    await forward(speed: speed, duration: const Duration(milliseconds: 300));
    await backward(speed: speed, duration: const Duration(milliseconds: 300));
    await stop();
  }

  /// Get bonded/paired devices. Assumes initPermissions() was already called.
  Future<List<Device>> getBondedDevices() async {
    debugPrint('[CarChassis] getBondedDevices() called');
    final devices = await _bt.getPairedDevices();
    debugPrint('[CarChassis] getPairedDevices() → ${devices.length} devices');
    for (final d in devices) {
      debugPrint('[CarChassis]   Device: name=${d.name}, address=${d.address}');
    }
    return devices;
  }

  /// Scan bonded devices for HC-05.
  Future<List<Device>> scanForHC05() async {
    final devices = await getBondedDevices();
    final hc05 = devices
        .where((d) =>
            d.name != null &&
            d.name!.toUpperCase().contains(CarProtocol.hc05Prefix))
        .toList();
    debugPrint('[CarChassis] scanForHC05() → ${hc05.length} HC-05 devices');
    return hc05;
  }
}
