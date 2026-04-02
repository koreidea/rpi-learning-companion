import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import '../../bluetooth/car_chassis.dart';
import '../../bluetooth/bluetooth_manager.dart';
import 'joystick.dart';

/// Overlay for car connection + joystick control.
/// Shows at bottom of face screen when car is connected.
class CarControlOverlay extends StatefulWidget {
  final BluetoothManager btManager;

  const CarControlOverlay({super.key, required this.btManager});

  @override
  State<CarControlOverlay> createState() => _CarControlOverlayState();
}

class _CarControlOverlayState extends State<CarControlOverlay> {
  bool _connecting = false;
  bool _showJoystick = false;
  String _status = 'Disconnected';

  BluetoothManager get _bt => widget.btManager;
  CarChassis get _car => _bt.car;

  @override
  Widget build(BuildContext context) {
    return Positioned(
      bottom: 10,
      right: 10,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.end,
        mainAxisSize: MainAxisSize.min,
        children: [
          // Joystick (shown when connected and toggled on)
          if (_showJoystick && _car.connected)
            Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Joystick(
                size: 140,
                onMove: _onJoystickMove,
                onStop: _onJoystickStop,
              ),
            ),

          // Control buttons row
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Status indicator
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                decoration: BoxDecoration(
                  color: Colors.black54,
                  borderRadius: BorderRadius.circular(15),
                ),
                child: Text(
                  _status,
                  style: TextStyle(
                    color: _car.connected ? Colors.green : Colors.grey,
                    fontSize: 12,
                  ),
                ),
              ),
              const SizedBox(width: 8),

              // Connect/Disconnect button
              _buildIconButton(
                icon: _car.connected ? Icons.bluetooth_connected : Icons.bluetooth,
                color: _car.connected ? Colors.blue : Colors.grey,
                onTap: _connecting ? null : (_car.connected ? _disconnect : _connect),
                loading: _connecting,
              ),

              // Joystick toggle (only when connected)
              if (_car.connected) ...[
                const SizedBox(width: 8),
                _buildIconButton(
                  icon: Icons.gamepad,
                  color: _showJoystick ? Colors.orange : Colors.grey,
                  onTap: () => setState(() => _showJoystick = !_showJoystick),
                ),
              ],

              // Dance button (only when connected)
              if (_car.connected) ...[
                const SizedBox(width: 8),
                _buildIconButton(
                  icon: Icons.music_note,
                  color: Colors.purple,
                  onTap: () => _car.dance(),
                ),
              ],
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildIconButton({
    required IconData icon,
    required Color color,
    VoidCallback? onTap,
    bool loading = false,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 40,
        height: 40,
        decoration: BoxDecoration(
          color: Colors.black54,
          shape: BoxShape.circle,
          border: Border.all(color: color.withValues(alpha: 0.5)),
        ),
        child: loading
            ? Padding(
                padding: const EdgeInsets.all(10),
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  color: color,
                ),
              )
            : Icon(icon, color: color, size: 20),
      ),
    );
  }

  Future<void> _connect() async {
    setState(() {
      _connecting = true;
      _status = 'Connecting...';
    });

    try {
      // Add a 15-second timeout so it doesn't hang forever
      final ok = await _bt.autoConnect().timeout(
        const Duration(seconds: 15),
        onTimeout: () {
          debugPrint('[CarOverlay] autoConnect() TIMED OUT after 15s');
          return false;
        },
      );
      if (mounted) {
        setState(() {
          _connecting = false;
          _status = ok ? 'Connected' : 'Not found';
        });
      }
    } catch (e) {
      debugPrint('[CarOverlay] _connect() ERROR: $e');
      if (mounted) {
        setState(() {
          _connecting = false;
          _status = 'Error: $e';
        });
      }
    }
  }

  Future<void> _disconnect() async {
    await _bt.disconnect();
    setState(() {
      _status = 'Disconnected';
      _showJoystick = false;
    });
  }

  void _onJoystickMove(JoystickDirection dir, double distance) {
    if (!_car.connected) return;
    final speed = (distance * 255).round().clamp(0, 255);

    switch (dir) {
      case JoystickDirection.forward:
        _car.forward(speed: speed);
      case JoystickDirection.backward:
        _car.backward(speed: speed);
      case JoystickDirection.left:
        _car.spinLeft(speed: speed);
      case JoystickDirection.right:
        _car.spinRight(speed: speed);
      case JoystickDirection.forwardLeft:
        _car.forwardLeft(speed: speed);
      case JoystickDirection.forwardRight:
        _car.forwardRight(speed: speed);
      case JoystickDirection.backwardLeft:
        _car.spinLeft(speed: speed); // approximate
      case JoystickDirection.backwardRight:
        _car.spinRight(speed: speed); // approximate
      case JoystickDirection.none:
        _car.stop();
    }
  }

  void _onJoystickStop() {
    if (_car.connected) _car.stop();
  }
}
