import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// PIN entry screen to protect parent dashboard.
/// Default PIN is 1234, configurable in settings.
class PinGateScreen extends ConsumerStatefulWidget {
  const PinGateScreen({super.key});

  @override
  ConsumerState<PinGateScreen> createState() => _PinGateScreenState();
}

class _PinGateScreenState extends ConsumerState<PinGateScreen> {
  String _enteredPin = '';
  String _storedPin = '1234';
  bool _error = false;

  @override
  void initState() {
    super.initState();
    // Allow portrait for parent screens
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
      DeviceOrientation.portraitDown,
      DeviceOrientation.landscapeLeft,
      DeviceOrientation.landscapeRight,
    ]);
    _loadPin();
  }

  Future<void> _loadPin() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _storedPin = prefs.getString('parent_pin') ?? '1234';
    });
  }

  void _onDigit(String digit) {
    if (_enteredPin.length >= 4) return;
    setState(() {
      _error = false;
      _enteredPin += digit;
    });
    if (_enteredPin.length == 4) {
      _checkPin();
    }
  }

  void _onBackspace() {
    if (_enteredPin.isEmpty) return;
    setState(() {
      _error = false;
      _enteredPin = _enteredPin.substring(0, _enteredPin.length - 1);
    });
  }

  void _checkPin() {
    if (_enteredPin == _storedPin) {
      context.go('/dashboard');
    } else {
      setState(() {
        _error = true;
        _enteredPin = '';
      });
      HapticFeedback.heavyImpact();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1A1A2E),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.white70),
          onPressed: () => context.go('/'),
        ),
      ),
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Icon(
                  Icons.lock_outline,
                  color: Colors.white54,
                  size: 48,
                ),
                const SizedBox(height: 16),
                const Text(
                  'Parent Dashboard',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 24,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  _error ? 'Incorrect PIN. Try again.' : 'Enter your 4-digit PIN',
                  style: TextStyle(
                    color: _error ? Colors.redAccent : Colors.white54,
                    fontSize: 14,
                  ),
                ),
                const SizedBox(height: 32),

                // PIN dots
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: List.generate(4, (i) {
                    final filled = i < _enteredPin.length;
                    return Container(
                      margin: const EdgeInsets.symmetric(horizontal: 10),
                      width: 20,
                      height: 20,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: filled
                            ? (_error ? Colors.redAccent : Colors.orangeAccent)
                            : Colors.transparent,
                        border: Border.all(
                          color: _error ? Colors.redAccent : Colors.white38,
                          width: 2,
                        ),
                      ),
                    );
                  }),
                ),
                const SizedBox(height: 40),

                // Number pad
                _buildNumberPad(),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildNumberPad() {
    return SizedBox(
      width: 280,
      child: Column(
        children: [
          _padRow(['1', '2', '3']),
          const SizedBox(height: 12),
          _padRow(['4', '5', '6']),
          const SizedBox(height: 12),
          _padRow(['7', '8', '9']),
          const SizedBox(height: 12),
          _padRow(['', '0', 'back']),
        ],
      ),
    );
  }

  Widget _padRow(List<String> keys) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: keys.map((key) {
        if (key.isEmpty) {
          return const SizedBox(width: 72, height: 72);
        }
        if (key == 'back') {
          return SizedBox(
            width: 72,
            height: 72,
            child: TextButton(
              onPressed: _onBackspace,
              style: TextButton.styleFrom(
                shape: const CircleBorder(),
              ),
              child: const Icon(
                Icons.backspace_outlined,
                color: Colors.white54,
                size: 24,
              ),
            ),
          );
        }
        return SizedBox(
          width: 72,
          height: 72,
          child: TextButton(
            onPressed: () => _onDigit(key),
            style: TextButton.styleFrom(
              shape: const CircleBorder(),
              backgroundColor: Colors.white.withValues(alpha: 0.08),
            ),
            child: Text(
              key,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 28,
                fontWeight: FontWeight.w300,
              ),
            ),
          ),
        );
      }).toList(),
    );
  }
}
