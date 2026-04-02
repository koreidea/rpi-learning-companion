/// ASCII commands matching the Arduino sketch — identical protocol to Pi.
/// Direction commands: F/B/L/R/G/H/I/J/S
/// Speed: '0'-'9' (maps from 0-255)
class CarProtocol {
  static const String forward = 'F';
  static const String backward = 'B';
  static const String spinLeft = 'L';
  static const String spinRight = 'R';
  static const String forwardLeft = 'G';
  static const String forwardRight = 'H';
  static const String backwardLeft = 'I';
  static const String backwardRight = 'J';
  static const String stop = 'S';

  /// Convert speed 0-255 to ASCII char '0'-'9'.
  static String speedChar(int speed) {
    final mapped = (speed * 9 / 255).round().clamp(0, 9);
    return mapped.toString();
  }

  /// HC-05 SPP UUID (lowercase — some Android devices require it)
  static const String sppUuid = '00001101-0000-1000-8000-00805f9b34fb';

  /// Default HC-05 PIN
  static const String defaultPin = '1234';

  /// Default HC-05 name prefix
  static const String hc05Prefix = 'HC-05';
}
