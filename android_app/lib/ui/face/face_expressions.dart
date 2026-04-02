import 'face_state.dart';

/// State colors — same as Pi's STATE_COLORS dict.
const Map<String, List<double>> stateColors = {
  'ready': [255, 120, 0],       // Dark Orange
  'listening': [30, 144, 255],  // Blue
  'processing': [255, 200, 50], // Yellow
  'speaking': [180, 100, 255],  // Purple
  'error': [255, 60, 60],       // Red
  'setup': [80, 80, 80],        // Gray
  'loading': [255, 200, 50],    // Yellow
  'sleeping': [80, 80, 80],     // Gray
};

/// Build target FaceParams for a given state — port of _build_expression() in tft_display.py.
FaceParams buildExpression(String state) {
  final c = stateColors[state] ?? [50, 205, 50];
  final r = c[0], g = c[1], b = c[2];

  EyeParams eye({
    double w = 90,
    double h = 75,
    double cr = 22,
    double px = 0,
    double py = 0,
    double ps = 0.5,
    double sl = 0,
    double sr = 0,
    double ga = 0.3,
  }) {
    return EyeParams(
      width: w, height: h, cornerRadius: cr,
      pupilX: px, pupilY: py, pupilScale: ps,
      slopeLeft: sl, slopeRight: sr,
      colorR: r, colorG: g, colorB: b,
      glowAlpha: ga,
    );
  }

  switch (state) {
    case 'ready':
      // Relaxed, green eyes, normal pupils, subtle sparkles
      return FaceParams(
        leftEye: eye(w: 120, h: 140, cr: 35, ps: 0.40),
        rightEye: eye(w: 120, h: 140, cr: 35, ps: 0.40),
        mouthStyle: 'none',
      );

    case 'listening':
      // Tall blue eyes, dilated pupils, looking slightly up
      return FaceParams(
        leftEye: eye(w: 120, h: 160, cr: 35, ps: 0.55, py: -0.15, ga: 0.5),
        rightEye: eye(w: 120, h: 160, cr: 35, ps: 0.55, py: -0.15, ga: 0.5),
        mouthStyle: 'none',
      );

    case 'processing':
      // Narrow yellow eyes, looking up-right, thinking
      return FaceParams(
        leftEye: eye(w: 120, h: 100, cr: 30, px: 0.5, py: -0.5, ps: 0.40),
        rightEye: eye(w: 120, h: 100, cr: 30, px: 0.5, py: -0.5, ps: 0.40),
        mouthStyle: 'none',
      );

    case 'speaking':
      // Purple, rounded, happy eyelids, animated mouth
      return FaceParams(
        leftEye: eye(w: 120, h: 100, cr: 45, ps: 0.40, sl: -0.3, sr: -0.3),
        rightEye: eye(w: 120, h: 100, cr: 45, ps: 0.40, sl: -0.3, sr: -0.3),
        mouthOpen: 0.5,
        mouthWidth: 40,
        mouthStyle: 'open',
      );

    case 'error':
      // Red X marks
      return FaceParams(
        leftEye: eye(w: 100, h: 100, cr: 20),
        rightEye: eye(w: 100, h: 100, cr: 20),
        mouthStyle: 'flat',
        mouthWidth: 50,
      );

    case 'loading':
      // Yellow, half-open sleepy eyes
      return FaceParams(
        leftEye: eye(w: 120, h: 65, cr: 30, py: 0.1),
        rightEye: eye(w: 120, h: 65, cr: 30, py: 0.1),
        mouthStyle: 'none',
      );

    case 'sleeping':
      // Thin horizontal lines
      return FaceParams(
        leftEye: eye(w: 120, h: 6, cr: 3),
        rightEye: eye(w: 120, h: 6, cr: 3),
        mouthStyle: 'none',
      );

    default:
      return FaceParams(
        leftEye: eye(),
        rightEye: eye(),
      );
  }
}
