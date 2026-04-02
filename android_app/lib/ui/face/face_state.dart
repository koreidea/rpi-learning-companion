import 'dart:ui';

/// Animatable parameters for a single eye — direct port from tft_display.py EyeParams.
class EyeParams {
  double x;
  double y;
  double width;
  double height;
  double cornerRadius;
  double slopeLeft;
  double slopeRight;
  double pupilX; // -1..1 normalized
  double pupilY; // -1..1 normalized
  double pupilScale; // 0..1
  double colorR;
  double colorG;
  double colorB;
  double glowAlpha;

  EyeParams({
    this.x = 0,
    this.y = 0,
    this.width = 90,
    this.height = 75,
    this.cornerRadius = 22,
    this.slopeLeft = 0,
    this.slopeRight = 0,
    this.pupilX = 0,
    this.pupilY = 0,
    this.pupilScale = 0.5,
    this.colorR = 50,
    this.colorG = 205,
    this.colorB = 50,
    this.glowAlpha = 0.3,
  });

  Color get color => Color.fromARGB(255, colorR.round().clamp(0, 255),
      colorG.round().clamp(0, 255), colorB.round().clamp(0, 255));

  EyeParams copy() => EyeParams(
        x: x, y: y, width: width, height: height,
        cornerRadius: cornerRadius,
        slopeLeft: slopeLeft, slopeRight: slopeRight,
        pupilX: pupilX, pupilY: pupilY, pupilScale: pupilScale,
        colorR: colorR, colorG: colorG, colorB: colorB,
        glowAlpha: glowAlpha,
      );
}

/// Complete animatable face state.
class FaceParams {
  EyeParams leftEye;
  EyeParams rightEye;
  double mouthOpen; // 0..1
  double mouthWidth;
  String mouthStyle; // "none", "happy", "flat", "open"

  FaceParams({
    EyeParams? leftEye,
    EyeParams? rightEye,
    this.mouthOpen = 0,
    this.mouthWidth = 40,
    this.mouthStyle = 'none',
  })  : leftEye = leftEye ?? EyeParams(),
        rightEye = rightEye ?? EyeParams();

  FaceParams copy() => FaceParams(
        leftEye: leftEye.copy(),
        rightEye: rightEye.copy(),
        mouthOpen: mouthOpen,
        mouthWidth: mouthWidth,
        mouthStyle: mouthStyle,
      );
}

/// Scheduler bookkeeping for procedural animations.
class AnimState {
  // Blink
  double blinkTimer = 4.0;
  double blinkPhase = -1.0;
  double blinkMult = 1.0;

  // Gaze drift
  double gazeTargetX = 0.0;
  double gazeTargetY = 0.0;
  double gazeTimer = 3.0;

  // Micro-saccade
  double saccadeOx = 0.0;
  double saccadeOy = 0.0;
  double saccadeTimer = 2.0;

  // Breathing
  double breathPhase = 0.0;

  // Processing dots
  int dotPhase = 0;
  double dotTimer = 0.0;

  // Speaking mouth
  double mouthPhase = 0.0;

  // Loading spinner
  int spinnerIdx = 0;
  double spinnerTimer = 0.0;

  // Sleep Z
  double zPhase = 0.0;

  // Transition
  double transitionProgress = 1.0;

  // Listening pulse
  double listenPhase = 0.0;

  // Idle look-around
  double lookPhase = 0.0;
  bool lookActive = false;
  double lookCooldown = 5.0;
  List<List<double>> lookSeq = [];
  List<double> lookFrom = [0.0, 0.0];

  // Tickle
  double tickleTimer = 0.0;
  double ticklePhase = 0.0;

  // Idle games
  double idleTimer = 0.0;
  bool gameActive = false;
}

/// Sparkle particle.
class Sparkle {
  double x, y, life, maxLife, size;
  Sparkle({
    this.x = 0,
    this.y = 0,
    this.life = 0.8,
    this.maxLife = 0.8,
    this.size = 2,
  });
}

/// Heart particle for tickle animation.
class Heart {
  double x, y, life, maxLife, size, driftX;
  Heart({
    this.x = 0,
    this.y = 0,
    this.life = 1.5,
    this.maxLife = 1.5,
    this.size = 6,
    this.driftX = 0,
  });
}

// ─── Lerp utilities ──────────────────────────────────────────────
double _lerp(double a, double b, double t) => a + (b - a) * t;

EyeParams lerpEye(EyeParams cur, EyeParams tgt,
    {double alpha = 0.25, double colorAlpha = 0.15, double pupilAlpha = 0.35}) {
  return EyeParams(
    x: _lerp(cur.x, tgt.x, alpha),
    y: _lerp(cur.y, tgt.y, alpha),
    width: _lerp(cur.width, tgt.width, alpha),
    height: _lerp(cur.height, tgt.height, alpha),
    cornerRadius: _lerp(cur.cornerRadius, tgt.cornerRadius, alpha),
    slopeLeft: _lerp(cur.slopeLeft, tgt.slopeLeft, alpha),
    slopeRight: _lerp(cur.slopeRight, tgt.slopeRight, alpha),
    pupilX: _lerp(cur.pupilX, tgt.pupilX, pupilAlpha),
    pupilY: _lerp(cur.pupilY, tgt.pupilY, pupilAlpha),
    pupilScale: _lerp(cur.pupilScale, tgt.pupilScale, alpha),
    colorR: _lerp(cur.colorR, tgt.colorR, colorAlpha),
    colorG: _lerp(cur.colorG, tgt.colorG, colorAlpha),
    colorB: _lerp(cur.colorB, tgt.colorB, colorAlpha),
    glowAlpha: _lerp(cur.glowAlpha, tgt.glowAlpha, alpha),
  );
}

FaceParams lerpFace(FaceParams cur, FaceParams tgt,
    {double alpha = 0.25, double colorAlpha = 0.15, double pupilAlpha = 0.35}) {
  return FaceParams(
    leftEye: lerpEye(cur.leftEye, tgt.leftEye,
        alpha: alpha, colorAlpha: colorAlpha, pupilAlpha: pupilAlpha),
    rightEye: lerpEye(cur.rightEye, tgt.rightEye,
        alpha: alpha, colorAlpha: colorAlpha, pupilAlpha: pupilAlpha),
    mouthOpen: _lerp(cur.mouthOpen, tgt.mouthOpen, alpha),
    mouthWidth: _lerp(cur.mouthWidth, tgt.mouthWidth, alpha),
    mouthStyle: tgt.mouthStyle,
  );
}
