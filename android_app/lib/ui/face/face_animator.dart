import 'dart:math';
import 'face_state.dart';

final _rng = Random();

// ─── Pre-defined look sequences (same as Pi) ──────────────────────
const List<List<List<double>>> lookSequences = [
  [[-0.7, 0.0], [0.0, 0.0], [0.7, 0.0], [0.0, 0.0]],
  [[-0.6, -0.5], [0.6, -0.5], [0.0, 0.0]],
  [[0.7, 0.0], [0.0, -0.5], [-0.7, 0.0], [0.0, 0.0]],
  [[-0.5, -0.4], [0.0, 0.0], [0.5, -0.4], [0.0, 0.0]],
  [[-0.7, 0.0], [0.0, -0.5], [0.7, 0.0], [0.0, 0.0]],
  [[0.0, -0.5], [-0.6, -0.2], [0.0, -0.5], [0.6, -0.2], [0.0, 0.0]],
];

double _easeInOut(double t) => (1.0 - cos(t * pi)) / 2.0;

/// Update blink — returns height multiplier 0..1.
double updateBlink(AnimState a, double dt) {
  if (a.blinkPhase >= 0) {
    a.blinkPhase += dt * 2.8;
    if (a.blinkPhase >= 1.0) {
      a.blinkPhase = -1.0;
      a.blinkTimer = 2.0 + _rng.nextDouble() * 3.0;
      return 1.0;
    }
    return cos(a.blinkPhase * pi).abs();
  } else {
    a.blinkTimer -= dt;
    if (a.blinkTimer <= 0) a.blinkPhase = 0.0;
    return 1.0;
  }
}

/// Update breathing — returns scale multiplier ~1.0 ± 0.03.
double updateBreathing(AnimState a, double dt) {
  a.breathPhase += dt * 1.5;
  if (a.breathPhase > 2 * pi) a.breathPhase -= 2 * pi;
  return 1.0 + 0.03 * sin(a.breathPhase);
}

/// Update small random gaze drift.
void updateGaze(AnimState a, double dt) {
  a.gazeTimer -= dt;
  if (a.gazeTimer <= 0) {
    a.gazeTargetX = -0.15 + _rng.nextDouble() * 0.3;
    a.gazeTargetY = -0.1 + _rng.nextDouble() * 0.2;
    a.gazeTimer = 2.0 + _rng.nextDouble() * 3.0;
  }
}

/// Update micro-saccade.
void updateSaccade(AnimState a, double dt) {
  a.saccadeTimer -= dt;
  if (a.saccadeTimer <= 0) {
    a.saccadeOx = -0.05 + _rng.nextDouble() * 0.1;
    a.saccadeOy = -0.03 + _rng.nextDouble() * 0.06;
    a.saccadeTimer = 1.0 + _rng.nextDouble() * 2.0;
  } else {
    a.saccadeOx *= 0.5;
    a.saccadeOy *= 0.5;
  }
}

/// Update idle look-around. Returns (pupilX, pupilY).
List<double> updateIdleLook(AnimState a, double dt) {
  if (!a.lookActive) {
    a.lookCooldown -= dt;
    if (a.lookCooldown <= 0) {
      a.lookActive = true;
      a.lookPhase = 0.0;
      a.lookSeq = lookSequences[_rng.nextInt(lookSequences.length)];
      a.lookFrom = [0.0, 0.0];
    }
    return [0.0, 0.0];
  }

  final seq = a.lookSeq;
  final n = seq.length;
  const moveTime = 0.6;
  const holdTime = 0.4;
  const stepTime = moveTime + holdTime;
  final totalTime = n * stepTime;

  a.lookPhase += dt;
  if (a.lookPhase >= totalTime) {
    a.lookActive = false;
    a.lookCooldown = 3.0 + _rng.nextDouble() * 5.0;
    a.lookPhase = 0.0;
    return [0.0, 0.0];
  }

  final stepIdx = min((a.lookPhase / stepTime).floor(), n - 1);
  final stepLocal = a.lookPhase - stepIdx * stepTime;
  final fromPos = a.lookFrom;
  final toPos = seq[stepIdx];

  if (stepLocal < moveTime) {
    final t = _easeInOut(stepLocal / moveTime);
    return [
      fromPos[0] + (toPos[0] - fromPos[0]) * t,
      fromPos[1] + (toPos[1] - fromPos[1]) * t,
    ];
  } else {
    a.lookFrom = List.from(toPos);
    return List.from(toPos);
  }
}

/// Update listening pulse phase.
double updateListenPulse(AnimState a, double dt) {
  a.listenPhase += dt * 1.8;
  return a.listenPhase;
}

/// Update processing dots. Returns phase 0-3.
int updateDots(AnimState a, double dt) {
  a.dotTimer += dt;
  if (a.dotTimer >= 0.5) {
    a.dotTimer = 0.0;
    a.dotPhase = (a.dotPhase + 1) % 4;
  }
  return a.dotPhase;
}

/// Update speaking mouth. Returns openness 0..1.
double updateMouth(AnimState a, double dt) {
  a.mouthPhase += dt * 8.0;
  return 0.3 + 0.7 * sin(a.mouthPhase).abs();
}

/// Update loading spinner. Returns index 0-7.
int updateSpinner(AnimState a, double dt) {
  a.spinnerTimer += dt;
  if (a.spinnerTimer >= 0.3) {
    a.spinnerTimer = 0.0;
    a.spinnerIdx = (a.spinnerIdx + 1) % 8;
  }
  return a.spinnerIdx;
}

/// Update sparkle particles.
List<Sparkle> updateSparkles(
    List<Sparkle> sparkles, double dt, double cx, double cy,
    {double spawnChance = 0.12}) {
  final alive = <Sparkle>[];
  for (final s in sparkles) {
    s.life -= dt;
    if (s.life > 0) alive.add(s);
  }
  if (_rng.nextDouble() < spawnChance && alive.length < 8) {
    final angle = _rng.nextDouble() * 2 * pi;
    final dist = 75 + _rng.nextDouble() * 40;
    alive.add(Sparkle(
      x: cx + dist * cos(angle),
      y: cy + dist * sin(angle),
      life: 0.8,
      maxLife: 0.8,
      size: 2.0 + _rng.nextDouble() * 2.0,
    ));
  }
  return alive;
}

/// Update sleep Z phase.
double updateSleepZ(AnimState a, double dt) {
  a.zPhase += dt * 0.8;
  if (a.zPhase > 2 * pi) a.zPhase -= 2 * pi;
  return a.zPhase;
}
