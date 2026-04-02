import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../bluetooth/bluetooth_manager.dart';
import '../../core/orchestrator/orchestrator.dart';
import '../../core/state/bot_state.dart';
import '../../core/state/shared_state.dart';
import '../widgets/car_control_overlay.dart';
import 'face_animator.dart';
import 'face_expressions.dart';
import 'face_painter.dart';
import 'face_state.dart';

/// Full-screen animated face -- the bot's personality on screen.
/// Tap to wake (tap-to-talk), long-press to navigate to parent dashboard.
///
/// When [embedded] is true, renders at a compact size without gesture
/// detectors (the parent widget handles taps). Used by HomeScreen.
class FaceScreen extends ConsumerStatefulWidget {
  final bool embedded;
  final String? startActivityId;

  const FaceScreen({super.key, this.embedded = false, this.startActivityId});

  @override
  ConsumerState<FaceScreen> createState() => _FaceScreenState();
}

class _FaceScreenState extends ConsumerState<FaceScreen>
    with SingleTickerProviderStateMixin, WidgetsBindingObserver {
  late Ticker _ticker;
  Duration _lastTick = Duration.zero;

  // Bluetooth manager (persists across rebuilds)
  final BluetoothManager _btManager = BluetoothManager();

  // Animation state
  final AnimState _anim = AnimState();
  FaceParams _current = buildExpression('ready');
  FaceParams _target = buildExpression('ready');
  List<Sparkle> _sparkles = [];
  String _prevState = 'ready';

  // Animation outputs
  double _blinkMult = 1.0;
  double _listenPhase = 0.0;
  int _dotPhase = 0;
  int _spinnerIdx = 0;
  double _sleepZPhase = 0.0;

  // UI state
  final bool _showCarControls = true;

  @override
  void initState() {
    super.initState();

    if (!widget.embedded) {
      // Lock to landscape for the full-screen face
      SystemChrome.setPreferredOrientations([
        DeviceOrientation.landscapeLeft,
        DeviceOrientation.landscapeRight,
      ]);
      SystemChrome.setEnabledSystemUIMode(SystemUiMode.immersiveSticky);
    }

    WidgetsBinding.instance.addObserver(this);

    _ticker = createTicker(_onTick);
    _ticker.start();

    if (!widget.embedded) {
      // Initialize orchestrator only in full-screen mode
      WidgetsBinding.instance.addPostFrameCallback((_) {
        final orchestrator = ref.read(orchestratorProvider);
        orchestrator.btManager = _btManager;
        orchestrator.init().then((_) {
          // If launched with an activity ID, start that activity
          if (widget.startActivityId != null) {
            orchestrator.startActivityById(widget.startActivityId!);
          }
        });
      });
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _ticker.dispose();
    if (!widget.embedded) {
      _btManager.disconnect();
      // Restore portrait for home screen when leaving
      SystemChrome.setPreferredOrientations([
        DeviceOrientation.portraitUp,
        DeviceOrientation.portraitDown,
      ]);
      SystemChrome.setEnabledSystemUIMode(SystemUiMode.edgeToEdge);
    }
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (widget.embedded) return;
    if (state == AppLifecycleState.paused ||
        state == AppLifecycleState.inactive) {
      // Stop continuous mode when app goes to background
      final orchestrator = ref.read(orchestratorProvider);
      if (orchestrator.continuousMode) {
        orchestrator.stopContinuousMode();
      }
    }
  }

  void _onTick(Duration elapsed) {
    final dt = _lastTick == Duration.zero
        ? 0.016
        : (elapsed - _lastTick).inMicroseconds / 1e6;
    _lastTick = elapsed;

    // Get current bot state
    final state = ref.read(sharedStateProvider);
    final stateStr = state.botState.name;

    // State change -> update target expression
    if (stateStr != _prevState) {
      _target = buildExpression(stateStr);
      _prevState = stateStr;
    }

    // -- Run animation updaters --
    _blinkMult = updateBlink(_anim, dt);
    final breathScale = updateBreathing(_anim, dt);

    // Gaze: idle look-around in ready state, micro-drift otherwise
    double gazeX = 0, gazeY = 0;
    if (stateStr == 'ready') {
      final look = updateIdleLook(_anim, dt);
      gazeX = look[0];
      gazeY = look[1];
      if (!_anim.lookActive) {
        updateGaze(_anim, dt);
        gazeX += _anim.gazeTargetX;
        gazeY += _anim.gazeTargetY;
      }
    }
    updateSaccade(_anim, dt);
    gazeX += _anim.saccadeOx;
    gazeY += _anim.saccadeOy;

    // Apply gaze to target
    _target.leftEye.pupilX = _target.leftEye.pupilX * 0.5 + gazeX * 0.5;
    _target.leftEye.pupilY = _target.leftEye.pupilY * 0.5 + gazeY * 0.5;
    _target.rightEye.pupilX = _target.rightEye.pupilX * 0.5 + gazeX * 0.5;
    _target.rightEye.pupilY = _target.rightEye.pupilY * 0.5 + gazeY * 0.5;

    // Apply breathing scale
    _target.leftEye.height *= breathScale;
    _target.leftEye.width *= (1.0 + (breathScale - 1.0) * 0.3);
    _target.rightEye.height *= breathScale;
    _target.rightEye.width *= (1.0 + (breathScale - 1.0) * 0.3);

    // State-specific updates
    if (stateStr == 'listening') {
      _listenPhase = updateListenPulse(_anim, dt);
    }
    if (stateStr == 'processing') {
      _dotPhase = updateDots(_anim, dt);
    }
    if (stateStr == 'speaking') {
      final mouthOpen = updateMouth(_anim, dt);
      _target.mouthOpen = mouthOpen;
    }
    if (stateStr == 'loading') {
      _spinnerIdx = updateSpinner(_anim, dt);
    }
    if (stateStr == 'sleeping') {
      _sleepZPhase = updateSleepZ(_anim, dt);
    }

    // Sparkles in ready/speaking states
    if (stateStr == 'ready' || stateStr == 'speaking') {
      _sparkles = updateSparkles(_sparkles, dt, 240, 150);
    } else {
      _sparkles = updateSparkles(_sparkles, dt, 240, 150, spawnChance: 0);
    }

    // Lerp current -> target
    _current = lerpFace(_current, _target);

    // Re-build target from expression (so breathing/gaze don't accumulate)
    _target = buildExpression(stateStr);

    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(sharedStateProvider);

    // Embedded mode: compact face widget, no gestures
    if (widget.embedded) {
      return CustomPaint(
        painter: FacePainter(
          face: _current,
          blinkMult: _blinkMult,
          isError: state.botState == BotState.error,
          sparkles: _sparkles,
          listenPhase: _listenPhase,
          dotPhase: _dotPhase,
          spinnerIdx: _spinnerIdx,
          sleepZPhase: _sleepZPhase,
          botState: state.botState.name,
        ),
        size: Size.infinite,
      );
    }

    // Full-screen mode
    return Stack(
      children: [
        // Face (full screen, tappable)
        GestureDetector(
          onTap: _onTap,
          onDoubleTap: _onDoubleTap,
          onLongPress: _onLongPress,
          child: Container(
            color: Colors.black,
            child: CustomPaint(
              painter: FacePainter(
                face: _current,
                blinkMult: _blinkMult,
                isError: state.botState == BotState.error,
                sparkles: _sparkles,
                listenPhase: _listenPhase,
                dotPhase: _dotPhase,
                spinnerIdx: _spinnerIdx,
                sleepZPhase: _sleepZPhase,
                botState: state.botState.name,
              ),
              size: Size.infinite,
            ),
          ),
        ),

        // Car control overlay (bottom-right)
        if (_showCarControls)
          CarControlOverlay(btManager: _btManager),

        // Continuous mode indicator (top-left)
        if (state.continuousMode)
          Positioned(
            top: 12,
            left: 12,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: Colors.green.withValues(alpha: 0.25),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(
                  color: Colors.greenAccent.withValues(alpha: 0.5),
                  width: 1,
                ),
              ),
              child: const Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.mic, color: Colors.greenAccent, size: 14),
                  SizedBox(width: 4),
                  Text(
                    'Continuous',
                    style: TextStyle(
                      color: Colors.greenAccent,
                      fontSize: 11,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            ),
          ),

        // Back button to home (top-right)
        Positioned(
          top: 12,
          right: 12,
          child: GestureDetector(
            onTap: () {
              HapticFeedback.lightImpact();
              context.go('/home');
            },
            child: Container(
              padding: const EdgeInsets.all(6),
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Icon(
                Icons.close,
                color: Colors.white.withValues(alpha: 0.5),
                size: 18,
              ),
            ),
          ),
        ),

        // Transcript overlay (bottom-left, shows what child said + bot response)
        if (state.currentTranscript.isNotEmpty ||
            state.currentResponse.isNotEmpty)
          Positioned(
            bottom: 60,
            left: 10,
            right: 160,
            child: Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (state.currentTranscript.isNotEmpty)
                    Text(
                      state.currentTranscript,
                      style: const TextStyle(color: Colors.white70, fontSize: 11),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                  if (state.currentResponse.isNotEmpty) ...[
                    const SizedBox(height: 2),
                    Text(
                      state.currentResponse,
                      style: const TextStyle(color: Colors.white, fontSize: 11),
                      maxLines: 3,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ],
                ],
              ),
            ),
          ),
      ],
    );
  }

  void _onTap() {
    final state = ref.read(sharedStateProvider);
    final currentState = state.botState;

    if (currentState == BotState.ready || currentState == BotState.sleeping) {
      // Tap-to-talk: use the orchestrator for real voice pipeline
      final orchestrator = ref.read(orchestratorProvider);
      orchestrator.handleTapToTalk();
    } else if (currentState == BotState.speaking) {
      // Tap while speaking -> interrupt
      final orchestrator = ref.read(orchestratorProvider);
      orchestrator.stop();
      ref.read(sharedStateProvider.notifier).setBotState(BotState.ready);
    }
  }

  void _onDoubleTap() {
    HapticFeedback.mediumImpact();
    final orchestrator = ref.read(orchestratorProvider);
    // Stop everything and go back to wake word detection mode
    orchestrator.stopAndResumeWakeWord();
    ref.read(sharedStateProvider.notifier).setBotState(BotState.ready);
  }

  void _onLongPress() {
    // Navigate to parent dashboard via PIN gate
    HapticFeedback.mediumImpact();
    context.push('/pin');
  }
}
