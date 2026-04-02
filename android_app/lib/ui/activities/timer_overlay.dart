import 'dart:async';
import 'package:flutter/material.dart';

/// Timer overlay widget for timed activities.
/// Shows countdown, pause/resume, and a progress bar.
class TimerOverlay extends StatefulWidget {
  /// Total duration for the timer.
  final Duration totalDuration;

  /// Called when the timer finishes.
  final VoidCallback? onComplete;

  /// Called when dismissed.
  final VoidCallback? onDismiss;

  const TimerOverlay({
    super.key,
    required this.totalDuration,
    this.onComplete,
    this.onDismiss,
  });

  @override
  State<TimerOverlay> createState() => _TimerOverlayState();
}

class _TimerOverlayState extends State<TimerOverlay> {
  late Duration _remaining;
  Timer? _timer;
  bool _paused = false;
  bool _done = false;

  @override
  void initState() {
    super.initState();
    _remaining = widget.totalDuration;
    _startTimer();
  }

  void _startTimer() {
    _timer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (_paused) return;
      setState(() {
        _remaining -= const Duration(seconds: 1);
        if (_remaining <= Duration.zero) {
          _remaining = Duration.zero;
          _done = true;
          _timer?.cancel();
          widget.onComplete?.call();
        }
      });
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  void _togglePause() {
    setState(() => _paused = !_paused);
  }

  String _formatDuration(Duration d) {
    final m = d.inMinutes.remainder(60).toString().padLeft(2, '0');
    final s = d.inSeconds.remainder(60).toString().padLeft(2, '0');
    return '$m:$s';
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final elapsed = widget.totalDuration - _remaining;
    final progress = widget.totalDuration.inSeconds > 0
        ? elapsed.inSeconds / widget.totalDuration.inSeconds
        : 0.0;

    return Positioned(
      top: 16,
      right: 16,
      child: Material(
        elevation: 4,
        borderRadius: BorderRadius.circular(16),
        child: Container(
          width: 160,
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: theme.colorScheme.surface,
            borderRadius: BorderRadius.circular(16),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Timer display
              Text(
                _done ? 'Done!' : _formatDuration(_remaining),
                style: theme.textTheme.headlineSmall?.copyWith(
                  fontWeight: FontWeight.bold,
                  fontFeatures: [const FontFeature.tabularFigures()],
                  color: _done
                      ? theme.colorScheme.primary
                      : theme.colorScheme.onSurface,
                ),
              ),
              const SizedBox(height: 8),

              // Progress bar
              ClipRRect(
                borderRadius: BorderRadius.circular(4),
                child: LinearProgressIndicator(
                  value: progress.clamp(0.0, 1.0),
                  minHeight: 4,
                  backgroundColor: theme.colorScheme.surfaceContainerHighest,
                ),
              ),
              const SizedBox(height: 10),

              if (_done)
                Text(
                  'Take a break!',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.primary,
                    fontWeight: FontWeight.w500,
                  ),
                )
              else
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    // Pause / Resume
                    IconButton(
                      onPressed: _togglePause,
                      icon: Icon(
                        _paused ? Icons.play_arrow : Icons.pause,
                        size: 20,
                      ),
                      style: IconButton.styleFrom(
                        minimumSize: const Size(36, 36),
                        padding: EdgeInsets.zero,
                      ),
                    ),
                    // Dismiss
                    IconButton(
                      onPressed: () {
                        _timer?.cancel();
                        widget.onDismiss?.call();
                      },
                      icon: const Icon(Icons.close, size: 20),
                      style: IconButton.styleFrom(
                        minimumSize: const Size(36, 36),
                        padding: EdgeInsets.zero,
                      ),
                    ),
                  ],
                ),
            ],
          ),
        ),
      ),
    );
  }
}
