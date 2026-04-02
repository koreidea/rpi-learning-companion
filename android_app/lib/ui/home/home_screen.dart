import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../core/state/bot_state.dart';
import '../../core/state/shared_state.dart';
import '../../models/skill.dart';
import '../face/face_animator.dart';
import '../face/face_expressions.dart';
import '../face/face_painter.dart';
import '../face/face_state.dart';
import '../progress/progress_screen.dart';
import 'skill_grid.dart';
import 'skill_icon_map.dart';

/// Main hub screen with bottom navigation: Home, Activities, Progress, Settings.
class HomeScreen extends ConsumerStatefulWidget {
  const HomeScreen({super.key});

  @override
  ConsumerState<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends ConsumerState<HomeScreen> {
  int _currentTab = 0;

  @override
  void initState() {
    super.initState();
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
      DeviceOrientation.portraitDown,
    ]);
    SystemChrome.setEnabledSystemUIMode(SystemUiMode.edgeToEdge);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _currentTab,
        children: const [
          _HomeTab(),
          SkillGrid(),
          ProgressScreen(),
          _SettingsTab(),
        ],
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _currentTab,
        onDestinationSelected: (i) => setState(() => _currentTab = i),
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.home_outlined),
            selectedIcon: Icon(Icons.home),
            label: 'Home',
          ),
          NavigationDestination(
            icon: Icon(Icons.grid_view_outlined),
            selectedIcon: Icon(Icons.grid_view),
            label: 'Activities',
          ),
          NavigationDestination(
            icon: Icon(Icons.insights_outlined),
            selectedIcon: Icon(Icons.insights),
            label: 'Progress',
          ),
          NavigationDestination(
            icon: Icon(Icons.settings_outlined),
            selectedIcon: Icon(Icons.settings),
            label: 'Settings',
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Home Tab
// ---------------------------------------------------------------------------

class _HomeTab extends ConsumerStatefulWidget {
  const _HomeTab();

  @override
  ConsumerState<_HomeTab> createState() => _HomeTabState();
}

class _HomeTabState extends ConsumerState<_HomeTab>
    with SingleTickerProviderStateMixin {
  late Ticker _ticker;
  Duration _lastTick = Duration.zero;

  final AnimState _anim = AnimState();
  FaceParams _current = buildExpression('ready');
  FaceParams _target = buildExpression('ready');
  List<Sparkle> _sparkles = [];
  String _prevState = 'ready';
  double _blinkMult = 1.0;
  double _listenPhase = 0.0;
  int _dotPhase = 0;
  int _spinnerIdx = 0;
  double _sleepZPhase = 0.0;

  @override
  void initState() {
    super.initState();
    _ticker = createTicker(_onTick);
    _ticker.start();
  }

  @override
  void dispose() {
    _ticker.dispose();
    super.dispose();
  }

  void _onTick(Duration elapsed) {
    final dt = _lastTick == Duration.zero
        ? 0.016
        : (elapsed - _lastTick).inMicroseconds / 1e6;
    _lastTick = elapsed;

    final state = ref.read(sharedStateProvider);
    final stateStr = state.botState.name;

    if (stateStr != _prevState) {
      _target = buildExpression(stateStr);
      _prevState = stateStr;
    }

    _blinkMult = updateBlink(_anim, dt);
    final breathScale = updateBreathing(_anim, dt);

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

    _target.leftEye.pupilX = _target.leftEye.pupilX * 0.5 + gazeX * 0.5;
    _target.leftEye.pupilY = _target.leftEye.pupilY * 0.5 + gazeY * 0.5;
    _target.rightEye.pupilX = _target.rightEye.pupilX * 0.5 + gazeX * 0.5;
    _target.rightEye.pupilY = _target.rightEye.pupilY * 0.5 + gazeY * 0.5;

    _target.leftEye.height *= breathScale;
    _target.leftEye.width *= (1.0 + (breathScale - 1.0) * 0.3);
    _target.rightEye.height *= breathScale;
    _target.rightEye.width *= (1.0 + (breathScale - 1.0) * 0.3);

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

    if (stateStr == 'ready' || stateStr == 'speaking') {
      _sparkles = updateSparkles(_sparkles, dt, 240, 150);
    } else {
      _sparkles = updateSparkles(_sparkles, dt, 240, 150, spawnChance: 0);
    }

    _current = lerpFace(_current, _target);
    _target = buildExpression(stateStr);

    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(sharedStateProvider);
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return SafeArea(
      child: CustomScrollView(
        slivers: [
          // Compact face hero
          SliverToBoxAdapter(
            child: GestureDetector(
              onTap: () {
                HapticFeedback.lightImpact();
                // Navigate to full-screen face for voice interaction
                context.push('/face');
              },
              child: Container(
                height: 140,
                margin: const EdgeInsets.fromLTRB(16, 12, 16, 0),
                decoration: BoxDecoration(
                  color: Colors.black,
                  borderRadius: BorderRadius.circular(20),
                ),
                clipBehavior: Clip.antiAlias,
                child: Stack(
                  children: [
                    CustomPaint(
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
                    Positioned(
                      bottom: 8,
                      right: 12,
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 8, vertical: 3),
                        decoration: BoxDecoration(
                          color: Colors.white.withValues(alpha: 0.15),
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(Icons.mic, size: 14,
                                color: Colors.white.withValues(alpha: 0.7)),
                            const SizedBox(width: 4),
                            Text(
                              'Tap to talk',
                              style: TextStyle(
                                fontSize: 11,
                                color: Colors.white.withValues(alpha: 0.7),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),

          // Greeting
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(20, 20, 20, 4),
              child: Text(
                'Hi there!',
                style: theme.textTheme.headlineMedium?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: colorScheme.onSurface,
                ),
              ),
            ),
          ),
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(20, 0, 20, 16),
              child: Text(
                'What would you like to learn today?',
                style: theme.textTheme.bodyLarge?.copyWith(
                  color: colorScheme.onSurfaceVariant,
                ),
              ),
            ),
          ),

          // Daily Pick card
          SliverToBoxAdapter(child: _DailyPickCard()),

          // Quick actions
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(20, 20, 20, 8),
              child: Text(
                'Quick Actions',
                style: theme.textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ),
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Row(
                children: [
                  Expanded(
                    child: _QuickActionCard(
                      icon: Icons.mic,
                      label: 'Talk',
                      color: const Color(0xFF5C6BC0),
                      onTap: () => context.push('/face'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _QuickActionCard(
                      icon: Icons.extension,
                      label: 'Play a Game',
                      color: const Color(0xFFFF7043),
                      onTap: () => context.push('/face'),
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Recent Activity
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(20, 24, 20, 8),
              child: Text(
                'Recent Activity',
                style: theme.textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ),
          SliverToBoxAdapter(child: _RecentActivitySection()),

          const SliverToBoxAdapter(child: SizedBox(height: 24)),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Daily Pick Card
// ---------------------------------------------------------------------------

class _DailyPickCard extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    // Pick a deterministic daily skill based on day of year
    final dayOfYear = DateTime.now().difference(
      DateTime(DateTime.now().year),
    ).inDays;
    final allSkills = SkillRegistry.all;
    final dailySkill = allSkills[dayOfYear % allSkills.length];
    final iconData = skillIconData(dailySkill.icon);

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: Card(
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
        color: Color(dailySkill.colorValue).withValues(alpha: 0.08),
        child: InkWell(
          borderRadius: BorderRadius.circular(16),
          onTap: () => context.push('/skill/${dailySkill.id.name}'),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                Container(
                  width: 52,
                  height: 52,
                  decoration: BoxDecoration(
                    color: Color(dailySkill.colorValue).withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: Icon(
                    iconData,
                    color: Color(dailySkill.colorValue),
                    size: 28,
                  ),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 8, vertical: 2),
                            decoration: BoxDecoration(
                              color: Color(dailySkill.colorValue)
                                  .withValues(alpha: 0.15),
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Text(
                              'Daily Pick',
                              style: TextStyle(
                                fontSize: 11,
                                fontWeight: FontWeight.w600,
                                color: Color(dailySkill.colorValue),
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 4),
                      Text(
                        dailySkill.shortName,
                        style: theme.textTheme.titleSmall?.copyWith(
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      Text(
                        dailySkill.description,
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ],
                  ),
                ),
                Icon(
                  Icons.chevron_right,
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Quick Action Card
// ---------------------------------------------------------------------------

class _QuickActionCard extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;
  final VoidCallback onTap;

  const _QuickActionCard({
    required this.icon,
    required this.label,
    required this.color,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      color: color.withValues(alpha: 0.08),
      child: InkWell(
        borderRadius: BorderRadius.circular(16),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 18, horizontal: 16),
          child: Row(
            children: [
              Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  color: color.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(icon, color: color, size: 22),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  label,
                  style: theme.textTheme.titleSmall?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Recent Activity Section
// ---------------------------------------------------------------------------

class _RecentActivitySection extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(sharedStateProvider);
    final history = state.conversationHistory;
    final theme = Theme.of(context);

    if (history.isEmpty) {
      return Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16),
        child: Card(
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Center(
              child: Column(
                children: [
                  Icon(
                    Icons.chat_bubble_outline,
                    size: 40,
                    color: theme.colorScheme.onSurfaceVariant
                        .withValues(alpha: 0.4),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'No conversations yet',
                    style: theme.textTheme.bodyMedium?.copyWith(
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Tap the face above to start talking!',
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: theme.colorScheme.onSurfaceVariant
                          .withValues(alpha: 0.7),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      );
    }

    final recent = history.reversed.take(5).toList();
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: Card(
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
        child: Column(
          children: [
            for (int i = 0; i < recent.length; i++) ...[
              ListTile(
                dense: true,
                leading: CircleAvatar(
                  radius: 16,
                  backgroundColor:
                      theme.colorScheme.primary.withValues(alpha: 0.1),
                  child: Icon(
                    Icons.chat_bubble_outline,
                    size: 16,
                    color: theme.colorScheme.primary,
                  ),
                ),
                title: Text(
                  recent[i].userText,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: theme.textTheme.bodyMedium,
                ),
                subtitle: Text(
                  _formatTime(recent[i].timestamp),
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                ),
              ),
              if (i < recent.length - 1)
                Divider(height: 1, indent: 56, endIndent: 16),
            ],
          ],
        ),
      ),
    );
  }

  String _formatTime(DateTime dt) {
    final now = DateTime.now();
    final diff = now.difference(dt);
    if (diff.inMinutes < 1) return 'Just now';
    if (diff.inMinutes < 60) return '${diff.inMinutes}m ago';
    if (diff.inHours < 24) return '${diff.inHours}h ago';
    return '${diff.inDays}d ago';
  }
}

// ---------------------------------------------------------------------------
// Settings Tab (redirects to existing settings)
// ---------------------------------------------------------------------------

class _SettingsTab extends ConsumerWidget {
  const _SettingsTab();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final theme = Theme.of(context);
    final state = ref.watch(sharedStateProvider);

    return SafeArea(
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(4, 8, 4, 16),
            child: Text(
              'Settings',
              style: theme.textTheme.headlineMedium?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
          _SettingsTile(
            icon: Icons.tune,
            label: 'App Settings',
            subtitle: 'API keys, language, audio',
            onTap: () => context.push('/pin'),
          ),
          _SettingsTile(
            icon: Icons.history,
            label: 'Conversation History',
            subtitle: '${state.conversationHistory.length} conversations',
            onTap: () => context.push('/history'),
          ),
          _SettingsTile(
            icon: Icons.bluetooth,
            label: 'Bluetooth',
            subtitle: state.carConnected ? 'Connected' : 'Not connected',
            onTap: () => context.push('/pin'),
          ),
          _SettingsTile(
            icon: Icons.info_outline,
            label: 'About',
            subtitle: 'HAI ROBO v1.0.0',
            onTap: () {},
          ),
        ],
      ),
    );
  }
}

class _SettingsTile extends StatelessWidget {
  final IconData icon;
  final String label;
  final String subtitle;
  final VoidCallback onTap;

  const _SettingsTile({
    required this.icon,
    required this.label,
    required this.subtitle,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      elevation: 0,
      margin: const EdgeInsets.only(bottom: 8),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: ListTile(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        leading: Container(
          width: 40,
          height: 40,
          decoration: BoxDecoration(
            color: theme.colorScheme.primary.withValues(alpha: 0.08),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Icon(icon, color: theme.colorScheme.primary, size: 22),
        ),
        title: Text(label),
        subtitle: Text(
          subtitle,
          style: TextStyle(color: theme.colorScheme.onSurfaceVariant),
        ),
        trailing: const Icon(Icons.chevron_right),
        onTap: onTap,
      ),
    );
  }
}
