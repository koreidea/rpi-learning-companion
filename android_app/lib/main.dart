import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'ui/face/face_screen.dart';
import 'ui/home/home_screen.dart';
import 'ui/parent/conversation_history_screen.dart';
import 'ui/parent/dashboard_screen.dart';
import 'ui/parent/pin_gate_screen.dart';
import 'ui/parent/settings_screen.dart';
import 'ui/setup/setup_screen.dart';
import 'ui/skills/skill_detail_screen.dart';
import 'ui/progress/progress_screen.dart';

/// Whether this is the first run (no API key configured and setup not done).
late final bool _isFirstRun;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Check first-run status before building the app
  final prefs = await SharedPreferences.getInstance();
  final hasApiKey = (prefs.getString('api_key_openai') ?? '').isNotEmpty ||
      (prefs.getString('api_key_gemini') ?? '').isNotEmpty ||
      (prefs.getString('api_key_claude') ?? '').isNotEmpty;
  final setupDone = prefs.getBool('first_run_done') ?? false;
  _isFirstRun = !hasApiKey && !setupDone;

  // Start in portrait for the home screen
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);

  // Show system UI for the home screen (face screen hides it)
  SystemChrome.setEnabledSystemUIMode(SystemUiMode.edgeToEdge);

  runApp(const ProviderScope(child: CompanionApp()));
}

/// App router configuration.
final _router = GoRouter(
  initialLocation: _isFirstRun ? '/setup' : '/home',
  routes: [
    GoRoute(
      path: '/home',
      builder: (context, state) => const HomeScreen(),
    ),
    GoRoute(
      path: '/face',
      builder: (context, state) {
        final activityId = state.uri.queryParameters['activityId'];
        return FaceScreen(startActivityId: activityId);
      },
    ),
    // Legacy route: redirect '/' to '/home'
    GoRoute(
      path: '/',
      redirect: (context, state) => '/home',
    ),
    GoRoute(
      path: '/pin',
      builder: (context, state) => const PinGateScreen(),
    ),
    GoRoute(
      path: '/dashboard',
      builder: (context, state) => const DashboardScreen(),
    ),
    GoRoute(
      path: '/settings',
      builder: (context, state) => const SettingsScreen(),
    ),
    GoRoute(
      path: '/setup',
      builder: (context, state) => const SetupScreen(),
    ),
    GoRoute(
      path: '/history',
      builder: (context, state) => const ConversationHistoryScreen(),
    ),
    GoRoute(
      path: '/skill/:id',
      builder: (context, state) => SkillDetailScreen(
        skillIdName: state.pathParameters['id']!,
      ),
    ),
    GoRoute(
      path: '/progress',
      builder: (context, state) => const ProgressScreen(),
    ),
  ],
);

class CompanionApp extends StatelessWidget {
  const CompanionApp({super.key});

  @override
  Widget build(BuildContext context) {
    // Define the Material 3 color scheme
    const seedColor = Color(0xFF5C6BC0); // Deep indigo/purple

    return MaterialApp.router(
      title: 'HAI ROBO',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: seedColor,
          brightness: Brightness.light,
        ),
        fontFamily: 'Roboto',
        cardTheme: CardThemeData(
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
        navigationBarTheme: NavigationBarThemeData(
          indicatorColor: seedColor.withValues(alpha: 0.15),
          labelBehavior: NavigationDestinationLabelBehavior.alwaysShow,
        ),
      ),
      routerConfig: _router,
    );
  }
}
