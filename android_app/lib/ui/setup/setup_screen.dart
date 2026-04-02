import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../core/config/config_manager.dart';
import '../../core/orchestrator/orchestrator.dart';

/// First-run setup wizard. Guides parents through initial configuration.
class SetupScreen extends ConsumerStatefulWidget {
  const SetupScreen({super.key});

  @override
  ConsumerState<SetupScreen> createState() => _SetupScreenState();
}

class _SetupScreenState extends ConsumerState<SetupScreen> {
  final PageController _pageController = PageController();
  int _currentPage = 0;

  // Config
  late ConfigManager _config;
  bool _initialized = false;

  // Step 2: API keys
  final _openaiController = TextEditingController();
  final _geminiController = TextEditingController();
  final _claudeController = TextEditingController();
  String _provider = 'openai';

  // Step 3: Language
  String _language = 'en';

  // Step 4: Bluetooth (informational only)
  bool _setupBluetooth = false;

  @override
  void initState() {
    super.initState();
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
      DeviceOrientation.portraitDown,
    ]);
    _initConfig();
  }

  Future<void> _initConfig() async {
    final orchestrator = ref.read(orchestratorProvider);
    await orchestrator.init();
    _config = orchestrator.config;
    setState(() {
      _openaiController.text = _config.openaiKey;
      _geminiController.text = _config.geminiKey;
      _claudeController.text = _config.claudeKey;
      _provider = _config.provider;
      _language = _config.language;
      _initialized = true;
    });
  }

  @override
  void dispose() {
    _pageController.dispose();
    _openaiController.dispose();
    _geminiController.dispose();
    _claudeController.dispose();
    super.dispose();
  }

  void _nextPage() {
    if (_currentPage < 4) {
      _pageController.nextPage(
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeInOut,
      );
    }
  }

  void _prevPage() {
    if (_currentPage > 0) {
      _pageController.previousPage(
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeInOut,
      );
    }
  }

  Future<void> _finish() async {
    // Save all config
    await _config.setOpenaiKey(_openaiController.text.trim());
    await _config.setGeminiKey(_geminiController.text.trim());
    await _config.setClaudeKey(_claudeController.text.trim());
    await _config.setProvider(_provider);
    await _config.setLanguage(_language);
    await _config.setFirstRunDone(true);

    // Update orchestrator
    final orchestrator = ref.read(orchestratorProvider);
    orchestrator.llmRouter.openaiKey = _openaiController.text.trim();
    orchestrator.llmRouter.geminiKey = _geminiController.text.trim();
    orchestrator.llmRouter.claudeKey = _claudeController.text.trim();
    orchestrator.llmRouter.activeProvider = _provider;
    orchestrator.cloudStt.updateApiKey(_openaiController.text.trim());

    if (mounted) {
      context.go('/');
    }
  }

  bool get _hasAnyApiKey =>
      _openaiController.text.trim().isNotEmpty ||
      _geminiController.text.trim().isNotEmpty ||
      _claudeController.text.trim().isNotEmpty;

  @override
  Widget build(BuildContext context) {
    if (!_initialized) {
      return const Scaffold(
        backgroundColor: Color(0xFF1A1A2E),
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      backgroundColor: const Color(0xFF1A1A2E),
      body: SafeArea(
        child: Column(
          children: [
            // Progress indicator
            Padding(
              padding: const EdgeInsets.fromLTRB(24, 16, 24, 0),
              child: Row(
                children: List.generate(5, (i) {
                  return Expanded(
                    child: Container(
                      height: 4,
                      margin: const EdgeInsets.symmetric(horizontal: 2),
                      decoration: BoxDecoration(
                        color: i <= _currentPage
                            ? Colors.orangeAccent
                            : Colors.white12,
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                  );
                }),
              ),
            ),

            // Pages
            Expanded(
              child: PageView(
                controller: _pageController,
                physics: const NeverScrollableScrollPhysics(),
                onPageChanged: (i) => setState(() => _currentPage = i),
                children: [
                  _welcomePage(),
                  _apiKeyPage(),
                  _languagePage(),
                  _bluetoothPage(),
                  _readyPage(),
                ],
              ),
            ),

            // Navigation buttons
            Padding(
              padding: const EdgeInsets.fromLTRB(24, 0, 24, 24),
              child: Row(
                children: [
                  if (_currentPage > 0)
                    TextButton(
                      onPressed: _prevPage,
                      child: const Text(
                        'Back',
                        style: TextStyle(color: Colors.white54),
                      ),
                    ),
                  const Spacer(),
                  if (_currentPage < 4)
                    ElevatedButton(
                      onPressed: _currentPage == 1 && !_hasAnyApiKey
                          ? null
                          : _nextPage,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.orangeAccent,
                        foregroundColor: Colors.black,
                        disabledBackgroundColor: Colors.grey[800],
                        padding: const EdgeInsets.symmetric(
                            horizontal: 32, vertical: 14),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                      child: const Text(
                        'Next',
                        style: TextStyle(fontWeight: FontWeight.w600),
                      ),
                    ),
                  if (_currentPage == 4)
                    ElevatedButton(
                      onPressed: _finish,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.green,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(
                            horizontal: 32, vertical: 14),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                      child: const Text(
                        'Get Started',
                        style: TextStyle(fontWeight: FontWeight.w600),
                      ),
                    ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ---- Step Pages ----

  Widget _welcomePage() {
    return Padding(
      padding: const EdgeInsets.all(32),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            width: 100,
            height: 100,
            decoration: BoxDecoration(
              color: Colors.orangeAccent.withValues(alpha: 0.15),
              shape: BoxShape.circle,
            ),
            child: const Icon(
              Icons.smart_toy_outlined,
              color: Colors.orangeAccent,
              size: 56,
            ),
          ),
          const SizedBox(height: 32),
          const Text(
            'Welcome!',
            style: TextStyle(
              color: Colors.white,
              fontSize: 32,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 12),
          const Text(
            'Let\'s set up your child\'s learning companion. '
            'This will only take a minute.',
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.white54, fontSize: 16, height: 1.5),
          ),
        ],
      ),
    );
  }

  Widget _apiKeyPage() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(32),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const SizedBox(height: 16),
          const Text(
            'API Key Setup',
            style: TextStyle(
              color: Colors.white,
              fontSize: 24,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 8),
          const Text(
            'Enter at least one API key to enable the AI companion. '
            'You can add more later in Settings.',
            style: TextStyle(color: Colors.white54, fontSize: 14, height: 1.4),
          ),
          const SizedBox(height: 24),

          // Provider selector
          const Text('Active Provider:',
              style: TextStyle(color: Colors.white70, fontSize: 13)),
          const SizedBox(height: 8),
          Row(
            children: [
              _setupProviderChip('openai', 'OpenAI', Colors.green),
              const SizedBox(width: 8),
              _setupProviderChip('gemini', 'Gemini', Colors.blue),
              const SizedBox(width: 8),
              _setupProviderChip('claude', 'Claude', Colors.orange),
            ],
          ),
          const SizedBox(height: 20),

          _setupKeyField('OpenAI API Key', _openaiController,
              _provider == 'openai', 'sk-...'),
          const SizedBox(height: 12),
          _setupKeyField('Gemini API Key', _geminiController,
              _provider == 'gemini', 'AIza...'),
          const SizedBox(height: 12),
          _setupKeyField('Claude API Key', _claudeController,
              _provider == 'claude', 'sk-ant-...'),
        ],
      ),
    );
  }

  Widget _languagePage() {
    const languages = [
      ('en', 'English', 'Hello! How are you?'),
      ('hi', 'Hindi', 'Namaste! Kaise ho?'),
      ('te', 'Telugu', 'Namaskaram! Ela unnaru?'),
    ];

    return Padding(
      padding: const EdgeInsets.all(32),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const SizedBox(height: 16),
          const Text(
            'Language',
            style: TextStyle(
              color: Colors.white,
              fontSize: 24,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 8),
          const Text(
            'Choose the primary language for your child. '
            'The bot can understand all three languages regardless of this setting.',
            style: TextStyle(color: Colors.white54, fontSize: 14, height: 1.4),
          ),
          const SizedBox(height: 32),

          ...languages.map((lang) {
            final selected = _language == lang.$1;
            return Padding(
              padding: const EdgeInsets.only(bottom: 12),
              child: GestureDetector(
                onTap: () => setState(() => _language = lang.$1),
                child: Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: selected
                        ? Colors.orangeAccent.withValues(alpha: 0.1)
                        : Colors.white.withValues(alpha: 0.05),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: selected ? Colors.orangeAccent : Colors.transparent,
                      width: 2,
                    ),
                  ),
                  child: Row(
                    children: [
                      Container(
                        width: 20,
                        height: 20,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: selected ? Colors.orangeAccent : Colors.transparent,
                          border: Border.all(
                            color: selected ? Colors.orangeAccent : Colors.white38,
                            width: 2,
                          ),
                        ),
                        child: selected
                            ? const Icon(Icons.check, size: 14, color: Colors.black)
                            : null,
                      ),
                      const SizedBox(width: 14),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            lang.$2,
                            style: TextStyle(
                              color: selected ? Colors.white : Colors.white70,
                              fontSize: 16,
                              fontWeight:
                                  selected ? FontWeight.w600 : FontWeight.normal,
                            ),
                          ),
                          const SizedBox(height: 2),
                          Text(
                            lang.$3,
                            style: const TextStyle(
                                color: Colors.white38, fontSize: 13),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            );
          }),
        ],
      ),
    );
  }

  Widget _bluetoothPage() {
    return Padding(
      padding: const EdgeInsets.all(32),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const SizedBox(height: 16),
          const Text(
            'Car Module',
            style: TextStyle(
              color: Colors.white,
              fontSize: 24,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 8),
          const Text(
            'Do you have a Bluetooth car chassis module? '
            'You can set this up later from the face screen.',
            style: TextStyle(color: Colors.white54, fontSize: 14, height: 1.4),
          ),
          const SizedBox(height: 32),

          GestureDetector(
            onTap: () => setState(() => _setupBluetooth = false),
            child: _optionCard(
              'Skip for now',
              'I just want to use the companion as a desk buddy.',
              Icons.smart_toy_outlined,
              !_setupBluetooth,
            ),
          ),
          const SizedBox(height: 12),
          GestureDetector(
            onTap: () => setState(() => _setupBluetooth = true),
            child: _optionCard(
              'I have a car module',
              'I\'ll connect it from the face screen using the car controls.',
              Icons.directions_car_outlined,
              _setupBluetooth,
            ),
          ),

          if (_setupBluetooth) ...[
            const SizedBox(height: 24),
            Container(
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: Colors.blueAccent.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: Colors.blueAccent.withValues(alpha: 0.3)),
              ),
              child: const Row(
                children: [
                  Icon(Icons.info_outline, color: Colors.blueAccent, size: 18),
                  SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      'Turn on your car module, then use the Bluetooth controls '
                      'on the face screen to pair.',
                      style: TextStyle(color: Colors.white54, fontSize: 13),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _readyPage() {
    return Padding(
      padding: const EdgeInsets.all(32),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            width: 100,
            height: 100,
            decoration: BoxDecoration(
              color: Colors.green.withValues(alpha: 0.15),
              shape: BoxShape.circle,
            ),
            child: const Icon(
              Icons.check_rounded,
              color: Colors.green,
              size: 56,
            ),
          ),
          const SizedBox(height: 32),
          const Text(
            'All Set!',
            style: TextStyle(
              color: Colors.white,
              fontSize: 32,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 12),
          const Text(
            'Your learning companion is ready. '
            'Tap the face to start a conversation, or long-press to access the parent dashboard.',
            textAlign: TextAlign.center,
            style: TextStyle(color: Colors.white54, fontSize: 16, height: 1.5),
          ),
          const SizedBox(height: 32),
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.05),
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Column(
              children: [
                _TipRow(icon: Icons.touch_app, text: 'Tap face to talk'),
                SizedBox(height: 10),
                _TipRow(icon: Icons.pan_tool, text: 'Long-press for parent dashboard'),
                SizedBox(height: 10),
                _TipRow(icon: Icons.settings, text: 'Settings accessible from dashboard'),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ---- Helper Widgets ----

  Widget _setupProviderChip(String value, String label, Color color) {
    final selected = _provider == value;
    return GestureDetector(
      onTap: () => setState(() => _provider = value),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        decoration: BoxDecoration(
          color: selected ? color.withValues(alpha: 0.2) : Colors.transparent,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: selected ? color : Colors.grey[700]!,
            width: selected ? 2 : 1,
          ),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: selected ? color : Colors.grey,
            fontWeight: selected ? FontWeight.bold : FontWeight.normal,
          ),
        ),
      ),
    );
  }

  Widget _setupKeyField(
    String label,
    TextEditingController controller,
    bool active,
    String hint,
  ) {
    return TextField(
      controller: controller,
      obscureText: true,
      onChanged: (_) => setState(() {}),
      style: TextStyle(
        color: active ? Colors.white : Colors.grey,
        fontSize: 14,
      ),
      decoration: InputDecoration(
        labelText: label,
        hintText: hint,
        hintStyle: TextStyle(color: Colors.grey[700]),
        labelStyle: TextStyle(color: active ? Colors.white70 : Colors.grey[700]),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: BorderSide(color: active ? Colors.white30 : Colors.grey[800]!),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: const BorderSide(color: Colors.orange),
        ),
        suffixIcon: controller.text.trim().isNotEmpty
            ? const Icon(Icons.check_circle, color: Colors.green, size: 18)
            : null,
        isDense: true,
        contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
      ),
    );
  }

  Widget _optionCard(String title, String subtitle, IconData icon, bool selected) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: selected
            ? Colors.orangeAccent.withValues(alpha: 0.1)
            : Colors.white.withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: selected ? Colors.orangeAccent : Colors.transparent,
          width: 2,
        ),
      ),
      child: Row(
        children: [
          Icon(icon, color: selected ? Colors.orangeAccent : Colors.white38, size: 28),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: TextStyle(
                    color: selected ? Colors.white : Colors.white70,
                    fontSize: 15,
                    fontWeight: selected ? FontWeight.w600 : FontWeight.normal,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  subtitle,
                  style: const TextStyle(color: Colors.white38, fontSize: 12),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _TipRow extends StatelessWidget {
  final IconData icon;
  final String text;

  const _TipRow({required this.icon, required this.text});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Icon(icon, color: Colors.orangeAccent, size: 18),
        const SizedBox(width: 10),
        Text(text, style: const TextStyle(color: Colors.white54, fontSize: 14)),
      ],
    );
  }
}
