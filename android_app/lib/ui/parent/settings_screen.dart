import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../core/config/config_manager.dart';
import '../../core/orchestrator/orchestrator.dart';
import '../../core/state/shared_state.dart';
import '../../services/illustration_service.dart';

/// Full settings page for the parent dashboard.
class SettingsScreen extends ConsumerStatefulWidget {
  const SettingsScreen({super.key});

  @override
  ConsumerState<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends ConsumerState<SettingsScreen> {
  late ConfigManager _config;
  bool _initialized = false;

  // Controllers
  late TextEditingController _openaiController;
  late TextEditingController _geminiController;
  late TextEditingController _claudeController;

  // Local state mirrors
  String _provider = 'openai';
  String _language = 'en';
  int _volume = 80;
  double _speechRate = 1.0;
  int _ageMin = 3;
  int _ageMax = 6;
  String _mode = 'online';
  bool _continuousMode = false;

  @override
  void initState() {
    super.initState();
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
      DeviceOrientation.portraitDown,
      DeviceOrientation.landscapeLeft,
      DeviceOrientation.landscapeRight,
    ]);
    _openaiController = TextEditingController();
    _geminiController = TextEditingController();
    _claudeController = TextEditingController();
    _loadConfig();
  }

  Future<void> _loadConfig() async {
    final orchestrator = ref.read(orchestratorProvider);
    await orchestrator.init();
    _config = orchestrator.config;

    setState(() {
      _openaiController.text = _config.openaiKey;
      _geminiController.text = _config.geminiKey;
      _claudeController.text = _config.claudeKey;
      _provider = _config.provider;
      _language = _config.language;
      _volume = _config.volume;
      _speechRate = _config.speechRate;
      _ageMin = _config.ageMin;
      _ageMax = _config.ageMax;
      _mode = _config.mode;
      _continuousMode = _config.continuousMode;
      _initialized = true;
    });
  }

  @override
  void dispose() {
    _openaiController.dispose();
    _geminiController.dispose();
    _claudeController.dispose();
    super.dispose();
  }

  Future<void> _saveAll() async {
    await _config.setOpenaiKey(_openaiController.text.trim());
    await _config.setGeminiKey(_geminiController.text.trim());
    await _config.setClaudeKey(_claudeController.text.trim());
    await _config.setProvider(_provider);
    await _config.setLanguage(_language);
    await _config.setVolume(_volume);
    await _config.setSpeechRate(_speechRate);
    await _config.setAgeMin(_ageMin);
    await _config.setAgeMax(_ageMax);
    await _config.setMode(_mode);
    await _config.setContinuousMode(_continuousMode);

    // Update shared state
    final notifier = ref.read(sharedStateProvider.notifier);
    notifier.setVolume(_volume);
    notifier.setLanguage(_language);
    notifier.setMode(_mode);
    notifier.setContinuousMode(_continuousMode);

    // Update orchestrator components and continuous mode
    final orchestrator = ref.read(orchestratorProvider);
    if (_continuousMode) {
      orchestrator.startContinuousMode();
    } else {
      orchestrator.stopContinuousMode();
    }
    orchestrator.llmRouter.openaiKey = _openaiController.text.trim();
    orchestrator.llmRouter.geminiKey = _geminiController.text.trim();
    orchestrator.llmRouter.claudeKey = _claudeController.text.trim();
    orchestrator.llmRouter.activeProvider = _provider;
    orchestrator.cloudStt.updateApiKey(_openaiController.text.trim());

    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Settings saved'),
          duration: Duration(seconds: 2),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!_initialized) {
      return const Scaffold(
        backgroundColor: Color(0xFF1A1A2E),
        body: Center(child: CircularProgressIndicator()),
      );
    }

    final state = ref.watch(sharedStateProvider);

    return Scaffold(
      backgroundColor: const Color(0xFF1A1A2E),
      appBar: AppBar(
        backgroundColor: const Color(0xFF16213E),
        elevation: 0,
        title: const Text('Settings'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => context.pop(),
        ),
        actions: [
          TextButton(
            onPressed: _saveAll,
            child: const Text(
              'Save',
              style: TextStyle(color: Colors.orangeAccent, fontWeight: FontWeight.w600),
            ),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // --- API Keys ---
          _sectionHeader('API Keys'),
          const SizedBox(height: 8),
          _providerSelector(),
          const SizedBox(height: 12),
          _apiKeyField('OpenAI API Key', _openaiController, _provider == 'openai'),
          const SizedBox(height: 8),
          _apiKeyField('Gemini API Key', _geminiController, _provider == 'gemini'),
          const SizedBox(height: 8),
          _apiKeyField('Claude API Key', _claudeController, _provider == 'claude'),
          const SizedBox(height: 24),

          // --- Language ---
          _sectionHeader('Language'),
          const SizedBox(height: 8),
          _languageSelector(),
          const SizedBox(height: 24),

          // --- Audio ---
          _sectionHeader('Audio'),
          const SizedBox(height: 8),
          _sliderRow(
            label: 'Volume',
            value: _volume.toDouble(),
            min: 0,
            max: 100,
            divisions: 20,
            valueLabel: '$_volume%',
            onChanged: (v) => setState(() => _volume = v.round()),
          ),
          const SizedBox(height: 8),
          _sliderRow(
            label: 'Speech Rate',
            value: _speechRate,
            min: 0.5,
            max: 2.0,
            divisions: 6,
            valueLabel: '${_speechRate.toStringAsFixed(1)}x',
            onChanged: (v) => setState(() => _speechRate = v),
          ),
          const SizedBox(height: 24),

          // --- Child Profile ---
          _sectionHeader('Child Age Range'),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: _sliderRow(
                  label: 'Min',
                  value: _ageMin.toDouble(),
                  min: 1,
                  max: 10,
                  divisions: 9,
                  valueLabel: '$_ageMin yrs',
                  onChanged: (v) {
                    final val = v.round();
                    setState(() {
                      _ageMin = val;
                      if (_ageMax < val) _ageMax = val;
                    });
                  },
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: _sliderRow(
                  label: 'Max',
                  value: _ageMax.toDouble(),
                  min: 1,
                  max: 12,
                  divisions: 11,
                  valueLabel: '$_ageMax yrs',
                  onChanged: (v) {
                    final val = v.round();
                    setState(() {
                      _ageMax = val;
                      if (_ageMin > val) _ageMin = val;
                    });
                  },
                ),
              ),
            ],
          ),
          const SizedBox(height: 24),

          // --- Mode ---
          _sectionHeader('Mode'),
          const SizedBox(height: 8),
          _modeToggle(),
          const SizedBox(height: 24),

          // --- Continuous Mode ---
          _sectionHeader('Conversation'),
          const SizedBox(height: 8),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.05),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Continuous Conversation Mode',
                        style: TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        'Bot keeps listening after speaking. No need to tap each time. '
                        'Double-tap the face to toggle on/off.',
                        style: TextStyle(
                          color: Colors.white.withValues(alpha: 0.4),
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
                ),
                Switch(
                  value: _continuousMode,
                  activeTrackColor: Colors.greenAccent.withValues(alpha: 0.5),
                  activeThumbColor: Colors.greenAccent,
                  onChanged: (v) => setState(() => _continuousMode = v),
                ),
              ],
            ),
          ),
          const SizedBox(height: 24),

          // --- Security ---
          _sectionHeader('Security'),
          const SizedBox(height: 8),
          ListTile(
            tileColor: Colors.white.withValues(alpha: 0.05),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
            leading: const Icon(Icons.lock, color: Colors.orangeAccent),
            title: const Text('Change PIN', style: TextStyle(color: Colors.white)),
            subtitle: const Text('Update parent dashboard PIN',
                style: TextStyle(color: Colors.white38, fontSize: 12)),
            trailing: const Icon(Icons.chevron_right, color: Colors.white38),
            onTap: _showChangePinDialog,
          ),
          const SizedBox(height: 24),

          // --- Bluetooth ---
          _sectionHeader('Bluetooth'),
          const SizedBox(height: 8),
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.05),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Row(
              children: [
                Icon(
                  Icons.bluetooth,
                  color: state.carConnected ? Colors.blueAccent : Colors.white38,
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        state.carConnected ? 'Connected' : 'Not connected',
                        style: TextStyle(
                          color: state.carConnected ? Colors.blueAccent : Colors.white54,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                      if (state.carMac != null)
                        Text(
                          state.carMac!,
                          style: const TextStyle(color: Colors.white38, fontSize: 12),
                        ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 24),

          // --- Illustration Cache ---
          _sectionHeader('Illustration Cache'),
          const SizedBox(height: 8),
          _illustrationCacheSection(),
          const SizedBox(height: 24),

          // --- Data ---
          _sectionHeader('Data'),
          const SizedBox(height: 8),
          ListTile(
            tileColor: Colors.red.withValues(alpha: 0.08),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
            leading: const Icon(Icons.delete_outline, color: Colors.redAccent),
            title: const Text('Clear Conversation History',
                style: TextStyle(color: Colors.redAccent)),
            onTap: _confirmClearHistory,
          ),
          const SizedBox(height: 24),

          // --- About ---
          _sectionHeader('About'),
          const SizedBox(height: 8),
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.05),
              borderRadius: BorderRadius.circular(10),
            ),
            child: const Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Learning Companion',
                    style: TextStyle(color: Colors.white, fontWeight: FontWeight.w600)),
                SizedBox(height: 4),
                Text('Version 1.0.0',
                    style: TextStyle(color: Colors.white38, fontSize: 12)),
                SizedBox(height: 4),
                Text('AI-powered learning companion for children ages 3-6. '
                    'Supports English, Hindi, and Telugu.',
                    style: TextStyle(color: Colors.white54, fontSize: 12)),
              ],
            ),
          ),
          const SizedBox(height: 32),
        ],
      ),
    );
  }

  // ---- Widgets ----

  Widget _sectionHeader(String title) {
    return Text(
      title,
      style: const TextStyle(
        color: Colors.white70,
        fontSize: 14,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.5,
      ),
    );
  }

  Widget _providerSelector() {
    return Row(
      children: [
        _providerChip('openai', 'OpenAI', Colors.green),
        const SizedBox(width: 8),
        _providerChip('gemini', 'Gemini', Colors.blue),
        const SizedBox(width: 8),
        _providerChip('claude', 'Claude', Colors.orange),
      ],
    );
  }

  Widget _providerChip(String value, String label, Color color) {
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

  Widget _apiKeyField(String label, TextEditingController controller, bool active) {
    return TextField(
      controller: controller,
      obscureText: true,
      style: TextStyle(
        color: active ? Colors.white : Colors.grey,
        fontSize: 13,
      ),
      decoration: InputDecoration(
        labelText: label,
        labelStyle: TextStyle(color: active ? Colors.white70 : Colors.grey[700]),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: BorderSide(color: active ? Colors.white30 : Colors.grey[800]!),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: const BorderSide(color: Colors.orange),
        ),
        suffixIcon: active
            ? const Icon(Icons.check_circle, color: Colors.green, size: 18)
            : null,
        isDense: true,
        contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
      ),
    );
  }

  Widget _languageSelector() {
    const languages = [
      ('en', 'English'),
      ('hi', 'Hindi'),
      ('te', 'Telugu'),
    ];
    return Row(
      children: languages.map((lang) {
        final selected = _language == lang.$1;
        return Padding(
          padding: const EdgeInsets.only(right: 8),
          child: ChoiceChip(
            label: Text(lang.$2),
            selected: selected,
            selectedColor: Colors.orangeAccent.withValues(alpha: 0.3),
            backgroundColor: Colors.white.withValues(alpha: 0.08),
            labelStyle: TextStyle(
              color: selected ? Colors.orangeAccent : Colors.white54,
            ),
            side: BorderSide(
              color: selected ? Colors.orangeAccent : Colors.transparent,
            ),
            onSelected: (_) => setState(() => _language = lang.$1),
          ),
        );
      }).toList(),
    );
  }

  Widget _sliderRow({
    required String label,
    required double value,
    required double min,
    required double max,
    required int divisions,
    required String valueLabel,
    required ValueChanged<double> onChanged,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(label, style: const TextStyle(color: Colors.white54, fontSize: 13)),
            Text(valueLabel, style: const TextStyle(color: Colors.orangeAccent, fontSize: 13)),
          ],
        ),
        SliderTheme(
          data: SliderTheme.of(context).copyWith(
            activeTrackColor: Colors.orangeAccent,
            inactiveTrackColor: Colors.white12,
            thumbColor: Colors.orangeAccent,
            overlayColor: Colors.orangeAccent.withValues(alpha: 0.2),
            trackHeight: 3,
          ),
          child: Slider(
            value: value,
            min: min,
            max: max,
            divisions: divisions,
            onChanged: onChanged,
          ),
        ),
      ],
    );
  }

  Widget _modeToggle() {
    return Container(
      padding: const EdgeInsets.all(4),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          Expanded(
            child: GestureDetector(
              onTap: () => setState(() => _mode = 'online'),
              child: Container(
                padding: const EdgeInsets.symmetric(vertical: 12),
                decoration: BoxDecoration(
                  color: _mode == 'online'
                      ? Colors.green.withValues(alpha: 0.2)
                      : Colors.transparent,
                  borderRadius: BorderRadius.circular(10),
                  border: _mode == 'online'
                      ? Border.all(color: Colors.green.withValues(alpha: 0.5))
                      : null,
                ),
                child: Center(
                  child: Text(
                    'Online (Cloud)',
                    style: TextStyle(
                      color: _mode == 'online' ? Colors.green : Colors.white38,
                      fontWeight: _mode == 'online' ? FontWeight.w600 : FontWeight.normal,
                    ),
                  ),
                ),
              ),
            ),
          ),
          Expanded(
            child: GestureDetector(
              onTap: () => setState(() => _mode = 'offline'),
              child: Container(
                padding: const EdgeInsets.symmetric(vertical: 12),
                decoration: BoxDecoration(
                  color: _mode == 'offline'
                      ? Colors.orange.withValues(alpha: 0.2)
                      : Colors.transparent,
                  borderRadius: BorderRadius.circular(10),
                  border: _mode == 'offline'
                      ? Border.all(color: Colors.orange.withValues(alpha: 0.5))
                      : null,
                ),
                child: Center(
                  child: Text(
                    'Offline (Local)',
                    style: TextStyle(
                      color: _mode == 'offline' ? Colors.orange : Colors.white38,
                      fontWeight: _mode == 'offline' ? FontWeight.w600 : FontWeight.normal,
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _illustrationCacheSection() {
    final serviceAsync = ref.watch(illustrationServiceProvider);
    final service = serviceAsync.valueOrNull;

    if (service == null) {
      return Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha: 0.05),
          borderRadius: BorderRadius.circular(10),
        ),
        child: const Row(
          children: [
            Icon(Icons.image_outlined, color: Colors.white38),
            SizedBox(width: 12),
            Expanded(
              child: Text(
                'Set an OpenAI API key to enable AI-generated educational illustrations.',
                style: TextStyle(color: Colors.white38, fontSize: 12),
              ),
            ),
          ],
        ),
      );
    }

    return FutureBuilder<(int, int)>(
      future: Future.wait([
        service.getCacheSize(),
        service.getCachedCount(),
      ]).then((results) => (results[0], results[1])),
      builder: (context, snapshot) {
        final cacheSize = snapshot.data?.$1 ?? 0;
        final cachedCount = snapshot.data?.$2 ?? 0;
        final sizeMb = (cacheSize / (1024 * 1024)).toStringAsFixed(1);

        return Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: Colors.white.withValues(alpha: 0.05),
            borderRadius: BorderRadius.circular(10),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  const Icon(Icons.image_outlined, color: Colors.white54),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          '$cachedCount illustrations cached',
                          style: const TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          '$sizeMb MB on disk',
                          style: const TextStyle(
                            color: Colors.white38,
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              if (cachedCount > 0) ...[
                const SizedBox(height: 12),
                SizedBox(
                  width: double.infinity,
                  child: OutlinedButton.icon(
                    onPressed: () => _confirmClearIllustrationCache(service),
                    icon: const Icon(Icons.delete_sweep, size: 16),
                    label: const Text('Clear Illustration Cache'),
                    style: OutlinedButton.styleFrom(
                      foregroundColor: Colors.redAccent,
                      side: BorderSide(
                        color: Colors.redAccent.withValues(alpha: 0.4),
                      ),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10),
                      ),
                      padding: const EdgeInsets.symmetric(vertical: 10),
                    ),
                  ),
                ),
              ],
            ],
          ),
        );
      },
    );
  }

  void _confirmClearIllustrationCache(IllustrationService service) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: const Color(0xFF16213E),
        title: const Text('Clear Illustration Cache?',
            style: TextStyle(color: Colors.white)),
        content: const Text(
          'This will delete all generated illustrations. '
          'You can regenerate them later from each content card.',
          style: TextStyle(color: Colors.white54),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(),
            child: const Text('Cancel',
                style: TextStyle(color: Colors.white54)),
          ),
          TextButton(
            onPressed: () async {
              await service.clearCache();
              Navigator.of(ctx).pop();
              // Force rebuild to update cache stats
              setState(() {});
              if (mounted) {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Illustration cache cleared')),
                );
              }
            },
            child: const Text('Clear',
                style: TextStyle(color: Colors.redAccent)),
          ),
        ],
      ),
    );
  }

  void _showChangePinDialog() {
    final currentController = TextEditingController();
    final newController = TextEditingController();
    final confirmController = TextEditingController();
    String? errorText;

    showDialog(
      context: context,
      builder: (ctx) {
        return StatefulBuilder(builder: (ctx, setDialogState) {
          return AlertDialog(
            backgroundColor: const Color(0xFF16213E),
            title: const Text('Change PIN', style: TextStyle(color: Colors.white)),
            content: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                TextField(
                  controller: currentController,
                  obscureText: true,
                  keyboardType: TextInputType.number,
                  maxLength: 4,
                  style: const TextStyle(color: Colors.white),
                  decoration: const InputDecoration(
                    labelText: 'Current PIN',
                    labelStyle: TextStyle(color: Colors.white54),
                    counterText: '',
                  ),
                ),
                const SizedBox(height: 8),
                TextField(
                  controller: newController,
                  obscureText: true,
                  keyboardType: TextInputType.number,
                  maxLength: 4,
                  style: const TextStyle(color: Colors.white),
                  decoration: const InputDecoration(
                    labelText: 'New PIN',
                    labelStyle: TextStyle(color: Colors.white54),
                    counterText: '',
                  ),
                ),
                const SizedBox(height: 8),
                TextField(
                  controller: confirmController,
                  obscureText: true,
                  keyboardType: TextInputType.number,
                  maxLength: 4,
                  style: const TextStyle(color: Colors.white),
                  decoration: const InputDecoration(
                    labelText: 'Confirm New PIN',
                    labelStyle: TextStyle(color: Colors.white54),
                    counterText: '',
                  ),
                ),
                if (errorText != null) ...[
                  const SizedBox(height: 8),
                  Text(errorText!, style: const TextStyle(color: Colors.redAccent, fontSize: 12)),
                ],
              ],
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.of(ctx).pop(),
                child: const Text('Cancel', style: TextStyle(color: Colors.white54)),
              ),
              TextButton(
                onPressed: () async {
                  if (currentController.text != _config.pin) {
                    setDialogState(() => errorText = 'Current PIN is incorrect');
                    return;
                  }
                  if (newController.text.length != 4) {
                    setDialogState(() => errorText = 'PIN must be 4 digits');
                    return;
                  }
                  if (newController.text != confirmController.text) {
                    setDialogState(() => errorText = 'PINs do not match');
                    return;
                  }
                  await _config.setPin(newController.text);
                  if (ctx.mounted) Navigator.of(ctx).pop();
                  if (mounted) {
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text('PIN updated')),
                    );
                  }
                },
                child: const Text('Save', style: TextStyle(color: Colors.orangeAccent)),
              ),
            ],
          );
        });
      },
    );
  }

  void _confirmClearHistory() {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: const Color(0xFF16213E),
        title: const Text('Clear History?', style: TextStyle(color: Colors.white)),
        content: const Text(
          'This will permanently delete all conversation history.',
          style: TextStyle(color: Colors.white54),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(),
            child: const Text('Cancel', style: TextStyle(color: Colors.white54)),
          ),
          TextButton(
            onPressed: () async {
              final notifier = ref.read(sharedStateProvider.notifier);
              notifier.clearHistory();
              final orchestrator = ref.read(orchestratorProvider);
              orchestrator.clearHistory();
              // Also clear persisted history
              await notifier.saveConversationHistory(orchestrator.config);
              Navigator.of(ctx).pop();
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('History cleared')),
              );
            },
            child: const Text('Clear', style: TextStyle(color: Colors.redAccent)),
          ),
        ],
      ),
    );
  }
}
