import 'package:flutter/material.dart';

import '../../core/config/config_manager.dart';

/// Dialog for entering API keys.
/// Shown on long-press of the face screen.
class ApiKeyDialog extends StatefulWidget {
  final ConfigManager config;
  final VoidCallback onSaved;

  const ApiKeyDialog({super.key, required this.config, required this.onSaved});

  @override
  State<ApiKeyDialog> createState() => _ApiKeyDialogState();
}

class _ApiKeyDialogState extends State<ApiKeyDialog> {
  late TextEditingController _openaiController;
  late TextEditingController _geminiController;
  late TextEditingController _claudeController;
  late String _selectedProvider;

  @override
  void initState() {
    super.initState();
    _openaiController = TextEditingController(text: widget.config.openaiKey);
    _geminiController = TextEditingController(text: widget.config.geminiKey);
    _claudeController = TextEditingController(text: widget.config.claudeKey);
    _selectedProvider = widget.config.provider;
  }

  @override
  void dispose() {
    _openaiController.dispose();
    _geminiController.dispose();
    _claudeController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Dialog(
      backgroundColor: Colors.grey[900],
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Settings',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 16),

              // Provider selector
              const Text('Active Provider:', style: TextStyle(color: Colors.white70, fontSize: 14)),
              const SizedBox(height: 8),
              Row(
                children: [
                  _providerChip('openai', 'OpenAI', Colors.green),
                  const SizedBox(width: 8),
                  _providerChip('gemini', 'Gemini', Colors.blue),
                  const SizedBox(width: 8),
                  _providerChip('claude', 'Claude', Colors.orange),
                ],
              ),
              const SizedBox(height: 16),

              // API Key fields
              _keyField('OpenAI API Key', _openaiController, _selectedProvider == 'openai'),
              const SizedBox(height: 8),
              _keyField('Gemini API Key', _geminiController, _selectedProvider == 'gemini'),
              const SizedBox(height: 8),
              _keyField('Claude API Key', _claudeController, _selectedProvider == 'claude'),
              const SizedBox(height: 16),

              // Save button
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.orange,
                    foregroundColor: Colors.black,
                  ),
                  onPressed: _save,
                  child: const Text('Save'),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _providerChip(String value, String label, Color color) {
    final selected = _selectedProvider == value;
    return GestureDetector(
      onTap: () => setState(() => _selectedProvider = value),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: selected ? color.withValues(alpha: 0.3) : Colors.transparent,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: selected ? color : Colors.grey,
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

  Widget _keyField(String label, TextEditingController controller, bool active) {
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
          borderSide: BorderSide(color: active ? Colors.white30 : Colors.grey[800]!),
        ),
        focusedBorder: const OutlineInputBorder(
          borderSide: BorderSide(color: Colors.orange),
        ),
        suffixIcon: active
            ? const Icon(Icons.check_circle, color: Colors.green, size: 18)
            : null,
        isDense: true,
        contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      ),
    );
  }

  Future<void> _save() async {
    await widget.config.setOpenaiKey(_openaiController.text.trim());
    await widget.config.setGeminiKey(_geminiController.text.trim());
    await widget.config.setClaudeKey(_claudeController.text.trim());
    await widget.config.setProvider(_selectedProvider);

    widget.onSaved();
    if (mounted) Navigator.of(context).pop();
  }
}
