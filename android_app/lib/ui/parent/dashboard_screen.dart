import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../core/state/shared_state.dart';

/// Main parent dashboard showing bot status, conversation stats, and history.
class DashboardScreen extends ConsumerStatefulWidget {
  const DashboardScreen({super.key});

  @override
  ConsumerState<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends ConsumerState<DashboardScreen> {
  @override
  void initState() {
    super.initState();
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
      DeviceOrientation.portraitDown,
      DeviceOrientation.landscapeLeft,
      DeviceOrientation.landscapeRight,
    ]);
  }

  @override
  Widget build(BuildContext context) {
    final state = ref.watch(sharedStateProvider);
    final history = state.conversationHistory;
    final today = DateTime.now();
    final todayConversations = history.where((e) =>
        e.timestamp.year == today.year &&
        e.timestamp.month == today.month &&
        e.timestamp.day == today.day).toList();

    final languagesUsed = todayConversations
        .map((e) => e.language)
        .toSet();

    return Scaffold(
      backgroundColor: const Color(0xFF1A1A2E),
      appBar: AppBar(
        backgroundColor: const Color(0xFF16213E),
        elevation: 0,
        title: const Text('Parent Dashboard'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => context.go('/home'),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () => context.push('/settings'),
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Bot status card
          _StatusCard(
            botState: state.botState.name,
            transcript: state.currentTranscript,
            response: state.currentResponse,
            mode: state.mode,
            language: state.language,
          ),
          const SizedBox(height: 16),

          // Quick stats
          Row(
            children: [
              Expanded(
                child: _StatCard(
                  label: 'Today',
                  value: '${todayConversations.length}',
                  subtitle: 'conversations',
                  icon: Icons.chat_bubble_outline,
                  color: Colors.blueAccent,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _StatCard(
                  label: 'Languages',
                  value: '${languagesUsed.length}',
                  subtitle: languagesUsed.isNotEmpty
                      ? languagesUsed.map(_langName).join(', ')
                      : 'none yet',
                  icon: Icons.language,
                  color: Colors.purpleAccent,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _StatCard(
                  label: 'Total',
                  value: '${history.length}',
                  subtitle: 'all time',
                  icon: Icons.history,
                  color: Colors.tealAccent,
                ),
              ),
            ],
          ),
          const SizedBox(height: 24),

          // Recent conversations
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Recent Conversations',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.w600,
                ),
              ),
              TextButton(
                onPressed: () => context.push('/history'),
                child: const Text(
                  'View All',
                  style: TextStyle(color: Colors.orangeAccent),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),

          if (history.isEmpty)
            Container(
              padding: const EdgeInsets.all(32),
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.05),
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Center(
                child: Text(
                  'No conversations yet.\nTap the face to start talking!',
                  textAlign: TextAlign.center,
                  style: TextStyle(color: Colors.white38, fontSize: 14),
                ),
              ),
            )
          else
            ...history.reversed.take(10).map((entry) => _ConversationTile(entry: entry)),
        ],
      ),
    );
  }

  String _langName(String code) {
    switch (code) {
      case 'hi': return 'Hindi';
      case 'te': return 'Telugu';
      case 'en':
      default: return 'English';
    }
  }
}

class _StatusCard extends StatelessWidget {
  final String botState;
  final String transcript;
  final String response;
  final String mode;
  final String language;

  const _StatusCard({
    required this.botState,
    required this.transcript,
    required this.response,
    required this.mode,
    required this.language,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            const Color(0xFF16213E),
            const Color(0xFF0F3460),
          ],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 12,
                height: 12,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: _stateColor(botState),
                ),
              ),
              const SizedBox(width: 8),
              Text(
                'Status: ${_stateLabel(botState)}',
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const Spacer(),
              _chip(mode == 'online' ? 'Online' : 'Offline',
                  mode == 'online' ? Colors.green : Colors.orange),
              const SizedBox(width: 6),
              _chip(_langLabel(language), Colors.blueAccent),
            ],
          ),
          if (transcript.isNotEmpty) ...[
            const SizedBox(height: 12),
            Text(
              'Child said: "$transcript"',
              style: const TextStyle(color: Colors.white70, fontSize: 13),
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
            ),
          ],
          if (response.isNotEmpty) ...[
            const SizedBox(height: 4),
            Text(
              'Bot replied: "$response"',
              style: const TextStyle(color: Colors.white54, fontSize: 13),
              maxLines: 3,
              overflow: TextOverflow.ellipsis,
            ),
          ],
        ],
      ),
    );
  }

  Widget _chip(String label, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.2),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withValues(alpha: 0.5)),
      ),
      child: Text(
        label,
        style: TextStyle(color: color, fontSize: 11, fontWeight: FontWeight.w500),
      ),
    );
  }

  Color _stateColor(String s) {
    switch (s) {
      case 'ready': return Colors.green;
      case 'listening': return Colors.blue;
      case 'processing': return Colors.yellow;
      case 'speaking': return Colors.purple;
      case 'error': return Colors.red;
      case 'sleeping': return Colors.grey;
      default: return Colors.white54;
    }
  }

  String _stateLabel(String s) {
    switch (s) {
      case 'ready': return 'Ready';
      case 'listening': return 'Listening';
      case 'processing': return 'Thinking';
      case 'speaking': return 'Speaking';
      case 'error': return 'Error';
      case 'sleeping': return 'Sleeping';
      case 'loading': return 'Loading';
      default: return s;
    }
  }

  String _langLabel(String code) {
    switch (code) {
      case 'hi': return 'Hindi';
      case 'te': return 'Telugu';
      case 'en':
      default: return 'English';
    }
  }
}

class _StatCard extends StatelessWidget {
  final String label;
  final String value;
  final String subtitle;
  final IconData icon;
  final Color color;

  const _StatCard({
    required this.label,
    required this.value,
    required this.subtitle,
    required this.icon,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, color: color, size: 20),
          const SizedBox(height: 8),
          Text(
            value,
            style: TextStyle(
              color: color,
              fontSize: 28,
              fontWeight: FontWeight.bold,
            ),
          ),
          Text(
            subtitle,
            style: const TextStyle(color: Colors.white54, fontSize: 11),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
        ],
      ),
    );
  }
}

class _ConversationTile extends StatelessWidget {
  final ConversationEntry entry;

  const _ConversationTile({required this.entry});

  @override
  Widget build(BuildContext context) {
    final time = _formatTime(entry.timestamp);
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                time,
                style: const TextStyle(color: Colors.white38, fontSize: 11),
              ),
              const SizedBox(width: 8),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
                decoration: BoxDecoration(
                  color: Colors.blueAccent.withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Text(
                  _langLabel(entry.language),
                  style: const TextStyle(color: Colors.blueAccent, fontSize: 10),
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            entry.userText,
            style: const TextStyle(color: Colors.white70, fontSize: 13),
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
          ),
          const SizedBox(height: 4),
          Text(
            entry.botResponse,
            style: const TextStyle(color: Colors.white54, fontSize: 13),
            maxLines: 3,
            overflow: TextOverflow.ellipsis,
          ),
        ],
      ),
    );
  }

  String _formatTime(DateTime dt) {
    final h = dt.hour.toString().padLeft(2, '0');
    final m = dt.minute.toString().padLeft(2, '0');
    return '$h:$m';
  }

  String _langLabel(String code) {
    switch (code) {
      case 'hi': return 'Hindi';
      case 'te': return 'Telugu';
      case 'en':
      default: return 'English';
    }
  }
}
