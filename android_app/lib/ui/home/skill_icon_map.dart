import 'package:flutter/material.dart';

/// Maps icon name strings from [Skill.icon] to Material [IconData].
IconData skillIconData(String iconName) {
  return _iconMap[iconName] ?? Icons.circle;
}

const Map<String, IconData> _iconMap = {
  'psychology': Icons.psychology,
  'palette': Icons.palette,
  'chat_bubble': Icons.chat_bubble,
  'group': Icons.group,
  'military_tech': Icons.military_tech,
  'favorite': Icons.favorite,
  'autorenew': Icons.autorenew,
  'account_balance': Icons.account_balance,
  'security': Icons.security,
  'eco': Icons.eco,
  'public': Icons.public,
  'self_improvement': Icons.self_improvement,
  'rocket_launch': Icons.rocket_launch,
  'balance': Icons.balance,
  'architecture': Icons.architecture,
  'menu_book': Icons.menu_book,
  'explore': Icons.explore,
  'videocam': Icons.videocam,
  'science': Icons.science,
  'schedule': Icons.schedule,
  // Common fallbacks
  'auto_awesome': Icons.auto_awesome,
  'record_voice_over': Icons.record_voice_over,
  'groups': Icons.groups,
  'extension': Icons.extension,
  'format_list_numbered': Icons.format_list_numbered,
  'grid_view': Icons.grid_view,
  'bug_report': Icons.bug_report,
  'account_tree': Icons.account_tree,
  'emoji_people': Icons.emoji_people,
  'auto_stories': Icons.auto_stories,
  'spellcheck': Icons.spellcheck,
  'calculate': Icons.calculate,
  'view_in_ar': Icons.view_in_ar,
  'fitness_center': Icons.fitness_center,
};
