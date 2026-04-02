import 'package:flutter/material.dart';

import '../../models/skill.dart';
import '../home/skill_icon_map.dart';

/// A small colored pill-shaped badge displaying a skill name.
class SkillChip extends StatelessWidget {
  final Skill skill;
  final bool showIcon;

  const SkillChip({
    super.key,
    required this.skill,
    this.showIcon = false,
  });

  @override
  Widget build(BuildContext context) {
    final color = Color(skill.colorValue);
    final iconData = skillIconData(skill.icon);

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: color.withValues(alpha: 0.3),
          width: 1,
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (showIcon) ...[
            Icon(iconData, size: 14, color: color),
            const SizedBox(width: 4),
          ],
          Text(
            skill.shortName,
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w500,
              color: color,
            ),
          ),
        ],
      ),
    );
  }
}
