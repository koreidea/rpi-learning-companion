import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../models/skill.dart';
import '../home/skill_icon_map.dart';
import 'activity_list.dart';
import 'encyclopedia_reader.dart';

/// Per-skill detail view with three tabs: Overview, Activities, Encyclopedia.
class SkillDetailScreen extends ConsumerWidget {
  final String skillIdName;

  const SkillDetailScreen({super.key, required this.skillIdName});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // Find the skill by enum name
    final skillId = SkillId.values.firstWhere(
      (s) => s.name == skillIdName,
      orElse: () => SkillId.criticalThinking,
    );
    final skill = SkillRegistry.skills[skillId]!;
    final color = Color(skill.colorValue);
    final iconData = skillIconData(skill.icon);
    final theme = Theme.of(context);

    return DefaultTabController(
      length: 3,
      child: Scaffold(
        body: NestedScrollView(
          headerSliverBuilder: (context, innerBoxIsScrolled) => [
            SliverAppBar(
              expandedHeight: 200,
              pinned: true,
              backgroundColor: color.withValues(alpha: 0.15),
              foregroundColor: theme.colorScheme.onSurface,
              flexibleSpace: FlexibleSpaceBar(
                background: SafeArea(
                  child: Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const SizedBox(height: 40),
                        Hero(
                          tag: 'skill_icon_${skill.id.name}',
                          child: Container(
                            width: 64,
                            height: 64,
                            decoration: BoxDecoration(
                              color: color.withValues(alpha: 0.15),
                              borderRadius: BorderRadius.circular(18),
                            ),
                            child: Icon(iconData, color: color, size: 36),
                          ),
                        ),
                        const SizedBox(height: 10),
                        Hero(
                          tag: 'skill_name_${skill.id.name}',
                          child: Material(
                            color: Colors.transparent,
                            child: Text(
                              skill.shortName,
                              style: theme.textTheme.headlineSmall?.copyWith(
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(height: 4),
                        Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 10, vertical: 3),
                          decoration: BoxDecoration(
                            color: color.withValues(alpha: 0.12),
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: Text(
                            skill.category[0].toUpperCase() +
                                skill.category.substring(1),
                            style: TextStyle(
                              fontSize: 12,
                              fontWeight: FontWeight.w500,
                              color: color,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
              bottom: TabBar(
                labelColor: color,
                unselectedLabelColor: theme.colorScheme.onSurfaceVariant,
                indicatorColor: color,
                tabs: const [
                  Tab(text: 'Overview'),
                  Tab(text: 'Activities'),
                  Tab(text: 'Encyclopedia'),
                ],
              ),
            ),
          ],
          body: TabBarView(
            children: [
              _OverviewTab(skill: skill),
              ActivityList(skill: skill),
              EncyclopediaReader(skill: skill),
            ],
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Overview Tab
// ---------------------------------------------------------------------------

class _OverviewTab extends StatelessWidget {
  final Skill skill;

  const _OverviewTab({required this.skill});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final color = Color(skill.colorValue);

    return ListView(
      padding: const EdgeInsets.all(20),
      children: [
        // Description card
        Card(
          elevation: 0,
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'About this Skill',
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  skill.description,
                  style: theme.textTheme.bodyMedium?.copyWith(
                    color: theme.colorScheme.onSurfaceVariant,
                    height: 1.5,
                  ),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 16),

        // Stats cards
        Row(
          children: [
            Expanded(
              child: _StatTile(
                icon: Icons.play_circle_outline,
                label: 'Sessions',
                value: '0',
                color: color,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: _StatTile(
                icon: Icons.timer_outlined,
                label: 'Time Spent',
                value: '0m',
                color: color,
              ),
            ),
          ],
        ),
        const SizedBox(height: 16),

        // Encyclopedia section preview
        Card(
          elevation: 0,
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          color: color.withValues(alpha: 0.06),
          child: ListTile(
            shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16)),
            leading: Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: color.withValues(alpha: 0.12),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Icon(Icons.auto_stories, color: color, size: 22),
            ),
            title: Text(
              skill.encyclopediaTitle,
              style: theme.textTheme.titleSmall?.copyWith(
                fontWeight: FontWeight.w600,
              ),
            ),
            subtitle: Text(
              'Stories, frameworks, and fun facts',
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            trailing: Icon(Icons.chevron_right, color: color),
            onTap: () {
              // Switch to encyclopedia tab
              DefaultTabController.of(context).animateTo(2);
            },
          ),
        ),
      ],
    );
  }
}

class _StatTile extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  final Color color;

  const _StatTile({
    required this.icon,
    required this.label,
    required this.value,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, color: color, size: 24),
            const SizedBox(height: 10),
            Text(
              value,
              style: theme.textTheme.headlineSmall?.copyWith(
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
            Text(
              label,
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
