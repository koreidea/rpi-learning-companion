import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../activities/activity_base.dart';
import '../../activities/activity_registry.dart';
import '../../core/orchestrator/orchestrator.dart';
import '../../models/skill.dart';

/// List of activities available for a given skill.
///
/// Uses the real [ActivityRegistry] from the [Orchestrator] to display
/// activities that match the skill's [SkillId]. The Start button navigates
/// to the face screen and triggers the activity.
class ActivityList extends ConsumerWidget {
  final Skill skill;

  const ActivityList({super.key, required this.skill});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final theme = Theme.of(context);
    final color = Color(skill.colorValue);

    // Build a temporary registry to get activities for display.
    // We use a lightweight registry without injected dependencies here
    // since we only need metadata (id, name, description, skillId).
    // The real orchestrator registry with full dependencies is used at runtime.
    final registry = ActivityRegistry();
    final activities = registry.getBySkill(skill.id);

    if (activities.isEmpty) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                Icons.construction,
                size: 48,
                color: theme.colorScheme.onSurfaceVariant
                    .withValues(alpha: 0.4),
              ),
              const SizedBox(height: 12),
              Text(
                'Activities coming soon!',
                style: theme.textTheme.titleMedium?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
            ],
          ),
        ),
      );
    }

    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: activities.length,
      itemBuilder: (context, i) {
        final activity = activities[i];
        return _ActivityCard(
          activity: activity,
          color: color,
        );
      },
    );
  }
}

class _ActivityCard extends StatelessWidget {
  final Activity activity;
  final Color color;

  const _ActivityCard({
    required this.activity,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    // Determine requirement badges from activity properties
    final badges = <String>[];
    final desc = activity.description.toLowerCase();
    final name = activity.name.toLowerCase();
    if (desc.contains('camera') ||
        desc.contains('show') ||
        desc.contains('draw') ||
        name.contains('show and tell') ||
        name.contains('drawing') ||
        name.contains('sketch')) {
      badges.add('camera');
    }
    if (desc.contains('car') ||
        desc.contains('drive') ||
        name.contains('exercise') ||
        name.contains('energizer')) {
      badges.add('car');
    }
    // Most activities use the LLM
    if (activity.skills.isNotEmpty) {
      badges.add('llm');
    }

    // Estimate time from age range
    final estimatedTime = '${activity.minAge == activity.maxAge ? 3 : 5} min';

    return Card(
      elevation: 0,
      margin: const EdgeInsets.only(bottom: 12),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    activity.name,
                    style: theme.textTheme.titleSmall?.copyWith(
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: theme.colorScheme.onSurfaceVariant
                        .withValues(alpha: 0.08),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    estimatedTime,
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: theme.colorScheme.onSurfaceVariant,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 6),
            Text(
              activity.description,
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                // Badges
                if (badges.contains('camera'))
                  const _Badge(icon: Icons.camera_alt, label: 'Camera'),
                if (badges.contains('car'))
                  const _Badge(icon: Icons.directions_car, label: 'Car'),
                if (badges.contains('llm'))
                  const _Badge(icon: Icons.smart_toy, label: 'AI'),
                const Spacer(),
                FilledButton.tonal(
                  onPressed: () {
                    // Navigate to face screen with the activity ID
                    context.push('/face?activityId=${activity.id}');
                  },
                  style: FilledButton.styleFrom(
                    backgroundColor: color.withValues(alpha: 0.12),
                    foregroundColor: color,
                    padding: const EdgeInsets.symmetric(
                        horizontal: 20, vertical: 8),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                  child: const Text(
                    'Start',
                    style: TextStyle(fontWeight: FontWeight.w600),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _Badge extends StatelessWidget {
  final IconData icon;
  final String label;

  const _Badge({required this.icon, required this.label});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      margin: const EdgeInsets.only(right: 6),
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 12, color: theme.colorScheme.onSurfaceVariant),
          const SizedBox(width: 3),
          Text(
            label,
            style: theme.textTheme.bodySmall?.copyWith(
              fontSize: 10,
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
        ],
      ),
    );
  }
}
