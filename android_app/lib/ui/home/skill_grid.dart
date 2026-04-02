import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../../models/skill.dart';
import '../widgets/progress_ring.dart';
import 'skill_icon_map.dart';

/// Activities tab: 20 skill cards in a categorized, searchable grid.
class SkillGrid extends StatefulWidget {
  const SkillGrid({super.key});

  @override
  State<SkillGrid> createState() => _SkillGridState();
}

class _SkillGridState extends State<SkillGrid> {
  String _query = '';
  final _searchController = TextEditingController();

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    // All categories in order
    const categories = ['cognitive', 'social', 'creative', 'practical', 'wellness'];
    const categoryLabels = {
      'cognitive': 'Cognitive',
      'social': 'Social',
      'creative': 'Creative',
      'practical': 'Practical',
      'wellness': 'Wellness',
    };

    final allSkills = SkillRegistry.all;
    final filtered = _query.isEmpty
        ? allSkills
        : allSkills
            .where((s) =>
                s.name.toLowerCase().contains(_query) ||
                s.shortName.toLowerCase().contains(_query) ||
                s.category.toLowerCase().contains(_query))
            .toList();

    return SafeArea(
      child: CustomScrollView(
        slivers: [
          // Title
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 4),
              child: Text(
                'Activities',
                style: theme.textTheme.headlineMedium?.copyWith(
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),

          // Search bar
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 8),
              child: SearchBar(
                controller: _searchController,
                hintText: 'Search skills...',
                leading: const Padding(
                  padding: EdgeInsets.only(left: 8),
                  child: Icon(Icons.search, size: 20),
                ),
                trailing: _query.isNotEmpty
                    ? [
                        IconButton(
                          icon: const Icon(Icons.clear, size: 20),
                          onPressed: () {
                            _searchController.clear();
                            setState(() => _query = '');
                          },
                        ),
                      ]
                    : null,
                elevation: WidgetStateProperty.all(0),
                onChanged: (v) => setState(() => _query = v.toLowerCase()),
              ),
            ),
          ),

          // If search active, show flat grid
          if (_query.isNotEmpty) ...[
            SliverPadding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              sliver: SliverGrid(
                gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount: 2,
                  mainAxisSpacing: 12,
                  crossAxisSpacing: 12,
                  childAspectRatio: 1.3,
                ),
                delegate: SliverChildBuilderDelegate(
                  (context, i) => _SkillCard(skill: filtered[i]),
                  childCount: filtered.length,
                ),
              ),
            ),
          ] else ...[
            // Grouped by category
            for (final cat in categories) ...[
              SliverToBoxAdapter(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(20, 16, 20, 8),
                  child: Text(
                    categoryLabels[cat] ?? cat,
                    style: theme.textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.w600,
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                  ),
                ),
              ),
              SliverPadding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                sliver: SliverGrid(
                  gridDelegate:
                      const SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: 2,
                    mainAxisSpacing: 12,
                    crossAxisSpacing: 12,
                    childAspectRatio: 1.3,
                  ),
                  delegate: SliverChildBuilderDelegate(
                    (context, i) {
                      final skills = SkillRegistry.getByCategory(cat);
                      return _SkillCard(skill: skills[i]);
                    },
                    childCount: SkillRegistry.getByCategory(cat).length,
                  ),
                ),
              ),
            ],
          ],

          const SliverToBoxAdapter(child: SizedBox(height: 24)),
        ],
      ),
    );
  }
}

class _SkillCard extends StatelessWidget {
  final Skill skill;

  const _SkillCard({required this.skill});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final color = Color(skill.colorValue);
    final iconData = skillIconData(skill.icon);

    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      color: color.withValues(alpha: 0.06),
      child: InkWell(
        borderRadius: BorderRadius.circular(16),
        onTap: () => context.push('/skill/${skill.id.name}'),
        child: Padding(
          padding: const EdgeInsets.all(14),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Container(
                    width: 38,
                    height: 38,
                    decoration: BoxDecoration(
                      color: color.withValues(alpha: 0.12),
                      borderRadius: BorderRadius.circular(11),
                    ),
                    child: Hero(
                      tag: 'skill_icon_${skill.id.name}',
                      child: Icon(iconData, color: color, size: 22),
                    ),
                  ),
                  const Spacer(),
                  ProgressRing(
                    progress: 0.0, // Placeholder -- no real data yet
                    size: 32,
                    color: color,
                    strokeWidth: 2.5,
                    showLabel: false,
                  ),
                ],
              ),
              const Spacer(),
              Hero(
                tag: 'skill_name_${skill.id.name}',
                child: Material(
                  color: Colors.transparent,
                  child: Text(
                    skill.shortName,
                    style: theme.textTheme.titleSmall?.copyWith(
                      fontWeight: FontWeight.w600,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ),
              const SizedBox(height: 2),
              Text(
                skill.description,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                  fontSize: 11,
                ),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
            ],
          ),
        ),
      ),
    );
  }
}
