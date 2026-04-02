import '../../models/skill.dart';

/// Identifiers for each drawable shape in [IllustrationShapes].
enum ShapeType {
  star,
  book,
  flask,
  leaf,
  gear,
  lightbulb,
  rocket,
  wave,
  hexagon,
  dome,
  atom,
  heart,
  cloud,
  mountain,
  tree,
  coin,
  music,
  globe,
  pencil,
  puzzle,
  shield,
  clock,
}

/// Configuration for the background scene of a skill illustration.
class SceneConfig {
  /// 3-5 shapes to scatter in the background.
  final List<ShapeType> shapes;

  /// Gradient angle in radians for the background sweep.
  final double bgAngle;

  const SceneConfig({required this.shapes, this.bgAngle = 0.0});
}

/// Maps each of the 20 skills to a themed scene configuration.
const Map<SkillId, SceneConfig> skillScenes = {
  SkillId.criticalThinking: SceneConfig(
    shapes: [ShapeType.gear, ShapeType.hexagon, ShapeType.puzzle],
  ),
  SkillId.creativity: SceneConfig(
    shapes: [ShapeType.lightbulb, ShapeType.star, ShapeType.pencil],
  ),
  SkillId.communication: SceneConfig(
    shapes: [ShapeType.book, ShapeType.music, ShapeType.cloud],
  ),
  SkillId.collaboration: SceneConfig(
    shapes: [ShapeType.heart, ShapeType.star, ShapeType.hexagon],
  ),
  SkillId.leadership: SceneConfig(
    shapes: [ShapeType.shield, ShapeType.star, ShapeType.mountain],
  ),
  SkillId.emotionalIntelligence: SceneConfig(
    shapes: [ShapeType.heart, ShapeType.cloud, ShapeType.wave],
  ),
  SkillId.adaptability: SceneConfig(
    shapes: [ShapeType.wave, ShapeType.gear, ShapeType.hexagon],
  ),
  SkillId.financialLiteracy: SceneConfig(
    shapes: [ShapeType.coin, ShapeType.star, ShapeType.mountain],
  ),
  SkillId.digitalCitizenship: SceneConfig(
    shapes: [ShapeType.shield, ShapeType.globe, ShapeType.hexagon],
  ),
  SkillId.environmental: SceneConfig(
    shapes: [ShapeType.leaf, ShapeType.tree, ShapeType.wave],
  ),
  SkillId.culturalAwareness: SceneConfig(
    shapes: [ShapeType.globe, ShapeType.dome, ShapeType.star],
  ),
  SkillId.healthWellness: SceneConfig(
    shapes: [ShapeType.heart, ShapeType.leaf, ShapeType.cloud],
  ),
  SkillId.entrepreneurial: SceneConfig(
    shapes: [ShapeType.rocket, ShapeType.lightbulb, ShapeType.coin],
  ),
  SkillId.ethics: SceneConfig(
    shapes: [ShapeType.shield, ShapeType.book, ShapeType.star],
  ),
  SkillId.designThinking: SceneConfig(
    shapes: [ShapeType.pencil, ShapeType.gear, ShapeType.lightbulb],
  ),
  SkillId.informationLiteracy: SceneConfig(
    shapes: [ShapeType.book, ShapeType.globe, ShapeType.puzzle],
  ),
  SkillId.selfDirection: SceneConfig(
    shapes: [ShapeType.mountain, ShapeType.star, ShapeType.rocket],
  ),
  SkillId.mediaCreation: SceneConfig(
    shapes: [ShapeType.pencil, ShapeType.music, ShapeType.star],
  ),
  SkillId.scientificThinking: SceneConfig(
    shapes: [ShapeType.flask, ShapeType.atom, ShapeType.gear],
  ),
  SkillId.timeManagement: SceneConfig(
    shapes: [ShapeType.clock, ShapeType.gear, ShapeType.mountain],
  ),
};

/// Maps content tags to accent shapes for contextual illustrations.
const Map<String, ShapeType> tagShapes = {
  'science': ShapeType.flask,
  'physics': ShapeType.atom,
  'space': ShapeType.rocket,
  'nature': ShapeType.leaf,
  'brain': ShapeType.atom,
  'India': ShapeType.dome,
  'history': ShapeType.clock,
  'technology': ShapeType.gear,
  'business': ShapeType.coin,
  'art': ShapeType.pencil,
  'music': ShapeType.music,
  'water': ShapeType.wave,
  'ocean': ShapeType.wave,
  'environment': ShapeType.tree,
  'health': ShapeType.heart,
  'leadership': ShapeType.shield,
  'reading': ShapeType.book,
  'math': ShapeType.hexagon,
  'money': ShapeType.coin,
  'globe': ShapeType.globe,
  'teamwork': ShapeType.heart,
  'emotion': ShapeType.heart,
  'courage': ShapeType.shield,
  'innovation': ShapeType.lightbulb,
  'detective': ShapeType.puzzle,
  'thinking': ShapeType.gear,
  'adventure': ShapeType.mountain,
  'communication': ShapeType.book,
  'culture': ShapeType.dome,
  'time': ShapeType.clock,
};
