/// The 20 21st Century Skills that form the learning framework.
///
/// Each skill has a unique identifier, metadata for display, and is
/// categorized into one of five domains: cognitive, social, practical,
/// creative, or wellness.
enum SkillId {
  /// s01 - Critical Thinking & Problem Solving
  criticalThinking,

  /// s02 - Creativity & Innovation
  creativity,

  /// s03 - Communication
  communication,

  /// s04 - Collaboration & Teamwork
  collaboration,

  /// s05 - Leadership
  leadership,

  /// s06 - Emotional Intelligence
  emotionalIntelligence,

  /// s07 - Adaptability & Resilience
  adaptability,

  /// s08 - Financial Literacy
  financialLiteracy,

  /// s09 - Digital Citizenship & Technology
  digitalCitizenship,

  /// s10 - Environmental Awareness
  environmental,

  /// s11 - Cultural Awareness & Global Citizenship
  culturalAwareness,

  /// s12 - Health & Wellness
  healthWellness,

  /// s13 - Entrepreneurial Thinking
  entrepreneurial,

  /// s14 - Ethics & Social Responsibility
  ethics,

  /// s15 - Design Thinking
  designThinking,

  /// s16 - Information Literacy
  informationLiteracy,

  /// s17 - Self-Direction & Initiative
  selfDirection,

  /// s18 - Media Creation & Digital Expression
  mediaCreation,

  /// s19 - Scientific Thinking & Inquiry
  scientificThinking,

  /// s20 - Time Management & Organization
  timeManagement,
}

/// A single 21st Century Skill with all its display and classification metadata.
class Skill {
  /// The enum identifier for this skill.
  final SkillId id;

  /// Full display name, e.g. "Critical Thinking & Problem Solving".
  final String name;

  /// Short display name for compact UI, e.g. "Critical Thinking".
  final String shortName;

  /// One-line description of the skill.
  final String description;

  /// Material icon name (as string for flexibility with icon mapping).
  final String icon;

  /// Color hex value for theming this skill in the UI.
  final int colorValue;

  /// The encyclopedia section title, e.g. "The Thinker's Vault".
  final String encyclopediaTitle;

  /// Category: "cognitive", "social", "practical", "creative", or "wellness".
  final String category;

  /// Creates a [Skill] with all required metadata.
  const Skill({
    required this.id,
    required this.name,
    required this.shortName,
    required this.description,
    required this.icon,
    required this.colorValue,
    required this.encyclopediaTitle,
    required this.category,
  });
}

/// Static registry of all 20 skills with their full metadata.
///
/// Use [SkillRegistry.get] for single lookups, [SkillRegistry.getByCategory]
/// for filtered lists, or [SkillRegistry.all] for the complete set.
class SkillRegistry {
  SkillRegistry._();

  /// Complete map of all skills indexed by [SkillId].
  static const Map<SkillId, Skill> skills = {
    SkillId.criticalThinking: Skill(
      id: SkillId.criticalThinking,
      name: 'Critical Thinking & Problem Solving',
      shortName: 'Critical Thinking',
      description: 'Analyze, evaluate, and solve problems logically.',
      icon: 'psychology',
      colorValue: 0xFF5C6BC0,
      encyclopediaTitle: "The Thinker's Vault",
      category: 'cognitive',
    ),
    SkillId.creativity: Skill(
      id: SkillId.creativity,
      name: 'Creativity & Innovation',
      shortName: 'Creativity',
      description: 'Generate original ideas and express them boldly.',
      icon: 'palette',
      colorValue: 0xFFFF7043,
      encyclopediaTitle: "The Inventor's Library",
      category: 'creative',
    ),
    SkillId.communication: Skill(
      id: SkillId.communication,
      name: 'Communication',
      shortName: 'Communication',
      description: 'Express ideas clearly and listen actively.',
      icon: 'chat_bubble',
      colorValue: 0xFF26A69A,
      encyclopediaTitle: "The Speaker's Stage",
      category: 'social',
    ),
    SkillId.collaboration: Skill(
      id: SkillId.collaboration,
      name: 'Collaboration & Teamwork',
      shortName: 'Collaboration',
      description: 'Work effectively with others toward shared goals.',
      icon: 'group',
      colorValue: 0xFF42A5F5,
      encyclopediaTitle: "The Team Tower",
      category: 'social',
    ),
    SkillId.leadership: Skill(
      id: SkillId.leadership,
      name: 'Leadership',
      shortName: 'Leadership',
      description: 'Inspire and guide others with confidence and empathy.',
      icon: 'military_tech',
      colorValue: 0xFFAB47BC,
      encyclopediaTitle: "The Captain's Quarters",
      category: 'social',
    ),
    SkillId.emotionalIntelligence: Skill(
      id: SkillId.emotionalIntelligence,
      name: 'Emotional Intelligence',
      shortName: 'Emotional Intelligence',
      description: 'Understand and manage emotions in self and others.',
      icon: 'favorite',
      colorValue: 0xFFEC407A,
      encyclopediaTitle: "The Heart's Garden",
      category: 'social',
    ),
    SkillId.adaptability: Skill(
      id: SkillId.adaptability,
      name: 'Adaptability & Resilience',
      shortName: 'Adaptability',
      description: 'Adjust to change and bounce back from setbacks.',
      icon: 'autorenew',
      colorValue: 0xFF66BB6A,
      encyclopediaTitle: "The Chameleon's Den",
      category: 'practical',
    ),
    SkillId.financialLiteracy: Skill(
      id: SkillId.financialLiteracy,
      name: 'Financial Literacy',
      shortName: 'Financial Literacy',
      description: 'Understand money, saving, spending, and value.',
      icon: 'account_balance',
      colorValue: 0xFF8D6E63,
      encyclopediaTitle: "The Treasure Chest",
      category: 'practical',
    ),
    SkillId.digitalCitizenship: Skill(
      id: SkillId.digitalCitizenship,
      name: 'Digital Citizenship & Technology',
      shortName: 'Digital Citizenship',
      description: 'Use technology responsibly and stay safe online.',
      icon: 'security',
      colorValue: 0xFF78909C,
      encyclopediaTitle: "The Digital Fortress",
      category: 'practical',
    ),
    SkillId.environmental: Skill(
      id: SkillId.environmental,
      name: 'Environmental Awareness',
      shortName: 'Environment',
      description: 'Care for the planet and understand ecosystems.',
      icon: 'eco',
      colorValue: 0xFF4CAF50,
      encyclopediaTitle: "The Green Globe",
      category: 'wellness',
    ),
    SkillId.culturalAwareness: Skill(
      id: SkillId.culturalAwareness,
      name: 'Cultural Awareness & Global Citizenship',
      shortName: 'Cultural Awareness',
      description: 'Appreciate diversity and connect across cultures.',
      icon: 'public',
      colorValue: 0xFF29B6F6,
      encyclopediaTitle: "The World Window",
      category: 'wellness',
    ),
    SkillId.healthWellness: Skill(
      id: SkillId.healthWellness,
      name: 'Health & Wellness',
      shortName: 'Health & Wellness',
      description: 'Build healthy habits for body and mind.',
      icon: 'self_improvement',
      colorValue: 0xFFEF5350,
      encyclopediaTitle: "The Wellness Workshop",
      category: 'wellness',
    ),
    SkillId.entrepreneurial: Skill(
      id: SkillId.entrepreneurial,
      name: 'Entrepreneurial Thinking',
      shortName: 'Entrepreneurship',
      description: 'Spot opportunities and turn ideas into action.',
      icon: 'rocket_launch',
      colorValue: 0xFFFFA726,
      encyclopediaTitle: "The Launchpad",
      category: 'wellness',
    ),
    SkillId.ethics: Skill(
      id: SkillId.ethics,
      name: 'Ethics & Social Responsibility',
      shortName: 'Ethics',
      description: 'Make fair choices and stand up for what is right.',
      icon: 'balance',
      colorValue: 0xFF7E57C2,
      encyclopediaTitle: "The Justice Hall",
      category: 'social',
    ),
    SkillId.designThinking: Skill(
      id: SkillId.designThinking,
      name: 'Design Thinking',
      shortName: 'Design Thinking',
      description: 'Empathize, define, ideate, prototype, and test.',
      icon: 'architecture',
      colorValue: 0xFF26C6DA,
      encyclopediaTitle: "The Design Studio",
      category: 'cognitive',
    ),
    SkillId.informationLiteracy: Skill(
      id: SkillId.informationLiteracy,
      name: 'Information Literacy',
      shortName: 'Info Literacy',
      description: 'Find, evaluate, and use information wisely.',
      icon: 'menu_book',
      colorValue: 0xFF5C6BC0,
      encyclopediaTitle: "The Knowledge Compass",
      category: 'cognitive',
    ),
    SkillId.selfDirection: Skill(
      id: SkillId.selfDirection,
      name: 'Self-Direction & Initiative',
      shortName: 'Self-Direction',
      description: 'Set goals, take initiative, and learn independently.',
      icon: 'explore',
      colorValue: 0xFFFFCA28,
      encyclopediaTitle: "The Pathfinder's Map",
      category: 'practical',
    ),
    SkillId.mediaCreation: Skill(
      id: SkillId.mediaCreation,
      name: 'Media Creation & Digital Expression',
      shortName: 'Media Creation',
      description: 'Create and share stories through digital media.',
      icon: 'videocam',
      colorValue: 0xFFFF8A65,
      encyclopediaTitle: "The Creator's Canvas",
      category: 'creative',
    ),
    SkillId.scientificThinking: Skill(
      id: SkillId.scientificThinking,
      name: 'Scientific Thinking & Inquiry',
      shortName: 'Scientific Thinking',
      description: 'Observe, hypothesize, experiment, and conclude.',
      icon: 'science',
      colorValue: 0xFF009688,
      encyclopediaTitle: "The Discovery Lab",
      category: 'cognitive',
    ),
    SkillId.timeManagement: Skill(
      id: SkillId.timeManagement,
      name: 'Time Management & Organization',
      shortName: 'Time Management',
      description: 'Plan, prioritize, and use time effectively.',
      icon: 'schedule',
      colorValue: 0xFF8E24AA,
      encyclopediaTitle: "The Clockwork Chamber",
      category: 'practical',
    ),
  };

  /// Look up a skill by its [SkillId]. Throws if the ID is not registered.
  static Skill get(SkillId id) => skills[id]!;

  /// Return all skills belonging to a given [category].
  ///
  /// Valid categories: "cognitive", "social", "practical", "creative", "wellness".
  static List<Skill> getByCategory(String category) =>
      skills.values.where((s) => s.category == category).toList();

  /// All 20 skills as an ordered list.
  static List<Skill> get all => skills.values.toList();
}
