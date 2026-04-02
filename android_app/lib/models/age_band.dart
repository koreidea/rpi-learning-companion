/// Age band classifications for content and interaction adaptation.
///
/// Each band defines an age range, session duration, and LLM prompt style
/// so that activities and conversations are age-appropriate.
enum AgeBand {
  /// Ages 3-6: Pre-school and kindergarten.
  nursery,

  /// Ages 7-10: Primary school.
  junior,

  /// Ages 11-14: Middle school.
  senior,
}

/// Extension providing metadata and behavior for each [AgeBand].
extension AgeBandExt on AgeBand {
  /// Human-readable label including the age range.
  String get label {
    switch (this) {
      case AgeBand.nursery:
        return 'Nursery (3-6)';
      case AgeBand.junior:
        return 'Junior (7-10)';
      case AgeBand.senior:
        return 'Senior (11-14)';
    }
  }

  /// Minimum age (inclusive) for this band.
  int get minAge {
    switch (this) {
      case AgeBand.nursery:
        return 3;
      case AgeBand.junior:
        return 7;
      case AgeBand.senior:
        return 11;
    }
  }

  /// Maximum age (inclusive) for this band.
  int get maxAge {
    switch (this) {
      case AgeBand.nursery:
        return 6;
      case AgeBand.junior:
        return 10;
      case AgeBand.senior:
        return 14;
    }
  }

  /// Description of the communication style for LLM system prompts.
  String get promptStyle {
    switch (this) {
      case AgeBand.nursery:
        return 'You are a fun, warm, playful friend. Use simple words. '
            'Lots of praise and encouragement. Keep responses short '
            '(2-3 sentences).';
      case AgeBand.junior:
        return 'You are a friendly teacher. Use age-appropriate vocabulary. '
            'Introduce frameworks and ask follow-up questions. '
            'Medium-length responses.';
      case AgeBand.senior:
        return 'You are a mentor. Ask probing questions. Encourage '
            'independence and critical thinking. Give substantive responses.';
    }
  }

  /// Recommended session duration for this age band.
  Duration get sessionDuration {
    switch (this) {
      case AgeBand.nursery:
        return const Duration(minutes: 10);
      case AgeBand.junior:
        return const Duration(minutes: 17);
      case AgeBand.senior:
        return const Duration(minutes: 25);
    }
  }

  /// Determine the appropriate [AgeBand] for a given age.
  ///
  /// Ages below 3 default to [AgeBand.nursery].
  /// Ages above 14 default to [AgeBand.senior].
  static AgeBand fromAge(int age) {
    if (age <= 6) return AgeBand.nursery;
    if (age <= 10) return AgeBand.junior;
    return AgeBand.senior;
  }
}
