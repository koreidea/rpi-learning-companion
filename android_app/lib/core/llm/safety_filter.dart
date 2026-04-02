/// Content safety filter for child-appropriate output.
///
/// Checks LLM responses for inappropriate content before TTS.
/// Designed for children aged 3-6.
class SafetyFilter {
  SafetyFilter._();

  /// Safe replacement message when unsafe content is detected.
  static const _safeReplacement =
      "Hmm, let's talk about something more fun! Do you want to hear a story or play a game?";

  // -- Blocked word/phrase lists by category --

  /// Violence, weapons, death (beyond fairy tale context).
  static const _violencePatterns = [
    'kill',
    'murder',
    'stab',
    'shoot',
    'gun',
    'rifle',
    'pistol',
    'bomb',
    'explode',
    'explosion',
    'grenade',
    'massacre',
    'slaughter',
    'torture',
    'strangle',
    'decapitate',
    'dismember',
    'bloodbath',
    'genocide',
    'assassinate',
    'execution',
    'beheading',
    'machete',
    'assault rifle',
    'shotgun',
    'sniper',
    'warfare',
    'molotov',
  ];

  /// Adult/sexual content.
  static const _adultPatterns = [
    'sex',
    'sexual',
    'nude',
    'naked',
    'porn',
    'pornography',
    'erotic',
    'orgasm',
    'genital',
    'intercourse',
    'masturbat',
    'prostitut',
    'stripper',
    'fetish',
    'bondage',
    'xxx',
    'hentai',
    'explicit',
    'obscene',
    'vulgar',
  ];

  /// Profanity/slurs.
  static const _profanityPatterns = [
    'fuck',
    'shit',
    'damn',
    'hell',
    'ass',
    'bitch',
    'bastard',
    'crap',
    'dick',
    'piss',
    'slut',
    'whore',
    'nigger',
    'nigga',
    'faggot',
    'retard',
    'chink',
    'spic',
    'wetback',
    'kike',
  ];

  /// Scary/horror content for young children.
  static const _scaryPatterns = [
    'demon',
    'devil',
    'satan',
    'possessed',
    'exorcism',
    'zombie',
    'undead',
    'nightmare',
    'haunted',
    'poltergeist',
    'serial killer',
    'creepypasta',
    'gore',
    'gory',
    'mutilat',
    'corpse',
    'cadaver',
    'blood dripping',
    'blood-soaked',
    'flesh eating',
    'dismembered',
    'severed head',
    'terrifying',
    'horrifying',
  ];

  /// Personal information requests.
  static const _personalInfoPatterns = [
    'what is your address',
    'where do you live',
    'your phone number',
    'your password',
    'tell me your name and address',
    'credit card',
    'social security',
    'bank account',
    'mother\'s maiden name',
    'give me your parents',
    'what school do you go to',
  ];

  /// Drug/alcohol references.
  static const _drugAlcoholPatterns = [
    'cocaine',
    'heroin',
    'meth',
    'methamphetamine',
    'marijuana',
    'cannabis',
    'weed',
    'crack',
    'ecstasy',
    'lsd',
    'opioid',
    'fentanyl',
    'overdose',
    'inject drugs',
    'snort',
    'drunk',
    'alcohol',
    'beer',
    'wine',
    'vodka',
    'whiskey',
    'liquor',
    'get high',
    'getting high',
    'smoke weed',
    'drug dealer',
  ];

  /// Self-harm content.
  static const _selfHarmPatterns = [
    'suicide',
    'kill myself',
    'kill yourself',
    'self-harm',
    'self harm',
    'cut myself',
    'cut yourself',
    'hang myself',
    'hang yourself',
    'jump off',
    'end my life',
    'end your life',
    'want to die',
    'wanna die',
    'slit wrist',
    'overdose',
  ];

  /// All pattern lists combined for efficient checking.
  static final List<List<String>> _allPatternLists = [
    _violencePatterns,
    _adultPatterns,
    _profanityPatterns,
    _scaryPatterns,
    _personalInfoPatterns,
    _drugAlcoholPatterns,
    _selfHarmPatterns,
  ];

  /// Words that are safe in child context even though they partially match
  /// blocked patterns (e.g., "assistant", "class", "pass", "grass",
  /// "assembly", "classic", "compass", "embarrass", "harass").
  static const _allowedWords = [
    'assistant',
    'class',
    'classic',
    'classroom',
    'pass',
    'passage',
    'passenger',
    'passing',
    'grass',
    'grasshopper',
    'compass',
    'glass',
    'glasses',
    'mass',
    'massive',
    'brass',
    'lasso',
    'molasses',
    'assassin bug', // insect
    'bass', // fish or music
    'sassafras',
    'cassette',
    'hassle',
    'tassel',
    'assembly',
    'embassy',
    'embarrass',
    'amass',
  ];

  /// Check if text is safe for children. Returns true if safe.
  static bool isSafe(String text) {
    if (text.trim().isEmpty) return true;
    final lower = text.toLowerCase();
    return !_containsUnsafeContent(lower);
  }

  /// Filter text: returns the original if safe, or a safe replacement.
  static String filter(String text) {
    if (text.trim().isEmpty) return text;
    if (isSafe(text)) return text;
    return _safeReplacement;
  }

  /// Internal check for unsafe content.
  static bool _containsUnsafeContent(String lowerText) {
    for (final patternList in _allPatternLists) {
      for (final pattern in patternList) {
        if (_matchesPattern(lowerText, pattern)) {
          return true;
        }
      }
    }
    return false;
  }

  /// Match a pattern in text, with word-boundary awareness for short patterns.
  /// For multi-word patterns (phrases), use simple contains.
  /// For single words, check that the match is not part of an allowed word.
  static bool _matchesPattern(String lowerText, String pattern) {
    if (!lowerText.contains(pattern)) return false;

    // Multi-word patterns (phrases) are specific enough — no need for
    // word boundary checking.
    if (pattern.contains(' ')) return true;

    // For short single-word patterns, verify it is not embedded in a safe word.
    for (final allowed in _allowedWords) {
      if (allowed.contains(pattern) && lowerText.contains(allowed)) {
        // The match is inside an allowed word — not unsafe.
        return false;
      }
    }

    // Check word boundaries using regex for very short patterns (<=4 chars)
    // to avoid false positives like "ass" in "class".
    if (pattern.length <= 4) {
      final regex = RegExp('\\b${RegExp.escape(pattern)}\\b');
      return regex.hasMatch(lowerText);
    }

    return true;
  }
}
