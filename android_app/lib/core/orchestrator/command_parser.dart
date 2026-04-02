/// Parses voice commands from child's speech.
/// Port of _parse_car_command, _parse_volume_command, _parse_song_command,
/// _is_stop_command from rpi/core/main.py.

/// Parsed command result.
class ParsedCommand {
  final CommandType type;
  final String? action;
  final int? speed;
  final double? duration;
  final int? volumeLevel;
  final String? songName;
  final String? activityId;

  const ParsedCommand({
    required this.type,
    this.action,
    this.speed,
    this.duration,
    this.volumeLevel,
    this.songName,
    this.activityId,
  });
}

enum CommandType {
  car,
  volume,
  song,
  follow,
  stop,
  vision,
  activity,
  none,
}

class CommandParser {
  /// Parse all command types from transcript.
  /// Returns first matching command, or null if none.
  ParsedCommand? parse(String transcript, {int currentVolume = 80}) {
    final lower = transcript.toLowerCase().trim();

    // Check stop first (simple)
    if (_isStopCommand(lower)) {
      return const ParsedCommand(type: CommandType.stop);
    }

    // Volume commands
    final vol = _parseVolume(lower, currentVolume);
    if (vol != null) {
      return ParsedCommand(type: CommandType.volume, volumeLevel: vol);
    }

    // Car commands
    final car = _parseCarCommand(lower);
    if (car != null) return car;

    // Follow commands (check before vision — "follow me" is distinct)
    final follow = _parseFollowCommand(lower);
    if (follow != null) return follow;

    // Activity commands: generic triggers only. Specific activity matching
    // is now handled by the ActivityRegistry via voiceTriggers on each
    // activity. The orchestrator checks the registry directly.
    final activity = _parseActivityCommand(lower);
    if (activity != null) return activity;

    // Vision commands
    if (_isVisionCommand(lower)) {
      return const ParsedCommand(type: CommandType.vision);
    }

    // Song commands
    final song = _parseSongCommand(lower);
    if (song != null) return song;

    return null;
  }

  // ── Stop command ──

  bool _isStopCommand(String lower) {
    const triggers = [
      'stop', 'stop the song', 'stop the music', 'stop singing',
      'stop playing', 'be quiet', 'quiet', 'enough',
      'no more', "that's enough", 'shut up',
      // Hindi
      'बंद करो', 'रुको', 'बस',
      // Telugu
      'ఆపు', 'ఆపండి', 'చాలు',
    ];
    return triggers.any((t) => lower.contains(t));
  }

  // ── Volume commands ──

  int? _parseVolume(String lower, int current) {
    // Direct percentage: "volume 50", "set volume to 80"
    final directMatch = RegExp(r'volume\s+(?:to\s+)?(\d+)\s*(?:%|percent)?').firstMatch(lower);
    if (directMatch != null) return int.parse(directMatch.group(1)!).clamp(0, 100);

    final setMatch = RegExp(r'(?:set|change)\s+(?:the\s+)?volume\s+(?:to\s+)?(\d+)').firstMatch(lower);
    if (setMatch != null) return int.parse(setMatch.group(1)!).clamp(0, 100);

    final pctMatch = RegExp(r'(\d+)\s*(?:%|percent)\s*volume').firstMatch(lower);
    if (pctMatch != null) return int.parse(pctMatch.group(1)!).clamp(0, 100);

    // Relative commands
    const upWords = ['increase volume', 'louder', 'volume up', 'turn up', 'raise volume', 'more volume'];
    const downWords = ['decrease volume', 'quieter', 'volume down', 'turn down', 'lower volume', 'less volume'];

    for (final phrase in upWords) {
      if (lower.contains(phrase)) return (current + 20).clamp(0, 100);
    }
    for (final phrase in downWords) {
      if (lower.contains(phrase)) return (current - 20).clamp(0, 100);
    }

    // Special
    if (lower.contains('mute') && !lower.contains('unmute')) return 0;
    if (lower.contains('unmute')) return 80;
    if (['full volume', 'maximum volume', 'max volume'].any((w) => lower.contains(w))) return 100;

    return null;
  }

  // ── Car commands ──

  ParsedCommand? _parseCarCommand(String lower) {
    // Forward
    const forwardTriggers = [
      'go forward', 'move forward', 'go ahead', 'move ahead',
      'go straight', 'drive forward', 'drive ahead',
      'आगे चलो', 'आगे जाओ',
      'ముందుకు వెళ్ళు', 'ముందుకు',
    ];
    for (final t in forwardTriggers) {
      if (lower.contains(t)) {
        return const ParsedCommand(type: CommandType.car, action: 'forward', speed: 200, duration: 1.0);
      }
    }

    // Backward
    const backwardTriggers = [
      'go back', 'move back', 'go backward', 'move backward',
      'reverse', 'drive back', 'back up',
      'पीछे जाओ', 'पीछे चलो',
      'వెనక్కు వెళ్ళు', 'వెనక్కు',
    ];
    for (final t in backwardTriggers) {
      if (lower.contains(t)) {
        return const ParsedCommand(type: CommandType.car, action: 'backward', speed: 200, duration: 1.0);
      }
    }

    // Turn left
    const leftTriggers = [
      'turn left', 'go left', 'move left',
      'बाएं मुड़ो', 'बाएं जाओ',
      'ఎడమకు తిరుగు', 'ఎడమకు',
    ];
    for (final t in leftTriggers) {
      if (lower.contains(t)) {
        return const ParsedCommand(type: CommandType.car, action: 'spinLeft', speed: 200, duration: 0.5);
      }
    }

    // Turn right
    const rightTriggers = [
      'turn right', 'go right', 'move right',
      'दाएं मुड़ो', 'दाएं जाओ',
      'కుడికి తిరుగు', 'కుడికి',
    ];
    for (final t in rightTriggers) {
      if (lower.contains(t)) {
        return const ParsedCommand(type: CommandType.car, action: 'spinRight', speed: 200, duration: 0.5);
      }
    }

    // Spin / turn around
    const spinTriggers = [
      'spin', 'turn around', 'rotate', 'do a spin',
      'घूमो', 'चक्कर',
      'తిరుగు', 'గిర గిర',
    ];
    for (final t in spinTriggers) {
      if (lower.contains(t)) {
        return const ParsedCommand(type: CommandType.car, action: 'spinRight', speed: 220, duration: 1.0);
      }
    }

    // Stop car
    const carStopTriggers = [
      'stop the car', 'car stop', 'stop moving', 'stop driving',
      'brake', 'halt',
      'गाड़ी रोको', 'रुक जाओ',
      'కారు ఆపు', 'ఆగు',
    ];
    for (final t in carStopTriggers) {
      if (lower.contains(t)) {
        return const ParsedCommand(type: CommandType.car, action: 'stop');
      }
    }

    // Dance
    const danceTriggers = [
      'dance', 'do a dance', 'car dance', 'wiggle',
      'नाचो', 'डांस',
      'డ్యాన్స్', 'నాట్యం',
    ];
    for (final t in danceTriggers) {
      if (lower.contains(t)) {
        return const ParsedCommand(type: CommandType.car, action: 'dance');
      }
    }

    // Come here
    const comeTriggers = [
      'come here', 'come to me', 'come over',
      'इधर आओ', 'मेरे पास आओ',
      'ఇక్కడికి రా', 'నా దగ్గరికి రా',
    ];
    for (final t in comeTriggers) {
      if (lower.contains(t)) {
        return const ParsedCommand(type: CommandType.car, action: 'forward', speed: 180, duration: 1.5);
      }
    }

    return null;
  }

  // ── Follow commands ──

  ParsedCommand? _parseFollowCommand(String lower) {
    // Stop following (check first — "stop following" should not trigger "follow")
    const stopFollowTriggers = [
      // English
      'stop following', 'stop following me',
      'stay there', 'stay here', 'stay put',
      "don't follow", 'do not follow',
      'stop chasing', 'stop chasing me',
      'wait there', 'wait here',
      // Hindi
      'रुको', 'वहीं रुको', 'मत आओ', 'पीछा मत करो',
      // Hindi (romanized)
      'ruko', 'mat aao', 'peecha mat karo',
      // Telugu
      'ఆగు', 'అక్కడే ఉండు', 'రాకు', 'వెంట రాకు',
      // Telugu (romanized)
      'aagu', 'akkade undu', 'raaku', 'venta raaku',
    ];
    for (final t in stopFollowTriggers) {
      if (lower.contains(t)) {
        return const ParsedCommand(type: CommandType.follow, action: 'stop');
      }
    }

    // Start following
    const startFollowTriggers = [
      // English
      'follow me', 'come with me', 'chase me',
      'come along', 'tag along', 'walk with me',
      'follow along', 'come on',
      // Hindi
      'मेरे पीछे आओ', 'मेरे साथ आओ', 'पीछे आओ',
      // Hindi (romanized)
      'mere peeche aao', 'mere saath aao', 'peeche aao',
      // Telugu
      'నా వెనక రా', 'నా వెంట రా', 'నాతో రా',
      // Telugu (romanized)
      'na venaka ra', 'na venta ra', 'naato ra',
    ];
    for (final t in startFollowTriggers) {
      if (lower.contains(t)) {
        return const ParsedCommand(type: CommandType.follow, action: 'start');
      }
    }

    return null;
  }

  // ── Vision commands ──

  bool _isVisionCommand(String lower) {
    const triggers = [
      // English
      'what do you see', 'what can you see', 'what is this',
      'what are these', 'what is that', 'look around',
      'describe what you see', 'tell me what you see',
      'describe this', 'look at this', 'what is in front of you',
      'use your eyes', 'can you see',
      // Hindi (romanized — Whisper often outputs romanized Hindi)
      'kya dikh raha hai', 'kya dikhai de raha hai',
      'dekho', 'ye kya hai', 'yeh kya hai',
      // Hindi (Devanagari)
      'क्या दिख रहा है', 'देखो', 'यह क्या है',
      // Telugu (romanized)
      'emi kanipistundi', 'em kanipistundi', 'chudandi', 'chodu',
      // Telugu (script)
      'ఏమి కనిపిస్తుంది', 'చూడండి', 'ఇది ఏమిటి',
    ];
    return triggers.any((t) => lower.contains(t));
  }

  // ── Activity commands ──
  //
  // Specific activity matching (e.g., "coding game" -> coding_sequence) is
  // now handled by ActivityRegistry.findByVoiceTrigger(). Each activity
  // defines its own voiceTriggers map. The command parser only detects
  // generic "play a game" intent so the orchestrator knows to consult the
  // registry or show a menu.

  ParsedCommand? _parseActivityCommand(String lower) {
    const genericTriggers = [
      'play a game', 'let\'s play a game', 'i want to play',
      'let\'s play', 'play game', 'game time',
      // Hindi
      'खेल खेलते हैं', 'खेल खेलो',
      // Telugu
      'ఆట ఆడదాం', 'ఆట ఆడు',
    ];
    for (final t in genericTriggers) {
      if (lower.contains(t)) {
        return const ParsedCommand(type: CommandType.activity);
      }
    }

    return null;
  }

  // ── Song commands ──

  ParsedCommand? _parseSongCommand(String lower) {
    // Available songs (matching the Pi's song library)
    const songs = [
      'twinkle_twinkle',
      'baa_baa_black_sheep',
      'row_row_row_your_boat',
      'london_bridge',
      'head_shoulders_knees_and_toes',
      'humpty_dumpty',
      'jack_and_jill',
      'mary_had_a_little_lamb',
      'old_macdonald',
      'itsy_bitsy_spider',
    ];

    // Check for specific song name
    for (final song in songs) {
      final songName = song.replaceAll('_', ' ');
      if (lower.contains(songName)) {
        return ParsedCommand(type: CommandType.song, songName: song);
      }
    }

    // Generic song triggers → random song
    const songTriggers = [
      'sing me a song', 'sing a song', 'sing song',
      'play a song', 'play song', 'play music',
      'sing me something', 'sing something',
      'nursery rhyme', 'play rhyme',
      'another song', 'next song', 'one more song',
      'sing for me', 'can you sing',
      // Hindi
      'गाना सुनाओ', 'गाना गाओ', 'एक गाना',
      // Telugu
      'పాట పాడు', 'పాట వినిపించు',
    ];

    for (final trigger in songTriggers) {
      if (lower.contains(trigger)) {
        // Pick random song
        final idx = DateTime.now().millisecondsSinceEpoch % songs.length;
        return ParsedCommand(type: CommandType.song, songName: songs[idx]);
      }
    }

    return null;
  }
}
