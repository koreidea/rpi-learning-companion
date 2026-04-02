import 'dart:math';

import 'package:shared_preferences/shared_preferences.dart';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_types.dart';

/// A language entry with hello, thank you, and a fun phrase.
class _LanguageEntry {
  final String language;
  final String country;
  final String hello;
  final String helloPronunciation;
  final String thankYou;
  final String thankYouPronunciation;
  final String funPhrase;
  final String funPhraseMeaning;
  final String funFact;

  const _LanguageEntry({
    required this.language,
    required this.country,
    required this.hello,
    required this.helloPronunciation,
    required this.thankYou,
    required this.thankYouPronunciation,
    required this.funPhrase,
    required this.funPhraseMeaning,
    required this.funFact,
  });
}

/// Language of the Day: learn a daily phrase in a new language.
///
/// Teaches: cultural awareness, vocabulary, pronunciation, global curiosity.
///
/// Each session teaches hello, thank you, and a fun phrase in one language.
/// Tracks which languages have been covered via SharedPreferences so the
/// child gets a new language each day.
class LanguageOfDay extends EpisodicActivity {
  final Random _rng = Random();

  bool _active = false;
  int _episode = 1;
  bool _todayDone = false;

  /// 0=intro+hello, 1=thankYou, 2=funPhrase, 3=funFact+practice, 4=done
  int _phase = 0;
  _LanguageEntry? _currentLanguage;
  int _score = 0;

  /// Indices of languages already covered, persisted.
  List<int> _coveredIndices = [];

  static const String _prefsKey = 'language_of_day_covered';
  static const String _prefsDateKey = 'language_of_day_date';

  static const List<_LanguageEntry> _languages = [
    _LanguageEntry(language: 'Swahili', country: 'Kenya and Tanzania', hello: 'Jambo', helloPronunciation: 'jahm-bo', thankYou: 'Asante', thankYouPronunciation: 'ah-sahn-teh', funPhrase: 'Hakuna Matata', funPhraseMeaning: 'No worries', funFact: 'Swahili is spoken by over 100 million people across Africa!'),
    _LanguageEntry(language: 'French', country: 'France and many African countries', hello: 'Bonjour', helloPronunciation: 'bon-zhoor', thankYou: 'Merci', thankYouPronunciation: 'mair-see', funPhrase: 'C\'est la vie', funPhraseMeaning: 'That is life', funFact: 'French is called the language of love!'),
    _LanguageEntry(language: 'Japanese', country: 'Japan', hello: 'Konnichiwa', helloPronunciation: 'koh-nee-chee-wah', thankYou: 'Arigatou', thankYouPronunciation: 'ah-ree-gah-toh', funPhrase: 'Kawaii', funPhraseMeaning: 'Cute', funFact: 'Japanese has three different writing systems!'),
    _LanguageEntry(language: 'Arabic', country: 'Egypt, Saudi Arabia, and many countries', hello: 'Marhaba', helloPronunciation: 'mar-ha-ba', thankYou: 'Shukran', thankYouPronunciation: 'shoo-kran', funPhrase: 'Inshallah', funPhraseMeaning: 'God willing', funFact: 'Arabic is written from right to left, the opposite of English!'),
    _LanguageEntry(language: 'Mandarin Chinese', country: 'China', hello: 'Ni Hao', helloPronunciation: 'nee-how', thankYou: 'Xie Xie', thankYouPronunciation: 'shee-eh shee-eh', funPhrase: 'Jia You', funPhraseMeaning: 'You can do it', funFact: 'Mandarin is the most spoken language in the world with over a billion speakers!'),
    _LanguageEntry(language: 'Spanish', country: 'Spain, Mexico, and most of South America', hello: 'Hola', helloPronunciation: 'oh-la', thankYou: 'Gracias', thankYouPronunciation: 'grah-see-ahs', funPhrase: 'Amigo', funPhraseMeaning: 'Friend', funFact: 'Spanish is spoken in over 20 countries around the world!'),
    _LanguageEntry(language: 'German', country: 'Germany, Austria, and Switzerland', hello: 'Hallo', helloPronunciation: 'hah-loh', thankYou: 'Danke', thankYouPronunciation: 'dahn-keh', funPhrase: 'Wunderbar', funPhraseMeaning: 'Wonderful', funFact: 'German words can be very long! One word for speed limit is Geschwindigkeitsbegrenzung!'),
    _LanguageEntry(language: 'Korean', country: 'South Korea', hello: 'Annyeong', helloPronunciation: 'ahn-nyung', thankYou: 'Gamsahamnida', thankYouPronunciation: 'gam-sa-ham-nee-da', funPhrase: 'Hwaiting', funPhraseMeaning: 'You can do it, fighting', funFact: 'The Korean alphabet, called Hangul, was invented by a king to help everyone read!'),
    _LanguageEntry(language: 'Portuguese', country: 'Brazil and Portugal', hello: 'Ola', helloPronunciation: 'oh-lah', thankYou: 'Obrigado', thankYouPronunciation: 'oh-bree-gah-doo', funPhrase: 'Saudade', funPhraseMeaning: 'Missing someone you love', funFact: 'Portuguese is the official language of 9 countries across 4 continents!'),
    _LanguageEntry(language: 'Russian', country: 'Russia', hello: 'Privet', helloPronunciation: 'pree-vyet', thankYou: 'Spasibo', thankYouPronunciation: 'spa-see-ba', funPhrase: 'Molodets', funPhraseMeaning: 'Well done', funFact: 'Russian uses a different alphabet called Cyrillic with 33 letters!'),
    _LanguageEntry(language: 'Italian', country: 'Italy', hello: 'Ciao', helloPronunciation: 'chow', thankYou: 'Grazie', thankYouPronunciation: 'grah-tsee-eh', funPhrase: 'Bellissimo', funPhraseMeaning: 'Very beautiful', funFact: 'Italian is the language of music! Many music words like piano and forte are Italian!'),
    _LanguageEntry(language: 'Turkish', country: 'Turkey', hello: 'Merhaba', helloPronunciation: 'mair-ha-ba', thankYou: 'Tesekkurler', thankYouPronunciation: 'teh-sheh-kur-lair', funPhrase: 'Guzel', funPhraseMeaning: 'Beautiful', funFact: 'Turkish uses special letters like a C with a tail underneath!'),
    _LanguageEntry(language: 'Thai', country: 'Thailand', hello: 'Sawasdee', helloPronunciation: 'sa-waht-dee', thankYou: 'Khop Khun', thankYouPronunciation: 'kohp-koon', funPhrase: 'Sanuk', funPhraseMeaning: 'Fun', funFact: 'Thailand is called the Land of Smiles because Thai people smile a lot!'),
    _LanguageEntry(language: 'Vietnamese', country: 'Vietnam', hello: 'Xin Chao', helloPronunciation: 'sin-chow', thankYou: 'Cam On', thankYouPronunciation: 'kahm-uhn', funPhrase: 'Dep Qua', funPhraseMeaning: 'So beautiful', funFact: 'Vietnamese uses the Latin alphabet with special marks on top!'),
    _LanguageEntry(language: 'Greek', country: 'Greece', hello: 'Yassou', helloPronunciation: 'yah-soo', thankYou: 'Efcharisto', thankYouPronunciation: 'ef-ha-ree-stoh', funPhrase: 'Opa', funPhraseMeaning: 'Wow, hooray', funFact: 'Many English words come from Greek, like dinosaur, telephone, and alphabet!'),
    _LanguageEntry(language: 'Dutch', country: 'Netherlands', hello: 'Hallo', helloPronunciation: 'hah-loh', thankYou: 'Dank je', thankYouPronunciation: 'dahnk-yuh', funPhrase: 'Gezellig', funPhraseMeaning: 'Cozy and fun together', funFact: 'The Netherlands is famous for windmills, tulips, and bicycles!'),
    _LanguageEntry(language: 'Polish', country: 'Poland', hello: 'Czesc', helloPronunciation: 'cheshch', thankYou: 'Dziekuje', thankYouPronunciation: 'jen-koo-yeh', funPhrase: 'Super', funPhraseMeaning: 'Super, great', funFact: 'Polish has some of the longest consonant clusters in the world!'),
    _LanguageEntry(language: 'Hebrew', country: 'Israel', hello: 'Shalom', helloPronunciation: 'sha-lohm', thankYou: 'Toda', thankYouPronunciation: 'toh-dah', funPhrase: 'Mazal Tov', funPhraseMeaning: 'Congratulations, good luck', funFact: 'Hebrew is read from right to left, and it was revived after not being spoken for centuries!'),
    _LanguageEntry(language: 'Filipino', country: 'Philippines', hello: 'Kumusta', helloPronunciation: 'koo-moos-tah', thankYou: 'Salamat', thankYouPronunciation: 'sah-lah-maht', funPhrase: 'Mabuhay', funPhraseMeaning: 'Long live, welcome', funFact: 'The Philippines has over 7,000 islands and over 170 languages!'),
    _LanguageEntry(language: 'Malay', country: 'Malaysia and Indonesia', hello: 'Selamat', helloPronunciation: 'seh-lah-maht', thankYou: 'Terima Kasih', thankYouPronunciation: 'teh-ree-mah kah-see', funPhrase: 'Boleh', funPhraseMeaning: 'Can do', funFact: 'Malay and Indonesian are very similar, so people from both countries can understand each other!'),
    _LanguageEntry(language: 'Zulu', country: 'South Africa', hello: 'Sawubona', helloPronunciation: 'sah-woo-boh-nah', thankYou: 'Ngiyabonga', thankYouPronunciation: 'ngee-yah-bohn-gah', funPhrase: 'Ubuntu', funPhraseMeaning: 'I am because we are', funFact: 'Zulu has click sounds that are made by clicking your tongue!'),
    _LanguageEntry(language: 'Amharic', country: 'Ethiopia', hello: 'Selam', helloPronunciation: 'seh-lahm', thankYou: 'Ameseginalehu', thankYouPronunciation: 'ah-meh-seh-gee-nah-leh-hoo', funPhrase: 'Betam Tiru', funPhraseMeaning: 'Very good', funFact: 'Ethiopia has its own unique alphabet with over 200 characters!'),
    _LanguageEntry(language: 'Nepali', country: 'Nepal', hello: 'Namaste', helloPronunciation: 'nah-mah-stay', thankYou: 'Dhanyabad', thankYouPronunciation: 'dahn-yah-bahd', funPhrase: 'Ramro', funPhraseMeaning: 'Beautiful, nice', funFact: 'Nepal is home to Mount Everest, the tallest mountain in the world!'),
    _LanguageEntry(language: 'Bengali', country: 'Bangladesh and eastern India', hello: 'Namaskar', helloPronunciation: 'noh-mosh-kar', thankYou: 'Dhonnobad', thankYouPronunciation: 'dhon-no-bahd', funPhrase: 'Bhalo', funPhraseMeaning: 'Good', funFact: 'Bengali is the seventh most spoken language in the world!'),
    _LanguageEntry(language: 'Tamil', country: 'Tamil Nadu in India and Sri Lanka', hello: 'Vanakkam', helloPronunciation: 'vah-nah-kahm', thankYou: 'Nandri', thankYouPronunciation: 'nahn-dree', funPhrase: 'Superb', funPhraseMeaning: 'Excellent', funFact: 'Tamil is one of the oldest languages in the world, over 2,000 years old!'),
    _LanguageEntry(language: 'Yoruba', country: 'Nigeria', hello: 'Bawo ni', helloPronunciation: 'bah-woh-nee', thankYou: 'E se', thankYouPronunciation: 'eh-sheh', funPhrase: 'Ire o', funPhraseMeaning: 'Good wishes to you', funFact: 'Yoruba is a tonal language where the same word can mean different things depending on pitch!'),
    _LanguageEntry(language: 'Icelandic', country: 'Iceland', hello: 'Hallo', helloPronunciation: 'hah-loh', thankYou: 'Takk', thankYouPronunciation: 'tahk', funPhrase: 'Bless', funPhraseMeaning: 'Goodbye, bless you', funFact: 'Iceland has no mosquitoes and people believe in elves and hidden people!'),
    _LanguageEntry(language: 'Maori', country: 'New Zealand', hello: 'Kia Ora', helloPronunciation: 'kee-ah oh-rah', thankYou: 'Ka pai', thankYouPronunciation: 'kah-pie', funPhrase: 'Whanau', funPhraseMeaning: 'Family', funFact: 'The Maori people do a traditional dance called the Haka before sports games!'),
    _LanguageEntry(language: 'Navajo', country: 'United States, Navajo Nation', hello: "Ya'at'eeh", helloPronunciation: 'yah-ah-teh', thankYou: 'Ahehee', thankYouPronunciation: 'ah-heh-heh', funPhrase: 'Hozho', funPhraseMeaning: 'Beauty and harmony', funFact: 'Navajo code talkers used their language as an unbreakable code in World War 2!'),
    _LanguageEntry(language: 'Hawaiian', country: 'Hawaii, United States', hello: 'Aloha', helloPronunciation: 'ah-loh-hah', thankYou: 'Mahalo', thankYouPronunciation: 'mah-hah-loh', funPhrase: 'Ohana', funPhraseMeaning: 'Family', funFact: 'Hawaiian has only 13 letters in its alphabet, the smallest in the world!'),
  ];

  LanguageOfDay();

  @override
  String get id => 'cultural_language_of_day';

  @override
  String get name => 'Language of the Day';

  @override
  String get category => 'world';

  @override
  String get description =>
      'Learn a daily phrase in a new language from around the world.';

  @override
  List<String> get skills => [
        'cultural awareness',
        'vocabulary',
        'pronunciation',
        'global curiosity',
      ];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 10;

  @override
  SkillId? get skillId => SkillId.culturalAwareness;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'language of the day',
          'teach me a language',
          'new language',
          'how do you say hello',
          'words in another language',
        ],
        'hi': [
          'आज की भाषा',
          'नई भाषा सिखाओ',
          'कैसे बोलते हैं',
        ],
        'te': [
          'ఈ రోజు భాష',
          'కొత్త భాష నేర్పించు',
          'ఎలా చెప్తారు',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.nursery;

  @override
  bool get isActive => _active;

  @override
  int get currentEpisode => _episode;

  @override
  String get episodeKey => _prefsKey;

  @override
  bool get hasNewEpisode => !_todayDone;

  @override
  String get progressSummary {
    return 'Learned ${_coveredIndices.length} of ${_languages.length} languages.';
  }

  @override
  Future<void> saveProgress() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setStringList(
      _prefsKey,
      _coveredIndices.map((i) => i.toString()).toList(),
    );
    await prefs.setString(
      _prefsDateKey,
      DateTime.now().toIso8601String().substring(0, 10),
    );
  }

  @override
  Future<void> loadProgress() async {
    final prefs = await SharedPreferences.getInstance();
    final stored = prefs.getStringList(_prefsKey);
    if (stored != null) {
      _coveredIndices = stored.map((s) => int.parse(s)).toList();
    }
    final dateStr = prefs.getString(_prefsDateKey);
    final today = DateTime.now().toIso8601String().substring(0, 10);
    _todayDone = dateStr == today;
    _episode = _coveredIndices.length + 1;
  }

  @override
  Future<String> start() async {
    await loadProgress();
    _active = true;
    _score = 0;
    _phase = 0;

    // Pick a language not yet covered
    final available = <int>[];
    for (int i = 0; i < _languages.length; i++) {
      if (!_coveredIndices.contains(i)) {
        available.add(i);
      }
    }

    if (available.isEmpty) {
      // All languages covered! Reset and start again.
      _coveredIndices.clear();
      for (int i = 0; i < _languages.length; i++) {
        available.add(i);
      }
    }

    final chosen = available[_rng.nextInt(available.length)];
    _currentLanguage = _languages[chosen];
    _coveredIndices.add(chosen);
    await saveProgress();

    final lang = _currentLanguage!;
    return 'Today we are learning ${lang.language}, spoken in ${lang.country}! '
        'To say hello in ${lang.language}, you say ${lang.hello}. '
        'It sounds like ${lang.helloPronunciation}. Can you try saying '
        '${lang.hello}?';
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_isQuitTrigger(lower)) {
      return await end();
    }

    final lang = _currentLanguage!;

    switch (_phase) {
      case 0:
        _phase = 1;
        _score += 10;
        return 'Great job! You said ${lang.hello}! Now let us learn how to '
            'say thank you. In ${lang.language}, thank you is ${lang.thankYou}. '
            'It sounds like ${lang.thankYouPronunciation}. Can you say '
            '${lang.thankYou}?';

      case 1:
        _phase = 2;
        _score += 10;
        return 'Wonderful! Now here is a fun phrase. In ${lang.language}, '
            'people say ${lang.funPhrase}, which means ${lang.funPhraseMeaning}. '
            'Can you try saying ${lang.funPhrase}?';

      case 2:
        _phase = 3;
        _score += 10;
        return 'You are doing amazing! Here is a fun fact about ${lang.language}: '
            '${lang.funFact} '
            'So today you learned hello is ${lang.hello}, thank you is '
            '${lang.thankYou}, and ${lang.funPhrase} means ${lang.funPhraseMeaning}!';

      case 3:
        _active = false;
        _score += 10;
        _todayDone = true;
        await saveProgress();
        return 'You just learned 3 phrases in ${lang.language}! That is '
            'language number ${_coveredIndices.length}! '
            'Score: $_score points! Come back tomorrow for a new language!';

      default:
        return await end();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    if (_currentLanguage == null) {
      return 'Come back anytime to learn a new language!';
    }
    return 'You started learning ${_currentLanguage!.language} today! '
        'Remember, hello is ${_currentLanguage!.hello} and thank you is '
        '${_currentLanguage!.thankYou}. See you next time!';
  }

  bool _isQuitTrigger(String lower) {
    const triggers = [
      'stop', 'quit', 'exit', "i'm done", 'no more',
      'finish', 'end',
    ];
    return triggers.any((t) => lower.contains(t));
  }
}
