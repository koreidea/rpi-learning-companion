import 'dart:async';
import 'dart:math';

import 'package:flutter/foundation.dart';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// Data for a country in the World Explorer activity.
class _Country {
  final String name;
  final String greeting;
  final String greetingMeaning;
  final String food;
  final String clothing;
  final String nature;
  final String funFact;

  const _Country({
    required this.name,
    required this.greeting,
    required this.greetingMeaning,
    required this.food,
    required this.clothing,
    required this.nature,
    required this.funFact,
  });
}

/// World Explorer: virtual visits to countries around the world.
///
/// Teaches: cultural awareness, geography, vocabulary, curiosity about the world.
///
/// Flow:
/// 1. Bot picks a country and introduces it with a greeting.
/// 2. Bot shares fun facts about food, clothing, nature.
/// 3. Child interacts and tries the greeting.
/// 4. Bot asks if they want to explore another country.
class WorldExplorer extends Activity {
  final Random _rng = Random();

  bool _active = false;
  int _countriesExplored = 0;
  int _score = 0;
  int _phase = 0; // 0=intro, 1=food, 2=clothing+nature, 3=greeting practice, 4=done
  _Country? _currentCountry;
  final List<String> _visitedCountries = [];

  static const List<_Country> _countries = [
    _Country(
      name: 'Japan',
      greeting: 'Konnichiwa',
      greetingMeaning: 'hello',
      food: 'In Japan, people eat sushi, that is rice with fish on top. They use chopsticks instead of spoons!',
      clothing: 'People in Japan wear beautiful clothes called kimonos for special festivals.',
      nature: 'Japan has amazing cherry blossom trees that turn pink in spring. They are so beautiful!',
      funFact: 'In Japan, people bow to say hello instead of shaking hands.',
    ),
    _Country(
      name: 'Brazil',
      greeting: 'Ola',
      greetingMeaning: 'hello',
      food: 'In Brazil, people love to eat acai bowls, which are made from purple berries. They are yummy and healthy!',
      clothing: 'During carnival, people in Brazil wear colorful costumes with feathers and sparkles!',
      nature: 'Brazil has the Amazon rainforest, the biggest forest in the world! It is home to parrots, monkeys, and jaguars.',
      funFact: 'Brazil has the most types of butterflies in the whole world!',
    ),
    _Country(
      name: 'Egypt',
      greeting: 'Marhaba',
      greetingMeaning: 'hello',
      food: 'In Egypt, people eat falafel, which are crispy little balls made from beans. They are delicious!',
      clothing: 'Long ago, Egyptian kings called pharaohs wore golden crowns and beautiful robes.',
      nature: 'Egypt has the pyramids, which are giant triangle buildings made thousands of years ago. They also have camels that walk through the desert!',
      funFact: 'The pyramids in Egypt are so old, they were already ancient when dinosaurs were still a memory!',
    ),
    _Country(
      name: 'Kenya',
      greeting: 'Jambo',
      greetingMeaning: 'hello',
      food: 'In Kenya, people eat ugali, which is like a thick porridge made from corn. They eat it with their hands!',
      clothing: 'The Maasai people in Kenya wear bright red cloth and beautiful beaded jewelry.',
      nature: 'Kenya has amazing safaris where you can see lions, elephants, giraffes, and zebras running free!',
      funFact: 'Kenya has the second tallest mountain in all of Africa, called Mount Kenya!',
    ),
    _Country(
      name: 'France',
      greeting: 'Bonjour',
      greetingMeaning: 'hello',
      food: 'In France, people eat croissants for breakfast. They are flaky, buttery, and so yummy!',
      clothing: 'French people are famous for fashion. They sometimes wear cute berets, which are round, flat hats!',
      nature: 'France has fields of lavender flowers that are purple and smell amazing! And the Eiffel Tower in Paris is super tall!',
      funFact: 'The Eiffel Tower in France is as tall as an 81 story building!',
    ),
    _Country(
      name: 'Australia',
      greeting: "G'day",
      greetingMeaning: 'hello',
      food: 'In Australia, people love to have barbecues. They also eat vegemite on toast, which is a brown spread!',
      clothing: 'Australians sometimes wear cork hats, which have little corks hanging from strings to keep flies away!',
      nature: 'Australia has kangaroos that hop around and koalas that sleep in eucalyptus trees. They also have the Great Barrier Reef, the biggest coral reef in the world!',
      funFact: 'Baby kangaroos are called joeys and they ride in their mommy\'s pouch!',
    ),
    _Country(
      name: 'India',
      greeting: 'Namaste',
      greetingMeaning: 'hello',
      food: 'In India, people eat colorful curries with rice and naan bread. The food is full of yummy spices!',
      clothing: 'Women in India wear beautiful saris, which are long colorful cloths wrapped around them.',
      nature: 'India has the Taj Mahal, a beautiful white palace. And elephants are very special in India!',
      funFact: 'India has more than a billion people and they speak over 100 different languages!',
    ),
    _Country(
      name: 'Mexico',
      greeting: 'Hola',
      greetingMeaning: 'hello',
      food: 'In Mexico, people eat tacos, which are tortillas filled with yummy things like beans, cheese, and salsa!',
      clothing: 'During celebrations, people in Mexico wear sombreros, which are big beautiful hats!',
      nature: 'Mexico has colorful butterflies called monarch butterflies that fly thousands of miles every year!',
      funFact: 'In Mexico, they have a fun celebration called Day of the Dead where they remember people they love with flowers and music!',
    ),
  ];

  WorldExplorer();

  // -- Activity metadata --

  @override
  String get id => 'world_explorer';

  @override
  String get name => 'World Explorer';

  @override
  String get category => 'world';

  @override
  String get description =>
      'Take a virtual trip around the world and learn about different countries.';

  @override
  List<String> get skills => ['cultural awareness', 'geography', 'vocabulary', 'curiosity'];

  @override
  int get minAge => 3;

  @override
  int get maxAge => 6;

  @override
  SkillId? get skillId => SkillId.culturalAwareness;

  @override
  Map<String, List<String>> get voiceTriggers => const {
    'en': ['explore the world', 'world explorer', 'visit a country', 'travel game', 'learn about countries'],
    'hi': ['दुनिया घूमो', 'देश के बारे में बताओ', 'यात्रा खेल'],
    'te': ['ప్రపంచం చూద్దాం', 'దేశాల గురించి చెప్పు', 'ప్రయాణం ఆట'],
  };

  @override
  AgeBand get targetAgeBand => AgeBand.nursery;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    return '$_countriesExplored countries explored. Score: $_score.';
  }

  // -- Lifecycle --

  @override
  Future<String> start() async {
    _countriesExplored = 0;
    _score = 0;
    _visitedCountries.clear();
    _active = true;
    _phase = 0;

    return _startNewCountry();
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;

    final lower = childSaid.toLowerCase().trim();

    if (_isQuitTrigger(lower)) {
      return await end();
    }

    final country = _currentCountry!;

    switch (_phase) {
      case 0:
        // After intro, share food fact
        _phase = 1;
        _score += 5;
        return country.food;

      case 1:
        // After food, share clothing + nature
        _phase = 2;
        _score += 5;
        return "${country.clothing} ${country.nature}";

      case 2:
        // Ask them to try the greeting
        _phase = 3;
        _score += 5;
        return "Here is a fun fact: ${country.funFact} "
            "Now, can you say ${country.greeting}? That means ${country.greetingMeaning} in ${country.name}!";

      case 3:
        // They tried the greeting
        _phase = 4;
        _score += 10;
        _countriesExplored++;
        return "Great job! You said ${country.greeting}! Now you know about ${country.name}! "
            "Would you like to explore another country?";

      case 4:
        // Check if they want another
        if (_wantsAnother(lower)) {
          return _startNewCountry();
        }
        return await end();

      default:
        return await end();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    debugPrint('[WorldExplorer] Ended, countries=$_countriesExplored, score=$_score');

    if (_countriesExplored == 0) {
      return "Okay, we'll explore the world another time! There are so many amazing places to visit!";
    }

    final names = _visitedCountries.join(', ');
    return "Wow, you explored $_countriesExplored "
        "countr${_countriesExplored > 1 ? 'ies' : 'y'}: $names! "
        "You are a real world explorer!";
  }

  // -- Helpers --

  String _startNewCountry() {
    // Pick a country not yet visited
    final available = _countries
        .where((c) => !_visitedCountries.contains(c.name))
        .toList();

    if (available.isEmpty) {
      _active = false;
      return "You have explored every country! You are an amazing world explorer! "
          "You visited all ${_countries.length} countries!";
    }

    _currentCountry = available[_rng.nextInt(available.length)];
    _visitedCountries.add(_currentCountry!.name);
    _phase = 0;

    return "Let's explore the world! We are going to ${_currentCountry!.name}! "
        "${_currentCountry!.greeting}! That means ${_currentCountry!.greetingMeaning} "
        "in ${_currentCountry!.name}! Are you ready to learn about this amazing place?";
  }

  bool _wantsAnother(String lower) {
    const triggers = [
      'yes', 'yeah', 'sure', 'okay', 'another', 'more',
      'one more', 'again', 'next', 'explore',
      'हाँ', 'और', 'एक और',
      'అవును', 'ఇంకొకటి', 'మరొకటి',
    ];
    return triggers.any((t) => lower.contains(t));
  }

  bool _isQuitTrigger(String lower) {
    const triggers = [
      'stop the game', 'quit', 'exit', "i'm done", 'no more',
      'stop playing', 'end the game', 'finish', 'no', 'no thanks',
      'बंद करो', 'खेल बंद', 'नहीं',
      'ఆపు', 'ఆట ఆపు', 'వద్దు',
    ];
    return triggers.any((t) => lower.contains(t));
  }
}
