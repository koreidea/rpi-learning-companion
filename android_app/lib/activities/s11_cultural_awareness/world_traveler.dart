import 'dart:math';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';

/// Data for a country with rich cultural content.
class _CountryData {
  final String name;
  final String greeting;
  final String greetingPronunciation;
  final String food;
  final String traditions;
  final String nature;
  final String funFact;
  final String landmark;
  final String greetingPractice;

  const _CountryData({
    required this.name,
    required this.greeting,
    required this.greetingPronunciation,
    required this.food,
    required this.traditions,
    required this.nature,
    required this.funFact,
    required this.landmark,
    required this.greetingPractice,
  });
}

/// World Traveler: deep virtual visits to countries with culture, history,
/// and daily life.
///
/// Teaches: cultural awareness, geography, vocabulary, global citizenship.
///
/// Flow per country:
/// 1. Greeting introduction and pronunciation.
/// 2. Food and cuisine.
/// 3. Traditions and daily life.
/// 4. Nature and landmarks.
/// 5. Fun fact and greeting practice.
///
/// Randomly selects a country each session from a pool of 12 countries,
/// each with multi-phase rich content.
class WorldTraveler extends Activity {
  final Random _rng = Random();

  bool _active = false;
  int _countriesVisited = 0;
  int _score = 0;

  /// Current phase within a country visit.
  /// 0=greeting, 1=food, 2=traditions, 3=nature+landmark, 4=funFact+practice, 5=done
  int _phase = 0;
  _CountryData? _currentCountry;
  final List<String> _visitedCountries = [];

  static const List<_CountryData> _countries = [
    _CountryData(
      name: 'Japan',
      greeting: 'Konnichiwa',
      greetingPronunciation: 'koh-nee-chee-wah',
      food: 'In Japan, people eat sushi, which is rice with fish on top, and '
          'ramen, which is a warm noodle soup. They use chopsticks to eat!',
      traditions: 'Japanese people fold paper into beautiful shapes called '
          'origami. They make cranes, flowers, and even dragons! During spring, '
          'everyone goes outside to look at the pink cherry blossom trees.',
      nature: 'Japan has tall mountains and beautiful gardens. Mount Fuji is '
          'the tallest mountain and it looks like a perfect triangle with snow on top!',
      funFact: 'Japan has super fast bullet trains called Shinkansen. They '
          'can go as fast as 300 kilometers per hour! That is faster than a race car!',
      landmark: 'The traditional Japanese robe is called a kimono. People '
          'wear kimonos during festivals and special celebrations.',
      greetingPractice: 'In Japan, people bow to say hello. Can you try bowing '
          'and saying Konnichiwa?',
    ),
    _CountryData(
      name: 'Brazil',
      greeting: 'Ola',
      greetingPronunciation: 'oh-lah',
      food: 'In Brazil, people love acai bowls made from purple berries, and '
          'they eat rice and beans every day. Brazilian barbeque, called churrasco, '
          'is famous all over the world!',
      traditions: 'Brazil has the biggest party in the world called Carnival! '
          'People wear colorful costumes with feathers and dance the samba in the streets. '
          'Brazilians also love football more than any other sport.',
      nature: 'Brazil has the Amazon rainforest, the biggest forest in the world! '
          'It is home to parrots, monkeys, jaguars, and millions of types of insects.',
      funFact: 'The Sugarloaf Mountain in Rio de Janeiro looks like a giant sugar cube '
          'sticking up from the ocean. You can ride a cable car to the top!',
      landmark: 'A giant statue called Christ the Redeemer stands on top of a mountain '
          'in Rio. It has arms spread wide and can be seen from everywhere in the city.',
      greetingPractice: 'In Brazil, people often greet each other with a hug. '
          'Can you wave and say Ola?',
    ),
    _CountryData(
      name: 'Egypt',
      greeting: 'Marhaba',
      greetingPronunciation: 'mar-ha-ba',
      food: 'In Egypt, people eat falafel, crispy balls made from beans, and '
          'koshari, a mix of rice, pasta, and lentils. They also love sweet dates!',
      traditions: 'Long ago, Egyptian kings called pharaohs built huge pyramids. '
          'They also wrote using pictures called hieroglyphics instead of letters! '
          'People still come from all over the world to see these amazing things.',
      nature: 'Egypt has the Nile, the longest river in the world! Farmers use '
          'its water to grow food in the middle of the desert. Camels walk through '
          'the sandy desert like ships sailing on sand.',
      funFact: 'The ancient Egyptians invented toothpaste thousands of years ago! '
          'They also loved cats so much that they treated them like royalty.',
      landmark: 'The Great Pyramid of Giza is one of the Seven Wonders of the '
          'Ancient World. It is made of over two million stone blocks!',
      greetingPractice: 'In Egypt, people place their hand on their heart when '
          'greeting someone. Can you try that and say Marhaba?',
    ),
    _CountryData(
      name: 'Kenya',
      greeting: 'Jambo',
      greetingPronunciation: 'jahm-bo',
      food: 'In Kenya, people eat ugali, which is like thick porridge made from '
          'corn, and nyama choma, which means roasted meat. They often eat together '
          'as a big family!',
      traditions: 'The Maasai people in Kenya are famous for their jumping dance. '
          'Warriors compete to see who can jump the highest! They also wear bright '
          'red cloth and beautiful beaded jewelry.',
      nature: 'Kenya has amazing safaris where you can see lions, elephants, '
          'giraffes, and zebras running free in the wild! The Great Rift Valley '
          'stretches across Kenya like a giant crack in the earth.',
      funFact: 'Kenya has the second tallest mountain in Africa, called Mount Kenya. '
          'Also, Kenyan runners are some of the fastest in the whole world!',
      landmark: 'Baobab trees in Kenya can live for thousands of years and their '
          'trunks can be as wide as a house! People call them upside-down trees '
          'because their branches look like roots.',
      greetingPractice: 'In Kenya, people shake hands warmly and smile big. '
          'Can you shake your hand and say Jambo?',
    ),
    _CountryData(
      name: 'France',
      greeting: 'Bonjour',
      greetingPronunciation: 'bon-zhoor',
      food: 'In France, people eat flaky croissants for breakfast and baguettes, '
          'which are long crunchy bread. French cheese is famous, and they have '
          'over 400 different kinds!',
      traditions: 'France has the Tour de France, the biggest bicycle race in the '
          'world! Riders pedal through mountains and fields for three whole weeks. '
          'French people also love art and have some of the best museums.',
      nature: 'France has fields of lavender flowers that are purple and smell '
          'amazing. The French countryside has rolling hills, vineyards, and castles.',
      funFact: 'The Eiffel Tower in Paris is as tall as an 81 story building! '
          'It was supposed to be taken down after 20 years but people loved it so much, '
          'they kept it forever.',
      landmark: 'The Louvre Museum in Paris has the Mona Lisa, the most famous '
          'painting in the world. The museum is so big, it would take days to see everything!',
      greetingPractice: 'French people sometimes wear cute berets, which are round '
          'flat hats. Can you pretend to tip a hat and say Bonjour?',
    ),
    _CountryData(
      name: 'Australia',
      greeting: "G'day",
      greetingPronunciation: 'guh-day',
      food: 'In Australia, people love barbecues. They also eat vegemite on toast, '
          'which is a dark brown spread. And they put shrimp on the barbie, which '
          'means they grill prawns!',
      traditions: 'Australians play a unique instrument called the didgeridoo, '
          'which makes a deep humming sound. Aboriginal Australians have been '
          'making art and music for over 50,000 years, making it one of the oldest '
          'cultures in the world!',
      nature: 'Australia has kangaroos that hop around and koalas that sleep in '
          'eucalyptus trees for 22 hours a day! The Great Barrier Reef is the '
          'biggest coral reef in the world, full of colorful fish.',
      funFact: 'Aboriginal Australians invented the boomerang, a curved stick that '
          'comes back to you when you throw it! They used it for hunting.',
      landmark: 'The Sydney Opera House looks like giant white sails on the water. '
          'It is one of the most famous buildings in the whole world!',
      greetingPractice: 'Australians are very friendly and casual. Can you wave '
          "and say G'day?",
    ),
    _CountryData(
      name: 'India',
      greeting: 'Namaste',
      greetingPronunciation: 'nah-mah-stay',
      food: 'In India, people eat colorful curries with rice and naan bread. '
          'The food is full of yummy spices like turmeric, cumin, and cardamom. '
          'Indian sweets like gulab jamun are super delicious!',
      traditions: 'India has so many festivals! Diwali is the festival of lights '
          'where people light lamps everywhere. Holi is the festival of colors where '
          'people throw colored powder at each other! Cricket is the most popular sport.',
      nature: 'India has the beautiful Himalaya mountains, where the tallest peaks '
          'in the world touch the clouds. Indian elephants are gentle giants that '
          'are treated with great respect.',
      funFact: 'India has more than a billion people and they speak over 100 different '
          'languages! The Taj Mahal took over 20 years and 20,000 workers to build.',
      landmark: 'Women in India wear beautiful saris, which are long colorful cloths '
          'wrapped elegantly around them. Each region has its own special style!',
      greetingPractice: 'In India, people press their palms together and bow slightly. '
          'Can you press your hands together and say Namaste?',
    ),
    _CountryData(
      name: 'Mexico',
      greeting: 'Hola',
      greetingPronunciation: 'oh-la',
      food: 'In Mexico, people eat tacos filled with beans, cheese, and salsa. '
          'They also make guacamole from avocados and eat churros, which are '
          'cinnamon sugar sticks. Yum!',
      traditions: 'Mexico has a beautiful celebration called Day of the Dead '
          'where people remember loved ones with flowers, music, and face painting. '
          'Kids love pinatas, which are colorful figures filled with candy!',
      nature: 'Mexico has amazing beaches, deserts, and jungles. Monarch butterflies '
          'fly thousands of miles every year to spend winter in Mexican forests!',
      funFact: 'The ancient Aztecs built giant pyramids in Mexico, just like in Egypt! '
          'They also invented chocolate! Mariachi bands play happy music with '
          'trumpets and guitars.',
      landmark: 'The Chichen Itza pyramid is so cleverly built that during the '
          'equinox, the shadow of a snake appears to slither down its steps!',
      greetingPractice: 'Mexican people are very warm and friendly. Can you smile '
          'big and say Hola?',
    ),
    _CountryData(
      name: 'China',
      greeting: 'Ni Hao',
      greetingPronunciation: 'nee-how',
      food: 'In China, people eat dumplings, which are little pockets of dough '
          'filled with meat and vegetables. They also eat noodles and rice with '
          'chopsticks, just like in Japan!',
      traditions: 'Chinese New Year is the biggest celebration in China. People '
          'set off fireworks, do dragon dances, and give red envelopes with lucky '
          'money inside! Dragon boat races are exciting too!',
      nature: 'China has giant pandas that eat bamboo all day long. They are black '
          'and white and so cute! The mountains of China often have mist floating '
          'around them like clouds.',
      funFact: 'The Great Wall of China is so long, it would take about 18 months '
          'to walk from one end to the other! Ancient people built it to protect '
          'their land. China is also where kung fu was invented!',
      landmark: 'The Forbidden City in Beijing was home to emperors for 500 years. '
          'It has 9,999 rooms!',
      greetingPractice: 'In China, a slight nod with a smile is a friendly greeting. '
          'Can you nod and say Ni Hao?',
    ),
    _CountryData(
      name: 'Russia',
      greeting: 'Privet',
      greetingPronunciation: 'pree-vyet',
      food: 'In Russia, people eat borscht, a bright purple soup made from beets. '
          'They also eat blini, which are thin pancakes, and pirozhki, little '
          'stuffed bread rolls.',
      traditions: 'Russia has beautiful matryoshka dolls, which are wooden dolls '
          'that fit inside each other, getting smaller and smaller! Russian ballet '
          'is famous all over the world for being graceful and beautiful.',
      nature: 'Siberia in Russia is one of the coldest places on Earth. It gets '
          'so cold that you can throw hot water in the air and it freezes instantly! '
          'Russia also has huge forests called taiga.',
      funFact: 'Russia was the first country to send a person into space! '
          'Yuri Gagarin flew around the Earth in 1961. Russia also has the longest '
          'railway in the world, the Trans-Siberian Railway.',
      landmark: 'The Kremlin in Moscow has colorful onion-shaped towers that look '
          'like ice cream swirls. It is where the government works.',
      greetingPractice: 'Russians often greet friends with a big warm handshake. '
          'Can you extend your hand and say Privet?',
    ),
    _CountryData(
      name: 'South Africa',
      greeting: 'Sawubona',
      greetingPronunciation: 'sah-woo-boh-nah',
      food: 'In South Africa, people love braai, which is a big outdoor barbeque. '
          'They also eat biltong, which is dried meat, and bobotie, a spiced '
          'meat dish with an egg topping.',
      traditions: 'South Africa is called the Rainbow Nation because it has so many '
          'different cultures living together. Nelson Mandela helped bring everyone '
          'together and taught the world about forgiveness and peace.',
      nature: 'South Africa has amazing safaris just like Kenya! You can see the '
          'Big Five: lions, elephants, rhinos, leopards, and buffalo. The ocean '
          'around South Africa has great white sharks!',
      funFact: 'Table Mountain in Cape Town is a flat-topped mountain that looks '
          'like a giant table. Sometimes clouds sit on top like a tablecloth! '
          'South Africa is the only country with three capital cities.',
      landmark: 'Robben Island is where Nelson Mandela was kept in prison for '
          '27 years. After he was freed, he became the president and changed '
          'the country forever.',
      greetingPractice: 'Sawubona means I see you in Zulu, and it is a beautiful '
          'way to say hello. Can you look at me and say Sawubona?',
    ),
    _CountryData(
      name: 'Italy',
      greeting: 'Ciao',
      greetingPronunciation: 'chow',
      food: 'In Italy, people eat pizza and pasta! Italy is where pizza was '
          'invented. They also love gelato, which is Italian ice cream that comes '
          'in amazing flavors like pistachio and stracciatella.',
      traditions: 'Italy has beautiful art everywhere. Leonardo da Vinci, who '
          'painted the Mona Lisa, was Italian! Italians love to eat meals together '
          'as a family, and dinner can last for hours with many courses.',
      nature: 'Italy has beautiful coastlines, rolling hills with olive trees, '
          'and even a volcano called Mount Vesuvius! Long ago, it buried an entire '
          'city called Pompeii in ash.',
      funFact: 'The Colosseum in Rome is almost 2,000 years old! Ancient Romans '
          'watched gladiator battles there. Venice is a city built on water where '
          'people use boats called gondolas instead of cars!',
      landmark: 'The Leaning Tower of Pisa actually leans to one side! It took '
          'almost 200 years to build because it kept tilting.',
      greetingPractice: 'Italians are very expressive and often use their hands '
          'when they talk. Can you wave your hand and say Ciao?',
    ),
  ];

  WorldTraveler();

  @override
  String get id => 'cultural_world_traveler';

  @override
  String get name => 'World Traveler';

  @override
  String get category => 'world';

  @override
  String get description =>
      'Take deep virtual visits to countries and learn about their culture, '
      'history, food, and daily life.';

  @override
  List<String> get skills => [
        'cultural awareness',
        'geography',
        'vocabulary',
        'global citizenship',
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
          'world traveler',
          'travel the world',
          'visit a country',
          'learn about countries',
          'tell me about a country',
          'virtual trip',
        ],
        'hi': [
          'दुनिया की सैर',
          'देश घूमो',
          'देश के बारे में बताओ',
        ],
        'te': [
          'ప్రపంచ యాత్ర',
          'దేశం చూద్దాం',
          'దేశాల గురించి',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.nursery;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_countriesVisited == 0) return 'No countries visited yet.';
    return 'Visited $_countriesVisited countries. Score: $_score points.';
  }

  @override
  Future<String> start() async {
    _countriesVisited = 0;
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
        // After greeting intro, share food
        _phase = 1;
        _score += 5;
        return country.food;

      case 1:
        // After food, share traditions
        _phase = 2;
        _score += 5;
        return country.traditions;

      case 2:
        // After traditions, share nature and landmark
        _phase = 3;
        _score += 5;
        return '${country.nature} ${country.landmark}';

      case 3:
        // Fun fact and greeting practice
        _phase = 4;
        _score += 5;
        return 'Here is a fun fact! ${country.funFact} '
            '${country.greetingPractice}';

      case 4:
        // They tried the greeting, celebrate and ask about next country
        _phase = 5;
        _score += 10;
        _countriesVisited++;
        return 'Amazing! You said ${country.greeting} like a real '
            '${country.name} local! You now know so much about ${country.name}! '
            'Would you like to fly to another country?';

      case 5:
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
    if (_countriesVisited == 0) {
      return 'No worries! The world is waiting for you whenever you are ready '
          'to explore!';
    }
    final names = _visitedCountries.join(', ');
    return 'What an incredible journey! You visited $_countriesVisited '
        'countr${_countriesVisited > 1 ? 'ies' : 'y'}: $names! '
        'You earned $_score points and learned so much about the world! '
        'You are a true world traveler!';
  }

  // -- Helpers --

  String _startNewCountry() {
    final available = _countries
        .where((c) => !_visitedCountries.contains(c.name))
        .toList();

    if (available.isEmpty) {
      _active = false;
      return 'You have visited every country on our list! You are an incredible '
          'world traveler! You explored all ${_countries.length} countries!';
    }

    _currentCountry = available[_rng.nextInt(available.length)];
    _visitedCountries.add(_currentCountry!.name);
    _phase = 0;

    return 'Pack your bags! We are flying to ${_currentCountry!.name}! '
        '${_currentCountry!.greeting}! That is how people say hello in '
        '${_currentCountry!.name}. You say it like ${_currentCountry!.greetingPronunciation}. '
        'Are you ready to explore this amazing place?';
  }

  bool _wantsAnother(String lower) {
    const triggers = [
      'yes', 'yeah', 'sure', 'okay', 'another', 'more', 'next',
      'one more', 'again', 'explore', 'fly',
    ];
    return triggers.any((t) => lower.contains(t));
  }

  bool _isQuitTrigger(String lower) {
    const triggers = [
      'stop', 'quit', 'exit', "i'm done", 'no more',
      'finish', 'no', 'no thanks', 'end',
    ];
    return triggers.any((t) => lower.contains(t));
  }
}
