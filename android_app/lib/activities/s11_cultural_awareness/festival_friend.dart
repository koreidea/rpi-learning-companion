import 'dart:math';

import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_types.dart';

/// Data for a festival or celebration.
class _Festival {
  final String name;
  final int month;
  final int day;

  /// Whether the date varies each year (lunar calendar, etc.).
  final bool dateVaries;
  final String story;
  final String traditions;
  final String food;
  final String howKidsCelebrate;

  const _Festival({
    required this.name,
    required this.month,
    required this.day,
    this.dateVaries = false,
    required this.story,
    required this.traditions,
    required this.food,
    required this.howKidsCelebrate,
  });
}

/// Festival Friend: calendar-aware celebration of festivals from around
/// the world.
///
/// Teaches: cultural awareness, history, traditions, empathy, global citizenship.
///
/// When a festival is within 3 days of today, the activity focuses on that
/// festival. Otherwise, it picks a random festival and says
/// "Did you know about this festival?"
///
/// For each festival: the story behind it, traditions, food, and how kids
/// celebrate.
class FestivalFriend extends CalendarActivity {
  final Random _rng = Random();

  bool _active = false;
  int _festivalsExplored = 0;
  int _score = 0;

  /// 0=intro+story, 1=traditions, 2=food, 3=kids celebrate, 4=done
  int _phase = 0;
  _Festival? _currentFestival;
  final List<String> _visitedFestivals = [];

  static const List<_Festival> _festivals = [
    _Festival(
      name: 'Diwali',
      month: 10,
      day: 24,
      dateVaries: true,
      story: 'Diwali is the festival of lights! It celebrates the victory of '
          'light over darkness and good over evil. The story goes that Lord Rama '
          'returned home after defeating an evil king, and people lit lamps '
          'to guide his way.',
      traditions: 'People clean and decorate their homes with beautiful '
          'rangoli patterns made from colored powder. They light rows of clay '
          'lamps called diyas and set off fireworks that fill the sky with colors!',
      food: 'Families make delicious sweets like ladoo, barfi, and gulab jamun. '
          'There are also crunchy snacks like chakli and namkeen. Everyone shares '
          'sweets with neighbors and friends!',
      howKidsCelebrate: 'Kids love lighting sparklers, wearing new clothes, and '
          'getting gifts! They also help make rangoli patterns and eat lots of '
          'yummy sweets. Some kids draw with chalk and make beautiful designs.',
    ),
    _Festival(
      name: 'Holi',
      month: 3,
      day: 14,
      dateVaries: true,
      story: 'Holi is the festival of colors! It celebrates the arrival of '
          'spring and the victory of good over evil. There is a story about a '
          'brave boy named Prahlad who was saved from fire because of his love '
          'and goodness.',
      traditions: 'People throw colored powder and water at each other! Everyone '
          'gets covered in bright pink, blue, yellow, and green. Even strangers '
          'become friends during Holi. People dance, sing, and say happy Holi!',
      food: 'People drink thandai, a sweet milk drink with nuts. They eat '
          'gujiya, which are sweet fried dumplings, and puran poli, a sweet '
          'stuffed flatbread.',
      howKidsCelebrate: 'Kids have the most fun during Holi! They fill water '
          'balloons and water guns with colored water and splash everyone. By '
          'the end of the day, everyone looks like a walking rainbow!',
    ),
    _Festival(
      name: 'Eid',
      month: 4,
      day: 10,
      dateVaries: true,
      story: 'Eid ul-Fitr marks the end of Ramadan, a month when Muslims fast '
          'from sunrise to sunset. It is a celebration of gratitude, sharing, '
          'and togetherness. The word Eid means happiness!',
      traditions: 'People wear their best clothes and go to the mosque for '
          'special prayers. They give gifts and money to those in need, which '
          'is called Zakat. Everyone hugs and says Eid Mubarak!',
      food: 'Families prepare biryani, kebabs, and sheer khurma, a sweet '
          'vermicelli pudding. There are also sewaiyan, dates, and many '
          'different sweets!',
      howKidsCelebrate: 'Kids get new clothes and Eidi, which is gift money '
          'from elders! They visit family, eat delicious food, and play with '
          'cousins. It is like a big family reunion!',
    ),
    _Festival(
      name: 'Christmas',
      month: 12,
      day: 25,
      story: 'Christmas celebrates the birth of Jesus Christ. People around '
          'the world celebrate with love, giving, and kindness. The story says '
          'Santa Claus flies through the sky in a sleigh pulled by reindeer!',
      traditions: 'People decorate Christmas trees with lights and ornaments. '
          'They hang stockings, sing carols, and exchange gifts. Families come '
          'together for a big dinner.',
      food: 'Families eat turkey, ham, Christmas pudding, and gingerbread '
          'cookies. Kids leave cookies and milk for Santa! Hot chocolate with '
          'marshmallows is a favorite drink.',
      howKidsCelebrate: 'Kids wake up early to open presents under the tree! '
          'They write letters to Santa, build snowmen if there is snow, and '
          'sing jingle bells. Making paper snowflakes is fun too!',
    ),
    _Festival(
      name: 'Pongal',
      month: 1,
      day: 14,
      story: 'Pongal is a harvest festival in South India. Farmers thank the '
          'Sun God and nature for a good harvest. The name Pongal means to '
          'boil over, and it symbolizes abundance and prosperity.',
      traditions: 'People cook sweet rice in a clay pot until it overflows, '
          'and everyone shouts Pongal O Pongal! They draw colorful kolam '
          'patterns in front of their homes and decorate cows with flowers.',
      food: 'The main dish is sweet pongal rice cooked with milk, jaggery, '
          'cashews, and raisins. They also make savory pongal and many '
          'other South Indian dishes.',
      howKidsCelebrate: 'Kids help decorate cows with bells and paint. They '
          'play traditional games, fly kites, and watch the rice pot overflow. '
          'Some kids ride on bullock carts!',
    ),
    _Festival(
      name: 'Republic Day',
      month: 1,
      day: 26,
      story: 'Republic Day celebrates the day India adopted its Constitution '
          'in 1950. The Constitution is like a rulebook that says everyone is '
          'equal and has rights. It is a very important day for all Indians!',
      traditions: 'There is a grand parade in New Delhi with soldiers, '
          'decorated trucks from every state, dancers, and fighter jets flying '
          'in formation. The President unfurls the national flag.',
      food: 'Families enjoy special meals together. Many people make sweets '
          'in the colors of the Indian flag: saffron, white, and green!',
      howKidsCelebrate: 'Kids watch the parade on TV, wave flags, and sing the '
          'national anthem. Schools have flag hoisting ceremonies and cultural '
          'programs where kids perform dances and skits.',
    ),
    _Festival(
      name: 'Independence Day',
      month: 8,
      day: 15,
      story: 'Independence Day celebrates the day India became free from '
          'British rule in 1947. Brave freedom fighters like Mahatma Gandhi, '
          'Jawaharlal Nehru, and many others worked hard for this freedom.',
      traditions: 'The Prime Minister raises the flag at the Red Fort in Delhi. '
          'People fly kites in the sky, which symbolizes freedom. Buildings '
          'and streets are decorated with the tricolor.',
      food: 'Families enjoy festive meals. Kids love the tricolor sweets and '
          'ice cream in saffron, white, and green.',
      howKidsCelebrate: 'Kids fly kites, wave flags, and participate in school '
          'programs. They dress up as freedom fighters and learn about the '
          'brave people who fought for India.',
    ),
    _Festival(
      name: 'Onam',
      month: 9,
      day: 5,
      dateVaries: true,
      story: 'Onam is a harvest festival in Kerala. It celebrates the mythical '
          'King Mahabali, who was so good that people believe he visits Kerala '
          'every year during Onam. It is a time of joy and togetherness.',
      traditions: 'People make beautiful flower carpets called pookalam in '
          'front of their homes. Snake boat races are held on rivers with teams '
          'rowing in rhythm. People wear white and gold clothes.',
      food: 'The Onam Sadhya is a grand feast served on banana leaves with '
          'over 20 dishes! There is rice, sambar, avial, payasam, and many more '
          'vegetarian dishes.',
      howKidsCelebrate: 'Kids help make the flower carpet using petals of '
          'different colors. They watch the exciting boat races, play '
          'traditional games, and eat the amazing Sadhya feast!',
    ),
    _Festival(
      name: 'Navratri',
      month: 10,
      day: 3,
      dateVaries: true,
      story: 'Navratri means nine nights. It celebrates the Goddess Durga '
          'and her victory over evil. For nine nights, people dance, pray, '
          'and celebrate goodness winning over darkness.',
      traditions: 'People do a special dance called Garba in Gujarat where '
          'they clap and spin in circles. In other states, people set up '
          'dolls called Golu on steps. Each night honors a different form of '
          'the Goddess.',
      food: 'Many people eat only vegetarian food during Navratri. Special '
          'dishes include sabudana khichdi, kuttu ki puri, and sweet halwa.',
      howKidsCelebrate: 'Kids love the Garba dancing with colorful sticks '
          'called dandiya! They dress up in traditional clothes, set up doll '
          'displays, and visit friends to see their collections.',
    ),
    _Festival(
      name: 'Ganesh Chaturthi',
      month: 9,
      day: 7,
      dateVaries: true,
      story: 'Ganesh Chaturthi celebrates the birthday of Lord Ganesha, the '
          'elephant-headed god of wisdom and new beginnings. People believe '
          'Ganesha removes obstacles and brings good luck.',
      traditions: 'Families bring home clay statues of Ganesha and decorate '
          'them with flowers. After days of celebration, the statues are '
          'immersed in water in a grand procession with drumming and dancing.',
      food: 'The favorite sweet is modak, which are sweet dumplings that '
          'Ganesha loves! People also make ladoo, karanji, and many other '
          'sweets.',
      howKidsCelebrate: 'Kids make their own small Ganesha statues from clay! '
          'They decorate them with paint and flowers. The grand procession '
          'with music and dancing is the most exciting part.',
    ),
    _Festival(
      name: 'Chinese New Year',
      month: 1,
      day: 25,
      dateVaries: true,
      story: 'Chinese New Year celebrates the start of a new year on the '
          'lunar calendar. The story says people used fireworks and the color '
          'red to scare away a monster called Nian. Now it is a time of '
          'family, feasting, and fresh starts!',
      traditions: 'People decorate with red lanterns and banners. Dragon '
          'dancers fill the streets. Families clean their homes to sweep '
          'away bad luck and welcome good fortune.',
      food: 'Families eat dumplings, which look like gold coins for luck. '
          'Fish is served because the word for fish sounds like the word '
          'for abundance. Nian gao is a sticky sweet rice cake.',
      howKidsCelebrate: 'Kids get red envelopes with lucky money inside! '
          'They watch dragon dances, light firecrackers, and stay up late '
          'on New Year Eve. Each year is named after an animal!',
    ),
    _Festival(
      name: 'Thanksgiving',
      month: 11,
      day: 28,
      dateVaries: true,
      story: 'Thanksgiving is celebrated in America. It started when the '
          'Pilgrims and Native Americans shared a feast to celebrate a good '
          'harvest. Now it is a day for families to be thankful for what '
          'they have.',
      traditions: 'Families gather for a big meal. People say what they are '
          'thankful for. There is a big parade in New York City with giant '
          'balloons floating through the streets!',
      food: 'The star of the meal is roasted turkey! There is also cranberry '
          'sauce, mashed potatoes, sweet corn, and pumpkin pie for dessert.',
      howKidsCelebrate: 'Kids make hand turkeys by tracing their hand and '
          'drawing a turkey. They help cook, watch the parade on TV, and '
          'say what they are grateful for. Then everyone eats together!',
    ),
    _Festival(
      name: 'Easter',
      month: 4,
      day: 20,
      dateVaries: true,
      story: 'Easter celebrates the resurrection of Jesus Christ. It is also '
          'a celebration of spring, new life, and hope. The Easter Bunny is '
          'a fun character who brings eggs and treats!',
      traditions: 'People color and decorate eggs in bright patterns. The '
          'Easter Bunny hides eggs for children to find. Churches hold special '
          'services, and many people wear their best spring clothes.',
      food: 'Families enjoy Easter ham, hot cross buns, and lamb. Chocolate '
          'Easter eggs and bunny-shaped treats are the favorite for kids!',
      howKidsCelebrate: 'Kids go on Easter egg hunts in the garden or park! '
          'They color eggs with dyes and stickers, eat chocolate bunnies, '
          'and sometimes make Easter bonnets decorated with flowers.',
    ),
    _Festival(
      name: 'Hanukkah',
      month: 12,
      day: 14,
      dateVaries: true,
      story: 'Hanukkah is the Festival of Lights celebrated by Jewish people. '
          'Long ago, a small amount of oil miraculously lasted eight nights '
          'in the Temple. So Hanukkah is celebrated for eight nights!',
      traditions: 'Each night, a new candle is lit on a special candle holder '
          'called a menorah. Families say prayers, play games, and spend '
          'time together.',
      food: 'People eat foods fried in oil to remember the miracle! Latkes '
          'are crispy potato pancakes, and sufganiyot are jelly-filled donuts.',
      howKidsCelebrate: 'Kids spin a dreidel, which is a four-sided top, and '
          'play a game with chocolate coins! They get a small gift each night '
          'for eight nights and help light the menorah.',
    ),
    _Festival(
      name: 'Baisakhi',
      month: 4,
      day: 13,
      story: 'Baisakhi celebrates the Sikh new year and the formation of the '
          'Khalsa, the Sikh community, by Guru Gobind Singh in 1699. It is '
          'also a harvest festival in Punjab.',
      traditions: 'People visit gurdwaras for special prayers. There are '
          'colorful processions called Nagar Kirtan with music and martial '
          'arts. The energetic Bhangra dance fills the streets with joy.',
      food: 'People share langar, a community meal at the gurdwara that is '
          'free for everyone. Families also make makki ki roti with sarson '
          'ka saag, and sweet halwa.',
      howKidsCelebrate: 'Kids join the processions, try Bhangra dancing, '
          'and eat delicious food at the community meal. Some kids wear '
          'colorful turbans and traditional Punjabi clothes.',
    ),
    _Festival(
      name: 'Buddha Purnima',
      month: 5,
      day: 23,
      dateVaries: true,
      story: 'Buddha Purnima celebrates the birthday of Gautama Buddha, who '
          'taught people about kindness, peace, and the middle path. He sat '
          'under a tree for a long time until he found wisdom.',
      traditions: 'People visit Buddhist temples, meditate, and offer flowers '
          'and candles. They set free caged birds as a symbol of compassion. '
          'Monks chant and share teachings of peace.',
      food: 'Many people eat only vegetarian food on this day. Sweet rice '
          'pudding called kheer is popular, along with fruits and simple meals.',
      howKidsCelebrate: 'Kids visit temples, light candles, and learn about '
          'being kind to everyone. Some kids feed birds and animals. It is a '
          'peaceful, calm day for families.',
    ),
    _Festival(
      name: 'World Environment Day',
      month: 6,
      day: 5,
      story: 'World Environment Day reminds everyone to take care of our planet! '
          'It started in 1973 and is celebrated in over 100 countries. Every year '
          'has a special theme about protecting nature.',
      traditions: 'People plant trees, clean up beaches and parks, and make '
          'promises to use less plastic. Schools and communities organize '
          'activities to learn about saving the environment.',
      food: 'Many people eat locally grown food and try to waste less. Some '
          'families plant their own vegetables in small gardens!',
      howKidsCelebrate: 'Kids plant saplings, make posters about saving the '
          'Earth, and learn about recycling. Some kids make bird feeders or '
          'start small gardens. It is a great day to be a planet protector!',
    ),
    _Festival(
      name: "Children's Day",
      month: 11,
      day: 14,
      story: "Children's Day in India is on November 14, the birthday of "
          "Jawaharlal Nehru, India's first Prime Minister. He loved children "
          'so much that kids called him Chacha Nehru, meaning Uncle Nehru!',
      traditions: 'Schools organize special programs with games, competitions, '
          'and fun activities. Teachers perform skits and dances for the '
          'students. It is a day when kids are the stars!',
      food: 'Schools give out chocolates and sweets. Some schools have picnics '
          'with cake, sandwiches, and juice boxes!',
      howKidsCelebrate: 'Kids get to enjoy fun activities at school instead of '
          'regular classes! There are dance shows, drama performances, games, '
          'and prizes. It is one of the most fun days at school!',
    ),
  ];

  FestivalFriend();

  @override
  String get id => 'cultural_festival_friend';

  @override
  String get name => 'Festival Friend';

  @override
  String get category => 'world';

  @override
  String get description =>
      'Learn about festivals and celebrations from around the world.';

  @override
  List<String> get skills => [
        'cultural awareness',
        'history',
        'traditions',
        'empathy',
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
          'festival friend',
          'tell me about a festival',
          'what festival is today',
          'celebrate',
          'festival game',
          'holiday today',
        ],
        'hi': [
          'त्योहार',
          'आज कौन सा त्योहार है',
          'त्योहार के बारे में बताओ',
        ],
        'te': [
          'పండుగ',
          'ఈ రోజు ఏ పండుగ',
          'పండుగ గురించి చెప్పు',
        ],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.nursery;

  @override
  bool get isActive => _active;

  @override
  String get progressSummary {
    if (_festivalsExplored == 0) return 'No festivals explored yet.';
    return 'Explored $_festivalsExplored festivals. Score: $_score points.';
  }

  @override
  bool isRelevantToday(DateTime date) {
    return _findNearbyFestival(date) != null;
  }

  @override
  String? getTodayEvent(DateTime date) {
    return _findNearbyFestival(date)?.name;
  }

  /// Find a festival within +/- 3 days of the given date.
  _Festival? _findNearbyFestival(DateTime date) {
    for (final f in _festivals) {
      final festivalDate = DateTime(date.year, f.month, f.day);
      final diff = (date.difference(festivalDate).inDays).abs();
      if (diff <= 3) return f;
    }
    return null;
  }

  @override
  Future<String> start() async {
    _festivalsExplored = 0;
    _score = 0;
    _visitedFestivals.clear();
    _active = true;
    _phase = 0;

    final now = DateTime.now();
    final nearby = _findNearbyFestival(now);

    if (nearby != null) {
      _currentFestival = nearby;
      _visitedFestivals.add(nearby.name);
      return 'Guess what! ${nearby.name} is happening right now! '
          '${nearby.story} Want to hear more about it?';
    }

    return _startRandomFestival();
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_isQuitTrigger(lower)) {
      return await end();
    }

    final festival = _currentFestival!;

    switch (_phase) {
      case 0:
        _phase = 1;
        _score += 5;
        return festival.traditions;

      case 1:
        _phase = 2;
        _score += 5;
        return festival.food;

      case 2:
        _phase = 3;
        _score += 5;
        return festival.howKidsCelebrate;

      case 3:
        _phase = 4;
        _score += 10;
        _festivalsExplored++;
        return 'Now you know all about ${festival.name}! That was fun! '
            'Would you like to learn about another festival?';

      case 4:
        if (_wantsAnother(lower)) {
          return _startRandomFestival();
        }
        return await end();

      default:
        return await end();
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    if (_festivalsExplored == 0) {
      return 'The world has so many wonderful celebrations! Come back '
          'anytime to learn about them!';
    }
    final names = _visitedFestivals.join(', ');
    return 'You learned about $_festivalsExplored '
        'festival${_festivalsExplored > 1 ? 's' : ''}: $names! '
        'Score: $_score points! You are a festival expert!';
  }

  String _startRandomFestival() {
    final available = _festivals
        .where((f) => !_visitedFestivals.contains(f.name))
        .toList();

    if (available.isEmpty) {
      _active = false;
      return 'You have learned about every festival! Amazing! '
          'You know so much about celebrations around the world!';
    }

    _currentFestival = available[_rng.nextInt(available.length)];
    _visitedFestivals.add(_currentFestival!.name);
    _phase = 0;

    return 'Did you know about ${_currentFestival!.name}? '
        '${_currentFestival!.story} Want to hear more?';
  }

  bool _wantsAnother(String lower) {
    const triggers = [
      'yes', 'yeah', 'sure', 'okay', 'another', 'more', 'next',
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
