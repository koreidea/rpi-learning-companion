import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../../audio/sentence_buffer.dart';
import '../../core/llm/llm_provider.dart';
import '../../core/llm/llm_router.dart';
import '../../models/age_band.dart';
import '../../models/skill.dart';
import '../activity_base.dart';
import '../activity_types.dart';

/// Callback type for streaming TTS -- receives a complete sentence to speak.
typedef SpeakSentenceCallback = Future<void> Function(String sentence);

/// Daily multi-episode puzzle stories that teach critical thinking.
///
/// Each mystery spans 3-5 episodes. In each episode the bot narrates a
/// segment of a real historical or scientific mystery, pauses, and asks
/// the child what they think happened next. The LLM responds to the
/// child's guesses and weaves them into the narrative before continuing.
///
/// Stories include a mix of Indian and global mysteries appropriate for
/// children, covering topics like Mangalyaan, John Snow's cholera
/// investigation, Archimedes, Ramanujan, and more.
class MysteryOfTheDay extends EpisodicActivity {
  final LlmRouter _llmRouter;
  final SentenceBuffer _sentenceBuffer;

  /// Optional callback for streaming TTS.
  SpeakSentenceCallback? onSpeakSentence;

  final Random _random = Random();

  bool _active = false;
  int _currentEpisode = 1;
  int _currentStoryIndex = 0;
  int _phase = 0;
  // Phase 0: intro / set scene
  // Phase 1: narrate + ask guess
  // Phase 2: respond to guess + continue
  // Phase 3: wrap episode

  final List<Map<String, String>> _conversationHistory = [];

  MysteryOfTheDay({
    required LlmRouter llmRouter,
    SentenceBuffer? sentenceBuffer,
    this.onSpeakSentence,
  })  : _llmRouter = llmRouter,
        _sentenceBuffer = sentenceBuffer ?? SentenceBuffer();

  @override
  String get id => 'critical_thinking_mystery_of_the_day';

  @override
  String get name => 'Mystery of the Day';

  @override
  String get category => 'thinking';

  @override
  String get description =>
      'Daily multi-episode mystery stories from history and science.';

  @override
  List<String> get skills =>
      ['critical thinking', 'deductive reasoning', 'curiosity'];

  @override
  int get minAge => 5;

  @override
  int get maxAge => 14;

  @override
  SkillId? get skillId => SkillId.criticalThinking;

  @override
  Map<String, List<String>> get voiceTriggers => const {
        'en': [
          'mystery of the day',
          'daily mystery',
          'mystery story',
          'puzzle story',
          'tell me a mystery',
        ],
        'hi': ['आज की पहेली', 'रहस्य कहानी', 'पहेली कहानी'],
        'te': ['ఈ రోజు రహస్యం', 'రహస్య కథ', 'పజిల్ కథ'],
      };

  @override
  AgeBand get targetAgeBand => AgeBand.junior;

  @override
  bool get isActive => _active;

  @override
  int get currentEpisode => _currentEpisode;

  @override
  String get episodeKey => 'mystery_of_the_day_progress';

  @override
  bool get hasNewEpisode {
    final story = _mysteries[_currentStoryIndex];
    return _currentEpisode <= story.episodes.length;
  }

  @override
  Future<void> saveProgress() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt('${episodeKey}_story', _currentStoryIndex);
    await prefs.setInt('${episodeKey}_episode', _currentEpisode);
    debugPrint(
        '[MysteryOfTheDay] Saved: story=$_currentStoryIndex ep=$_currentEpisode');
  }

  @override
  Future<void> loadProgress() async {
    final prefs = await SharedPreferences.getInstance();
    _currentStoryIndex = prefs.getInt('${episodeKey}_story') ?? 0;
    _currentEpisode = prefs.getInt('${episodeKey}_episode') ?? 1;

    // If we finished all episodes of the current story, move to next
    if (_currentStoryIndex < _mysteries.length) {
      final story = _mysteries[_currentStoryIndex];
      if (_currentEpisode > story.episodes.length) {
        _currentStoryIndex = (_currentStoryIndex + 1) % _mysteries.length;
        _currentEpisode = 1;
      }
    } else {
      _currentStoryIndex = 0;
      _currentEpisode = 1;
    }

    debugPrint(
        '[MysteryOfTheDay] Loaded: story=$_currentStoryIndex ep=$_currentEpisode');
  }

  @override
  Future<String> start() async {
    _active = true;
    _phase = 0;
    _conversationHistory.clear();
    await loadProgress();

    final story = _mysteries[_currentStoryIndex];
    final episodeIndex = _currentEpisode - 1;

    if (episodeIndex >= story.episodes.length) {
      // All episodes done, advance story
      _currentStoryIndex = (_currentStoryIndex + 1) % _mysteries.length;
      _currentEpisode = 1;
      await saveProgress();
      return start();
    }

    final episode = story.episodes[episodeIndex];
    _phase = 1;

    debugPrint(
        '[MysteryOfTheDay] Starting story=${story.title} ep=$_currentEpisode');

    if (_currentEpisode == 1) {
      return "Welcome to Mystery of the Day! Today we begin a new mystery: "
          "${story.title}. ${episode.narration} ${episode.question}";
    }

    return "Welcome back to our mystery: ${story.title}! "
        "Episode $_currentEpisode. ${episode.narration} ${episode.question}";
  }

  @override
  Future<String?> processResponse(String childSaid) async {
    if (!_active) return null;
    final lower = childSaid.toLowerCase().trim();

    if (_containsQuit(lower)) {
      _active = false;
      return await end();
    }

    final story = _mysteries[_currentStoryIndex];
    final episodeIndex = _currentEpisode - 1;

    if (episodeIndex >= story.episodes.length) {
      _active = false;
      return await end();
    }

    final episode = story.episodes[episodeIndex];

    switch (_phase) {
      case 1:
        // Child made a guess -- use LLM to respond
        _conversationHistory.add({'role': 'user', 'content': childSaid});
        _phase = 2;
        final response = await _getLlmResponse(
          story,
          episode,
          childSaid,
          isWrapUp: false,
        );
        _conversationHistory
            .add({'role': 'assistant', 'content': response});
        return "$response What else do you think might have happened?";

      case 2:
        // Second guess or follow-up -- wrap up this episode
        _conversationHistory.add({'role': 'user', 'content': childSaid});
        _phase = 3;
        final response = await _getLlmResponse(
          story,
          episode,
          childSaid,
          isWrapUp: true,
        );

        // Advance episode
        _currentEpisode++;
        await saveProgress();

        if (_currentEpisode > story.episodes.length) {
          _active = false;
          return "$response And that is the end of the mystery: "
              "${story.title}! ${story.conclusion} "
              "You were a brilliant detective! Come back tomorrow for a new mystery!";
        }

        _active = false;
        return "$response Great thinking! That is the end of today's episode. "
            "Come back tomorrow for episode $_currentEpisode to find out what happens next!";

      default:
        return "Tell me what you think happened!";
    }
  }

  @override
  Future<String> end() async {
    _active = false;
    await saveProgress();
    return "Great detective work today! We are on episode $_currentEpisode "
        "of ${_mysteries[_currentStoryIndex].title}. "
        "Come back soon to continue the mystery!";
  }

  @override
  String get progressSummary {
    final story = _mysteries[_currentStoryIndex];
    return 'Story: ${story.title}, Episode $_currentEpisode of ${story.episodes.length}.';
  }

  // -- LLM response generation --

  Future<String> _getLlmResponse(
    _MysteryStory story,
    _MysteryEpisode episode,
    String childSaid, {
    required bool isWrapUp,
  }) async {
    final wrapGuidance = isWrapUp
        ? 'This is the end of the episode. Reveal the key fact: '
            '"${episode.keyFact}". Celebrate the child\'s thinking. '
            'Do not ask a follow-up question.'
        : 'The child just made a guess. Respond with excitement. If their '
            'guess is close to the real answer, praise them. If not, give a '
            'gentle hint without revealing the answer yet.';

    final systemPrompt =
        'You are Buddy, a friendly storytelling detective companion for '
        'children aged 5-14. You are telling the mystery: "${story.title}". '
        'Current episode context: "${episode.narration}" '
        '$wrapGuidance '
        'Rules: Keep response to 2-3 sentences. Be encouraging and excited. '
        'Do not use markdown, bullet points, or emojis. Speak naturally.';

    final messages = <Map<String, String>>[
      {'role': 'system', 'content': systemPrompt},
      ..._conversationHistory,
    ];

    if (_conversationHistory.isEmpty ||
        _conversationHistory.last['content'] != childSaid) {
      messages.add({'role': 'user', 'content': childSaid});
    }

    try {
      final provider = _llmRouter.getProvider();

      if (onSpeakSentence != null) {
        return await _streamWithTts(provider, messages);
      }

      final buffer = StringBuffer();
      await for (final token in provider.stream(messages)) {
        buffer.write(token);
      }
      final result = buffer.toString().trim();
      return result.isNotEmpty
          ? result
          : _fallbackResponse(isWrapUp, episode);
    } catch (e) {
      debugPrint('[MysteryOfTheDay] LLM error: $e');
      return _fallbackResponse(isWrapUp, episode);
    }
  }

  Future<String> _streamWithTts(
    LlmProvider provider,
    List<Map<String, String>> messages,
  ) async {
    _sentenceBuffer.reset();
    final fullResponse = <String>[];

    try {
      await for (final token in provider.stream(messages)) {
        final sentence = _sentenceBuffer.feed(token);
        if (sentence != null) {
          fullResponse.add(sentence);
          await onSpeakSentence!(sentence);
        }
      }

      final remaining = _sentenceBuffer.flush();
      if (remaining != null) {
        fullResponse.add(remaining);
        await onSpeakSentence!(remaining);
      }

      final result = fullResponse.join(' ');
      return result.isNotEmpty ? result : 'That is interesting thinking!';
    } catch (e) {
      if (fullResponse.isNotEmpty) return fullResponse.join(' ');
      return 'That is interesting thinking!';
    }
  }

  String _fallbackResponse(bool isWrapUp, _MysteryEpisode episode) {
    if (isWrapUp) {
      return "Great guess! Here is what actually happened: ${episode.keyFact}";
    }
    return "Interesting idea! That is a really creative guess. Let me tell you a bit more.";
  }

  bool _containsQuit(String text) {
    const quitWords = [
      'quit', 'exit', 'stop', 'done', 'finish', 'no more', 'enough',
      'i want to stop', "i don't want to play", 'end game',
    ];
    return quitWords.any((w) => text.contains(w));
  }

  // -- Mystery story data --

  static const List<_MysteryStory> _mysteries = [
    _MysteryStory(
      title: 'The Cholera Detective',
      conclusion:
          'Doctor John Snow proved that dirty water caused cholera, not bad air. '
          'He changed how we think about diseases forever!',
      episodes: [
        _MysteryEpisode(
          narration:
              'In 1854, people in London were getting very sick. Hundreds of '
              'people in one neighborhood were dying from a disease called cholera. '
              'Everyone thought it was caused by bad smells in the air.',
          question: 'What do you think was really making people sick?',
          keyFact:
              'A doctor named John Snow suspected it was the water, not the air.',
        ),
        _MysteryEpisode(
          narration:
              'Doctor John Snow started making a map. He marked every house '
              'where someone got sick. He noticed something strange. All the '
              'sick people lived near one water pump on Broad Street.',
          question:
              'Why do you think the people near that pump were getting sick?',
          keyFact:
              'The water pump was contaminated with sewage. The dirty water was spreading the disease.',
        ),
        _MysteryEpisode(
          narration:
              'John Snow convinced the town officials to remove the handle '
              'from the water pump so nobody could use it. The number of sick '
              'people started going down almost immediately.',
          question:
              'What do you think this proved about how the disease spread?',
          keyFact:
              'It proved that cholera spread through contaminated water. '
              'This changed medicine forever and led to clean water systems.',
        ),
      ],
    ),
    _MysteryStory(
      title: 'The Mangalyaan Mission',
      conclusion:
          'India became the first country to reach Mars on its very first try, '
          'and did it for less money than it costs to make a Hollywood movie!',
      episodes: [
        _MysteryEpisode(
          narration:
              'In 2013, Indian scientists at ISRO had a dream. They wanted to '
              'send a spacecraft to Mars. But they had a problem. Their budget '
              'was very small compared to other space agencies.',
          question:
              'How do you think they managed to build a Mars mission with so little money?',
          keyFact:
              'ISRO scientists found clever shortcuts. They reused technology '
              'from their moon mission and kept the spacecraft very light.',
        ),
        _MysteryEpisode(
          narration:
              'The rocket they had was not powerful enough to send the spacecraft '
              'directly to Mars. They needed to be really clever about how to '
              'get there. They came up with a brilliant trick.',
          question:
              'What clever trick do you think they used to reach Mars?',
          keyFact:
              'They used Earth\'s gravity like a slingshot! The spacecraft '
              'orbited Earth several times, going faster each time, before '
              'shooting off toward Mars.',
        ),
        _MysteryEpisode(
          narration:
              'After ten months of traveling through space, Mangalyaan reached '
              'Mars. The whole country held its breath. The spacecraft had to '
              'slow down and enter orbit perfectly, or it would fly right past.',
          question: 'Do you think it made it? What could go wrong?',
          keyFact:
              'Mangalyaan entered Mars orbit perfectly on September 24, 2014. '
              'India became the first country to succeed on its first Mars attempt!',
        ),
      ],
    ),
    _MysteryStory(
      title: 'Archimedes and the Golden Crown',
      conclusion:
          'Archimedes discovered that objects displace water equal to their '
          'volume. This is now called the Archimedes Principle!',
      episodes: [
        _MysteryEpisode(
          narration:
              'Long ago in ancient Greece, a king gave a goldsmith some pure '
              'gold to make a crown. When the crown came back, it looked '
              'beautiful. But the king was suspicious. Had the goldsmith '
              'secretly mixed in cheaper silver?',
          question:
              'How would you figure out if the crown was pure gold without breaking it?',
          keyFact:
              'The king asked the great thinker Archimedes to solve this puzzle.',
        ),
        _MysteryEpisode(
          narration:
              'Archimedes thought and thought but could not figure it out. '
              'Then one day, he stepped into a full bathtub. Water spilled '
              'over the sides! He suddenly had an idea and was so excited '
              'he ran through the streets shouting Eureka, which means I found it!',
          question: 'What do you think he figured out from the bathtub?',
          keyFact:
              'He realized that his body pushed water out of the tub. The '
              'amount of water displaced equals the volume of the object.',
        ),
        _MysteryEpisode(
          narration:
              'Archimedes put the crown in water and measured how much water '
              'it displaced. Then he put the same weight of pure gold in water. '
              'If the crown was pure gold, both should displace the same water.',
          question: 'What do you think happened? Was the crown pure gold?',
          keyFact:
              'The crown displaced more water than the pure gold! This meant '
              'silver had been mixed in, because silver is lighter than gold '
              'and takes up more space. The goldsmith was caught!',
        ),
      ],
    ),
    _MysteryStory(
      title: 'The Self-Taught Genius: Ramanujan',
      conclusion:
          'Ramanujan went on to Cambridge and became one of the greatest '
          'mathematicians ever, all because he never stopped being curious.',
      episodes: [
        _MysteryEpisode(
          narration:
              'In a small town in India called Kumbakonam, a young boy named '
              'Srinivasa Ramanujan loved numbers more than anything. He had '
              'almost no math textbooks, just one old book he borrowed.',
          question:
              'How do you think he learned advanced mathematics with just one book?',
          keyFact:
              'Ramanujan worked through every single problem in that book and '
              'then started discovering new math formulas all by himself!',
        ),
        _MysteryEpisode(
          narration:
              'Ramanujan filled notebooks with amazing mathematical formulas '
              'that nobody in India had seen before. He tried to share his work, '
              'but people did not understand it. He was just a poor clerk at an office.',
          question: 'What do you think he did to get people to notice his work?',
          keyFact:
              'He wrote a letter to a famous mathematician in England named '
              'G.H. Hardy, including nine pages of his formulas.',
        ),
        _MysteryEpisode(
          narration:
              'When Professor Hardy in Cambridge got the letter, he was '
              'stunned. Some of the formulas were things mathematicians had '
              'been trying to prove for years. Others were completely new.',
          question:
              'What do you think Hardy did when he saw the letter from an unknown clerk in India?',
          keyFact:
              'Hardy invited Ramanujan to Cambridge! He said the formulas '
              'must be true because nobody could have invented them.',
        ),
      ],
    ),
    _MysteryStory(
      title: 'The Clever Minister: Chanakya',
      conclusion:
          'Chanakya used intelligence over brute force to build one of the '
          'greatest empires in Indian history, the Maurya Empire.',
      episodes: [
        _MysteryEpisode(
          narration:
              'Over 2,000 years ago, a powerful and cruel king named Dhana '
              'Nanda ruled a large part of India. A brilliant teacher named '
              'Chanakya was insulted by this king and vowed to replace him.',
          question: 'How do you think a teacher could defeat a powerful king?',
          keyFact:
              'Chanakya did not use an army. He used his intelligence. He '
              'found a brave young man named Chandragupta and trained him.',
        ),
        _MysteryEpisode(
          narration:
              'Chanakya trained Chandragupta in strategy, warfare, and '
              'leadership. But they still had a huge problem. Nanda had the '
              'largest army in all of India. A direct attack would be impossible.',
          question:
              'What strategy do you think Chanakya used to defeat such a large army?',
          keyFact:
              'Chanakya first weakened the king by turning his own officials '
              'against him. He used spies and alliances instead of brute force.',
        ),
        _MysteryEpisode(
          narration:
              'One by one, provinces started breaking away from Nanda. His '
              'generals switched sides. By the time Chandragupta attacked, '
              'the once-mighty kingdom was already crumbling from within.',
          question:
              'What does this teach us about how brains can be mightier than muscles?',
          keyFact:
              'Chandragupta defeated Nanda and built the Maurya Empire. '
              'Chanakya became his chief advisor and wrote the Arthashastra, '
              'one of the oldest books on strategy.',
        ),
      ],
    ),
    _MysteryStory(
      title: 'The Dinosaur Bone Hunters',
      conclusion:
          'Thanks to fossil hunters, we now know about hundreds of dinosaur '
          'species. Every fossil tells a story millions of years old!',
      episodes: [
        _MysteryEpisode(
          narration:
              'In the 1800s, a young girl named Mary Anning was walking along '
              'cliffs in England when she found strange stone shapes in the '
              'rock. They looked like bones, but they were enormous.',
          question: 'What do you think she had found in those cliffs?',
          keyFact:
              'Mary had found fossils! These were the bones of creatures '
              'that lived millions of years ago, turned to stone over time.',
        ),
        _MysteryEpisode(
          narration:
              'Mary kept searching and found a complete skeleton of a creature '
              'that was 5 meters long with a snout full of teeth. Nobody had '
              'ever seen anything like it. Scientists were baffled.',
          question: 'What kind of creature do you think it was?',
          keyFact:
              'It was an Ichthyosaur, a marine reptile that lived 200 million '
              'years ago! Mary\'s discovery changed science forever.',
        ),
        _MysteryEpisode(
          narration:
              'More fossil hunters started finding giant bones all over the '
              'world. In India, scientists found fossils of dinosaurs in '
              'Gujarat and Madhya Pradesh. The bones told a story of creatures '
              'that ruled the Earth for 165 million years.',
          question:
              'Why do you think the dinosaurs disappeared? What could have happened?',
          keyFact:
              'An asteroid hit Earth 66 million years ago, causing massive '
              'changes to the climate. The dinosaurs could not survive, but '
              'small animals and birds did.',
        ),
      ],
    ),
    _MysteryStory(
      title: 'The Mystery of the Indus Valley Script',
      conclusion:
          'The Indus Valley script remains one of the greatest unsolved '
          'puzzles in history. Maybe one day you will help crack the code!',
      episodes: [
        _MysteryEpisode(
          narration:
              'About 5,000 years ago, one of the world\'s greatest cities '
              'stood where Pakistan and India are today. The Indus Valley '
              'civilization had perfect streets, amazing drainage, and a '
              'mysterious writing system.',
          question:
              'Why do you think nobody can read their writing today?',
          keyFact:
              'The Indus script has never been decoded. We have found over '
              '4,000 objects with the writing, but the texts are very short.',
        ),
        _MysteryEpisode(
          narration:
              'Scientists have found over 400 different symbols. Some look '
              'like animals, some like plants, and some are completely abstract. '
              'The longest text found is only 26 symbols long.',
          question:
              'What do you think these symbols might mean? Are they letters, words, or pictures?',
          keyFact:
              'Nobody knows for sure! Some scientists think they are words, '
              'others think they are names or titles. The texts are too short '
              'to decode using computers.',
        ),
        _MysteryEpisode(
          narration:
              'The civilization suddenly disappeared around 1900 BCE. The '
              'cities were abandoned. With them went any knowledge of how '
              'to read the script. It is one of the last great undeciphered '
              'scripts in human history.',
          question:
              'If you could travel back in time, how would you learn to read their writing?',
          keyFact:
              'To decode a script, you usually need a bilingual text, like '
              'the Rosetta Stone that helped decode Egyptian hieroglyphs. '
              'No such text has been found for the Indus script.',
        ),
      ],
    ),
    _MysteryStory(
      title: 'The Accidental Discovery: Penicillin',
      conclusion:
          'Alexander Fleming\'s messy lab led to antibiotics, which have '
          'saved hundreds of millions of lives around the world!',
      episodes: [
        _MysteryEpisode(
          narration:
              'In 1928, a scientist named Alexander Fleming came back from '
              'vacation to his messy lab in London. He had left some bacteria '
              'growing in dishes. But something strange had happened.',
          question: 'What do you think he found in his messy lab?',
          keyFact:
              'Mold had grown on one of the dishes! But around the mold, '
              'all the bacteria had died. Something in the mold was killing germs.',
        ),
        _MysteryEpisode(
          narration:
              'Fleming was curious about the mold. He grew more of it and '
              'found that it produced a substance that could kill many types '
              'of harmful bacteria. He called it penicillin.',
          question:
              'Why is it important to have something that can kill harmful bacteria?',
          keyFact:
              'Before penicillin, even a small cut could be deadly if it got '
              'infected. There was no medicine to fight bacterial infections.',
        ),
        _MysteryEpisode(
          narration:
              'It took over 10 more years before other scientists figured out '
              'how to make enough penicillin to give to patients. During World '
              'War 2, it saved thousands of soldiers who would have died from '
              'infected wounds.',
          question:
              'What does Fleming\'s story teach us about accidents and curiosity?',
          keyFact:
              'Sometimes great discoveries happen by accident! But it takes a '
              'curious mind to notice something important. Fleming could have '
              'thrown away that moldy dish, but his curiosity saved millions.',
        ),
      ],
    ),
    _MysteryStory(
      title: 'The Invention of Zero',
      conclusion:
          'Indian mathematicians gave the world zero, without which we would '
          'not have computers, space travel, or modern science!',
      episodes: [
        _MysteryEpisode(
          narration:
              'Thousands of years ago, people had a big problem with counting. '
              'If you had 5 apples and gave them all away, how would you write '
              'that you had nothing? There was no number for nothing!',
          question:
              'Why do you think having no symbol for nothing was a problem?',
          keyFact:
              'Without zero, you cannot tell the difference between 1, 10, '
              'and 100. You need a placeholder to show empty positions.',
        ),
        _MysteryEpisode(
          narration:
              'In ancient India, a mathematician named Brahmagupta wrote rules '
              'for using zero in the year 628. He said that any number plus '
              'zero equals the same number, and any number times zero equals zero.',
          question:
              'Can you think of why zero is special? What happens when you add or multiply with zero?',
          keyFact:
              'Brahmagupta was the first person to treat zero as a real number '
              'with its own rules. This was a revolutionary idea!',
        ),
        _MysteryEpisode(
          narration:
              'The idea of zero traveled from India to the Arab world and then '
              'to Europe. It took centuries! European mathematicians were amazed '
              'by this simple but powerful idea from India.',
          question:
              'How do you think zero changed the world? What would be impossible without it?',
          keyFact:
              'Without zero, we could not have modern math, computers, or even '
              'phone numbers. The entire digital world runs on zeros and ones!',
        ),
      ],
    ),
    _MysteryStory(
      title: 'The Puzzle of Gravity',
      conclusion:
          'Newton\'s discovery of gravity explains why planets orbit the sun, '
          'why the moon stays near Earth, and why we stay on the ground!',
      episodes: [
        _MysteryEpisode(
          narration:
              'In 1665, a young man named Isaac Newton was sitting in his '
              'garden when he saw an apple fall from a tree. Everyone had seen '
              'apples fall before, but Newton asked a question nobody else had.',
          question: 'What question do you think Newton asked about the falling apple?',
          keyFact:
              'Newton asked: why does the apple fall DOWN and not sideways or '
              'up? What invisible force pulls it toward the ground?',
        ),
        _MysteryEpisode(
          narration:
              'Newton started thinking bigger. If the Earth pulls the apple '
              'toward it, could the Earth also be pulling the Moon? But if '
              'the Earth pulls the Moon, why does the Moon not fall down and '
              'crash into us?',
          question: 'Why do you think the Moon does not fall into the Earth?',
          keyFact:
              'The Moon IS falling toward Earth! But it is also moving '
              'sideways so fast that it keeps missing. It falls in a circle, '
              'which we call an orbit.',
        ),
        _MysteryEpisode(
          narration:
              'Newton realized that every object in the universe pulls on '
              'every other object. The bigger the object, the stronger the '
              'pull. This is why the Sun holds all the planets in orbit.',
          question:
              'If you went to the Moon, would you weigh more or less? Why?',
          keyFact:
              'You would weigh much less on the Moon because the Moon is '
              'smaller than Earth and has weaker gravity. You could jump '
              'six times higher on the Moon!',
        ),
      ],
    ),
  ];
}

/// A complete mystery story with multiple episodes.
class _MysteryStory {
  final String title;
  final String conclusion;
  final List<_MysteryEpisode> episodes;

  const _MysteryStory({
    required this.title,
    required this.conclusion,
    required this.episodes,
  });
}

/// A single episode within a mystery story.
class _MysteryEpisode {
  final String narration;
  final String question;
  final String keyFact;

  const _MysteryEpisode({
    required this.narration,
    required this.question,
    required this.keyFact,
  });
}
