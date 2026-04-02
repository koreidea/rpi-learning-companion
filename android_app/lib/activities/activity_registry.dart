import '../audio/sentence_buffer.dart';
import '../bluetooth/car_chassis.dart';
import '../core/llm/llm_router.dart';
import '../models/skill.dart';
import '../vision/camera_manager.dart';
import '../vision/vision_describer.dart';
import 'activity_base.dart';

// Skill 01 - Critical Thinking
import 's01_critical_thinking/debate_buddy.dart';
import 's01_critical_thinking/mystery_of_the_day.dart';
import 's01_critical_thinking/mystery_solver.dart';
import 's01_critical_thinking/odd_one_out.dart';
import 's01_critical_thinking/sorting_challenge.dart';

// Skill 02 - Creativity
import 's02_creativity/drawing_prompt.dart';
import 's02_creativity/invention_time.dart';
import 's02_creativity/sketch_and_tell.dart';
import 's02_creativity/story_builder.dart';
import 's02_creativity/story_remix.dart';
import 's02_creativity/what_if_machine.dart';

// Skill 03 - Communication
import 's03_communication/interview_game.dart';
import 's03_communication/one_minute_expert.dart';
import 's03_communication/pronunciation_coach.dart';
import 's03_communication/show_and_tell.dart';
import 's03_communication/teach_the_bot.dart';
import 's03_communication/tell_me_about_day.dart';

// Skill 04 - Collaboration
import 's04_collaboration/build_it_together.dart';
import 's04_collaboration/compliment_circle.dart';
import 's04_collaboration/story_relay.dart';

// Skill 05 - Leadership
import 's05_leadership/leader_of_the_day.dart';
import 's05_leadership/what_would_you_do.dart';

// Skill 06 - Emotional Intelligence
import 's06_emotional_intelligence/bot_emotions.dart';
import 's06_emotional_intelligence/emotion_checkin.dart';
import 's06_emotional_intelligence/emotion_detective.dart';
import 's06_emotional_intelligence/manners_roleplay.dart';

// Skill 07 - Adaptability
import 's07_adaptability/new_day_new_way.dart';
import 's07_adaptability/plot_twist.dart';

// Skill 08 - Financial Literacy
import 's08_financial_literacy/budget_boss.dart';
import 's08_financial_literacy/kores_shop.dart';
import 's08_financial_literacy/lemonade_stand.dart';

// Skill 09 - Digital Citizenship
import 's09_digital_citizenship/privacy_guard.dart';
import 's09_digital_citizenship/real_or_fake.dart';
import 's09_digital_citizenship/screen_time_coach.dart';

// Skill 10 - Environmental Awareness
import 's10_environmental/eco_audit.dart';
import 's10_environmental/eco_detective.dart';
import 's10_environmental/future_city.dart';
import 's10_environmental/nature_narrator.dart';

// Skill 11 - Cultural Awareness
import 's11_cultural_awareness/festival_friend.dart';
import 's11_cultural_awareness/language_of_day.dart';
import 's11_cultural_awareness/world_explorer.dart';
import 's11_cultural_awareness/world_traveler.dart';

// Skill 12 - Health & Wellness
import 's12_health_wellness/exercise_buddy.dart';
import 's12_health_wellness/mindful_minute.dart';
import 's12_health_wellness/morning_energizer.dart';
import 's12_health_wellness/nutrition_navigator.dart';
import 's12_health_wellness/sleep_stories.dart';

// Skill 13 - Entrepreneurial Thinking
import 's13_entrepreneurial/market_day.dart';
import 's13_entrepreneurial/pitch_to_kore.dart';
import 's13_entrepreneurial/problem_spotter.dart';

// Skill 14 - Ethics & Civic Responsibility
import 's14_ethics/ethics_engine.dart';
import 's14_ethics/good_citizen.dart';
import 's14_ethics/news_buddy.dart';

// Skill 15 - Design Thinking / Coding
import 's15_design_thinking/debug_the_route.dart';
import 's15_design_thinking/design_detective.dart';
import 's15_design_thinking/loop_game.dart';
import 's15_design_thinking/rapid_prototype.dart';
import 's15_design_thinking/sequence_challenge.dart';

// Skill 16 - Information Literacy
import 's16_information_literacy/ask_better_questions.dart';
import 's16_information_literacy/citation_station.dart';
import 's16_information_literacy/source_safari.dart';

// Skill 17 - Self Direction
import 's17_self_direction/curiosity_hour.dart';
import 's17_self_direction/goal_buddy.dart';
import 's17_self_direction/habit_tracker.dart';

// Skill 18 - Media Creation
import 's18_media_creation/comic_creator.dart';
import 's18_media_creation/kore_news_channel.dart';
import 's18_media_creation/podcast_producer.dart';

// Skill 19 - Scientific Thinking
import 's19_scientific_thinking/hypothesis_of_day.dart';
import 's19_scientific_thinking/lab_partner.dart';
import 's19_scientific_thinking/nature_explorer.dart';
import 's19_scientific_thinking/number_friends.dart';
import 's19_scientific_thinking/science_myth_buster.dart';

// Skill 20 - Time Management
import 's20_time_management/deadline_countdown.dart';
import 's20_time_management/plan_my_day.dart';
import 's20_time_management/pomodoro_buddy.dart';
import 's20_time_management/reflection_time.dart';

/// Registry of all available activities organized by 21st century skills.
///
/// Provides lookup by voice trigger (multi-language), skill ID, category,
/// and unique activity ID. Voice triggers are defined on each activity
/// itself, not hardcoded in this registry.
class ActivityRegistry {
  final List<Activity> _activities = [];

  /// Create the registry, injecting dependencies for various activities.
  ///
  /// [car] - optional car chassis for coding/physical activities.
  /// [llmRouter] - optional LLM router for activities that use the LLM.
  /// [sentenceBuffer] - optional sentence buffer for streaming TTS.
  /// [cameraManager] - optional camera for vision-based activities.
  /// [visionDescriber] - optional vision describer for Show and Tell.
  /// [openaiApiKey] - optional API key for vision API calls.
  /// [childAge] - child's age for difficulty calibration (default 4).
  ActivityRegistry({
    CarChassis? car,
    LlmRouter? llmRouter,
    SentenceBuffer? sentenceBuffer,
    CameraManager? cameraManager,
    VisionDescriber? visionDescriber,
    String openaiApiKey = '',
    int childAge = 4,
  }) {
    _registerSkill01CriticalThinking(
      llmRouter: llmRouter,
      sentenceBuffer: sentenceBuffer,
    );
    _registerSkill02Creativity(
      llmRouter: llmRouter,
      sentenceBuffer: sentenceBuffer,
      cameraManager: cameraManager,
      visionDescriber: visionDescriber,
      openaiApiKey: openaiApiKey,
    );
    _registerSkill03Communication(
      llmRouter: llmRouter,
      sentenceBuffer: sentenceBuffer,
      cameraManager: cameraManager,
      visionDescriber: visionDescriber,
      openaiApiKey: openaiApiKey,
    );
    _registerSkill04Collaboration();
    _registerSkill05Leadership(
      llmRouter: llmRouter,
      sentenceBuffer: sentenceBuffer,
    );
    _registerSkill06EmotionalIntelligence(
      llmRouter: llmRouter,
      sentenceBuffer: sentenceBuffer,
    );
    _registerSkill07Adaptability(
      llmRouter: llmRouter,
      sentenceBuffer: sentenceBuffer,
    );
    _registerSkill08FinancialLiteracy();
    _registerSkill09DigitalCitizenship();
    _registerSkill10Environmental(
      llmRouter: llmRouter,
      sentenceBuffer: sentenceBuffer,
      cameraManager: cameraManager,
      visionDescriber: visionDescriber,
      openaiApiKey: openaiApiKey,
    );
    _registerSkill11CulturalAwareness();
    _registerSkill12HealthWellness(
      car: car,
      llmRouter: llmRouter,
      sentenceBuffer: sentenceBuffer,
    );
    _registerSkill13Entrepreneurial(
      llmRouter: llmRouter,
      sentenceBuffer: sentenceBuffer,
    );
    _registerSkill14Ethics(
      llmRouter: llmRouter,
      sentenceBuffer: sentenceBuffer,
    );
    _registerSkill15DesignThinking(
      car: car,
      llmRouter: llmRouter,
      sentenceBuffer: sentenceBuffer,
      cameraManager: cameraManager,
      visionDescriber: visionDescriber,
    );
    _registerSkill16InformationLiteracy(
      llmRouter: llmRouter,
      sentenceBuffer: sentenceBuffer,
    );
    _registerSkill17SelfDirection(
      llmRouter: llmRouter,
      sentenceBuffer: sentenceBuffer,
    );
    _registerSkill18MediaCreation(
      llmRouter: llmRouter,
      sentenceBuffer: sentenceBuffer,
    );
    _registerSkill19ScientificThinking(
      llmRouter: llmRouter,
      sentenceBuffer: sentenceBuffer,
      cameraManager: cameraManager,
      visionDescriber: visionDescriber,
      openaiApiKey: openaiApiKey,
      childAge: childAge,
    );
    _registerSkill20TimeManagement();
  }

  // -- Registration methods for each skill --

  void _registerSkill01CriticalThinking({
    LlmRouter? llmRouter,
    SentenceBuffer? sentenceBuffer,
  }) {
    _activities.addAll([
      MysterySolver(),
      OddOneOut(),
      SortingChallenge(),
    ]);
    if (llmRouter != null) {
      _activities.add(DebateBuddy(
        llmRouter: llmRouter,
        sentenceBuffer: sentenceBuffer,
      ));
      _activities.add(MysteryOfTheDay(
        llmRouter: llmRouter,
        sentenceBuffer: sentenceBuffer,
      ));
    }
  }

  void _registerSkill02Creativity({
    LlmRouter? llmRouter,
    SentenceBuffer? sentenceBuffer,
    CameraManager? cameraManager,
    VisionDescriber? visionDescriber,
    String openaiApiKey = '',
  }) {
    if (llmRouter != null) {
      _activities.add(WhatIfMachine(
        llmRouter: llmRouter,
        sentenceBuffer: sentenceBuffer,
      ));
      _activities.add(StoryBuilder(llmRouter: llmRouter));
      _activities.add(InventionTime(llmRouter: llmRouter));
      _activities.add(StoryRemix(llmRouter: llmRouter));
    }
    _activities.add(DrawingPrompt(
      camera: cameraManager,
      visionDescriber: visionDescriber,
      apiKey: openaiApiKey,
    ));
    _activities.add(SketchAndTell(
      camera: cameraManager,
      visionDescriber: visionDescriber,
      apiKey: openaiApiKey,
    ));
  }

  void _registerSkill03Communication({
    LlmRouter? llmRouter,
    SentenceBuffer? sentenceBuffer,
    CameraManager? cameraManager,
    VisionDescriber? visionDescriber,
    String openaiApiKey = '',
  }) {
    if (cameraManager != null && visionDescriber != null) {
      _activities.add(ShowAndTell(
        cameraManager: cameraManager,
        visionDescriber: visionDescriber,
        apiKey: openaiApiKey,
      ));
    }
    if (llmRouter != null) {
      _activities.add(TeachTheBot(
        llmRouter: llmRouter,
        sentenceBuffer: sentenceBuffer,
      ));
      _activities.add(InterviewGame(
        llmRouter: llmRouter,
        sentenceBuffer: sentenceBuffer,
      ));
      _activities.add(OneMinuteExpert(llmRouter: llmRouter));
      _activities.add(TellMeAboutDay(
        llmRouter: llmRouter,
        sentenceBuffer: sentenceBuffer,
      ));
    }
    _activities.add(PronunciationCoach());
  }

  void _registerSkill04Collaboration() {
    _activities.addAll([
      BuildItTogether(),
      ComplimentCircle(),
      StoryRelay(),
    ]);
  }

  void _registerSkill05Leadership({
    LlmRouter? llmRouter,
    SentenceBuffer? sentenceBuffer,
  }) {
    if (llmRouter != null) {
      _activities.add(LeaderOfTheDay(llmRouter: llmRouter));
      _activities.add(WhatWouldYouDo(
        llmRouter: llmRouter,
        sentenceBuffer: sentenceBuffer,
      ));
    }
  }

  void _registerSkill06EmotionalIntelligence({
    LlmRouter? llmRouter,
    SentenceBuffer? sentenceBuffer,
  }) {
    _activities.add(BotEmotions());
    if (llmRouter != null) {
      _activities.add(EmotionCheckin(llmRouter: llmRouter));
      _activities.add(MannersRoleplay(llmRouter: llmRouter));
      _activities.add(EmotionDetective(
        llmRouter: llmRouter,
        sentenceBuffer: sentenceBuffer,
      ));
    }
  }

  void _registerSkill07Adaptability({
    LlmRouter? llmRouter,
    SentenceBuffer? sentenceBuffer,
  }) {
    _activities.add(PlotTwist());
    if (llmRouter != null) {
      _activities.add(NewDayNewWay(
        llmRouter: llmRouter,
        sentenceBuffer: sentenceBuffer,
      ));
    }
  }

  void _registerSkill08FinancialLiteracy() {
    _activities.addAll([
      BudgetBoss(),
      KoresShop(),
      LemonadeStand(),
    ]);
  }

  void _registerSkill09DigitalCitizenship() {
    _activities.addAll([
      PrivacyGuard(),
      RealOrFake(),
      ScreenTimeCoach(),
    ]);
  }

  void _registerSkill10Environmental({
    LlmRouter? llmRouter,
    SentenceBuffer? sentenceBuffer,
    CameraManager? cameraManager,
    VisionDescriber? visionDescriber,
    String openaiApiKey = '',
  }) {
    _activities.add(EcoDetective());
    _activities.add(NatureNarrator(
      camera: cameraManager,
      visionDescriber: visionDescriber,
      apiKey: openaiApiKey,
    ));
    if (llmRouter != null) {
      _activities.add(EcoAudit(
        llmRouter: llmRouter,
        sentenceBuffer: sentenceBuffer,
      ));
      _activities.add(FutureCity(
        llmRouter: llmRouter,
        sentenceBuffer: sentenceBuffer,
      ));
    }
  }

  void _registerSkill11CulturalAwareness() {
    _activities.addAll([
      WorldExplorer(),
      WorldTraveler(),
      FestivalFriend(),
      LanguageOfDay(),
    ]);
  }

  void _registerSkill12HealthWellness({
    CarChassis? car,
    LlmRouter? llmRouter,
    SentenceBuffer? sentenceBuffer,
  }) {
    _activities.add(ExerciseBuddy(car: car));
    _activities.add(MorningEnergizer(car: car));
    _activities.add(MindfulMinute());
    if (llmRouter != null) {
      _activities.add(NutritionNavigator(
        llmRouter: llmRouter,
        sentenceBuffer: sentenceBuffer,
      ));
      _activities.add(SleepStories(
        llmRouter: llmRouter,
        sentenceBuffer: sentenceBuffer,
      ));
    }
  }

  void _registerSkill13Entrepreneurial({
    LlmRouter? llmRouter,
    SentenceBuffer? sentenceBuffer,
  }) {
    _activities.add(MarketDay());
    if (llmRouter != null) {
      _activities.add(PitchToKore(
        llmRouter: llmRouter,
        sentenceBuffer: sentenceBuffer,
      ));
      _activities.add(ProblemSpotter(
        llmRouter: llmRouter,
        sentenceBuffer: sentenceBuffer,
      ));
    }
  }

  void _registerSkill14Ethics({
    LlmRouter? llmRouter,
    SentenceBuffer? sentenceBuffer,
  }) {
    _activities.add(GoodCitizen());
    if (llmRouter != null) {
      _activities.add(EthicsEngine(
        llmRouter: llmRouter,
        sentenceBuffer: sentenceBuffer,
      ));
      _activities.add(NewsBuddy(
        llmRouter: llmRouter,
        sentenceBuffer: sentenceBuffer,
      ));
    }
  }

  void _registerSkill15DesignThinking({
    CarChassis? car,
    LlmRouter? llmRouter,
    SentenceBuffer? sentenceBuffer,
    CameraManager? cameraManager,
    VisionDescriber? visionDescriber,
  }) {
    _activities.addAll([
      SequenceChallenge(car: car),
      DebugTheRoute(),
      LoopGame(car: car),
    ]);
    if (llmRouter != null) {
      _activities.add(DesignDetective(
        llmRouter: llmRouter,
        sentenceBuffer: sentenceBuffer,
        camera: cameraManager,
        visionDescriber: visionDescriber,
      ));
      _activities.add(RapidPrototype(
        llmRouter: llmRouter,
        sentenceBuffer: sentenceBuffer,
        camera: cameraManager,
      ));
    }
  }

  void _registerSkill16InformationLiteracy({
    LlmRouter? llmRouter,
    SentenceBuffer? sentenceBuffer,
  }) {
    _activities.addAll([
      CitationStation(),
      SourceSafari(),
    ]);
    if (llmRouter != null) {
      _activities.add(AskBetterQuestions(
        llmRouter: llmRouter,
        sentenceBuffer: sentenceBuffer,
      ));
    }
  }

  void _registerSkill17SelfDirection({
    LlmRouter? llmRouter,
    SentenceBuffer? sentenceBuffer,
  }) {
    _activities.addAll([
      GoalBuddy(),
      HabitTracker(),
    ]);
    if (llmRouter != null) {
      _activities.add(CuriosityHour(
        llmRouter: llmRouter,
        sentenceBuffer: sentenceBuffer,
      ));
    }
  }

  void _registerSkill18MediaCreation({
    LlmRouter? llmRouter,
    SentenceBuffer? sentenceBuffer,
  }) {
    if (llmRouter != null) {
      _activities.add(ComicCreator(
        llmRouter: llmRouter,
        sentenceBuffer: sentenceBuffer,
      ));
      _activities.add(KoreNewsChannel(
        llmRouter: llmRouter,
        sentenceBuffer: sentenceBuffer,
      ));
      _activities.add(PodcastProducer(
        llmRouter: llmRouter,
        sentenceBuffer: sentenceBuffer,
      ));
    }
  }

  void _registerSkill19ScientificThinking({
    LlmRouter? llmRouter,
    SentenceBuffer? sentenceBuffer,
    CameraManager? cameraManager,
    VisionDescriber? visionDescriber,
    String openaiApiKey = '',
    int childAge = 4,
  }) {
    _activities.add(HypothesisOfDay());
    if (llmRouter != null) {
      _activities.add(NatureExplorer(
        camera: cameraManager,
        visionDescriber: visionDescriber,
        llmRouter: llmRouter,
        apiKey: openaiApiKey,
      ));
      _activities.add(LabPartner(
        llmRouter: llmRouter,
        sentenceBuffer: sentenceBuffer,
      ));
      _activities.add(ScienceMythBuster(
        llmRouter: llmRouter,
        sentenceBuffer: sentenceBuffer,
      ));
    }
    _activities.add(NumberFriends(startingAge: childAge));
  }

  void _registerSkill20TimeManagement() {
    _activities.addAll([
      DeadlineCountdown(),
      PlanMyDay(),
      PomodoroBuddy(),
      ReflectionTime(),
    ]);
  }

  // -- Lookup methods --

  /// All registered activities.
  List<Activity> getAll() => List.unmodifiable(_activities);

  /// Activities in a given category.
  List<Activity> getByCategory(String category) =>
      _activities.where((a) => a.category == category).toList();

  /// Get activities for a specific 21st century skill.
  List<Activity> getBySkill(SkillId skillId) =>
      _activities.where((a) => a.skillId == skillId).toList();

  /// Look up an activity by its unique ID. Returns null if not found.
  Activity? getById(String id) {
    for (final a in _activities) {
      if (a.id == id) return a;
    }
    return null;
  }

  /// Find an activity by voice trigger in any supported language.
  ///
  /// Scans all registered activities' [voiceTriggers] maps and returns the
  /// first activity whose trigger phrase is contained in [transcript].
  /// Returns null if no trigger matches.
  Activity? findByVoiceTrigger(String transcript) {
    final lower = transcript.toLowerCase().trim();
    for (final activity in _activities) {
      for (final triggers in activity.voiceTriggers.values) {
        for (final trigger in triggers) {
          if (lower.contains(trigger.toLowerCase())) {
            return activity;
          }
        }
      }
    }
    return null;
  }
}
