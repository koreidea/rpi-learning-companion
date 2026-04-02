/// Abstract interface for LLM providers (cloud and local).
abstract class LlmProvider {
  /// Stream tokens from the LLM.
  Stream<String> stream(List<Map<String, String>> messages);

  /// Provider name for logging.
  String get name;
}
