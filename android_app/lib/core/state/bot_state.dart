/// Bot state machine — mirrors the Pi's BotState enum.
enum BotState {
  setup,
  loading,
  ready,
  listening,
  processing,
  speaking,
  error,
  sleeping,
}
