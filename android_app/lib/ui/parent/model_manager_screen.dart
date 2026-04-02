import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../core/model_manager.dart';

/// Riverpod provider for ModelManager instance.
final modelManagerProvider = Provider<ModelManager>((ref) {
  final manager = ModelManager();
  ref.onDispose(() => manager.dispose());
  return manager;
});

/// Notifier that tracks model states and triggers UI rebuilds.
class ModelStatesNotifier extends StateNotifier<List<ModelState>> {
  final ModelManager _manager;
  Timer? _refreshTimer;

  ModelStatesNotifier(this._manager) : super([]) {
    _init();
  }

  Future<void> _init() async {
    await _manager.init();
    state = _manager.allModelStates;
  }

  /// Refresh model states from disk.
  Future<void> refresh() async {
    await _manager.init();
    state = List.from(_manager.allModelStates);
  }

  /// Start downloading a model. Periodically refreshes state during download.
  Future<void> download(String modelId) async {
    // Start periodic UI refresh during download
    _refreshTimer?.cancel();
    _refreshTimer = Timer.periodic(
      const Duration(milliseconds: 500),
      (_) => state = List.from(_manager.allModelStates),
    );

    await _manager.downloadModel(
      modelId,
      onProgress: (id, progress) {
        // State is refreshed by the timer above
      },
    );

    _refreshTimer?.cancel();
    _refreshTimer = null;
    state = List.from(_manager.allModelStates);
  }

  /// Cancel an ongoing download.
  void cancelDownload(String modelId) {
    _manager.cancelDownload(modelId);
    _refreshTimer?.cancel();
    _refreshTimer = null;
    state = List.from(_manager.allModelStates);
  }

  /// Delete a downloaded model.
  Future<void> delete(String modelId) async {
    await _manager.deleteModel(modelId);
    state = List.from(_manager.allModelStates);
  }

  @override
  void dispose() {
    _refreshTimer?.cancel();
    super.dispose();
  }
}

final modelStatesProvider =
    StateNotifierProvider<ModelStatesNotifier, List<ModelState>>((ref) {
  final manager = ref.watch(modelManagerProvider);
  return ModelStatesNotifier(manager);
});

/// Screen for managing ML model downloads.
///
/// Shows all available models with their status, download/delete controls,
/// and disk space usage.
class ModelManagerScreen extends ConsumerStatefulWidget {
  const ModelManagerScreen({super.key});

  @override
  ConsumerState<ModelManagerScreen> createState() => _ModelManagerScreenState();
}

class _ModelManagerScreenState extends ConsumerState<ModelManagerScreen> {
  String _diskUsage = '...';

  @override
  void initState() {
    super.initState();
    _loadDiskUsage();
  }

  Future<void> _loadDiskUsage() async {
    final manager = ref.read(modelManagerProvider);
    final usage = await manager.getTotalDiskUsageDisplay();
    if (mounted) {
      setState(() => _diskUsage = usage);
    }
  }

  @override
  Widget build(BuildContext context) {
    final models = ref.watch(modelStatesProvider);
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('ML Models'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () {
              ref.read(modelStatesProvider.notifier).refresh();
              _loadDiskUsage();
            },
            tooltip: 'Refresh',
          ),
        ],
      ),
      body: Column(
        children: [
          // Disk usage header
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(16),
            color: theme.colorScheme.surfaceContainerHighest,
            child: Row(
              children: [
                Icon(
                  Icons.storage,
                  color: theme.colorScheme.onSurfaceVariant,
                ),
                const SizedBox(width: 12),
                Text(
                  'Disk usage: $_diskUsage',
                  style: theme.textTheme.bodyLarge?.copyWith(
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                ),
              ],
            ),
          ),
          // Model list
          Expanded(
            child: models.isEmpty
                ? const Center(child: CircularProgressIndicator())
                : ListView.separated(
                    padding: const EdgeInsets.all(16),
                    itemCount: models.length,
                    separatorBuilder: (_, __) => const SizedBox(height: 12),
                    itemBuilder: (context, index) {
                      return _ModelCard(
                        modelState: models[index],
                        onDownload: () {
                          ref
                              .read(modelStatesProvider.notifier)
                              .download(models[index].info.id);
                        },
                        onCancel: () {
                          ref
                              .read(modelStatesProvider.notifier)
                              .cancelDownload(models[index].info.id);
                        },
                        onDelete: () async {
                          final confirm = await showDialog<bool>(
                            context: context,
                            builder: (ctx) => AlertDialog(
                              title: const Text('Delete Model'),
                              content: Text(
                                'Delete ${models[index].info.displayName}? '
                                'You can re-download it later.',
                              ),
                              actions: [
                                TextButton(
                                  onPressed: () =>
                                      Navigator.of(ctx).pop(false),
                                  child: const Text('Cancel'),
                                ),
                                TextButton(
                                  onPressed: () =>
                                      Navigator.of(ctx).pop(true),
                                  child: const Text('Delete'),
                                ),
                              ],
                            ),
                          );
                          if (confirm == true) {
                            await ref
                                .read(modelStatesProvider.notifier)
                                .delete(models[index].info.id);
                            _loadDiskUsage();
                          }
                        },
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }
}

class _ModelCard extends StatelessWidget {
  final ModelState modelState;
  final VoidCallback onDownload;
  final VoidCallback onCancel;
  final VoidCallback onDelete;

  const _ModelCard({
    required this.modelState,
    required this.onDownload,
    required this.onCancel,
    required this.onDelete,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final info = modelState.info;
    final status = modelState.status;

    return Card(
      elevation: 1,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header row: name + status badge
            Row(
              children: [
                Expanded(
                  child: Text(
                    info.displayName,
                    style: theme.textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
                _StatusBadge(status: status),
              ],
            ),
            const SizedBox(height: 4),
            // Description
            Text(
              info.description,
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 8),
            // Size info
            Text(
              'Size: ${modelState.estimatedSizeDisplay}',
              style: theme.textTheme.bodySmall,
            ),
            // Progress bar (during download)
            if (status == ModelStatus.downloading) ...[
              const SizedBox(height: 12),
              LinearProgressIndicator(
                value: modelState.progress > 0 ? modelState.progress : null,
                minHeight: 6,
                borderRadius: BorderRadius.circular(3),
              ),
              const SizedBox(height: 4),
              Text(
                '${modelState.downloadedSizeDisplay} / ${modelState.estimatedSizeDisplay} '
                '(${(modelState.progress * 100).toStringAsFixed(0)}%)',
                style: theme.textTheme.bodySmall,
              ),
            ],
            // Error message
            if (status == ModelStatus.error &&
                modelState.errorMessage != null) ...[
              const SizedBox(height: 8),
              Text(
                modelState.errorMessage!,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.error,
                ),
              ),
            ],
            const SizedBox(height: 12),
            // Action buttons
            Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                if (status == ModelStatus.notDownloaded ||
                    status == ModelStatus.error)
                  FilledButton.icon(
                    onPressed: onDownload,
                    icon: const Icon(Icons.download, size: 18),
                    label: const Text('Download'),
                  ),
                if (status == ModelStatus.downloading)
                  OutlinedButton.icon(
                    onPressed: onCancel,
                    icon: const Icon(Icons.close, size: 18),
                    label: const Text('Cancel'),
                  ),
                if (status == ModelStatus.ready)
                  OutlinedButton.icon(
                    onPressed: onDelete,
                    icon: Icon(
                      Icons.delete_outline,
                      size: 18,
                      color: theme.colorScheme.error,
                    ),
                    label: Text(
                      'Delete',
                      style: TextStyle(color: theme.colorScheme.error),
                    ),
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _StatusBadge extends StatelessWidget {
  final ModelStatus status;

  const _StatusBadge({required this.status});

  @override
  Widget build(BuildContext context) {
    final (String label, Color color) = switch (status) {
      ModelStatus.notDownloaded => ('Not Downloaded', Colors.grey),
      ModelStatus.downloading => ('Downloading', Colors.blue),
      ModelStatus.ready => ('Ready', Colors.green),
      ModelStatus.error => ('Error', Colors.red),
    };

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Text(
        label,
        style: TextStyle(
          fontSize: 11,
          fontWeight: FontWeight.w500,
          color: color,
        ),
      ),
    );
  }
}
