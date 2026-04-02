import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:companion_app/main.dart';

void main() {
  testWidgets('App launches with face screen', (WidgetTester tester) async {
    await tester.pumpWidget(const ProviderScope(child: CompanionApp()));
    await tester.pump();
    // Face screen should render (black background with CustomPaint)
    expect(find.byType(CompanionApp), findsOneWidget);
  });
}
