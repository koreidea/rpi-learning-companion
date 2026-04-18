"""Professional-quality line art drawings for projector sketch/tracing mode.

Realistic, detailed drawings with many polyline control points.
Designed for user-testing demos with kids and parents.

Each drawing has steps with spoken instructions and drawing primitives.
All coordinates on a 0-100 scale, mapped to screen pixels at render time.

Primitive types:
  line:    start=(x,y), end=(x,y)
  circle:  center=(x,y), radius=float
  ellipse: rect=(x,y,w,h)
  arc:     rect=(x,y,w,h), start_angle=float, end_angle=float (radians)
  lines:   points=[(x,y),...], closed=bool  (polyline/polygon)
  dot:     center=(x,y), radius=float (filled circle)
  dashed:  points=[(x,y),...] (dashed polyline — for fold lines in crafts)
"""

import math

PI = math.pi

# ═══════════════════════════════════════════════════════════════
# REALISTIC DOG — Sitting Labrador Retriever, side profile
# Detailed contour with ~200+ coordinate points across 8 steps
# ═══════════════════════════════════════════════════════════════

REALISTIC_DOG = {
    "name": "Realistic Dog (Labrador)",
    "steps": [
        # ── Step 1: Head outline ──────────────────────────────
        {
            "instruction": "Let's start with the dog's head. Trace the outline of the skull and muzzle carefully.",
            "elements": [
                # Skull + forehead + muzzle outline (top of head → nose)
                {"type": "lines", "points": [
                    (38, 22), (39, 20), (40, 18.5), (41.5, 17), (43, 16),
                    (45, 15.5), (47, 15.5), (49, 16), (50.5, 16.5),
                    (52, 17.5), (53, 19), (53.5, 20.5), (53.5, 22),
                    # Forehead dip to muzzle
                    (53, 23.5), (52, 25), (51, 26), (50.5, 27),
                    (50.5, 28), (51, 29), (52, 29.5), (53, 30),
                    (54, 30.5), (54.5, 31), (54.5, 32), (54, 33),
                    # Nose tip
                    (53.5, 33.5), (52.5, 34), (51.5, 34), (50.5, 33.5),
                ], "closed": False},
                # Lower jaw (nose → chin → throat)
                {"type": "lines", "points": [
                    (50.5, 33.5), (50, 34.5), (49.5, 35.5),
                    (48.5, 36), (47.5, 36), (46.5, 35.5),
                    (45.5, 35), (44.5, 35), (43.5, 35.5),
                    (42, 36), (41, 36.5), (40, 36.5),
                    (39, 36), (38, 35),
                ], "closed": False},
            ]
        },
        # ── Step 2: Ear ──────────────────────────────────────
        {
            "instruction": "Now trace the floppy ear hanging down from the head.",
            "elements": [
                {"type": "lines", "points": [
                    # Ear starts from top of head
                    (43, 17), (42, 17.5), (41, 18.5), (40, 20),
                    (39, 22), (38, 24), (37, 26), (36.5, 28),
                    (36, 30), (36, 32), (36.5, 34), (37, 35.5),
                    (37.5, 36.5), (38, 37.5), (38.5, 38),
                    # Ear bottom curve
                    (39, 38.5), (39.5, 38.5), (40, 38),
                    (40.5, 37.5), (41, 37), (41, 36.5),
                    # Back up along inner edge
                    (40.5, 35.5), (40, 34), (39.5, 32),
                    (39, 30), (38.5, 28), (38.5, 26),
                    (38.5, 24), (38.5, 22.5), (38, 22),
                ], "closed": False},
            ]
        },
        # ── Step 3: Eye, nose detail, mouth ───────────────────
        {
            "instruction": "Add the eye, nose, and mouth details. Take your time with the eye.",
            "elements": [
                # Eye outline (almond shape)
                {"type": "lines", "points": [
                    (46, 24), (47, 23), (48, 22.5), (49, 22.5),
                    (50, 23), (50.5, 24), (50, 25), (49, 25.5),
                    (48, 25.5), (47, 25), (46, 24),
                ], "closed": True},
                # Pupil
                {"type": "dot", "center": (48.5, 24), "radius": 1.2},
                # Eye highlight
                {"type": "dot", "center": (49, 23.5), "radius": 0.4},
                # Nose (filled rounded shape)
                {"type": "lines", "points": [
                    (50.5, 31), (51, 30.5), (52, 30), (53, 30.5),
                    (53.5, 31), (54, 32), (53.5, 33), (52.5, 33.5),
                    (51.5, 33.5), (50.5, 33), (50.5, 31),
                ], "closed": True},
                # Nostril
                {"type": "dot", "center": (51.5, 32), "radius": 0.5},
                {"type": "dot", "center": (53, 32), "radius": 0.5},
                # Mouth line
                {"type": "lines", "points": [
                    (50.5, 33.5), (49.5, 34), (48, 34.5),
                    (46.5, 35), (45, 35), (44, 34.5),
                ], "closed": False},
                # Eyebrow ridge
                {"type": "lines", "points": [
                    (45.5, 22.5), (46.5, 21.5), (48, 21), (49.5, 21),
                    (50.5, 21.5), (51, 22),
                ], "closed": False},
            ]
        },
        # ── Step 4: Neck and chest ────────────────────────────
        {
            "instruction": "Now draw the neck going down and the chest curving forward.",
            "elements": [
                # Back of neck (from head down)
                {"type": "lines", "points": [
                    (38, 22), (37.5, 24), (37, 27), (36.5, 30),
                    (36, 33), (35.5, 36), (35, 39), (34.5, 42),
                    (34, 44), (33.5, 46),
                ], "closed": False},
                # Front of neck + chest (from jaw down)
                {"type": "lines", "points": [
                    (38, 35), (39, 37), (40, 39),
                    (41, 41), (42, 43), (43, 45),
                    (44, 47), (45, 49), (46, 51),
                    (47, 53), (47.5, 55), (48, 57),
                    (48, 59), (47.5, 61),
                ], "closed": False},
                # Chest fur texture lines
                {"type": "lines", "points": [
                    (41, 42), (42, 44), (41.5, 46),
                ], "closed": False},
                {"type": "lines", "points": [
                    (43, 44), (44, 46), (43.5, 48),
                ], "closed": False},
            ]
        },
        # ── Step 5: Back and body ─────────────────────────────
        {
            "instruction": "Trace the back from the neck all the way down to the hip.",
            "elements": [
                # Back line (neck → shoulder → back → hip)
                {"type": "lines", "points": [
                    (33.5, 46), (33, 48), (32.5, 50),
                    (32, 52), (31.5, 54), (31, 56),
                    # Slight dip at shoulder blade
                    (30.5, 58), (30, 60), (29.5, 62),
                    # Back rises slightly
                    (29, 64), (28.5, 66), (28.5, 68),
                    (29, 70), (29.5, 71), (30, 72),
                    # Hip curve
                    (31, 73), (32, 73.5), (33, 74),
                    (34, 74), (35, 73.5),
                ], "closed": False},
                # Belly line
                {"type": "lines", "points": [
                    (47.5, 61), (46, 63), (44.5, 64.5),
                    (43, 65.5), (41, 66), (39, 66),
                    (37, 66.5), (35, 67), (34, 68),
                    (33, 69), (32.5, 70),
                ], "closed": False},
            ]
        },
        # ── Step 6: Front legs ────────────────────────────────
        {
            "instruction": "Draw the two front legs. The near leg is in front, and the far one peeks behind.",
            "elements": [
                # Near front leg (full detail)
                {"type": "lines", "points": [
                    (47.5, 61), (47, 63), (46.5, 65),
                    (46, 67), (46, 69), (46, 71),
                    (46, 73), (46, 75), (46, 77),
                    # Paw
                    (45.5, 78.5), (45, 79.5), (44.5, 80),
                    (44, 80.5), (43, 80.5), (42.5, 80),
                    (42, 79.5), (42, 79),
                    # Back of leg going up
                    (42.5, 78), (43, 77), (43, 75),
                    (43, 73), (43, 71), (43, 69),
                    (43, 67), (43.5, 65.5),
                ], "closed": False},
                # Paw toes (near leg)
                {"type": "lines", "points": [
                    (43.5, 80), (44, 80.5),
                ], "closed": False},
                {"type": "lines", "points": [
                    (44.5, 79.5), (44.5, 80.5),
                ], "closed": False},
                # Far front leg (partially visible behind near leg)
                {"type": "lines", "points": [
                    (41, 66), (40.5, 68), (40, 70),
                    (40, 72), (40, 74), (40, 76),
                    (40, 77.5),
                    # Far paw
                    (39.5, 79), (39, 79.5), (38.5, 80),
                    (38, 80.5), (37.5, 80.5), (37, 80),
                    (37, 79.5), (37.5, 79),
                    (38, 78), (38, 76), (38, 74),
                ], "closed": False},
            ]
        },
        # ── Step 7: Hind legs ─────────────────────────────────
        {
            "instruction": "Now the back legs. The dog is sitting, so the hind leg is folded.",
            "elements": [
                # Near hind leg — thigh (sitting position, folded)
                {"type": "lines", "points": [
                    (35, 73.5), (36, 74), (37, 75),
                    (37.5, 76), (37, 77), (36, 78),
                    (35, 79), (34, 79.5),
                    # Hock joint
                    (33, 79), (32, 78), (31.5, 77),
                    # Lower leg extends forward
                    (31, 77.5), (30.5, 78), (30, 79),
                    (29.5, 79.5), (29, 80),
                    # Paw
                    (28.5, 80.5), (28, 80.5), (27.5, 80.5),
                    (27, 80), (26.5, 79.5), (27, 79),
                    (27.5, 78.5), (28, 78), (28.5, 78),
                    (29, 78.5),
                ], "closed": False},
                # Paw toes (hind)
                {"type": "lines", "points": [
                    (27.5, 80), (28, 80.5),
                ], "closed": False},
                # Thigh curve (visible portion)
                {"type": "lines", "points": [
                    (32.5, 70), (33, 72), (34, 74),
                    (35, 75.5), (35.5, 76.5),
                ], "closed": False},
                # Far hind leg (just the paw peeking out)
                {"type": "lines", "points": [
                    (25, 79), (24.5, 79.5), (24, 80),
                    (23.5, 80.5), (23, 80.5), (22.5, 80),
                    (23, 79.5), (23.5, 79), (24, 78.5),
                ], "closed": False},
            ]
        },
        # ── Step 8: Tail and finishing details ────────────────
        {
            "instruction": "Finally, add the tail curving up and some fur details. Your dog is complete!",
            "elements": [
                # Tail (curving up from hip)
                {"type": "lines", "points": [
                    (30, 72), (29, 70), (28, 68),
                    (27, 66), (26.5, 64), (26, 62),
                    (25.5, 60), (25.5, 58), (26, 56.5),
                    # Tail tip curves
                    (26.5, 55.5), (27, 55), (27.5, 55),
                    # Other side of tail
                    (28, 55.5), (28, 56.5), (28, 58),
                    (28.5, 60), (29, 62), (29.5, 64),
                    (30, 66), (30.5, 68), (30, 70),
                ], "closed": False},
                # Whisker dots
                {"type": "dot", "center": (49, 32), "radius": 0.3},
                {"type": "dot", "center": (49, 33), "radius": 0.3},
                {"type": "dot", "center": (48, 32.5), "radius": 0.3},
                {"type": "dot", "center": (48, 33.5), "radius": 0.3},
                # Fur texture on chest
                {"type": "lines", "points": [
                    (44, 50), (45, 52), (44.5, 54),
                ], "closed": False},
                {"type": "lines", "points": [
                    (45.5, 52), (46.5, 54), (46, 56),
                ], "closed": False},
                {"type": "lines", "points": [
                    (46, 56), (47, 58), (46.5, 60),
                ], "closed": False},
                # Fur texture on back
                {"type": "lines", "points": [
                    (31, 56), (30.5, 58), (31, 60),
                ], "closed": False},
                {"type": "lines", "points": [
                    (30, 62), (29.5, 64), (30, 66),
                ], "closed": False},
                # Collar
                {"type": "lines", "points": [
                    (36, 38), (37, 40), (38.5, 41.5),
                    (40, 42.5), (41.5, 43), (43, 43),
                    (44, 42.5),
                ], "closed": False, "width": 4},
                # Collar tag
                {"type": "circle", "center": (40, 44), "radius": 1.2},
            ]
        },
    ]
}


# ═══════════════════════════════════════════════════════════════
# PAPER CRAFT — 3D Stand-up Dog
# Cut along solid lines, fold along dashed lines, glue tabs
# Projects onto cardboard/paper for kids to cut and assemble
# ═══════════════════════════════════════════════════════════════

CRAFT_DOG = {
    "name": "Paper Dog (Cut & Fold)",
    "steps": [
        # ── Step 1: Body (main piece) ─────────────────────────
        {
            "instruction": "This is the main body piece. Cut along the solid white lines carefully.",
            "elements": [
                # Body outline — side profile of dog body
                # Left side of body (from tail → back → head → chest → belly → back to tail)
                {"type": "lines", "points": [
                    # Bottom tab fold line start
                    (15, 62),
                    # Tail top
                    (15, 45), (16, 42), (17, 40), (18, 39),
                    # Back
                    (20, 38), (22, 37.5), (24, 37), (26, 36.5),
                    (28, 36), (30, 36), (32, 36), (34, 36),
                    # Neck rise
                    (36, 35.5), (38, 34.5), (40, 33),
                    # Head top
                    (41, 31.5), (42, 30), (43, 29), (44, 28.5),
                    (46, 28), (48, 28), (50, 28.5),
                    # Ear bump
                    (51, 29), (52, 28.5), (53, 27.5), (54, 27),
                    (55, 27.5), (55.5, 28.5), (55, 29.5),
                    # Forehead to muzzle
                    (55.5, 30.5), (56, 31.5), (57, 32),
                    (58, 32.5), (59, 33), (60, 33),
                    # Nose
                    (61, 33.5), (61, 34.5), (60.5, 35.5),
                    # Mouth and chin
                    (59.5, 36), (58.5, 36), (57.5, 36.5),
                    (56.5, 37), (55.5, 38), (54.5, 39),
                    # Chest
                    (54, 40), (53.5, 42), (53, 44),
                    (53, 46), (53.5, 48),
                    # Front leg
                    (54, 50), (54.5, 52), (55, 54),
                    (55, 56), (55, 58), (55, 60),
                    # Front paw
                    (55, 62), (56, 63), (56.5, 63.5),
                    (57, 63.5), (58, 63), (58, 62),
                    # Between legs
                    (58, 60.5),
                ], "closed": False},
                # Continue: inner front leg → belly → inner back leg → back paw → tab
                {"type": "lines", "points": [
                    (58, 60.5),
                    (57, 59), (56, 58), (55.5, 56),
                    # Belly
                    (54, 53), (52, 52), (50, 51.5),
                    (48, 51.5), (46, 52), (44, 52),
                    (42, 52), (40, 52.5), (38, 53),
                    (36, 54), (34, 55),
                    # Hind leg
                    (32, 56), (30, 57), (28, 58),
                    (26, 59.5), (25, 61),
                    # Hind paw
                    (24, 62), (23, 63), (22, 63.5),
                    (21, 63.5), (20, 63), (20, 62),
                    # Up hind leg
                    (20, 60), (21, 58), (22, 56),
                    (22, 54), (21, 52),
                    # Under tail
                    (19, 51), (17, 52), (16, 54),
                    (15, 57), (15, 60), (15, 62),
                ], "closed": False},
            ]
        },
        # ── Step 2: Face details on body piece ────────────────
        {
            "instruction": "Now add the eye and nose details on the head part.",
            "elements": [
                # Eye
                {"type": "lines", "points": [
                    (52, 31), (53, 30.5), (54, 30.5),
                    (55, 31), (54.5, 31.5), (53.5, 31.5), (52, 31),
                ], "closed": True},
                {"type": "dot", "center": (53.5, 31), "radius": 0.6},
                # Nose
                {"type": "dot", "center": (60.5, 34), "radius": 1.0},
                # Mouth curve
                {"type": "lines", "points": [
                    (60, 35.5), (59, 36), (58, 36.5),
                ], "closed": False},
                # Ear inner line
                {"type": "lines", "points": [
                    (52, 29), (53, 28), (54, 27.5),
                ], "closed": False},
            ]
        },
        # ── Step 3: Bottom stand tab ──────────────────────────
        {
            "instruction": "Now trace the fold tab at the bottom. This part folds under so the dog can stand up!",
            "elements": [
                # Bottom stand tab (dashed = fold line)
                # Fold line along the bottom of the body
                {"type": "dashed", "points": [
                    (15, 62), (20, 62), (24, 62),
                    (30, 62), (36, 62), (42, 62),
                    (48, 62), (55, 62), (58, 62),
                ]},
                # Tab outline (solid — cut this)
                {"type": "lines", "points": [
                    (15, 62), (15, 70), (18, 71),
                    (22, 71.5), (26, 72), (30, 72),
                    (34, 72), (38, 72), (42, 71.5),
                    (46, 71.5), (50, 71), (54, 71),
                    (58, 70), (58, 62),
                ], "closed": False},
                # "FOLD" label hint — small arrows pointing up
                {"type": "lines", "points": [(36, 66), (36.5, 64), (37, 66)], "closed": False},
                {"type": "lines", "points": [(40, 66), (40.5, 64), (41, 66)], "closed": False},
            ]
        },
        # ── Step 4: Second side piece (mirror) ────────────────
        {
            "instruction": "Now trace the other side of the dog. It's the same shape mirrored! Cut this piece too.",
            "elements": [
                # Mirror body — shifted to the right side of projection
                # Simplified mirror outline (same shape, x = 95 - original_x, shifted)
                # This is placed to the right, offset by +20
                {"type": "lines", "points": [
                    # "CUT ANOTHER ONE" indicator — a simple duplicate outline
                    # We show a smaller version as reference
                    (65, 35), (66, 33), (67, 31.5),
                    (68, 30.5), (69.5, 30), (71, 30),
                    (72.5, 30.5), (73.5, 31.5),
                    # Head
                    (74, 32.5), (74.5, 33.5), (75, 34.5),
                    (75.5, 35.5), (75, 36.5),
                    # Down
                    (74, 37.5), (73, 39), (72, 41),
                    (71.5, 43), (72, 45), (73, 47),
                    (74, 49), (74, 51),
                    # Paw
                    (74, 53), (74.5, 54), (75, 54.5),
                    (75, 53),
                    # Belly back
                    (74, 51), (72, 49), (70, 48.5),
                    (68, 49), (66, 50),
                    # Hind
                    (65, 52), (64, 53.5),
                    (63.5, 54), (63, 54.5), (63, 53),
                    (64, 51), (65, 49), (65, 47),
                    (65, 45), (65, 42), (65, 39),
                    (65, 37), (65, 35),
                ], "closed": True},
                # Label
                {"type": "lines", "points": [
                    (67, 42), (68, 42),  # "x2" hint — just a small mark
                ], "closed": False},
                # Fold tab (dashed)
                {"type": "dashed", "points": [
                    (63, 54.5), (66, 54.5), (69, 54.5),
                    (72, 54.5), (75, 54.5),
                ]},
                # Tab outline
                {"type": "lines", "points": [
                    (63, 54.5), (63, 59), (66, 59.5),
                    (69, 60), (72, 59.5), (75, 59),
                    (75, 54.5),
                ], "closed": False},
            ]
        },
    ]
}


# ═══════════════════════════════════════════════════════════════
# PAPER CRAFT — 3D House Box (beginner-friendly)
# A simple house that folds into a 3D box shape
# ═══════════════════════════════════════════════════════════════

CRAFT_HOUSE = {
    "name": "Paper House (Cut & Fold)",
    "steps": [
        # ── Step 1: Front wall with door and window ───────────
        {
            "instruction": "This is the front wall of the house. Cut the outside shape, trace the door and window.",
            "elements": [
                # Front wall rectangle
                {"type": "lines", "points": [
                    (20, 35), (50, 35), (50, 65), (20, 65),
                ], "closed": True},
                # Roof triangle on top
                {"type": "lines", "points": [
                    (18, 35), (35, 18), (52, 35),
                ], "closed": True},
                # Door
                {"type": "lines", "points": [
                    (30, 65), (30, 50), (38, 50), (38, 65),
                ], "closed": False},
                # Door handle
                {"type": "dot", "center": (36, 58), "radius": 0.7},
                # Window
                {"type": "lines", "points": [
                    (24, 40), (32, 40), (32, 48), (24, 48),
                ], "closed": True},
                # Window cross
                {"type": "line", "start": (28, 40), "end": (28, 48)},
                {"type": "line", "start": (24, 44), "end": (32, 44)},
            ]
        },
        # ── Step 2: Side walls with fold lines ────────────────
        {
            "instruction": "Here are the side walls. Cut the outer edges and fold where the dashes are.",
            "elements": [
                # Right side wall
                {"type": "lines", "points": [
                    (50, 35), (75, 35), (75, 65), (50, 65),
                ], "closed": True},
                # Fold line where it meets front wall
                {"type": "dashed", "points": [
                    (50, 35), (50, 65),
                ]},
                # Side window
                {"type": "lines", "points": [
                    (57, 42), (68, 42), (68, 55), (57, 55),
                ], "closed": True},
                {"type": "line", "start": (62.5, 42), "end": (62.5, 55)},
                {"type": "line", "start": (57, 48.5), "end": (68, 48.5)},
                # Left side wall (on the left)
                {"type": "lines", "points": [
                    (20, 35), (20, 65), (-5, 65), (-5, 35),
                ], "closed": True},
                # Fold line
                {"type": "dashed", "points": [
                    (20, 35), (20, 65),
                ]},
                # Side window
                {"type": "lines", "points": [
                    (2, 42), (13, 42), (13, 55), (2, 55),
                ], "closed": True},
                {"type": "line", "start": (7.5, 42), "end": (7.5, 55)},
                {"type": "line", "start": (2, 48.5), "end": (13, 48.5)},
            ]
        },
        # ── Step 3: Back wall and glue tab ────────────────────
        {
            "instruction": "The back wall goes on the right. The small tab is for gluing the house together.",
            "elements": [
                # Back wall
                {"type": "lines", "points": [
                    (75, 35), (100, 35), (100, 65), (75, 65),
                ], "closed": True},
                # Fold line
                {"type": "dashed", "points": [
                    (75, 35), (75, 65),
                ]},
                # Back window
                {"type": "lines", "points": [
                    (82, 42), (93, 42), (93, 55), (82, 55),
                ], "closed": True},
                # Glue tab on far right
                {"type": "lines", "points": [
                    (100, 38), (104, 40), (104, 60), (100, 62),
                ], "closed": False},
                {"type": "dashed", "points": [
                    (100, 35), (100, 65),
                ]},
            ]
        },
        # ── Step 4: Floor tab ─────────────────────────────────
        {
            "instruction": "Last part! These tabs at the bottom fold inward to make the floor. Now cut, fold, and glue!",
            "elements": [
                # Floor tab on front wall (folds inward)
                {"type": "lines", "points": [
                    (20, 65), (50, 65), (50, 78), (20, 78),
                ], "closed": True},
                {"type": "dashed", "points": [
                    (20, 65), (50, 65),
                ]},
                # Floor tab on back wall
                {"type": "lines", "points": [
                    (75, 65), (100, 65), (100, 78), (75, 78),
                ], "closed": True},
                {"type": "dashed", "points": [
                    (75, 65), (100, 65),
                ]},
                # Side floor tabs (smaller)
                {"type": "lines", "points": [
                    (50, 65), (75, 65), (73, 72), (52, 72),
                ], "closed": True},
                {"type": "dashed", "points": [
                    (50, 65), (75, 65),
                ]},
                {"type": "lines", "points": [
                    (-5, 65), (20, 65), (18, 72), (-3, 72),
                ], "closed": True},
                {"type": "dashed", "points": [
                    (-5, 65), (20, 65),
                ]},
            ]
        },
    ]
}


# ═══════════════════════════════════════════════════════════════
# Registry: maps voice-friendly names to drawing data
# ═══════════════════════════════════════════════════════════════

PRO_DRAWINGS = {
    "realistic dog": REALISTIC_DOG,
    "labrador": REALISTIC_DOG,
    "real dog": REALISTIC_DOG,
    "craft dog": CRAFT_DOG,
    "paper dog": CRAFT_DOG,
    "dog craft": CRAFT_DOG,
    "craft house": CRAFT_HOUSE,
    "paper house": CRAFT_HOUSE,
    "house craft": CRAFT_HOUSE,
}

PRO_NAMES = list(PRO_DRAWINGS.keys())
