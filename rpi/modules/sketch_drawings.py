"""Step-by-step line drawings for projector sketch/tracing mode.

Each drawing has steps. Each step has an instruction (spoken aloud)
and a list of drawing primitives. All coordinates are on a 0-100 scale
and get mapped to screen pixels at render time.

Primitive types:
  line:    start=(x,y), end=(x,y)
  circle:  center=(x,y), radius=float
  ellipse: rect=(x,y,w,h)
  arc:     rect=(x,y,w,h), start_angle=float, end_angle=float (radians)
  lines:   points=[(x,y),...], closed=bool  (polyline/polygon)
  dot:     center=(x,y), radius=float (filled circle)
"""

import math

PI = math.pi

SKETCH_DRAWINGS = {
    "cat": {
        "name": "Cat",
        "steps": [
            {
                "instruction": "Draw a big circle for the head",
                "elements": [
                    {"type": "circle", "center": (50, 35), "radius": 16},
                ]
            },
            {
                "instruction": "Add two pointy ears on top",
                "elements": [
                    {"type": "lines", "points": [(37, 23), (34, 10), (44, 21)], "closed": True},
                    {"type": "lines", "points": [(63, 23), (66, 10), (56, 21)], "closed": True},
                ]
            },
            {
                "instruction": "Draw two round eyes",
                "elements": [
                    {"type": "circle", "center": (43, 33), "radius": 3},
                    {"type": "circle", "center": (57, 33), "radius": 3},
                    {"type": "dot", "center": (43, 33), "radius": 1.5},
                    {"type": "dot", "center": (57, 33), "radius": 1.5},
                ]
            },
            {
                "instruction": "Add a small nose and a smiley mouth",
                "elements": [
                    {"type": "dot", "center": (50, 39), "radius": 1.5},
                    {"type": "arc", "rect": (44, 39, 6, 6), "start_angle": PI, "end_angle": 2 * PI},
                    {"type": "arc", "rect": (50, 39, 6, 6), "start_angle": PI, "end_angle": 2 * PI},
                ]
            },
            {
                "instruction": "Draw long whiskers on each side",
                "elements": [
                    {"type": "line", "start": (39, 38), "end": (22, 35)},
                    {"type": "line", "start": (39, 40), "end": (22, 42)},
                    {"type": "line", "start": (61, 38), "end": (78, 35)},
                    {"type": "line", "start": (61, 40), "end": (78, 42)},
                ]
            },
            {
                "instruction": "Draw an oval body below the head",
                "elements": [
                    {"type": "ellipse", "rect": (36, 50, 28, 30)},
                ]
            },
            {
                "instruction": "Add a curvy tail on the side",
                "elements": [
                    {"type": "arc", "rect": (60, 60, 20, 25), "start_angle": PI * 0.5, "end_angle": PI * 2},
                ]
            },
        ]
    },

    "dog": {
        "name": "Dog",
        "steps": [
            {
                "instruction": "Draw a big circle for the head",
                "elements": [
                    {"type": "circle", "center": (50, 32), "radius": 16},
                ]
            },
            {
                "instruction": "Add two floppy ears hanging down",
                "elements": [
                    {"type": "ellipse", "rect": (28, 25, 12, 22)},
                    {"type": "ellipse", "rect": (60, 25, 12, 22)},
                ]
            },
            {
                "instruction": "Draw two eyes and a big nose",
                "elements": [
                    {"type": "circle", "center": (44, 30), "radius": 3},
                    {"type": "circle", "center": (56, 30), "radius": 3},
                    {"type": "dot", "center": (44, 30), "radius": 1.5},
                    {"type": "dot", "center": (56, 30), "radius": 1.5},
                    {"type": "ellipse", "rect": (46, 35, 8, 6)},
                ]
            },
            {
                "instruction": "Add a happy mouth with tongue",
                "elements": [
                    {"type": "arc", "rect": (42, 38, 16, 10), "start_angle": PI, "end_angle": 2 * PI},
                    {"type": "ellipse", "rect": (48, 42, 5, 6)},
                ]
            },
            {
                "instruction": "Draw a round body below",
                "elements": [
                    {"type": "ellipse", "rect": (34, 48, 32, 30)},
                ]
            },
            {
                "instruction": "Add four short legs at the bottom",
                "elements": [
                    {"type": "line", "start": (40, 76), "end": (40, 90)},
                    {"type": "line", "start": (47, 76), "end": (47, 90)},
                    {"type": "line", "start": (53, 76), "end": (53, 90)},
                    {"type": "line", "start": (60, 76), "end": (60, 90)},
                ]
            },
            {
                "instruction": "Add a wagging tail!",
                "elements": [
                    {"type": "arc", "rect": (62, 50, 18, 15), "start_angle": PI * 1.5, "end_angle": PI * 2.3},
                ]
            },
        ]
    },

    "house": {
        "name": "House",
        "steps": [
            {
                "instruction": "Draw a big square for the walls",
                "elements": [
                    {"type": "lines", "points": [(25, 45), (75, 45), (75, 85), (25, 85)], "closed": True},
                ]
            },
            {
                "instruction": "Add a triangle roof on top",
                "elements": [
                    {"type": "lines", "points": [(20, 45), (50, 15), (80, 45)], "closed": True},
                ]
            },
            {
                "instruction": "Draw a door in the middle",
                "elements": [
                    {"type": "lines", "points": [(43, 85), (43, 62), (57, 62), (57, 85)], "closed": False},
                    {"type": "dot", "center": (54, 74), "radius": 1.2},
                ]
            },
            {
                "instruction": "Add two windows on each side",
                "elements": [
                    {"type": "lines", "points": [(29, 55), (39, 55), (39, 68), (29, 68)], "closed": True},
                    {"type": "line", "start": (34, 55), "end": (34, 68)},
                    {"type": "line", "start": (29, 61), "end": (39, 61)},
                    {"type": "lines", "points": [(61, 55), (71, 55), (71, 68), (61, 68)], "closed": True},
                    {"type": "line", "start": (66, 55), "end": (66, 68)},
                    {"type": "line", "start": (61, 61), "end": (71, 61)},
                ]
            },
            {
                "instruction": "Add a chimney on the roof",
                "elements": [
                    {"type": "lines", "points": [(62, 30), (62, 15), (70, 15), (70, 25)], "closed": False},
                ]
            },
        ]
    },

    "tree": {
        "name": "Tree",
        "steps": [
            {
                "instruction": "Draw a thick trunk going up",
                "elements": [
                    {"type": "lines", "points": [(44, 90), (44, 50), (56, 50), (56, 90)], "closed": True},
                ]
            },
            {
                "instruction": "Add a big round cloud of leaves",
                "elements": [
                    {"type": "circle", "center": (50, 35), "radius": 20},
                ]
            },
            {
                "instruction": "Add more leaf bumps on the sides",
                "elements": [
                    {"type": "circle", "center": (35, 40), "radius": 13},
                    {"type": "circle", "center": (65, 40), "radius": 13},
                    {"type": "circle", "center": (42, 22), "radius": 12},
                    {"type": "circle", "center": (58, 22), "radius": 12},
                ]
            },
            {
                "instruction": "Draw some apples in the tree!",
                "elements": [
                    {"type": "dot", "center": (42, 30), "radius": 2.5},
                    {"type": "dot", "center": (55, 38), "radius": 2.5},
                    {"type": "dot", "center": (48, 44), "radius": 2.5},
                    {"type": "dot", "center": (60, 28), "radius": 2.5},
                ]
            },
        ]
    },

    "fish": {
        "name": "Fish",
        "steps": [
            {
                "instruction": "Draw an oval body for the fish",
                "elements": [
                    {"type": "ellipse", "rect": (25, 30, 40, 30)},
                ]
            },
            {
                "instruction": "Add a triangle tail at the back",
                "elements": [
                    {"type": "lines", "points": [(65, 45), (82, 32), (82, 58)], "closed": True},
                ]
            },
            {
                "instruction": "Draw a round eye",
                "elements": [
                    {"type": "circle", "center": (36, 42), "radius": 4},
                    {"type": "dot", "center": (37, 41), "radius": 2},
                ]
            },
            {
                "instruction": "Add a smiley mouth",
                "elements": [
                    {"type": "arc", "rect": (30, 48, 10, 6), "start_angle": PI, "end_angle": 2 * PI},
                ]
            },
            {
                "instruction": "Draw fins on top and bottom",
                "elements": [
                    {"type": "lines", "points": [(42, 30), (50, 18), (55, 30)], "closed": True},
                    {"type": "lines", "points": [(42, 60), (48, 70), (55, 60)], "closed": True},
                ]
            },
            {
                "instruction": "Add some scales and bubbles!",
                "elements": [
                    {"type": "arc", "rect": (40, 38, 8, 8), "start_angle": 0, "end_angle": PI},
                    {"type": "arc", "rect": (48, 38, 8, 8), "start_angle": 0, "end_angle": PI},
                    {"type": "arc", "rect": (44, 46, 8, 8), "start_angle": 0, "end_angle": PI},
                    {"type": "circle", "center": (22, 35), "radius": 2},
                    {"type": "circle", "center": (18, 28), "radius": 1.5},
                    {"type": "circle", "center": (20, 22), "radius": 1},
                ]
            },
        ]
    },

    "star": {
        "name": "Star",
        "steps": [
            {
                "instruction": "Draw a line going up to the top point",
                "elements": [
                    {"type": "line", "start": (40, 55), "end": (50, 15)},
                ]
            },
            {
                "instruction": "Draw a line going down to the right",
                "elements": [
                    {"type": "line", "start": (50, 15), "end": (60, 55)},
                ]
            },
            {
                "instruction": "Draw a line going to the far left",
                "elements": [
                    {"type": "line", "start": (60, 55), "end": (22, 35)},
                ]
            },
            {
                "instruction": "Draw a line straight across to the far right",
                "elements": [
                    {"type": "line", "start": (22, 35), "end": (78, 35)},
                ]
            },
            {
                "instruction": "Close it back to where we started!",
                "elements": [
                    {"type": "line", "start": (78, 35), "end": (40, 55)},
                ]
            },
            {
                "instruction": "Add a smiley face in the middle!",
                "elements": [
                    {"type": "dot", "center": (46, 36), "radius": 1.5},
                    {"type": "dot", "center": (54, 36), "radius": 1.5},
                    {"type": "arc", "rect": (44, 38, 12, 7), "start_angle": PI, "end_angle": 2 * PI},
                ]
            },
        ]
    },

    "flower": {
        "name": "Flower",
        "steps": [
            {
                "instruction": "Draw a circle in the center for the flower middle",
                "elements": [
                    {"type": "circle", "center": (50, 35), "radius": 8},
                ]
            },
            {
                "instruction": "Add petals all around — top and bottom",
                "elements": [
                    {"type": "ellipse", "rect": (44, 12, 12, 16)},
                    {"type": "ellipse", "rect": (44, 42, 12, 16)},
                ]
            },
            {
                "instruction": "Add petals on the left and right",
                "elements": [
                    {"type": "ellipse", "rect": (28, 28, 16, 12)},
                    {"type": "ellipse", "rect": (56, 28, 16, 12)},
                ]
            },
            {
                "instruction": "Add petals in the corners",
                "elements": [
                    {"type": "ellipse", "rect": (31, 15, 14, 12)},
                    {"type": "ellipse", "rect": (55, 15, 14, 12)},
                    {"type": "ellipse", "rect": (31, 42, 14, 12)},
                    {"type": "ellipse", "rect": (55, 42, 14, 12)},
                ]
            },
            {
                "instruction": "Draw a long stem going down",
                "elements": [
                    {"type": "line", "start": (50, 55), "end": (50, 92)},
                ]
            },
            {
                "instruction": "Add two leaves on the stem",
                "elements": [
                    {"type": "ellipse", "rect": (33, 68, 17, 8)},
                    {"type": "ellipse", "rect": (50, 76, 17, 8)},
                ]
            },
        ]
    },

    "sun": {
        "name": "Sun",
        "steps": [
            {
                "instruction": "Draw a big circle in the middle",
                "elements": [
                    {"type": "circle", "center": (50, 45), "radius": 16},
                ]
            },
            {
                "instruction": "Add a happy face — eyes and smile!",
                "elements": [
                    {"type": "dot", "center": (44, 42), "radius": 2},
                    {"type": "dot", "center": (56, 42), "radius": 2},
                    {"type": "arc", "rect": (42, 46, 16, 8), "start_angle": PI, "end_angle": 2 * PI},
                ]
            },
            {
                "instruction": "Draw rays going up, down, left, and right",
                "elements": [
                    {"type": "line", "start": (50, 27), "end": (50, 12)},
                    {"type": "line", "start": (50, 63), "end": (50, 78)},
                    {"type": "line", "start": (32, 45), "end": (17, 45)},
                    {"type": "line", "start": (68, 45), "end": (83, 45)},
                ]
            },
            {
                "instruction": "Add rays in between — the diagonal ones!",
                "elements": [
                    {"type": "line", "start": (38, 33), "end": (27, 22)},
                    {"type": "line", "start": (62, 33), "end": (73, 22)},
                    {"type": "line", "start": (38, 57), "end": (27, 68)},
                    {"type": "line", "start": (62, 57), "end": (73, 68)},
                ]
            },
        ]
    },

    "car": {
        "name": "Car",
        "steps": [
            {
                "instruction": "Draw a long rectangle for the car body",
                "elements": [
                    {"type": "lines", "points": [(15, 55), (85, 55), (85, 72), (15, 72)], "closed": True},
                ]
            },
            {
                "instruction": "Add the cabin on top with windows",
                "elements": [
                    {"type": "lines", "points": [(28, 55), (35, 35), (65, 35), (72, 55)], "closed": True},
                ]
            },
            {
                "instruction": "Draw a line in the middle for two windows",
                "elements": [
                    {"type": "line", "start": (50, 36), "end": (50, 55)},
                ]
            },
            {
                "instruction": "Add two round wheels at the bottom",
                "elements": [
                    {"type": "circle", "center": (30, 74), "radius": 8},
                    {"type": "circle", "center": (70, 74), "radius": 8},
                    {"type": "dot", "center": (30, 74), "radius": 3},
                    {"type": "dot", "center": (70, 74), "radius": 3},
                ]
            },
            {
                "instruction": "Add headlights and a door handle",
                "elements": [
                    {"type": "circle", "center": (17, 62), "radius": 3},
                    {"type": "circle", "center": (83, 62), "radius": 3},
                    {"type": "line", "start": (52, 62), "end": (58, 62)},
                ]
            },
        ]
    },

    "butterfly": {
        "name": "Butterfly",
        "steps": [
            {
                "instruction": "Draw a thin oval body in the center",
                "elements": [
                    {"type": "ellipse", "rect": (47, 25, 6, 40)},
                ]
            },
            {
                "instruction": "Add a round head on top",
                "elements": [
                    {"type": "circle", "center": (50, 22), "radius": 5},
                    {"type": "dot", "center": (48, 21), "radius": 1},
                    {"type": "dot", "center": (52, 21), "radius": 1},
                ]
            },
            {
                "instruction": "Draw two curly antennae",
                "elements": [
                    {"type": "arc", "rect": (38, 8, 12, 14), "start_angle": 0, "end_angle": PI},
                    {"type": "arc", "rect": (50, 8, 12, 14), "start_angle": 0, "end_angle": PI},
                ]
            },
            {
                "instruction": "Draw the top left wing — a big curve",
                "elements": [
                    {"type": "ellipse", "rect": (18, 22, 30, 22)},
                ]
            },
            {
                "instruction": "Draw the top right wing to match",
                "elements": [
                    {"type": "ellipse", "rect": (52, 22, 30, 22)},
                ]
            },
            {
                "instruction": "Add bottom wings — slightly smaller",
                "elements": [
                    {"type": "ellipse", "rect": (22, 42, 26, 18)},
                    {"type": "ellipse", "rect": (52, 42, 26, 18)},
                ]
            },
            {
                "instruction": "Decorate the wings with circles!",
                "elements": [
                    {"type": "circle", "center": (33, 33), "radius": 5},
                    {"type": "circle", "center": (67, 33), "radius": 5},
                    {"type": "circle", "center": (35, 50), "radius": 3.5},
                    {"type": "circle", "center": (65, 50), "radius": 3.5},
                ]
            },
        ]
    },
}

# List of available drawings for voice matching
SKETCH_NAMES = list(SKETCH_DRAWINGS.keys())
