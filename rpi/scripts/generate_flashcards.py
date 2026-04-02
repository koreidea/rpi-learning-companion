#!/usr/bin/env python3
"""Generate colorful flashcard images for the projector module.

Creates 500x500 PNG images using PIL. Each image is a simple, bold,
child-friendly illustration suitable for wall projection.

Output: rpi/assets/projector/flashcards/{category}/{name}.png

Usage:
    python rpi/scripts/generate_flashcards.py
"""

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

# Output directory
OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "projector" / "flashcards"
SIZE = 500  # 500x500 images


def new_image(bg=(30, 30, 45)):
    img = Image.new("RGB", (SIZE, SIZE), bg)
    return img, ImageDraw.Draw(img)


def save(img, category, name):
    path = OUT_DIR / category / f"{name.lower()}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), "PNG")
    print(f"  {path.name}")


# ═══════════════════════════════════════════════════════════════
# SHAPES
# ═══════════════════════════════════════════════════════════════

def draw_circle():
    img, d = new_image()
    # Outer glow
    d.ellipse([80, 80, 420, 420], fill=(255, 80, 80))
    # Inner highlight
    d.ellipse([100, 100, 400, 400], fill=(255, 100, 100))
    d.ellipse([140, 120, 280, 220], fill=(255, 150, 150))  # shine
    return img


def draw_square():
    img, d = new_image()
    d.rounded_rectangle([90, 90, 410, 410], radius=15, fill=(30, 144, 255))
    d.rounded_rectangle([110, 110, 390, 390], radius=12, fill=(50, 160, 255))
    d.polygon([(120, 120), (250, 120), (120, 220)], fill=(80, 180, 255))  # shine
    return img


def draw_triangle():
    img, d = new_image()
    pts = [(250, 60), (430, 420), (70, 420)]
    d.polygon(pts, fill=(76, 175, 80))
    inner = [(250, 100), (400, 400), (100, 400)]
    d.polygon(inner, fill=(90, 190, 95))
    d.polygon([(250, 120), (310, 240), (170, 240)], fill=(120, 210, 125))  # shine
    return img


def draw_star():
    img, d = new_image()
    cx, cy, r_out, r_in = 250, 250, 200, 80
    pts = []
    for i in range(10):
        angle = math.radians(i * 36 - 90)
        r = r_out if i % 2 == 0 else r_in
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    d.polygon(pts, fill=(255, 215, 0))
    # Smaller star shine
    pts2 = []
    for i in range(10):
        angle = math.radians(i * 36 - 90)
        r = (r_out * 0.7) if i % 2 == 0 else (r_in * 0.7)
        pts2.append((cx + r * math.cos(angle), cy - 20 + r * math.sin(angle)))
    d.polygon(pts2, fill=(255, 230, 80))
    return img


def draw_heart():
    img, d = new_image()
    # Two circles for top of heart
    d.ellipse([95, 100, 275, 280], fill=(255, 50, 80))
    d.ellipse([225, 100, 405, 280], fill=(255, 50, 80))
    # Triangle for bottom
    d.polygon([(95, 210), (405, 210), (250, 430)], fill=(255, 50, 80))
    # Shine
    d.ellipse([130, 130, 230, 230], fill=(255, 100, 120))
    return img


def draw_rectangle():
    img, d = new_image()
    d.rounded_rectangle([60, 130, 440, 370], radius=15, fill=(156, 39, 176))
    d.rounded_rectangle([80, 150, 420, 350], radius=12, fill=(170, 60, 190))
    d.polygon([(90, 160), (220, 160), (90, 230)], fill=(190, 100, 210))
    return img


def draw_diamond():
    img, d = new_image()
    pts = [(250, 50), (440, 250), (250, 450), (60, 250)]
    d.polygon(pts, fill=(0, 188, 212))
    inner = [(250, 90), (410, 250), (250, 410), (90, 250)]
    d.polygon(inner, fill=(30, 200, 220))
    d.polygon([(250, 110), (310, 200), (180, 200)], fill=(80, 220, 235))
    return img


def draw_oval():
    img, d = new_image()
    d.ellipse([70, 120, 430, 380], fill=(255, 152, 0))
    d.ellipse([90, 140, 410, 360], fill=(255, 170, 40))
    d.ellipse([120, 150, 270, 250], fill=(255, 200, 100))
    return img


SHAPE_DRAWERS = {
    "circle": draw_circle,
    "square": draw_square,
    "triangle": draw_triangle,
    "star": draw_star,
    "heart": draw_heart,
    "rectangle": draw_rectangle,
    "diamond": draw_diamond,
    "oval": draw_oval,
}


# ═══════════════════════════════════════════════════════════════
# COLORS
# ═══════════════════════════════════════════════════════════════

COLOR_VALUES = {
    "red": (220, 40, 40),
    "blue": (30, 120, 255),
    "green": (40, 180, 60),
    "yellow": (255, 220, 30),
    "orange": (255, 140, 0),
    "purple": (140, 40, 200),
    "pink": (255, 105, 180),
    "white": (240, 240, 240),
    "black": (30, 30, 30),
    "brown": (139, 90, 43),
}


def draw_color(name, rgb):
    bg = (50, 50, 60) if name != "black" else (100, 100, 110)
    img, d = new_image(bg)
    # Large circle swatch
    d.ellipse([60, 60, 440, 440], fill=rgb)
    # Highlight/shine
    if name not in ("black", "white"):
        shine = tuple(min(255, c + 60) for c in rgb)
        d.ellipse([120, 90, 250, 200], fill=shine)
    elif name == "white":
        d.ellipse([120, 90, 250, 200], fill=(255, 255, 255))
    else:
        d.ellipse([120, 90, 250, 200], fill=(60, 60, 60))
    return img


# ═══════════════════════════════════════════════════════════════
# ANIMALS — Simple geometric cartoon faces
# ═══════════════════════════════════════════════════════════════

def draw_dog():
    img, d = new_image()
    # Ears (floppy)
    d.ellipse([60, 80, 180, 280], fill=(160, 100, 50))
    d.ellipse([320, 80, 440, 280], fill=(160, 100, 50))
    # Head
    d.ellipse([100, 100, 400, 400], fill=(200, 140, 70))
    # Muzzle
    d.ellipse([160, 250, 340, 400], fill=(230, 190, 130))
    # Eyes
    d.ellipse([170, 190, 220, 250], fill=(255, 255, 255))
    d.ellipse([280, 190, 330, 250], fill=(255, 255, 255))
    d.ellipse([185, 205, 210, 235], fill=(40, 40, 40))
    d.ellipse([293, 205, 318, 235], fill=(40, 40, 40))
    # Eye shine
    d.ellipse([192, 208, 202, 218], fill=(255, 255, 255))
    d.ellipse([300, 208, 310, 218], fill=(255, 255, 255))
    # Nose
    d.ellipse([225, 290, 275, 330], fill=(50, 30, 20))
    # Mouth
    d.arc([210, 310, 250, 360], 0, 180, fill=(50, 30, 20), width=3)
    d.arc([250, 310, 290, 360], 0, 180, fill=(50, 30, 20), width=3)
    # Tongue
    d.ellipse([235, 345, 270, 395], fill=(255, 120, 120))
    return img


def draw_cat():
    img, d = new_image()
    # Ears (pointed)
    d.polygon([(100, 180), (150, 40), (210, 170)], fill=(180, 180, 180))
    d.polygon([(290, 170), (350, 40), (400, 180)], fill=(180, 180, 180))
    # Inner ears
    d.polygon([(125, 175), (155, 70), (195, 170)], fill=(255, 160, 160))
    d.polygon([(305, 170), (345, 70), (380, 175)], fill=(255, 160, 160))
    # Head
    d.ellipse([100, 120, 400, 420], fill=(200, 200, 200))
    # Eyes
    d.ellipse([160, 210, 230, 280], fill=(100, 200, 100))
    d.ellipse([270, 210, 340, 280], fill=(100, 200, 100))
    # Pupils (vertical slits)
    d.ellipse([190, 220, 205, 270], fill=(20, 20, 20))
    d.ellipse([298, 220, 313, 270], fill=(20, 20, 20))
    # Eye shine
    d.ellipse([192, 225, 200, 233], fill=(255, 255, 255))
    d.ellipse([300, 225, 308, 233], fill=(255, 255, 255))
    # Nose
    d.polygon([(240, 300), (260, 300), (250, 320)], fill=(255, 140, 140))
    # Whiskers
    for y_off in [-5, 5, 15]:
        d.line([(100, 310 + y_off), (185, 320 + y_off)], fill=(150, 150, 150), width=2)
        d.line([(315, 320 + y_off), (400, 310 + y_off)], fill=(150, 150, 150), width=2)
    # Mouth
    d.arc([220, 315, 250, 350], 0, 180, fill=(100, 100, 100), width=2)
    d.arc([250, 315, 280, 350], 0, 180, fill=(100, 100, 100), width=2)
    return img


def draw_elephant():
    img, d = new_image()
    # Big ears
    d.ellipse([30, 100, 200, 380], fill=(130, 140, 160))
    d.ellipse([300, 100, 470, 380], fill=(130, 140, 160))
    # Inner ears
    d.ellipse([60, 140, 180, 340], fill=(170, 160, 180))
    d.ellipse([320, 140, 440, 340], fill=(170, 160, 180))
    # Head
    d.ellipse([110, 80, 390, 380], fill=(150, 160, 180))
    # Eyes
    d.ellipse([180, 180, 220, 230], fill=(255, 255, 255))
    d.ellipse([280, 180, 320, 230], fill=(255, 255, 255))
    d.ellipse([192, 192, 212, 218], fill=(40, 40, 40))
    d.ellipse([292, 192, 312, 218], fill=(40, 40, 40))
    d.ellipse([196, 196, 204, 204], fill=(255, 255, 255))
    d.ellipse([296, 196, 304, 204], fill=(255, 255, 255))
    # Trunk
    d.rounded_rectangle([220, 250, 280, 440], radius=30, fill=(140, 150, 170))
    d.rounded_rectangle([230, 420, 310, 460], radius=15, fill=(140, 150, 170))
    # Trunk lines
    for y in range(280, 420, 25):
        d.arc([225, y, 275, y + 20], 0, 180, fill=(120, 130, 150), width=2)
    return img


def draw_lion():
    img, d = new_image()
    # Mane (spiky rays)
    cx, cy = 250, 250
    for angle_deg in range(0, 360, 15):
        a = math.radians(angle_deg)
        x1 = cx + 130 * math.cos(a)
        y1 = cy + 130 * math.sin(a)
        x2 = cx + 210 * math.cos(a - 0.15)
        y2 = cy + 210 * math.sin(a - 0.15)
        x3 = cx + 210 * math.cos(a + 0.15)
        y3 = cy + 210 * math.sin(a + 0.15)
        d.polygon([(x1, y1), (x2, y2), (x3, y3)], fill=(200, 130, 20))
    # Face
    d.ellipse([120, 120, 380, 380], fill=(240, 190, 60))
    # Eyes
    d.ellipse([175, 210, 225, 260], fill=(255, 255, 255))
    d.ellipse([275, 210, 325, 260], fill=(255, 255, 255))
    d.ellipse([190, 220, 215, 250], fill=(50, 40, 20))
    d.ellipse([288, 220, 313, 250], fill=(50, 40, 20))
    d.ellipse([195, 225, 205, 235], fill=(255, 255, 255))
    d.ellipse([293, 225, 303, 235], fill=(255, 255, 255))
    # Nose
    d.polygon([(235, 280), (265, 280), (250, 305)], fill=(80, 50, 20))
    # Mouth
    d.arc([215, 300, 250, 340], 0, 180, fill=(80, 50, 20), width=3)
    d.arc([250, 300, 285, 340], 0, 180, fill=(80, 50, 20), width=3)
    return img


def draw_fish():
    img, d = new_image()
    # Tail
    d.polygon([(370, 250), (460, 170), (460, 330)], fill=(255, 140, 0))
    # Body
    d.ellipse([60, 140, 400, 360], fill=(255, 165, 50))
    # Stripe
    d.ellipse([80, 160, 380, 340], fill=(255, 200, 80))
    # Fin (top)
    d.polygon([(200, 150), (250, 60), (280, 150)], fill=(255, 120, 0))
    # Eye
    d.ellipse([130, 210, 190, 270], fill=(255, 255, 255))
    d.ellipse([145, 222, 175, 258], fill=(20, 20, 20))
    d.ellipse([150, 228, 162, 240], fill=(255, 255, 255))
    # Mouth
    d.arc([90, 250, 140, 290], 320, 40, fill=(200, 100, 0), width=3)
    # Scales
    for sx, sy in [(220, 200), (270, 230), (310, 260), (250, 280), (300, 200)]:
        d.arc([sx, sy, sx + 40, sy + 30], 0, 180, fill=(255, 140, 30), width=2)
    return img


def draw_bird():
    img, d = new_image()
    # Wing
    d.ellipse([180, 170, 430, 340], fill=(0, 150, 200))
    # Body
    d.ellipse([100, 150, 340, 370], fill=(50, 180, 230))
    # Head
    d.ellipse([70, 100, 220, 250], fill=(50, 180, 230))
    # Eye
    d.ellipse([120, 150, 170, 200], fill=(255, 255, 255))
    d.ellipse([133, 160, 158, 190], fill=(20, 20, 20))
    d.ellipse([137, 164, 147, 174], fill=(255, 255, 255))
    # Beak
    d.polygon([(70, 185), (30, 200), (70, 215)], fill=(255, 180, 0))
    # Tail feathers
    d.polygon([(320, 230), (440, 180), (430, 250)], fill=(0, 130, 180))
    d.polygon([(320, 260), (450, 220), (440, 290)], fill=(0, 120, 170))
    d.polygon([(310, 290), (440, 270), (430, 330)], fill=(0, 110, 160))
    # Legs
    d.line([(200, 360), (180, 430)], fill=(255, 160, 0), width=4)
    d.line([(240, 360), (260, 430)], fill=(255, 160, 0), width=4)
    d.line([(160, 430), (200, 430)], fill=(255, 160, 0), width=3)
    d.line([(240, 430), (280, 430)], fill=(255, 160, 0), width=3)
    return img


def draw_rabbit():
    img, d = new_image()
    # Long ears
    d.rounded_rectangle([140, 10, 200, 210], radius=30, fill=(220, 200, 200))
    d.rounded_rectangle([300, 10, 360, 210], radius=30, fill=(220, 200, 200))
    # Inner ears
    d.rounded_rectangle([155, 30, 185, 190], radius=20, fill=(255, 170, 170))
    d.rounded_rectangle([315, 30, 345, 190], radius=20, fill=(255, 170, 170))
    # Head
    d.ellipse([110, 150, 390, 420], fill=(240, 230, 225))
    # Cheeks
    d.ellipse([120, 280, 210, 360], fill=(255, 210, 210))
    d.ellipse([290, 280, 380, 360], fill=(255, 210, 210))
    # Eyes
    d.ellipse([180, 230, 225, 290], fill=(60, 40, 40))
    d.ellipse([275, 230, 320, 290], fill=(60, 40, 40))
    d.ellipse([195, 242, 210, 260], fill=(255, 255, 255))
    d.ellipse([288, 242, 303, 260], fill=(255, 255, 255))
    # Nose
    d.ellipse([237, 310, 263, 335], fill=(255, 140, 150))
    # Mouth
    d.line([(250, 335), (250, 360)], fill=(180, 140, 140), width=2)
    d.arc([225, 345, 250, 375], 0, 180, fill=(180, 140, 140), width=2)
    d.arc([250, 345, 275, 375], 0, 180, fill=(180, 140, 140), width=2)
    # Teeth
    d.rectangle([240, 360, 250, 385], fill=(255, 255, 255))
    d.rectangle([252, 360, 262, 385], fill=(255, 255, 255))
    return img


def draw_monkey():
    img, d = new_image()
    # Ears
    d.ellipse([60, 180, 160, 310], fill=(180, 120, 60))
    d.ellipse([340, 180, 440, 310], fill=(180, 120, 60))
    d.ellipse([80, 200, 150, 290], fill=(230, 190, 140))
    d.ellipse([350, 200, 420, 290], fill=(230, 190, 140))
    # Head
    d.ellipse([110, 80, 390, 400], fill=(160, 100, 40))
    # Face area
    d.ellipse([140, 150, 360, 380], fill=(230, 190, 140))
    # Eyes
    d.ellipse([180, 200, 230, 260], fill=(255, 255, 255))
    d.ellipse([270, 200, 320, 260], fill=(255, 255, 255))
    d.ellipse([195, 215, 220, 248], fill=(40, 30, 20))
    d.ellipse([282, 215, 307, 248], fill=(40, 30, 20))
    d.ellipse([200, 220, 210, 230], fill=(255, 255, 255))
    d.ellipse([287, 220, 297, 230], fill=(255, 255, 255))
    # Nose
    d.ellipse([220, 275, 250, 305], fill=(120, 70, 30))
    d.ellipse([250, 275, 280, 305], fill=(120, 70, 30))
    # Mouth
    d.arc([210, 300, 290, 360], 0, 180, fill=(120, 70, 30), width=3)
    return img


def draw_cow():
    img, d = new_image()
    # Horns
    d.polygon([(140, 120), (120, 40), (170, 100)], fill=(200, 180, 120))
    d.polygon([(360, 120), (380, 40), (330, 100)], fill=(200, 180, 120))
    # Head
    d.ellipse([110, 100, 390, 380], fill=(255, 255, 255))
    # Spots
    d.ellipse([150, 120, 220, 190], fill=(60, 60, 60))
    d.ellipse([310, 140, 370, 200], fill=(60, 60, 60))
    d.ellipse([270, 100, 320, 150], fill=(60, 60, 60))
    # Eyes
    d.ellipse([180, 210, 225, 260], fill=(40, 40, 40))
    d.ellipse([275, 210, 320, 260], fill=(40, 40, 40))
    d.ellipse([195, 220, 210, 238], fill=(255, 255, 255))
    d.ellipse([288, 220, 303, 238], fill=(255, 255, 255))
    # Muzzle
    d.ellipse([165, 280, 335, 400], fill=(230, 200, 170))
    # Nostrils
    d.ellipse([210, 310, 240, 345], fill=(180, 140, 120))
    d.ellipse([260, 310, 290, 345], fill=(180, 140, 120))
    # Ears
    d.ellipse([90, 140, 145, 220], fill=(255, 230, 230))
    d.ellipse([355, 140, 410, 220], fill=(255, 230, 230))
    return img


def draw_horse():
    img, d = new_image()
    # Mane
    for i in range(8):
        y = 60 + i * 30
        d.ellipse([95, y, 160, y + 50], fill=(100, 60, 20))
    # Head (elongated)
    d.ellipse([110, 80, 400, 350], fill=(180, 130, 70))
    # Muzzle
    d.ellipse([150, 280, 370, 440], fill=(200, 155, 90))
    # Eyes
    d.ellipse([190, 170, 240, 230], fill=(40, 30, 20))
    d.ellipse([290, 170, 340, 230], fill=(40, 30, 20))
    d.ellipse([205, 180, 220, 200], fill=(255, 255, 255))
    d.ellipse([303, 180, 318, 200], fill=(255, 255, 255))
    # Nostrils
    d.ellipse([200, 340, 240, 380], fill=(120, 80, 40))
    d.ellipse([280, 340, 320, 380], fill=(120, 80, 40))
    # Ears
    d.polygon([(160, 100), (140, 20), (200, 80)], fill=(180, 130, 70))
    d.polygon([(340, 100), (360, 20), (300, 80)], fill=(180, 130, 70))
    d.polygon([(165, 95), (148, 35), (195, 82)], fill=(210, 160, 100))
    d.polygon([(335, 95), (352, 35), (305, 82)], fill=(210, 160, 100))
    return img


ANIMAL_DRAWERS = {
    "dog": draw_dog,
    "cat": draw_cat,
    "elephant": draw_elephant,
    "lion": draw_lion,
    "fish": draw_fish,
    "bird": draw_bird,
    "rabbit": draw_rabbit,
    "monkey": draw_monkey,
    "cow": draw_cow,
    "horse": draw_horse,
}


# ═══════════════════════════════════════════════════════════════
# FRUITS — Colorful produce illustrations
# ═══════════════════════════════════════════════════════════════

def draw_apple():
    img, d = new_image()
    # Body
    d.ellipse([100, 130, 400, 430], fill=(220, 30, 30))
    # Top indent
    d.ellipse([190, 115, 310, 200], fill=(30, 30, 45))
    d.ellipse([195, 130, 305, 210], fill=(220, 30, 30))
    # Highlight
    d.ellipse([140, 170, 250, 300], fill=(240, 70, 70))
    # Stem
    d.rounded_rectangle([240, 80, 255, 150], radius=5, fill=(100, 70, 30))
    # Leaf
    d.ellipse([255, 80, 340, 135], fill=(60, 180, 60))
    d.line([(255, 108), (335, 100)], fill=(40, 140, 40), width=2)
    return img


def draw_banana():
    img, d = new_image()
    # Banana curve (using arc-like shapes)
    # Main body
    for offset in range(60):
        x = 100 + offset
        y_top = 120 + int(40 * math.sin(math.radians(offset * 2.5)))
        y_bot = y_top + 120
        color_val = 255 - offset // 2
        d.line([(x, y_top), (x, y_bot)], fill=(color_val, color_val - 20, 30), width=2)
    # Better banana shape
    # Crescent shape
    d.ellipse([60, 50, 440, 450], fill=(255, 230, 40))
    d.ellipse([40, 80, 430, 480], fill=(30, 30, 45))  # cut out inner curve
    d.ellipse([60, 50, 440, 400], fill=(255, 230, 40))
    d.ellipse([50, 100, 440, 450], fill=(30, 30, 45))  # cut more
    # Simplified: just draw a curved banana
    img2, d2 = new_image()
    # Banana body
    pts = []
    for i in range(100):
        t = i / 99
        x = 80 + t * 340
        y = 250 - 120 * math.sin(t * math.pi)
        pts.append((x, y))
    for i in range(99, -1, -1):
        t = i / 99
        x = 80 + t * 340
        y = 310 - 100 * math.sin(t * math.pi)
        pts.append((x, y))
    d2.polygon(pts, fill=(255, 230, 40))
    # Highlight
    pts_h = []
    for i in range(100):
        t = i / 99
        x = 100 + t * 300
        y = 260 - 110 * math.sin(t * math.pi)
        pts_h.append((x, y))
    for i in range(99, -1, -1):
        t = i / 99
        x = 100 + t * 300
        y = 280 - 105 * math.sin(t * math.pi)
        pts_h.append((x, y))
    d2.polygon(pts_h, fill=(255, 245, 100))
    # Tip
    d2.ellipse([80, 230, 120, 290], fill=(140, 120, 30))
    # Stem
    d2.rounded_rectangle([400, 190, 430, 220], radius=5, fill=(140, 120, 30))
    return img2


def draw_orange():
    img, d = new_image()
    # Body
    d.ellipse([80, 80, 420, 420], fill=(255, 165, 0))
    d.ellipse([90, 90, 410, 410], fill=(255, 180, 30))
    # Highlight
    d.ellipse([130, 110, 260, 230], fill=(255, 210, 80))
    # Texture dots
    import random
    rng = random.Random(42)
    for _ in range(30):
        x = rng.randint(120, 380)
        y = rng.randint(120, 380)
        # Only draw if inside the circle
        if (x - 250) ** 2 + (y - 250) ** 2 < 160 ** 2:
            d.ellipse([x, y, x + 4, y + 4], fill=(240, 150, 0))
    # Stem
    d.rounded_rectangle([235, 55, 260, 95], radius=5, fill=(80, 120, 30))
    # Leaf
    d.ellipse([255, 50, 330, 95], fill=(60, 170, 50))
    return img


def draw_grape():
    img, d = new_image()
    # Stem
    d.rounded_rectangle([240, 40, 255, 120], radius=5, fill=(80, 120, 30))
    d.ellipse([250, 35, 320, 80], fill=(60, 160, 50))
    # Grape cluster
    grape_color = (140, 40, 180)
    grape_light = (170, 70, 210)
    positions = [
        (250, 140), (210, 160), (290, 160),
        (180, 200), (250, 200), (320, 200),
        (195, 250), (265, 250), (335, 250),
        (210, 300), (280, 300),
        (250, 340),
    ]
    r = 38
    for px, py in positions:
        d.ellipse([px - r, py - r, px + r, py + r], fill=grape_color)
    for px, py in positions:
        d.ellipse([px - r + 8, py - r + 5, px - 5, py - 5], fill=grape_light)
    return img


def draw_mango():
    img, d = new_image()
    # Body (oval, slightly tilted via skew)
    d.ellipse([80, 90, 420, 410], fill=(255, 180, 30))
    d.ellipse([90, 100, 410, 400], fill=(255, 200, 50))
    # Red blush on top
    d.ellipse([80, 80, 300, 250], fill=(240, 100, 30))
    d.ellipse([100, 100, 280, 230], fill=(250, 140, 40))
    # Highlight
    d.ellipse([140, 130, 250, 220], fill=(255, 210, 80))
    # Stem
    d.rounded_rectangle([240, 60, 260, 105], radius=5, fill=(100, 80, 30))
    return img


def draw_watermelon():
    img, d = new_image()
    # Rind (full circle, cut to semicircle later)
    d.ellipse([50, 80, 450, 420], fill=(40, 160, 50))
    # Inner rind
    d.ellipse([65, 95, 435, 405], fill=(140, 220, 140))
    # Red flesh
    d.ellipse([80, 110, 420, 390], fill=(255, 60, 60))
    # Cut to show flat bottom — cover bottom with bg
    d.rectangle([0, 300, 500, 500], fill=(30, 30, 45))
    # Re-draw rind curve at bottom
    d.chord([50, 80, 450, 420], 0, 180, fill=(40, 160, 50))
    d.chord([65, 95, 435, 405], 0, 180, fill=(140, 220, 140))
    d.chord([80, 110, 420, 390], 0, 180, fill=(255, 60, 60))
    # Seeds
    seed_color = (40, 30, 20)
    seeds = [(180, 220), (250, 250), (320, 220), (210, 280), (290, 280), (250, 190)]
    for sx, sy in seeds:
        d.ellipse([sx - 6, sy - 8, sx + 6, sy + 8], fill=seed_color)
    # Highlight on flesh
    d.ellipse([150, 160, 260, 220], fill=(255, 100, 100))
    return img


def draw_strawberry():
    img, d = new_image()
    # Body (triangle-ish with rounded top)
    d.ellipse([120, 100, 380, 300], fill=(230, 30, 40))
    d.polygon([(130, 230), (370, 230), (250, 440)], fill=(230, 30, 40))
    # Highlight
    d.ellipse([150, 130, 280, 240], fill=(255, 80, 80))
    # Seeds
    seed_color = (255, 230, 100)
    import random
    rng = random.Random(99)
    for _ in range(25):
        x = rng.randint(160, 340)
        y = rng.randint(150, 380)
        # Rough bounds check
        if y > 250:
            max_x = 250 + (440 - y) * 0.6
            min_x = 250 - (440 - y) * 0.6
            if not (min_x < x < max_x):
                continue
        d.ellipse([x - 3, y - 4, x + 3, y + 4], fill=seed_color)
    # Leaves on top
    for angle in range(-60, 70, 25):
        a = math.radians(angle)
        lx = 250 + 70 * math.cos(a - math.pi / 2)
        ly = 110 + 70 * math.sin(a - math.pi / 2)
        d.ellipse([lx - 20, ly - 10, lx + 20, ly + 20], fill=(40, 170, 40))
    # Stem
    d.rounded_rectangle([243, 55, 257, 110], radius=5, fill=(80, 130, 30))
    return img


def draw_pineapple():
    img, d = new_image()
    # Body (oval)
    d.ellipse([110, 140, 390, 450], fill=(220, 170, 30))
    d.ellipse([120, 150, 380, 440], fill=(240, 190, 40))
    # Cross-hatch pattern
    for i in range(8):
        y = 170 + i * 35
        d.line([(130, y), (370, y)], fill=(200, 150, 20), width=2)
    for i in range(7):
        x = 140 + i * 35
        d.line([(x, 160), (x, 430)], fill=(200, 150, 20), width=2)
    # Crown leaves
    leaves = [
        (250, 140, 230, 30), (250, 140, 270, 30),
        (250, 140, 190, 50), (250, 140, 310, 50),
        (250, 140, 170, 80), (250, 140, 330, 80),
        (250, 140, 200, 15), (250, 140, 300, 15),
        (250, 140, 160, 40), (250, 140, 340, 40),
    ]
    for x1, y1, x2, y2 in leaves:
        d.line([(x1, y1), (x2, y2)], fill=(40, 160, 40), width=8)
        d.line([(x1, y1), (x2, y2)], fill=(60, 180, 50), width=4)
    return img


FRUIT_DRAWERS = {
    "apple": draw_apple,
    "banana": draw_banana,
    "orange": draw_orange,
    "grape": draw_grape,
    "mango": draw_mango,
    "watermelon": draw_watermelon,
    "strawberry": draw_strawberry,
    "pineapple": draw_pineapple,
}


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("Generating flashcard images...")
    print(f"Output: {OUT_DIR}\n")

    print("Shapes:")
    for name, drawer in SHAPE_DRAWERS.items():
        save(drawer(), "shapes", name)

    print("\nColors:")
    for name, rgb in COLOR_VALUES.items():
        save(draw_color(name, rgb), "colors", name)

    print("\nAnimals:")
    for name, drawer in ANIMAL_DRAWERS.items():
        save(drawer(), "animals", name)

    print("\nFruits:")
    for name, drawer in FRUIT_DRAWERS.items():
        save(drawer(), "fruits", name)

    total = len(SHAPE_DRAWERS) + len(COLOR_VALUES) + len(ANIMAL_DRAWERS) + len(FRUIT_DRAWERS)
    print(f"\nDone! Generated {total} flashcard images.")


if __name__ == "__main__":
    main()
