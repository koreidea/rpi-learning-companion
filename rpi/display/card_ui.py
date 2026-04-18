"""Card-based UI system for TFT display (480x320).

Replaces the old settings panel. Activated by long press on home screen.
Horizontal scrollable cards → tap to enter → sub-screens.

Cards:
  0. Voice Companion — returns to home (animated face)
  1. Arts & Crafts — drawing list + Imagine Art (projector integration)
  2. Bedtime Stories — illustrated stories with TTS narration
  3. Encyclopedia — 21st century skills with activities
  4. Settings — volume, language, mode, mic, projector, car, wifi, sleep
"""

# ─── Card Definitions ──────────────────────────────────────

MAIN_CARDS = [
    {
        "id": "voice",
        "title": "Voice",
        "subtitle": "Buddy",
        "icon": "mic",
        "color": (80, 200, 120),       # Green
        "bg": (20, 60, 35),
    },
    {
        "id": "arts",
        "title": "Arts &",
        "subtitle": "Crafts",
        "icon": "palette",
        "color": (255, 130, 60),        # Orange
        "bg": (70, 35, 15),
    },
    {
        "id": "stories",
        "title": "Bedtime",
        "subtitle": "Stories",
        "icon": "moon",
        "color": (180, 140, 255),       # Lavender
        "bg": (40, 25, 65),
    },
    {
        "id": "encyclopedia",
        "title": "Encyclo",
        "subtitle": "pedia",
        "icon": "book",
        "color": (100, 160, 255),       # Blue
        "bg": (25, 40, 70),
    },
    {
        "id": "settings",
        "title": "Settings",
        "subtitle": "",
        "icon": "gear",
        "color": (180, 180, 200),       # Gray
        "bg": (40, 40, 50),
    },
]

# ─── Arts & Crafts Items ──────────────────────────────────

ARTS_ITEMS = [
    {"id": "imagine", "name": "Imagine Art", "desc": "Describe & trace!", "color": (255, 100, 200), "icon": "star"},
    {"id": "realistic dog", "name": "Realistic Dog", "desc": "Labrador portrait", "color": (200, 150, 80)},
    {"id": "cat", "name": "Cat", "desc": "Cute kitty", "color": (180, 130, 200)},
    {"id": "dog", "name": "Dog", "desc": "Cartoon puppy", "color": (140, 180, 100)},
    {"id": "butterfly", "name": "Butterfly", "desc": "Wings & antenna", "color": (255, 180, 80)},
    {"id": "fish", "name": "Fish", "desc": "Underwater friend", "color": (80, 180, 255)},
    {"id": "house", "name": "House", "desc": "Cozy home", "color": (200, 120, 80)},
    {"id": "tree", "name": "Tree", "desc": "Apple tree", "color": (80, 200, 80)},
    {"id": "flower", "name": "Flower", "desc": "Garden bloom", "color": (255, 100, 150)},
    {"id": "star", "name": "Star", "desc": "Twinkle twinkle", "color": (255, 220, 60)},
    {"id": "sun", "name": "Sun", "desc": "Sunshine face", "color": (255, 200, 50)},
    {"id": "car", "name": "Car", "desc": "Vroom vroom", "color": (100, 150, 255)},
    {"id": "craft dog", "name": "Paper Dog", "desc": "Cut & fold 3D", "color": (200, 160, 100), "icon": "scissors"},
    {"id": "craft house", "name": "Paper House", "desc": "Cut & fold 3D", "color": (180, 120, 80), "icon": "scissors"},
]

# ─── 21st Century Skills (Encyclopedia) ──────────────────

SKILLS_DATA = [
    {"id": "critical_thinking", "name": "Critical Thinking", "short": "Think & Solve",
     "color": (92, 107, 192), "desc": "Learn to analyze, question, and solve problems like a detective!",
     "activities": ["Mystery Solver", "Odd One Out", "Sorting Challenge"]},
    {"id": "creativity", "name": "Creativity", "short": "Create & Invent",
     "color": (255, 112, 67), "desc": "Use your imagination to create amazing new things!",
     "activities": ["Drawing Prompt", "What If Machine", "Story Builder"]},
    {"id": "communication", "name": "Communication", "short": "Talk & Listen",
     "color": (38, 166, 154), "desc": "Express your ideas clearly and listen to understand others!",
     "activities": ["Show and Tell", "Interview Game", "Teach the Bot"]},
    {"id": "collaboration", "name": "Collaboration", "short": "Teamwork",
     "color": (66, 165, 245), "desc": "Work together with friends to achieve great things!",
     "activities": ["Build Together", "Compliment Circle", "Story Relay"]},
    {"id": "leadership", "name": "Leadership", "short": "Lead the Way",
     "color": (171, 71, 188), "desc": "Inspire others and make good decisions for everyone!",
     "activities": ["Leader of the Day", "What Would You Do"]},
    {"id": "emotional_intelligence", "name": "Emotional Intelligence", "short": "Feel & Care",
     "color": (236, 64, 122), "desc": "Understand your feelings and care about how others feel!",
     "activities": ["Emotion Check-in", "Emotion Detective", "Manners Roleplay"]},
    {"id": "adaptability", "name": "Adaptability", "short": "Be Flexible",
     "color": (102, 187, 106), "desc": "Learn to handle changes and bounce back stronger!",
     "activities": ["Plot Twist", "New Day New Way"]},
    {"id": "financial_literacy", "name": "Financial Literacy", "short": "Money Smart",
     "color": (141, 110, 99), "desc": "Learn how money works and make smart choices!",
     "activities": ["Budget Boss", "Lemonade Stand"]},
    {"id": "environment", "name": "Environment", "short": "Go Green",
     "color": (76, 175, 80), "desc": "Protect our planet and love nature!",
     "activities": ["Eco Detective", "Nature Narrator", "Future City"]},
    {"id": "cultural", "name": "Cultural Awareness", "short": "World Explorer",
     "color": (41, 182, 246), "desc": "Discover amazing cultures and traditions around the world!",
     "activities": ["World Traveler", "Festival Friend", "Language of Day"]},
    {"id": "health", "name": "Health & Wellness", "short": "Stay Healthy",
     "color": (239, 83, 80), "desc": "Take care of your body and mind to feel great!",
     "activities": ["Exercise Buddy", "Mindful Minute", "Sleep Stories"]},
    {"id": "entrepreneurial", "name": "Entrepreneurship", "short": "Big Ideas",
     "color": (255, 167, 38), "desc": "Turn your big ideas into real projects!",
     "activities": ["Market Day", "Problem Spotter"]},
    {"id": "ethics", "name": "Ethics", "short": "Do Right",
     "color": (126, 87, 194), "desc": "Learn what's fair and be a good citizen!",
     "activities": ["Good Citizen", "Ethics Engine"]},
    {"id": "design_thinking", "name": "Design Thinking", "short": "Design & Build",
     "color": (38, 198, 218), "desc": "Design creative solutions step by step!",
     "activities": ["Sequence Challenge", "Design Detective", "Rapid Prototype"]},
    {"id": "info_literacy", "name": "Info Literacy", "short": "Smart Search",
     "color": (92, 107, 192), "desc": "Find and use information wisely!",
     "activities": ["Source Safari", "Ask Better Questions"]},
    {"id": "self_direction", "name": "Self-Direction", "short": "Own Your Path",
     "color": (255, 202, 40), "desc": "Set goals and take charge of your learning!",
     "activities": ["Goal Buddy", "Curiosity Hour"]},
    {"id": "media_creation", "name": "Media Creation", "short": "Be a Creator",
     "color": (255, 138, 101), "desc": "Make videos, stories, and cool digital art!",
     "activities": ["Comic Creator", "Podcast Producer"]},
    {"id": "scientific", "name": "Scientific Thinking", "short": "Discover!",
     "color": (0, 150, 136), "desc": "Ask questions and discover how the world works!",
     "activities": ["Nature Explorer", "Lab Partner", "Science Myth Buster"]},
    {"id": "time_management", "name": "Time Management", "short": "Use Time Well",
     "color": (142, 36, 170), "desc": "Plan your day and make every minute count!",
     "activities": ["Plan My Day", "Pomodoro Buddy"]},
    {"id": "digital_citizenship", "name": "Digital Citizenship", "short": "Safe Online",
     "color": (120, 144, 156), "desc": "Stay safe and be kind in the digital world!",
     "activities": ["Privacy Guard", "Real or Fake"]},
]

# ─── Settings Items ──────────────────────────────────────

SETTINGS_ITEMS = [
    {"id": "volume",   "label": "Volume",     "icon": "speaker",  "color": (100, 200, 255)},
    {"id": "language", "label": "Language",    "icon": "globe",    "color": (255, 180, 60)},
    {"id": "mode",     "label": "AI Mode",     "icon": "brain",    "color": (180, 100, 255)},
    {"id": "mic",      "label": "Microphone",  "icon": "mic",      "color": (80, 200, 120)},
    {"id": "projector","label": "Projector",   "icon": "projector","color": (255, 130, 60)},
    {"id": "car",      "label": "Car",         "icon": "car_icon", "color": (100, 160, 255)},
    {"id": "wifi",     "label": "WiFi",        "icon": "wifi",     "color": (60, 220, 180)},
    {"id": "sleep",    "label": "Sleep",       "icon": "moon",     "color": (140, 120, 200)},
]

LANGUAGES = [
    {"id": "en", "name": "English",  "flag": "EN", "color": (100, 200, 255)},
    {"id": "hi", "name": "Hindi",    "flag": "HI", "color": (255, 160, 50)},
    {"id": "te", "name": "Telugu",   "flag": "TE", "color": (80, 200, 120)},
]

# ─── Bedtime Stories ─────────────────────────────────────

BEDTIME_STORIES = [
    {
        "id": "imagine_story",
        "title": "Imagine Story",
        "desc": "Create your own story!",
        "color": (255, 200, 80),
        "bg_color": (30, 20, 10),
        "pages": [],  # Generated dynamically by AI
    },
    {
        "id": "sleepy_moon",
        "title": "The Sleepy Moon",
        "desc": "Moon yawns and stars come out",
        "color": (200, 180, 255),
        "bg_color": (15, 10, 40),
        "pages": [
            {
                "text": "High above the rooftops, the big round Moon yawned wide. 'I'm so sleepy,' she said softly.",
                "scene": "moon_yawn",
            },
            {
                "text": "One by one, tiny stars popped out of the dark blue sky. 'We'll keep watch!' they twinkled.",
                "scene": "stars_appear",
            },
            {
                "text": "The Moon pulled a fluffy cloud blanket over herself and smiled. 'Goodnight, little stars.'",
                "scene": "moon_cloud",
            },
            {
                "text": "Down below, a little child looked up and whispered, 'Goodnight Moon, goodnight stars.' And drifted off to sleep.",
                "scene": "child_window",
            },
        ],
    },
    {
        "id": "little_star",
        "title": "The Littlest Star",
        "desc": "A tiny star learns to shine",
        "color": (255, 220, 80),
        "bg_color": (10, 10, 35),
        "pages": [
            {
                "text": "In a sky full of bright stars, one tiny star named Twinkle felt sad. 'I'm too small to shine,' she sighed.",
                "scene": "tiny_star_sad",
            },
            {
                "text": "The wise old North Star said, 'Every star has its own special glow. Just believe in yourself!'",
                "scene": "north_star",
            },
            {
                "text": "Twinkle took a deep breath and shined with all her heart. A warm golden glow burst out!",
                "scene": "star_glow",
            },
            {
                "text": "A child on Earth pointed up and said, 'Look! That's the prettiest star!' Twinkle beamed with joy.",
                "scene": "child_pointing",
            },
        ],
    },
    {
        "id": "bear_dream",
        "title": "Little Bear's Dream",
        "desc": "A bear cub's dreamy adventure",
        "color": (180, 140, 100),
        "bg_color": (20, 15, 30),
        "pages": [
            {
                "text": "Little Bear curled up in his cozy cave. Mama Bear kissed his head. 'Sweet dreams, my little one.'",
                "scene": "bear_cave",
            },
            {
                "text": "In his dream, Little Bear floated on a fluffy cloud over a rainbow river full of honey fish!",
                "scene": "cloud_ride",
            },
            {
                "text": "He met a friendly owl who said, 'Welcome to Dreamland! Everything here is made of starlight.'",
                "scene": "owl_dreamland",
            },
            {
                "text": "Little Bear hugged the owl goodbye and floated back to his cave. He smiled in his sleep, warm and safe.",
                "scene": "bear_sleeping",
            },
        ],
    },
    {
        "id": "night_garden",
        "title": "The Night Garden",
        "desc": "Flowers that bloom under moonlight",
        "color": (255, 130, 200),
        "bg_color": (10, 20, 15),
        "pages": [
            {
                "text": "When the sun goes down and everyone sleeps, a secret garden comes alive with moonflowers!",
                "scene": "moonflowers",
            },
            {
                "text": "Tiny fireflies dance between the glowing petals, lighting up the garden like fairy lanterns.",
                "scene": "fireflies",
            },
            {
                "text": "A little fairy named Luna waters each flower with drops of starlight from her tiny watering can.",
                "scene": "fairy_water",
            },
            {
                "text": "As dawn peeks over the hill, the moonflowers close gently. 'See you tonight,' Luna whispers. Sweet dreams!",
                "scene": "dawn_close",
            },
        ],
    },
    {
        "id": "owl_kitten",
        "title": "The Owl & The Kitten",
        "desc": "Unlikely friends at night",
        "color": (160, 200, 140),
        "bg_color": (15, 20, 10),
        "pages": [
            {
                "text": "A tiny kitten couldn't sleep. She crept outside and met a wise old owl sitting on a branch. 'Hoo are you?'",
                "scene": "owl_meets_kitten",
            },
            {
                "text": "'I can't sleep,' said the kitten. 'Let me sing you a lullaby,' hooted the owl softly.",
                "scene": "owl_singing",
            },
            {
                "text": "The owl sang of distant mountains, sparkling streams, and fields of soft, swaying grass.",
                "scene": "lullaby_scene",
            },
            {
                "text": "The kitten purred and curled up under the branch. 'Thank you, Mr. Owl,' she whispered, and fell fast asleep.",
                "scene": "kitten_asleep",
            },
        ],
    },
    {
        "id": "counting_sheep",
        "title": "Counting Sheep",
        "desc": "One sheep, two sheep, zzz...",
        "color": (220, 220, 240),
        "bg_color": (15, 18, 35),
        "pages": [
            {
                "text": "A little lamb couldn't fall asleep. 'Try counting us!' said the sheep in the meadow.",
                "scene": "sheep_meadow",
            },
            {
                "text": "One fluffy sheep jumped over the fence. Then two. Then three! Each one softer than the last.",
                "scene": "sheep_jumping",
            },
            {
                "text": "By the time the little lamb counted ten sheep, her eyes were getting heavy. 'Seven... eight...'",
                "scene": "sheep_sleepy",
            },
            {
                "text": "The sheep tucked the lamb in with a woolly blanket. 'Goodnight, little one.' Zzzzz...",
                "scene": "sheep_tuck_in",
            },
        ],
    },
    {
        "id": "dream_train",
        "title": "The Dream Train",
        "desc": "A magical ride to Dreamland",
        "color": (100, 180, 255),
        "bg_color": (10, 15, 35),
        "pages": [
            {
                "text": "Every night at bedtime, a magical silver train appears at the edge of your pillow. All aboard!",
                "scene": "train_pillow",
            },
            {
                "text": "The Dream Train chugs through cotton-candy clouds, past chocolate mountains and lemonade lakes.",
                "scene": "candy_land",
            },
            {
                "text": "At each station, friendly dream creatures wave hello. A rainbow dragon, a dancing penguin!",
                "scene": "dream_station",
            },
            {
                "text": "The last stop is your very own dream. The train whistles softly. 'See you tomorrow night!' Choo choo...",
                "scene": "last_stop",
            },
        ],
    },
    {
        "id": "goodnight_forest",
        "title": "Goodnight Forest",
        "desc": "Saying goodnight to everyone",
        "color": (80, 180, 100),
        "bg_color": (10, 20, 10),
        "pages": [
            {
                "text": "Goodnight tall trees with your leafy arms. Goodnight flowers closing your petals tight.",
                "scene": "forest_trees",
            },
            {
                "text": "Goodnight rabbits in your burrow. Goodnight deer resting by the stream.",
                "scene": "forest_animals",
            },
            {
                "text": "Goodnight cricket with your sleepy song. Goodnight firefly with your gentle glow.",
                "scene": "cricket_firefly",
            },
            {
                "text": "Goodnight forest, goodnight world, goodnight me. Tomorrow we'll play again. Sweet dreams...",
                "scene": "forest_night",
            },
        ],
    },
]


# ─── Story Scene Drawing ────────────────────────────────────

def draw_scene(draw, scene_id, width, height, frame=0):
    """Draw an illustrated scene for a story page using PIL primitives.

    All coordinates are proportional to width/height so scenes scale
    correctly from TFT (480×320) to projector (1920×1080).

    Args:
        draw: PIL ImageDraw object
        scene_id: which scene to render
        width, height: drawing area dimensions
        frame: animation frame counter for subtle motion
    """
    import math
    cx, cy = width // 2, height // 2
    # Scale factor relative to 480×320 base
    sx = width / 480.0
    sy = height / 320.0
    s = min(sx, sy)  # uniform scale for circles/sizes

    def S(v):
        """Scale a pixel value."""
        return max(1, int(v * s))

    def SX(v):
        """Scale x-coordinate."""
        return int(v * sx)

    def SY(v):
        """Scale y-coordinate."""
        return int(v * sy)

    def W(v):
        """Scale line width."""
        return max(1, int(v * s))

    def _stars(count=12, area_w=None, area_h=None):
        """Scatter stars in the scene."""
        import random
        rng = random.Random(hash(scene_id) + 42)
        aw = area_w or width
        ah = area_h or (height * 2 // 3)
        for _ in range(count):
            x = rng.randint(5, aw - 5)
            y = rng.randint(5, ah - 5)
            sz = S(rng.choice([1, 1, 2, 2, 3]))
            bright = rng.randint(150, 255)
            c = (bright, bright, rng.randint(180, 255))
            draw.ellipse([x - sz, y - sz, x + sz, y + sz], fill=c)

    def _twinkle_star(x, y, size=6):
        """Draw a 4-point twinkle star."""
        sz = S(size)
        draw.line([x, y - sz, x, y + sz], fill=(255, 255, 200), width=W(2))
        draw.line([x - sz, y, x + sz, y], fill=(255, 255, 200), width=W(2))
        draw.line([x - sz//2, y - sz//2, x + sz//2, y + sz//2], fill=(255, 255, 180), width=W(1))
        draw.line([x + sz//2, y - sz//2, x - sz//2, y + sz//2], fill=(255, 255, 180), width=W(1))

    def _moon(mx, my, r, phase="full"):
        """Draw moon at position."""
        r = S(r)
        draw.ellipse([mx - r, my - r, mx + r, my + r], fill=(255, 250, 200))
        # Subtle glow
        for g in range(3):
            gr = r + S(3 + g * 3)
            draw.ellipse([mx - gr, my - gr, mx + gr, my + gr],
                         outline=(60, 55, 30), width=W(1))
        if phase == "crescent":
            off = S(12)
            draw.ellipse([mx - r + off, my - r - S(2), mx + r + off, my + r + S(2)],
                         fill=None)

    def _ground(y_start, color=(20, 50, 20)):
        draw.rectangle([0, y_start, width, height], fill=color)
        # Subtle grass texture
        import random
        rng = random.Random(hash(scene_id) + 99)
        for _ in range(int(20 * sx)):
            gx = rng.randint(0, width)
            gy = rng.randint(y_start, min(y_start + S(8), height))
            gc = tuple(min(255, c + rng.randint(-10, 15)) for c in color)
            draw.line([gx, gy, gx + S(rng.randint(2, 5)), gy - S(rng.randint(2, 4))],
                      fill=gc, width=W(1))

    def _hills(y_base, color=(20, 45, 20), count=3):
        """Draw rolling hills."""
        import random
        rng = random.Random(hash(scene_id) + 77)
        for i in range(count):
            hx = int(width * (i + 0.5) / count) + rng.randint(-SX(30), SX(30))
            hw = S(rng.randint(80, 150))
            hh = S(rng.randint(25, 50))
            hc = tuple(min(255, c + rng.randint(-8, 8)) for c in color)
            draw.ellipse([hx - hw, y_base - hh, hx + hw, y_base + hh], fill=hc)

    def _tree(tx, ty, trunk_h=30, crown_r=18):
        th = S(trunk_h)
        cr = S(crown_r)
        tw = S(5)
        draw.rectangle([tx - tw, ty - th, tx + tw, ty], fill=(90, 60, 30))
        # Multi-layer crown for depth
        draw.ellipse([tx - cr, ty - th - cr, tx + cr, ty - th + cr // 2], fill=(25, 80, 25))
        draw.ellipse([tx - cr + S(4), ty - th - cr + S(3),
                       tx + cr - S(4), ty - th + cr // 2 - S(2)], fill=(35, 100, 35))

    def _pine_tree(tx, ty, h=50):
        """Draw a pine/christmas tree shape."""
        th = S(h)
        tw = S(4)
        draw.rectangle([tx - tw, ty - S(10), tx + tw, ty], fill=(80, 55, 25))
        for i in range(3):
            level_y = ty - S(10) - i * (th // 3)
            bw = S(22 - i * 5)
            draw.polygon([(tx - bw, level_y), (tx + bw, level_y),
                          (tx, level_y - th // 3)], fill=(20, 70 + i * 10, 20))

    def _house(hx, hy, w=50, h=35):
        w, h = S(w), S(h)
        draw.rectangle([hx, hy, hx + w, hy + h], fill=(80, 60, 50))
        draw.polygon([(hx - S(5), hy), (hx + w + S(5), hy), (hx + w // 2, hy - S(20))],
                     fill=(150, 60, 40))
        # Window with warm glow + glow effect
        wx, wy = hx + w // 2 - S(7), hy + S(8)
        ww, wh = S(14), S(14)
        # Glow
        for g in range(3):
            gg = S(3 + g * 2)
            draw.rectangle([wx - gg, wy - gg, wx + ww + gg, wy + wh + gg],
                           fill=(80 - g * 15, 60 - g * 10, 15))
        draw.rectangle([wx, wy, wx + ww, wy + wh], fill=(255, 220, 100))
        # Window cross
        draw.line([wx + ww // 2, wy, wx + ww // 2, wy + wh], fill=(120, 90, 50), width=W(1))
        draw.line([wx, wy + wh // 2, wx + ww, wy + wh // 2], fill=(120, 90, 50), width=W(1))
        # Door
        dx = hx + w // 2 - S(5)
        draw.rectangle([dx, hy + h - S(15), dx + S(10), hy + h], fill=(100, 70, 40))

    def _cloud(ccx, ccy, cw=40):
        """Draw a fluffy cloud."""
        cw = S(cw)
        ch = cw // 2
        for ox, oy, rr in [(-cw//3, 0, ch), (0, -ch//3, ch), (cw//3, 0, ch),
                            (-cw//6, ch//4, ch*2//3), (cw//6, ch//4, ch*2//3)]:
            draw.ellipse([ccx + ox - rr, ccy + oy - rr, ccx + ox + rr, ccy + oy + rr],
                         fill=(200, 200, 220))

    def _bear(bx, by, size=20):
        """Draw a simple bear."""
        r = S(size)
        # Body
        draw.ellipse([bx - r, by - r // 2, bx + r, by + r], fill=(140, 95, 60))
        # Head
        hr = r * 2 // 3
        draw.ellipse([bx - hr, by - r - hr, bx + hr, by - r + hr // 3], fill=(140, 95, 60))
        # Ears
        er = hr // 3
        draw.ellipse([bx - hr + er, by - r - hr - er, bx - hr + er * 3, by - r - hr + er],
                     fill=(120, 80, 50))
        draw.ellipse([bx + hr - er * 3, by - r - hr - er, bx + hr - er, by - r - hr + er],
                     fill=(120, 80, 50))
        # Snout
        draw.ellipse([bx - er, by - r - S(3), bx + er, by - r + er], fill=(180, 140, 100))
        # Eyes
        draw.ellipse([bx - hr // 2 - S(2), by - r - hr // 2 - S(2),
                       bx - hr // 2 + S(2), by - r - hr // 2 + S(2)], fill=(30, 20, 10))
        draw.ellipse([bx + hr // 2 - S(2), by - r - hr // 2 - S(2),
                       bx + hr // 2 + S(2), by - r - hr // 2 + S(2)], fill=(30, 20, 10))

    def _owl(ox, oy, size=16):
        """Draw an owl."""
        r = S(size)
        # Body
        draw.ellipse([ox - r, oy - r, ox + r, oy + r], fill=(130, 110, 80))
        # Eye circles
        er = r * 2 // 3
        draw.ellipse([ox - er - S(2), oy - er, ox - S(2), oy], fill=(255, 255, 200))
        draw.ellipse([ox + S(2), oy - er, ox + er + S(2), oy], fill=(255, 255, 200))
        # Pupils
        pr = S(3)
        draw.ellipse([ox - er // 2 - pr, oy - er // 2 - pr,
                       ox - er // 2 + pr, oy - er // 2 + pr], fill=(30, 30, 30))
        draw.ellipse([ox + er // 2 - pr, oy - er // 2 - pr,
                       ox + er // 2 + pr, oy - er // 2 + pr], fill=(30, 30, 30))
        # Beak
        draw.polygon([(ox - S(3), oy + S(2)), (ox + S(3), oy + S(2)), (ox, oy + S(7))],
                     fill=(210, 160, 50))
        # Ear tufts
        draw.polygon([(ox - r, oy - r), (ox - r + S(5), oy - r - S(8)), (ox - r + S(10), oy - r)],
                     fill=(130, 110, 80))
        draw.polygon([(ox + r, oy - r), (ox + r - S(5), oy - r - S(8)), (ox + r - S(10), oy - r)],
                     fill=(130, 110, 80))

    def _kitten(kx, ky, size=10, sleeping=False):
        """Draw a kitten."""
        r = S(size)
        # Body (horizontal oval if sleeping)
        if sleeping:
            draw.ellipse([kx - r * 3 // 2, ky - r // 2, kx + r * 3 // 2, ky + r], fill=(210, 190, 170))
        else:
            draw.ellipse([kx - r, ky - r // 2, kx + r, ky + r], fill=(210, 190, 170))
        # Head
        hr = r * 2 // 3
        hx = kx - r if sleeping else kx
        draw.ellipse([hx - hr, ky - r - hr // 2, hx + hr, ky - r // 2 + hr // 2], fill=(210, 190, 170))
        # Ears
        ear = hr // 2
        draw.polygon([(hx - hr + S(2), ky - r), (hx - hr // 2, ky - r - ear * 2), (hx, ky - r)],
                     fill=(220, 200, 180))
        draw.polygon([(hx, ky - r), (hx + hr // 2, ky - r - ear * 2), (hx + hr - S(2), ky - r)],
                     fill=(220, 200, 180))
        # Eyes (closed if sleeping)
        if sleeping:
            draw.arc([hx - S(4), ky - r - S(2), hx - S(1), ky - r + S(2)], 0, 180,
                     fill=(80, 60, 40), width=W(1))
            draw.arc([hx + S(1), ky - r - S(2), hx + S(4), ky - r + S(2)], 0, 180,
                     fill=(80, 60, 40), width=W(1))
        else:
            draw.ellipse([hx - S(4), ky - r - S(2), hx - S(1), ky - r + S(2)], fill=(50, 180, 50))
            draw.ellipse([hx + S(1), ky - r - S(2), hx + S(4), ky - r + S(2)], fill=(50, 180, 50))
        # Tail
        tail_x = kx + r if sleeping else kx + r // 2
        draw.arc([tail_x, ky - S(5), tail_x + S(15), ky + S(10)], 180, 360,
                 fill=(210, 190, 170), width=W(2))

    def _sheep(shx, shy, size=14):
        """Draw a fluffy sheep."""
        r = S(size)
        # Fluffy body (overlapping circles)
        for dx, dy in [(-r//3, 0), (r//3, 0), (0, -r//4), (0, r//4), (-r//5, -r//3), (r//5, -r//3)]:
            draw.ellipse([shx + dx - r//2, shy + dy - r//2, shx + dx + r//2, shy + dy + r//2],
                         fill=(245, 245, 250))
        # Head
        draw.ellipse([shx - r//2, shy - r - S(2), shx + r//3, shy - r//2], fill=(70, 70, 75))
        # Legs
        for lx in [shx - r//3, shx + r//4]:
            draw.rectangle([lx, shy + r//3, lx + S(3), shy + r], fill=(60, 60, 60))

    def _fairy(fx, fy, size=8):
        """Draw a small fairy with wings."""
        r = S(size)
        # Wings
        wr = S(14)
        draw.ellipse([fx - wr, fy - r, fx - S(2), fy + r // 2],
                     outline=(200, 200, 255), width=W(1))
        draw.ellipse([fx + S(2), fy - r, fx + wr, fy + r // 2],
                     outline=(200, 200, 255), width=W(1))
        # Body
        draw.rectangle([fx - S(3), fy - S(2), fx + S(3), fy + r], fill=(200, 150, 255))
        # Head
        draw.ellipse([fx - S(4), fy - r, fx + S(4), fy - S(1)], fill=(255, 225, 200))
        # Wand with sparkle
        draw.line([fx + S(5), fy, fx + S(18), fy - S(12)], fill=(255, 230, 100), width=W(1))
        _twinkle_star(fx + S(18), fy - S(12), 3)

    def _train(tx, ty, size=40):
        """Draw a small train."""
        w = S(size)
        h = S(14)
        # Engine
        draw.rounded_rectangle([tx, ty, tx + w, ty + h], radius=S(3), fill=(180, 180, 210))
        # Cabin
        draw.rounded_rectangle([tx + w, ty + S(2), tx + w + S(18), ty + h], radius=S(2),
                               fill=(160, 160, 185))
        # Smokestack
        draw.rectangle([tx + S(8), ty - S(8), tx + S(14), ty], fill=(150, 150, 170))
        # Wheels
        for wx in [tx + S(8), tx + S(28), tx + w + S(8)]:
            draw.ellipse([wx - S(4), ty + h - S(2), wx + S(4), ty + h + S(5)],
                         fill=(100, 100, 120))
        # Smoke puffs
        for i, (dx, dy, pr) in enumerate([(S(-5), S(-15), S(6)), (S(-15), S(-22), S(8)), (S(-28), S(-28), S(5))]):
            draw.ellipse([tx + S(11) + dx - pr, ty + dy - pr,
                          tx + S(11) + dx + pr, ty + dy + pr], fill=(80, 80, 100))

    def _flower(fx, fy, petal_color=(200, 180, 255), center_color=(255, 255, 200), size=8):
        """Draw a flower with petals."""
        sz = S(size)
        # Stem
        draw.line([fx, fy, fx, fy + S(20)], fill=(30, 80, 30), width=W(2))
        # Petals
        for angle in range(0, 360, 60):
            px = fx + int(sz * math.cos(math.radians(angle)))
            py = fy + int(sz * math.sin(math.radians(angle)))
            pr = sz * 2 // 3
            draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=petal_color)
        # Center
        cr = sz // 2
        draw.ellipse([fx - cr, fy - cr, fx + cr, fy + cr], fill=center_color)

    def _music_notes(nx, ny, count=3):
        """Draw floating music notes."""
        for i in range(count):
            mx = nx + S(i * 18)
            my = ny - S(i * 12)
            draw.ellipse([mx - S(3), my, mx + S(3), my + S(4)], fill=(200, 180, 255))
            draw.line([mx + S(3), my - S(12), mx + S(3), my + S(2)], fill=(200, 180, 255), width=W(1))
            if i % 2 == 0:
                draw.line([mx + S(3), my - S(12), mx + S(9), my - S(10)], fill=(200, 180, 255), width=W(1))

    # ── Scene implementations ──

    if scene_id == "moon_yawn":
        _stars(18)
        _moon(cx, cy - SY(10), 50)
        mr = S(50)
        # Yawn mouth (wide open O)
        draw.arc([cx - S(16), cy - SY(10), cx + S(16), cy + S(14)], 0, 180,
                 fill=(120, 90, 60), width=W(3))
        # Sleepy half-closed eyes
        draw.arc([cx - S(25), cy - S(40), cx - S(8), cy - S(28)], 0, 180,
                 fill=(120, 90, 60), width=W(2))
        draw.arc([cx + S(8), cy - S(40), cx + S(25), cy - S(28)], 0, 180,
                 fill=(120, 90, 60), width=W(2))
        # Rooftops at bottom
        _ground(height - S(25), (20, 20, 35))
        for i in range(5):
            rx = SX(i * 100 + 20)
            draw.polygon([(rx, height - S(25)), (rx + SX(50), height - S(25)),
                          (rx + SX(25), height - S(45))], fill=(35, 30, 45))

    elif scene_id == "stars_appear":
        _stars(30)
        _moon(width - SX(60), SY(45), 30)
        # Big twinkling stars
        for px_frac, py_frac in [(0.17, 0.1), (0.42, 0.2), (0.73, 0.08), (0.31, 0.28), (0.6, 0.35)]:
            _twinkle_star(int(width * px_frac), int(height * py_frac), 8)

    elif scene_id == "moon_cloud":
        _stars(12)
        # Moon
        _moon(cx, cy - SY(20), 42)
        # Cloud blanket over bottom of moon
        for ox in range(-S(60), S(65), S(18)):
            draw.ellipse([cx + ox - S(25), cy + S(5), cx + ox + S(25), cy + S(30)],
                         fill=(190, 190, 215))
        # Extra cloud puffs
        for ox in range(-S(45), S(50), S(22)):
            draw.ellipse([cx + ox - S(18), cy + S(15), cx + ox + S(18), cy + S(35)],
                         fill=(180, 180, 210))

    elif scene_id == "child_window":
        _stars(12)
        _moon(width - SX(55), SY(35), 24)
        # Ground with grass
        _hills(height - S(20), (22, 40, 22), 4)
        _ground(height - S(20), (25, 38, 25))
        # House (bigger)
        _house(cx - S(35), height - S(80))
        # Child silhouette in window
        wx = cx - S(35) + S(50) // 2 - S(7)
        wy = height - S(80) + S(8)
        # Child head
        draw.ellipse([wx + S(2), wy + S(1), wx + S(12), wy + S(11)], fill=(70, 50, 35))
        # Small trees
        _tree(SX(60), height - S(20), 30, 14)
        _tree(width - SX(60), height - S(20), 25, 12)

    elif scene_id == "tiny_star_sad":
        _stars(14)
        _moon(width - SX(55), SY(40), 22, "crescent")
        # Sad tiny star in center (bigger for visibility)
        r = S(12)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(200, 200, 120))
        # Glow
        for g in range(3):
            gr = r + S(3 + g * 4)
            draw.ellipse([cx - gr, cy - gr, cx + gr, cy + gr],
                         outline=(100, 100, 60), width=W(1))
        # Sad face (bigger)
        draw.ellipse([cx - S(5), cy - S(4), cx - S(2), cy - S(1)], fill=(80, 80, 40))
        draw.ellipse([cx + S(2), cy - S(4), cx + S(5), cy - S(1)], fill=(80, 80, 40))
        draw.arc([cx - S(5), cy + S(2), cx + S(5), cy + S(8)], 180, 360,
                 fill=(80, 80, 40), width=W(2))
        # Other big stars around (happy)
        for px_frac, py_frac in [(0.15, 0.15), (0.8, 0.25), (0.25, 0.7), (0.7, 0.65)]:
            bx, by = int(width * px_frac), int(height * py_frac)
            br = S(6)
            draw.ellipse([bx - br, by - br, bx + br, by + br], fill=(255, 255, 200))

    elif scene_id == "north_star":
        _stars(10)
        # Big north star at top with rays
        nsx, nsy = cx, SY(40)
        _twinkle_star(nsx, nsy, 20)
        # Glow rings
        for g in range(4):
            gr = S(20 + g * 8)
            draw.ellipse([nsx - gr, nsy - gr, nsx + gr, nsy + gr],
                         outline=(120, 120, 80), width=W(1))
        # Tiny star below looking up
        r = S(8)
        draw.ellipse([cx - r, cy + S(30), cx + r, cy + S(30) + r * 2], fill=(220, 220, 140))
        # Hopeful face
        draw.ellipse([cx - S(3), cy + S(34), cx - S(1), cy + S(36)], fill=(100, 100, 60))
        draw.ellipse([cx + S(1), cy + S(34), cx + S(3), cy + S(36)], fill=(100, 100, 60))

    elif scene_id == "star_glow":
        # Radiant burst of light from center
        for r in range(S(80), S(5), -S(5)):
            alpha = max(30, 255 - int(r / s * 2))
            c = (alpha, alpha, min(255, alpha + 40))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=None, outline=c, width=W(2))
        # Bright center
        r = S(14)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 220))
        # Light rays
        for angle in range(0, 360, 30):
            rx = cx + int(S(60) * math.cos(math.radians(angle)))
            ry = cy + int(S(60) * math.sin(math.radians(angle)))
            draw.line([cx, cy, rx, ry], fill=(200, 200, 150), width=W(1))
        _stars(8)

    elif scene_id == "child_pointing":
        _stars(14)
        # Bright star at top right with glow
        star_x, star_y = cx + SX(80), SY(25)
        _twinkle_star(star_x, star_y, 10)
        # Ground
        _ground(height - S(30), (20, 42, 20))
        # Child silhouette at bottom
        chx, chy = cx - SX(25), height - S(65)
        draw.ellipse([chx - S(9), chy - S(9), chx + S(9), chy + S(9)], fill=(80, 55, 40))  # head
        draw.rectangle([chx - S(7), chy + S(9), chx + S(7), chy + S(35)], fill=(80, 55, 40))  # body
        # Pointing arm toward star
        draw.line([chx + S(7), chy + S(14), chx + S(45), chy - S(15)],
                  fill=(80, 55, 40), width=W(3))
        # Other arm down
        draw.line([chx - S(7), chy + S(14), chx - S(15), chy + S(28)],
                  fill=(80, 55, 40), width=W(2))
        # Grass
        _tree(SX(80), height - S(30), 28, 14)

    elif scene_id == "bear_cave":
        _ground(height - S(25), (45, 35, 28))
        # Cave (larger, more detailed)
        cave_w, cave_h = S(80), S(60)
        draw.arc([cx - cave_w, cy - cave_h // 2, cx + cave_w, cy + cave_h],
                 180, 360, fill=(90, 80, 65), width=W(5))
        draw.rectangle([cx - cave_w, cy + S(8), cx + cave_w, height - S(25)], fill=(35, 28, 22))
        # Cave interior darkness
        draw.ellipse([cx - cave_w + S(10), cy - S(5), cx + cave_w - S(10), cy + cave_h - S(15)],
                     fill=(25, 20, 15))
        # Mama bear
        _bear(cx - S(10), cy + S(10), 22)
        # Baby bear (smaller, curled up next to mama)
        draw.ellipse([cx + S(15), cy + S(12), cx + S(35), cy + S(28)], fill=(110, 75, 45))
        draw.ellipse([cx + S(18), cy + S(5), cx + S(32), cy + S(15)], fill=(110, 75, 45))
        _stars(5)

    elif scene_id == "cloud_ride":
        _stars(10)
        # Big fluffy cloud
        _cloud(cx, cy, 55)
        # Bear sitting on cloud
        _bear(cx, cy - S(25), 16)
        # Rainbow below
        colors = [(255, 50, 50), (255, 165, 0), (255, 255, 0), (0, 200, 0), (0, 120, 255), (148, 0, 255)]
        for i, c in enumerate(colors):
            r = S(90 - i * 10)
            draw.arc([cx - r, cy + S(25), cx + r, cy + S(25) + r], 0, 180, fill=c, width=W(4))
        # Honey fish in rainbow river
        for fx_off in [-S(30), S(10), S(50)]:
            fx = cx + fx_off
            fy = cy + S(80)
            draw.ellipse([fx - S(5), fy - S(3), fx + S(5), fy + S(3)], fill=(255, 200, 50))
            draw.polygon([(fx + S(5), fy), (fx + S(10), fy - S(4)), (fx + S(10), fy + S(4))],
                         fill=(255, 200, 50))

    elif scene_id == "owl_dreamland":
        _stars(18)
        # Sparkly dreamland background
        import random
        rng = random.Random(hash("owl_dream") + 42)
        for _ in range(15):
            dx = rng.randint(0, width)
            dy = rng.randint(0, height)
            dr = S(rng.randint(1, 3))
            draw.ellipse([dx - dr, dy - dr, dx + dr, dy + dr], fill=(180, 180, 255))
        # Owl
        _owl(cx - SX(50), cy, 20)
        # Bear
        _bear(cx + SX(50), cy, 16)
        # Dreamland text effect (starlight swirls)
        for angle in range(0, 360, 20):
            px = cx + int(S(70) * math.cos(math.radians(angle)))
            py = cy + int(S(50) * math.sin(math.radians(angle)))
            draw.ellipse([px - S(1), py - S(1), px + S(1), py + S(1)], fill=(200, 200, 255))

    elif scene_id == "bear_sleeping":
        _stars(10)
        _ground(height - S(30), (38, 30, 24))
        # Sleeping bear (curled up)
        bx, by = cx, cy + S(10)
        # Curled body
        draw.ellipse([bx - S(30), by - S(5), bx + S(30), by + S(25)], fill=(130, 90, 55))
        # Head resting
        draw.ellipse([bx - S(20), by - S(15), bx + S(5), by + S(5)], fill=(125, 85, 50))
        # Closed eyes
        draw.arc([bx - S(14), by - S(10), bx - S(6), by - S(5)], 0, 180,
                 fill=(60, 40, 25), width=W(1))
        # Smile
        draw.arc([bx - S(12), by - S(4), bx - S(4), by + S(2)], 0, 180,
                 fill=(60, 40, 25), width=W(1))
        # Zzz (floating, getting bigger)
        for i, (zx, zy, zsz) in enumerate([(S(25), -S(20), 10), (S(38), -S(35), 13), (S(52), -S(52), 16)]):
            from PIL import ImageFont
            try:
                fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", S(zsz))
            except Exception:
                fnt = ImageFont.load_default()
            draw.text((bx + zx, by + zy), "z", fill=(180, 180, 230), font=fnt)

    elif scene_id == "moonflowers":
        _stars(8)
        _moon(width - SX(50), SY(30), 22)
        _ground(height - S(40), (10, 35, 18))
        # Glowing moonflowers
        positions = [(0.13, 0.83), (0.31, 0.80), (0.50, 0.82), (0.69, 0.81), (0.87, 0.83)]
        for px_frac, py_frac in positions:
            fx = int(width * px_frac)
            fy = int(height * py_frac)
            _flower(fx, fy - S(15), (200, 180, 255), (255, 255, 200), 10)
            # Glow effect
            draw.ellipse([fx - S(12), fy - S(25), fx + S(12), fy - S(5)],
                         outline=(100, 80, 140), width=W(1))

    elif scene_id == "fireflies":
        _ground(height - S(40), (10, 28, 14))
        _stars(4)
        # Flowers on ground
        for px_frac in [0.17, 0.42, 0.73]:
            fx = int(width * px_frac)
            _flower(fx, height - S(55), (180, 150, 225), (200, 180, 255), 7)
        # Fireflies with glowing aura
        import random
        rng = random.Random(hash("fireflies") + frame // 3)
        for _ in range(14):
            fx = rng.randint(S(20), width - S(20))
            fy = rng.randint(height // 4, height - S(45))
            # Glow
            draw.ellipse([fx - S(6), fy - S(6), fx + S(6), fy + S(6)], fill=(60, 55, 20))
            draw.ellipse([fx - S(3), fy - S(3), fx + S(3), fy + S(3)], fill=(255, 255, 100))

    elif scene_id == "fairy_water":
        _stars(6)
        _moon(width - SX(45), SY(25), 18)
        _ground(height - S(35), (12, 32, 15))
        # Fairy
        _fairy(cx, cy)
        # Flowers below being watered
        for fx_frac in [0.3, 0.5, 0.7]:
            fx = int(width * fx_frac)
            _flower(fx, height - S(50), (180, 200, 255), (220, 220, 255), 6)
        # Water drops
        for dy in range(0, S(20), S(6)):
            draw.ellipse([cx + S(6), cy + S(12) + dy, cx + S(9), cy + S(16) + dy],
                         fill=(150, 200, 255))

    elif scene_id == "dawn_close":
        # Gradient sky (dawn colors)
        for yy in range(height * 2 // 3):
            t = yy / (height * 2 // 3)
            r = int(15 + t * 80)
            g = int(10 + t * 50)
            b = int(45 - t * 20)
            draw.line([0, yy, width, yy], fill=(max(0, r), max(0, g), max(0, min(255, b))))
        # Horizon warm glow
        glow_y = height * 2 // 3 - S(10)
        draw.rectangle([0, glow_y, width, glow_y + S(15)], fill=(100, 60, 35))
        _ground(glow_y + S(15), (18, 35, 15))
        # Closed flowers (buds)
        for fx_frac in [0.15, 0.35, 0.55, 0.75]:
            fx = int(width * fx_frac)
            fy = height - S(30)
            draw.line([fx, fy, fx, fy + S(18)], fill=(30, 65, 30), width=W(2))
            draw.ellipse([fx - S(5), fy - S(8), fx + S(5), fy], fill=(120, 90, 150))

    elif scene_id == "owl_meets_kitten":
        _stars(10)
        _moon(width - SX(55), SY(30), 22)
        # Big tree
        _tree(cx - SX(20), height - S(25), 60, 30)
        # Branch
        draw.line([cx - SX(20), height - S(80), cx + SX(30), height - S(85)],
                  fill=(85, 60, 35), width=W(4))
        # Owl on branch
        _owl(cx + SX(10), height - S(95), 18)
        # Ground
        _ground(height - S(25), (28, 40, 28))
        # Kitten below
        _kitten(cx + SX(40), height - S(35), 12)

    elif scene_id == "owl_singing":
        _stars(8)
        _moon(SX(55), SY(30), 18)
        # Owl (bigger, center)
        _owl(cx, cy - S(5), 22)
        # Open beak (singing)
        draw.polygon([(cx - S(4), cy + S(8)), (cx + S(4), cy + S(8)), (cx, cy + S(14))],
                     fill=(210, 160, 50))
        # Music notes floating
        _music_notes(cx + S(28), cy - S(30), 4)
        # Musical waves
        for i in range(3):
            r = S(35 + i * 12)
            draw.arc([cx - r, cy - r, cx + r, cy + r], -60, 60,
                     fill=(180, 160, 220), width=W(1))

    elif scene_id == "lullaby_scene":
        _stars(12)
        # Layered mountains
        colors = [(30, 40, 60), (25, 35, 55), (32, 42, 58)]
        for i, (peaks, c) in enumerate(zip(
            [((0, 0.17, 0.33), -0.25), ((0.25, 0.5, 0.75), -0.3), ((0.6, 0.83, 1.0), -0.22)],
            colors
        )):
            fracs, peak_h = peaks
            for f in fracs:
                px = int(width * f)
                ph = int(height * abs(peak_h)) + S(i * 10)
                pw = SX(100)
                draw.polygon([(px - pw, height - S(22)), (px, height - S(22) - ph),
                              (px + pw, height - S(22))], fill=c)
        # Stream at bottom
        _ground(height - S(22), (15, 28, 45))
        for sx_off in range(0, width, S(30)):
            draw.arc([sx_off, height - S(20), sx_off + S(28), height - S(10)], 0, 180,
                     fill=(70, 110, 190), width=W(2))
        # Swaying grass
        for sx_off in range(0, width, S(35)):
            draw.arc([sx_off, height - S(22) - S(12), sx_off + S(8), height - S(22)], 200, 340,
                     fill=(40, 60, 30), width=W(1))

    elif scene_id == "kitten_asleep":
        _stars(8)
        _moon(width - SX(50), SY(30), 20)
        # Tree branch across screen
        draw.line([0, cy + S(12), width * 2 // 3, cy + S(8)], fill=(85, 65, 38), width=W(5))
        # Owl perched on branch
        _owl(width // 4, cy - S(5), 16)
        # Ground
        _ground(height - S(28), (28, 40, 28))
        # Sleeping kitten below
        _kitten(cx + SX(10), height - S(38), 13, sleeping=True)
        # Zzz
        for i, (zx, zy, zsz) in enumerate([(S(18), -S(12), 9), (S(28), -S(22), 11), (S(40), -S(34), 13)]):
            kx = cx + SX(10)
            ky = height - S(38)
            try:
                from PIL import ImageFont
                fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", S(zsz))
            except Exception:
                from PIL import ImageFont
                fnt = ImageFont.load_default()
            draw.text((kx + zx, ky + zy), "z", fill=(180, 180, 230), font=fnt)

    elif scene_id == "sheep_meadow":
        _stars(6)
        _moon(width - SX(45), SY(25), 18)
        _hills(height - S(25), (25, 50, 25), 3)
        _ground(height - S(25), (22, 45, 22))
        # Fence (longer, more detailed)
        for fx in range(S(25), width - S(25), S(55)):
            draw.rectangle([fx, height - S(52), fx + S(5), height - S(25)], fill=(130, 105, 75))
        draw.line([S(25), height - S(46), width - S(25), height - S(46)],
                  fill=(130, 105, 75), width=W(3))
        draw.line([S(25), height - S(35), width - S(25), height - S(35)],
                  fill=(130, 105, 75), width=W(2))
        # Sheep scattered
        for px_frac in [0.2, 0.45, 0.7]:
            _sheep(int(width * px_frac), height - S(58), 16)

    elif scene_id == "sheep_jumping":
        _stars(6)
        _hills(height - S(22), (22, 45, 22), 2)
        _ground(height - S(22), (22, 45, 22))
        # Fence center
        draw.rectangle([cx - S(4), height - S(55), cx + S(4), height - S(22)], fill=(130, 105, 75))
        draw.line([cx - S(45), height - S(46), cx + S(45), height - S(46)],
                  fill=(130, 105, 75), width=W(3))
        draw.line([cx - S(45), height - S(35), cx + S(45), height - S(35)],
                  fill=(130, 105, 75), width=W(2))
        # Jumping sheep (arc trajectory)
        sx, sy = cx + S(25), height - S(85)
        _sheep(sx, sy, 16)
        # Motion arc
        draw.arc([cx - S(20), height - S(95), cx + S(50), height - S(40)], 180, 360,
                 fill=(180, 180, 200), width=W(1))

    elif scene_id == "sheep_sleepy":
        _stars(5)
        _ground(height - S(28), (22, 45, 22))
        # Sleepy lamb center (bigger)
        lx, ly = cx, cy + S(10)
        _sheep(lx, ly, 20)
        # Half-closed eyes
        draw.arc([lx - S(8), ly - S(20) - S(4), lx - S(2), ly - S(20)], 0, 180,
                 fill=(60, 60, 60), width=W(2))
        # Numbers floating up
        for i, (nx_off, ny_off) in enumerate([(S(30), -S(30)), (S(48), -S(50)), (S(62), -S(70))]):
            try:
                from PIL import ImageFont
                fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", S(14 + i * 2))
            except Exception:
                from PIL import ImageFont
                fnt = ImageFont.load_default()
            draw.text((lx + nx_off, ly + ny_off), str(8 + i), fill=(180, 180, 230), font=fnt)

    elif scene_id == "sheep_tuck_in":
        _stars(8)
        _ground(height - S(22), (22, 40, 22))
        # Woolly blanket (bigger)
        lx, ly = cx, cy + S(15)
        draw.rounded_rectangle([lx - S(40), ly - S(5), lx + S(40), ly + S(25)],
                               radius=S(10), fill=(235, 235, 245))
        # Lamb head poking out
        draw.ellipse([lx - S(12), ly - S(20), lx + S(8), ly - S(2)], fill=(245, 245, 250))
        # Closed eyes
        draw.arc([lx - S(8), ly - S(14), lx - S(2), ly - S(9)], 0, 180,
                 fill=(60, 60, 60), width=W(2))
        draw.arc([lx + S(2), ly - S(14), lx + S(8), ly - S(9)], 0, 180,
                 fill=(60, 60, 60), width=W(2))
        # Gentle smile
        draw.arc([lx - S(4), ly - S(6), lx + S(4), ly - S(2)], 0, 180,
                 fill=(60, 60, 60), width=W(1))

    elif scene_id == "train_pillow":
        _stars(10)
        # Big fluffy pillow
        draw.rounded_rectangle([cx - S(80), cy - S(10), cx + S(80), cy + S(35)],
                               radius=S(18), fill=(210, 210, 235))
        draw.rounded_rectangle([cx - S(70), cy - S(5), cx + S(70), cy + S(28)],
                               radius=S(14), fill=(220, 220, 240))
        # Magical train on pillow edge
        _train(cx - S(30), cy - S(20), 45)
        # Sparkles around train
        for angle in range(0, 360, 40):
            px = cx + int(S(55) * math.cos(math.radians(angle)))
            py = cy - S(10) + int(S(30) * math.sin(math.radians(angle)))
            draw.ellipse([px - S(1), py - S(1), px + S(1), py + S(1)], fill=(200, 200, 255))

    elif scene_id == "candy_land":
        # Cotton candy clouds (colorful sky)
        for ox_frac, c in [(-0.18, (255, 180, 210)), (0, (190, 210, 255)), (0.18, (255, 210, 190))]:
            ccx = cx + int(width * ox_frac)
            _cloud(ccx, SY(35), 35)
            # Tint the cloud
            draw.ellipse([ccx - S(25), SY(25), ccx + S(25), SY(50)], fill=c)
        # Chocolate mountains
        draw.polygon([(0, height - S(18)), (SX(110), height - S(70)), (SX(220), height - S(18))],
                     fill=(90, 55, 30))
        draw.polygon([(SX(190), height - S(18)), (SX(310), height - S(85)), (SX(430), height - S(18))],
                     fill=(100, 60, 28))
        # Lemonade lake
        draw.ellipse([cx - S(70), height - S(35), cx + S(70), height - S(5)], fill=(255, 255, 140))
        # Train chugging through
        _train(cx - S(35), cy + S(12), 50)

    elif scene_id == "dream_station":
        _stars(8)
        # Platform
        draw.rectangle([0, height - S(30), width, height], fill=(65, 60, 75))
        draw.rectangle([0, height - S(33), width, height - S(30)], fill=(110, 100, 120))
        # Arched station roof
        draw.arc([SX(30), SY(15), width - SX(30), height - S(30)], 180, 360,
                 fill=(80, 70, 90), width=W(3))
        # Rainbow dragon (chain of colored circles)
        dx, dy = cx - SX(60), cy
        colors = [(255, 80, 80), (255, 160, 50), (255, 255, 80), (80, 200, 80), (80, 120, 255)]
        for i, c in enumerate(colors):
            draw.ellipse([dx + S(i * 14) - S(7), dy - S(7), dx + S(i * 14) + S(7), dy + S(7)], fill=c)
        # Dragon head
        draw.ellipse([dx - S(12), dy - S(10), dx + S(5), dy + S(10)], fill=(255, 80, 80))
        draw.ellipse([dx - S(8), dy - S(6), dx - S(4), dy - S(2)], fill=(255, 255, 200))
        # Dancing penguin
        px = cx + SX(60)
        draw.ellipse([px - S(10), cy - S(15), px + S(10), cy + S(15)], fill=(30, 30, 42))
        draw.ellipse([px - S(7), cy - S(5), px + S(7), cy + S(12)], fill=(235, 235, 245))
        draw.ellipse([px - S(2), cy - S(10), px + S(2), cy - S(6)], fill=(255, 255, 200))
        # Penguin flippers up (dancing)
        draw.line([px - S(10), cy, px - S(16), cy - S(10)], fill=(30, 30, 42), width=W(2))
        draw.line([px + S(10), cy, px + S(16), cy - S(10)], fill=(30, 30, 42), width=W(2))

    elif scene_id == "last_stop":
        _stars(12)
        _moon(SX(55), SY(35), 20)
        # Cozy bed
        draw.rounded_rectangle([cx - S(60), cy, cx + S(60), cy + S(30)], radius=S(10),
                               fill=(190, 170, 210))
        draw.rounded_rectangle([cx - S(65), cy + S(25), cx + S(65), cy + S(45)], radius=S(6),
                               fill=(210, 190, 225))
        # Pillow
        draw.rounded_rectangle([cx - S(35), cy - S(8), cx + S(12), cy + S(12)], radius=S(10),
                               fill=(235, 235, 250))
        # Child shape sleeping
        draw.ellipse([cx - S(20), cy - S(4), cx - S(5), cy + S(10)], fill=(180, 140, 110))
        # Train going away (small, in distance)
        _train(width - SX(80), cy - S(35), 25)

    elif scene_id == "forest_trees":
        _stars(6)
        _moon(width - SX(45), SY(28), 20)
        _hills(height - S(22), (18, 40, 18), 4)
        _ground(height - S(22), (18, 40, 18))
        # Row of trees with variety
        positions = [0.1, 0.25, 0.42, 0.6, 0.78, 0.92]
        for i, px_frac in enumerate(positions):
            tx = int(width * px_frac)
            if i % 2 == 0:
                _tree(tx, height - S(22), 50, 25)
            else:
                _pine_tree(tx, height - S(22), 55)

    elif scene_id == "forest_animals":
        _stars(5)
        _moon(width - SX(40), SY(20), 14)
        _ground(height - S(28), (18, 40, 18))
        _tree(SX(50), height - S(28), 40, 20)
        _tree(width - SX(60), height - S(28), 35, 18)
        # Rabbit
        rx, ry = SX(120), height - S(42)
        draw.ellipse([rx - S(8), ry - S(6), rx + S(8), ry + S(6)], fill=(185, 165, 145))
        draw.ellipse([rx - S(5), ry - S(12), rx, ry - S(3)], fill=(185, 165, 145))
        draw.ellipse([rx, ry - S(12), rx + S(5), ry - S(3)], fill=(185, 165, 145))
        # Ear insides
        draw.ellipse([rx - S(4), ry - S(11), rx - S(1), ry - S(5)], fill=(220, 180, 170))
        draw.ellipse([rx + S(1), ry - S(11), rx + S(4), ry - S(5)], fill=(220, 180, 170))
        # Deer
        dx, dy = SX(320), height - S(55)
        draw.ellipse([dx - S(16), dy - S(6), dx + S(16), dy + S(12)], fill=(170, 130, 75))
        draw.ellipse([dx - S(7), dy - S(20), dx + S(7), dy - S(4)], fill=(170, 130, 75))
        # Antlers
        draw.line([dx - S(3), dy - S(20), dx - S(10), dy - S(32)], fill=(130, 95, 55), width=W(2))
        draw.line([dx - S(10), dy - S(32), dx - S(15), dy - S(30)], fill=(130, 95, 55), width=W(1))
        draw.line([dx + S(3), dy - S(20), dx + S(10), dy - S(32)], fill=(130, 95, 55), width=W(2))
        draw.line([dx + S(10), dy - S(32), dx + S(15), dy - S(30)], fill=(130, 95, 55), width=W(1))
        # Legs
        for lx in [dx - S(8), dx + S(8)]:
            draw.line([lx, dy + S(10), lx, dy + S(22)], fill=(140, 100, 55), width=W(2))
        # Stream
        for sx_off in range(SX(180), SX(280), S(25)):
            draw.arc([sx_off, height - S(30), sx_off + S(28), height - S(16)], 0, 180,
                     fill=(65, 105, 185), width=W(2))

    elif scene_id == "cricket_firefly":
        _ground(height - S(22), (14, 32, 14))
        _stars(3)
        # Cricket (bigger)
        cx2, cy2 = SX(130), height - S(32)
        draw.ellipse([cx2 - S(8), cy2 - S(5), cx2 + S(8), cy2 + S(5)], fill=(55, 90, 35))
        draw.ellipse([cx2 - S(10), cy2 - S(3), cx2 - S(3), cy2 + S(3)], fill=(60, 95, 38))
        # Legs
        for lx, ly in [(cx2 - S(5), cy2 + S(5)), (cx2, cy2 + S(5)), (cx2 + S(5), cy2 + S(5))]:
            draw.line([lx, ly, lx + S(3), ly + S(8)], fill=(50, 80, 30), width=W(1))
            draw.line([lx, ly, lx - S(3), ly + S(8)], fill=(50, 80, 30), width=W(1))
        # Musical notes from cricket
        _music_notes(cx2 + S(12), cy2 - S(20), 3)
        # Fireflies (bigger glow)
        import random
        rng = random.Random(42 + frame // 3)
        for _ in range(10):
            fx = rng.randint(S(50), width - S(50))
            fy = rng.randint(S(20), height - S(30))
            # Glow aura
            draw.ellipse([fx - S(8), fy - S(8), fx + S(8), fy + S(8)], fill=(50, 45, 15))
            draw.ellipse([fx - S(4), fy - S(4), fx + S(4), fy + S(4)], fill=(200, 200, 80))
            draw.ellipse([fx - S(2), fy - S(2), fx + S(2), fy + S(2)], fill=(255, 255, 130))

    elif scene_id == "forest_night":
        _stars(18)
        _moon(cx, SY(35), 28)
        _hills(height - S(18), (14, 35, 14), 3)
        _ground(height - S(18), (14, 34, 14))
        _pine_tree(SX(70), height - S(18), 50)
        _tree(SX(180), height - S(18), 40, 22)
        _pine_tree(SX(320), height - S(18), 45)
        _tree(SX(420), height - S(18), 35, 20)
        # Peaceful moonlight glow
        moon_y = SY(35)
        for r in range(S(55), S(12), -S(5)):
            draw.ellipse([cx - r, moon_y - r // 2, cx + r, moon_y + r // 2],
                         outline=(50, 50, min(90, 35 + r)), width=W(1))

    else:
        # Default: starry sky with big moon
        _stars(25)
        _moon(cx, cy - SY(20), 35)


# ─── Icon Drawing Helpers ─────────────────────────────────

def draw_icon(draw, icon_type, cx, cy, size, color):
    """Draw a simple icon centered at (cx, cy) using PIL ImageDraw."""
    s = size // 2
    if icon_type == "mic":
        # Microphone shape
        draw.ellipse([cx - s//3, cy - s, cx + s//3, cy], fill=color)
        draw.rectangle([cx - s//3, cy - s//3, cx + s//3, cy + s//4], fill=color)
        draw.arc([cx - s//2, cy - s//4, cx + s//2, cy + s//2], 0, 180, fill=color, width=2)
        draw.line([cx, cy + s//2, cx, cy + s], fill=color, width=2)
    elif icon_type == "palette":
        # Paint palette
        draw.ellipse([cx - s, cy - s + 2, cx + s, cy + s - 2], outline=color, width=2)
        for dx, dy in [(-4, -3), (3, -5), (6, 0), (2, 4)]:
            draw.ellipse([cx + dx - 2, cy + dy - 2, cx + dx + 2, cy + dy + 2], fill=color)
    elif icon_type == "book":
        # Open book
        draw.rectangle([cx - s, cy - s + 3, cx - 2, cy + s - 3], outline=color, width=2)
        draw.rectangle([cx + 2, cy - s + 3, cx + s, cy + s - 3], outline=color, width=2)
        draw.line([cx, cy - s + 5, cx, cy + s - 5], fill=color, width=1)
    elif icon_type == "gear":
        # Simple gear
        draw.ellipse([cx - s//2, cy - s//2, cx + s//2, cy + s//2], outline=color, width=2)
        draw.ellipse([cx - s//4, cy - s//4, cx + s//4, cy + s//4], fill=color)
        for angle in range(0, 360, 45):
            import math
            ex = cx + int(s * 0.8 * math.cos(math.radians(angle)))
            ey = cy + int(s * 0.8 * math.sin(math.radians(angle)))
            draw.ellipse([ex - 2, ey - 2, ex + 2, ey + 2], fill=color)
    elif icon_type == "star":
        # Star shape
        import math
        pts = []
        for i in range(10):
            angle = math.radians(i * 36 - 90)
            r = s if i % 2 == 0 else s // 2
            pts.append((cx + int(r * math.cos(angle)), cy + int(r * math.sin(angle))))
        draw.polygon(pts, fill=color)
    elif icon_type == "scissors":
        # Simple scissors
        draw.ellipse([cx - s, cy - s//2, cx - 2, cy + 2], outline=color, width=2)
        draw.ellipse([cx - s, cy - 2, cx - 2, cy + s//2], outline=color, width=2)
        draw.line([cx - 2, cy - 1, cx + s, cy - s//2], fill=color, width=2)
        draw.line([cx - 2, cy + 1, cx + s, cy + s//2], fill=color, width=2)
    elif icon_type == "speaker":
        # Speaker / volume
        draw.rectangle([cx - s//2, cy - s//3, cx - s//6, cy + s//3], fill=color)
        draw.polygon([(cx - s//6, cy - s//3), (cx + s//3, cy - s*2//3),
                       (cx + s//3, cy + s*2//3), (cx - s//6, cy + s//3)], fill=color)
        draw.arc([cx + s//6, cy - s//2, cx + s, cy + s//2], -45, 45, fill=color, width=2)
    elif icon_type == "globe":
        # Globe / world
        draw.ellipse([cx - s, cy - s, cx + s, cy + s], outline=color, width=2)
        draw.arc([cx - s//2, cy - s, cx + s//2, cy + s], 0, 360, fill=color, width=1)
        draw.line([cx - s, cy, cx + s, cy], fill=color, width=1)
        draw.line([cx, cy - s, cx, cy + s], fill=color, width=1)
    elif icon_type == "brain":
        # Brain / AI
        draw.ellipse([cx - s + 2, cy - s + 2, cx + 2, cy + s - 2], outline=color, width=2)
        draw.ellipse([cx - 2, cy - s + 2, cx + s - 2, cy + s - 2], outline=color, width=2)
        draw.arc([cx - s//2, cy - s//3, cx + s//2, cy + s//3], 0, 360, fill=color, width=1)
    elif icon_type == "projector":
        # Projector
        draw.rounded_rectangle([cx - s, cy - s//3, cx + s//3, cy + s//3], radius=3, outline=color, width=2)
        draw.ellipse([cx + s//3, cy - s//4, cx + s, cy + s//4], outline=color, width=2)
        draw.line([cx + s//2, cy + s//3, cx + s, cy + s*2//3], fill=color, width=1)
        draw.line([cx - s//2, cy + s//3, cx - s, cy + s*2//3], fill=color, width=1)
    elif icon_type == "car_icon":
        # Simple car top-down
        draw.rounded_rectangle([cx - s, cy - s//2, cx + s, cy + s//2], radius=4, outline=color, width=2)
        draw.rounded_rectangle([cx - s//2, cy - s + 2, cx + s//2, cy + s - 2], radius=3, outline=color, width=1)
        draw.ellipse([cx - s - 2, cy - s//3, cx - s + 4, cy - s//3 + 6], fill=color)
        draw.ellipse([cx - s - 2, cy + s//3 - 6, cx - s + 4, cy + s//3], fill=color)
        draw.ellipse([cx + s - 4, cy - s//3, cx + s + 2, cy - s//3 + 6], fill=color)
        draw.ellipse([cx + s - 4, cy + s//3 - 6, cx + s + 2, cy + s//3], fill=color)
    elif icon_type == "wifi":
        # WiFi arcs
        draw.ellipse([cx - 2, cy + s//3, cx + 2, cy + s//3 + 4], fill=color)
        draw.arc([cx - s//3, cy, cx + s//3, cy + s], -135, -45, fill=color, width=2)
        draw.arc([cx - s*2//3, cy - s//3, cx + s*2//3, cy + s*2//3], -135, -45, fill=color, width=2)
        draw.arc([cx - s, cy - s*2//3, cx + s, cy + s//3], -135, -45, fill=color, width=2)
    elif icon_type == "moon":
        # Crescent moon
        draw.ellipse([cx - s, cy - s, cx + s, cy + s], fill=color)
        draw.ellipse([cx - s//3, cy - s, cx + s + s//3, cy + s], fill=(10, 10, 20))
