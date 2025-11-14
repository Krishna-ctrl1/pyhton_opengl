import os
import math
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GLUT import glutSolidSphere
from OpenGL.GLUT import GLUT_BITMAP_HELVETICA_18, GLUT_BITMAP_HELVETICA_12
from PIL import Image
import numpy as np
import sys
import time as pytime
import random

# ============= CONFIG =============
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
FPS = 60
SPHERE_SLICES = 128
SPHERE_STACKS = 128

MOUSE_SENSITIVITY = 0.5
DRAG_DAMPING = 0.92
WHEEL_ZOOM_STEP = 80.0

TEX_DIR = "textures"
MUSIC_DIR = "music"

# --- Animation Config ---
BIG_BANG_DURATION = 4.0
TRANSITION_DURATION = 5.0
INFO_CARD_TRANSITION = 0.5

# --- Music Config ---
BIG_BANG_MUSIC = "big_bang.ogg"
TRANSITION_MUSIC = "creation.ogg"
SOLAR_SYSTEM_MUSIC = "ambient.ogg"

# --- Asteroid Belt Config ---
NUM_ASTEROIDS = 200
ASTEROID_BELT_INNER = 260
ASTEROID_BELT_OUTER = 310

# --- Satellite Config ---
NUM_SATELLITES = 3

# ============= SCENE DATA =============
SOLAR_SYSTEM_DATA = {
    "Sun":     (25.0,   "sun.jpg",       0,     0.0,   0.02,   7.25,   (1.0, 0.95, 0.7)),
    "Mercury": (2.0,    "mercury.jpg",   60,    4.7,   0.04,   0.03,   (0.8, 0.8, 0.8)),
    "Venus":   (4.0,    "venus.jpg",     100,   3.5,   -0.02,  177.3,  (1.0, 0.9, 0.5)),
    "Earth":   (4.2,    "earth.jpg",     150,   2.9,   1.0,    23.44,  (0.5, 0.7, 1.0)),
    "Mars":    (2.5,    "mars.jpg",      220,   2.4,   0.97,   25.19,  (1.0, 0.5, 0.3)),
    "Jupiter": (15.0,   "jupiter.jpg",   350,   1.3,   2.4,    3.13,   (0.95, 0.85, 0.7)),
    "Saturn":  (12.0,   "saturn.jpg",    520,   0.9,   2.2,    26.73,  (1.0, 0.95, 0.8)),
    "Uranus":  (8.0,    "uranus.jpg",    700,   0.6,   -1.4,   97.77,  (0.5, 0.8, 1.0)),
    "Neptune": (7.5,    "neptune.jpg",   900,   0.5,   1.5,    28.32,  (0.2, 0.4, 1.0)),
}

PLANET_ORDER = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]

# Planet information database
PLANET_INFO = {
    "Sun": {
        "type": "Star",
        "diameter": "1,391,000 km",
        "mass": "1.989 × 10³⁰ kg",
        "temperature": "5,778 K (surface)",
        "age": "4.6 billion years",
        "composition": "Hydrogen 73%, Helium 25%",
        "facts": [
            "The Sun contains 99.86% of the Solar System's mass",
            "Light from the Sun takes 8 minutes to reach Earth",
            "The Sun will become a red giant in 5 billion years",
            "Core temperature reaches 15 million degrees Celsius"
        ]
    },
    "Mercury": {
        "type": "Terrestrial Planet",
        "diameter": "4,879 km",
        "mass": "3.285 × 10²³ kg",
        "distance": "57.9 million km from Sun",
        "day_length": "59 Earth days",
        "year_length": "88 Earth days",
        "temperature": "-173°C to 427°C",
        "moons": "0",
        "facts": [
            "Smallest planet in our Solar System",
            "Has no atmosphere to retain heat",
            "Surface covered with impact craters",
            "Named after Roman messenger god"
        ]
    },
    "Venus": {
        "type": "Terrestrial Planet",
        "diameter": "12,104 km",
        "mass": "4.867 × 10²⁴ kg",
        "distance": "108.2 million km from Sun",
        "day_length": "243 Earth days",
        "year_length": "225 Earth days",
        "temperature": "462°C (hottest planet)",
        "moons": "0",
        "facts": [
            "Rotates backwards (retrograde rotation)",
            "Hottest planet due to greenhouse effect",
            "Thick atmosphere of carbon dioxide",
            "Named after Roman goddess of love"
        ]
    },
    "Earth": {
        "type": "Terrestrial Planet",
        "diameter": "12,742 km",
        "mass": "5.972 × 10²⁴ kg",
        "distance": "149.6 million km from Sun",
        "day_length": "24 hours",
        "year_length": "365.25 days",
        "temperature": "-88°C to 58°C",
        "moons": "1 (Moon)",
        "facts": [
            "Only known planet to support life",
            "71% of surface covered by water",
            "Atmosphere: 78% nitrogen, 21% oxygen",
            "Has a powerful magnetic field"
        ]
    },
    "Mars": {
        "type": "Terrestrial Planet",
        "diameter": "6,779 km",
        "mass": "6.39 × 10²³ kg",
        "distance": "227.9 million km from Sun",
        "day_length": "24.6 hours",
        "year_length": "687 Earth days",
        "temperature": "-87°C to -5°C",
        "moons": "2 (Phobos, Deimos)",
        "facts": [
            "Known as the Red Planet due to iron oxide",
            "Has the largest volcano: Olympus Mons",
            "Evidence of ancient water flows",
            "Thin atmosphere, mostly carbon dioxide"
        ]
    },
    "Jupiter": {
        "type": "Gas Giant",
        "diameter": "139,820 km",
        "mass": "1.898 × 10²⁷ kg",
        "distance": "778.5 million km from Sun",
        "day_length": "10 hours",
        "year_length": "12 Earth years",
        "temperature": "-108°C",
        "moons": "95+ known moons",
        "facts": [
            "Largest planet in Solar System",
            "Great Red Spot is a massive storm",
            "Has faint rings made of dust",
            "Named after king of Roman gods"
        ]
    },
    "Saturn": {
        "type": "Gas Giant",
        "diameter": "116,460 km",
        "mass": "5.683 × 10²⁶ kg",
        "distance": "1.43 billion km from Sun",
        "day_length": "10.7 hours",
        "year_length": "29 Earth years",
        "temperature": "-139°C",
        "moons": "146+ known moons",
        "facts": [
            "Famous for its spectacular ring system",
            "Rings are made of ice and rock particles",
            "Least dense planet (would float on water)",
            "Named after Roman god of agriculture"
        ]
    },
    "Uranus": {
        "type": "Ice Giant",
        "diameter": "50,724 km",
        "mass": "8.681 × 10²⁵ kg",
        "distance": "2.87 billion km from Sun",
        "day_length": "17.2 hours",
        "year_length": "84 Earth years",
        "temperature": "-197°C",
        "moons": "27 known moons",
        "facts": [
            "Rotates on its side (98° tilt)",
            "Coldest planetary atmosphere",
            "Has 13 faint rings",
            "Appears blue-green due to methane"
        ]
    },
    "Neptune": {
        "type": "Ice Giant",
        "diameter": "49,244 km",
        "mass": "1.024 × 10²⁶ kg",
        "distance": "4.5 billion km from Sun",
        "day_length": "16 hours",
        "year_length": "165 Earth years",
        "temperature": "-201°C",
        "moons": "14 known moons",
        "facts": [
            "Most distant planet from Sun",
            "Has the strongest winds in Solar System",
            "Dark blue color from methane",
            "Named after Roman god of the sea"
        ]
    }
}

MOON_DATA = {
    "Moon": (1.2, "moon.jpg", 12.0, 13.0, 0.0)
}

SATURN_RING_TEXTURE = "saturn_rings.jpg" 
STARFIELD_TEXTURE = "HDR_rich_multi_nebulae_1.hdr"

# ============= HELPERS =============
def resource_path(filename):
    return os.path.join(TEX_DIR, filename)

def music_path(filename):
    return os.path.join(MUSIC_DIR, filename)

def load_texture(path, flip_y=True, is_hdr=False):
    full_path = resource_path(path)
    if not os.path.exists(full_path):
        print(f"[WARN] Texture not found: {full_path}")
        return 0
    
    tex_id = 0
    try:
        if is_hdr:
            try:
                import imageio.v3 as iio
                img_data = iio.imread(full_path)
            except ImportError:
                img = Image.open(full_path)
                img_data = np.array(img, dtype=np.uint8)
                is_hdr = False
            except Exception as e:
                img = Image.open(full_path)
                img_data = np.array(img, dtype=np.uint8)
                is_hdr = False
            
            if is_hdr:
                if img_data.dtype != np.float32:
                    img_data = img_data.astype(np.float32) / 255.0
                
                if len(img_data.shape) == 2:
                    img_data = np.stack([img_data, img_data, img_data], axis=-1)
                
                height, width = img_data.shape[:2]
                channels = img_data.shape[2] if len(img_data.shape) > 2 else 3
                
                gl_format = GL_RGBA if channels == 4 else GL_RGB
                internal_format = GL_RGBA16F if channels == 4 else GL_RGB16F
                data_type = GL_FLOAT
                
                if flip_y:
                    img_data = np.flipud(img_data)
                img_data_bytes = img_data.tobytes()
        
        if not is_hdr:
            img = Image.open(full_path)
            if img.mode not in ["RGB", "RGBA"]:
                img = img.convert("RGBA")
            if flip_y:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            
            width, height = img.size
            gl_format = GL_RGBA if img.mode == "RGBA" else GL_RGB
            internal_format = GL_RGBA8 if img.mode == "RGBA" else GL_RGB8
            data_type = GL_UNSIGNED_BYTE
            img_data_bytes = img.tobytes()
        
        tex_id = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, gl_format, data_type, img_data_bytes)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)
        
        print(f"[OK] {path} ({width}x{height}, HDR:{is_hdr})")
        return tex_id
    
    except Exception as e:
        print(f"[ERROR] Loading {full_path}: {e}")
        if tex_id != 0:
            glDeleteTextures(1, [tex_id])
        return 0

def draw_textured_sphere(radius, tex_id, color=(1,1,1), alpha=1.0, slices=SPHERE_SLICES, stacks=SPHERE_STACKS):
    r, g, b = color
    glColor4f(r, g, b, alpha)
    
    if tex_id > 0:
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tex_id)
    
    quad = gluNewQuadric()
    gluQuadricTexture(quad, GL_TRUE if tex_id > 0 else GL_FALSE)
    gluQuadricNormals(quad, GLU_SMOOTH)
    gluSphere(quad, radius, slices, stacks)
    gluDeleteQuadric(quad)
    
    if tex_id > 0:
        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)

def draw_orbit_path(radius, num_segments=300, alpha=1.0):
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(0.5, 0.6, 1.0, 0.2 * alpha)
    glLineWidth(1.0)
    glBegin(GL_LINE_LOOP)
    for i in range(num_segments):
        angle = (i / num_segments) * 2.0 * math.pi
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        glVertex3f(x, 0.0, z)
    glEnd()
    glDisable(GL_BLEND)
    glEnable(GL_LIGHTING)

def draw_label(x, y, z, text, scale=1.0, alpha=1.0):
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glColor4f(1.0, 1.0, 1.0, alpha)
    glRasterPos3f(x, y, z)
    for ch in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)

def generate_starfield_texture(width=2048, height=1024):
    np.random.seed(123)
    texture = np.zeros((height, width, 3), dtype=np.uint8)
    
    for _ in range(8000):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        brightness = np.random.randint(150, 255)
        size = np.random.randint(1, 3)
        texture[max(0, y-size):min(height, y+size), max(0, x-size):min(width, x+size)] = [brightness, brightness, brightness]
    
    for _ in range(500):
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        r_val = np.random.randint(30, 120)
        g_val = np.random.randint(20, 100)
        b_val = np.random.randint(60, 180)
        
        for i in range(-30, 30):
            for j in range(-30, 30):
                if i*i + j*j < 900:
                    x = (cx + i) % width
                    y = (cy + j) % height
                    if np.random.random() > 0.7:
                        texture[y, x] = [r_val, g_val, b_val]
    
    return texture

def generate_glow_texture(size=256):
    texture = np.zeros((size, size, 4), dtype=np.uint8)
    center_x, center_y = size // 2, size // 2
    max_dist = math.sqrt(center_x**2 + center_y**2)

    for y in range(size):
        for x in range(size):
            dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            falloff = max(0, (1.0 - (dist / max_dist))) ** 2.5
            alpha = int(255 * falloff)
            
            if alpha > 0:
                texture[y, x] = [255, 255, 255, alpha]
    return texture

def generate_nebula_texture(size=256):
    noise = np.random.rand(size, size) * 0.2
    for _ in range(4):
        scale = np.random.randint(16, 32)
        amplitude = np.random.uniform(0.2, 0.5)
        small_grid = np.random.rand(max(1, size // scale), max(1, size // scale))
        scaled_grid = np.kron(small_grid, np.ones((scale, scale)))
        
        temp_grid = np.zeros((size, size))
        h, w = min(size, scaled_grid.shape[0]), min(size, scaled_grid.shape[1])
        temp_grid[:h, :w] = scaled_grid[:h, :w]
        
        noise += temp_grid * amplitude

    noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))

    texture = np.zeros((size, size, 4), dtype=np.uint8)
    center_x, center_y = size // 2, size // 2
    max_dist = math.sqrt(center_x**2 + center_y**2)

    for y in range(size):
        for x in range(size):
            dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            falloff = max(0, (1.0 - (dist / max_dist))) ** 2.0
            alpha = int(255 * falloff * noise[y, x]**2)
            val = noise[y, x]
            if val < 0.3: r, g, b = 40, 20, 60
            elif val < 0.6: r, g, b = 220, 80, 30
            else: r, g, b = 255, 230, 180
            texture[y, x] = [r, g, b, alpha]
    return texture

def draw_billboard(cx, cy, cz, scale, tex_id, color=(1.0, 1.0, 1.0), alpha=1.0):
    modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
    cam_right = [modelview[0][0], modelview[1][0], modelview[2][0]]
    cam_up = [modelview[0][1], modelview[1][1], modelview[2][1]]

    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE)
    glDepthMask(GL_FALSE)

    r, g, b = color
    glColor4f(r, g, b, alpha)

    glBegin(GL_QUADS)
    v1_x=cx+(-cam_right[0]-cam_up[0])*scale; v1_y=cy+(-cam_right[1]-cam_up[1])*scale; v1_z=cz+(-cam_right[2]-cam_up[2])*scale
    glTexCoord2f(0,0); glVertex3f(v1_x,v1_y,v1_z)
    v2_x=cx+(cam_right[0]-cam_up[0])*scale; v2_y=cy+(cam_right[1]-cam_up[1])*scale; v2_z=cz+(cam_right[2]-cam_up[2])*scale
    glTexCoord2f(1,0); glVertex3f(v2_x,v2_y,v2_z)
    v3_x=cx+(cam_right[0]+cam_up[0])*scale; v3_y=cy+(cam_right[1]+cam_up[1])*scale; v3_z=cz+(cam_right[2]+cam_up[2])*scale
    glTexCoord2f(1,1); glVertex3f(v3_x,v3_y,v3_z)
    v4_x=cx+(-cam_right[0]+cam_up[0])*scale; v4_y=cy+(-cam_right[1]+cam_up[1])*scale; v4_z=cz+(-cam_right[2]+cam_up[2])*scale
    glTexCoord2f(0,1); glVertex3f(v4_x,v4_y,v4_z)
    glEnd()

    glDepthMask(GL_TRUE)
    glDisable(GL_BLEND)
    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)

def draw_text_2d(x, y, text, font=GLUT_BITMAP_HELVETICA_18, color=(1,1,1)):
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glColor3f(*color)
    glRasterPos2f(x, y)
    for ch in text:
        glutBitmapCharacter(font, ord(ch))
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def draw_gradient_rect_2d(x, y, width, height, color_top, color_bottom, alpha=1.0):
    """Draw a vertical gradient rectangle"""
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    glBegin(GL_QUADS)
    glColor4f(color_bottom[0], color_bottom[1], color_bottom[2], alpha)
    glVertex2f(x, y)
    glVertex2f(x + width, y)
    glColor4f(color_top[0], color_top[1], color_top[2], alpha)
    glVertex2f(x + width, y + height)
    glVertex2f(x, y + height)
    glEnd()
    
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def draw_rounded_rect_2d(x, y, width, height, color, alpha=1.0):
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    glColor4f(color[0], color[1], color[2], alpha)
    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x + width, y)
    glVertex2f(x + width, y + height)
    glVertex2f(x, y + height)
    glEnd()
    
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

# ============= MAIN APP =============
class SolarSystemApp:
    def __init__(self, width, height):
        pygame.init()
        pygame.mixer.init()
        
        pygame.mixer.set_num_channels(2)
        self.music_channel_a = pygame.mixer.Channel(0)
        self.music_channel_b = pygame.mixer.Channel(1)
        self.active_channel = self.music_channel_a
        self.inactive_channel = self.music_channel_b
        self.music_tracks = {}
        
        self.screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        glutInit()
        pygame.display.set_caption("Cinematic Solar System - The Beginning")
        
        self.width, self.height = width, height
        self.fov = 45.0
        self.cam_dist = -1500.0
        self.cam_rot_x = -30.0
        self.cam_rot_y = 0.0
        self.dragging = False
        self.last_mouse = None
        self.vel_x, self.vel_y = 0.0, 0.0
        
        self.camera_target = None
        self.planet_positions = {}
        self.textures = {}

        self.info_card_active = False
        self.info_card_planet = None
        self.info_card_progress = 0.0
        self.info_card_target_progress = 0.0

        self.state = "BIG_BANG"
        self.state_timer = 0.0
        
        # Initialize asteroids
        self.asteroids = self.generate_asteroids()
        
        # Initialize satellites
        self.satellites = self.generate_satellites()
        
        self.load_all_textures()
        self.load_all_music()
        self.init_opengl()

    def generate_asteroids(self):
        """Generate random asteroids in the belt between Mars and Jupiter"""
        random.seed(42)
        asteroids = []
        for i in range(NUM_ASTEROIDS):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(ASTEROID_BELT_INNER, ASTEROID_BELT_OUTER)
            size = random.uniform(0.3, 1.2)
            speed = random.uniform(0.5, 1.5)
            offset_y = random.uniform(-5, 5)
            rotation_speed = random.uniform(0.5, 2.0)
            color_var = random.uniform(0.8, 1.2)
            asteroids.append({
                'angle': angle,
                'radius': radius,
                'size': size,
                'speed': speed,
                'offset_y': offset_y,
                'rotation_speed': rotation_speed,
                'color_var': color_var
            })
        return asteroids

    def generate_satellites(self):
        """Generate satellites orbiting Earth"""
        random.seed(100)
        satellites = []
        for i in range(NUM_SATELLITES):
            angle = random.uniform(0, 2 * math.pi)
            distance = 10.0 + i * 3.0
            speed = 8.0 + random.uniform(-2, 2)
            inclination = random.uniform(-15, 15)
            satellites.append({
                'angle': angle,
                'distance': distance,
                'speed': speed,
                'inclination': inclination
            })
        return satellites

    def load_all_textures(self):
        print("\n=== Loading Textures ===")
        for name, data in SOLAR_SYSTEM_DATA.items(): 
            self.textures[name] = load_texture(data[1])
        for name, data in MOON_DATA.items(): 
            self.textures[name] = load_texture(data[1])
        self.ring_tex = load_texture(SATURN_RING_TEXTURE)
        
        self.starfield_tex = load_texture(STARFIELD_TEXTURE, flip_y=False, is_hdr=True)
        if self.starfield_tex == 0:
            print("[INFO] Generating procedural starfield...")
            self.starfield_tex = self.create_procedural_starfield()

        print("[INFO] Generating procedural glow texture...")
        self.sun_glow_tex = self.create_procedural_glow_texture()
        print("[INFO] Generating procedural nebula texture...")
        self.nebula_tex = self.create_procedural_nebula_texture()
        print("=== Ready ===\n")

    def load_all_music(self):
        print("=== Loading Music ===")
        tracks_to_load = {
            "BIG_BANG": BIG_BANG_MUSIC,
            "TRANSITION": TRANSITION_MUSIC,
            "SOLAR_SYSTEM": SOLAR_SYSTEM_MUSIC
        }
        
        for name, filename in tracks_to_load.items():
            full_path = music_path(filename)
            if not os.path.exists(full_path):
                print(f"[WARN] Music file not found: {full_path}")
                continue
            
            try:
                sound = pygame.mixer.Sound(full_path)
                self.music_tracks[name] = sound
                print(f"[OK] Loaded music: {filename}")
            except pygame.error as e:
                print(f"[ERROR] Loading music {full_path}: {e}")
        print("=== Music Ready ===\n")

    def create_procedural_starfield(self):
        texture_data = generate_starfield_texture(2048, 1024)
        tex_id = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, 2048, 1024, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data.tobytes())
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)
        return tex_id

    def create_procedural_glow_texture(self):
        texture_data = generate_glow_texture(256)
        tex_id = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 256, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data.tobytes())
        glBindTexture(GL_TEXTURE_2D, 0)
        return tex_id
        
    def create_procedural_nebula_texture(self):
        texture_data = generate_nebula_texture(256)
        tex_id = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 256, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data.tobytes())
        glBindTexture(GL_TEXTURE_2D, 0)
        return tex_id

    def play_music_for_state(self, state):
        print(f"[MUSIC] Crossfading to {state}")
        sound_to_play = self.music_tracks.get(state)
        if not sound_to_play:
            print(f"[MUSIC] No track loaded for state: {state}")
            self.active_channel.fadeout(1500)
            self.inactive_channel.fadeout(1500)
            return
        try:
            self.active_channel.fadeout(2000)
            self.inactive_channel.set_volume(0.7)
            self.inactive_channel.play(sound_to_play, loops=-1, fade_ms=2000)
            temp = self.active_channel
            self.active_channel = self.inactive_channel
            self.inactive_channel = temp
            print(f"[MUSIC] Playing: {state}")
        except pygame.error as e:
            print(f"[ERROR] Could not play music for {state}: {e}")

    def set_projection(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, float(self.width)/self.height, 0.1, 50000.0)
        glMatrixMode(GL_MODELVIEW)

    def init_opengl(self):
        glViewport(0, 0, self.width, self.height)
        self.set_projection()
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_NORMALIZE)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 0.0, 0.0, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.35, 1.0])
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.25, 0.25, 0.3, 1.0])
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glMaterialf(GL_FRONT, GL_SHININESS, 32)

    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == QUIT or (ev.type == KEYDOWN and ev.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            
            if self.state != "SOLAR_SYSTEM" and ev.type == KEYDOWN and ev.key == K_SPACE:
                self.state = "SOLAR_SYSTEM"
                self.play_music_for_state("SOLAR_SYSTEM")
                pygame.display.set_caption("Solar System - 1-9: Planet Info | 0: Free View")

            if self.state == "SOLAR_SYSTEM":
                if ev.type == KEYDOWN:
                    if K_1 <= ev.key <= K_9:
                        idx = ev.key - K_1
                        if idx < len(PLANET_ORDER):
                            planet_name = PLANET_ORDER[idx]
                            if self.info_card_active and self.info_card_planet == planet_name:
                                self.info_card_target_progress = 0.0
                                self.info_card_active = False
                                print(f"Closing info card: {planet_name}")
                            else:
                                self.info_card_planet = planet_name
                                self.info_card_target_progress = 1.0
                                self.info_card_active = True
                                self.camera_target = planet_name
                                print(f"Opening info card: {planet_name}")
                    elif ev.key == K_0:
                        self.info_card_target_progress = 0.0
                        self.info_card_active = False
                        self.camera_target = None
                        print("Free view")
                
                if not self.info_card_active:
                    if ev.type == MOUSEBUTTONDOWN:
                        if ev.button == 1:
                            self.dragging = True
                            self.last_mouse = ev.pos
                        elif ev.button == 4: 
                            self.cam_dist += WHEEL_ZOOM_STEP
                        elif ev.button == 5: 
                            self.cam_dist -= WHEEL_ZOOM_STEP
                    if ev.type == MOUSEBUTTONUP and ev.button == 1: 
                        self.dragging = False
                    if ev.type == MOUSEMOTION and self.dragging:
                        x, y = ev.pos
                        lx, ly = self.last_mouse
                        self.vel_y += (x - lx) * MOUSE_SENSITIVITY
                        self.vel_x += (y - ly) * MOUSE_SENSITIVITY
                        self.last_mouse = (x, y)

    def update_camera(self, dt):
        self.cam_rot_y += self.vel_y * dt * 60
        self.cam_rot_x += self.vel_x * dt * 60
        self.vel_x *= DRAG_DAMPING
        self.vel_y *= DRAG_DAMPING
        self.cam_rot_x = max(-89.0, min(89.0, self.cam_rot_x))

    def update_info_card(self, dt):
        if self.info_card_progress < self.info_card_target_progress:
            self.info_card_progress = min(self.info_card_target_progress, 
                                         self.info_card_progress + dt / INFO_CARD_TRANSITION)
        elif self.info_card_progress > self.info_card_target_progress:
            self.info_card_progress = max(self.info_card_target_progress, 
                                         self.info_card_progress - dt / INFO_CARD_TRANSITION)

    def draw_starfield(self, alpha=1.0):
        glPushMatrix()
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glCullFace(GL_FRONT)
        draw_textured_sphere(30000, self.starfield_tex, color=(1,1,1), alpha=alpha, slices=64, stacks=32)
        glCullFace(GL_BACK)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glPopMatrix()

    def render_big_bang(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -1000)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glDepthMask(GL_FALSE)
        progress = self.state_timer / BIG_BANG_DURATION
        if progress < 1.0:
            alpha = (1.0 - progress) ** 1.5
            num_layers = 4
            for i in range(num_layers):
                glPushMatrix()
                rotation_speed = 20 + i * 15
                rotation_axis = [(i*0.5)%1, (i*0.2)%1, 1]
                glRotatef(self.state_timer * rotation_speed, *rotation_axis)
                scale = (progress ** 1.5) * (1000 + i * 200)
                draw_billboard(0, 0, 0, scale, self.nebula_tex, color=(1.5, 1.5, 1.5), alpha=alpha * 0.7)
                glPopMatrix()
        if progress < 0.3:
            flash_alpha = (1.0 - (progress / 0.3))
            flash_scale = 50 + (1000 * (progress / 0.3))
            if self.sun_glow_tex > 0:
                draw_billboard(0, 0, 0, flash_scale, self.sun_glow_tex, color=(2.5, 2.5, 2.2), alpha=flash_alpha)
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

    def render_minimap_scene(self, t, viewport_rect):
        vx, vy, vw, vh = viewport_rect
        glViewport(vx, vy, vw, vh)
        glScissor(vx, vy, vw, vh)
        glEnable(GL_SCISSOR_TEST)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(vw)/vh, 0.1, 50000.0)
        glMatrixMode(GL_MODELVIEW)
        glClear(GL_DEPTH_BUFFER_BIT)
        glPushMatrix()
        glLoadIdentity()
        glRotatef(self.cam_rot_x, 1, 0, 0)
        glRotatef(self.cam_rot_y, 0, 1, 0)
        if self.starfield_tex > 0:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            self.draw_starfield(alpha=0.5)
            glDisable(GL_BLEND)
        glPopMatrix()
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -2000.0)
        glRotatef(-20.0, 1, 0, 0)
        glRotatef(self.cam_rot_y * 0.3, 0, 1, 0)
        glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 0.0, 0.0, 1.0])
        glPushMatrix()
        sun_data = SOLAR_SYSTEM_DATA["Sun"]
        radius = sun_data[0]
        glDisable(GL_LIGHTING)
        draw_textured_sphere(radius, self.textures.get("Sun"), color=(2.0, 1.6, 1.2))
        glEnable(GL_LIGHTING)
        glPopMatrix()
        for name, data in SOLAR_SYSTEM_DATA.items():
            if name == "Sun": continue
            radius, _, orbit_r, _, spin, tilt, color = data
            x, y, z = self.planet_positions.get(name, (0, 0, 0))
            glPushMatrix()
            glTranslatef(x, y, z)
            draw_textured_sphere(radius, self.textures.get(name), color)
            glPopMatrix()
        glDisable(GL_SCISSOR_TEST)
        glViewport(0, 0, self.width, self.height)
        self.set_projection()

    def draw_info_card(self, planet_name, progress):
        if planet_name not in PLANET_INFO:
            return
        info = PLANET_INFO[planet_name]
        card_width = 700
        card_height = self.height - 100
        card_x = self.width - card_width - 30 + int((1.0 - progress) * (card_width + 100))
        card_y = 50
        bg_alpha = 0.92 * progress

        # NEW: Replaced the solid rectangle with a gradient for a nicer aesthetic
        color_top = (0.1, 0.1, 0.3) # Dark blue
        color_bottom = (0.05, 0.05, 0.15) # Darker blue/purple
        draw_gradient_rect_2d(card_x, card_y, card_width, card_height, color_top, color_bottom, bg_alpha)
        # OLD line: draw_rounded_rect_2d(card_x, card_y, card_width, card_height, (0.05, 0.05, 0.15), bg_alpha)
        
        # This is the border
        draw_rounded_rect_2d(card_x - 2, card_y - 2, card_width + 4, card_height + 4, (0.3, 0.5, 0.8), 0.3 * progress)
        
        current_y = self.height - card_y - 40
        line_height = 25
        small_line_height = 20
        title_color = SOLAR_SYSTEM_DATA[planet_name][6]
        draw_text_2d(card_x + 30, current_y, planet_name.upper(), GLUT_BITMAP_HELVETICA_18, title_color)
        current_y -= 35
        draw_text_2d(card_x + 30, current_y, info["type"], GLUT_BITMAP_HELVETICA_12, (0.7, 0.7, 0.7))
        current_y -= 40
        draw_rounded_rect_2d(card_x + 30, current_y, card_width - 60, 2, (0.3, 0.5, 0.8), 0.5 * progress)
        current_y -= 30
        draw_text_2d(card_x + 30, current_y, "PHYSICAL CHARACTERISTICS", GLUT_BITMAP_HELVETICA_12, (0.5, 0.7, 1.0))
        current_y -= 25
        characteristics = [
            ("Diameter:", info.get("diameter", "N/A")),
            ("Mass:", info.get("mass", "N/A")),
            ("Temperature:", info.get("temperature", "N/A")),
        ]
        if planet_name != "Sun":
            characteristics.extend([
                ("Distance from Sun:", info.get("distance", "N/A")),
                ("Day Length:", info.get("day_length", "N/A")),
                ("Year Length:", info.get("year_length", "N/A")),
                ("Moons:", info.get("moons", "N/A")),
            ])
        else:
            characteristics.extend([
                ("Age:", info.get("age", "N/A")),
                ("Composition:", info.get("composition", "N/A")),
            ])
        for label, value in characteristics:
            draw_text_2d(card_x + 40, current_y, label, GLUT_BITMAP_HELVETICA_12, (0.6, 0.6, 0.6))
            draw_text_2d(card_x + 200, current_y, value, GLUT_BITMAP_HELVETICA_12, (1.0, 1.0, 1.0))
            current_y -= small_line_height
        current_y -= 20
        draw_rounded_rect_2d(card_x + 30, current_y, card_width - 60, 2, (0.3, 0.5, 0.8), 0.5 * progress)
        current_y -= 30
        draw_text_2d(card_x + 30, current_y, "INTERESTING FACTS", GLUT_BITMAP_HELVETICA_12, (0.5, 0.7, 1.0))
        current_y -= 25
        for i, fact in enumerate(info["facts"], 1):
            words = fact.split()
            line = f"{i}. "
            max_width = 60
            for word in words:
                if len(line) + len(word) + 1 <= max_width:
                    line += word + " "
                else:
                    draw_text_2d(card_x + 40, current_y, line.strip(), GLUT_BITMAP_HELVETICA_12, (0.9, 0.9, 0.9))
                    current_y -= small_line_height
                    line = "   " + word + " "
            if line.strip():
                draw_text_2d(card_x + 40, current_y, line.strip(), GLUT_BITMAP_HELVETICA_12, (0.9, 0.9, 0.9))
                current_y -= small_line_height
            current_y -= 10
        current_y = card_y + 30
        draw_text_2d(card_x + 30, current_y, f"Press {PLANET_ORDER.index(planet_name) + 1} again to close | Press 0 for free view", 
                    GLUT_BITMAP_HELVETICA_12, (0.5, 0.5, 0.5))

    def render_scene(self, t, transition_progress=1.0):
        self.planet_positions.clear()
        for name, data in SOLAR_SYSTEM_DATA.items():
            if name == "Sun": 
                self.planet_positions["Sun"] = (0, 0, 0)
                continue
            _, _, orbit_r, orbit_speed, _, _, _ = data
            angle = t * orbit_speed
            x = orbit_r * math.sin(angle)
            z = orbit_r * math.cos(angle)
            self.planet_positions[name] = (x, 0.0, z)
        
        if self.info_card_progress > 0.01:
            minimap_width = 400
            minimap_height = 300
            minimap_x = 20
            minimap_y = 20
            minimap_scale = 0.5 + 0.5 * self.info_card_progress
            final_width = int(minimap_width * minimap_scale)
            final_height = int(minimap_height * minimap_scale)
            draw_rounded_rect_2d(minimap_x - 3, minimap_y - 3, final_width + 6, final_height + 6, 
                               (0.3, 0.5, 0.8), 0.5 * self.info_card_progress)
            draw_rounded_rect_2d(minimap_x, minimap_y, final_width, final_height, 
                               (0.0, 0.0, 0.0), 0.8 * self.info_card_progress)
            self.render_minimap_scene(t, (minimap_x, minimap_y, final_width, final_height))
            draw_text_2d(minimap_x + 10, minimap_y + final_height - 20, "SOLAR SYSTEM", 
                        GLUT_BITMAP_HELVETICA_12, (0.5, 0.7, 1.0))
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()
        glLoadIdentity()
        glRotatef(self.cam_rot_x, 1, 0, 0)
        glRotatef(self.cam_rot_y, 0, 1, 0)
        if self.starfield_tex > 0:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            self.draw_starfield(alpha=transition_progress)
            glDisable(GL_BLEND)
        glPopMatrix()
        glLoadIdentity()
        if self.info_card_progress > 0.01 and self.info_card_planet:
            focus_dist = -300.0 - (self.info_card_progress * 200.0)
            glTranslatef(0.0, 0.0, focus_dist)
            glRotatef(-20.0 + self.cam_rot_x * 0.3, 1, 0, 0)
            glRotatef(self.cam_rot_y * 0.3, 0, 1, 0)
            if self.info_card_planet in self.planet_positions:
                px, py, pz = self.planet_positions[self.info_card_planet]
                glTranslatef(-px, -py, -pz)
        else:
            glTranslatef(0.0, 0.0, self.cam_dist)
            glRotatef(self.cam_rot_x, 1, 0, 0)
            glRotatef(self.cam_rot_y, 0, 1, 0)
            if self.camera_target and self.camera_target in self.planet_positions:
                px, py, pz = self.planet_positions[self.camera_target]
                glTranslatef(-px, -py, -pz)
        glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 0.0, 0.0, 1.0])
        glPushMatrix()
        sun_data = SOLAR_SYSTEM_DATA["Sun"]
        radius, tex, _, _, spin, tilt, _ = sun_data
        glRotatef(tilt, 1, 0, 0)
        glRotatef((t * spin * 50.0) % 360.0, 0, 1, 0)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        current_radius = radius * transition_progress
        if self.info_card_planet == "Sun" and self.info_card_progress > 0.01:
            current_radius *= (1.0 + self.info_card_progress * 2.0)
        draw_textured_sphere(current_radius, self.textures.get("Sun"), color=(2.0, 1.6, 1.2), alpha=transition_progress)
        if self.sun_glow_tex > 0:
            glDisable(GL_LIGHTING)
            draw_billboard(0, 0, 0, current_radius * 5.0, self.sun_glow_tex, color=(1.6, 0.8, 0.2), alpha=transition_progress)
        glEnable(GL_LIGHTING)
        glColor3f(1.0, 1.0, 1.0)
        glDisable(GL_BLEND)
        glPopMatrix()
        if self.info_card_progress < 0.5:
            draw_label(0, sun_data[0] + 6, 0, "SUN", alpha=transition_progress * (1.0 - self.info_card_progress * 2))
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        for name, data in SOLAR_SYSTEM_DATA.items():
            if name == "Sun": continue
            radius, _, orbit_r, _, spin, tilt, color = data
            x, y, z = self.planet_positions[name]
            orbit_alpha = transition_progress * (1.0 - self.info_card_progress * 0.7)
            draw_orbit_path(orbit_r, alpha=orbit_alpha)
            glPushMatrix()
            glTranslatef(x, y, z)
            glRotatef(tilt, 1, 0, 0)
            glRotatef((t * spin * 50.0) % 360.0, 0, 1, 0)
            planet_scale = 1.0
            if self.info_card_planet == name and self.info_card_progress > 0.01:
                planet_scale = 1.0 + self.info_card_progress * 3.0
            draw_textured_sphere(radius * planet_scale, self.textures.get(name), color, alpha=transition_progress)
            if name == "Earth":
                
                # Draw Satellites orbiting Earth
                for sat in self.satellites:
                    sat_angle = t * sat['speed'] + sat['angle']
                    dist = sat['distance'] / planet_scale
                    sx = dist * math.sin(sat_angle)
                    sz = dist * math.cos(sat_angle)
                    sy = sx * math.sin(math.radians(sat['inclination'])) # cheap inclination
                    glPushMatrix()
                    glTranslatef(sx, sy, sz)
                    draw_textured_sphere(0.15, 0, color=(0.8, 0.8, 0.9), slices=8, stacks=8)
                    glPopMatrix()
                
                # Draw Moon
                m_data = MOON_DATA["Moon"]
                m_rad, _, m_dist, m_speed, m_spin = m_data
                m_angle = t * m_speed
                mx, mz = m_dist * math.sin(m_angle), m_dist * math.cos(m_angle)
                glPushMatrix()
                glTranslatef(mx / planet_scale, 0, mz / planet_scale)
                glRotatef((t * m_spin * 50) % 360, 0, 1, 0)
                draw_textured_sphere(m_rad, self.textures.get("Moon"), (1,1,1), alpha=transition_progress)
                glPopMatrix()
            elif name == "Saturn":
                if self.ring_tex > 0:
                    # --- START FIX ---
                    glEnable(GL_TEXTURE_2D) # Corrected from GL_TEXTURE_D
                    # --- END FIX ---
                    glBindTexture(GL_TEXTURE_2D, self.ring_tex)
                    glDisable(GL_LIGHTING)
                    glColor4f(1.0, 1.0, 1.0, 0.8 * transition_progress)
                    inner, outer, segments = (radius + 3) * planet_scale, (radius + 14) * planet_scale, 128
                    glBegin(GL_TRIANGLE_STRIP)
                    for i in range(segments + 1):
                        angle = (i / segments) * 2.0 * math.pi
                        u = i / segments
                        xi, zi = inner * math.cos(angle), inner * math.sin(angle)
                        xo, zo = outer * math.cos(angle), outer * math.sin(angle)
                        glTexCoord2f(u, 0); glVertex3f(xi, 0, zi)
                        glTexCoord2f(u, 1); glVertex3f(xo, 0, zo)
                    glEnd()
                    glEnable(GL_LIGHTING)
                    glBindTexture(GL_TEXTURE_2D, 0)
                    glDisable(GL_TEXTURE_2D)
            glPopMatrix()
            if self.info_card_progress < 0.5:
                label_alpha = transition_progress * (1.0 - self.info_card_progress * 2)
                draw_label(x, y + radius + 3, z, name.upper(), alpha=label_alpha)
        
        # Draw Asteroid Belt
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        asteroid_alpha = transition_progress * (1.0 - self.info_card_progress)
        if asteroid_alpha > 0.01:
            for asteroid in self.asteroids:
                cv = asteroid['color_var']
                asteroid_color = (0.6 * cv, 0.55 * cv, 0.5 * cv)
                
                angle = t * asteroid['speed'] + asteroid['angle']
                x = asteroid['radius'] * math.sin(angle)
                z = asteroid['radius'] * math.cos(angle)
                y = asteroid['offset_y']
                
                glPushMatrix()
                glTranslatef(x, y, z)
                glRotatef(t * 100 * asteroid['rotation_speed'], 0.5, 1, 0)
                draw_textured_sphere(asteroid['size'], 0, color=asteroid_color, alpha=asteroid_alpha, slices=8, stacks=8)
                glPopMatrix()
        glDisable(GL_BLEND)
        
        glDisable(GL_BLEND) 
        if self.info_card_progress > 0.01 and self.info_card_planet:
            self.draw_info_card(self.info_card_planet, self.info_card_progress)
            
    def run(self):
        clock = pygame.time.Clock()
        last_time = pytime.time()
        self.play_music_for_state(self.state)
        while True:
            now = pytime.time()
            dt = now - last_time
            last_time = now
            self.handle_events()
            if self.state != "SOLAR_SYSTEM":
                self.state_timer += dt
            if self.state == "BIG_BANG":
                self.render_big_bang()
                if self.state_timer > BIG_BANG_DURATION:
                    self.state = "TRANSITION"
                    self.state_timer = 0.0
                    self.play_music_for_state("TRANSITION")
                    pygame.display.set_caption("Cinematic Solar System - Forming...")
            elif self.state == "TRANSITION":
                t = pygame.time.get_ticks() / 12000.0
                progress = min(1.0, self.state_timer / TRANSITION_DURATION)
                self.render_scene(t, transition_progress=progress)
                if self.state_timer > TRANSITION_DURATION:
                    self.state = "SOLAR_SYSTEM"
                    self.play_music_for_state("SOLAR_SYSTEM")
                    pygame.display.set_caption("Solar System - 1-9: Planet Info | 0: Free View")
            elif self.state == "SOLAR_SYSTEM":
                self.update_camera(dt)
                self.update_info_card(dt)
                t = pygame.time.get_ticks() / 12000.0
                self.render_scene(t)
            pygame.display.flip()
            clock.tick(FPS)

if __name__ == "__main__":
    if not os.path.isdir(TEX_DIR):
        print(f"ERROR: Create '{TEX_DIR}' folder with texture files.")
    elif not os.path.isdir(MUSIC_DIR):
         print(f"ERROR: Create '{MUSIC_DIR}' folder with music files.")
    else:
        app = SolarSystemApp(WINDOW_WIDTH, WINDOW_HEIGHT)
        app.run()