import os
import math
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GLUT import glutSolidSphere
from OpenGL.GLUT import GLUT_BITMAP_HELVETICA_18
from PIL import Image
import numpy as np
import sys
import time as pytime

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
BIG_BANG_DURATION = 4.0  # seconds
TRANSITION_DURATION = 5.0 # seconds

# --- Music Config ---
# (Make sure you have these files in your 'music' folder!)
BIG_BANG_MUSIC = "big_bang.ogg"     # A dramatic, explosive track
TRANSITION_MUSIC = "creation.ogg"   # A soothing, building track
SOLAR_SYSTEM_MUSIC = "ambient.ogg"  # A calm, looping space track

# ============= SCENE DATA =============
# Format: name: (radius, texture, orbit_radius, orbit_speed, self_spin, axial_tilt, color)
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

MOON_DATA = {
    "Moon": (1.2, "moon.jpg", 12.0, 13.0, 0.0)
}

SATURN_RING_TEXTURE = "saturn_rings.jpg" 
STARFIELD_TEXTURE = "HDR_rich_multi_nebulae_1.hdr"

# ============= HELPERS =============
def resource_path(filename):
    """Get the full path to a resource in the TEX_DIR."""
    return os.path.join(TEX_DIR, filename)

def music_path(filename):
    """Get the full path to a resource in the MUSIC_DIR."""
    return os.path.join(MUSIC_DIR, filename)

def load_texture(path, flip_y=True, is_hdr=False):
    """Load a texture, return GL texture id."""
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
                print(f"[WARN] 'imageio' not installed. Cannot load HDR: {path}. Trying PIL.")
                img = Image.open(full_path)
                img_data = np.array(img, dtype=np.uint8)
                is_hdr = False
            except Exception as e:
                print(f"[WARN] Could not load HDR {path} with imageio: {e}. Trying PIL.")
                img = Image.open(full_path)
                img_data = np.array(img, dtype=np.uint8)
                is_hdr = False
            
            if is_hdr:
                if img_data.dtype != np.float32:
                    img_data = img_data.astype(np.float32) / 255.0 # Normalize if not float
                
                if len(img_data.shape) == 2: # Grayscale
                    img_data = np.stack([img_data, img_data, img_data], axis=-1)
                
                height, width = img_data.shape[:2]
                channels = img_data.shape[2] if len(img_data.shape) > 2 else 3
                
                gl_format = GL_RGBA if channels == 4 else GL_RGB
                internal_format = GL_RGBA16F if channels == 4 else GL_RGB16F
                data_type = GL_FLOAT
                
                if flip_y:
                    img_data = np.flipud(img_data)
                img_data_bytes = img_data.tobytes()
        
        # Standard image loading (PIL)
        if not is_hdr:
            img = Image.open(full_path)
            if img.mode not in ["RGB", "RGBA"]:
                img = img.convert("RGBA") # Convert to RGBA for consistency
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
    """Draw a textured sphere with alpha."""
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
    """Draw orbit path around the sun"""
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
    """Draw 3D text label"""
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glColor4f(1.0, 1.0, 1.0, alpha)
    glRasterPos3f(x, y, z)
    for ch in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)

def generate_starfield_texture(width=2048, height=1024):
    """Generate procedural starfield"""
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
    """Generates a procedural radial glow texture (RGBA)."""
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
    """Generates a procedural nebula/cloud texture (RGBA)."""
    noise = np.random.rand(size, size) * 0.2
    for _ in range(4):
        scale = np.random.randint(16, 32)
        amplitude = np.random.uniform(0.2, 0.5)
        small_grid = np.random.rand(max(1, size // scale), max(1, size // scale))
        scaled_grid = np.kron(small_grid, np.ones((scale, scale)))
        
        # FIX: Pad or crop the scaled grid to ensure it's the correct size
        # This prevents the broadcasting error.
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
    """Draw a texture on a quad that always faces the camera."""
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

# ============= MAIN APP =============
class SolarSystemApp:
    def __init__(self, width, height):
        pygame.init()
        pygame.mixer.init() # <-- Initialize the mixer
        
        # --- NEW MUSIC CHANNEL SETUP ---
        pygame.mixer.set_num_channels(2) # We want two channels for crossfading
        self.music_channel_a = pygame.mixer.Channel(0)
        self.music_channel_b = pygame.mixer.Channel(1)
        self.active_channel = self.music_channel_a   # Start with A
        self.inactive_channel = self.music_channel_b # B is ready for next track
        self.music_tracks = {} # Dictionary to hold our loaded music
        # --- END NEW MUSIC SETUP ---
        
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

        # --- State Machine ---
        self.state = "BIG_BANG"
        self.state_timer = 0.0
        
        self.load_all_textures()
        self.load_all_music() # <-- Load music at start
        self.init_opengl()

    def load_all_textures(self):
        """Load all textures"""
        print("\n=== Loading Textures ===")
        for name, data in SOLAR_SYSTEM_DATA.items(): self.textures[name] = load_texture(data[1])
        for name, data in MOON_DATA.items(): self.textures[name] = load_texture(data[1])
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
        """Loads all music tracks into memory as Sound objects."""
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
        """Crossfades music using two channels."""
        print(f"[MUSIC] Crossfading to {state}")
        
        # 1. Find the sound object to play
        sound_to_play = self.music_tracks.get(state)
            
        if not sound_to_play:
            print(f"[MUSIC] No track loaded for state: {state}")
            # Stop all music if no track is found
            self.active_channel.fadeout(1500)
            self.inactive_channel.fadeout(1500)
            return
            
        try:
            # 2. Fade out the *currently active* channel (e.g., Channel A)
            self.active_channel.fadeout(2000) # 2-second fade-out
            
            # 3. Play and fade in the new track on the *inactive* channel (e.g., Channel B)
            self.inactive_channel.set_volume(0.7)
            self.inactive_channel.play(sound_to_play, loops=-1, fade_ms=2000) # 2-second fade-in

            # 4. Swap the channels. Channel B is now the active one.
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
        """Initialize OpenGL"""
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
        """Handle input"""
        for ev in pygame.event.get():
            if ev.type == QUIT or (ev.type == KEYDOWN and ev.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            
            if self.state != "SOLAR_SYSTEM" and ev.type == KEYDOWN and ev.key == K_SPACE:
                self.state = "SOLAR_SYSTEM"
                self.play_music_for_state("SOLAR_SYSTEM") # <-- Trigger music
                pygame.display.set_caption("Cinematic Solar System - Use Mouse to Rotate | 1-9: Planet Focus | 0: Free View")

            if self.state == "SOLAR_SYSTEM":
                if ev.type == KEYDOWN:
                    if K_1 <= ev.key <= K_9:
                        idx = ev.key - K_1
                        if idx < len(PLANET_ORDER):
                            self.camera_target = PLANET_ORDER[idx]
                            print(f"Tracking: {self.camera_target}")
                    elif ev.key == K_0:
                        self.camera_target = None
                        print("Free view")
                
                if ev.type == MOUSEBUTTONDOWN:
                    if ev.button == 1:
                        self.dragging = True
                        self.last_mouse = ev.pos
                    elif ev.button == 4: self.cam_dist += WHEEL_ZOOM_STEP
                    elif ev.button == 5: self.cam_dist -= WHEEL_ZOOM_STEP
                
                if ev.type == MOUSEBUTTONUP and ev.button == 1: self.dragging = False
                
                if ev.type == MOUSEMOTION and self.dragging:
                    x, y = ev.pos
                    lx, ly = self.last_mouse
                    self.vel_y += (x - lx) * MOUSE_SENSITIVITY
                    self.vel_x += (y - ly) * MOUSE_SENSITIVITY
                    self.last_mouse = (x, y)

    def update_camera(self, dt):
        """Update camera"""
        self.cam_rot_y += self.vel_y * dt * 60
        self.cam_rot_x += self.vel_x * dt * 60
        self.vel_x *= DRAG_DAMPING
        self.vel_y *= DRAG_DAMPING
        self.cam_rot_x = max(-89.0, min(89.0, self.cam_rot_x))

    def draw_starfield(self, alpha=1.0):
        """Draw background starfield"""
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
        """Render the new nebula expansion explosion."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -1000) # Fixed camera

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

    def render_scene(self, t, transition_progress=1.0):
        """Render the main solar system scene."""
        self.planet_positions.clear()
        for name, data in SOLAR_SYSTEM_DATA.items():
            if name == "Sun": self.planet_positions["Sun"] = (0, 0, 0); continue
            _, _, orbit_r, orbit_speed, _, _, _ = data
            angle = t * orbit_speed
            x = orbit_r * math.sin(angle)
            z = orbit_r * math.cos(angle)
            self.planet_positions[name] = (x, 0.0, z)
        
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
        glTranslatef(0.0, 0.0, self.cam_dist)
        glRotatef(self.cam_rot_x, 1, 0, 0)
        glRotatef(self.cam_rot_y, 0, 1, 0)
        
        glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 0.0, 0.0, 1.0])
        
        if self.camera_target and self.camera_target in self.planet_positions:
            px, py, pz = self.planet_positions[self.camera_target]
            glTranslatef(-px, -py, -pz)
        
        glPushMatrix()
        sun_data = SOLAR_SYSTEM_DATA["Sun"]
        radius, tex, _, _, spin, tilt, _ = sun_data
        glRotatef(tilt, 1, 0, 0)
        glRotatef((t * spin * 50.0) % 360.0, 0, 1, 0)
        
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        current_radius = radius * transition_progress
        draw_textured_sphere(current_radius, self.textures.get("Sun"), color=(2.0, 1.6, 1.2), alpha=transition_progress)
        
        if self.sun_glow_tex > 0:
            glDisable(GL_LIGHTING)
            draw_billboard(0, 0, 0, current_radius * 5.0, self.sun_glow_tex, color=(1.6, 0.8, 0.2), alpha=transition_progress)

        glEnable(GL_LIGHTING)
        glColor3f(1.0, 1.0, 1.0)
        glDisable(GL_BLEND)

        glPopMatrix()
        draw_label(0, sun_data[0] + 6, 0, "SUN", alpha=transition_progress)
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        for name, data in SOLAR_SYSTEM_DATA.items():
            if name == "Sun": continue
            
            radius, _, orbit_r, _, spin, tilt, color = data
            x, y, z = self.planet_positions[name]
            
            draw_orbit_path(orbit_r, alpha=transition_progress)
            
            glPushMatrix()
            glTranslatef(x, y, z)
            glRotatef(tilt, 1, 0, 0)
            glRotatef((t * spin * 50.0) % 360.0, 0, 1, 0)
            
            draw_textured_sphere(radius, self.textures.get(name), color, alpha=transition_progress)
            
            if name == "Earth":
                m_data = MOON_DATA["Moon"]
                m_rad, _, m_dist, m_speed, m_spin = m_data
                m_angle = t * m_speed
                mx, mz = m_dist * math.sin(m_angle), m_dist * math.cos(m_angle)
                
                glPushMatrix()
                glTranslatef(mx, 0, mz)
                glRotatef((t * m_spin * 50) % 360, 0, 1, 0)
                draw_textured_sphere(m_rad, self.textures.get("Moon"), (1,1,1), alpha=transition_progress)
                glPopMatrix()
            
            elif name == "Saturn":
                if self.ring_tex > 0:
                    glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, self.ring_tex)
                    glDisable(GL_LIGHTING); glColor4f(1.0, 1.0, 1.0, 0.8 * transition_progress)
                    
                    inner, outer, segments = radius + 3, radius + 14, 128
                    
                    glBegin(GL_TRIANGLE_STRIP)
                    for i in range(segments + 1):
                        angle = (i / segments) * 2.0 * math.pi
                        u = i / segments
                        xi, zi = inner * math.cos(angle), inner * math.sin(angle)
                        xo, zo = outer * math.cos(angle), outer * math.sin(angle)
                        glTexCoord2f(u, 0); glVertex3f(xi, 0, zi)
                        glTexCoord2f(u, 1); glVertex3f(xo, 0, zo)
                    glEnd()
                    
                    glEnable(GL_LIGHTING); glBindTexture(GL_TEXTURE_2D, 0); glDisable(GL_TEXTURE_2D)
            
            glPopMatrix()
            draw_label(x, y + radius + 3, z, name.upper(), alpha=transition_progress)
        glDisable(GL_BLEND)

    def run(self):
        """Main loop"""
        clock = pygame.time.Clock()
        last_time = pytime.time()
        
        # Start the initial music
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
                    self.play_music_for_state("TRANSITION") # <-- Trigger music
                    pygame.display.set_caption("Cinematic Solar System - Forming...")

            elif self.state == "TRANSITION":
                t = pygame.time.get_ticks() / 12000.0
                progress = min(1.0, self.state_timer / TRANSITION_DURATION)
                self.render_scene(t, transition_progress=progress)
                if self.state_timer > TRANSITION_DURATION:
                    self.state = "SOLAR_SYSTEM"
                    self.play_music_for_state("SOLAR_SYSTEM") # <-- Trigger music
                    pygame.display.set_caption("Cinematic Solar System - Use Mouse to Rotate | 1-9: Planet Focus | 0: Free View")

            elif self.state == "SOLAR_SYSTEM":
                self.update_camera(dt)
                t = pygame.time.get_ticks() / 12000.0
                self.render_scene(t)
            
            pygame.display.flip()
            clock.tick(FPS)

if __name__ == "__main__":
    if not os.path.isdir(TEX_DIR):
        print(f"ERROR: Create a folder named '{TEX_DIR}' in the same directory as this script and add the required texture files inside it.")
    elif not os.path.isdir(MUSIC_DIR):
         print(f"ERROR: Create a folder named '{MUSIC_DIR}' in the same directory as this script and add your music files inside it.")
    else:
        app = SolarSystemApp(WINDOW_WIDTH, WINDOW_HEIGHT)
        app.run()