import os
import math
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
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

# ============= SCENE DATA =============
# Format: name: (radius, texture, orbit_radius, orbit_speed, self_spin, axial_tilt, color)
SOLAR_SYSTEM_DATA = {
    "Sun":     (25.0,  "sun.jpg",       0,    0.0,  0.02,   7.25,  (1.0, 0.95, 0.7)),
    "Mercury": (2.0,   "mercury.jpg",   60,   4.7,  0.04,   0.03,  (0.8, 0.8, 0.8)),
    "Venus":   (4.0,   "venus.jpg",     100,  3.5,  -0.02,  177.3, (1.0, 0.9, 0.5)),
    "Earth":   (4.2,   "earth.jpg",     150,  2.9,  1.0,    23.44, (0.5, 0.7, 1.0)),
    "Mars":    (2.5,   "mars.jpg",      220,  2.4,  0.97,   25.19, (1.0, 0.5, 0.3)),
    "Jupiter": (15.0,  "jupiter.jpg",   350,  1.3,  2.4,    3.13,  (0.95, 0.85, 0.7)),
    "Saturn":  (12.0,  "saturn.jpg",    520,  0.9,  2.2,    26.73, (1.0, 0.95, 0.8)),
    "Uranus":  (8.0,   "uranus.jpg",    700,  0.6,  -1.4,   97.77, (0.5, 0.8, 1.0)),
    "Neptune": (7.5,   "neptune.jpg",   900,  0.5,  1.5,    28.32, (0.2, 0.4, 1.0)),
}

PLANET_ORDER = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]

MOON_DATA = {
    "Moon": (1.2, "moon.jpg", 12.0, 13.0, 0.0)
}

SATURN_RING_TEXTURE = "saturn_rings.jpg"
SUN_GLOW_TEXTURE = "sun_glow.jpg"
STARFIELD_TEXTURE = "HDR_rich_blue_nebulae_1.hdr"

# ============= HELPERS =============
def resource_path(filename):
    return os.path.join(TEX_DIR, filename)

def load_texture(path, flip_y=True, is_hdr=False):
    """Load a texture, return GL texture id."""
    full_path = resource_path(path)
    if not os.path.exists(full_path):
        print(f"[WARN] Texture not found: {full_path}")
        return 0
    
    tex_id = 0
    try:
        if is_hdr:
            # Try loading HDR with imageio
            try:
                import imageio.v3 as iio
                img_data = iio.imread(full_path)
            except:
                print(f"[WARN] Could not load HDR {path}, trying as standard image")
                img = Image.open(full_path)
                img_data = np.array(img, dtype=np.uint8)
                is_hdr = False
            
            if is_hdr:
                if img_data.dtype != np.float32:
                    img_data = img_data.astype(np.float32)
                
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
            else:
                img = Image.open(full_path)
                if img.mode not in ["RGB", "RGBA"]:
                    img = img.convert("RGB")
                if flip_y:
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)
                
                width, height = img.size
                gl_format = GL_RGBA if img.mode == "RGBA" else GL_RGB
                internal_format = GL_RGBA8 if img.mode == "RGBA" else GL_RGB8
                data_type = GL_UNSIGNED_BYTE
                img_data_bytes = img.tobytes()
        else:
            # Standard LDR loading
            img = Image.open(full_path)
            if img.mode not in ["RGB", "RGBA"]:
                img = img.convert("RGB")
            if flip_y:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            
            width, height = img.size
            gl_format = GL_RGBA if img.mode == "RGBA" else GL_RGB
            internal_format = GL_RGBA8 if img.mode == "RGBA" else GL_RGB8
            data_type = GL_UNSIGNED_BYTE
            img_data_bytes = img.tobytes()
        
        # Upload to GPU
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
        print(f"[ERROR] {full_path}: {e}")
        if tex_id != 0:
            glDeleteTextures(1, [tex_id])
        return 0

def draw_textured_sphere(radius, tex_id, color=(1,1,1), slices=SPHERE_SLICES, stacks=SPHERE_STACKS):
    """Draw a textured sphere"""
    glColor3f(*color)
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

def draw_orbit_path(radius, num_segments=300):
    """Draw orbit path around the sun"""
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(0.5, 0.6, 1.0, 0.2)
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

def draw_label(x, y, z, text, scale=1.0):
    """Draw 3D text label"""
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glColor3f(1.0, 1.0, 1.0)
    glRasterPos3f(x, y, z)
    for ch in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(ch))
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)

def draw_glow_sphere(radius, scale=1.5, color=(1.0, 1.0, 1.0), intensity=1.0):
    """Draw a glowing sphere around a planet"""
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE)
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    
    quad = gluNewQuadric()
    gluQuadricNormals(quad, GLU_SMOOTH)
    r, g, b = color
    glColor4f(r, g, b, 0.8 * intensity)
    gluSphere(quad, radius * scale, 64, 64)
    gluDeleteQuadric(quad)
    
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glDisable(GL_BLEND)

def generate_starfield_texture(width=2048, height=1024):
    """Generate procedural starfield"""
    np.random.seed(123)
    texture = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add bright stars
    for _ in range(8000):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        brightness = np.random.randint(150, 255)
        size = np.random.randint(1, 3)
        texture[max(0, y-size):min(height, y+size), max(0, x-size):min(width, x+size)] = [brightness, brightness, brightness]
    
    # Add nebula clouds
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

# ============= MAIN APP =============
class SolarSystemApp:
    def __init__(self, width, height):
        pygame.init()
        glutInit()
        self.screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Cinematic Solar System - Use Mouse to Rotate | 1-9: Planet Focus | 0: Free View")
        
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
        
        self.load_all_textures()
        self.init_opengl()

    def load_all_textures(self):
        """Load all textures"""
        print("\n=== Loading Textures ===")
        for name, data in SOLAR_SYSTEM_DATA.items():
            self.textures[name] = load_texture(data[1])
        for name, data in MOON_DATA.items():
            self.textures[name] = load_texture(data[1])
        self.ring_tex = load_texture(SATURN_RING_TEXTURE)
        self.sun_glow_tex = load_texture(SUN_GLOW_TEXTURE)
        
        # Load starfield (HDR or fallback to procedural)
        self.starfield_tex = load_texture(STARFIELD_TEXTURE, flip_y=False, is_hdr=True)
        if self.starfield_tex == 0:
            print("[INFO] Generating procedural starfield...")
            self.starfield_tex = self.create_procedural_starfield()
        print("=== Ready ===\n")

    def create_procedural_starfield(self):
        """Create procedural starfield texture"""
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
        
        # Enhanced lighting for better clarity
        glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 0.0, 0.0, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.35, 1.0])
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.25, 0.25, 0.3, 1.0])
        
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        
        # Enable material properties for better texture clarity
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 32)

    def handle_events(self):
        """Handle input"""
        for ev in pygame.event.get():
            if ev.type == QUIT or (ev.type == KEYDOWN and ev.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            
            if ev.type == KEYDOWN:
                if K_1 <= ev.key <= K_9:
                    idx = ev.key - K_1
                    if idx < len(PLANET_ORDER):
                        self.camera_target = PLANET_ORDER[idx]
                        print(f"Tracking: {self.camera_target}")
                elif ev.key == K_0:
                    self.camera_target = None
                    print("Free view")
                elif ev.key == K_PLUS or ev.key == K_EQUALS:
                    self.fov = max(10.0, self.fov - 5)
                    self.set_projection()
                elif ev.key == K_MINUS:
                    self.fov = min(120.0, self.fov + 5)
                    self.set_projection()
            
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
        """Update camera"""
        self.cam_rot_y += self.vel_y * dt * 60
        self.cam_rot_x += self.vel_x * dt * 60
        self.vel_x *= DRAG_DAMPING
        self.vel_y *= DRAG_DAMPING
        self.cam_rot_x = max(-89.0, min(89.0, self.cam_rot_x))

    def draw_starfield(self):
        """Draw background starfield"""
        glPushMatrix()
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glColor3f(1.0, 1.0, 1.0)
        glCullFace(GL_FRONT)
        draw_textured_sphere(30000, self.starfield_tex, slices=64, stacks=32)
        glCullFace(GL_BACK)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glPopMatrix()

    def render_scene(self, t):
        """Render everything"""
        # Update planet positions
        self.planet_positions.clear()
        for name, data in SOLAR_SYSTEM_DATA.items():
            if name == "Sun":
                self.planet_positions["Sun"] = (0, 0, 0)
                continue
            radius, _, orbit_r, orbit_speed, _, _, _ = data
            angle = t * orbit_speed
            x = orbit_r * math.sin(angle)
            z = orbit_r * math.cos(angle)
            self.planet_positions[name] = (x, 0.0, z)
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Draw starfield (independent of camera zoom)
        glPushMatrix()
        glLoadIdentity()
        glRotatef(self.cam_rot_x, 1, 0, 0)
        glRotatef(self.cam_rot_y, 0, 1, 0)
        self.draw_starfield()
        glPopMatrix()
        
        # Setup main camera with enhanced lighting
        glLoadIdentity()
        glTranslatef(0.0, 0.0, self.cam_dist)
        glRotatef(self.cam_rot_x, 1, 0, 0)
        glRotatef(self.cam_rot_y, 0, 1, 0)
        
        # Update sun light position
        glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 0.0, 0.0, 1.0])
        
        # Follow target if selected
        if self.camera_target and self.camera_target in self.planet_positions:
            px, py, pz = self.planet_positions[self.camera_target]
            glTranslatef(-px, -py, -pz)
        
        # Draw Sun
        glPushMatrix()
        sun_data = SOLAR_SYSTEM_DATA["Sun"]
        radius, tex, _, _, spin, tilt, color = sun_data
        glRotatef(tilt, 1, 0, 0)
        glRotatef((t * spin * 50.0) % 360.0, 0, 1, 0)
        
        # Enhanced lighting for sun clarity
        glEnable(GL_LIGHTING)
        glColor3f(1.0, 1.0, 1.0)
        draw_textured_sphere(radius, self.textures.get("Sun"), (1.0, 1.0, 1.0))
        
        glPopMatrix()
        draw_label(0, sun_data[0] + 6, 0, "SUN")
        
        # Draw planets
        # Draw planets
        for name, data in SOLAR_SYSTEM_DATA.items():
            if name == "Sun":
                continue
            
            radius, tex, orbit_r, orbit_speed, spin, tilt, color = data
            x, y, z = self.planet_positions[name]
            
            # --- FIX IS HERE ---
            # 1. Draw the orbit path FIRST, centered at the sun (0,0,0).
            draw_orbit_path(orbit_r)

            # 2. NOW, move to the planet's position to draw the planet itself.
            glPushMatrix()
            glTranslatef(x, y, z)
            
            # The call to draw_orbit_path() was incorrectly here. It has been moved up.
            
            glRotatef(tilt, 1, 0, 0)
            glRotatef((t * spin * 50.0) % 360.0, 0, 1, 0)
            
            draw_textured_sphere(radius, self.textures.get(name), color)
            
            # Special effects (this part remains the same)
            if name == "Earth":
                # Subtle atmosphere glow
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                glDisable(GL_LIGHTING)
                glColor4f(0.3, 0.5, 0.9, 0.15)
                draw_glow_sphere(radius, 1.05, (0.3, 0.5, 0.9), 0.3)
                glEnable(GL_LIGHTING)
                glDisable(GL_BLEND)
                
                # Moon
                m_data = MOON_DATA["Moon"]
                m_rad, m_tex, m_dist, m_speed, m_spin = m_data
                m_angle = t * m_speed
                mx = m_dist * math.sin(m_angle)
                mz = m_dist * math.cos(m_angle)
                
                glPushMatrix()
                glTranslatef(mx, 0, mz)
                glRotatef((t * m_spin * 50) % 360, 0, 1, 0)
                draw_textured_sphere(m_rad, self.textures.get("Moon"), (1.0, 1.0, 1.0))
                glPopMatrix()
            
            elif name == "Saturn":
                # Saturn rings
                if self.ring_tex > 0:
                    glEnable(GL_TEXTURE_2D)
                    glBindTexture(GL_TEXTURE_2D, self.ring_tex)
                    glEnable(GL_BLEND)
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                    glDisable(GL_LIGHTING)
                    glColor4f(1.0, 1.0, 1.0, 0.7)
                    
                    inner = radius + 3
                    outer = radius + 14
                    segments = 200
                    
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
                    glDisable(GL_BLEND)
                    glBindTexture(GL_TEXTURE_2D, 0)
                    glDisable(GL_TEXTURE_2D)
            
            glPopMatrix()
            draw_label(x, y + radius + 3, z, name.upper())
        
    def run(self):
        """Main loop"""
        clock = pygame.time.Clock()
        last_time = pytime.time()
        
        while True:
            now = pytime.time()
            dt = now - last_time
            last_time = now
            
            self.handle_events()
            self.update_camera(dt)
            
            t = pygame.time.get_ticks() / 8000.0  # Slower animation
            self.render_scene(t)
            
            pygame.display.flip()
            clock.tick(FPS)

if __name__ == "__main__":
    if not os.path.isdir(TEX_DIR):
        print(f"ERROR: Create folder '{TEX_DIR}' and add texture files")
    else:
        app = SolarSystemApp(WINDOW_WIDTH, WINDOW_HEIGHT)
        app.run()