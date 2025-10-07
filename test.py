import os
import math
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GLUT import GLUT_BITMAP_HELVETICA_12
from PIL import Image
import numpy as np
import imageio.v3 as iio
import sys
import time as pytime

# ---------------- CONFIG ----------------
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
FPS = 60
SPHERE_SLICES = 256  # Increased for smoother planets
SPHERE_STACKS = 256

# Camera inertia / sensitivity
MOUSE_SENSITIVITY = 0.2
DRAG_DAMPING = 0.90
WHEEL_ZOOM_STEP = 50.0

# Textures folder
TEX_DIR = "textures"

# ---------------- SCENE DATA ----------------
# Format: name: (radius, texture, orbit_radius, orbit_speed, self_spin_speed, axial_tilt)
SOLAR_SYSTEM_DATA = {
    "Sun":     (20.0, "sun.jpg",       0,    0.0,  0.05, 7.25),
    "Mercury": (1.2,  "mercury.jpg",   40,   4.7,  0.1,  0.03),
    "Venus":   (2.8,  "venus.jpg",     70,   3.5, -0.05, 177.3),
    "Earth":   (3.0,  "earth.jpg",     100,  2.9,  1.0,  23.44),
    "Mars":    (1.8,  "mars.jpg",      150,  2.4,  0.9,  25.19),
    "Jupiter": (10.0, "jupiter.jpg",   250,  1.3,  2.4,  3.13),
    "Saturn":  (8.5,  "saturn.jpg",    350,  0.9,  2.2,  26.73),
    "Uranus":  (6.0,  "uranus.jpg",    450,  0.6, -1.4,  97.77),
    "Neptune": (5.8,  "neptune.jpg",   550,  0.5,  1.5,  28.32),
}

PLANET_ORDER = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]

MOON_DATA = {
    "Moon": (1.0, "moon.jpg", 8.0, 10.0, 0.4)
}

SATURN_RING_TEXTURE = "saturn_rings.png"
SUN_GLOW_TEXTURE = "sun_glow.png"
STARFIELD_TEXTURE = "HDR_rich_blue_nebulae_1.hdr" 

# ----------------- HELPERS -----------------
def resource_path(filename):
    return os.path.join(TEX_DIR, filename)

def load_texture(path, flip_y=True, is_hdr=False):
    """Load a texture, return GL texture id. Supports LDR (8-bit) and HDR (float)."""
    full_path = resource_path(path)
    if not os.path.exists(full_path):
        print(f"[WARN] Texture not found: {full_path}")
        return 0
    
    tex_id = 0
    try:
        if is_hdr:
            # FIX: Removed the invalid plugin='imageio' argument.
            # imageio will now auto-detect the correct plugin for the .hdr file.
            img_data = iio.imread(full_path, index=0)
            
            if img_data.dtype != np.float32:
                img_data = img_data.astype(np.float32)

            height, width, channels = img_data.shape
            
            if channels == 3:
                gl_format = GL_RGB
                internal_format = GL_RGB16F
            elif channels == 4:
                gl_format = GL_RGBA
                internal_format = GL_RGBA16F
            else:
                print(f"[ERROR] Unsupported channel count for HDR: {channels}")
                return 0
            
            data_type = GL_FLOAT
            
            if flip_y:
                 # Flip the data along the vertical axis (rows)
                img_data = np.flipud(img_data)

            img_data_bytes = img_data.tobytes()

        else:
            # LDR loading logic
            img = Image.open(full_path)
            if flip_y:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            
            mode = img.mode
            if mode == "P":
                img = img.convert("RGBA")
                mode = "RGBA"

            img_data_bytes = img.tobytes()
            width, height = img.size
            
            gl_format = GL_RGB
            internal_format = GL_RGB8
            data_type = GL_UNSIGNED_BYTE
            if mode == "RGBA":
                gl_format = GL_RGBA
                internal_format = GL_RGBA8
        
        # Common OpenGL Texture Setup
        tex_id_scalar = glGenTextures(1)
        tex_id = int(tex_id_scalar) if isinstance(tex_id_scalar, np.generic) else tex_id_scalar
        
        glBindTexture(GL_TEXTURE_2D, tex_id)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        glTexImage2D(GL_TEXTURE_2D, 0, internal_format, width, height, 0, gl_format, data_type, img_data_bytes)
        
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)
        print(f"[INFO] Loaded texture: {path} (HDR: {is_hdr})")
        return tex_id
    
    except Exception as e:
        print(f"[ERROR] Failed to load texture {full_path}: {e}")
        if tex_id != 0:
             glDeleteTextures(1, [tex_id]) 
        return 0

def draw_textured_sphere(radius, tex_id, slices=SPHERE_SLICES, stacks=SPHERE_STACKS):
    if tex_id > 0:
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tex_id)
    
    quad = gluNewQuadric()
    gluQuadricTexture(quad, GL_TRUE)
    gluQuadricNormals(quad, GLU_SMOOTH)
    gluSphere(quad, radius, slices, stacks)
    gluDeleteQuadric(quad)

    if tex_id > 0:
        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)

def draw_orbit(radius, alpha=0.1):
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(1.0, 1.0, 1.0, alpha)
    glBegin(GL_LINE_LOOP)
    for i in range(360):
        t = math.radians(i)
        x = radius * math.cos(t)
        z = radius * math.sin(t)
        glVertex3f(x, 0.0, z)
    glEnd()
    glDisable(GL_BLEND)
    glEnable(GL_LIGHTING)

def draw_label(x, y, z, text):
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glColor3f(0.8, 0.8, 0.8)
    glRasterPos3f(x, y, z)
    for ch in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(ch))
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)

def draw_billboard_quad(size, tex_id):
    if tex_id <= 0: return
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glPushMatrix()
    modelview = (GLfloat * 16)()
    glGetFloatv(GL_MODELVIEW_MATRIX, modelview)
    # Undo rotation
    for i in range(3):
        for j in range(3):
            modelview[i*4 + j] = 1.0 if i == j else 0.0
    glLoadMatrixf(modelview)
    
    half = size / 2.0
    glBegin(GL_QUADS)
    glTexCoord2f(0,0); glVertex3f(-half, -half, 0)
    glTexCoord2f(1,0); glVertex3f( half, -half, 0)
    glTexCoord2f(1,1); glVertex3f( half,  half, 0)
    glTexCoord2f(0,1); glVertex3f(-half,  half, 0)
    glEnd()
    glPopMatrix()
    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)

def draw_textured_ring(inner_r, outer_r, tex_id, segments=256):
    if tex_id <= 0: return
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    glDisable(GL_LIGHTING)
    glColor4f(1.0, 1.0, 1.0, 0.8) # Ring transparency
    glBegin(GL_TRIANGLE_STRIP)
    for i in range(segments + 1):
        theta = (i / segments) * (2.0 * math.pi)
        u = i / segments
        x_outer, z_outer = outer_r * math.cos(theta), outer_r * math.sin(theta)
        x_inner, z_inner = inner_r * math.cos(theta), inner_r * math.sin(theta)
        glTexCoord2f(u, 1.0); glVertex3f(x_outer, 0.0, z_outer)
        glTexCoord2f(u, 0.0); glVertex3f(x_inner, 0.0, z_inner)
    glEnd()
    glEnable(GL_LIGHTING)
    glDisable(GL_BLEND)
    glBindTexture(GL_TEXTURE_2D, 0)
    glDisable(GL_TEXTURE_2D)

# ----------------- APP -----------------
class SolarSystemApp:
    def __init__(self, width, height):
        pygame.init()
        glutInit()
        self.screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Cinematic Solar System")

        self.width, self.height = width, height
        self.fov = 45.0
        self.cam_dist = -1200.0
        self.cam_rot_x, self.cam_rot_y = -20.0, 0.0
        self.dragging, self.last_mouse = False, None
        self.vel_x, self.vel_y = 0.0, 0.0
        
        self.camera_target = None # Planet name to track
        self.planet_positions = {}

        self.textures = {}
        self.load_all_textures()
        self.init_opengl()

    def load_all_textures(self):
        for name, data in SOLAR_SYSTEM_DATA.items():
            self.textures[name] = load_texture(data[1])
        for name, data in MOON_DATA.items():
            self.textures[name] = load_texture(data[1])
        self.ring_tex = load_texture(SATURN_RING_TEXTURE)
        self.sun_glow_tex = load_texture(SUN_GLOW_TEXTURE)
        self.starfield_tex = load_texture(STARFIELD_TEXTURE, flip_y=False, is_hdr=True)

    def set_projection(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, float(self.width)/self.height, 0.1, 40000.0)
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
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 0.95, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.05, 0.05, 0.05, 1.0]) # Darker shadows
        
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_CULL_FACE)

    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == QUIT or (ev.type == KEYDOWN and ev.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            # New camera controls
            if ev.type == KEYDOWN:
                if K_1 <= ev.key <= K_9:
                    self.camera_target = PLANET_ORDER[ev.key - K_1]
                elif ev.key == K_0:
                    self.camera_target = None
                elif ev.key == K_PLUS or ev.key == K_EQUALS:
                    self.fov = max(15.0, self.fov - 5)
                    self.set_projection()
                elif ev.key == K_MINUS:
                    self.fov = min(90.0, self.fov + 5)
                    self.set_projection()
            
            # Mouse controls
            if ev.type == MOUSEBUTTONDOWN:
                if ev.button == 1: self.dragging, self.last_mouse = True, ev.pos
                elif ev.button == 4: self.cam_dist += WHEEL_ZOOM_STEP
                elif ev.button == 5: self.cam_dist -= WHEEL_ZOOM_STEP
            if ev.type == MOUSEBUTTONUP:
                if ev.button == 1: self.dragging, self.last_mouse = False, None
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
        self.cam_rot_x = max(-89.9, min(89.9, self.cam_rot_x))
        
    def draw_starfield(self):
        glPushMatrix()
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glColor3f(1.0, 1.0, 1.0)
        glCullFace(GL_FRONT) # Render inside of sphere
        draw_textured_sphere(20000, self.starfield_tex)
        glCullFace(GL_BACK) # Reset
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glPopMatrix()

    def render_scene(self, t):
        self.planet_positions.clear() # Recalculate positions each frame
        for name, (radius, _, orbit_r, orbit_speed, _, _) in SOLAR_SYSTEM_DATA.items():
            if name == "Sun":
                self.planet_positions["Sun"] = (0,0,0)
                continue
            angle = t * orbit_speed
            x = orbit_r * math.sin(angle)
            z = orbit_r * math.cos(angle)
            self.planet_positions[name] = (x, 0.0, z)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # --- Draw Starfield ---
        # We draw the starfield first, wrapped in a matrix push/pop.
        # We apply only the camera's rotation so that the background moves
        # with the mouse, but doesn't zoom or pan, feeling infinitely distant.
        glPushMatrix()
        glLoadIdentity()
        glRotatef(self.cam_rot_x, 1, 0, 0)
        glRotatef(self.cam_rot_y, 0, 1, 0)
        self.draw_starfield()
        glPopMatrix()

        # --- Draw Main Scene ---
        # Now, set up the full camera for the solar system objects.
        glLoadIdentity()
        glTranslatef(0.0, 0.0, self.cam_dist)
        glRotatef(self.cam_rot_x, 1, 0, 0)
        glRotatef(self.cam_rot_y, 0, 1, 0)
        
        if self.camera_target and self.camera_target in self.planet_positions:
            px, py, pz = self.planet_positions[self.camera_target]
            glTranslatef(-px, -py, -pz)

        # --- Sun ---
        glPushMatrix()
        sun_data = SOLAR_SYSTEM_DATA["Sun"]
        glRotatef(sun_data[5], 1, 0, 0) # Axial tilt
        glRotatef((t * sun_data[4] * 100.0) % 360.0, 0.0, 1.0, 0.0) # Self-rotation
        
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 1.0, 1.0)
        draw_textured_sphere(sun_data[0], self.textures.get("Sun"))
        glEnable(GL_LIGHTING)
        
        # Sun Glow
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE) # Additive
        glColor4f(1.0, 0.9, 0.6, 1.0)
        draw_billboard_quad(sun_data[0] * 6.0, self.sun_glow_tex)
        glDisable(GL_BLEND)
        glPopMatrix()
        
        draw_label(0, sun_data[0] + 4, 0, "Sun")

        # --- Planets ---
        for name, (radius, _, orbit_r, _, spin, tilt) in SOLAR_SYSTEM_DATA.items():
            if name == "Sun": continue
            x, y, z = self.planet_positions[name]
            glPushMatrix()
            glTranslatef(x, y, z)
            
            draw_orbit(orbit_r)
            
            # Axial tilt first, then self-rotation
            glRotatef(tilt, 1, 0, 0) 
            glRotatef((t * spin * 100.0) % 360.0, 0.0, 1.0, 0.0)
            
            draw_textured_sphere(radius, self.textures.get(name))
            
            # Earth's Atmosphere
            if name == "Earth":
                glEnable(GL_BLEND)
                glBlendFunc(GL_SRC_ALPHA, GL_ONE) # Additive blend for glow
                glDisable(GL_LIGHTING)
                glColor4f(0.5, 0.7, 1.0, 0.3) # Blueish transparent glow
                draw_textured_sphere(radius * 1.05, 0) # Draw untextured sphere
                glEnable(GL_LIGHTING)
                glDisable(GL_BLEND)
                
                # Moon
                m_data = MOON_DATA["Moon"]
                mangle = t * m_data[3]; mx, mz = m_data[2]*math.sin(mangle), m_data[2]*math.cos(mangle)
                glPushMatrix()
                glTranslatef(mx, 0.0, mz)
                glRotatef((t * m_data[4] * 360.0) % 360.0, 0,1,0)
                draw_textured_sphere(m_data[0], self.textures.get("Moon"))
                glPopMatrix()

            # Saturn Rings
            if name == "Saturn":
                draw_textured_ring(radius + 2.5, radius + 10.0, self.ring_tex)

            glPopMatrix()
            draw_label(x, y + radius + 1.5, z, name)
            
    def run(self):
        clock = pygame.time.Clock()
        last_time = pytime.time()
        while True:
            now = pytime.time()
            dt = now - last_time
            last_time = now
            
            self.handle_events()
            self.update_camera(dt)
            
            t = pygame.time.get_ticks() / 5000.0 # Slow down simulation time
            self.render_scene(t)
            
            pygame.display.flip()
            clock.tick(FPS)

if __name__ == "__main__":
    if not os.path.isdir(TEX_DIR):
        print(f"ERROR: Create a folder named '{TEX_DIR}' and place texture files inside.")
    else:
        app = SolarSystemApp(WINDOW_WIDTH, WINDOW_HEIGHT)
        app.run()