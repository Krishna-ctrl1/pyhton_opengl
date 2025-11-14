import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math

vertices = (
    (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1),
    (1, -1, 1), (1, 1, 1), (-1, -1, 1), (-1, 1, 1)
)

faces = (
    (0, 1, 2, 3),
    (4, 5, 7, 6),
    (0, 1, 5, 4),
    (2, 3, 6, 7),
    (1, 2, 7, 5),
    (0, 3, 6, 4)
)

normals = (
    (0, 0, -1),
    (0, 0, 1),
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0)
)

material_A_highly_reflective_metallic = {
    'name': 'Highly Reflective Metallic (Chrome)',
    'ambient': (0.1, 0.1, 0.1, 1.0),
    'diffuse': (0.2, 0.2, 0.2, 1.0),
    'specular': (1.0, 1.0, 1.0, 1.0),
    'shininess': 128.0,
    'emission': (0.0, 0.0, 0.0, 1.0)
}

material_B_semi_metallic = {
    'name': 'Semi-Metallic Medium Reflectance (Gold)',
    'ambient': (0.3, 0.2, 0.0, 1.0),
    'diffuse': (0.8, 0.6, 0.0, 1.0),
    'specular': (0.5, 0.4, 0.2, 1.0),
    'shininess': 40.0,
    'emission': (0.0, 0.0, 0.0, 1.0)
}

material_C_matte_low_reflectance = {
    'name': 'Matte Surface Low Reflectance (Green Rubber)',
    'ambient': (0.0, 0.1, 0.0, 1.0),
    'diffuse': (0.2, 0.8, 0.2, 1.0),
    'specular': (0.0, 0.0, 0.0, 1.0),
    'shininess': 1.0,
    'emission': (0.0, 0.0, 0.0, 1.0)
}

face_materials = [
    material_A_highly_reflective_metallic,
    material_A_highly_reflective_metallic,
    material_B_semi_metallic,
    material_B_semi_metallic,
    material_C_matte_low_reflectance,
    material_C_matte_low_reflectance
]

def setup_lighting():
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_NORMALIZE)
    
    light_position = [20.0, 20.0, 20.0, 1.0]
    light_ambient = [0.5, 0.5, 0.5, 1.0]
    light_diffuse = [1.5, 1.5, 1.5, 1.0]
    light_specular = [2.0, 2.0, 2.0, 1.0]
    
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)
    
    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 1.0)
    glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.0)
    glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0.0)

def apply_material(material):
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, material['ambient'])
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, material['diffuse'])
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, material['specular'])
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, material['shininess'])
    glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, material['emission'])

def draw_cube():
    glBegin(GL_QUADS)
    for i, face in enumerate(faces):
        apply_material(face_materials[i])
        glNormal3fv(normals[i])
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()

def draw_light_source():
    glDisable(GL_LIGHTING)
    glPushMatrix()
    glTranslatef(20.0, 20.0, 20.0)
    glColor3f(1.0, 1.0, 0.0)
    quad = gluNewQuadric()
    gluSphere(quad, 0.5, 20, 20)
    glPopMatrix()
    glEnable(GL_LIGHTING)

def draw_environment_grid():
    glDisable(GL_LIGHTING)
    glColor3f(0.3, 0.3, 0.3)
    glLineWidth(1)
    glBegin(GL_LINES)
    for i in range(-20, 21, 2):
        glVertex3f(i, -10, -20)
        glVertex3f(i, -10, 20)
        glVertex3f(-20, -10, i)
        glVertex3f(20, -10, i)
    glEnd()
    glEnable(GL_LIGHTING)

def draw_axes():
    glDisable(GL_LIGHTING)
    glLineWidth(3)
    glBegin(GL_LINES)
    glColor3f(1, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(5, 0, 0)
    glColor3f(0, 1, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 5, 0)
    glColor3f(0, 0, 1)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, 5)
    glEnd()
    glEnable(GL_LIGHTING)

def get_pentagon_vertices(radius=8):
    pentagon = []
    for i in range(5):
        angle = 2 * math.pi * i / 5 - math.pi / 2
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        pentagon.append((x, y, 0))
    return pentagon

def draw_pentagon_path(pentagon_vertices):
    glDisable(GL_LIGHTING)
    glColor3f(0.8, 0.8, 1.0)
    glLineWidth(4)
    glBegin(GL_LINE_LOOP)
    for vertex in pentagon_vertices:
        glVertex3fv(vertex)
    glEnd()
    
    glPointSize(15)
    glColor3f(1, 1, 0)
    glBegin(GL_POINTS)
    for vertex in pentagon_vertices:
        glVertex3fv(vertex)
    glEnd()
    glEnable(GL_LIGHTING)

def interpolate_position(start, end, t):
    return (
        start[0] + (end[0] - start[0]) * t,
        start[1] + (end[1] - start[1]) * t,
        start[2] + (end[2] - start[2]) * t
    )

class Camera:
    def __init__(self):
        self.distance = 25
        self.rotation_x = 25
        self.rotation_y = 45
        self.translation_x = 0
        self.translation_y = 0
        self.target_x = 0
        self.target_y = 0
        self.target_z = 0
        self.fov = 45
    
    def apply_view(self):
        glLoadIdentity()
        glTranslatef(self.translation_x, self.translation_y, -self.distance)
        glRotatef(self.rotation_x, 1, 0, 0)
        glRotatef(self.rotation_y, 0, 1, 0)
        glTranslatef(-self.target_x, -self.target_y, -self.target_z)
    
    def zoom(self, amount):
        self.distance = max(5, min(60, self.distance + amount))
    
    def rotate(self, dx, dy):
        self.rotation_y += dx
        self.rotation_x += dy
        self.rotation_x = max(-89, min(89, self.rotation_x))
    
    def pan(self, dx, dy):
        self.translation_x += dx
        self.translation_y += dy

def draw_info_text(screen, camera, current_material):
    font = pygame.font.Font(None, 24)
    y_offset = 10
    
    texts = [
        f"Camera Distance: {camera.distance:.1f}",
        f"Camera Rotation: X={camera.rotation_x:.1f}° Y={camera.rotation_y:.1f}°",
        f"Current Material: {current_material}",
        "",
        "Controls:",
        "Mouse Drag - Rotate View",
        "W/S - Zoom In/Out",
        "Arrow Keys - Pan Camera",
        "Q/E - Adjust FOV",
        "R - Reset Camera",
        "Space - Toggle Auto-Rotate",
        "ESC - Exit"
    ]
    
    for text in texts:
        surface = font.render(text, True, (255, 255, 255))
        screen.blit(surface, (10, y_offset))
        y_offset += 25

def main():
    pygame.init()
    display = (1400, 900)
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Enhanced 3D Cube - Camera Control, Materials & Lighting System")
    
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (display[0] / display[1]), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    setup_lighting()
    
    glClearColor(0.05, 0.05, 0.1, 1.0)
    
    camera = Camera()
    pentagon_vertices = get_pentagon_vertices()
    current_vertex = 0
    next_vertex = 1
    transition_progress = 0
    transition_speed = 0.004
    rotation_angle = 0
    auto_rotate = True
    
    mouse_down = False
    last_mouse_pos = (0, 0)
    
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    print("\n" + "="*70)
    print("ENHANCED 3D CUBE VISUALIZATION")
    print("="*70)
    print("\nCAMERA CONTROLS:")
    print("  Mouse Drag     : Rotate camera around scene")
    print("  W / S          : Zoom in / Zoom out")
    print("  Arrow Keys     : Pan camera view")
    print("  Q / E          : Decrease / Increase FOV")
    print("  R              : Reset camera to default position")
    print("  Space          : Toggle auto-rotation")
    print("  ESC            : Exit application")
    print("\nMATERIAL PROPERTIES:")
    print("  Material A (Faces 0,1): HIGHLY REFLECTIVE METALLIC (CHROME MIRROR)")
    print("    - Shininess: 128 (MAXIMUM)")
    print("    - Specular: (1.0, 1.0, 1.0) - PURE WHITE MIRROR REFLECTIONS")
    print("    - Looks like a POLISHED MIRROR - EXTREMELY SHINY")
    print("  Material B (Faces 2,3): SEMI-METALLIC MEDIUM REFLECTANCE (GOLD)")
    print("    - Shininess: 40 (MEDIUM)")
    print("    - Bright GOLDEN/YELLOW color")
    print("    - MODERATE shine - clearly less than chrome")
    print("  Material C (Faces 4,5): MATTE SURFACE LOW REFLECTANCE (GREEN RUBBER)")
    print("    - Shininess: 1 (MINIMUM)")
    print("    - Specular: (0.0, 0.0, 0.0) - NO REFLECTIONS AT ALL")
    print("    - Looks COMPLETELY FLAT - NO SHINE WHATSOEVER")
    print("\nLIGHTING:")
    print("  Single Point Light Source at position (20, 20, 20)")
    print("  INTENSE lighting with boosted specular (2.0) for dramatic effect")
    print("="*70 + "\n")
    
    frame_count = 0
    
    while True:
        frame_count += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                elif event.key == pygame.K_r:
                    camera = Camera()
                elif event.key == pygame.K_SPACE:
                    auto_rotate = not auto_rotate
                elif event.key == pygame.K_q:
                    camera.fov = max(30, camera.fov - 2)
                    glMatrixMode(GL_PROJECTION)
                    glLoadIdentity()
                    gluPerspective(camera.fov, (display[0] / display[1]), 0.1, 100.0)
                    glMatrixMode(GL_MODELVIEW)
                elif event.key == pygame.K_e:
                    camera.fov = min(90, camera.fov + 2)
                    glMatrixMode(GL_PROJECTION)
                    glLoadIdentity()
                    gluPerspective(camera.fov, (display[0] / display[1]), 0.1, 100.0)
                    glMatrixMode(GL_MODELVIEW)
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_down = True
                    last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 4:
                    camera.zoom(-1)
                elif event.button == 5:
                    camera.zoom(1)
            
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_down = False
            
            if event.type == pygame.MOUSEMOTION and mouse_down:
                current_pos = pygame.mouse.get_pos()
                dx = current_pos[0] - last_mouse_pos[0]
                dy = current_pos[1] - last_mouse_pos[1]
                camera.rotate(dx * 0.5, dy * 0.5)
                last_mouse_pos = current_pos
        
        keys = pygame.key.get_pressed()
        if keys[K_w]:
            camera.zoom(-0.3)
        if keys[K_s]:
            camera.zoom(0.3)
        if keys[K_LEFT]:
            camera.pan(-0.15, 0)
        if keys[K_RIGHT]:
            camera.pan(0.15, 0)
        if keys[K_UP]:
            camera.pan(0, 0.15)
        if keys[K_DOWN]:
            camera.pan(0, -0.15)
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        camera.apply_view()
        
        draw_environment_grid()
        draw_light_source()
        draw_pentagon_path(pentagon_vertices)
        
        start_pos = pentagon_vertices[current_vertex]
        end_pos = pentagon_vertices[next_vertex]
        current_pos = interpolate_position(start_pos, end_pos, transition_progress)
        
        glPushMatrix()
        glTranslatef(current_pos[0], current_pos[1], current_pos[2])
        if auto_rotate:
            glRotatef(rotation_angle, 1, 1, 0.5)
        draw_cube()
        glPopMatrix()
        
        current_face_index = int((rotation_angle / 60) % 6)
        current_material_name = face_materials[current_face_index]['name']
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, display[0], 0, display[1])
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        info_y = display[1] - 30
        info_texts = [
            f"FPS: {int(clock.get_fps())}",
            f"Camera: Dist={camera.distance:.1f} RotX={camera.rotation_x:.1f}° RotY={camera.rotation_y:.1f}° FOV={camera.fov}°",
            f"Viewing Material: {current_material_name}",
            f"Auto-Rotate: {'ON' if auto_rotate else 'OFF'}"
        ]
        
        for text in info_texts:
            surface = font.render(text, True, (255, 255, 255))
            screen.blit(surface, (10, info_y))
            info_y -= 25
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        
        if auto_rotate:
            rotation_angle += 0.8
            if rotation_angle >= 360:
                rotation_angle = 0
        
        transition_progress += transition_speed
        if transition_progress >= 1.0:
            transition_progress = 0
            current_vertex = next_vertex
            next_vertex = (next_vertex + 1) % len(pentagon_vertices)
        
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()