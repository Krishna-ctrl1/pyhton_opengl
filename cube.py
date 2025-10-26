import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math

# PART 1: 3D Cube with Different Colors on Each Face

vertices = (
    (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1),
    (1, -1, 1), (1, 1, 1), (-1, -1, 1), (-1, 1, 1)
)

edges = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 7), (3, 6)
)

faces = (
    (0, 1, 2, 3),
    (4, 5, 7, 6),
    (0, 1, 5, 4),
    (2, 3, 6, 7),
    (1, 2, 7, 5),
    (0, 3, 6, 4)
)

colors = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 0, 1),
    (0, 1, 1)
)

def draw_cube():
    glBegin(GL_QUADS)
    for i, face in enumerate(faces):
        glColor3fv(colors[i])
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()
    
    glColor3f(0, 0, 0)
    glLineWidth(2)
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

# PART 2: Automatic Rotation

def apply_rotation(angle):
    glRotatef(angle, 1, 1, 0.5)

# PART 3: Pentagon Path Translation

def get_pentagon_vertices(radius=8):
    pentagon = []
    for i in range(5):
        angle = 2 * math.pi * i / 5 - math.pi / 2
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        pentagon.append((x, y, 0))
    return pentagon

def draw_pentagon_path(pentagon_vertices):
    glColor3f(1, 1, 1)
    glLineWidth(3)
    glBegin(GL_LINE_LOOP)
    for vertex in pentagon_vertices:
        glVertex3fv(vertex)
    glEnd()
    
    glPointSize(12)
    glColor3f(1, 1, 0)
    glBegin(GL_POINTS)
    for vertex in pentagon_vertices:
        glVertex3fv(vertex)
    glEnd()

def interpolate_position(start, end, t):
    return (
        start[0] + (end[0] - start[0]) * t,
        start[1] + (end[1] - start[1]) * t,
        start[2] + (end[2] - start[2]) * t
    )

def main():
    pygame.init()
    display = (1200, 800)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("3D Rotating Cube - Pentagon Path")
    
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    
    glMatrixMode(GL_MODELVIEW)
    glTranslatef(0.0, 0.0, -20)
    glEnable(GL_DEPTH_TEST)
    
    pentagon_vertices = get_pentagon_vertices()
    
    rotation_angle = 0
    current_vertex = 0
    next_vertex = 1
    transition_progress = 0
    transition_speed = 0.005
    
    clock = pygame.time.Clock()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glPushMatrix()
        
        draw_pentagon_path(pentagon_vertices)
        
        start_pos = pentagon_vertices[current_vertex]
        end_pos = pentagon_vertices[next_vertex]
        current_pos = interpolate_position(start_pos, end_pos, transition_progress)
        
        glTranslatef(current_pos[0], current_pos[1], current_pos[2])
        
        apply_rotation(rotation_angle)
        
        draw_cube()
        
        glPopMatrix()
        
        rotation_angle += 1
        if rotation_angle > 360:
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