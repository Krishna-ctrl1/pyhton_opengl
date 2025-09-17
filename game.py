import sys
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

x_rot = y_rot = 0.0
last_x = last_y = None

def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glEnable(GL_DEPTH_TEST)

def display():
    global x_rot, y_rot
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    glTranslatef(0.0, 0.0, -6.0)

    glRotatef(x_rot, 1.0, 0.0, 0.0)
    glRotatef(y_rot, 0.0, 1.0, 0.0)

    glBegin(GL_QUADS)

    # Face 1: Light Blue
    glColor3f(0.6, 0.8, 1.0)
    glVertex3f(-1.0, -1.0, 1.0)
    glVertex3f(1.0, -1.0, 1.0)
    glVertex3f(1.0, 1.0, 1.0)
    glVertex3f(-1.0, 1.0, 1.0)

    # Face 2: Orange
    glColor3f(1.0, 0.6, 0.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glVertex3f(-1.0, 1.0, -1.0)
    glVertex3f(1.0, 1.0, -1.0)
    glVertex3f(1.0, -1.0, -1.0)

    # Face 3: Green
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(-1.0, 1.0, -1.0)
    glVertex3f(-1.0, 1.0, 1.0)
    glVertex3f(1.0, 1.0, 1.0)
    glVertex3f(1.0, 1.0, -1.0)

    # Face 4: Purple
    glColor3f(0.8, 0.4, 0.8)
    glVertex3f(-1.0, -1.0, -1.0)
    glVertex3f(1.0, -1.0, -1.0)
    glVertex3f(1.0, -1.0, 1.0)
    glVertex3f(-1.0, -1.0, 1.0)

    # Face 5: Yellow
    glColor3f(1.0, 1.0, 0.0)
    glVertex3f(1.0, -1.0, -1.0)
    glVertex3f(1.0, 1.0, -1.0)
    glVertex3f(1.0, 1.0, 1.0)
    glVertex3f(1.0, -1.0, 1.0)

    # Face 6: Pink
    glColor3f(1.0, 0.7, 0.8)
    glVertex3f(-1.0, -1.0, -1.0)
    glVertex3f(-1.0, -1.0, 1.0)
    glVertex3f(-1.0, 1.0, 1.0)
    glVertex3f(-1.0, 1.0, -1.0)

    glEnd()
    glutSwapBuffers()

def mouse_motion(x, y):
    global x_rot, y_rot, last_x, last_y
    if last_x is not None and last_y is not None:
        dx = x - last_x
        dy = y - last_y
        y_rot += dx * 0.5   
        x_rot += dy * 0.5   
    last_x, last_y = x, y
    glutPostRedisplay()

def reshape(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, w/h, 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)

def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(600, 600)
    glutCreateWindow(b"Rotating Cube - Mouse Controlled")
    init()
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutMotionFunc(mouse_motion)      
    glutPassiveMotionFunc(mouse_motion) 
    glutMainLoop()

if __name__ == "__main__":
    main()
