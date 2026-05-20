import glfw
from OpenGL.GL import *
from OpenGL.GLU import *

rotation = 0.0


def draw_sphere(radius):
    """Dibuja esfera usando GLU (sin GLUT)"""
    quad = gluNewQuadric()
    gluSphere(quad, radius, 32, 32)
    gluDeleteQuadric(quad)


def draw_eye():
    glPushMatrix()

    # Exterior (piel)
    glColor3f(0.85, 0.67, 0.65)
    glPushMatrix()
    glTranslatef(0.7, 0, 0)
    draw_sphere(0.54)
    glPopMatrix()

    # Blanco
    glColor3f(1, 1, 1)
    glPushMatrix()
    glTranslatef(0.56, 0, 0)
    draw_sphere(0.6)
    glPopMatrix()

    # Iris
    glColor3f(0.84, 0.85, 0.92)
    glPushMatrix()
    glTranslatef(0.49, 0, 0)
    draw_sphere(0.55)
    glPopMatrix()

    # Pupila
    glColor3f(0, 0, 0)
    glPushMatrix()
    glTranslatef(0.3, 0, 0)
    draw_sphere(0.4)
    glPopMatrix()

    glPopMatrix()


def setup_lighting():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)

    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    light_position = [2.0, 2.0, 2.0, 1.0]
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)

    ambient = [0.3, 0.3, 0.3, 1.0]
    diffuse = [1.0, 1.0, 1.0, 1.0]

    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse)


def resize(window, width, height):
    if height == 0:
        height = 1

    glViewport(0, 0, width, height)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, width / height, 0.1, 100.0)

    glMatrixMode(GL_MODELVIEW)


def main():
    global rotation

    if not glfw.init():
        print("Error iniciando GLFW")
        return

    window = glfw.create_window(800, 600, "Ojo 3D", None, None)
    if not window:
        print("No se pudo crear ventana")
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_window_size_callback(window, resize)

    glClearColor(0.54, 0.72, 0.84, 1.0)

    setup_lighting()
    resize(window, 800, 600)

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()
        glTranslatef(0, 0, -5)

        # rotación
        rotation += 0.5
        glRotatef(rotation, 0, 1, 0)

        draw_eye()

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == "__main__":
    main()