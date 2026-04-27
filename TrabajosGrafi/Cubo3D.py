
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import sys

# Variables globales
window = None
angle = 0  # Declaramos angle en el nivel superior

def init():
    # Configuración inicial de OpenGL
    glClearColor(0.0, 0.0, 0.0, 1.0)  # Color de fondo
    glEnable(GL_DEPTH_TEST)  # Activar prueba de profundidad para 3D
    # Configuración de proyección
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    
    gluPerspective(60, 1.0, 0.1, 50.0)
    # Cambiar a la matriz de modelo para los objetos
    glMatrixMode(GL_MODELVIEW)

def draw_cube():
    global angle
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Limpiar pantalla y buffer de profundidad
    #Cubo1
    glLoadIdentity()
    glTranslatef(-3.0, 0.0, -10.0) # Lo movemos a la izquierda y atrás
    glRotatef(angle, 1, 0, 0)      # Solo eje X
    dibujar_geometria_cubo()

    #Cubo 2
    glLoadIdentity()
    glTranslatef(0.0, 0.0, -10.0) 
    glRotatef(angle, 0, 1, 0)      
    dibujar_geometria_cubo()

    #Cybo 3 
    glLoadIdentity()
    glTranslatef(3.0, 0.0, -10.0) 
    glRotatef(angle, 0, 0, 1)     
    dibujar_geometria_cubo()

    glfw.swap_buffers(window)  
    angle += 0.1 

def dibujar_geometria_cubo():
    glBegin(GL_QUADS)  # Iniciar el cubo como un conjunto de caras (quads)

    # Cada conjunto de cuatro vértices representa una cara del cubo
    glColor3f(1.0, 0.0, 1.0)  # Rojo
    glVertex3f( 1, 1,-1)
    glColor3f(0.4, 1.0, 1.0)  # Verde
    glVertex3f(-1, 1,-1)
    glColor3f(0.4, 1.0, 1.0)  # Verde
    glVertex3f(-1, 1, 1)
    glColor3f(0.3, 0.8, 0.1)  # Verde
    glVertex3f( 1, 1, 1)

    glColor3f(0.0, 1.0, 0.0)  # Verde
    glVertex3f( 1,-1, 1)
    glVertex3f(-1,-1, 1)
    glVertex3f(-1,-1,-1)
    glVertex3f( 1,-1,-1)

    glColor3f(0.0, 0.0, 1.0)  # Azul
    glVertex3f( 1, 1, 1)
    glVertex3f(-1, 1, 1)
    glVertex3f(-1,-1, 1)
    glVertex3f( 1,-1, 1)

    glColor3f(1.0, 1.0, 0.0)  # Amarillo
    glVertex3f( 1,-1,-1)
    glVertex3f(-1,-1,-1)
    glVertex3f(-1, 1,-1)
    glVertex3f( 1, 1,-1)

    glColor3f(1.0, 0.0, 1.0)  # Magenta
    glVertex3f(-1, 1, 1)
    glVertex3f(-1, 1,-1)
    glVertex3f(-1,-1,-1)
    glVertex3f(-1,-1, 1)

    glColor3f(0.0, 1.0, 1.0)  # Cyan
    glVertex3f( 1, 1,-1)
    glVertex3f( 1, 1, 1)
    glVertex3f( 1,-1, 1)
    glVertex3f( 1,-1,-1)

    glEnd()
    glFlush()

 # Incrementar el ángulo para rotación

def main():
    global window

    # Inicializar GLFW
    if not glfw.init():
        sys.exit()

    # Crear ventana de GLFW
    width, height = 500, 500
    window = glfw.create_window(width, height, "Cubo 3D Rotando con GLFW", None, None)
    if not window:
        glfw.terminate()
        sys.exit()

    # Configurar el contexto de OpenGL en la ventana
    glfw.make_context_current(window)

    # Configuración de viewport y OpenGL
    glViewport(0, 1, width, height)
    init()

    # Bucle principal
    while not glfw.window_should_close(window):
        draw_cube()
        glfw.poll_events()

    glfw.terminate()  # Cerrar GLFW al salir
    glEnd()
    glFlush()

    glfw.swap_buffers(window)  # Intercambiar buffers para animación suave
    angle += 0.1  # Incrementar el ángulo para rotación

if __name__ == "__main__":
    main()