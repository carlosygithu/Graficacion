import os
import sys
import math
import time
import random
import glfw
import cv2
import numpy as np
import mediapipe as mp
from OpenGL.GL import *
from OpenGL.GLU import *

# ============================================================
# CIUDAD SEMI-REALISTA HIGH POLY 3D
# CONTROL DE MANOS REDISEÑADO:
#   - PINCH (pulgar+índice): zoom in/out
#   - POSICIÓN X DE LA MANO: pan izquierda/derecha
#   - POSICIÓN Y DE LA MANO: pan arriba/abajo
# ============================================================

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20),
]

# ------------------ Estado cámara ------------------
angle_x, angle_y = 35.0, 45.0
zoom = -70.0
pan_x, pan_y = 0.0, 0.0
START_TIME = time.time()

# ------------------ Estado gestos ------------------
# Historial suavizado de posición de mano y pinch
_hand_pos_history = []   # últimas N posiciones normalizadas (nx, ny)
_pinch_history = []      # últimas N distancias de pinch normalizadas
_HISTORY_LEN = 6         # cuántos frames promediar (suavizado)

def elapsed():
    return time.time() - START_TIME

# ============================================================
# PRIMITIVOS (igual que antes)
# ============================================================

def clamp(v, a=0.0, b=1.0):
    return max(a, min(b, v))

def color(r, g, b):
    glColor3f(clamp(r), clamp(g), clamp(b))

def draw_cube(w, h, d, r, g, b):
    hw, hd = w / 2, d / 2
    faces = [
        ([(-hw,0, hd),( hw,0, hd),( hw,h, hd),(-hw,h, hd)], 1.00),
        ([(-hw,0,-hd),( hw,0,-hd),( hw,h,-hd),(-hw,h,-hd)], 0.75),
        ([(-hw,0,-hd),(-hw,0, hd),(-hw,h, hd),(-hw,h,-hd)], 0.65),
        ([( hw,0,-hd),( hw,0, hd),( hw,h, hd),( hw,h,-hd)], 0.65),
        ([(-hw,h,-hd),( hw,h,-hd),( hw,h, hd),(-hw,h, hd)], 0.90),
        ([(-hw,0,-hd),( hw,0,-hd),( hw,0, hd),(-hw,0, hd)], 0.55),
    ]
    glBegin(GL_QUADS)
    for verts, br in faces:
        color(r*br, g*br, b*br)
        for v in verts:
            glVertex3fv(v)
    glEnd()

def draw_cube_outline(w, h, d):
    hw, hd = w/2, d/2
    glColor3f(0.10, 0.10, 0.11)
    glLineWidth(1)
    glBegin(GL_LINES)
    pts = [(-hw,0,-hd),(hw,0,-hd),(hw,0,hd),(-hw,0,hd),
           (-hw,h,-hd),(hw,h,-hd),(hw,h,hd),(-hw,h,hd)]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for a,b in edges:
        glVertex3fv(pts[a]); glVertex3fv(pts[b])
    glEnd()

def draw_pyramid(w, h, d, r, g, b):
    hw, hd = w/2, d/2
    apex = (0,h,0)
    base = [(-hw,0,hd),(hw,0,hd),(hw,0,-hd),(-hw,0,-hd)]
    faces = [(base[0],base[1],apex,1.0),(base[1],base[2],apex,0.8),
             (base[2],base[3],apex,0.7),(base[3],base[0],apex,0.7)]
    glBegin(GL_TRIANGLES)
    for a,b2,c,br in faces:
        color(r*br,g*br,b*br)
        glVertex3fv(a); glVertex3fv(b2); glVertex3fv(c)
    glEnd()

def draw_cylinder(radius, height, r, g, b, segs=48):
    glBegin(GL_QUADS)
    for i in range(segs):
        a0 = 2*math.pi*i/segs
        a1 = 2*math.pi*(i+1)/segs
        x0,z0 = math.cos(a0)*radius, math.sin(a0)*radius
        x1,z1 = math.cos(a1)*radius, math.sin(a1)*radius
        br = 0.65 + 0.35*(i/segs)
        color(r*br, g*br, b*br)
        glVertex3f(x0,0,z0); glVertex3f(x1,0,z1)
        glVertex3f(x1,height,z1); glVertex3f(x0,height,z0)
    glEnd()

def set_material(r, g, b, shininess=35.0):
    glColor3f(clamp(r), clamp(g), clamp(b))
    try:
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, [clamp(r), clamp(g), clamp(b), 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.25, 0.25, 0.25, 1.0])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess)
    except Exception:
        pass

def draw_smooth_cylinder(radius, height, r, g, b, segs=48):
    set_material(r, g, b)
    q = gluNewQuadric()
    gluQuadricNormals(q, GLU_SMOOTH)
    glPushMatrix()
    glRotatef(-90, 1, 0, 0)
    gluCylinder(q, radius, radius, height, segs, 8)
    gluDisk(q, 0, radius, segs, 1)
    glTranslatef(0, 0, height)
    gluDisk(q, 0, radius, segs, 1)
    glPopMatrix()
    gluDeleteQuadric(q)

def draw_smooth_sphere(radius, r, g, b, slices=48, stacks=24):
    set_material(r, g, b, 45.0)
    q = gluNewQuadric()
    gluQuadricNormals(q, GLU_SMOOTH)
    gluSphere(q, radius, slices, stacks)
    gluDeleteQuadric(q)

def draw_ellipsoid(rx, ry, rz, r, g, b, slices=48, stacks=24):
    glPushMatrix()
    glScalef(rx, ry, rz)
    draw_smooth_sphere(1.0, r, g, b, slices, stacks)
    glPopMatrix()

def draw_rounded_car_body(w, h, d, r, g, b):
    draw_cube(w*0.82, h*0.72, d, r, g, b)
    glPushMatrix(); glTranslatef(0, h*0.36, d*0.42)
    draw_ellipsoid(w*0.42, h*0.34, d*0.18, r*1.02, g*1.02, b*1.02, 40, 18)
    glPopMatrix()
    glPushMatrix(); glTranslatef(0, h*0.36, -d*0.42)
    draw_ellipsoid(w*0.42, h*0.34, d*0.18, r*0.85, g*0.85, b*0.85, 40, 18)
    glPopMatrix()

def draw_glass_panel(w, h, d, r=0.14, g=0.50, b=0.75):
    draw_cube(w, h, d, r, g, b)

def draw_flat_rect(w, h, r, g, b, z=0.01):
    hw = w/2
    glBegin(GL_QUADS)
    color(r,g,b)
    glVertex3f(-hw,0,z); glVertex3f(hw,0,z)
    glVertex3f(hw,h,z); glVertex3f(-hw,h,z)
    glEnd()

# ============================================================
# SEMÁFOROS
# ============================================================

def traffic_phase():
    t = elapsed() % 18.0
    if t < 7.0:   return "EW_GREEN"
    if t < 9.0:   return "ALL_RED"
    if t < 16.0:  return "NS_GREEN"
    return "ALL_RED"

def vehicle_light_color(direction):
    phase = traffic_phase()
    if phase == "ALL_RED": return "red"
    if direction == "EW":  return "green" if phase == "EW_GREEN" else "red"
    if direction == "NS":  return "green" if phase == "NS_GREEN" else "red"
    return "red"

def pedestrian_light_for_crossing(crossing_kind):
    phase = traffic_phase()
    if crossing_kind == "horizontal_road": return "walk" if phase == "NS_GREEN" else "stop"
    if crossing_kind == "vertical_road":   return "walk" if phase == "EW_GREEN" else "stop"
    return "stop"

def draw_traffic_light(direction="EW"):
    state = vehicle_light_color(direction)
    draw_cylinder(0.09, 3.8, 0.05, 0.05, 0.06, 14)
    glPushMatrix()
    glTranslatef(0,3.55,0)
    draw_cube(0.55,1.25,0.38,0.02,0.02,0.025)
    draw_cube_outline(0.55,1.25,0.38)
    lights = [
        ("red",    0.90, (1.0,0.05,0.05), (0.20,0.0,0.0)),
        ("yellow", 0.55, (1.0,0.85,0.05), (0.25,0.18,0.0)),
        ("green",  0.20, (0.05,1.0,0.10), (0.0,0.22,0.03)),
    ]
    for name,y,on,off in lights:
        glPushMatrix()
        glTranslatef(0,y,0.205)
        if state == name:
            draw_cube(0.27,0.22,0.035,*on)
        else:
            draw_cube(0.19,0.16,0.025,*off)
        glPopMatrix()
    ped_kind = "vertical_road" if direction == "NS" else "horizontal_road"
    ped = pedestrian_light_for_crossing(ped_kind)
    glPushMatrix()
    glTranslatef(0,-0.38,0.21)
    if ped == "walk": draw_cube(0.18,0.28,0.03,0.95,0.95,0.95)
    else:             draw_cube(0.18,0.28,0.03,0.95,0.05,0.05)
    glPopMatrix()
    glPopMatrix()

def draw_crosswalk():
    glBegin(GL_QUADS)
    for i in range(-3,4):
        color(0.95,0.95,0.9)
        x0 = i * 0.65
        glVertex3f(x0,0.045,-3.0); glVertex3f(x0+0.32,0.045,-3.0)
        glVertex3f(x0+0.32,0.045,3.0); glVertex3f(x0,0.045,3.0)
    glEnd()

# ============================================================
# CIUDAD
# ============================================================

def draw_cartoon_windows(w,h,d,rows,cols):
    hw, hd = w/2, d/2
    ww = w/(cols*2.2)
    wh = h/(rows*2.5)
    z = hd + 0.018
    glBegin(GL_QUADS)
    for row in range(rows):
        for col in range(cols):
            if (row+col+int(elapsed())) % 5 == 0: color(0.12,0.16,0.25)
            elif (row+col) % 2 == 0:              color(0.78,0.70,0.35)
            else:                                  color(0.30,0.55,0.70)
            x = -hw + col*(w/cols) + ww/2
            y = row*(h/rows) + wh/2
            glVertex3f(x,y,z); glVertex3f(x+ww,y,z)
            glVertex3f(x+ww,y+wh,z); glVertex3f(x,y+wh,z)
    glEnd()

def draw_billboard(w,h,text_color=(1,1,1), bg=(1,0,0), blink=False):
    if blink and int(elapsed()*2) % 2 == 0:
        bg = (bg[0]*0.5, bg[1]*0.5, bg[2]*0.5)
    draw_flat_rect(w,h,*bg,z=0.03)
    glPushMatrix()
    glTranslatef(0,h*0.35,0.04)
    for i in range(4):
        glPushMatrix()
        glTranslatef(-w*0.32+i*w*0.21,0,0)
        draw_cube(w*0.10,h*0.16,0.02,*text_color)
        glPopMatrix()
    glPopMatrix()

def draw_tower_building(w,h,d,base_color, billboard_color):
    r,g,b = base_color
    draw_cube(w,h,d,r,g,b)
    draw_cube_outline(w,h,d)
    draw_cartoon_windows(w,h,d,max(4,int(h//2)),4)
    glPushMatrix()
    glTranslatef(0,h*0.45,d/2+0.04)
    draw_billboard(w*0.82,h*0.28,bg=billboard_color, blink=True)
    glPopMatrix()
    glPushMatrix()
    glTranslatef(-w/2-0.04,h*0.35,0)
    glRotatef(-90,0,1,0)
    draw_billboard(d*0.8,h*0.35,bg=(1.0,0.85,0.05), blink=False)
    glPopMatrix()

def draw_shop_building(w,h,d,r,g,b):
    draw_cube(w,h,d,r,g,b)
    draw_cube_outline(w,h,d)
    glPushMatrix()
    glTranslatef(0,0.7,d/2+0.03)
    draw_flat_rect(w*0.75,1.1,0.35,0.95,1.0,z=0.02)
    glPopMatrix()
    glPushMatrix()
    glTranslatef(0,2.0,d/2+0.04)
    draw_billboard(w*0.85,0.55,bg=(1.0,0.1,0.25), blink=True)
    glPopMatrix()
    if h > 3: draw_cartoon_windows(w,h,d,2,3)

def draw_ground_and_streets():
    glBegin(GL_QUADS)
    color(0.20,0.32,0.20)
    glVertex3f(-70,-0.03,70); glVertex3f(70,-0.03,70)
    glVertex3f(70,-0.03,-70); glVertex3f(-70,-0.03,-70)
    glEnd()
    street = (0.13,0.13,0.14)
    glBegin(GL_QUADS)
    color(*street)
    glVertex3f(-70,0.01,-4); glVertex3f(70,0.01,-4); glVertex3f(70,0.01,4); glVertex3f(-70,0.01,4)
    glVertex3f(-4,0.01,-70); glVertex3f(4,0.01,-70); glVertex3f(4,0.01,70); glVertex3f(-4,0.01,70)
    for z in [-35,35]:
        glVertex3f(-70,0.01,z-3); glVertex3f(70,0.01,z-3); glVertex3f(70,0.01,z+3); glVertex3f(-70,0.01,z+3)
    for x in [-35,35]:
        glVertex3f(x-3,0.01,-70); glVertex3f(x+3,0.01,-70); glVertex3f(x+3,0.01,70); glVertex3f(x-3,0.01,70)
    glEnd()
    sidewalk = (0.58,0.58,0.55)
    glBegin(GL_QUADS)
    color(*sidewalk)
    for z in [-5.2,5.2,-31.2,-38.8,31.2,38.8]:
        glVertex3f(-70,0.035,z-0.65); glVertex3f(70,0.035,z-0.65)
        glVertex3f(70,0.035,z+0.65); glVertex3f(-70,0.035,z+0.65)
    for x in [-5.2,5.2,-31.2,-38.8,31.2,38.8]:
        glVertex3f(x-0.65,0.035,-70); glVertex3f(x+0.65,0.035,-70)
        glVertex3f(x+0.65,0.035,70); glVertex3f(x-0.65,0.035,70)
    glEnd()
    glBegin(GL_QUADS)
    color(0.85,0.72,0.08)
    for x in range(-65,66,9):
        glVertex3f(x,0.05,-0.10); glVertex3f(x+4.5,0.05,-0.10)
        glVertex3f(x+4.5,0.05,0.10); glVertex3f(x,0.05,0.10)
    for z in range(-65,66,9):
        glVertex3f(-0.10,0.05,z); glVertex3f(0.10,0.05,z)
        glVertex3f(0.10,0.05,z+4.5); glVertex3f(-0.10,0.05,z+4.5)
    glEnd()
    glPushMatrix(); draw_crosswalk(); glPopMatrix()
    glPushMatrix(); glRotatef(90,0,1,0); draw_crosswalk(); glPopMatrix()

def draw_tree():
    draw_smooth_cylinder(0.32, 3.5, 0.36, 0.20, 0.10, 36)
    glPushMatrix(); glTranslatef(0, 3.45, 0)
    draw_ellipsoid(1.25, 1.05, 1.25, 0.05, 0.36, 0.12, 48, 24)
    glTranslatef(-0.75, -0.28, 0.18)
    draw_ellipsoid(0.95, 0.78, 0.95, 0.06, 0.42, 0.14, 48, 20)
    glTranslatef(1.45, 0.10, -0.30)
    draw_ellipsoid(0.90, 0.75, 0.90, 0.05, 0.40, 0.13, 48, 20)
    glTranslatef(-0.65, 0.65, 0.18)
    draw_ellipsoid(0.85, 0.72, 0.85, 0.08, 0.48, 0.16, 48, 20)
    glPopMatrix()

def draw_pine_big():
    draw_smooth_cylinder(0.24, 2.4, 0.34, 0.18, 0.08, 32)
    glPushMatrix(); glTranslatef(0, 1.55, 0)
    draw_pyramid(3.0, 2.2, 3.0, 0.04, 0.30, 0.10)
    glTranslatef(0, 1.10, 0); draw_pyramid(2.35, 1.9, 2.35, 0.05, 0.38, 0.12)
    glTranslatef(0, 0.95, 0); draw_pyramid(1.55, 1.45, 1.55, 0.06, 0.45, 0.15)
    glPopMatrix()

def draw_lamp():
    draw_smooth_cylinder(0.07, 3.7, 0.06, 0.06, 0.07, 32)
    glPushMatrix(); glTranslatef(0,3.45,0.38)
    draw_cube(0.18,0.16,0.85,0.06,0.06,0.07)
    glTranslatef(0,-0.03,0.38)
    draw_ellipsoid(0.24,0.15,0.24,1.0,0.88,0.35,32,16)
    glPopMatrix()

def draw_static_scenery():
    draw_ground_and_streets()
    buildings = [
        (-18,-18,7,22,7,(0.20,0.32,0.48),(1.0,0.05,0.15)),
        (-30,-18,6,30,6,(0.18,0.22,0.35),(1.0,0.8,0.05)),
        (-48,-18,8,18,7,(0.45,0.30,0.22),(0.05,0.55,1.0)),
        (18,-18,7,28,7,(0.25,0.25,0.38),(1.0,0.25,0.7)),
        (32,-18,6,20,6,(0.15,0.35,0.42),(0.05,1.0,0.4)),
        (50,-18,9,24,8,(0.42,0.25,0.30),(1.0,0.35,0.05)),
        (-18,18,7,26,7,(0.28,0.30,0.42),(0.2,0.8,1.0)),
        (-34,18,7,17,6,(0.35,0.25,0.32),(1.0,0.15,0.15)),
        (-50,18,9,25,7,(0.20,0.35,0.30),(0.95,0.95,0.1)),
        (18,18,7,22,7,(0.22,0.22,0.34),(0.1,0.4,1.0)),
        (34,18,6,34,6,(0.16,0.24,0.34),(1.0,0.05,0.35)),
        (52,18,8,19,8,(0.42,0.35,0.25),(0.0,0.9,0.7)),
    ]
    for x,z,w,h,d,bc,bill in buildings:
        glPushMatrix(); glTranslatef(x,0,z); draw_tower_building(w,h,d,bc,bill); glPopMatrix()
    shops = [
        (-18,48,8,4,7,0.85,0.45,0.35), (-6,48,7,4,7,0.35,0.65,0.85),
        (8,48,8,4,7,0.75,0.65,0.30),   (22,48,9,4,7,0.55,0.75,0.55),
        (48,48,8,4,7,0.85,0.55,0.75),  (-48,48,9,4,7,0.60,0.50,0.85),
        (-50,-48,9,4,7,0.85,0.55,0.25),(-36,-48,8,4,7,0.25,0.65,0.75),
        (36,-48,8,4,7,0.75,0.45,0.35), (50,-48,9,4,7,0.45,0.75,0.45),
    ]
    for x,z,w,h,d,r,g,b in shops:
        glPushMatrix(); glTranslatef(x,0,z); draw_shop_building(w,h,d,r,g,b); glPopMatrix()
    glBegin(GL_QUADS)
    color(0.12,0.50,0.18)
    glVertex3f(-68,0.025,68); glVertex3f(-42,0.025,68)
    glVertex3f(-42,0.025,42); glVertex3f(-68,0.025,42)
    glEnd()
    for x in [-64,-58,-52,-46]:
        for z in [46,52,58,64]:
            glPushMatrix(); glTranslatef(x,0,z); draw_tree(); glPopMatrix()
    extra_tree_positions = []
    for x in range(-60, 61, 12):
        extra_tree_positions += [(x,8.0),(x,-8.0),(x,42.0),(x,-42.0)]
    for z in range(-60, 61, 12):
        extra_tree_positions += [(8.0,z),(-8.0,z),(42.0,z),(-42.0,z)]
    for i,(tx,tz) in enumerate(extra_tree_positions):
        if abs(tx) < 12 and abs(tz) < 12: continue
        glPushMatrix(); glTranslatef(tx,0,tz)
        if i % 3 == 0: draw_tree()
        else:           draw_pine_big()
        glPopMatrix()
    for x,z,rot in [(-5,-5,0),(5,-5,90),(-5,5,-90),(5,5,180),
                    (-35,-35,0),(35,-35,90),(-35,35,-90),(35,35,180)]:
        glPushMatrix(); glTranslatef(x,0,z); glRotatef(rot,0,1,0)
        draw_traffic_light("EW" if abs(z) > abs(x) else "NS")
        glPopMatrix()
    for p in [(-25,5.5),(-10,5.5),(10,5.5),(25,5.5),(45,5.5),
              (-25,-5.5),(-10,-5.5),(10,-5.5),(25,-5.5),(45,-5.5),
              (5.5,-25),(5.5,-45),(-5.5,-25),(-5.5,-45),(5.5,25),(-5.5,25)]:
        glPushMatrix(); glTranslatef(p[0],0,p[1]); draw_lamp(); glPopMatrix()
    for p in [(-60,5.5),(-50,-5.5),(60,5.5),(50,-5.5),(5.5,55),(-5.5,55)]:
        glPushMatrix(); glTranslatef(p[0],0,p[1]); draw_tree(); glPopMatrix()
    t = elapsed()*0.25
    for cx,cz,sz in [(-45,-45,7),(-10,-30,5),(35,40,7),(10,15,6)]:
        x = ((cx + t*12 + 90) % 180) - 90
        for ox,oz,sc in [(0,0,1),(-sz*.45,0,.7),(sz*.45,0,.7)]:
            glPushMatrix(); glTranslatef(x+ox,32,cz+oz)
            draw_cube(sz*sc,sz*.42*sc,sz*sc,0.96,0.96,1.0)
            glPopMatrix()

# ============================================================
# LÓGICA DE TRÁFICO
# ============================================================

def point_segment_distance(px,pz,ax,az,bx,bz):
    abx,abz = bx-ax,bz-az
    apx,apz = px-ax,pz-az
    ab2 = abx*abx+abz*abz
    if ab2 <= 0.0001: return math.hypot(px-ax,pz-az)
    t = max(0.0,min(1.0,(apx*abx+apz*abz)/ab2))
    return math.hypot(px-(ax+abx*t),pz-(az+abz*t))

def segment_crosses_street(ax,az,bx,bz):
    road_lines = [0,35,-35]
    crosswalk_offsets = [-8,8,-35,35]
    for rz in road_lines:
        if (az-rz)*(bz-rz) < 0:
            x_mid = (ax+bx)/2
            if any(abs(x_mid-cw) < 2.4 for cw in crosswalk_offsets):
                return True,x_mid,rz,"horizontal_road"
    for rx in road_lines:
        if (ax-rx)*(bx-rx) < 0:
            z_mid = (az+bz)/2
            if any(abs(z_mid-cw) < 2.4 for cw in crosswalk_offsets):
                return True,rx,z_mid,"vertical_road"
    return False,None,None,None

def cars_near_crossing(cx,cz,vehicles,radius=8.5):
    return any(math.hypot(v.x-cx,v.z-cz)<radius and not getattr(v,"stopped",False) for v in (vehicles or []))

def vehicle_ahead(me,other,max_dist=9.5,side_dist=1.45):
    if other is me or getattr(other,'lane_id',None)!=getattr(me,'lane_id',None): return False
    tx,tz = me.route[me.wp]
    fx,fz = tx-me.x,tz-me.z
    flen = math.hypot(fx,fz)
    if flen < 0.001: return False
    fx,fz = fx/flen,fz/flen
    otx,otz = other.route[other.wp]
    ofx,ofz = otx-other.x,otz-other.z
    oflen = math.hypot(ofx,ofz)
    if oflen > 0.001:
        ofx,ofz = ofx/oflen,ofz/oflen
        if fx*ofx+fz*ofz < 0.75: return False
    ox,oz = other.x-me.x,other.z-me.z
    forward = ox*fx+oz*fz
    lateral = abs(ox*(-fz)+oz*fx)
    return 0.0 < forward < max_dist and lateral < side_dist

def vehicle_intersection_conflict(me,other,conflict_radius=4.2):
    if other is me: return False
    intersections = [(0,0),(35,35),(-35,35),(35,-35),(-35,-35),(0,35),(0,-35),(35,0),(-35,0)]
    tx,tz = me.route[me.wp]
    fx,fz = tx-me.x,tz-me.z
    flen = math.hypot(fx,fz)
    if flen < 0.001: return False
    fx,fz = fx/flen,fz/flen
    for ix,iz in intersections:
        mx,mz = ix-me.x,iz-me.z
        forward = mx*fx+mz*fz
        lateral = abs(mx*(-fz)+mz*fx)
        if 1.5 < forward < 10.5 and lateral < 4.3 and math.hypot(other.x-ix,other.z-iz)<conflict_radius:
            return True
    return False

def pedestrian_on_road_ahead(me,people,max_dist=9.0,side_dist=2.2):
    if not people: return False
    tx,tz = me.route[me.wp]
    fx,fz = tx-me.x,tz-me.z
    flen = math.hypot(fx,fz)
    if flen < 0.001: return False
    fx,fz = fx/flen,fz/flen
    for p in people:
        on_road = (abs(p.z)<4.6 or abs(p.x)<4.6 or abs(abs(p.z)-35)<3.8 or abs(abs(p.x)-35)<3.8)
        if not on_road: continue
        ox,oz = p.x-me.x,p.z-me.z
        if 0.0 < ox*fx+oz*fz < max_dist and abs(ox*(-fz)+oz*fx) < side_dist: return True
    return False

def approaching_red_light(vehicle,lookahead=11.0):
    if vehicle_light_color(vehicle.direction) != "red": return False
    tx,tz = vehicle.route[vehicle.wp]
    fx,fz = tx-vehicle.x,tz-vehicle.z
    flen = math.hypot(fx,fz)
    if flen < 0.001: return False
    fx,fz = fx/flen,fz/flen
    for ix,iz in [(0,0),(35,35),(-35,35),(35,-35),(-35,-35),(0,35),(0,-35),(35,0),(-35,0)]:
        ox,oz = ix-vehicle.x,iz-vehicle.z
        if 4.0 < ox*fx+oz*fz < lookahead and abs(ox*(-fz)+oz*fx) < 4.4: return True
    return False

# ============================================================
# PERSONAS
# ============================================================

class CartoonPerson:
    def __init__(self,route,speed,skin,hair,shirt,pants,scale=1.0,can_cross=False):
        self.route=route; self.wp=1
        self.x,self.z=route[0]; self.speed=speed; self.angle=0
        self.skin=skin; self.hair=hair; self.shirt=shirt; self.pants=pants
        self.scale=scale; self.can_cross=can_cross; self.waiting=False
        self.phase=random.random()*math.pi*2

    def update(self,dt,vehicles=None):
        tx,tz=self.route[self.wp]; dx,dz=tx-self.x,tz-self.z; dist=math.hypot(dx,dz)
        if dist<0.35:
            self.wp=(self.wp+1)%len(self.route); self.waiting=False; return
        crossing,cx,cz,crossing_kind=segment_crosses_street(self.x,self.z,tx,tz)
        if crossing:
            if pedestrian_light_for_crossing(crossing_kind)=="stop": self.waiting=True; return
            if cars_near_crossing(cx,cz,vehicles,radius=7.5): self.waiting=True; return
        self.waiting=False
        self.angle=math.degrees(math.atan2(dx,dz))
        step=self.speed*dt; self.x+=dx/dist*step; self.z+=dz/dist*step

    def draw(self):
        s=self.scale; swing=0 if self.waiting else math.sin(elapsed()*6+self.phase)*16
        glPushMatrix(); glTranslatef(self.x,0,self.z); glRotatef(self.angle,0,1,0); glScalef(s,s,s)
        for sx,rot in [(-0.11,swing),(0.11,-swing)]:
            glPushMatrix(); glTranslatef(sx,0.08,0); glRotatef(rot,1,0,0)
            draw_smooth_cylinder(0.075,0.46,*self.pants,segs=28)
            glTranslatef(0,0.43,0); draw_smooth_cylinder(0.085,0.38,*self.pants,segs=28)
            glTranslatef(0,-0.08,0.08); draw_ellipsoid(0.13,0.055,0.22,0.07,0.055,0.04,28,12)
            glPopMatrix()
        glPushMatrix(); glTranslatef(0,0.78,0); draw_ellipsoid(0.28,0.46,0.18,*self.shirt,40,20); glPopMatrix()
        for sx,rot in [(-0.31,-swing),(0.31,swing)]:
            glPushMatrix(); glTranslatef(sx,1.02,0); glRotatef(rot,1,0,0)
            draw_smooth_cylinder(0.055,0.42,*self.skin,segs=28)
            glTranslatef(0,0.39,0); draw_ellipsoid(0.07,0.06,0.06,*self.skin,24,12); glPopMatrix()
        glPushMatrix(); glTranslatef(0,1.22,0)
        draw_smooth_cylinder(0.065,0.15,*self.skin,segs=24)
        glTranslatef(0,0.26,0); draw_ellipsoid(0.20,0.27,0.18,*self.skin,48,24)
        glTranslatef(0,0.19,0); draw_ellipsoid(0.21,0.09,0.19,*self.hair,36,16)
        for ex in [-0.075,0.075]:
            glPushMatrix(); glTranslatef(ex,-0.08,0.17)
            draw_ellipsoid(0.025,0.018,0.012,0.02,0.02,0.02,16,8); glPopMatrix()
        glPopMatrix()
        if self.waiting:
            glPushMatrix(); glTranslatef(0,1.85,0)
            draw_ellipsoid(0.12,0.12,0.12,1.0,0.82,0.05,24,12); glPopMatrix()
        glPopMatrix()

# ============================================================
# VEHÍCULOS
# ============================================================

class CartoonVehicle:
    def __init__(self,vtype,route,speed,body,roof=(0.7,0.9,1.0),lane_id=0,direction="EW"):
        self.vtype=vtype; self.route=route; self.wp=1
        self.x,self.z=route[0]; self.speed=speed; self.base_speed=speed
        self.angle=0; self.body=body; self.roof=roof
        self.lane_id=lane_id; self.direction=direction
        self.stopped=False; self.phase=random.random()*10

    def update(self,dt,vehicles=None,people=None):
        tx,tz=self.route[self.wp]; dx,dz=tx-self.x,tz-self.z; dist=math.hypot(dx,dz)
        if dist<0.55:
            self.wp+=1
            if self.wp>=len(self.route): self.x,self.z=self.route[0]; self.wp=1
            self.stopped=False; return
        spd=self.base_speed
        if (any(vehicle_ahead(self,o) for o in (vehicles or [])) or
            any(vehicle_intersection_conflict(self,o) for o in (vehicles or [])) or
            pedestrian_on_road_ahead(self,people) or approaching_red_light(self)):
            spd=0.0
        self.stopped=spd<0.05
        self.angle=math.degrees(math.atan2(dx,dz))
        if spd<=0.0: return
        step=min(spd*dt,dist); self.x+=dx/dist*step; self.z+=dz/dist*step

    def wheel(self,x,z,radius=0.27):
        glPushMatrix(); glTranslatef(x,0,z); glRotatef(90,0,0,1)
        draw_smooth_cylinder(radius,0.20,0.015,0.015,0.018,48)
        glTranslatef(0,0,0.01); draw_smooth_cylinder(radius*0.48,0.215,0.78,0.78,0.74,40)
        glPopMatrix()

    def draw(self):
        r,g,b=self.body
        glPushMatrix(); glTranslatef(self.x,0.24,self.z); glRotatef(self.angle,0,1,0)
        if self.vtype=="taxi":
            draw_rounded_car_body(1.55,0.58,2.85,1.0,0.72,0.05)
            glPushMatrix(); glTranslatef(0,0.52,-0.08)
            draw_ellipsoid(0.58,0.36,0.55,0.12,0.48,0.70,48,18); glPopMatrix()
            glPushMatrix(); glTranslatef(0,1.02,0.05); draw_cube(0.48,0.12,0.30,1.0,0.90,0.18); glPopMatrix()
        elif self.vtype=="pickup":
            draw_rounded_car_body(1.65,0.62,2.95,r,g,b)
            glPushMatrix(); glTranslatef(0,0.58,0.62)
            draw_ellipsoid(0.55,0.38,0.45,0.12,0.45,0.62,48,18); glPopMatrix()
            glPushMatrix(); glTranslatef(0,0.52,-0.72)
            draw_cube(1.25,0.10,1.05,r*0.55,g*0.55,b*0.55); glPopMatrix()
        elif self.vtype=="bus":
            draw_cube(1.95,1.20,4.45,r,g,b)
            glPushMatrix(); glTranslatef(0,1.08,0)
            draw_ellipsoid(0.98,0.20,2.05,r*0.90,g*0.90,b*0.90,48,12); glPopMatrix()
            for z in [-1.45,-0.55,0.35,1.25]:
                glPushMatrix(); glTranslatef(0,0.78,z)
                draw_glass_panel(1.98,0.28,0.04,0.14,0.42,0.62); glPopMatrix()
        else:
            draw_rounded_car_body(1.50,0.58,2.90,r,g,b)
            glPushMatrix(); glTranslatef(0,0.55,-0.10)
            draw_ellipsoid(0.56,0.34,0.56,*self.roof,48,18); glPopMatrix()
        for x in [-0.36,0.36]:
            glPushMatrix(); glTranslatef(x,0.34,1.43)
            draw_ellipsoid(0.10,0.055,0.025,1.0,0.95,0.55,18,8); glPopMatrix()
            glPushMatrix(); glTranslatef(x,0.34,-1.43)
            draw_ellipsoid(0.11,0.060,0.025,1.0,0.02 if not self.stopped else 0.0,0.02,18,8); glPopMatrix()
        wz=1.08 if self.vtype!="bus" else 1.72
        wx=0.78 if self.vtype!="bus" else 0.95
        for x,z in [(-wx,wz),(wx,wz),(-wx,-wz),(wx,-wz)]:
            self.wheel(x,z,0.31 if self.vtype!="bus" else 0.38)
        glPopMatrix()

# ============================================================
# INSTANCIAS
# ============================================================

def offset_route(route,offset):
    return route[offset:]+route[:offset]

def build_people():
    skins = [(0.95,0.72,0.50),(0.80,0.55,0.36),(0.98,0.80,0.60),(0.62,0.40,0.26),(0.90,0.66,0.46),(0.72,0.50,0.34)]
    hairs = [(0.10,0.06,0.03),(0.75,0.35,0.08),(0.85,0.75,0.18),(0.05,0.05,0.05),(0.45,0.28,0.12),(0.68,0.68,0.62)]
    shirts= [(0.75,0.12,0.10),(0.12,0.32,0.70),(0.12,0.55,0.22),(0.92,0.58,0.12),(0.50,0.18,0.62),(0.82,0.82,0.78),(0.10,0.58,0.60),(0.58,0.14,0.26),(0.28,0.28,0.30)]
    pants = [(0.08,0.16,0.42),(0.12,0.12,0.14),(0.22,0.36,0.48),(0.40,0.26,0.14),(0.50,0.50,0.46),(0.10,0.24,0.20)]
    sidewalk_routes = [
        [(-65,5.3),(-45,5.3),(-25,5.3),(-5,5.3),(15,5.3),(35,5.3),(65,5.3)],
        [(65,-5.3),(45,-5.3),(25,-5.3),(5,-5.3),(-15,-5.3),(-35,-5.3),(-65,-5.3)],
        [(5.3,-65),(5.3,-45),(5.3,-25),(5.3,-5),(5.3,15),(5.3,35),(5.3,65)],
        [(-5.3,65),(-5.3,45),(-5.3,25),(-5.3,5),(-5.3,-15),(-5.3,-35),(-5.3,-65)],
        [(-65,38.8),(-45,38.8),(-25,38.8),(-5,38.8),(15,38.8),(35,38.8),(65,38.8)],
        [(65,31.2),(45,31.2),(25,31.2),(5,31.2),(-15,31.2),(-35,31.2),(-65,31.2)],
        [(-65,-31.2),(-45,-31.2),(-25,-31.2),(-5,-31.2),(15,-31.2),(35,-31.2),(65,-31.2)],
        [(65,-38.8),(45,-38.8),(25,-38.8),(5,-38.8),(-15,-38.8),(-35,-38.8),(-65,-38.8)],
    ]
    crossing_routes = [
        [(-8,5.3),(-8,3.0),(-8,-3.0),(-8,-5.3),(-18,-5.3),(-18,5.3)],
        [(8,-5.3),(8,-3.0),(8,3.0),(8,5.3),(18,5.3),(18,-5.3)],
        [(5.3,8),(3.0,8),(-3.0,8),(-5.3,8),(-5.3,18),(5.3,18)],
        [(-5.3,-8),(-3.0,-8),(3.0,-8),(5.3,-8),(5.3,-18),(-5.3,-18)],
    ]
    people=[]; idx=0
    for route in sidewalk_routes:
        for off in [0,2,4]:
            r=offset_route(route,off)
            people.append(CartoonPerson(r,1.0+(idx%5)*0.18,skins[idx%len(skins)],hairs[(idx*2+1)%len(hairs)],shirts[(idx*3+2)%len(shirts)],pants[(idx*5+3)%len(pants)],0.90+(idx%4)*0.045,can_cross=False))
            idx+=1
    for route in crossing_routes:
        for off in [0,3]:
            r=offset_route(route,off)
            people.append(CartoonPerson(r,0.90+(idx%4)*0.16,skins[idx%len(skins)],hairs[(idx*2+1)%len(hairs)],shirts[(idx*3+2)%len(shirts)],pants[(idx*5+3)%len(pants)],0.95,can_cross=True))
            idx+=1
    return people

def build_vehicles():
    colors=[(0.92,0.72,0.12),(0.62,0.08,0.06),(0.08,0.22,0.52),(0.10,0.38,0.18),(0.55,0.30,0.10),(0.28,0.18,0.45),(0.10,0.42,0.45),(0.78,0.78,0.76)]
    lane_specs=[
        ("EW",[(-72,-1.55),(-50,-1.55),(-25,-1.55),(-9,-1.55),(9,-1.55),(25,-1.55),(50,-1.55),(72,-1.55)]),
        ("EW",[(72,1.55),(50,1.55),(25,1.55),(9,1.55),(-9,1.55),(-25,1.55),(-50,1.55),(-72,1.55)]),
        ("NS",[(1.55,-72),(1.55,-50),(1.55,-25),(1.55,-9),(1.55,9),(1.55,25),(1.55,50),(1.55,72)]),
        ("NS",[(-1.55,72),(-1.55,50),(-1.55,25),(-1.55,9),(-1.55,-9),(-1.55,-25),(-1.55,-50),(-1.55,-72)]),
        ("EW",[(-72,33.45),(-50,33.45),(-25,33.45),(-9,33.45),(9,33.45),(25,33.45),(50,33.45),(72,33.45)]),
        ("EW",[(72,36.55),(50,36.55),(25,36.55),(9,36.55),(-9,36.55),(-25,36.55),(-50,36.55),(-72,36.55)]),
        ("EW",[(-72,-36.55),(-50,-36.55),(-25,-36.55),(-9,-36.55),(9,-36.55),(25,-36.55),(50,-36.55),(72,-36.55)]),
        ("EW",[(72,-33.45),(50,-33.45),(25,-33.45),(9,-33.45),(-9,-33.45),(-25,-33.45),(-50,-33.45),(-72,-33.45)]),
        ("NS",[(33.45,-72),(33.45,-50),(33.45,-25),(33.45,-9),(33.45,9),(33.45,25),(33.45,50),(33.45,72)]),
        ("NS",[(36.55,72),(36.55,50),(36.55,25),(36.55,9),(36.55,-9),(36.55,-25),(36.55,-50),(36.55,-72)]),
        ("NS",[(-36.55,-72),(-36.55,-50),(-36.55,-25),(-36.55,-9),(-36.55,9),(-36.55,25),(-36.55,50),(-36.55,72)]),
        ("NS",[(-33.45,72),(-33.45,50),(-33.45,25),(-33.45,9),(-33.45,-9),(-33.45,-25),(-33.45,-50),(-33.45,-72)]),
    ]
    vehicles=[]; idx=0; types=["sedan","taxi","pickup","sedan","bus"]
    for lane_id,(direction,route) in enumerate(lane_specs):
        for off in [0,4]:
            r=offset_route(route,off); vtype=types[(idx+lane_id)%len(types)]
            speed=3.8+(idx%3)*0.45
            if vtype=="bus": speed=3.0
            vehicles.append(CartoonVehicle(vtype,r,speed,colors[idx%len(colors)],lane_id=lane_id,direction=direction))
            idx+=1
    return vehicles

# ============================================================
# CONTROL DE MANOS REDISEÑADO
# ============================================================

def smooth_append(history, value, max_len=6):
    """Agrega un valor a la historia y mantiene solo los últimos max_len."""
    history.append(value)
    if len(history) > max_len:
        history.pop(0)

def smooth_average(history):
    """Promedio suavizado de la historia."""
    if not history:
        return None
    return sum(history) / len(history)

def smooth_average_2d(history):
    if not history:
        return None, None
    xs = [h[0] for h in history]
    ys = [h[1] for h in history]
    return sum(xs)/len(xs), sum(ys)/len(ys)

def compute_pinch_distance(keypoints, frame_w, frame_h):
    """Calcula distancia normalizada entre pulgar (4) e índice (8).
    Se normaliza dividiendo por el ancho de la mano para ser independiente
    de la distancia de la mano a la cámara.
    """
    thumb_tip  = keypoints[4]   # punta del pulgar
    index_tip  = keypoints[8]   # punta del índice
    wrist      = keypoints[0]   # muñeca
    middle_mcp = keypoints[9]   # nudillo del dedo medio

    # Distancia pulgar-índice en píxeles
    raw_dist = math.hypot(thumb_tip[0]-index_tip[0], thumb_tip[1]-index_tip[1])

    # Escala de referencia: tamaño de la mano (muñeca → nudillo medio)
    hand_scale = math.hypot(wrist[0]-middle_mcp[0], wrist[1]-middle_mcp[1])
    if hand_scale < 1.0:
        hand_scale = 1.0

    # Distancia normalizada: 0 = pellizcado, 1 = completamente abierto
    return min(1.0, raw_dist / (hand_scale * 1.1))

def compute_hand_center(keypoints):
    """Centro de la palma normalizado [0,1] basado en landmarks de la muñeca y nudillos."""
    palm_pts = [0, 5, 9, 13, 17]
    cx = sum(keypoints[i][0] for i in palm_pts) / len(palm_pts)
    cy = sum(keypoints[i][1] for i in palm_pts) / len(palm_pts)
    return cx, cy   # en píxeles

def draw_hand_overlay_new(frame, keypoints, pinch_dist, hand_cx, hand_cy, fw, fh):
    """Overlay visual claro: muestra el gesto actual y guías de control."""

    # Dibujar esqueleto
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, keypoints[a], keypoints[b], (100, 220, 100), 2)
    for pt in keypoints:
        cv2.circle(frame, pt, 4, (255, 200, 50), cv2.FILLED)

    # Resaltar pulgar e índice
    cv2.circle(frame, keypoints[4], 9, (0, 255, 255), cv2.FILLED)   # pulgar
    cv2.circle(frame, keypoints[8], 9, (0, 255, 255), cv2.FILLED)   # índice
    cv2.line(frame, keypoints[4], keypoints[8], (0, 255, 255), 2)   # línea pinch


    # Centro de palma con cruz de dirección
    cx_px, cy_px = int(hand_cx), int(hand_cy)
    cv2.drawMarker(frame, (cx_px, cy_px), (255, 80, 0),
                   cv2.MARKER_CROSS, 28, 2)

    # Flecha de dirección respecto al centro de la pantalla
    center_x, center_y = fw // 2, fh // 2
    dx = cx_px - center_x
    dy = cy_px - center_y
    if abs(dx) > 35 or abs(dy) > 35:
        end_x = center_x + int(dx * 0.25)
        end_y = center_y + int(dy * 0.25)
        cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y),
                        (255, 80, 0), 3, tipLength=0.35)

def process_hands_new(results, fw, fh, dt):
    """Sistema de control de manos rediseñado:

    1. PINCH (pulgar + índice):
       - Distancia pequeña (<28% del tamaño de mano) → zoom IN
       - Distancia grande (>55% del tamaño de mano) → zoom OUT
       - Intermedio → sin cambio de zoom
       La velocidad del zoom es proporcional al grado de apertura/cierre.

    2. POSICIÓN DE LA PALMA:
       - Mueve la ciudad en X e Y según la posición horizontal y vertical
         de la palma respecto al centro de la imagen.
       - Zona muerta central del 20% para evitar movimiento accidental.
       - Velocidad proporcional a la distancia del centro.
    """
    global zoom, pan_x, pan_y
    global _hand_pos_history, _pinch_history

    if not results.hand_landmarks:
        # Sin mano: resetear historiales para evitar valores obsoletos
        _hand_pos_history.clear()
        _pinch_history.clear()
        return None

    # Tomar la PRIMERA mano detectada (más estable que alternar entre manos)
    hand_lm = results.hand_landmarks[0]
    keypoints = [(int(lm.x * fw), int(lm.y * fh)) for lm in hand_lm]

    # --- Calcular pinch ---
    pinch_dist = compute_pinch_distance(keypoints, fw, fh)
    smooth_append(_pinch_history, pinch_dist, _HISTORY_LEN)
    smooth_pinch = smooth_average(_pinch_history)

    # --- Calcular posición de la palma ---
    hand_cx, hand_cy = compute_hand_center(keypoints)
    # Normalizar a [0, 1]
    nx, ny = hand_cx / fw, hand_cy / fh
    smooth_append(_hand_pos_history, (nx, ny), _HISTORY_LEN)
    snx, sny = smooth_average_2d(_hand_pos_history)

    # === ZOOM por PINCH ===
    PINCH_CLOSED = 0.22   # por debajo → acercar zoom
    PINCH_OPEN   = 0.52   # por arriba → alejar zoom
    ZOOM_SPEED   = 55.0   # unidades por segundo (sensible pero controlable)

    if smooth_pinch < PINCH_CLOSED:
        # Más cerrado = más rápido zoom in
        factor = 1.0 - (smooth_pinch / PINCH_CLOSED)  # 0..1
        zoom = min(-8.0, zoom + ZOOM_SPEED * factor * dt)
    elif smooth_pinch > PINCH_OPEN:
        # Más abierto = más rápido zoom out
        factor = (smooth_pinch - PINCH_OPEN) / (1.0 - PINCH_OPEN)  # 0..1
        zoom = max(-140.0, zoom - ZOOM_SPEED * factor * dt * 0.85)

    # === PAN por POSICIÓN DE PALMA ===
    DEAD_ZONE = 0.20    # zona muerta central: ±20% desde el centro
    PAN_SPEED  = 28.0   # unidades por segundo a máxima desviación

    off_x = snx - 0.5   # -0.5 .. +0.5
    off_y = sny - 0.5

    # Aplicar zona muerta con suavidad (rampa lineal)
    def apply_dead_zone(v, dz):
        if abs(v) < dz:
            return 0.0
        sign = 1.0 if v > 0 else -1.0
        return sign * (abs(v) - dz) / (0.5 - dz)

    pan_factor_x = apply_dead_zone(off_x, DEAD_ZONE)
    pan_factor_y = apply_dead_zone(off_y, DEAD_ZONE)

    # Invertir Y para que arriba en pantalla = acercarse en 3D
    pan_x -= pan_factor_x * PAN_SPEED * dt   # izq en pantalla → ciudad se mueve derecha
    pan_y += pan_factor_y * PAN_SPEED * dt   # arriba en pantalla → pan hacia adelante

    return keypoints, pinch_dist, hand_cx, hand_cy

# ============================================================
# MAIN
# ============================================================

def main():
    global angle_x, angle_y, zoom, pan_x, pan_y

    use_hands = os.path.exists(MODEL_PATH)
    if not use_hands:
        print("\n[AVISO] No se encontró hand_landmarker.task.")
        print("  Descárgalo desde: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        print("  y colócalo en la misma carpeta que este script.\n")

    if not glfw.init():
        sys.exit("No se pudo iniciar GLFW")

    WIN_W, WIN_H = 1280, 900
    window = glfw.create_window(WIN_W, WIN_H, "Ciudad 3D - Control de Manos", None, None)
    if not window:
        glfw.terminate()
        sys.exit("No se pudo crear la ventana")

    glfw.make_context_current(window)
    glViewport(0, 0, WIN_W, WIN_H)
    glClearColor(0.62, 0.70, 0.78, 1.0)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_LIGHTING); glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glLightfv(GL_LIGHT0, GL_POSITION, [30.0, 80.0, 40.0, 1.0])
    glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.35, 0.35, 0.35, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  [0.90, 0.90, 0.86, 1.0])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [0.35, 0.35, 0.35, 1.0])
    glEnable(GL_FOG)
    glFogfv(GL_FOG_COLOR, (GLfloat * 4)(0.62, 0.70, 0.78, 1.0))
    glFogf(GL_FOG_DENSITY, 0.006); glFogi(GL_FOG_MODE, GL_EXP2)

    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    gluPerspective(50, WIN_W/WIN_H, 0.5, 600)
    glMatrixMode(GL_MODELVIEW)

    # Mouse callbacks (igual que antes)
    mouse_drag = {"active": False, "last": (0,0), "button": -1}
    def mouse_button_cb(win, button, action, mods):
        if action == glfw.PRESS:
            mouse_drag["active"] = True; mouse_drag["button"] = button
            mouse_drag["last"] = glfw.get_cursor_pos(win)
        elif action == glfw.RELEASE:
            mouse_drag["active"] = False
    def cursor_pos_cb(win, xpos, ypos):
        global angle_x, angle_y, pan_x, pan_y
        if not mouse_drag["active"]: return
        lx, ly = mouse_drag["last"]; dx, dy = xpos-lx, ypos-ly
        mouse_drag["last"] = (xpos, ypos)
        if mouse_drag["button"] == glfw.MOUSE_BUTTON_LEFT:
            angle_y += dx*0.35; angle_x = max(5, min(88, angle_x + dy*0.35))
        elif mouse_drag["button"] == glfw.MOUSE_BUTTON_RIGHT:
            pan_x += dx*0.08; pan_y -= dy*0.08
    def scroll_cb(win, xoff, yoff):
        global zoom
        zoom = max(-140, min(-8, zoom + yoff*3))
    glfw.set_mouse_button_callback(window, mouse_button_cb)
    glfw.set_cursor_pos_callback(window, cursor_pos_cb)
    glfw.set_scroll_callback(window, scroll_cb)

    people   = build_people()
    vehicles = build_vehicles()

    cap = None; landmarker = None
    if use_hands:
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1,                          # 1 mano = más velocidad y estabilidad
            min_hand_detection_confidence=0.60,
            min_hand_presence_confidence=0.60,
            min_tracking_confidence=0.55,
        )
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        landmarker = HandLandmarker.create_from_options(options)

    print("=" * 55)
    print("  CIUDAD 3D ")
    print("=" * 55)
    print("  GESTOS:")
    print("    Pulgar + índice JUNTOS  → zoom IN  (acercar)")
    print("    Pulgar + índice ABIERTOS→ zoom OUT (alejar)")
    print("    Mano a la IZQUIERDA    → ciudad se mueve")
    print("    Mano a la DERECHA      → ciudad se mueve")
    print("    Mano ARRIBA / ABAJO    → pan vertical")
    print("")
    print("  TECLADO:")
    print("    WASD / Flechas → mover vista")
    print("    Q / E          → zoom")
    print("    Mouse izq      → rotar cámara")
    print("    Mouse der      → mover ciudad")
    print("    Rueda scroll   → zoom")
    print("    ESC            → salir")
    print("=" * 55)

    last = time.time()

    while not glfw.window_should_close(window):
        now = time.time(); dt = min(now-last, 0.05); last = now

        spd = 18*dt
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS: break
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS or glfw.get_key(window, glfw.KEY_UP)    == glfw.PRESS: pan_y += spd
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS or glfw.get_key(window, glfw.KEY_DOWN)  == glfw.PRESS: pan_y -= spd
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS or glfw.get_key(window, glfw.KEY_LEFT)  == glfw.PRESS: pan_x -= spd
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS: pan_x += spd
        if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS: zoom = max(-140, zoom - spd*2)
        if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS: zoom = min(-8,   zoom + spd*2)

        for v in vehicles: v.update(dt, vehicles=vehicles, people=people)
        for p in people:   p.update(dt, vehicles=vehicles)

        # --- Procesamiento de manos ---
        if use_hands and cap:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                fh_cam, fw_cam, _ = frame.shape
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                results = landmarker.detect(mp_img)
                hand_data = process_hands_new(results, fw_cam, fh_cam, dt)

                if hand_data:
                    kpts, pdist, hcx, hcy = hand_data
                    draw_hand_overlay_new(frame, kpts, pdist, hcx, hcy, fw_cam, fh_cam)

                # HUD de instrucciones en la cámara
                cv2.rectangle(frame, (0, fh_cam-48), (fw_cam, fh_cam), (0,0,0), -1)
                # tText(frame,
                  #  "Pinch pulgar+indice = ZOOM  |  Mueve la palma = MOVER CIUDAD",
                  #  (8, fh_cam-14), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180,255,180), 1)
#
                # Indicador de zoom actual
                #zoom_pct = int(((-zoom) - 8) / (140 - 8) * 100)
               # cv2.putText(frame, f"Zoom: {zoom_pct}%",
                 #   (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,100), 2)

                cv2.imshow("Control de Manos", frame)
                cv2.waitKey(1)

        # --- Render OpenGL ---
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        cam_z = -zoom
        gluLookAt(pan_x, cam_z*0.62 + pan_y, cam_z,
                  pan_x, 2.0 + pan_y, 0.0,
                  0.0, 1.0, 0.0)
        glRotatef(angle_x, 1, 0, 0)
        glRotatef(angle_y, 0, 1, 0)

        draw_static_scenery()
        for v in vehicles: v.draw()
        for p in people:   p.draw()

        glfw.swap_buffers(window)
        glfw.poll_events()

    if cap: cap.release()
    if landmarker: landmarker.close()
    cv2.destroyAllWindows()
    glfw.terminate()

if __name__ == "__main__":
    main()