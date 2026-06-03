from __future__ import annotations
import math, os, sys
from pathlib import Path

import cv2
import glfw
import numpy as np
import mediapipe as mp
from OpenGL.GL import *
from OpenGL.GLU import *

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel",      "3")

# ── Configuración ArUco ──────────────────────────────────────────────────────
CAMERA_INDEX    = 0
MARKER_LENGTH_M = 0.10
ARUCO_DICT      = cv2.aruco.DICT_4X4_50
MARKER_ID       = 0
ZNear, ZFar     = 0.001, 200.0

SCRIPT_DIR = Path(__file__).resolve().parent
CALIB_NPZ  = SCRIPT_DIR / "camera_ar.npz"
MODEL_PATH = str(SCRIPT_DIR / "hand_landmarker.task")

# ── Estado gestual ───────────────────────────────────────────────────────────
city_yaw   =   0.0
city_pitch =  30.0
city_zoom  =   1.0
_prev_index = None
_prev_pinch = None

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),
    (15,16),(13,17),(0,17),(17,18),(18,19),(19,20),
]

# ── MediaPipe ────────────────────────────────────────────────────────────────
BaseOptions        = mp.tasks.BaseOptions
HandLandmarker     = mp.tasks.vision.HandLandmarker
HandLandmarkerOpts = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode  = mp.tasks.vision.RunningMode

# ── Calibración ──────────────────────────────────────────────────────────────
def default_camera_matrix(w, h):
    f = float(max(w, h))
    return np.array([[f,0,w/2.],[0,f,h/2.],[0,0,1]], dtype=np.float64)

def load_calibration(w, h):
    if CALIB_NPZ.is_file():
        d = np.load(CALIB_NPZ)
        return d["camera_matrix"], d["dist_coeffs"]
    return default_camera_matrix(w, h), np.zeros((5,1), dtype=np.float64)

# ── ArUco ────────────────────────────────────────────────────────────────────
def make_aruco_detector():
    dic    = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dic, params), dic
    return None, dic

def detect_marker(gray, detector, dictionary):
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=cv2.aruco.DetectorParameters())
    if ids is None or len(ids) == 0:
        return None
    matches = np.where(ids.flatten() == MARKER_ID)[0]
    if len(matches) == 0:
        return None
    return corners[int(matches[0])]

def marker_object_points(side):
    s = side / 2.0
    return np.array([[-s,s,0],[s,s,0],[s,-s,0],[-s,-s,0]], dtype=np.float32)

def estimate_pose(corners, K, dist):
    pts2d = np.asarray(
        corners[0] if corners.ndim == 3 else corners,
        dtype=np.float32).reshape(-1, 2)
    obj   = marker_object_points(MARKER_LENGTH_M)
    flags = (cv2.SOLVEPNP_IPPE_SQUARE
             if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE")
             else cv2.SOLVEPNP_ITERATIVE)
    ok, rvec, tvec = cv2.solvePnP(obj, pts2d, K, dist, flags=flags)
    if not ok:
        raise RuntimeError("solvePnP falló")
    return rvec, tvec

# ── Matrices OpenGL ───────────────────────────────────────────────────────────
def projection_from_k(K, w, h, znear, zfar):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    P = np.zeros((4,4), dtype=np.float32)
    P[0,0] =  2.*fx/w;  P[1,1] =  2.*fy/h
    P[0,2] =  (w-2.*cx)/w;  P[1,2] = (2.*cy-h)/h
    P[2,2] = -(zfar+znear)/(zfar-znear);  P[2,3] = -1.
    P[3,2] = -2.*zfar*znear/(zfar-znear)
    return P

def modelview_from_pose(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    M    = np.eye(4, dtype=np.float64)
    M[:3,:3] = R;  M[:3,3] = tvec.flatten()
    return (np.diag([1.,-1.,-1.,1.]) @ M).T.astype(np.float32)

# ── Textura fondo webcam ──────────────────────────────────────────────────────
_tex_id = None; _tex_buf = None

def upload_frame_texture(frame_bgr, w, h):
    global _tex_id, _tex_buf
    rgb = cv2.flip(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), 0)
    if _tex_buf is None or _tex_buf.shape[:2] != (h, w):
        _tex_buf = np.empty((h, w, 3), dtype=np.uint8)
    np.copyto(_tex_buf, rgb)
    if _tex_id is None:
        _tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, _tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,w,h,0,GL_RGB,GL_UNSIGNED_BYTE,_tex_buf)

def draw_background_quad(w, h):
    glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
    glOrtho(0,w,0,h,-1,1)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
    glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, _tex_id)
    glColor3f(1,1,1)
    glBegin(GL_QUADS)
    glTexCoord2f(0,0); glVertex2f(0,0);  glTexCoord2f(1,0); glVertex2f(w,0)
    glTexCoord2f(1,1); glVertex2f(w,h);  glTexCoord2f(0,1); glVertex2f(0,h)
    glEnd()
    glDisable(GL_TEXTURE_2D)
    glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix()
    glMatrixMode(GL_MODELVIEW); glEnable(GL_DEPTH_TEST)

# ════════════════════════════════════════════════════════════════════════════
#  PRIMITIVAS DE LA CIUDAD
# ════════════════════════════════════════════════════════════════════════════
def draw_generic_cube(w, h, d, r, g, b):
    w_h, d_h = w/2., d/2.
    glBegin(GL_QUADS)
    glColor3f(r,g,b)
    glVertex3f(-w_h,0,d_h);  glVertex3f(w_h,0,d_h);  glVertex3f(w_h,h,d_h);  glVertex3f(-w_h,h,d_h)
    glColor3f(r*.8,g*.8,b*.8)
    glVertex3f(-w_h,0,-d_h); glVertex3f(w_h,0,-d_h); glVertex3f(w_h,h,-d_h); glVertex3f(-w_h,h,-d_h)
    glColor3f(r*.7,g*.7,b*.7)
    glVertex3f(-w_h,0,-d_h); glVertex3f(-w_h,0,d_h); glVertex3f(-w_h,h,d_h); glVertex3f(-w_h,h,-d_h)
    glVertex3f( w_h,0,-d_h); glVertex3f( w_h,0,d_h); glVertex3f( w_h,h,d_h); glVertex3f( w_h,h,-d_h)
    glColor3f(r*.9,g*.9,b*.9)
    glVertex3f(-w_h,h,-d_h); glVertex3f(w_h,h,-d_h); glVertex3f(w_h,h,d_h); glVertex3f(-w_h,h,d_h)
    glEnd()

def draw_pyramid(w, h, d, r, g, b):
    w_h, d_h = w/2., d/2.
    glBegin(GL_TRIANGLES)
    glColor3f(r,g,b)
    glVertex3f(-w_h,0,d_h);  glVertex3f(w_h,0,d_h);  glVertex3f(0,h,0)
    glColor3f(r*.8,g*.8,b*.8)
    glVertex3f(-w_h,0,-d_h); glVertex3f(w_h,0,-d_h); glVertex3f(0,h,0)
    glColor3f(r*.7,g*.7,b*.7)
    glVertex3f(-w_h,0,-d_h); glVertex3f(-w_h,0,d_h); glVertex3f(0,h,0)
    glVertex3f( w_h,0,-d_h); glVertex3f( w_h,0,d_h); glVertex3f(0,h,0)
    glEnd()

def draw_windows(w, h, d, rows, cols):
    w_half, d_half = w/2., d/2.
    win_w = w/(cols*2); win_h = h/(rows*2)
    glColor3f(0.95,0.95,0.4)
    glBegin(GL_QUADS)
    z_f = d_half+0.01
    for r in range(rows):
        for c in range(cols):
            if (r+c)%3==0: continue
            x = -w_half+(c*(w/cols))+win_w/2
            y = (r*(h/rows))+win_h/2
            glVertex3f(x,y,z_f); glVertex3f(x+win_w,y,z_f)
            glVertex3f(x+win_w,y+win_h,z_f); glVertex3f(x,y+win_h,z_f)
    glEnd()

def draw_detailed_house(w, h, d, r, g, b):
    draw_generic_cube(w,h,d,r,g,b)
    glPushMatrix(); glTranslatef(0,0.3,0); draw_windows(w*.8,h*.6,d,2,2); glPopMatrix()
    w_h, d_h = (w+.2)/2., (d+.2)/2.; roof_y = h+.8
    glBegin(GL_TRIANGLES)
    glColor3f(.85,.35,.1)
    glVertex3f(-w_h,h,d_h); glVertex3f(w_h,h,d_h); glVertex3f(0,roof_y,0)
    glVertex3f(-w_h,h,-d_h); glVertex3f(w_h,h,-d_h); glVertex3f(0,roof_y,0)
    glEnd()

def draw_cafe():
    draw_generic_cube(4.,2.5,3.5,.75,.6,.45)
    glBegin(GL_QUADS)
    glColor3f(.9,.9,.5)
    glVertex3f(-1.5,.5,1.76); glVertex3f(1.5,.5,1.76); glVertex3f(1.5,1.8,1.76); glVertex3f(-1.5,1.8,1.76)
    glEnd()
    glPushMatrix(); glTranslatef(0,2.3,.3)
    draw_generic_cube(4.4,.2,4.,.4,.25,.15)
    glTranslatef(0,.5,1.6); draw_generic_cube(2.,.5,.1,.9,.85,.7)
    glPopMatrix()

def draw_traffic_light(is_green):
    draw_generic_cube(.15,3.5,.15,.25,.25,.25)
    glPushMatrix(); glTranslatef(0,3.5,0)
    draw_generic_cube(.4,1.,.4,.1,.1,.1)
    r_l = 1. if not is_green else .15
    g_l = 1. if is_green     else .15
    glPushMatrix(); glTranslatef(0,.25,.21);  draw_generic_cube(.2,.2,.02,r_l,0.,0.);  glPopMatrix()
    glPushMatrix(); glTranslatef(0,-.25,.21); draw_generic_cube(.2,.2,.02,0.,g_l,0.);  glPopMatrix()
    glPopMatrix()

def draw_pink_motorcycle():
    glPushMatrix()
    draw_generic_cube(.25,.4,1.,1.,.4,.7)
    glPushMatrix(); glTranslatef(0,-.05,.4);  draw_generic_cube(.12,.25,.25,.12,.12,.12); glPopMatrix()
    glPushMatrix(); glTranslatef(0,-.05,-.4); draw_generic_cube(.12,.25,.25,.12,.12,.12); glPopMatrix()
    glPushMatrix(); glTranslatef(0,.4,.1);    draw_generic_cube(.2,.12,.3,.2,.2,.2);      glPopMatrix()
    glPopMatrix()

def draw_school():
    draw_generic_cube(9.,4.5,4.5,.7,.7,.7)
    glPushMatrix(); glTranslatef(2.5,0,3.); draw_generic_cube(3.,4.5,3.,.65,.65,.65); glPopMatrix()
    draw_windows(8.,3.5,4.5,2,4)

def draw_church():
    draw_generic_cube(5.,5.5,8.,.85,.82,.75)
    glPushMatrix(); glTranslatef(0,0,3.2)
    draw_generic_cube(2.5,10.,2.5,.75,.72,.65)
    glTranslatef(0,10.,0); draw_pyramid(2.8,2.5,2.8,.3,.3,.35)
    glTranslatef(0,2.5,0); draw_generic_cube(.15,1.,.15,.9,.8,.2)
    glTranslatef(0,.3,0);  draw_generic_cube(.6,.15,.15,.9,.8,.2)
    glPopMatrix()

def draw_small_playground():
    glBegin(GL_QUADS)
    glColor3f(.35,.65,.3)
    glVertex3f(-4.5,.02,-4.5); glVertex3f(4.5,.02,-4.5)
    glVertex3f(4.5,.02,4.5);   glVertex3f(-4.5,.02,4.5)
    glEnd()
    glPushMatrix(); glTranslatef(-2.,0,-1.)
    draw_generic_cube(.1,1.8,.1,.2,.2,.2)
    glTranslatef(2.5,0,0); draw_generic_cube(.1,1.8,.1,.2,.2,.2)
    glTranslatef(-1.25,1.8,0); draw_generic_cube(2.7,.1,.1,.2,.2,.2)
    glTranslatef(0,-1.,0); draw_generic_cube(.8,.08,.3,.8,.2,.2)
    glPopMatrix()
    glPushMatrix(); glTranslatef(1.8,0,-1.5)
    draw_generic_cube(.6,1.4,.6,.2,.4,.8)
    glPushMatrix(); glTranslatef(0,.5,.8); glRotatef(30,1,0,0)
    draw_generic_cube(.5,.1,1.6,.8,.8,.8); glPopMatrix()
    glPopMatrix()

def draw_kiosk():
    glPushMatrix()
    draw_generic_cube(2.6,.5,2.6,.5,.35,.25)
    for sx in [-1.1,1.1]:
        for sz in [-1.1,1.1]:
            glPushMatrix(); glTranslatef(sx,.5,sz)
            draw_generic_cube(.12,1.8,.12,.8,.7,.5); glPopMatrix()
    glTranslatef(0,2.3,0); draw_pyramid(3.,1.2,3.,.7,.2,.2)
    glPopMatrix()

def draw_dog():
    glPushMatrix()
    draw_generic_cube(.25,.25,.5,.55,.27,.07)
    glPushMatrix()
    for ox in [-.08,.08]:
        for oz in [-.18,.18]:
            glPushMatrix(); glTranslatef(ox,-.12,oz)
            draw_generic_cube(.06,.15,.06,.4,.2,.0); glPopMatrix()
    glPopMatrix()
    glTranslatef(0,.2,.2); draw_generic_cube(.2,.2,.2,.55,.27,.07)
    glPopMatrix()

def draw_car(r, g, b):
    glPushMatrix()
    draw_generic_cube(1.,.38,1.8,r,g,b)
    glPushMatrix(); glTranslatef(0,.38,-.1)
    draw_generic_cube(.8,.32,1.,r*.6,g*.6,b*.6); glPopMatrix()
    glPopMatrix()

def draw_truck():
    glPushMatrix()
    draw_generic_cube(1.4,1.6,3.6,.85,.85,.85)
    glTranslatef(0,0,1.3); draw_generic_cube(1.3,1.,1.,.8,.1,.1)
    glPopMatrix()

def draw_volleyball_court():
    glBegin(GL_QUADS)
    glColor3f(.9,.5,.2)
    glVertex3f(-3.5,.02,-5.5); glVertex3f(3.5,.02,-5.5)
    glVertex3f(3.5,.02,5.5);   glVertex3f(-3.5,.02,5.5)
    glColor3f(.1,.4,.7)
    glVertex3f(-2.8,.025,-4.8); glVertex3f(2.8,.025,-4.8)
    glVertex3f(2.8,.025,4.8);   glVertex3f(-2.8,.025,4.8)
    glEnd()
    draw_generic_cube(.08,1.8,.08,.6,.6,.6)
    glPushMatrix(); glTranslatef(0,1.1,0); draw_generic_cube(5.4,.5,.02,.9,.9,.9); glPopMatrix()

def draw_basketball_court():
    glBegin(GL_QUADS)
    glColor3f(.75,.52,.3)
    glVertex3f(-3.5,.02,-6.); glVertex3f(3.5,.02,-6.)
    glVertex3f(3.5,.02,6.);   glVertex3f(-3.5,.02,6.)
    glEnd()
    glPushMatrix(); glTranslatef(0,0,-5.6)
    draw_generic_cube(.12,2.5,.12,.2,.2,.2)
    glTranslatef(0,2.5,.15); draw_generic_cube(1.5,.9,.04,1.,1.,1.); glPopMatrix()
    glPushMatrix(); glTranslatef(0,0,5.6)
    draw_generic_cube(.12,2.5,.12,.2,.2,.2)
    glTranslatef(0,2.5,-.15); draw_generic_cube(1.5,.9,.04,1.,1.,1.); glPopMatrix()

def draw_tree_round():
    draw_generic_cube(.2,.8,.2,.4,.25,.15)
    glPushMatrix(); glTranslatef(0,.8,0); draw_generic_cube(1.,.9,1.,.2,.55,.2); glPopMatrix()

def draw_tree_pine():
    draw_generic_cube(.2,.6,.2,.4,.25,.15)
    glPushMatrix()
    glTranslatef(0,.5,0); draw_pyramid(1.2,.8,1.2,.1,.38,.15)
    glTranslatef(0,.5,0); draw_pyramid(.9,.7,.9,.12,.42,.18)
    glPopMatrix()

def draw_animated_helicopter(t):
    glPushMatrix()
    draw_generic_cube(1.4,1.,3.,.2,.2,.8)
    glPushMatrix(); glTranslatef(0,.2,-2.); draw_generic_cube(.3,.3,1.5,.2,.2,.8); glPopMatrix()
    glPushMatrix(); glTranslatef(0,1.1,0); glRotatef(t*800,0,1,0)
    draw_generic_cube(4.,.04,.25,.9,.9,.9); glPopMatrix()
    glPopMatrix()

def draw_scenery(t):
    # Suelo
    glBegin(GL_QUADS)
    glColor3f(.14,.14,.15)
    glVertex3f(-55,-.01,55); glVertex3f(55,-.01,55)
    glVertex3f(55,-.01,-55); glVertex3f(-55,-.01,-55)
    glEnd()

    # Manzanas
    manzanas = [
        (-32.,-30.,24.,24.),(-32.,5.,24.,20.),(-32.,38.,24.,22.),
        (0.,-30.,30.,24.),  (0.,5.,30.,20.),  (0.,38.,30.,22.),
        (35.,-30.,30.,24.), (35.,5.,30.,20.),  (35.,38.,30.,22.)
    ]
    glBegin(GL_QUADS)
    for mx,mz,mw,md in manzanas:
        glColor3f(.08,.08,.09)
        wh,dh = mw/2.,md/2.
        glVertex3f(mx-wh,.005,mz+dh); glVertex3f(mx+wh,.005,mz+dh)
        glVertex3f(mx+wh,.005,mz-dh); glVertex3f(mx-wh,.005,mz-dh)
    glEnd()

    # Semáforos animados
    traffic_phase   = int(t/6.)%2
    vertical_green  = (traffic_phase==0)
    horizontal_green= not vertical_green
    glPushMatrix(); glTranslatef(-17.,0,14.);  draw_traffic_light(vertical_green);  glPopMatrix()
    glPushMatrix(); glTranslatef(-13.,0,17.5); glRotatef(90,0,1,0); draw_traffic_light(horizontal_green); glPopMatrix()

    # Parque
    glPushMatrix(); glTranslatef(-32.,0,5.)
    glBegin(GL_QUADS); glColor3f(.2,.45,.22)
    glVertex3f(-11.5,.01,9.5); glVertex3f(11.5,.01,9.5)
    glVertex3f(11.5,.01,-9.5); glVertex3f(-11.5,.01,-9.5); glEnd()
    glPushMatrix(); glTranslatef(-5.5,0,-3.);  draw_volleyball_court(); glPopMatrix()
    glPushMatrix(); glTranslatef(-5.5,0,4.5);  draw_small_playground(); glPopMatrix()
    glPushMatrix(); glTranslatef(-1.,.2,2.5);  draw_dog();              glPopMatrix()
    glPushMatrix(); glTranslatef(4.5,0,0.);    draw_kiosk();            glPopMatrix()
    for az in range(-8,9,4):
        glPushMatrix(); glTranslatef(-10.,0,az); draw_tree_round(); glPopMatrix()
        glPushMatrix(); glTranslatef(9.5,0,az);  draw_tree_pine();  glPopMatrix()
    glPopMatrix()

    # Zona comunitaria
    glPushMatrix(); glTranslatef(0.,0,-30.)
    glPushMatrix(); glTranslatef(-7.,0,-4.); draw_school();            glPopMatrix()
    glPushMatrix(); glTranslatef(1.,0,2.);   draw_church();            glPopMatrix()
    glPushMatrix(); glTranslatef(9.,0,-2.);  draw_basketball_court();  glPopMatrix()
    glPopMatrix()

    # Rascacielos
    for x,z,w,h,d,r,g,b in [
        (-32.,-30.,4.5,18.,4.5,.25,.35,.5),(-25.,-26.,4.,14.,4.,.3,.3,.35),
        (0.,38.,5.,26.,5.,.15,.4,.45),(8.,42.,4.5,22.,4.5,.2,.2,.3),
        (-8.,35.,4.,17.,4.,.35,.35,.4),(35.,38.,5.5,29.,5.5,.1,.25,.5),
        (42.,44.,4.,20.,4.,.22,.45,.4),(28.,34.,4.2,15.,4.2,.4,.4,.45),
    ]:
        glPushMatrix(); glTranslatef(x,0,z)
        draw_generic_cube(w,h,d,r,g,b); draw_windows(w,h,d,int(h//1.5),4)
        glPopMatrix()

    # Casas
    house_positions = [
        (-38.,32.),(-32.,32.),(-26.,32.),(-35.,42.),(-29.,42.),
        (-10.,2.),(-4.,2.),(-10.,9.),(-4.,9.),
        (22.,1.),(28.,1.),(34.,1.),(25.,9.),(31.,9.),
        (24.,-35.),(30.,-35.),(38.,-35.),(32.,-25.)
    ]
    house_colors = [(.85,.45,.45),(.45,.65,.85),(.55,.75,.55),(.85,.80,.55),(.75,.60,.80)]
    for idx,(hx,hz) in enumerate(house_positions):
        rc,gc,bc = house_colors[idx%len(house_colors)]
        glPushMatrix(); glTranslatef(hx,0,hz)
        draw_detailed_house(2.4,2.,2.4,rc,gc,bc); glPopMatrix()

    glPushMatrix(); glTranslatef(4.,0,5.); draw_cafe(); glPopMatrix()

    # Tráfico vertical
    for idx, x_lane in enumerate([-15.,15.,49.]):
        sp = 9.+(idx%2)*3.
        for off in [0.,50.,25.]:
            z1 = -50.+((t*sp+off)%100.)
            if x_lane==-15. and not vertical_green and 4.<z1<16.: z1=4.
            glPushMatrix(); glTranslatef(x_lane-.7,.15,z1); draw_car(.85,.1,.1); glPopMatrix()
        z2 = -50.+((t*sp+50.)%100.)
        glPushMatrix(); glTranslatef(x_lane+.7,.15,z2); draw_truck(); glPopMatrix()
        z3 = -50.+((t*sp+25.)%100.)
        glPushMatrix(); glTranslatef(x_lane,.15,z3); draw_pink_motorcycle(); glPopMatrix()

    # Tráfico horizontal
    for idx, z_lane in enumerate([-18.,16.,49.]):
        sp = 10.+(idx%2)*3.
        for off in [0.,50.,75.]:
            x1 = -50.+((t*sp+off)%100.)
            if z_lane==16. and not horizontal_green and -27.<x1<-15.: x1=-27.
            glPushMatrix(); glTranslatef(x1,.15,z_lane-.7); glRotatef(90,0,1,0); draw_car(.9,.8,.1); glPopMatrix()

    # Helicóptero
    glPushMatrix()
    glRotatef(t*22,0,1,0); glTranslatef(15.,24.,10.)
    draw_animated_helicopter(t)
    glPopMatrix()

# ── Iluminación ──────────────────────────────────────────────────────────────
def setup_lighting():
    glEnable(GL_LIGHTING); glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glLightfv(GL_LIGHT0, GL_POSITION, (.5, 1., .5, 0.))
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  (1.,1.,.95,1.))
    glLightfv(GL_LIGHT0, GL_AMBIENT,  (.3,.3,.3,1.))
    glEnable(GL_NORMALIZE)

# ── Escena 3D Anclada Al Marcador + Reflejo Efecto Lupa ──────────────────────
def draw_scene_3d(rvec, tvec, K, w, h, t):
    P  = projection_from_k(K, w, h, ZNear, ZFar)
    MV = modelview_from_pose(rvec, tvec)

    glMatrixMode(GL_PROJECTION); glLoadMatrixf(P)
    glMatrixMode(GL_MODELVIEW);  glLoadIdentity()
    glMultMatrixf(MV)

    # Extraemos la distancia 'z' real entre la cámara y el marcador ArUco
    camera_distance = float(tvec[2][0])
    
    # Efecto Lupa: Aumenta la geometría si nos aproximamos a menos de 45cm
    lens_zoom_effect = 1.0
    if camera_distance < 0.45:
        lens_zoom_effect = 1.0 + (0.45 - camera_distance) * 4.5

    # Rotación + zoom gestual de la mano
    glRotatef(city_pitch, 1, 0, 0)
    glRotatef(city_yaw,   0, 1, 0)
    glScalef(city_zoom, city_zoom, city_zoom)

    # Aplicamos el aumento dinámico por software
    glScalef(lens_zoom_effect, lens_zoom_effect, lens_zoom_effect)

    # Escalar ciudad (~110 unidades) para que quepa sobre el marcador (~0.1 m)
    s = 0.0008
    glScalef(s, s, s)

    setup_lighting()
    draw_scenery(t)

# ── Interfaz HUD De Cámara Y Guías ───────────────────────────────────────────
def draw_camera_ui(frame, w, h, tvec):
    """Genera un visor óptico en el plano 2D para denotar el aumento por reflejo"""
    if tvec is None: return
    dist = float(tvec[2][0])
    
    length = 30
    color = (0, 255, 0) if dist >= 0.45 else (0, 165, 255) # Naranja si el zoom se activa
    
    # Esquina Superior Izquierda
    cv2.line(frame, (40, 40), (40 + length, 40), color, 2)
    cv2.line(frame, (40, 40), (40, 40 + length), color, 2)
    # Esquina Superior Derecha
    cv2.line(frame, (w - 40, 40), (w - 40 - length, 40), color, 2)
    cv2.line(frame, (w - 40, 40), (w - 40, 40 + length), color, 2)
    # Esquina Inferior Izquierda
    cv2.line(frame, (40, h - 40), (40 + length, h - 40), color, 2)
    cv2.line(frame, (40, h - 40), (40, h - 40 - length), color, 2)
    # Esquina Inferior Derecha
    cv2.line(frame, (w - 40, h - 40), (w - 40 - length, h - 40), color, 2)
    cv2.line(frame, (w - 40, h - 40), (w - 40, h - 40 - length), color, 2)
    
   
# ── Capa Gestual MediaPipe ───────────────────────────────────────────────────
def draw_hand_overlay(frame, hand_landmarks, fw, fh):
    kps = [(int(lm.x*fw), int(lm.y*fh)) for lm in hand_landmarks]
    for a,b in HAND_CONNECTIONS:
        cv2.line(frame, kps[a], kps[b], (0,255,0), 2)
    for pt in kps:
        cv2.circle(frame, pt, 4, (0,120,255), cv2.FILLED)
    

def process_gestures(results, fw, fh):
    global city_yaw, city_pitch, city_zoom, _prev_index, _prev_pinch
    if not results or not results.hand_landmarks:
        _prev_index = None; _prev_pinch = None; return
    hand_lm = results.hand_landmarks[0]
    kps     = [(lm.x, lm.y) for lm in hand_lm]
    if len(kps) < 21: return
    index_tip = kps[8]; thumb_tip = kps[4]
    pinch = math.hypot(index_tip[0]-thumb_tip[0], index_tip[1]-thumb_tip[1])
    if _prev_index is not None:
        dx = (index_tip[0]-_prev_index[0])*250.
        dy = (index_tip[1]-_prev_index[1])*250.
        city_yaw  += dx
        city_pitch = max(-85., min(85., city_pitch+dy))
    if _prev_pinch is not None:
        city_zoom = max(.2, min(5., city_zoom-(pinch-_prev_pinch)*3.))
    _prev_index = index_tip; _prev_pinch = pinch

# ── Main Loop ────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: falta '{MODEL_PATH}'")
        print("Descárgalo de: https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task\n")
        sys.exit(1)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.", file=sys.stderr); sys.exit(1)

    ret, probe = cap.read()
    if not ret: sys.exit(1)
    cam_h, cam_w = probe.shape[:2]

    K, dist         = load_calibration(cam_w, cam_h)
    detector, dictn = make_aruco_detector()

    if not glfw.init(): sys.exit(1)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    window = glfw.create_window(
        cam_w, cam_h,
        "RA Ciudad | Lente de Camara Magnificado Activo | ESC=salir",
        None, None)
    if not window:
        glfw.terminate(); sys.exit(1)

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    def on_key(win, key, _sc, action, _mods):
        if action==glfw.PRESS and key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
            glfw.set_window_should_close(win, True)

    glfw.set_key_callback(window, on_key)
    glEnable(GL_DEPTH_TEST)

    opts = HandLandmarkerOpts(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
    )

    with HandLandmarker.create_from_options(opts) as landmarker:
        while not glfw.window_should_close(window):
            ret, frame = cap.read()
            if not ret: continue

            h, w = frame.shape[:2]
            t    = glfw.get_time()

            # 1) Detectar ArUco en frame ORIGINAL
            gray_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners_orig = detect_marker(gray_orig, detector, dictn)

            # 2) Flip para display y mano
            frame = cv2.flip(frame, 1)

            # 3) Si hay marcador, calcular pose en espejo
            rvec, tvec = None, None
            if corners_orig is not None:
                corners_flip = corners_orig.copy()
                corners_flip[..., 0] = w - corners_orig[..., 0]
                rvec, tvec = estimate_pose(corners_flip, K, dist)

            # 4) Deteccion mano sobre frame flipado
            mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB,
                               data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = landmarker.detect(mp_img)
            process_gestures(results, w, h)

            # 5) Capas de interfaz OpenCV (Overlay Mano + UI Lente)
            if results and results.hand_landmarks:
                draw_hand_overlay(frame, results.hand_landmarks[0], w, h)
            if tvec is not None:
                draw_camera_ui(frame, w, h, tvec)

            # 6) Render Matrix en OpenGL
            glViewport(0, 0, w, h)
            upload_frame_texture(frame, w, h)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            draw_background_quad(w, h)

            if rvec is not None:
                draw_scene_3d(rvec, tvec, K, w, h, t)

            glfw.swap_buffers(window)
            glfw.poll_events()

    cap.release()
    glfw.terminate()


if __name__ == "__main__":
    main()