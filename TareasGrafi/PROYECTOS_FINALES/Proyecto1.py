import time
import math
import numpy as np
import cv2
import os
 
# --- CONFIGURACIÓN GLOBAL ---
W, H = 800, 600
FPS = 30
DURATION = 60.0  # 6 escenas de ~10 segundos cada una
 
def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
 
def smoothstep(a, b, x):
    x = clamp01((x - a) / (b - a))
    return x * x * (3 - 2 * x)
 
def poly_param(fx, fy, t0, t1, n, cx, cy, sx, sy):
    """Genera puntos para cv2.polylines basados en funciones paramétricas"""
    ts = np.linspace(t0, t1, n, dtype=np.float32)
    xs = fx(ts) * sx + cx
    ys = fy(ts) * sy + cy
    return np.round(np.stack([xs, ys], 1)).astype(np.int32).reshape((-1, 1, 2))
 
def hsv_to_bgr(h, s, v):
    hsv = np.uint8([[[h % 180, np.clip(s, 0, 255), np.clip(v, 0, 255)]]])
    return tuple(int(x) for x in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
 
def background_radial(img, t, h0=10, h1=40):
    """Fondo procedural con gradiente radial en HSV"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx = (xx - W * 0.5) / (W * 0.5)
    ny = (yy - H * 0.5) / (H * 0.5)
    r = np.sqrt(nx * nx + ny * ny)
    hsv = np.zeros((H, W, 3), np.uint8)
    hue = (h0 + (h1 - h0) * r + 6 * np.sin(t * 0.7 + r * 4.0)).astype(np.float32)
    hsv[:, :, 0] = np.clip(hue, 0, 179).astype(np.uint8)
    hsv[:, :, 1] = 200
    hsv[:, :, 2] = (20 + 80 * np.clip(1 - r * 0.7, 0, 1)).astype(np.uint8)
    img[:] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
 
def draw_sparkles(img, t, seed=77, n=70):
    """Fondo: partículas brillantes que pulsan"""
    rng = np.random.default_rng(seed)
    xs = rng.integers(0, W, n)
    ys = rng.integers(0, H, n)
    for i in range(n):
        bright = int(120 + 135 * abs(math.sin(t * 2.5 + i * 1.3)))
        size = 1 if bright < 200 else 2
        cv2.circle(img, (int(xs[i]), int(ys[i])), size, (bright, bright, bright), -1)
 
# --- POST-PROCESAMIENTO ---
def post_vignette(img, strength=0.7):
    """Viñeta oscura en los bordes"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx = (xx - W * 0.5) / (W * 0.5)
    ny = (yy - H * 0.5) / (H * 0.5)
    r2 = nx * nx + ny * ny
    mask = np.clip(1.0 - strength * r2, 0.0, 1.0)
    return (img.astype(np.float32) * mask[..., None]).astype(np.uint8)
 
# ─────────────────────────────────────────────
# ESCENA 1 — Presentación + Rosa Polar + Hexágonos
# ─────────────────────────────────────────────
def scene_1_presentacion(img, t):
    background_radial(img, t, h0=5, h1=30)
    draw_sparkles(img, t, seed=11, n=90)
 
    cx, cy = W // 2, H // 2
 
    # Hexágonos concéntricos rotando (en lugar de cuadros)
    for i in range(1, 7):
        r = i * 50 + int(15 * math.sin(t * 1.8 + i))
        angle_off = t * 0.4 * (1 if i % 2 == 0 else -1)
        pts_hex = []
        for k in range(6):
            ang = angle_off + k * math.pi / 3
            pts_hex.append([cx + int(r * math.cos(ang)), cy + int(r * math.sin(ang))])
        pts_hex = np.array(pts_hex, np.int32)
        cv2.polylines(img, [pts_hex], True, hsv_to_bgr(10 + i * 8, 210, 160), 2, cv2.LINE_AA)
 
    # Rosa polar de 5 pétalos en las esquinas
    fx_rosa = lambda th: np.cos(5 * th)
    fy_rosa = lambda th: np.sin(5 * th)
    for (rx, ry) in [(100, 90), (W - 100, 90), (100, H - 90), (W - 100, H - 90)]:
        pts_r = poly_param(
            lambda th: np.cos(5 * th) * np.cos(th),
            lambda th: np.cos(5 * th) * np.sin(th),
            0, 2 * math.pi, 500, rx, ry, 55, 55)
        cv2.polylines(img, [pts_r], True, hsv_to_bgr(20, 200, 220), 1, cv2.LINE_AA)
 
    cv2.putText(img, "DEMO PROCEDURAL - GRAFICACION POR COMPUTADORA",
                (28, H // 2 - 18), cv2.FONT_HERSHEY_TRIPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "OpenCV  |  Curvas Matematicas  |  Transformaciones",
                (40, H // 2 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 220, 150), 2, cv2.LINE_AA)
 
 
# ─────────────────────────────────────────────
# ESCENA 2 — Rosa Polar (múltiples pétalos)
# ─────────────────────────────────────────────
def scene_2_rosa_polar(img, t):
    background_radial(img, t, h0=0, h1=20)
    draw_sparkles(img, t, seed=22, n=55)
 
    cx, cy = W // 2, H // 2
 
    # Rosa central grande (k=7 pétalos animados)
    k_val = 7
    scale_anim = 1.0 + 0.12 * math.sin(t * 3.5)
    pts_main = poly_param(
        lambda th: np.cos(k_val * th) * np.cos(th),
        lambda th: np.cos(k_val * th) * np.sin(th),
        0, 2 * math.pi, 1200, cx, cy,
        int(130 * scale_anim), int(130 * scale_anim))
    cv2.polylines(img, [pts_main], True, hsv_to_bgr(5, 230, 255), 2, cv2.LINE_AA)
 
    # Rosa satélite izquierda (k=3)
    ox1 = int(W * 0.2 + 25 * math.cos(t))
    oy1 = int(H * 0.38 + 25 * math.sin(t))
    pts_s1 = poly_param(
        lambda th: np.cos(3 * th) * np.cos(th),
        lambda th: np.cos(3 * th) * np.sin(th),
        0, 2 * math.pi, 500, ox1, oy1, 40, 40)
    cv2.polylines(img, [pts_s1], True, hsv_to_bgr(15, 220, 240), 1, cv2.LINE_AA)
 
    # Rosa satélite derecha (k=5)
    ox2 = int(W * 0.8 + 25 * math.sin(t))
    oy2 = int(H * 0.62 + 25 * math.cos(t))
    pts_s2 = poly_param(
        lambda th: np.cos(5 * th) * np.cos(th),
        lambda th: np.cos(5 * th) * np.sin(th),
        0, 2 * math.pi, 500, ox2, oy2, 35, 35)
    cv2.polylines(img, [pts_s2], True, hsv_to_bgr(25, 200, 230), 1, cv2.LINE_AA)
 
    cv2.putText(img, "Curva 1: Rosa Polar  r = cos(k*theta)",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
 
 
# ─────────────────────────────────────────────
# ESCENA 3 — Triángulos + Espiral de Fermat
# ─────────────────────────────────────────────
def scene_3_triangulos(img, t):
    background_radial(img, t, h0=70, h1=110)
 
    cx, cy = W // 2, H // 2
 
    # Espirales de Fermat decorativas en esquinas
    for (ex, ey) in [(140, 130), (W - 140, 130), (140, H - 130), (W - 140, H - 130)]:
        pts_f = poly_param(
            lambda th: np.sqrt(np.abs(th)) * np.cos(th + t),
            lambda th: np.sqrt(np.abs(th)) * np.sin(th + t),
            0, 6 * math.pi, 300, ex, ey, 8, 8)
        cv2.polylines(img, [pts_f], False, hsv_to_bgr(90, 180, 120), 1, cv2.LINE_AA)
 
    # Rejilla de triángulos equiláteros con transformación afín
    rot_ang = 0.4 * math.sin(t * 1.2)
    scale_v = 0.85 + 0.15 * math.cos(t * 1.8)
    M = np.array([
        [scale_v * math.cos(rot_ang), -math.sin(rot_ang), cx * (1 - scale_v)],
        [math.sin(rot_ang),  scale_v * math.cos(rot_ang), cy * (1 - scale_v)]
    ], dtype=np.float32)
 
    layer = np.zeros_like(img)
    r_size = 22
    for x in range(90, W, 110):
        for y in range(70, H, 110):
            h_val = math.sqrt(3) / 2 * r_size
            pts_tri = np.array([
                [x, int(y - r_size)],
                [int(x + r_size * 0.87), int(y + r_size * 0.5)],
                [int(x - r_size * 0.87), int(y + r_size * 0.5)]
            ], np.int32)
            color_t = hsv_to_bgr(int(85 + x * 0.04), 220, 190)
            cv2.fillPoly(layer, [pts_tri], color_t, cv2.LINE_AA)
            cv2.polylines(layer, [pts_tri], True, (255, 255, 255), 1, cv2.LINE_AA)
 
    transformed = cv2.warpAffine(layer, M, (W, H))
    img[:] = cv2.addWeighted(img, 1.0, transformed, 0.9, 0)
 
    cv2.putText(img, "Transformaciones: Rotacion + Escala + Espiral de Fermat",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
 
 
# ─────────────────────────────────────────────
# ESCENA 4 — Ondas de anillos + Hipocicloide + Astroide
# ─────────────────────────────────────────────
def scene_4_hipocicloide(img, t):
    background_radial(img, t, h0=55, h1=90)
    draw_sparkles(img, t, seed=44, n=60)
 
    # Emisores de anillos desde tres puntos
    for (pcx, pcy, hue_v) in [(W // 2, H // 2, 65), (W // 4, H // 3, 80), (3 * W // 4, 2 * H // 3, 72)]:
        for i in range(5):
            radius = int((t * 60 + i * 55) % 240)
            alpha = max(0, 1.0 - radius / 240.0)
            cv2.circle(img, (pcx, pcy), radius, hsv_to_bgr(hue_v, 200, int(220 * alpha)), 2, cv2.LINE_AA)
 
    # Curva 2: Hipocicloide (R=5, r=3 → astroide de 2 puntas)
    R, r = 5, 3
    fx_hipo = lambda th: (R - r) * np.cos(th) + r * np.cos((R - r) / r * th)
    fy_hipo = lambda th: (R - r) * np.sin(th) - r * np.sin((R - r) / r * th)
    pts_hipo = poly_param(fx_hipo, fy_hipo, 0, 6 * math.pi, 600, W // 5, H // 4, 40, 40)
    cv2.polylines(img, [pts_hipo], True, (255, 200, 100), 2, cv2.LINE_AA)
 
    # Curva 3: Astroide (hipocicloide clásica, 4 puntas)
    fx_ast = lambda th: np.cos(th) ** 3
    fy_ast = lambda th: np.sin(th) ** 3
    pts_ast = poly_param(fx_ast, fy_ast, 0, 2 * math.pi, 400, 4 * W // 5, 3 * H // 4, 100, 100)
    cv2.polylines(img, [pts_ast], True, (100, 255, 220), 2, cv2.LINE_AA)
 
    cv2.putText(img, "Curvas 2 y 3: Hipocicloide y Astroide",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
 
 
# ─────────────────────────────────────────────
# ESCENA 5 — Tormenta de Pentágonos y Estrellas
# ─────────────────────────────────────────────
def scene_5_tormenta(img, t, rng):
    background_radial(img, t, h0=95, h1=130)
 
    # Partículas de lluvia diagonal
    state_rng = np.random.default_rng(555)
    n_lluvia = 50
    lx = (state_rng.integers(0, W, n_lluvia) + int(t * 90)) % W
    ly = (state_rng.integers(0, H, n_lluvia) + int(t * 140)) % H
    for i in range(n_lluvia):
        cv2.line(img, (lx[i], ly[i]), (lx[i] + 3, ly[i] + 5), (180, 220, 255), 1, cv2.LINE_AA)
 
    # Campo denso de pentágonos y estrellas de 5 puntas
    n_particles = 200
    for i in range(n_particles):
        sx = i * 7.3
        sy = i * 12.1
        x = int((sx + t * 45 + 55 * math.sin(t + sy)) % W)
        y = int((sy + t * 28 + 45 * math.cos(t * 0.7 + sx)) % H)
        size = int(5 + 6 * abs(math.sin(t * 1.5 + i)))
        color_p = hsv_to_bgr(int(t * 10 + i * 2), 220, 235)
 
        if i % 3 == 0:
            # Pentágono relleno
            pts_penta = []
            for k in range(5):
                ang = -math.pi / 2 + k * 2 * math.pi / 5
                pts_penta.append([x + int(size * math.cos(ang)), y + int(size * math.sin(ang))])
            cv2.fillPoly(img, [np.array(pts_penta, np.int32)], color_p, cv2.LINE_AA)
        elif i % 3 == 1:
            # Estrella de 5 puntas
            pts_star = []
            for k in range(10):
                ang = -math.pi / 2 + k * math.pi / 5
                r_k = size if k % 2 == 0 else size * 0.4
                pts_star.append([x + int(r_k * math.cos(ang)), y + int(r_k * math.sin(ang))])
            cv2.fillPoly(img, [np.array(pts_star, np.int32)], color_p, cv2.LINE_AA)
        else:
            # Círculo pequeño
            cv2.circle(img, (x, y), size // 2, color_p, -1, cv2.LINE_AA)
 
    cv2.putText(img, "Primitivas Colectivas: Pentagons, Stars & Circles",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
 
 
# ─────────────────────────────────────────────
# ESCENA 6 — Clímax: Curva de Lissajous, Cissoide, Espiral Logarítmica
# ─────────────────────────────────────────────
def scene_6_final(img, t):
    background_radial(img, t, h0=130, h1=175)
    draw_sparkles(img, t, seed=99, n=100)
    cx, cy = W // 2, H // 2
 
    # Curva 4: Lissajous (a=3, b=5, delta animado)
    delta = t * 0.8
    pts_lis = poly_param(
        lambda th: np.sin(3 * th + delta),
        lambda th: np.sin(5 * th),
        0, 2 * math.pi, 800, cx, cy, 95, 95)
    cv2.polylines(img, [pts_lis], True, (120, 255, 255), 3, cv2.LINE_AA)
 
    # Curva 5: Estrofoide / Cissoide de Diocles
    fx_ciss = lambda th: 2 * np.sin(th) * np.tan(th)
    fy_ciss = lambda th: 2 * np.sin(th) ** 2
    pts_ciss = poly_param(fx_ciss, fy_ciss,
                          -math.pi * 0.45, math.pi * 0.45, 400,
                          cx - 200, cy + 130, 40, 40)
    cv2.polylines(img, [pts_ciss], False, (220, 180, 255), 2, cv2.LINE_AA)
 
    # Curva 6: Espiral Logarítmica
    a_spiral = 0.15
    fx_log = lambda th: np.exp(a_spiral * th) * np.cos(th + t)
    fy_log = lambda th: np.exp(a_spiral * th) * np.sin(th + t)
    pts_log = poly_param(fx_log, fy_log, 0, 5 * math.pi, 700,
                         cx + 190, cy + 130, 8, 8)
    cv2.polylines(img, [pts_log], False, (255, 160, 160), 2, cv2.LINE_AA)
 
    # Anillo orbital de estrellas de 5 puntas
    n_orbit = 10
    for i in range(n_orbit):
        ang = t * 0.7 + i * (2 * math.pi / n_orbit)
        rx, ry = int(cx + 155 * math.cos(ang)), int(cy + 155 * math.sin(ang))
        pts_star = []
        for k in range(10):
            a_k = -math.pi / 2 + k * math.pi / 5
            r_k = 9 if k % 2 == 0 else 4
            pts_star.append([rx + int(r_k * math.cos(a_k)), ry + int(r_k * math.sin(a_k))])
        cv2.fillPoly(img, [np.array(pts_star, np.int32)], (255, 255, 200), cv2.LINE_AA)
 
    # Elipse rotatoria central
    cv2.ellipse(img, (cx, cy),
                (int(70 + 12 * math.sin(t * 3)), 35),
                math.degrees(t * 0.9), 0, 360, (255, 255, 255), 2, cv2.LINE_AA)
 
    cv2.putText(img, "Curvas 4-6: Lissajous, Cissoide, Espiral Logaritmica",
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
 
 
# ─────────────────────────────────────────────
# TIMELINE Y TRANSICIONES
# ─────────────────────────────────────────────
def render_scene_router(buf, scene_id, t, rng):
    if scene_id == 0:
        scene_1_presentacion(buf, t)
    elif scene_id == 1:
        scene_2_rosa_polar(buf, t)
    elif scene_id == 2:
        scene_3_triangulos(buf, t)
    elif scene_id == 3:
        scene_4_hipocicloide(buf, t)
    elif scene_id == 4:
        scene_5_tormenta(buf, t, rng)
    else:
        scene_6_final(buf, t)
 
def timeline(t, rng, bufA, bufB):
    block = int(min(5, max(0, t // 10)))
    t_in = t - block * 10
 
    render_scene_router(bufA, block, t, rng)
    frame = bufA
 
    # Crossfade entre escenas
    if block < 5 and t_in >= 8.5:
        render_scene_router(bufB, block + 1, t, rng)
        alpha = smoothstep(8.5, 10.0, t_in)
        frame = cv2.addWeighted(bufA, 1.0 - alpha, bufB, alpha, 0)
 
    # Fade in / fade out global
    fade_in = smoothstep(0.0, 1.8, t)
    fade_out = 1.0 - smoothstep(DURATION - 1.8, DURATION, t)
    total_fade = fade_in * fade_out
 
    if total_fade < 0.999:
        frame = (frame.astype(np.float32) * total_fade).astype(np.uint8)
 
    return frame
 
 
# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    rng = np.random.default_rng(2025)
    bufA = np.zeros((H, W, 3), np.uint8)
    bufB = np.zeros((H, W, 3), np.uint8)
 
    total_frames = int(DURATION * FPS)
 
    os.makedirs("renders", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter("renders/demo_proyecto.mp4", fourcc, FPS, (W, H))
 
    print("-> Renderizando 'renders/demo_proyecto.mp4'...")
 
    for i in range(total_frames):
        t = i / FPS
        frame = timeline(t, rng, bufA, bufB)
        frame = post_vignette(frame, 0.7)
        video_writer.write(frame)
 
        cv2.imshow("Demo Procedural - Graficacion", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
 
    video_writer.release()
    cv2.destroyAllWindows()
    print("Listo! Video guardado en renders/demo_proyecto.mp4")
 
if __name__ == "__main__":
    main()
