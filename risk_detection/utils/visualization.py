# risk_detection/utils/visualization.py
import cv2
import numpy as np

def draw_polygon(frame, poly_np, active=False):
    color = (0, 255, 0) if not active else (0, 0, 255)
    cv2.polylines(frame, [poly_np.astype(np.int32)], True, color, 2)

def draw_line(frame, p1, p2, active=False):
    color = (0, 255, 0) if not active else (0, 0, 255)
    cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 3)

def put_text(frame, text, org=(20, 40), color=(255,255,255), scale=0.8, thick=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

def draw_hud(frame, fps=None, lines=[], detections={}):
    y = 450
    if fps is not None:
        put_text(frame, f"FPS: {fps:.1f}", (20, y), (0,255,0), 0.9, 2); y += 30
    for s in lines:
        if "Riesgo: NO" in s:
            put_text(frame, s, (20, y), (0,255,0), 0.8, 2); y += 28
        else:
            put_text(frame, s, (20, y), (0,0,255), 0.8, 2); y += 28

    # Definir las conexiones del esqueleto (Formato COCO 17 puntos)
    # Cada tupla representa dos índices de keypoints que deben unirse
    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4),       # Cabeza (Nariz a ojos/oídos)
        (5, 6),                               # Hombros
        (5, 7), (7, 9),                       # Brazo izquierdo
        (6, 8), (8, 10),                      # Brazo derecho
        (5, 11), (6, 12),                     # Torso (Hombros a caderas)
        (11, 12),                             # Caderas
        (11, 13), (13, 15),                   # Pierna izquierda
        (12, 14), (14, 16)                    # Pierna derecha
    ]

    hand_points = [9, 10]
    foot_points = [15, 16]
    COLOR_HAND = (0, 255, 255)  # Amarillo
    COLOR_FOOT = (255, 0, 255)  # Magenta
    COLOR_SKELETON = (200, 100, 0) # Azulado para el esqueleto

    # Verificación de detección
    if hasattr(detections["pose"][0], "keypoints") and detections["pose"][0].keypoints is not None:
        # Obtener coordenadas (x, y)
        keypoints = detections["pose"][0].keypoints.xy.cpu().numpy()
        
        # Iterar sobre cada persona detectada
        for kp_set in keypoints:
            
            # --- 1. DIBUJAR EL ESQUELETO (LÍNEAS) ---
            for idx1, idx2 in SKELETON_CONNECTIONS:
                # Obtener las coordenadas de los dos puntos a conectar
                pt1 = (int(kp_set[idx1][0]), int(kp_set[idx1][1]))
                pt2 = (int(kp_set[idx2][0]), int(kp_set[idx2][1]))
                
                # Verificar que los puntos no sean (0,0) (puntos no detectados)
                if pt1[0] != 0 and pt1[1] != 0 and pt2[0] != 0 and pt2[1] != 0:
                    cv2.line(frame, pt1, pt2, COLOR_SKELETON, 2)

            # --- 2. DIBUJAR PUNTOS Y RESALTES ---
            for idx, (x, y) in enumerate(kp_set):
                x, y = int(x), int(y)
                
                # Si el punto es (0,0), saltarlo
                if x == 0 and y == 0:
                    continue

                # Dibujar articulación general
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                
                # Resaltar manos
                if idx in hand_points:
                    cv2.circle(frame, (x, y), 3, COLOR_HAND, -1)
                    # cv2.putText(frame, "Hand", (x+5, y-5),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HAND, 2)
                # Resaltar pies
                elif idx in foot_points:
                    cv2.circle(frame, (x, y), 6, COLOR_FOOT, -1)
                    cv2.putText(frame, "Foot", (x+5, y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_FOOT, 2)


