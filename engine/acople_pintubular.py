# risk_detection/engine/acople_pintubular.py
import time, numpy as np
from shapely.geometry import Polygon, Point
from .base_scene import BaseScene
from utils.geometry_utils import has_all_classes, boxes_to_polys_by_name, make_line_from_stickout_to_llavetm, point_in_or_touch_poly
from utils.pose_utils import iter_feet
from utils.visualization import draw_polygon, draw_line, put_text

class AcoplePintubular(BaseScene):
    name = "acople_pintubular"

    def __init__(self, cfg):
        super().__init__(cfg)

    def _update_height(self, det_obj):
        req = ["stickout"]
        if not has_all_classes(det_obj, req):
            return None, None
        
        polys = boxes_to_polys_by_name(det_obj, req)
        s = polys["stickout"]
        h = (s.bounds[3] - s.bounds[1])
        area = s.area

        self.heights_stickout.append(h)
        if len(self.heights_stickout) > self.cfg.ACOPLE_HEIGHT_BUFFER:
            self.heights_stickout.pop(0)
        return h, area

    def _confirm_scene(self, h, area):
        if len(self.heights_stickout) > 5:
            altura_promedio_pasada = np.mean(self.heights_stickout[:-3])
        else:
            altura_promedio_pasada = h
        inc_rel = (h - altura_promedio_pasada) / max(altura_promedio_pasada, 1.0) # Incremento Relativo

        # --- Evaluar estabilidad (altura mantenida) ---
        # Si el incremento fue grande, comenzamos a verificar si se mantiene
        scene = inc_rel > self.cfg.ACOPLE_INC_MIN and area > self.cfg.ACOPLE_AREA_MIN_STICKOUT
        self.increment_scene_active_pos_neg(scene)

        # --- Confirmar acople si se mantiene estable varios frames ---
        if not self.scene_active and self.scene_active_pos >= self.cfg.ACOPLE_SCENE_ON:
            self.activate_scene()
            self.initialize_time()

    def _window_remaining(self):
        if not self.scene_active or self.t0 is None:
            return 0.0
        left = self.cfg.ACOPLE_WINDOW_SEC - (time.time() - self.t0)
        return max(0.0, left)

    def _risk_window_polygon(self, det_obj, res_pose, frame):
        poly_np = self.cfg.POLIGONO_RIESGO_STICKOUT_LLAVETM120
        poly = Polygon(poly_np)
        risk = False
        for x, y in iter_feet(res_pose, self.cfg.FEET_IDXS):
            if point_in_or_touch_poly([x,y], poly):
                risk = True

        if frame is not None and self.cfg.VISUALIZE:
            draw_polygon(frame, poly_np, active=risk)
            rem = int(self._window_remaining())
            put_text(frame, f"Ventana acople: {rem}s", (20, 90))

        return risk

    def evaluate(self, det_obj, res_pose, frame):
        h, area = self._update_height(det_obj)
        if h is not None:
            self._confirm_scene(h, area)

        risk = False
        if self.scene_active and self._window_remaining() > 0:
            risk = self._risk_window_polygon(det_obj, res_pose, frame)

            self.increment_risk_active_pos_neg(risk)

            if self.risk_active_pos >= self.cfg.ACOPLE_RISK_ON:
                self.activate_risk()
            elif self.risk_active_neg >= self.cfg.ACOPLE_RISK_OFF:
                self.deactivate_risk()

        elif self.scene_active and self._window_remaining() <= 0:
            # cerrar escena y reset ventana
            self.deactivate_scene()

        # print(f"Escena: {self.scene_active}, {self.scene_active_pos}, {self.scene_active_neg}")
        # print(f"Riesgo: {self.risk_active}, {self.risk_active_pos}, {self.risk_active_neg}")
        # print("------------------------------------------------------------")
        self.log_state()
        return self.make_result(self.scene_active, self.risk_active)
