# risk_detection/engine/extraccion_stickout.py
import numpy as np
from shapely.geometry import Point, Polygon
from .base_scene import BaseScene
from utils.geometry_utils import has_all_classes, boxes_to_polys_by_name, point_in_or_touch_poly
from utils.pose_utils import iter_feet
from utils.visualization import draw_polygon

class ExtraccionStickout(BaseScene):
    name = "extraccion_stickout"

    def __init__(self, cfg):
        super().__init__(cfg)

    def _instant_condition(self, det_obj):
        """
        Detecta la escena de 'Extracción de Stickout' usando geometría con Shapely.
        
        Se evalúan:
        - Solapamiento espacial entre el stickout y el brazo.
        - Distancia vertical entre sus cajas.
        - Alineación horizontal aproximada.
        """
        
        req = ["stickout", "brazotaladro"]
        if not has_all_classes(det_obj, req):
            return False
        polys = boxes_to_polys_by_name(det_obj, req)
        s, b = polys["stickout"], polys["brazotaladro"]

        inter = s.intersection(b).area
        amin = max(min(s.area, b.area), 1.0)
        ratio = inter / amin

        if ratio > self.cfg.EXTR_OVERLAP_MIN:
            return True

        dist = s.distance(b)
        cx_s = s.centroid.x; cx_b = b.centroid.x
        w_avg = ((s.bounds[2]-s.bounds[0]) + (b.bounds[2]-b.bounds[0]))/2.0
        aligned = abs(cx_s - cx_b) < w_avg * self.cfg.EXTR_ALIGN_RATIO

        return (dist <= self.cfg.EXTR_DIST_PX) and aligned

    def _risk_polygon(self, res_pose):
        """
        Detecta si algún landmark del pie (izquierdo o derecho)
        está dentro o sobre el polígono de riesgo definido.
        """
        poly_np = self.cfg.POLIGONO_RIESGO_STICKOUT

        poly = Polygon(poly_np)
        for x, y in iter_feet(res_pose, self.cfg.FEET_IDXS):
            if point_in_or_touch_poly([x,y], poly):
                return True
        return False

    def evaluate(self, det_obj, res_pose, frame):
        scene = self._instant_condition(det_obj)
        self.increment_scene_active_pos_neg(scene)

        if self.scene_active_pos >= self.cfg.EXTR_SCENE_ON:
            self.activate_scene()
        elif self.scene_active_neg >= self.cfg.EXTR_SCENE_OFF:
            self.deactivate_scene()

        risk = False

        if self.scene_active:
            risk = self._risk_polygon(res_pose)
            self.increment_risk_active_pos_neg(risk)

            if self.risk_active_pos >= self.cfg.EXTR_RISK_ON:
                self.activate_risk()
            elif self.risk_active_neg >= self.cfg.EXTR_RISK_OFF:
                self.deactivate_risk()

            if frame is not None and self.cfg.VISUALIZE:
                draw_polygon(frame, self.cfg.POLIGONO_RIESGO_STICKOUT, active=self.risk_active)
                
        self.log_state()
        # print(f"Escena activa: {self.scene_active}, Riesgo activo: {self.risk_active}, Frames positivos: {self.risk_active_pos}, Frames_negativos: {self.risk_active_neg}")
        return self.make_result(self.scene_active, self.risk_active)
