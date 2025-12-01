# risk_detection/engine/zona_riesgo_pickup_tubular.py
from shapely.geometry import Point, Polygon
from .base_scene import BaseScene
from utils.geometry_utils import has_all_classes, boxes_to_polys_by_name
from utils.pose_utils import iter_feet
from utils.visualization import draw_polygon
import cv2


class zona_riesgo_pickup_tubular(BaseScene):
    """
    Detecta el riesgo cuando el pie de una persona se encuentra
    dentro de la zona de riesgo durante la operación de pickup tubular.
    """
    name = "zona_riesgo_pickup_tubular"

    def __init__(self, cfg):
        super().__init__(cfg)
    
    def _instant_condition(self, det_obj):
        """True si tubular está solapado/cerca a brazotaladro."""
        
        req = ["brazotaladro", "tubular"]
        if not has_all_classes(det_obj, req):
            return False
        polys = boxes_to_polys_by_name(det_obj, req)
        braz = polys["brazotaladro"]
        tub = polys["tubular"]

        inter = tub.intersection(braz).area
        amin = max(min(tub.area, braz.area), 1.0)
        ratio = inter / amin
        dist = braz.distance(tub)

        return (ratio > self.cfg.PICKUP_ZONE_OVERLAP_MIN) and (dist < self.cfg.PICKUP_ZONE_DIST_PX)

    def _risk_feet_inside_zone(self, res_pose):
        """
        Detecta si algún landmark del pie (izquierdo o derecho)
        está dentro o sobre el polígono de riesgo definido.
        """

        poly_np = self.cfg.POLIGONO_RIESGO_PICK_UP_TUBULAR
        poly = Polygon(poly_np)

        for x, y in iter_feet(res_pose, self.cfg.FEET_IDXS):
            if Point(x, y).within(poly) or Point(x, y).touches(poly):
                return True, poly_np
        return False, poly_np
    
    def evaluate(self, det_obj, res_pose, frame):
        scene = self._instant_condition(det_obj)
        self.increment_scene_active_pos_neg(scene)

        if self.scene_active_pos >= self.cfg.PICKUP_ZONE_SCENE_ON:
            self.activate_scene()
        elif self.scene_active_neg >= self.cfg.PICKUP_ZONE_SCENE_OFF:
            self.deactivate_scene()
        
        risk = False

        if self.scene_active:
            risk, poly_np = self._risk_feet_inside_zone(res_pose)
            self.increment_risk_active_pos_neg(risk)

            if self.risk_active_pos >= self.cfg.PICKUP_ZONE_RISK_ON:
                self.activate_risk()
            elif self.risk_active_neg >= self.cfg.PICKUP_ZONE_RISK_OFF:
                self.deactivate_risk()

            if frame is not None and self.cfg.VISUALIZE:
                draw_polygon(frame, poly_np, active=self.risk_active)

        # print(f"Escena activa: {self.scene_active}, Riesgo activo: {self.risk_active}, Frames positivos: {self.risk_active_pos}, Frames_negativos: {self.risk_active_neg}")

        self.log_state()
        return self.make_result(self.scene_active, self.risk_active)
