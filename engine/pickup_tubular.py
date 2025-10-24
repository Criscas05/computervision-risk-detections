# risk_detection/engine/pickup_tubular.py
from shapely.geometry import Point, Polygon
from .base_scene import BaseScene
from utils.geometry_utils import has_all_classes, boxes_to_polys_by_name
from utils.pose_utils import iter_keypoints

class PickupTubular(BaseScene):
    name = "pickup_tubular"

    def __init__(self, cfg):
        super().__init__(cfg)
    
    def _instant_condition(self, det_obj):
        """True si tubular está solapado/cerca a brazotaladro."""
        
        req = ["brazotaladro", "tubular"]
        if not has_all_classes(det_obj, req):
            return False
        polys = boxes_to_polys_by_name(det_obj, req)
        braz = polys["brazotaladro"]; tub = polys["tubular"]

        inter = tub.intersection(braz).area
        amin = max(min(tub.area, braz.area), 1.0)
        ratio = inter / amin
        dist = braz.distance(tub)

        return (ratio > self.cfg.PICKUP_OVERLAP_MIN) and (dist < self.cfg.PICKUP_DIST_PX)

    def _risk_hands_on_brazotaladro(self, res_pose, det_obj):
        """True si mano (izq/der) cae dentro del bbox de 'brazotaladro'."""

        req = ["brazotaladro"]
        if not has_all_classes(det_obj, req):
            return False
        polys = boxes_to_polys_by_name(det_obj, req)
        braz = polys["brazotaladro"]

        if res_pose and hasattr(res_pose[0], "keypoints"):
            for kp in iter_keypoints(res_pose):
                for idx in self.cfg.HAND_IDXS:
                    x, y = kp[idx]
                    if Point(x, y).within(braz):
                        return True
        return False

    def evaluate(self, det_obj, res_pose, frame):
        scene = self._instant_condition(det_obj)
        self.increment_scene_active_pos_neg(scene)

        if self.scene_active_pos >= self.cfg.PICKUP_SCENE_ON:
            self.activate_scene()
        elif self.scene_active_neg >= self.cfg.PICKUP_SCENE_OFF:
            self.deactivate_scene()
        
        risk = False

        if self.scene_active:
            risk = self._risk_hands_on_brazotaladro(res_pose, det_obj)
            self.increment_risk_active_pos_neg(risk)

            if self.risk_active_pos >= self.cfg.PICKUP_RISK_ON:
                self.activate_risk()
            elif self.risk_active_neg >= self.cfg.PICKUP_RISK_OFF:
                self.deactivate_risk()

        # print(f"Escena activa: {self.scene_active}, Riesgo activo: {self.risk_active}, Frames positivos: {self.risk_active_pos}, Frames_negativos: {self.risk_active_neg}")

        self.log_state()
        return self.make_result(self.scene_active, self.risk_active)
