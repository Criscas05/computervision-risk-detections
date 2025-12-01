# risk_detection/engine/cabron_abierto.py
from shapely.geometry import box as shapely_box
from .base_scene import BaseScene
from utils.geometry_utils import has_all_classes, boxes_to_polys_by_name, feet_distance_to_geom
from utils.pose_utils import iter_feet

class CabronAbierto(BaseScene):
    name = "cabron_abierto"

    def __init__(self, cfg):
        super().__init__(cfg)

    def evaluate(self, det_obj, res_pose, frame):
        req = ["cabron"]
        active = has_all_classes(det_obj, req)

        risk = False
        self.increment_scene_active_pos_neg(active)

        if self.scene_active_pos >= self.cfg.CABRON_SCENE_ON:
            self.activate_scene()
        elif self.scene_active_neg >= self.cfg.CABRON_SCENE_OFF:
            self.deactivate_scene()
        
        if self.scene_active:
            polys = boxes_to_polys_by_name(det_obj, req)
            # print(polys)
            # print(self.risk_active,self.risk_active_pos, self.risk_active_neg)
            # print("------------------")
            if polys:
                cabron_geom = polys["cabron"]
                feet = list(iter_feet(res_pose, self.cfg.FEET_IDXS))
                risk = feet_distance_to_geom(feet, cabron_geom, self.cfg.CABRON_PIE_PROX_PX)
            else:
                risk = False
            
            self.increment_risk_active_pos_neg(risk)

            if self.risk_active_pos >= self.cfg.CABRON_RISK_ON:
                self.activate_risk()
            elif self.risk_active_neg >= self.cfg.CABRON_RISK_OFF:
                self.deactivate_risk()
        self.log_state()
        # print(f"Escena activa: {self.scene_active}, Riesgo activo: {self.risk_active}, Frames positivos: {self.risk_active_pos}, Frames_negativos: {self.risk_active_neg}")
        return self.make_result(self.scene_active, self.risk_active)
