# risk_detection/engine/tubular_pendulando.py
import numpy as np
from shapely.geometry import Polygon, Point
from .base_scene import BaseScene
from utils.geometry_utils import has_all_classes, boxes_to_polys_by_name
from utils.pose_utils import iter_feet
from utils.visualization import draw_polygon
import cv2

class TubularPendulando(BaseScene):
    name = "tubular_pendulando"

    def __init__(self, cfg):
        super().__init__(cfg)
    
    def _instant_condition(self, det_obj):
        """
        Detecta la escena 'tubular pendulando'.
        
        Condiciones:
        - Existencia de pin tubular y stick out.
        - Pin tubular ubicado a la derecha de la línea vertical (60% del frame).
        """
        # Calcular la posición X de la línea vertical (60% del ancho del frame)
        x_linea_vertical = int(self.cfg.RESIZE[0] * self.cfg.PEND_LINE_RATIO_X)
        
        req = ["stickout", "pintubular"]
        if not has_all_classes(det_obj, req):
            return False, x_linea_vertical
        polys = boxes_to_polys_by_name(det_obj, req)

        # Verificar si el pin_tubular está a la derecha de la línea vertical
        # Usando el centroide de la caja del pin_tubular
        pin_x = polys["pintubular"].centroid.x

        return (pin_x > x_linea_vertical), x_linea_vertical

        
    def _risk_polygon_golpeo_tubular(self, res_pose):
        """
        Detecta si algún landmark del pie (izquierdo o derecho)
        está dentro o sobre el polígono de riesgo definido.
        """

        poly_np = self.cfg.POLIGONO_RIESGO_PIN_TUBULAR

        poly = Polygon(poly_np)

        for x, y in iter_feet(res_pose, self.cfg.FEET_IDXS):
            if Point(x, y).within(poly) or Point(x, y).touches(poly):
                return True, poly_np
        return False, poly_np
    
    def evaluate(self, det_obj, res_pose, frame):
        scene, x_linea_vertical = self._instant_condition(det_obj)
        self.increment_scene_active_pos_neg(scene)

        if self.scene_active_pos >= self.cfg.PEND_SCENE_ON:
            self.activate_scene()
        elif self.scene_active_neg >= self.cfg.PEND_SCENE_OFF:
            self.deactivate_scene()
        
        risk = False

        if self.scene_active:
            risk, poly_np = self._risk_polygon_golpeo_tubular(res_pose)
            self.increment_risk_active_pos_neg(risk)

            if self.risk_active_pos >= self.cfg.PEND_RISK_ON:
                self.activate_risk()
            elif self.risk_active_neg >= self.cfg.PEND_RISK_OFF:
                self.deactivate_risk()

            if frame is not None and self.cfg.VISUALIZE:
                # cv2.line(frame, (x_linea_vertical, 0), (x_linea_vertical, self.cfg.RESIZE[1]), (255,0,255), 2)
                draw_polygon(frame, poly_np, active=self.risk_active)

        # print(f"Escena activa: {self.scene_active}, Riesgo activo: {self.risk_active}, Frames positivos: {self.risk_active_pos}, Frames_negativos: {self.risk_active_neg}")
        self.log_state()
        return self.make_result(self.scene_active, self.risk_active)
