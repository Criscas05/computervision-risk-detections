# risk_detection/engine/acople_pintubular_mano_safata.py
import numpy as np
import cv2
from shapely.geometry import Point, Polygon, box as shapely_box
from .base_scene import BaseScene
from utils.geometry_utils import has_all_classes, boxes_to_polys_by_name, point_in_or_touch_poly, feet_distance_to_geom
from utils.pose_utils import iter_keypoints, iter_feet

class AcoplePintubularManoSafata(BaseScene):
    name = "acople_pintubular_mano_safata"

    def __init__(self, cfg):
        super().__init__(cfg)

    def _instant_condition(self, det_obj, res_pose):
        """
        Detecta si la escena de 'Acople Pin Tubular' está activa.
        Condición: Stickout y Safata solapados y cercanos.
        """
        req = ["stickout", "safata"]
        if not has_all_classes(det_obj, req):
            return False
            
        polys = boxes_to_polys_by_name(det_obj, req)
        stickout = polys["stickout"]
        safata = polys["safata"]

        # Calcular solapamiento
        inter = stickout.intersection(safata).area
        amin = max(min(stickout.area, safata.area), 1.0)
        ratio = inter / amin

        # Calcular cercania del pie al stickout para disminuir falsos positivos
        if polys:
            feet = list(iter_feet(res_pose, self.cfg.FEET_IDXS))
            pie_cerca_stickout = feet_distance_to_geom(feet, stickout, self.cfg.MANO_PIE_PROX_PX)
        else:
            pie_cerca_stickout = False
        
        # Calcular distancia entre stickout y safata
        dist = stickout.distance(safata)

        # print(ratio, dist, pie_cerca_stickout)

        # La escena es válida si se tocan/solapan O están muy cerca y el pie está cerca al stickout
        is_active = ((ratio > self.cfg.MANO_OVERLAP_MIN) or (dist < self.sfg.MANO_DIST_PX)) and pie_cerca_stickout
        return is_active
    
    def _get_safata_danger_zone(self, safata_poly):
        """
        Crea un sub-polígono que representa SOLO la entrada de la safata.
        La entrada está en la parte superior.
        """
        minx, miny, maxx, maxy = safata_poly.bounds
        width = maxx - minx
        height = maxy - miny
        
        # DEFINICIÓN DE LA ENTRADA DE LA SAFATA
        # danger_minx = minx + (width * 0.20) 
        # danger_maxx = maxx - (width * 0.10)
        danger_minx = minx + (width * 0.20) 
        danger_maxx = maxx
        danger_miny = miny + (height * 0.05) # Un poco abajo del borde superior
        danger_maxy = miny + (height * 0.45) # Hasta casi la mitad de la caja
        
        return shapely_box(danger_minx, danger_miny, danger_maxx, danger_maxy)

    def _calculate_virtual_fingertip(self, elbow, wrist):
        """
        Proyecta la posición de los dedos basándose en el vector Codo -> Muñeca.
        Retorna coordenadas (x, y) de la punta de los dedos estimada.
        """
        w = np.array(wrist)
        # Vector del antebrazo
        vec_forearm = w - np.array(elbow)
        
        # Longitud del antebrazo
        arm_length = np.linalg.norm(vec_forearm)
        
        if arm_length < 1:
            return w # Evitar errores si puntos están pegados
        
        # Calcular vector de la mano
        # Multiplicamos el vector unitario del antebrazo (netamente direccion) por la longitud que se quiere aumentar, obteniendo un vector con la misma dirección del vector antebrazo y con la longitud tomada del procentaje deseado del antebrazo
        vec_hand = (vec_forearm / arm_length) * (arm_length * self.cfg.MANO_EXTENSION_FACTOR)
        
        # Punta de dedos = Muñeca + Vector Mano
        fingertip = w + vec_hand

        return fingertip

    def _risk_condition(self, det_obj, res_pose):
        """
        Evalúa si la 'Mano Proyectada' entra en la parte peligrosa de la safata.
        """
        req = ["safata"]
        if not has_all_classes(det_obj, req):
            return False
            
        polys = boxes_to_polys_by_name(det_obj, req)
        safata_poly = polys["safata"]

        # Obtener la zona específica de peligro en safata (la entrada)
        danger_zone = self._get_safata_danger_zone(safata_poly)

        if not res_pose or not hasattr(res_pose[0], "keypoints") or res_pose[0].keypoints is None:
            return False

        # Iterar sobre todas las personas detectadas
        keypoints = res_pose[0].keypoints.xy.cpu().numpy()
        
        for person_kps in keypoints:

            for elbow_idx, wrist_idx in self.cfg.ARMS_IDXS:

                elbow = person_kps[elbow_idx]
                wrist = person_kps[wrist_idx]

                # Calcular dónde estarían los dedos
                fingertip_xy = self._calculate_virtual_fingertip(elbow, wrist)
                fingertip_point = Point(fingertip_xy[0], fingertip_xy[1])

                # Verificar si la PUNTA DE LOS DEDOS está en la zona peligrosa
                if fingertip_point.within(danger_zone):
                    return True
                    
        return False

    def evaluate(self, det_obj, res_pose, frame):
        # Evaluar Escena
        scene_active = self._instant_condition(det_obj, res_pose)
        self.increment_scene_active_pos_neg(scene_active)

        if self.scene_active_pos >= self.cfg.MANO_SCENE_ON:
            self.activate_scene()
        elif self.scene_active_neg >= self.cfg.MANO_SCENE_OFF:
            self.deactivate_scene()

        risk_active = False

        # Evaluar Riesgo (solo si la escena está activa)
        if self.scene_active:
            risk_active = self._risk_condition(det_obj, res_pose)
            self.increment_risk_active_pos_neg(risk_active)

            if self.risk_active_pos >= self.cfg.MANO_RISK_ON:
                self.activate_risk()
            elif self.risk_active_neg >= self.cfg.MANO_RISK_OFF:
                self.deactivate_risk()

            if frame is not None and self.cfg.VISUALIZE:
                # Dibujar zona peligrosa (Azul)
                try:
                    polys = boxes_to_polys_by_name(det_obj, ["safata"])
                    if "safata" in polys:
                        danger_zone = self._get_safata_danger_zone(polys["safata"])
                        # Extraer coords
                        x_min, y_min, x_max, y_max = danger_zone.bounds
                        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
                        
                        # Dibujar proyección de mano si hay personas
                        kps = res_pose[0].keypoints.xy.cpu().numpy()
                        for pk in kps:
                            for e_idx, w_idx in self.cfg.ARMS_IDXS:
                                e, w = pk[e_idx], pk[w_idx]
                                if e[0] > 1 and w[0] > 1:
                                    tip = self._calculate_virtual_fingertip(e, w)
                                    # Línea brazo (verde)
                                    cv2.line(frame, (int(e[0]), int(e[1])), (int(w[0]), int(w[1])), (0, 255, 0), 2)
                                    # Línea mano proyectada (azul)
                                    cv2.line(frame, (int(w[0]), int(w[1])), (int(tip[0]), int(tip[1])), (255, 0, 0), 2)
                                    # Punta (círculo azul)
                                    cv2.circle(frame, (int(tip[0]), int(tip[1])), 4, (255, 0, 0), -1)
                except Exception:
                    pass
                
        self.log_state()
        return self.make_result(self.scene_active, self.risk_active)