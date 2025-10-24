# ============================================================
# risk_detection/engine/base_scene.py
# ------------------------------------------------------------
# Clase base para todas las escenas del RiskEngine.
# Proporciona estructura estándar, persistencia y utilidades
# para control de estado, visualización y reinicio.
# ============================================================

from abc import ABC, abstractmethod
import time
import datetime

class BaseScene(ABC):
    """
    Clase base abstracta para las escenas y riesgos.
    Todas las escenas deben heredar de esta clase y sobrescribir:
        - evaluate(det_obj, res_pose, frame)
    """

    name: str = "base_scene"

    def __init__(self, cfg):

        self.cfg = cfg

        self.scene_active = False
        self.scene_active_pos = 0
        self.scene_active_neg = 0

        self.risk_active = False
        self.risk_active_pos = 0
        self.risk_active_neg = 0

        self.heights_stickout = []
        self.t0 = None

        self.logs_enabled = True 

    # ============================================================
    # Métodos principales a implementar
    # ============================================================

    @abstractmethod
    def evaluate(self, det_obj, res_pose, frame):
        """
        Evalúa la escena y su riesgo en un frame.
        Debe retornar un diccionario con la forma:
        {
            "scene": bool,
            "risk": bool,
            "extras": dict(opcional)
        }

        Args:
            det_obj: detecciones YOLO / Supervision (sv.Detections)
            res_pose: resultado de modelo de pose YOLO
            frame: imagen actual (para visualización opcional)
        """
        raise NotImplementedError("Cada subclase debe implementar evaluate().")

    # ============================================================
    # Métodos utilitarios comunes
    # ============================================================

    def activate_scene(self):
        """Activa la escena (o riesgo)"""
        self.scene_active = True

    def deactivate_scene(self):
        """Desactiva la escena y resetea contadores."""
        self.scene_active = False
        self.scene_active_pos = 0
        self.scene_active_neg = 0

        self.risk_active = False
        self.risk_active_pos = 0
        self.risk_active_neg = 0

        self.t0 = None
        self.heights_stickout.clear()

    def activate_risk(self):
        """Marca si el riesgo está activo (solo para tracking visual)."""
        self.risk_active = True
    
    def deactivate_risk(self):
        """Marca si el riesgo está desactivado (solo para tracking visual)."""
        self.risk_active = False

    def increment_scene_active_pos_neg(self, condition: bool):
        """
        Incrementa contadores de frames consecutivos positivos/negativos
        y maneja activación por histéresis.
        """
        if condition:
            self.scene_active_pos += 1
            self.scene_active_neg = 0
        else:
            self.scene_active_neg += 1
            self.scene_active_pos = 0

    def increment_risk_active_pos_neg(self, condition: bool):
        """
        Incrementa contadores de frames consecutivos positivos/negativos
        y maneja activación por histéresis.
        """
        if condition:
            self.risk_active_pos += 1
            self.risk_active_neg = 0
        else:
            self.risk_active_neg += 1
            self.risk_active_pos = 0

    def initialize_time(self):
        "Inicializa la variable temporal para indicar un momento"
        self.t0 = time.time()

    def log_state(self):
        """Imprime mensaje de debug si está habilitado."""
        if self.logs_enabled and self.scene_active and self.risk_active:
            print(f"[{self.name}]")

    # ============================================================
    # Plantilla de retorno estándar
    # ============================================================

    def make_result(self, scene: bool, risk: bool, extras: dict = None):
        """Devuelve el formato de salida estándar para el motor."""
        ts = datetime.datetime.now().isoformat()
        return {
            "time": ts,
            "scene": bool(scene),
            "risk": bool(risk),
        }
