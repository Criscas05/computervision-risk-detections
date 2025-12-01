# risk_detection/risk_engine.py
from engine.extraccion_stickout import ExtraccionStickout
from engine.cabron_abierto import CabronAbierto
from engine.acople_pintubular import AcoplePintubular
from engine.pickup_tubular import PickupTubular
from engine.tubular_pendulando import TubularPendulando
from engine.zona_riesgo_pickup_tubular import zona_riesgo_pickup_tubular
from engine.mano_safata import AcoplePintubularManoSafata

class RiskEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.scenes = [
            ExtraccionStickout(cfg),
            CabronAbierto(cfg),
            PickupTubular(cfg),
            TubularPendulando(cfg),
            AcoplePintubular(cfg),
            zona_riesgo_pickup_tubular(cfg),
            AcoplePintubularManoSafata(cfg)
        ]

    def process(self, det_obj, res_pose, frame=None):
        results = {}
        for s in self.scenes:
            results[s.name] = s.evaluate(det_obj, res_pose, frame) # Evaluamos cada unas de escenas de riesgos inicializadas arriba con sus respectivios riesgos
        return results
