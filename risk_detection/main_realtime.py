# main_realtime_refactored.py
import time
import cv2
import signal
import sys
import logging
import pytz
import numpy as np
from collections import deque
from datetime import datetime
from ultralytics import YOLO
from supervision import Detections, BoxAnnotator, LabelAnnotator, ColorLookup
from config import Config
from in_out.beacon_controller import BeaconController
from in_out.db_logger import DBLogger
from in_out.video_clip_writer import VideoClipWriter
from utils.visualization import draw_hud
from risk_engine import RiskEngine

import json
import psutil
import statistics
from datetime import datetime
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


# =============================
# Configuración del Logger
# =============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================================
# Clase para monitoreo de rendimiento del sistema
# ============================================================
class PerformanceMonitor:
    def __init__(self):
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.vram_usage = []
        self.fps_values = []
        self.start_time = datetime.now(pytz.timezone("America/Bogota"))
        self.frames_processed = 0

        self.gpu_handle = None
        if GPU_AVAILABLE:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self.gpu_handle = None

    def update(self, fps):
        """Llamado en cada frame procesado."""
        self.frames_processed += 1
        if fps:
            self.fps_values.append(fps)
        self.cpu_usage.append(psutil.cpu_percent(interval=None))
        self.memory_usage.append(psutil.virtual_memory().percent)

        if self.gpu_handle:
            try:
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                self.gpu_usage.append(gpu_util.gpu)
                self.vram_usage.append(mem_info.used / (1024 ** 2))  # en MiB
            except Exception:
                pass

    def finalize(self, output_dir="logs"):
        """Guarda un archivo JSON con el resumen de métricas."""
        duration = (datetime.now(pytz.timezone("America/Bogota")) - self.start_time).total_seconds()
        summary = {
            "timestamp": self.start_time.isoformat(),
            "duration_sec": round(duration, 2),
            "frames_processed": self.frames_processed,
            "fps_mean": round(statistics.mean(self.fps_values), 2) if self.fps_values else 0,
            "fps_max": round(max(self.fps_values), 2) if self.fps_values else 0,
            "cpu_mean_percent": round(statistics.mean(self.cpu_usage), 2) if self.cpu_usage else 0,
            "memory_mean_percent": round(statistics.mean(self.memory_usage), 2) if self.memory_usage else 0,
            "gpu_mean_percent": round(statistics.mean(self.gpu_usage), 2) if self.gpu_usage else None,
            "vram_mean_mib": round(statistics.mean(self.vram_usage), 2) if self.vram_usage else None,
        }

        output_path = f"{output_dir}/performance_metrics_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        logger.info(f"📊 Métricas de rendimiento guardadas en {output_path}")
        return summary

# =============================
# Clase principal del sistema
# =============================
class RiskDetectionApp:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.keep_running = True
        self.bogota = pytz.timezone("America/Bogota")
        self.db_logger = DBLogger()
        self.previous_risk_states = {}
        self.video_writer = None
        self.cap = None
        self.engine = RiskEngine(cfg)
        self.clip_writer = None
        self.beacon = None
        self.pre_roll_buffer = deque()
        self.fps_smoothed = None
        self.monitor = PerformanceMonitor() if getattr(cfg, "MONITOR_PERFORMANCE", False) else None

    # -------------------------
    # Inicialización
    # -------------------------
    def setup(self):
        logger.info("🚀 Iniciando Risk Detection Service...")
        self._load_models()
        self._setup_db_logger()
        self._setup_beacon()
        self._setup_clip_writer()
        self._setup_video_capture()
        logger.info("✅ Sistema completamente inicializado.")

    def _load_models(self):
        start = time.time()
        self.model_obj = YOLO(self.cfg.MODEL_OBJ)
        self.model_pose = YOLO(self.cfg.MODEL_POSE)
        logger.info(f"📦 Modelos YOLO cargados en {time.time() - start:.2f}s")

    def _setup_db_logger(self):
        db_path = self.db_logger.start_logger(output_dir=self.cfg.LOG_DIR)
        logger.info(f"🗄️ Logs de BBDD → {db_path}")

    def _setup_beacon(self):
        if self.cfg.BEACON_ENABLED:
            self.beacon = BeaconController(self.cfg)
            self.beacon.start_controller()
            logger.info("🚨 Baliza conectada")

    def _setup_clip_writer(self):
        if self.cfg.CLIP_ENABLED:
            self.clip_writer = VideoClipWriter(self.cfg)
            self.clip_writer.start_controller()

    def _setup_video_capture(self):
        logger.info(f"📹 Conectando a video fuente: {self.cfg.VIDEO_SOURCE}")
        self.cap = cv2.VideoCapture(self.cfg.VIDEO_SOURCE)
        if not self.cap.isOpened():
            logger.error(f"❌ No se pudo abrir la fuente: {self.cfg.VIDEO_SOURCE}")
            raise RuntimeError(f"No se pudo abrir fuente: {self.cfg.VIDEO_SOURCE}")

        fps = self.cap.get(cv2.CAP_PROP_FPS) or 15.0
        pre_roll_size = int(fps * self.cfg.CLIP_PREROLL_SEC)
        self.pre_roll_buffer = deque(maxlen=pre_roll_size)
        logger.info(f"📹 FPS: {fps:.2f} | Buffer pre-roll: {pre_roll_size} frames")

        if self.cfg.WRITE_OUTPUT:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.cfg.OUTPUT_PATH, fourcc, fps, self.cfg.RESIZE)
            logger.info(f"💾 Grabación activada → {self.cfg.OUTPUT_PATH}")

    # -------------------------
    # Horarios y señales
    # -------------------------
    def is_within_schedule(self):
        now = datetime.now(self.bogota)
        if now.hour in self.cfg.HOURS_PAUSE and now.minute < self.cfg.MINUTES_PAUSE:
            return False
        return True

    def graceful_shutdown(self, signum, frame):
        """
        Manejador de señal para SIGINT (Ctrl+C) y SIGTERM (docker compose down).
        """
        logger.warning("🛑 Señal de apagado recibida. Deteniendo ejecución...")
        self.keep_running = False

    # -------------------------
    # Bucle principal
    # -------------------------
    def run(self):
        signal.signal(signal.SIGINT, self.graceful_shutdown)
        signal.signal(signal.SIGTERM, self.graceful_shutdown)

        fps_timer = time.time()
        while self.keep_running:
            if not self.is_within_schedule():
                logger.info(f"⏸️ Pausa por {self.cfg.MINUTES_PAUSE} minutos...")
                time.sleep(60 * self.cfg.MINUTES_PAUSE)
                break

            ok, frame = self.cap.read()
            if not ok:
                logger.warning("⚠️ No se pudo leer frame.")
                break

            frame = cv2.resize(frame, self.cfg.RESIZE)
            frame_copy = frame.copy()
            self._process_frame(frame, frame_copy)

            now = time.time()
            inst_fps = 1.0 / max(now - fps_timer, 1e-6)
            fps_timer = now
            self.fps_smoothed = inst_fps if self.fps_smoothed is None else (self.fps_smoothed * 0.9 + inst_fps * 0.1)

            if self.monitor:
                self.monitor.update(self.fps_smoothed)

            if self.cfg.VISUALIZE:
                cv2.imshow("RiskEngine", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("👤 Cierre manual (tecla 'q')")
                    break

        self.cleanup()

    # -------------------------
    # Procesamiento por frame
    # -------------------------
    def _process_frame(self, frame, frame_copy):
        if self.cfg.CLIP_ENABLED:
            self.pre_roll_buffer.append(frame_copy)
            self.clip_writer.put_frame(frame_copy)

        detections = self._run_inference(frame)
        results = self.engine.process(detections["objects"], detections["pose"], frame if self.cfg.VISUALIZE else None)
        self._handle_risks(results)
        self._visualize(frame, detections, results)

        if self.video_writer:
            self.video_writer.write(frame)

    def _run_inference(self, frame):
        res_obj = self.model_obj(frame, device=self.cfg.DEVICE, conf=self.cfg.CONF_OBJ, verbose=False)
        det_obj = Detections.from_ultralytics(res_obj[0])
        names = res_obj[0].names
        det_obj.data["class_name"] = np.array([names[i] for i in det_obj.class_id.astype(int)])
        res_pose = self.model_pose(frame, device=self.cfg.DEVICE, conf=self.cfg.CONF_POSE, verbose=False)
        return {"objects": det_obj, "pose": res_pose}

    def _handle_risks(self, results):
        any_risk = False # Bandera para la Baliza
        for scene, data in results.items():
            current = data["risk"]
            prev = self.previous_risk_states.get(scene, False)

            video_file_name = None

            # Lógica para el Grabador de Clips
            if self.cfg.CLIP_ENABLED:
                # video_file_name = None

                # Transición: De NO RIESGO a RIESGO
                if current and not prev:
                    ts_str = data["time"].replace(":", "-").replace(".", "_")
                    video_file_name = f"{scene}_{ts_str}.mp4"
                    logger.info(f"🎬 [Clip] Comando START enviado para: {scene}")
                    self.clip_writer.start_clip(scene, list(self.pre_roll_buffer), video_file_name)
                # Transición: De RIESGO a NO RIESGO
                elif not current and prev:
                    logger.info(f"🎬 [Clip] Comando STOP enviado para: {scene}")
                    self.clip_writer.stop_clip(scene)

            # Actualizar el estado anterior
            self.previous_risk_states[scene] = current
            if current:
                any_risk = True
                self.db_logger.log_event(
                    scene_name=scene,
                    ts=data["time"],
                    scene_active=data["scene"],
                    risk_active=data["risk"],
                    video_file=video_file_name
                )
        # --- Activar Baliza (no bloqueante) ---
        if self.cfg.BEACON_ENABLED and any_risk:
            self.beacon.trigger_alarm()

    # -------------------------
    # Visualización
    # -------------------------
    def _visualize(self, frame, detections, results):
        if not self.cfg.VISUALIZE:
            return

        box_annot = BoxAnnotator(thickness=1, color_lookup=ColorLookup.INDEX)
        lab_annot = LabelAnnotator(color_lookup=ColorLookup.INDEX, text_padding=3, text_scale=0.35, text_thickness=0, smart_position=True)
        labels = [detections["objects"].data["class_name"][i] for i in range(len(detections["objects"].data["class_name"]))]

        frame = box_annot.annotate(scene=frame, detections=detections["objects"])
        frame = lab_annot.annotate(scene=frame, detections=detections["objects"], labels=labels)

        risk_dict = {
            "extraccion_stickout": "Pie dentro del Radio de Giro",
            "acople_pintubular": "Persona entre la llaveTM120 y Tubular",
            "cabron_abierto": "Persona cerca al Cabron Abierto",
            "pickup_tubular": "Manos en el Elevador/Brazotaladro",
            "tubular_pendulando": "Golpe por Tubular Pendulado",
            "zona_riesgo_pickup_tubular": "Golpe por Pickup Tubular",
            "acople_pintubular_mano_safata": "Atrapamiento por mano en la safata"
        }
        

        lines = [
            f"Escena {k}: {'SI' if v['scene'] else 'NO'} | Riesgo: {risk_dict[k] if v['risk'] else 'NO'}"
            for k, v in results.items()
        ]

        # print(lines)
        draw_hud(frame, self.fps_smoothed, lines, detections)


    # -------------------------
    # Limpieza
    # -------------------------
    def cleanup(self):
        logger.info("🔻 Finalizando y liberando recursos...")
        if self.beacon: self.beacon.stop_controller()
        if self.clip_writer: self.clip_writer.stop()
        self.db_logger.stop_logger()
        if self.cap: self.cap.release()
        if self.video_writer: self.video_writer.release()
        cv2.destroyAllWindows()
        if self.monitor:
            self.monitor.finalize(output_dir=self.cfg.LOG_DIR)
        logger.info("✅ Sesión finalizada correctamente.")


# =============================
# Punto de entrada
# =============================
if __name__ == "__main__":
    cfg = Config()
    app = RiskDetectionApp(cfg)
    while app.keep_running:
        try:
            app.setup()
            app.run()
        except Exception as e:
            logger.exception(f"❌ Error en ejecución principal: {e}")
            logger.info("♻️ Reiniciando en 15 segundos...")
            time.sleep(15)
        if not app.keep_running:
            break
    sys.exit(0)
