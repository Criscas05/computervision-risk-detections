# risk_detection/main_realtime.py
import time, cv2, supervision as sv
from ultralytics import YOLO
from config import Config
from risk_engine import RiskEngine
from utils.visualization import draw_hud
import numpy as np
from in_out.db_logger import DBLogger
# import threading

DBLogger = DBLogger()

def main():
    cfg = Config()
    print(f"🟢 Device: {cfg.DEVICE.upper()}")

    model_obj = YOLO(cfg.MODEL_OBJ)
    model_pose = YOLO(cfg.MODEL_POSE)
    
    db_path = DBLogger.start_logger(output_dir="logs")

    cap = cv2.VideoCapture(cfg.VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir fuente: {cfg.VIDEO_SOURCE}")

    if cfg.WRITE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        out = cv2.VideoWriter(cfg.OUTPUT_PATH, fourcc, fps, cfg.RESIZE)
    else:
        out = None

    engine = RiskEngine(cfg)
    prev = time.time()
    fps_smoothed = None

    try:
        while True:

            # print(f"Bucle CV ejecutándose en hilo: {threading.current_thread().name}")

            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, cfg.RESIZE)

            # ---- inference objetos ----
            res_o = model_obj(frame, device=cfg.DEVICE, conf=cfg.CONF_OBJ, verbose=False)
            det_obj = sv.Detections.from_ultralytics(res_o[0])

            # Creamos un "class_name" ndarray (para acelerar filtros)
            class_ids = det_obj.class_id.astype(int)
            names = res_o[0].names
            det_obj.data["class_name"] = np.array([names[i] for i in class_ids])

            # ---- inference pose ----
            res_p = model_pose(frame, device=cfg.DEVICE, conf=cfg.CONF_POSE, verbose=False)

            # ---- engine ----
            results = engine.process(det_obj, res_p, frame if cfg.VISUALIZE else None)

            # --- Registrar eventos ---
            for scene_name, data in results.items():
                if data["risk"]:
                    DBLogger.log_event(
                            scene_name=scene_name,
                            ts=data["time"],
                            scene_active=data["scene"],
                            risk_active=data["risk"]
                        )

            # ---- overlay: cajas y labels ----
            if cfg.VISUALIZE:
                box_annot = sv.BoxAnnotator(thickness=1, color_lookup=sv.ColorLookup.INDEX)
                lab_annot = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX, text_padding=3,
                                            text_scale=0.35, text_thickness=0, smart_position=True)
                labels = [names[i] for i in class_ids]
                frame = box_annot.annotate(scene=frame, detections=det_obj)
                frame = lab_annot.annotate(scene=frame, detections=det_obj, labels=labels)

            # ---- fps ----
            now = time.time()
            inst_fps = 1.0 / max(now - prev, 1e-6); prev = now
            fps_smoothed = inst_fps if fps_smoothed is None else (fps_smoothed * 0.9 + inst_fps * 0.1)

            # ---- HUD ----
            if cfg.VISUALIZE:
                lines = []
                for k, v in results.items():
                    tag = f"{k}: {'ON' if v['scene'] else 'off'} | risk: {'Alerta' if v['risk'] else '-'}"
                    lines.append(tag)
                draw_hud(frame, fps_smoothed, lines)

            # ---- IO ----
            if cfg.VISUALIZE:
                cv2.imshow("RiskEngine", frame)
            if out is not None:
                out.write(frame)
            if cfg.VISUALIZE and (cv2.waitKey(1) & 0xFF == ord('q')):
                break
    finally:
        print("Cerrando recursos...")
        DBLogger.stop_logger()

        cap.release()
        if out: out.release()
        cv2.destroyAllWindows()

        print(f"Sesión finalizada. Eventos guardados en: {db_path}")


if __name__ == "__main__":
    main()
