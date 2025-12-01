# risk_detection/in_out/video_clip_writer.py
import cv2
import queue
import threading
import os
import sys
import time
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

class VideoClipWriter:
    """
    Un controlador asíncrono (en un hilo) para grabar clips de video.
    Maneja múltiples grabaciones de "escenas" simultáneamente.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.running = False
        self.thread = None
        
        # Dos colas: una para comandos (START/STOP) y otra para frames
        self.command_queue = queue.Queue()
        self.frame_queue = queue.Queue(maxsize=120) # maxsize previene uso de RAM infinito si el hilo se atora
        
        # Diccionario para almacenar los grabadores de video activos
        # Clave: scene_name (str)
        # Valor: (cv2.VideoWriter, str: file_path)
        self.active_writers = {}
        
        # Asegurarse de que el directorio de clips exista
        os.makedirs(cfg.CLIPS_DIR, exist_ok=True)
        logger.info(f"📹 VideoClipWriter inicializado. Clips se guardarán en: {cfg.CLIPS_DIR}")

    def start_controller(self):
        """Inicia el hilo worker."""
        if self.running:
            logger.warning("[ClipWriter] El controlador ya está en ejecución.")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_worker, daemon=True)
        self.thread.start()
        logger.info("🟢 [ClipWriter] Hilo worker iniciado.")

    def stop(self):
        """Detiene el hilo worker de forma limpia."""
        logger.info("🟡 [ClipWriter] Deteniendo hilo worker...")
        self.running = False
        self.command_queue.put(("STOP_ALL", None, None)) # Comando para destrabar .get()
        if self.thread:
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                logger.error("🔴 [ClipWriter] El hilo no pudo detenerse a tiempo.")
        
        # Cerrar todos los archivos de video que hayan quedado abiertos
        for writer, file_path in self.active_writers.values():
            try:
                writer.release()
                logger.info(f"  > Archivo de clip cerrado (por stop): {file_path}")
            except Exception as e:
                logger.error(f"  > Error cerrando clip {file_path}: {e}")
        self.active_writers.clear()
        logger.info("🟢 [ClipWriter] Hilo worker detenido limpiamente.")

    # --- Métodos llamados por el Hilo Principal (Productor) ---

    def put_frame(self, frame):
        """Añade un frame a la cola de frames (no bloqueante)."""
        if not self.running: return
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            # Esto es normal si el hilo de I/O se atrasa.
            # Se descarta el frame más antiguo para priorizar el tiempo real.
            pass 

    def start_clip(self, scene_name, pre_roll_frames, video_file_name):
        """Envía un comando para INICIAR la grabación de un clip (no bloqueante)."""
        if not self.running: return
        logger.debug(f"[ClipWriter] Comando START recibido para: {scene_name}")
        self.command_queue.put(("START", scene_name, (pre_roll_frames, video_file_name)))

    def stop_clip(self, scene_name):
        """Envía un comando para DETENER la grabación de un clip (no bloqueante)."""
        if not self.running: return
        logger.debug(f"[ClipWriter] Comando STOP recibido para: {scene_name}")
        self.command_queue.put(("STOP", scene_name, None))

    # --- Método ejecutado por el Hilo Worker (Consumidor) ---

    def _run_worker(self):
        """
        Half-private method. El bucle principal del hilo worker.
        """
        while self.running:
            # Procesar todos los comandos pendientes (START/STOP)
            try:
                while not self.command_queue.empty():
                    cmd, scene_name, data = self.command_queue.get_nowait()
                    
                    if cmd == "START":
                        pre_roll_frames, video_file_name = data
                        self._handle_start(scene_name, pre_roll_frames, video_file_name)
                    elif cmd == "STOP":
                        self._handle_stop(scene_name)
                    elif cmd == "STOP_ALL":
                        return # Salir del bucle

            except queue.Empty:
                pass # No hay comandos, continuar
            except Exception as e:
                logger.error(f"[ClipWriter] Error procesando comando: {e}", exc_info=True)

            # Procesar todos los frames pendientes
            try:
                if not self.active_writers:
                    # Si no hay grabaciones activas, vaciar la cola de frames
                    # para evitar que el búfer de pre-roll crezca indefinidamente
                    while not self.frame_queue.empty():
                        self.frame_queue.get_nowait()
                else:
                    # Si hay grabaciones, escribir frames
                    while not self.frame_queue.empty():
                        frame = self.frame_queue.get_nowait()
                        for writer, _ in self.active_writers.values():
                            writer.write(frame)
            
            except queue.Empty:
                pass # No hay frames, continuar
            except Exception as e:
                logger.error(f"[ClipWriter] Error escribiendo frame: {e}", exc_info=True)

            # Dormir un poco para ceder el GIL y no consumir 100% CPU
            # en un bucle vacío si no hay frames/comandos.
            time.sleep(0.001)

    def _handle_start(self, scene_name, pre_roll_frames, video_file_name):
        """Lógica para iniciar una grabación (llamado por el worker)."""
        if scene_name in self.active_writers:
            logger.warning(f"[ClipWriter] 'START' ignorado: {scene_name} ya está grabando.")
            return

        try:
            # Definir dimensiones y FPS del video
            if not pre_roll_frames:
                logger.error(f"[ClipWriter] No se puede iniciar {scene_name}: búfer de pre-roll vacío.")
                return
            
            first_frame = pre_roll_frames[0]
            height, width, _ = first_frame.shape
            # Asumimos que el FPS del pre-roll es el FPS de grabación
            fps = len(pre_roll_frames) / max(self.cfg.CLIP_PREROLL_SEC, 1.0)
            
            # Crear nombre de archivo único
            # ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            # file_name = f"{scene_name}_{ts}.mp4"
            file_path = os.path.join(self.cfg.CLIPS_DIR, video_file_name)
            
            # Crear el objeto VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
            
            if not writer.isOpened():
                raise IOError(f"cv2.VideoWriter no pudo abrir el archivo: {file_path}")

            # Escribir el búfer de pre-roll
            for frame in pre_roll_frames:
                writer.write(frame)
                
            # Añadir a la lista de grabadores activos
            self.active_writers[scene_name] = (writer, file_path)
            logger.info(f"🟢 [ClipWriter] Grabación INICIADA para {scene_name} → {file_path}")

        except Exception as e:
            logger.error(f"[ClipWriter] Fallo al iniciar clip para {scene_name}: {e}", exc_info=True)
            if 'writer' in locals() and writer.isOpened():
                writer.release()

    def _handle_stop(self, scene_name):
        """Lógica para detener una grabación (llamado por el worker)."""
        if scene_name not in self.active_writers:
            logger.warning(f"[ClipWriter] 'STOP' ignorado: {scene_name} no estaba grabando.")
            return
        
        try:
            writer, file_path = self.active_writers.pop(scene_name)
            writer.release()
            logger.info(f"🔴 [ClipWriter] Grabación DETENIDA para {scene_name} → {file_path}")
        except Exception as e:
            logger.error(f"[ClipWriter] Fallo al detener clip para {scene_name}: {e}", exc_info=True)
