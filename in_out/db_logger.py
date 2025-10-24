import sqlite3
import queue
import threading
import json
import time
import os
from datetime import datetime

# La cola que comunica el hilo principal (CV) con el hilo de la BBDD
data_queue = queue.Queue()

# Variables globales para manejar el hilo y la ruta del archivo
worker_thread = None
db_path = None

class DBLogger:
    def database_worker(self, db_file_path):
        """
        Este es el "worker" que se ejecuta en un hilo separado.
        Su único trabajo es escuchar la cola y escribir en la BBDD.
        """
        # Cada hilo DEBE crear su propia conexión a SQLite
        conn = None
        try:
            conn = sqlite3.connect(db_file_path)
            cursor = conn.cursor()
            
            # Definimos la tabla (si no existe)
            # Esta tabla coincide con los datos que quieres guardar
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS riesgos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    scene_name TEXT,
                    scene_active BOOLEAN,
                    risk_active BOOLEAN
                )
            """)
            conn.commit()
            print(f"🟢 [Logger] Hilo worker conectado a BBDD: {db_file_path}")

            while True:
                try:
                    # data_queue.get() bloqueará el hilo worker 
                    # No el hilo principal hasta que haya un item.
                    data = data_queue.get()

                    # 'None' es la señal que usamos para detener el hilo
                    if data is None:
                        print("🟢 [Logger] Señal de parada recibida. Terminando hilo worker.")
                        break
                    # print(f"DB Worker escribiendo en hilo: {threading.current_thread().name}")
                    # time.sleep(3)
                    # Desempaquetamos los datos que envía el hilo principal
                    (scene_name, ts, scene_active, risk_active) = data
                    
                    # Convertimos los datos para un almacenamiento seguro
                    ts_str = str(ts)  # Convertimos timestamp/datetime a string

                    # Insertamos en la base de datos
                    cursor.execute(
                        "INSERT INTO riesgos (timestamp, scene_name, scene_active, risk_active) VALUES (?, ?, ?, ?)",
                        (ts_str, scene_name, scene_active, risk_active)
                    )
                    
                    # Hacemos commit de la transacción
                    conn.commit()

                except sqlite3.Error as e:
                    print(f"🔴 [Logger] Error de SQLite: {e}")
                except Exception as e:
                    print(f"🔴 [Logger] Error en hilo worker: {e}")

        finally:
            if conn:
                conn.close()
                print("🟢 [Logger] Conexión a BBDD cerrada.")


    def start_logger(self, output_dir="logs"):
        """
        Inicia el hilo worker de la base de datos.
        Debe llamarse una vez al inicio del programa.
        """
        global worker_thread, db_path
        
        # Asegurarse de que la carpeta de logs exista
        os.makedirs(output_dir, exist_ok=True)
        
        # Crear un nombre de archivo único por día
        today = datetime.now().strftime('%Y-%m-%d')
        db_path = os.path.join(output_dir, f"registros_riesgos_{today}.db")
        
        # Iniciamos el hilo worker
        # 'daemon=True' asegura que el hilo se cierre si el script principal falla
        worker_thread = threading.Thread(target=self.database_worker, args=(db_path,), daemon=True)
        worker_thread.start()
        
        print(f"🟢 [Logger] Hilo worker iniciado. Guardando en: {db_path}")
        return db_path


    def log_event(self, scene_name: str, ts, scene_active: bool, risk_active: bool):
        """
        Esta es la función que tu hilo principal llamará.
        Es súper rápida porque solo pone datos en una cola (RAM).
        """
        try:
            data = (scene_name, ts, scene_active, risk_active)
            # data_queue.put_nowait(data) # Opción si no quieres bloquear nunca
            data_queue.put(data) # .put() es seguro y rápido
        except Exception as e:
            print(f"🔴 [Logger] Error al encolar evento: {e}")


    def stop_logger(self):
        """
        Envía la señal de parada al hilo worker y espera a que termine.
        Debe llamarse al final de tu programa para un cierre limpio.
        """
        global worker_thread
        if worker_thread and worker_thread.is_alive():
            print("🟡 [Logger] Enviando señal de parada al worker...")
            data_queue.put(None)  # Envía la señal 'None'
            worker_thread.join(timeout=5.0) # Espera a que el hilo termine
            if worker_thread.is_alive():
                print("🔴 [Logger] El hilo worker no terminó a tiempo.")
            else:
                print("🟢 [Logger] Hilo worker detenido limpiamente.")
        else:
            print("🟡 [Logger] El hilo worker no estaba corriendo.")