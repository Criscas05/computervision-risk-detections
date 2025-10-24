# risk_detection/config.py
import numpy as np
import torch

class Config:
    # -------- modelos --------
    MODEL_OBJ = "risk_detection/trained_model/yolo11s_sintest.pt"
    MODEL_POSE = "risk_detection/trained_model/yolo11m-pose.pt"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- fuente de video --------
    # VIDEO_SOURCE="rtsp://user:pass@ip:port/..."

    # VIDEO_SOURCE = "videos/ch0005_20250522T123857Z_20250522T133145Z_X00010000868000000-00.17.29.040-00.22.49.650-seg04.mp4"
    # VIDEO_SOURCE = "videos/ch0005_20250810T112823Z_20250810T122124Z_X00010000345000000-00.44.12.724-00.53.00.594.mp4" # Escenas sinteticas completas
    # VIDEO_SOURCE = "videos/ch0005_20250522T133145Z_20250522T142433Z_X00010000867000000-00.23.05.353-00.24.22.387-seg07.mp4" #Escena real - Extracción del stickout/pie dentro del circulo
    # VIDEO_SOURCE = "videos/escena_extraccion_stickout_y_brazotaladro_agarra_tubular.mp4"
    VIDEO_SOURCE = "risk_detection/videos/escena_sintetica_cabron_abierto_pie_cerca.mp4" #Escena sintetica- Cabron Abierto con pie cerca
    # VIDEO_SOURCE = "videos/escena_sintetica_elevador_con_tubular_manos_en_elevador.mp4" #Escena sintetica - Elevador agarra el tubular
    # VIDEO_SOURCE = "videos/escena_pie_dentro_de_circulo_manoasafata_pintubular_cerca.mp4" # Escena sintetica - Pie dentro del circulo con tubular cerca
    # VIDEO_SOURCE = "videos/escena_pie_dentro_de_circulo_manoasafata_pintubular_cerca2.mp4" # Escena sintetica 2 - Pie dentro del circulo con tubular cerca
    # VIDEO_SOURCE = "videos/escena_sintetica_riesgo_persona_entre_tubular_stickout.mp4" # Escena sintetica - persona entre tubular pendulado y stickout
    # VIDEO_SOURCE = "videos/escena_real_acople_tubular_stickout_activacion_llavetm120.mp4"
    # VIDEO_SOURCE = "videos/escena_sintetica_acople_tubular_stickout_persona_entre_stickout_llavetm120.mp4"
    # VIDEO_SOURCE = "videos/escena_sintetica_acople_tubular_stickout_persona_entre_stickout_llavetm120-cut-merged-1760744002530-cut-merged-1760744172973.mp4"
    # VIDEO_SOURCE = "risk_detection/videos/ch0005_20250522T133145Z_20250522T142433Z_X00010000867000000-00.23.05.353-00.24.22.387-seg07-merged-1761015701433.mp4"


    RESIZE = (1152, 648)
    CONF_OBJ = 0.4
    CONF_POSE = 0.5

    # -------- visualización --------
    VISUALIZE = True
    WRITE_OUTPUT = False
    OUTPUT_PATH = "output.mp4"

    # -------- índices pose --------
    FEET_IDXS = (15, 16)
    HAND_IDXS = (9, 10)

    # -------- polígonos zona riesgo --------
    POLIGONO_RIESGO_STICKOUT = np.array([
        [447.6,404.861],[447.6,420.289],[465,432],[517,439],[578,432],
        [606.154,416.745],[612.412,400.691],[604.485,386.93],[581.328,378.382],
        [542,373],[524.999,373.587],[505,375],[475,383],[459.909,392.351],[447.6,404.861]
    ], dtype=np.int32)

    # POLIGONO_RIESGO_STICKOUT = np.array([
    # [441.735, 356.126],[453.298, 343.954],[471.555, 335.034],[493.465, 328.952],[511.722, 324.491],
    # [535.66, 322.87],[562.032, 325.302],[585.97, 330.574],[601.794, 338.683],[607.879, 348.82],
    # [607.068, 358.146],[597.331, 367.877],[581.102, 376.392],[564.872, 382.879],[547.832, 386.934],
    # [532.009, 388.556],[512.128, 390.178],[493.059, 388.556],[478.453, 386.529],[461.818, 382.068],
    # [449.24, 376.392],[442.343, 366.661]
    # ], dtype=np.int32)

    # Polígono para riesgo de 'Tubular Pendulando'
    POLIGONO_RIESGO_PIN_TUBULAR = np.array([
        [648.09,299.612],[694.554,376.517],[618.179,389.497],[578.395,312.429]
    ], dtype=np.int32)

    # Polígono para riesgo de 'Acople Pintubular usuario cerca a llavetm120'
    POLIGONO_RIESGO_STICKOUT_LLAVETM120 = np.array([
        [445.351,281.2],[447.405,324.957],[628.626,318.8],[625.545,280.835]
    ], dtype=np.int32)

    # -------- umbrales y persistencias --------

    # Escena -> Extracción stickout (Brazotaladro extrae el Stickout de la profundidad y persona coloca el pie dentro del circulo de rotación de las orejas)

    EXTR_OVERLAP_MIN = 0.05     # Umbral minimo de solapamiento entre el brazotaladro y el stickout
    EXTR_ALIGN_RATIO = 0.8      # Indica, máximo cuanto procentaje pueden estar desalineados ambos boundingbox verticalmente
    EXTR_DIST_PX = 10.0         # Umbral de distancia máxima en pixeles que debe haber entre ambos objetos
    EXTR_SCENE_ON = 5           # frames consecutivos para activar la escena
    EXTR_SCENE_OFF = 5          # frames consecutivos para desactivar la escena
    EXTR_RISK_ON = 5            # frames consecutivos para activar el riesgo
    EXTR_RISK_OFF = 5           # frames consecutivos para desactivar el riesgo

    # Escena -> Cabron abierto (Persona cerca al cabrón cuando se encuentra abierto)
    CABRON_PIE_PROX_PX = 20     # Umbral de distancia maxima en pixeles entre el cabrón y la persona
    CABRON_SCENE_ON = 5         # frames consecutivos para activar la escena
    CABRON_SCENE_OFF = 5        # frames consecutivos para desactivar la escena
    CABRON_RISK_ON = 5          # frames consecutivos para activar el riesgo
    CABRON_RISK_OFF = 5         # frames consecutivos para desactivar el riesgo

    #Escena -> Pickup Tubular (Persona toca el brazotaladro cuando él se encuentra acoplado al tubular horizontal)
    PICKUP_OVERLAP_MIN = 0.05   # Umbral minimo de solapamiento entre el brazotaladro y el tubular
    PICKUP_DIST_PX = 30.0       # Umbral de distancia máxima en pixeles que debe haber entre ambos objetos
    PICKUP_SCENE_ON = 5         # frames consecutivos para activar la escena
    PICKUP_SCENE_OFF = 5        # frames consecutivos para desactivar la escena
    PICKUP_RISK_ON = 5          # frames consecutivos para activar el riesgo
    PICKUP_RISK_OFF = 5         # frames consecutivos para desactivar el riesgo

    # Escena -> Tubular pendulando (Persona pasa entre el stickout y el pintubular pendulado)
    PEND_LINE_RATIO_X = 0.6     # línea vertical al 60% del ancho
    PEND_SCENE_ON = 5           # frames consecutivos para activar la escena
    PEND_SCENE_OFF = 5          # frames consecutivos para desactivar la escena
    PEND_RISK_ON = 5            # frames consecutivos para activar el riesgo
    PEND_RISK_OFF = 5           # frames consecutivos para desactivar el riesgo

    # Escena -> Acople pintubular (Usuario pasa entre el stickout y la llavetm120 ACOPLE_WINDOW_SEC segundos despues de acoplados manualmente)
    ACOPLE_HEIGHT_BUFFER = 20   # Cantidad de alturas/frames que almacenaria la ventana de validación
    ACOPLE_INC_MIN = 0.40       # +40% de variación respecto a la media reciente
    ACOPLE_WINDOW_SEC = 100.0    # ventana de riesgo post-confirmación
    ACOPLE_AREA_MIN_STICKOUT = 9000
    ACOPLE_SCENE_ON = 6        # frames consecutivos para activar la escena
    ACOPLE_RISK_ON = 5          # frames consecutivos para activar el riesgo
    ACOPLE_RISK_OFF = 5         # frames consecutivos para desactivar el riesgo
