# Bootcamp de Visión Artificial

Repaso práctico e intensivo de visión por computador aplicada a vídeo: teoría + código ejecutable, desde PyTorch y
YOLO hasta tracking multiobjeto, Re-Identification, homografías, análisis temporal, optimización de inferencia y
puesta en producción. Termina con un **proyecto propio** (detección → tracking → ReID → homografía → datos
estructurados → optimización) sobre vídeo real, orientado a analítica deportiva y de personas.

---

## Cómo usar cada módulo

Cada carpeta tiene **un notebook Jupyter** (`NN_nombre.ipynb`) con:
- Teoría explicada desde cero (intuición → fórmula → código).
- Celdas ejecutables. Las marcadas ✍️ **EJERCICIO** son para que las implementes tú (la solución está justo debajo, plegada en una celda `# SOLUCIÓN`; intenta no mirarla).

Abre Jupyter con:
```bash
git clone https://github.com/jcandel01/vision-bootcamp.git && cd vision-bootcamp
python -m venv .venv && source .venv/bin/activate && pip install -r 00_setup/requirements.txt
jupyter lab
```

**Cada notebook es independiente**: su primera celda **⚙️ SETUP** instala los paquetes que falten, descarga el
vídeo de ejemplo a `data/` y (si el notebook usa `vision_utils/` o `11_produccion/`) clona el repo. Los pesos YOLO
los descarga Ultralytics solo. Ejecútala siempre primero. `00_setup` es opcional (comprobar entorno y descargar todo).

### En Google Colab
1. [colab.research.google.com](https://colab.research.google.com) → **Abrir notebook → GitHub** → busca
   `jcandel01/vision-bootcamp` y elige el notebook.
2. **Entorno de ejecución → Cambiar tipo → GPU (T4)**.
3. Colab no guarda en el repo: **Archivo → Guardar una copia en GitHub/Drive** si quieres conservar tus ejercicios.

Regla de oro: **no leas en pasivo**. Ejecuta, cambia parámetros, mira qué se rompe.

---

## Plan (10 h/día)

Prioridad: ⭐⭐⭐ imprescindible · ⭐⭐ importante · ⭐ si da tiempo

### Día 1 — Refresco de PyTorch + YOLO, detección y datos
| Horas | Notebook | Prioridad |
|---|---|---|
| 0:00–0:30 | `00_setup` — entorno, Jupyter, pesos y vídeos (opcional: cada notebook se prepara solo) | ⭐⭐⭐ |
| 0:30–1:30 | `01_imagen_video` — OpenCV, imagen como tensor, vídeo, FFmpeg | ⭐⭐ |
| 1:30–3:30 | `02_pytorch` — tensores, autograd, bucle de entrenamiento, CNN, transfer learning | ⭐⭐⭐ |
| 3:30–7:00 | `03_deteccion` — IoU ✍️, NMS ✍️, mAP, YOLO por dentro, Ultralytics, fine-tuning, DETR | ⭐⭐⭐ |
| 7:00–8:30 | `04_segmentacion_pose` — segmentación, SAM, pose, OKS | ⭐⭐ |
| 8:30–10:00 | `05_datasets` — formatos, CVAT/Roboflow, leakage, limpieza ✍️ | ⭐⭐⭐ |

### Día 2 — El núcleo del puesto: tracking, identidades, geometría
| Horas | Notebook | Prioridad |
|---|---|---|
| 0:00–4:00 | `06_tracking` — Kalman ✍️, húngaro, SORT desde cero ✍️, ByteTrack/OC-SORT/BoT-SORT, MOTA/IDF1/HOTA | ⭐⭐⭐ |
| 4:00–6:00 | `07_reid` — embeddings, distancia coseno, triplet loss, re-link de tracklets ✍️ | ⭐⭐⭐ |
| 6:00–8:30 | `08_geometria` — cámara pinhole, calibración, homografía DLT ✍️, vista cenital, multicámara | ⭐⭐ |
| 8:30–10:00 | `09_temporal` — trayectorias, suavizado, velocidad, interacciones, acciones | ⭐⭐ |

### Día 3 — Rendimiento, producción y proyecto
| Horas | Notebook | Prioridad |
|---|---|---|
| 0:00–2:30 | `10_optimizacion` — profiling, ONNX, CoreML/TensorRT, FP16/INT8, pipeline | ⭐⭐⭐ |
| 2:30–3:30 | `11_produccion` — código modular, FastAPI, Docker, MLOps | ⭐⭐ |
| 3:30–7:30 | `12_proyecto_final` — pipeline completo sobre tu vídeo + evaluación | ⭐⭐⭐ |

**Si solo tienes 2 días:** Día 1 = 00, 02 (rápido), 03, 05, 06. Día 2 = 07, 08, 10, 12 (versión corta).
Los notebooks 01, 04 y 09 léelos en diagonal.

---

## Estructura

```
NN_tema/NN_tema.ipynb   ← un notebook por módulo (teoría + código + ejercicios ✍️)
vision_utils/           ← código compartido (soluciones de referencia, reutilizadas en 07 y 12):
  synthetic.py          escena sintética con GT: cruces, oclusiones, desapariciones, embeddings
  mot_metrics.py        MOTA, IDF1 y HOTA implementadas desde cero (léelo: ~130 líneas)
  tracker.py            Kalman + húngaro: SORT / ByteTrack / + ReID
  postprocess.py        interpolación de huecos y stitching de tracklets
  analytics.py          cinemática (Savitzky–Golay), eventos con histéresis, resúmenes
11_produccion/          ← mini-servicio real: FastAPI + Dockerfile + tests (pytest)
data/                   ← pesos, vídeos y salidas
```

Las soluciones de los ejercicios están en celdas `# SOLUCIÓN` **plegadas** justo debajo de cada ejercicio
(en JupyterLab aparecen como "…": haz clic para desplegar). Si ejecutas todo de corrido funciona igual, porque la
solución redefine la función.

---

## Checklist final ("¿sabría explicarlo en 1 minuto en una pizarra?")

- [ ] IoU, NMS y por qué DETR no necesita NMS
- [ ] Precision / Recall / mAP@50 / mAP@50:95
- [ ] Qué cambió de YOLOv5 → YOLOv8 → YOLO11 (anchor-free, decoupled head…)
- [ ] Filtro de Kalman en tracking (predict / update, estado)
- [ ] Algoritmo húngaro y matriz de costes
- [ ] SORT vs DeepSORT vs ByteTrack vs OC-SORT vs BoT-SORT (una frase para cada uno)
- [ ] MOTA vs IDF1 vs HOTA (y cuál prefieres para identidades: IDF1/HOTA)
- [ ] Qué es un ID switch y 3 formas de reducirlos
- [ ] ReID: embedding, distancia coseno, triplet loss, galería
- [ ] Homografía: qué es, cuántos puntos, cuándo es válida, qué punto del bbox proyectar
- [ ] Intrínsecos vs extrínsecos, distorsión
- [ ] ONNX vs TensorRT, FP16 vs INT8, dónde está el cuello de botella de un pipeline de vídeo
- [ ] Data leakage al dividir por frames vs por vídeo
- [ ] El proyecto final resumido en 2 minutos con números (FPS, mAP, IDF1)
