# 00 · Setup (30 min)

## 1. Entorno

Probado en tu Mac (M4) con el Python 3.14 de Homebrew: PyTorch 2.14, Ultralytics 8.4, MPS activo.

```bash
cd ~/Documents/cursos/vision
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r 00_setup/requirements.txt
python -m ipykernel install --user --name vision --display-name "Python (vision)"
jupyter lab
```

En Jupyter elige el kernel **Python (vision)** en cada notebook. (Si algún paquete fallase con 3.14:
`brew install python@3.12` y crea el venv con `python3.12`.)

## 2. Abre `00_setup.ipynb`

Opcional: cada notebook se prepara solo con su celda ⚙️ SETUP. Este comprueba el entorno y descarga a `data/` los pesos YOLO11 y un vídeo de personas caminando. (Ya están
descargados: los dejé al probar los notebooks.)

**Recomendado:** consigue también un vídeo **deportivo** de 20–60 s con cámara fija
(Pexels → "football match", "basketball game", o grábalo con el móvil) y guárdalo como
`data/sport.mp4`. El proyecto final (notebook 12) lo usará automáticamente. Para recortar uno largo:

```bash
ffmpeg -ss 00:01:00 -i entrada.mp4 -t 30 -c:v libx264 -an data/sport.mp4
#      inicio        entrada        duración  recodifica   sin audio
```

## 3. Notas

- **MPS** (Metal) = GPU de Apple Silicon → `device="mps"`. En la empresa: **CUDA** → `device="cuda"`/`0`.
- **TensorRT** solo existe en NVIDIA: el notebook 10 trae un bloque para probarlo en Google Colab (GPU T4 gratis).
- Tu configuración de Ultralytics tiene `datasets_dir` apuntando a `~/Documents/proyectos/yolo_ros2/datasets`:
  los datasets de ejemplo (`coco8`, `coco128`) se descargarán ahí. Cámbialo si quieres con
  `yolo settings datasets_dir=/ruta`.
- Los notebooks 02, 03 y 05 entrenan modelos durante unos minutos; el resto va rápido.
