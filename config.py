# config.py

import numpy as np
import cv2

# =============================================================================
# PENGATURAN UTAMA (WAJIB DISESuaikan)
# =============================================================================

YOUTUBE_URL = "https://www.youtube.com/live/P2QdljtuKwo?si=iAfnFoQSf7xcccrH"

# --- PENGATURAN DEVICE ---
# Diubah ke "cpu" untuk mengatasi error "Torch not compiled with CUDA enabled".
# PERINGATAN: Ini akan membuat program berjalan jauh lebih lambat.
# Solusi permanen ada di penjelasan saya.
MODEL_PATH = 'yolo11m.pt'  # Menggunakan model yang terbukti ada dan cepat
DEVICE = "cuda"
STREAM_RESOLUTION = (720, 360)

# Zona default, disesuaikan dengan resolusi baru (opsional, bisa digambar ulang)
ZONE_POLYGON = np.array([
    [20, 150], [620, 150], [620, 300], [20, 300]
], np.int32)

# =============================================================================
# PENGATURAN KELAS KENDARAAN (FLEKSIBEL)
# =============================================================================

CLASS_DATA = {
    2: {"name": "Mobil", "color": (0, 255, 0)},
    3: {"name": "Motor", "color": (255, 100, 0)},
    5: {"name": "Bus", "color": (0, 0, 255)},
    7: {"name": "Truk", "color": (255, 255, 0)}
}
CLASS_ID_VEHICLES = list(CLASS_DATA.keys())

# =============================================================================
# PENGATURAN TAMPILAN (OPSIONAL)
# =============================================================================

ZONE_COLOR = (255, 255, 255)
ZONE_THICKNESS = 2
TEXT_START_POSITION = (20, 30)
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.7
TEXT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)
TEXT_LINE_HEIGHT = 25
CENTER_CIRCLE_COLOR = (0, 255, 255)
CENTER_CIRCLE_RADIUS = 5
