# config.py

import numpy as np
import cv2

# =============================================================================
# PENGATURAN UTAMA (WAJIB DISESUAIKAN)
# =============================================================================

YOUTUBE_URL = "https://www.youtube.com/live/MJOolZOl3i0?si=yTLpdCVw4L58tspT"

# --- PERUBAHAN UTAMA UNTUK PERFORMA ---
# Mengganti model ke versi 's' (Small) yang jauh lebih cepat dari 'l' (Large).
# Ini adalah kompromi terbaik antara kecepatan dan akurasi.
# Jika masih lambat, bisa diturunkan lagi ke 'yolov10n.pt' (Nano).
MODEL_PATH = 'yolov10s.pt'

ZONE_POLYGON = np.array([
    [150, 250], [1150, 250], [1150, 500], [100, 500]
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

TEXT_START_POSITION = (50, 70)
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 1.1
TEXT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)
TEXT_LINE_HEIGHT = 40

CENTER_CIRCLE_COLOR = (0, 255, 255)
CENTER_CIRCLE_RADIUS = 5
