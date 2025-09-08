import numpy as np

# =============================================================================
# PENGATURAN UTAMA (WAJIB DISESUAIKAN)
# =============================================================================

# Link CCTV YouTube yang akan dianalisis
YOUTUBE_URL = "https://www.youtube.com/live/MJOolZOl3i0?si=yTLpdCVw4L58tspT"

# Path ke model YOLOv8. 'yolov8n.pt' adalah versi nano yang paling ringan.
MODEL_PATH = 'yolov8n.pt'

# KOORDINAT ZONA HITUNG (POLIGON)
# Ini adalah bagian PALING PENTING untuk Anda sesuaikan.
# Titik-titik ini membentuk area di mana kendaraan akan dihitung.
# Format: np.array([[x1, y1], [x2, y2], [x3, y3], ...], np.int32)
ZONE_POLYGON = np.array([
    [150, 250],  # Titik kiri-atas
    [1150, 250],  # Titik kanan-atas
    [1150, 500],  # Titik kanan-bawah
    [100, 500]   # Titik kiri-bawah
], np.int32)

# =============================================================================
# PENGATURAN TAMPILAN (OPSIONAL)
# =============================================================================

# Pengaturan untuk garis poligon zona
ZONE_COLOR = (0, 255, 255)  # Kuning (BGR)
ZONE_THICKNESS = 2

# Pengaturan untuk teks display (jumlah kendaraan)
TEXT_COLOR = (255, 255, 255)  # Putih
TEXT_POSITION = (50, 70)
TEXT_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 2
TEXT_THICKNESS = 3

# Pengaturan lingkaran pada objek yang baru dihitung
CENTER_CIRCLE_COLOR = (0, 0, 255)  # Merah
CENTER_CIRCLE_RADIUS = 5

# ID Kelas Objek Kendaraan pada dataset COCO
# 2: car, 3: motorcycle, 5: bus, 7: truck
CLASS_ID_VEHICLES = [2, 3, 5, 7]
