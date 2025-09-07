import cv2
import numpy as np
from ultralytics import YOLO
from vidgear.gears import CamGear

# Impor dari file lokal
import config
from vehicle_counter import VehicleCounter

# --- PERUBAHAN: Tambahkan konstanta untuk frame skipping ---
# Angka ini berarti kita hanya akan memproses 1 dari setiap 3 frame.
# Anda bisa menaikkan angka ini (misal, 5) jika video masih terasa lambat.
FRAME_SKIP_RATE = 3
# --------------------------------------------------------

# Variabel global untuk menyimpan titik-titik yang diklik oleh user
points = []


def mouse_callback(event, x, y, flags, param):
    """
    Fungsi callback untuk menangani event mouse.
    Menyimpan koordinat saat user mengklik tombol kiri mouse.
    """
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"Titik ditambahkan: ({x}, {y}). Total: {len(points)} titik.")


def draw_polygon_ui(frame):
    """
    Menampilkan UI interaktif bagi user untuk menggambar poligon di atas frame.
    Mengembalikan poligon yang sudah jadi atau None jika dibatalkan.
    """
    global points
    window_name = "Gambar Zona Hitung Anda"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n--- INSTRUKSI MENGGAMBAR ZONA ---")
    print("1. Klik pada gambar untuk menambahkan titik sudut poligon.")
    print("2. Minimal 3 titik diperlukan untuk membuat sebuah zona.")
    print("3. Tekan tombol 'ENTER' untuk mengkonfirmasi dan memulai deteksi.")
    print("4. Tekan tombol 'C' untuk menghapus semua titik dan mengulang dari awal.")
    print("5. Tekan tombol 'Q' untuk membatalkan dan menggunakan zona default.")
    print("---------------------------------\n")

    while True:
        temp_frame = frame.copy()
        for point in points:
            cv2.circle(temp_frame, tuple(point), 5, (0, 0, 255), -1)
        if len(points) > 1:
            cv2.polylines(temp_frame, [np.array(points, np.int32)], isClosed=False, color=(
                0, 255, 255), thickness=2)

        cv2.putText(temp_frame, "Klik utk menambah titik. Tekan ENTER utk selesai.",
                    (20, 40), 0, 1, (255, 255, 255), 2)
        cv2.putText(temp_frame, "Tekan 'c' utk hapus, 'q' utk batal.",
                    (20, 80), 0, 1, (255, 255, 255), 2)

        cv2.imshow(window_name, temp_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            cv2.destroyWindow(window_name)
            return None
        if key == ord('c'):
            points = []
            print("Semua titik dihapus. Silakan gambar ulang.")
        if key == 13:
            if len(points) < 3:
                print("Error: Minimal 3 titik diperlukan. Silakan tambahkan titik lagi.")
            else:
                cv2.destroyWindow(window_name)
                return np.array(points, np.int32)


def main():
    model = YOLO(config.MODEL_PATH)
    stream = CamGear(source=config.YOUTUBE_URL,
                     stream_mode=True, logging=True).start()

    first_frame = stream.read()
    if first_frame is None:
        print("Error: Gagal membaca frame dari stream CCTV.")
        stream.stop()
        return

    user_defined_zone = draw_polygon_ui(first_frame)

    if user_defined_zone is None:
        print("Tidak ada zona yang digambar, menggunakan zona default dari config.py.")
        zone_polygon = config.ZONE_POLYGON
    else:
        print("Zona berhasil dibuat oleh user.")
        zone_polygon = user_defined_zone

    counter = VehicleCounter(zone_polygon, config)

    # --- PERUBAHAN: Variabel untuk logika frame skipping ---
    frame_count = 0
    last_results = None
    # -----------------------------------------------------

    print("\nMemulai deteksi dan penghitungan...")
    while True:
        frame = stream.read()
        if frame is None:
            break

        frame_count += 1
        annotated_frame = frame.copy()  # Mulai dengan frame asli

        # Hanya jalankan deteksi berat pada frame tertentu
        if frame_count % FRAME_SKIP_RATE == 0:
            results = model.track(frame, persist=True)
            last_results = results  # Simpan hasil deteksi terakhir

        # Untuk semua frame (termasuk yang di-skip), gunakan hasil terakhir untuk diproses
        # Ini membuat tampilan visual tetap mulus
        if last_results is not None:
            annotated_frame = counter.process_frame(frame, last_results)

        cv2.imshow("Analisis Volume Kendaraan", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    stream.stop()
    print("Aplikasi ditutup.")


if __name__ == "__main__":
    main()
