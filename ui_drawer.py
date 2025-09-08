# ui_drawer.py

import cv2
import numpy as np


class UIDrawer:
    """
    Kelas terpisah untuk menangani semua interaksi UI,
    khususnya untuk menggambar poligon zona hitung.
    """

    def __init__(self):
        self.points = []

    def _mouse_callback(self, event, x, y, flags, param):
        """Callback untuk menangani klik mouse."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x, y])
            print(
                f"Titik ditambahkan: ({x}, {y}). Total: {len(self.points)} titik.")

    def draw_polygon_ui(self, frame):
        """
        Menampilkan UI interaktif untuk menggambar poligon.
        Mengembalikan poligon atau None jika dibatalkan.
        """
        self.points = []  # Reset poin setiap kali fungsi dipanggil
        window_name = "Gambar Zona Hitung Anda"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        print("\n--- INSTRUKSI MENGGAMBAR ZONA ---")
        print("1. Klik pada gambar untuk menambahkan titik sudut poligon.")
        print("2. Minimal 3 titik diperlukan.")
        print("3. Tekan 'ENTER' untuk konfirmasi.")
        print("4. Tekan 'C' untuk hapus semua titik.")
        print("5. Tekan 'Q' untuk batal dan pakai zona default.")
        print("---------------------------------\n")

        while True:
            temp_frame = frame.copy()
            for point in self.points:
                cv2.circle(temp_frame, tuple(point), 5, (0, 0, 255), -1)
            if len(self.points) > 1:
                cv2.polylines(temp_frame, [np.array(
                    self.points, np.int32)], isClosed=False, color=(0, 255, 255), thickness=2)

            cv2.putText(temp_frame, "Klik: Tambah Titik | ENTER: Selesai",
                        (20, 40), 0, 1, (255, 255, 255), 2)
            cv2.putText(temp_frame, "C: Hapus | Q: Batal/Default",
                        (20, 80), 0, 1, (255, 255, 255), 2)

            cv2.imshow(window_name, temp_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                cv2.destroyWindow(window_name)
                return None
            if key == ord('c'):
                self.points = []
                print("Semua titik dihapus. Silakan gambar ulang.")
            if key == 13:  # Tombol Enter
                if len(self.points) < 3:
                    print("Error: Minimal 3 titik diperlukan.")
                else:
                    cv2.destroyWindow(window_name)
                    return np.array(self.points, np.int32)
