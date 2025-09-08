# main.py

import cv2
import sys
import threading
from queue import Queue, Empty
from ultralytics import YOLO
from vidgear.gears import CamGear

import config
from vehicle_counter import VehicleCounter
from ui_drawer import UIDrawer


class VehicleDetectionApp:
    def __init__(self):
        self.config = config
        self.model = YOLO(self.config.MODEL_PATH)
        self.ui_drawer = UIDrawer()

        self.frame_queue = Queue(maxsize=2)  # Antrean untuk frame mentah
        self.processed_frame = None  # Frame yang sudah diolah
        self.is_running = threading.Event()
        self.is_running.set()

    def _stream_reader(self):
        """Thread untuk membaca frame dari stream secepat mungkin."""
        print(f"Memulai stream dari: {self.config.YOUTUBE_URL}...")
        try:
            stream = CamGear(source=self.config.YOUTUBE_URL,
                             stream_mode=True, logging=True).start()
        except Exception as e:
            print(f"Error fatal: Gagal memulai stream. Detail: {e}")
            self.is_running.clear()
            return

        while self.is_running.is_set():
            frame = stream.read()
            if frame is None:
                print("Stream berakhir atau terputus.")
                self.is_running.clear()
                break

            if not self.frame_queue.full():
                self.frame_queue.put(frame)

        stream.stop()
        print("Thread pembaca stream berhenti.")

    def _frame_processor(self, zone_polygon):
        """Thread untuk memproses frame dengan model YOLO."""
        counter = VehicleCounter(zone_polygon, self.config)
        print("Thread prosesor siap.")

        while self.is_running.is_set():
            try:
                # Ambil frame terbaru dari antrean, jangan menunggu
                frame = self.frame_queue.get(block=False)

                # Jalankan deteksi
                results = self.model.track(
                    frame, persist=True, classes=self.config.CLASS_ID_VEHICLES, verbose=False)

                # Anotasi frame dan simpan sebagai frame yang sudah diproses
                self.processed_frame = counter.process_frame(frame, results)

            except Empty:
                # Jika antrean kosong, tidak ada yang perlu diproses
                continue

        print("Thread prosesor berhenti.")

    def run(self):
        """Mempersiapkan UI dan menjalankan semua thread."""
        # Dapatkan frame pertama untuk UI gambar zona
        print("Mengambil frame pertama untuk UI...")
        stream = CamGear(source=self.config.YOUTUBE_URL,
                         stream_mode=True).start()
        first_frame = stream.read()
        stream.stop()

        if first_frame is None:
            print("Error: Gagal mendapatkan frame pertama. Aplikasi tidak bisa dimulai.")
            return

        zone_polygon = self.ui_drawer.draw_polygon_ui(first_frame)
        if zone_polygon is None:
            zone_polygon = self.config.ZONE_POLYGON
            print("Zona kustom dibatalkan, menggunakan zona default.")
        else:
            print("Zona berhasil dibuat oleh pengguna.")

        # Inisialisasi frame yang diproses dengan frame pertama
        self.processed_frame = first_frame

        # Jalankan thread
        reader_thread = threading.Thread(
            target=self._stream_reader, daemon=True)
        processor_thread = threading.Thread(
            target=self._frame_processor, args=(zone_polygon,), daemon=True)

        reader_thread.start()
        processor_thread.start()

        print("\nMemulai aplikasi... Tekan 'q' pada jendela video untuk keluar.")

        # Loop utama untuk menampilkan video
        while self.is_running.is_set():
            cv2.imshow("Analisis Volume Kendaraan - Dioptimalkan",
                       self.processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()

        # Tunggu thread selesai
        reader_thread.join(timeout=2)
        processor_thread.join(timeout=2)

    def stop(self):
        """Menghentikan semua proses dengan aman."""
        print("Perintah berhenti diterima, menutup aplikasi...")
        self.is_running.clear()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = VehicleDetectionApp()
    app.run()
    print("Aplikasi ditutup.")
