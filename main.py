# main.py

import cv2
import sys
import threading
import torch
from queue import Queue, Empty, Full
from ultralytics import YOLO
from vidgear.gears import CamGear

import config
from vehicle_counter import VehicleCounter
from ui_drawer import UIDrawer


class VehicleDetectionApp:
    def __init__(self):
        self.config = config

        # Inisialisasi model
        print(
            f"Menginisialisasi model {self.config.MODEL_PATH} pada device: {self.config.DEVICE}...")
        self.model = YOLO(self.config.MODEL_PATH).to(self.config.DEVICE)
        print("Model berhasil dimuat.")

        self.ui_drawer = UIDrawer()

        # PERUBAHAN: Menaikkan ukuran antrean untuk buffer yang lebih baik,
        # ini akan menciptakan delay tapi video lebih mulus.
        self.raw_frame_queue = Queue(maxsize=50)
        self.processed_frame_queue = Queue(maxsize=50)

        self.is_running = threading.Event()
        self.is_running.set()

    def _stream_reader(self):
        """Thread untuk membaca frame dari stream dan memasukkannya ke antrean mentah."""
        print(f"Memulai stream dari: {self.config.YOUTUBE_URL}...")
        stream_options = {
            "STREAM_RESOLUTION": f"{self.config.STREAM_RESOLUTION[0]}x{self.config.STREAM_RESOLUTION[1]}"}

        try:
            stream = CamGear(source=self.config.YOUTUBE_URL, stream_mode=True,
                             logging=True, backend=cv2.CAP_GSTREAMER, **stream_options).start()
        except Exception:
            print("Peringatan: GStreamer backend gagal, mencoba FFMPEG...")
            stream = CamGear(source=self.config.YOUTUBE_URL, stream_mode=True,
                             logging=True, backend=cv2.CAP_FFMPEG, **stream_options).start()

        while self.is_running.is_set():
            frame = stream.read()
            if frame is None:
                self.is_running.clear()
                break

            # PERUBAHAN UTAMA: Logika Anti-lag Dihapus.
            # Sekarang thread ini akan menunggu jika antrean penuh,
            # memaksa pemrosesan berjalan secara berurutan.
            # Ini adalah kunci untuk menghilangkan "speed run".
            self.raw_frame_queue.put(frame)

        stream.stop()
        print("Thread pembaca stream berhenti.")

    def _frame_processor(self, zone_polygon):
        """Thread untuk mengambil frame mentah, memprosesnya dengan YOLO, dan memasukkan ke antrean jadi."""
        counter = VehicleCounter(zone_polygon, self.config)
        print("Thread prosesor siap.")

        while self.is_running.is_set():
            try:
                # Mengambil frame dari antrean buffer
                frame = self.raw_frame_queue.get(timeout=1)

                # Jalankan deteksi
                with torch.no_grad():  # Optimasi memori
                    results = self.model.track(
                        frame, persist=True, classes=self.config.CLASS_ID_VEHICLES,
                        verbose=False, device=self.config.DEVICE)

                # Anotasi frame dan masukkan ke antrean output
                processed_frame = counter.process_frame(frame, results)
                self.processed_frame_queue.put(processed_frame)

            except Empty:
                continue

        print("Thread prosesor berhenti.")

    def run(self):
        """Mempersiapkan UI dan menjalankan semua thread."""
        print("Mengambil frame pertama untuk UI...")
        stream = CamGear(source=self.config.YOUTUBE_URL,
                         stream_mode=True).start()
        first_frame = stream.read()
        stream.stop()

        if first_frame is None:
            print("Error: Gagal mendapatkan frame pertama.")
            return

        zone_polygon = self.ui_drawer.draw_polygon_ui(first_frame)
        if zone_polygon is None:
            zone_polygon = self.config.ZONE_POLYGON

        # Jalankan thread
        reader_thread = threading.Thread(
            target=self._stream_reader, daemon=True)
        processor_thread = threading.Thread(
            target=self._frame_processor, args=(zone_polygon,), daemon=True)

        reader_thread.start()
        processor_thread.start()

        print("\nMemulai aplikasi... Tekan 'q' untuk keluar.")

        # Loop utama untuk menampilkan video
        while self.is_running.is_set():
            try:
                frame_to_show = self.processed_frame_queue.get(timeout=2)
                cv2.imshow("Deteksi Kendaraan", frame_to_show)
            except Empty:
                if not processor_thread.is_alive():
                    print("Stream berakhir.")
                    break
                continue

            # Mengatur kecepatan playback agar normal (~25 FPS)
            if cv2.waitKey(40) & 0xFF == ord('q'):
                self.stop()

        reader_thread.join(timeout=2)
        processor_thread.join(timeout=2)

    def stop(self):
        """Menghentikan semua proses dengan aman."""
        print("Menutup aplikasi...")
        self.is_running.clear()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = VehicleDetectionApp()
    app.run()
    print("Aplikasi ditutup.")
