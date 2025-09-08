# vehicle_counter.py

import cv2
import numpy as np


class VehicleCounter:
    """
    Mengelola logika deteksi, pelacakan, dan penghitungan kendaraan
    secara dinamis berdasarkan konfigurasi.
    """

    def __init__(self, zone_polygon, config):
        """
        Inisialisasi penghitung.

        Args:
            zone_polygon (np.array): Koordinat poligon zona hitung.
            config (module): Modul konfigurasi (config.py).
        """
        self.zone = zone_polygon
        self.config = config
        self.counted_track_ids = set()

        # Inisialisasi penghitung secara dinamis dari config
        self.vehicle_counts = {
            class_id: 0 for class_id in self.config.CLASS_ID_VEHICLES}
        self.total_count = 0

    def _get_center_bottom(self, box):
        """Menghitung titik tengah bawah dari bounding box."""
        x1, y1, x2, y2 = box
        return (int((x1 + x2) / 2), int(y2))

    def process_frame(self, frame, results):
        """
        Memproses satu frame video, menganotasi, dan menghitung kendaraan.

        Args:
            frame (np.array): Frame video asli.
            results: Hasil deteksi dari model YOLO.

        Returns:
            np.array: Frame video yang telah dianotasi.
        """
        annotated_frame = frame.copy()

        # Gambar zona hitung
        cv2.polylines(annotated_frame, [self.zone], isClosed=True,
                      color=self.config.ZONE_COLOR, thickness=self.config.ZONE_THICKNESS)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                # Proses hanya jika class_id adalah kendaraan yang didefinisikan di config
                if class_id in self.config.CLASS_ID_VEHICLES:
                    class_info = self.config.CLASS_DATA[class_id]
                    color = class_info["color"]
                    name = class_info["name"]

                    # Gambar bounding box dan label
                    x1, y1, x2, y2 = box
                    cv2.rectangle(annotated_frame, (x1, y1),
                                  (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"{name} #{track_id}", (x1, y1 - 10),
                                self.config.TEXT_FONT, 0.7, color, 2)

                    # Titik untuk pengecekan masuk zona
                    point_to_check = self._get_center_bottom(box)

                    # Cek apakah kendaraan berada di dalam zona
                    if cv2.pointPolygonTest(self.zone, point_to_check, False) >= 0:
                        # Jika kendaraan belum pernah dihitung, hitung sekarang
                        if track_id not in self.counted_track_ids:
                            self.counted_track_ids.add(track_id)
                            self.vehicle_counts[class_id] += 1
                            self.total_count += 1

                            # Tandai kendaraan yang baru dihitung dengan lingkaran
                            cv2.circle(annotated_frame, point_to_check,
                                       self.config.CENTER_CIRCLE_RADIUS,
                                       self.config.CENTER_CIRCLE_COLOR, -1)

        self._draw_counts(annotated_frame)
        return annotated_frame

    def _draw_counts(self, frame):
        """Menampilkan teks hitungan kendaraan di layar."""
        pos_x = self.config.TEXT_START_POSITION[0]
        pos_y = self.config.TEXT_START_POSITION[1]

        # Tampilkan total
        cv2.putText(frame, f"Total Kendaraan: {self.total_count}",
                    (pos_x, pos_y), self.config.TEXT_FONT, self.config.TEXT_SCALE,
                    self.config.TEXT_COLOR, self.config.TEXT_THICKNESS)

        # Tampilkan hitungan per jenis kendaraan secara dinamis
        for i, class_id in enumerate(self.config.CLASS_ID_VEHICLES):
            class_info = self.config.CLASS_DATA[class_id]
            name = class_info["name"]
            count = self.vehicle_counts[class_id]

            # Hitung posisi y untuk baris teks berikutnya
            current_y = pos_y + (self.config.TEXT_LINE_HEIGHT * (i + 1))

            cv2.putText(frame, f"{name}: {count}",
                        (pos_x, current_y), self.config.TEXT_FONT, self.config.TEXT_SCALE,
                        self.config.TEXT_COLOR, self.config.TEXT_THICKNESS)
