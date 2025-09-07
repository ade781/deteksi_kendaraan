import cv2


class VehicleCounter:
    """
    Sebuah kelas untuk mengelola logika deteksi dan penghitungan kendaraan.
    """

    def __init__(self, zone_polygon, config):
        """
        Inisialisasi penghitung.

        Args:
            zone_polygon (np.array): Array numpy berisi koordinat poligon zona hitung.
            config (module): Modul konfigurasi yang berisi pengaturan tampilan.
        """
        self.zone = zone_polygon
        self.config = config
        self.counted_ids = set()

        # --- PERUBAHAN: Inisialisasi penghitung terpisah ---
        self.total_count = 0
        self.car_count = 0
        self.motorcycle_count = 0
        # Di sini bisa ditambahkan jenis kendaraan lain jika perlu (bus, truk)
        # ----------------------------------------------------

    def process_frame(self, frame, results):
        """
        Memproses satu frame video untuk mendeteksi dan menghitung kendaraan.

        Args:
            frame (np.array): Frame video asli.
            results (list): Hasil deteksi dari model YOLO.

        Returns:
            np.array: Frame video yang sudah dianotasi dengan deteksi dan hitungan.
        """
        # --- PERUBAHAN: Kita akan menggambar manual agar bisa filter object ---
        # Kita mulai dengan frame asli, bukan yang sudah di-plot
        annotated_frame = frame.copy()
        # ------------------------------------------------------------------

        # Gambar zona hitung pada frame
        cv2.polylines(annotated_frame, [self.zone], isClosed=True,
                      color=self.config.ZONE_COLOR, thickness=self.config.ZONE_THICKNESS)

        # Cek jika ada objek yang berhasil dilacak
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            class_names = results[0].names  # Ambil nama kelas dari hasil

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                # --- PERUBAHAN: Fokus HANYA pada kendaraan ---
                if class_id in self.config.CLASS_ID_VEHICLES:
                    # Gambar bounding box untuk kendaraan
                    x1, y1, x2, y2 = box
                    cv2.rectangle(annotated_frame, (x1, y1),
                                  (x2, y2), (0, 255, 0), 2)

                    # --- Logika penghitungan tetap sama ---
                    point_to_check = (int((x1 + x2) / 2), int(y2))
                    is_inside = cv2.pointPolygonTest(
                        self.zone, point_to_check, False)
                    # -------------------------------------

                    if is_inside >= 0:
                        if track_id not in self.counted_ids:
                            self.counted_ids.add(track_id)
                            self.total_count += 1

                            # --- PERUBAHAN: Tambah hitungan berdasarkan kelas ---
                            if class_id == 2:  # 2 adalah ID untuk 'car'
                                self.car_count += 1
                            elif class_id == 3:  # 3 adalah ID untuk 'motorcycle'
                                self.motorcycle_count += 1
                            # --------------------------------------------------

                            cv2.circle(annotated_frame, point_to_check,
                                       self.config.CENTER_CIRCLE_RADIUS, self.config.CENTER_CIRCLE_COLOR, -1)

        # --- PERUBAHAN: Tampilkan semua hitungan ---
        cv2.putText(annotated_frame, f"Total Kendaraan: {self.total_count}",
                    (self.config.TEXT_POSITION[0],
                     self.config.TEXT_POSITION[1]),
                    self.config.TEXT_FONT, 1.2, self.config.TEXT_COLOR, self.config.TEXT_THICKNESS)

        cv2.putText(annotated_frame, f"Mobil: {self.car_count}",
                    (self.config.TEXT_POSITION[0],
                     self.config.TEXT_POSITION[1] + 40),
                    self.config.TEXT_FONT, 1.2, self.config.TEXT_COLOR, self.config.TEXT_THICKNESS)

        cv2.putText(annotated_frame, f"Motor: {self.motorcycle_count}",
                    (self.config.TEXT_POSITION[0],
                     self.config.TEXT_POSITION[1] + 80),
                    self.config.TEXT_FONT, 1.2, self.config.TEXT_COLOR, self.config.TEXT_THICKNESS)
        # -------------------------------------------

        return annotated_frame
