import math


class Tracker:
    def __init__(self):
        self.center_points = {}   # id -> (cx, cy)
        self.disappeared = {}     # id -> số frame bị mất
        self.id_count = 0

        self.MAX_DISTANCE = 180     # khoảng cách match (tăng để giảm nhảy)
        self.MAX_DISAPPEAR = 25     # giữ ID lâu hơn

    def update(self, objects_rect):
        objects_bbs_ids = []

        # Tính center của detections
        detections = []
        for rect in objects_rect:
            x1, y1, x2, y2 = rect
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            detections.append((x1, y1, x2, y2, cx, cy))

        new_center_points = {}
        used_ids = set()

        # ===== MATCH TỐT NHẤT =====
        for (x1, y1, x2, y2, cx, cy) in detections:

            best_id = None
            best_dist = float("inf")

            for track_id, (px, py) in self.center_points.items():

                if track_id in used_ids:
                    continue

                dist = math.hypot(cx - px, cy - py)

                if dist < self.MAX_DISTANCE and dist < best_dist:
                    best_dist = dist
                    best_id = track_id

            # ===== MATCH THÀNH CÔNG =====
            if best_id is not None:
                new_center_points[best_id] = (cx, cy)
                self.disappeared[best_id] = 0
                used_ids.add(best_id)

                objects_bbs_ids.append((x1, y1, x2, y2, best_id))

            # ===== TẠO ID MỚI =====
            else:
                new_center_points[self.id_count] = (cx, cy)
                self.disappeared[self.id_count] = 0

                objects_bbs_ids.append((x1, y1, x2, y2, self.id_count))
                self.id_count += 1

        # ===== XỬ LÝ OBJECT BỊ MẤT =====
        for track_id in list(self.center_points.keys()):
            if track_id not in new_center_points:

                self.disappeared[track_id] += 1

                # giữ lại nếu chưa mất quá lâu
                if self.disappeared[track_id] < self.MAX_DISAPPEAR:
                    new_center_points[track_id] = self.center_points[track_id]
                else:
                    # xoá hẳn
                    del self.disappeared[track_id]

        # cập nhật lại
        self.center_points = new_center_points.copy()

        return objects_bbs_ids