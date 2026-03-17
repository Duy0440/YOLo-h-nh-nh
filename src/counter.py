import numpy as np

class CentroidTracker:

    def __init__(self, max_disappeared=40):

        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid):

        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):

        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):

        if len(rects) == 0:

            for oid in list(self.disappeared.keys()):

                self.disappeared[oid] += 1

                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

            return self.objects

        input_centroids = np.array([
            ((x1+x2)//2, (y1+y2)//2) for (x1,y1,x2,y2) in rects
        ])

        if len(self.objects) == 0:

            for c in input_centroids:
                self.register(c)

        else:

            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))

            D = np.linalg.norm(
                object_centroids[:, None] - input_centroids,
                axis=2
            )

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for r, c in zip(rows, cols):

                if r in used_rows or c in used_cols:
                    continue

                if D[r, c] > 50:
                    continue

                oid = object_ids[r]

                self.objects[oid] = input_centroids[c]
                self.disappeared[oid] = 0

                used_rows.add(r)
                used_cols.add(c)

            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            for r in unused_rows:

                oid = object_ids[r]
                self.disappeared[oid] += 1

                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

            for c in unused_cols:
                self.register(input_centroids[c])

        return self.objects


class Counter:

    def __init__(self):

        self.tracker = CentroidTracker()
        self.counts = set()

    def update(self, detections):

        rects = [d["xyxy"] for d in detections]

        objects = self.tracker.update(rects)

        for oid in objects:
            self.counts.add(oid)

        return objects

    def total(self):

        return len(self.counts)