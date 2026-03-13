import numpy as np


class CentroidTracker:
    """Simple centroid tracker that assigns IDs to bounding boxes.

    This keeps track of object centroids between consecutive frames. When a
    bounding box is close enough to an existing tracked centroid, the tracker
    assumes it is the same object and preserves the object ID.

    If an object disappears for too many frames, it is deregistered.
    """

    def __init__(self, max_disappeared=50):
        # Next ID to assign to a newly seen object.
        self.next_object_id = 0

        # Mapping of object ID -> centroid (x, y)
        self.objects = {}

        # Mapping of object ID -> number of consecutive frames it has been missing
        self.disappeared = {}

        # How many frames an object can be missing before we stop tracking it
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        """Register a new object with an ID and initial centroid."""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """Stop tracking an object ID."""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        """Update tracking given a list of bounding boxes.

        Args:
            rects: list of (x1, y1, x2, y2) bounding boxes for the current frame.

        Returns:
            dict mapping object_id -> centroid (x, y).
        """

        # If there are no detections, mark existing objects as disappeared.
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        # Convert bounding boxes to centroids.
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)

        # If we have no currently tracked objects, register all detections now.
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            # Build matrices of distances between existing objects and new centroids.
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.linalg.norm(
                np.array(object_centroids)[:, np.newaxis] - input_centroids,
                axis=2,
            )

            # For each existing object, find the closest new centroid.
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                # Ignore matches that are too far away.
                if D[row, col] > 50:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Determine which existing objects have disappeared.
            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)

            # If there are more existing objects than new detections, some objects disappeared.
            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                # If there are more detections than existing objects, register them.
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects


class Counter:
    """Keeps a simple count of unique tracked object IDs.

    This is a very basic 'counting' mechanism: every time a new object ID is
    created by the tracker, we increment the total. Objects can exit the frame
    and be removed, but the total count will still reflect the number of unique
    objects seen so far.
    """

    def __init__(self):
        self.tracker = CentroidTracker()
        self.counts = {}

    def update(self, detections):
        """Update counts from a list of detection dicts."""

        rects = [d['xyxy'] for d in detections]
        objects = self.tracker.update(rects)

        for oid in objects.keys():
            if oid not in self.counts:
                # Count each object once, when it is first seen.
                self.counts[oid] = 1

        return objects

    def totals(self):
        """Return the number of unique objects seen so far."""
        return len(self.counts)
