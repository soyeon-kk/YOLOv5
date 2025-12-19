import math

class PersonTracker:
    """
    간단 인물 추적기
    """

    def __init__(self, max_dist=800.0, max_lost=60):
        self.max_dist = max_dist
        self.max_lost = max_lost
        self.next_id = 0
        self.tracks = {}

    def update(self, detections):
        used_tracks = set()
        current_ids = set()

        for x1, y1, x2, y2 in detections:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            best_id = None
            best_dist = self.max_dist

            for tid, info in self.tracks.items():
                if tid in used_tracks:
                    continue
                tx, ty = info["center"]
                dist = math.hypot(cx - tx, cy - ty)
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is None:
                tid = self.next_id
                self.next_id += 1
            else:
                tid = best_id

            self.tracks[tid] = {
                "center": (cx, cy),
                "lost": 0,
                "bbox": (x1, y1, x2, y2),
            }
            used_tracks.add(tid)
            current_ids.add(tid)

        for tid in list(self.tracks.keys()):
            if tid not in used_tracks:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]

        track_boxes = []
        for tid, info in self.tracks.items():
            if tid in current_ids:
                x1, y1, x2, y2 = info["bbox"]
                track_boxes.append((x1, y1, x2, y2, tid))

        return current_ids, track_boxes
