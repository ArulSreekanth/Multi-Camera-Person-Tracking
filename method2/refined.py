# detectq.py
"""
Multi-camera ReID proof-of-concept:
- YOLOv8 for person detection
- DeepSORT (or fallback) for per-camera tracking
- TorchReID (OSNet) for embeddings
- Matches tracklets across cameras using cosine similarity
- Overlays LocalID->GlobalID(conf) on video
- Saves annotated videos + global_id_map.csv
"""

import os
import cv2
import time
import argparse
import numpy as np
import csv
from collections import defaultdict

# ---------------- YOLO ----------------
try:
    from ultralytics import YOLO
except Exception as e:
    raise ImportError("Install ultralytics: pip install ultralytics") from e

# ---------------- TorchReID ----------------
try:
    from torchreid.utils import FeatureExtractor
    import torch
except Exception as e:
    raise ImportError("Install torchreid and torch: pip install torch torchreid") from e

# ---------------- DeepSORT ----------------
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort  # pip install deep-sort-realtime
    HAVE_DEEPSORT = True
except Exception:
    HAVE_DEEPSORT = False
    # Minimal fallback tracker
    class SimpleCentroidTracker:
        def __init__(self, max_lost=30):
            self.next_id = 1
            self.tracks = {}
        def update(self, detections):
            outs = []
            for d in detections:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = (d, 0)
                outs.append((d, tid))
            return outs

# ---------------- Helpers ----------------
def l2norm(x, eps=1e-12):
    return x / (np.linalg.norm(x) + eps)

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

track_memory = defaultdict(list)

# ---------------- Cross-camera gallery ----------------
class CrossCameraGallery:
    def __init__(self, max_items=500):
        self.items = {}  # global_id -> dict(cam, track_id, embedding, last_seen)
        self.last_global_id = 0
        self.max_items = max_items

    def new_global_id(self):
        self.last_global_id += 1
        return self.last_global_id

    def add_or_update(self, global_id, cam, cam_track_id, embedding, ts):
        self.items[global_id] = dict(cam=cam, cam_track_id=cam_track_id,
                                     embedding=embedding, last_seen=ts)

    def match(self, embedding, exclude_cam=None, topk=5):
        scores = []
        for gid, info in self.items.items():
            if exclude_cam is not None and info['cam'] == exclude_cam:
                continue
            scores.append((gid, cosine(embedding, info['embedding'])))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topk]

    def prune_old(self, age_threshold=300.0, now_ts=None):
        now_ts = now_ts or time.time()
        to_delete = [gid for gid, info in self.items.items()
                     if now_ts - info['last_seen'] > age_threshold]
        for gid in to_delete:
            del self.items[gid]

# ---------------- Main pipeline ----------------
def process_pair(cam1_path, cam2_path, outdir, match_thresh=0.62, reid_model='osnet_x0_25'):
    os.makedirs(outdir, exist_ok=True)

    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"âœ… Using device: {device}")

    # TorchReID extractor
    extractor = FeatureExtractor(
        model_name=reid_model,
        model_path='',
        device=device
    )

    # YOLO detector
    ymodel = YOLO('yolo11n.pt')
    ymodel.to(device)
    PERSON_CLASS_ID = 0

    # Trackers
    if HAVE_DEEPSORT:
        print("âœ… Using DeepSORT tracker")
        tracker1 = DeepSort(max_age=100, n_init=8)
        tracker2 = DeepSort(max_age=100, n_init=8)  
    else:
        tracker1 = SimpleCentroidTracker()
        tracker2 = SimpleCentroidTracker()

    cap1 = cv2.VideoCapture(cam1_path)
    cap2 = cv2.VideoCapture(cam2_path)

    # Video writers
    w1, h1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2, h2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS) or 20
    out1 = cv2.VideoWriter(os.path.join(outdir, 'cam1_out.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w1, h1))
    out2 = cv2.VideoWriter(os.path.join(outdir, 'cam2_out.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w2, h2))

    gallery = CrossCameraGallery()
    frame_idx = 0

    # CSV logging
    csv_path = os.path.join(outdir, "global_id_map.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["global_id", "camera", "local_track_id", "confidence", "start_frame", "end_frame"])
    track_frames = {}
    local_to_global = {}   # <-- add this here


    def process_frame(frame, cam_name, tracker, writer):
        nonlocal frame_idx
        if frame is None:
            return

        detections = []
        results = ymodel(frame, imgsz=640, conf=0.25, classes=[PERSON_CLASS_ID], verbose=False)

        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                conf = float(b.conf[0])
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, PERSON_CLASS_ID))

        out_tracks = []
        if HAVE_DEEPSORT:
            tracks = tracker.update_tracks(detections, frame=frame)
            for t in tracks:
                if not t.is_confirmed():
                    continue
                tid = t.track_id
                tlbr = t.to_tlbr()
                out_tracks.append((tid, [int(tlbr[0]), int(tlbr[1]), int(tlbr[2]), int(tlbr[3])], 1.0))
        else:
            outs = tracker.update(detections)
            for d, tid in outs:
                out_tracks.append((tid, [int(d[0]), int(d[1]), int(d[2]), int(d[3])], d[4]))

        ts = time.time()
        for tid, bbox, conf in out_tracks:
            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_resized = cv2.resize(crop, (128, 256))
            feats = extractor(crop_resized)
            if isinstance(feats, torch.Tensor):
                feats = feats.detach().cpu().numpy()
            feat = feats[0]
            track_memory[tid].append(feat)

            avg_feat = l2norm(np.mean(track_memory[tid], axis=0))

            # Match with gallery
            # Assign stable global IDs
            if (cam_name, tid) in local_to_global:
                gid = local_to_global[(cam_name, tid)]
                score = cosine(avg_feat, gallery.items[gid]['embedding'])
            else:
                if len(track_memory[tid]) < 5:   # MIN_FRAMES_FOR_GID
                    # show pending instead of assigning a new GID
                    cv2.putText(frame, f"T{tid} (pending)", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,200), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,200,200), 2)
                    continue
                match_results = gallery.match(avg_feat, exclude_cam=cam_name)
                if match_results and match_results[0][1] > 0.55:   # lowered threshold
                    gid, score = match_results[0]
                else:
                    gid = gallery.new_global_id()
                    score = 1.0
                local_to_global[(cam_name, tid)] = gid


            gallery.add_or_update(gid, cam_name, tid, avg_feat, ts)

            # Update frame ranges
            if (cam_name, tid) not in track_frames:
                track_frames[(cam_name, tid)] = [frame_idx, frame_idx]
            else:
                track_frames[(cam_name, tid)][1] = frame_idx

            # Draw overlay

            np.random.seed(gid)  # same gid = same color always
            color = tuple(int(c) for c in np.random.randint(0, 255, size=3))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text_x = int((x1 + x2) / 2)
            text_y = int((y1 + y2) / 2)
            cv2.putText(frame, f"T{tid} -> G{gid} ({score:.2f})",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            lineType=cv2.LINE_AA)


            # Log to CSV
            csv_writer.writerow([gid, cam_name, tid, f"{score:.2f}",
                                 track_frames[(cam_name, tid)][0],
                                 track_frames[(cam_name, tid)][1]])

        writer.write(frame)

    while True:
        ok1, f1 = cap1.read()
        ok2 = False
        f2 = None
        if cap2.isOpened():
            ok2, f2 = cap2.read()
        if not ok1 and not ok2:
            break
        frame_idx += 1
        if ok1:
            process_frame(f1, "A", tracker1, out1)
        if ok2:
            process_frame(f2, "B", tracker2, out2)

    cap1.release(); cap2.release()
    out1.release(); out2.release()
    csv_file.close()
    print(f"Outputs saved to {outdir}")
    print(f"   - cam1_out.mp4, cam2_out.mp4")
    print(f"   - global_id_map.csv")

# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam1", required=True, help="Path to camera 1 video")
    parser.add_argument("--cam2", required=True, help="Path to camera 2 video")
    parser.add_argument("--out_dir", default="output", help="Output directory")
    args = parser.parse_args()

    process_pair(args.cam1, args.cam2, args.out_dir)