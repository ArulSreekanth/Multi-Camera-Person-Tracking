"""
cross_video_reid.py
Simple pipeline:
- Detect with YOLOv8 (ultralytics package)
- Track per-video with deep_sort_realtime
- Extract embeddings with a ResNet50 (ImageNet) as a fallback (replace with a ReID model for better results)
- Match tracklets across videos using cosine distance + Hungarian
- Writes a CSV of matches and optional annotated videos
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torchvision.transforms as T
import torchvision.models as models
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.optimize import linear_sum_assignment
import argparse
import csv
from collections import defaultdict
from tqdm import tqdm
import pickle

# -------------------
# Utils
# -------------------
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def crop_box(img, box, pad=0):
    x1, y1, x2, y2 = [int(x) for x in box]
    h, w = img.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)
    return img[y1:y2, x1:x2]

# -------------------
# Appearance model (fallback: ResNet50 pretrained on ImageNet)
# Replace with a proper ReID model (OSNet, etc.) for best accuracy
# -------------------
import torchreid

class AppearanceEncoder:
    def __init__(self, device='mps' if torch.backends.mps.is_available() else 'cpu'):
        self.device = device
        # Build pretrained OSNet model
        self.model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=True
        )
        self.model.eval().to(self.device)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),  # ReID aspect ratio
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def encode(self, crop):
        if crop.size == 0:
            return None
        x = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(x).cpu().numpy()[0]
        # L2 normalize
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        return feat


# -------------------
# Per-video processing: detect + track
# returns: tracklets dict: {track_id: {'frames':[frame_idx,...], 'boxes':[box,...], 'crops':[img arrays...]}}
# -------------------
def process_video(video_path, detector, tracker, encoder, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)
    tracklets = {}   # <-- plain dict now
    frame_idx = 0

    pbar = tqdm(total=total, desc=f"Processing {os.path.basename(video_path)}")
    while frame_idx < total:
        ret, frame = cap.read()
        if not ret:
            break

        # detect people with YOLO (class 0 usually = person)
        results = detector(frame, imgsz=640, device='mps',classes=[0])  # device='cpu' or 'cuda:0' if available
        # results is list of Results; take first
        boxes = []
        scores = []
        for r in results:
            if len(r.boxes) == 0:
                continue
            for box in r.boxes:
                xyxy = box.xyxy.cpu().numpy()[0]  # x1,y1,x2,y2
                conf = float(box.conf.cpu().numpy()[0])
                boxes.append(xyxy.tolist())
                scores.append(conf)

        # prepare detections for deep_sort_realtime: list of dicts with tlbr and confidence
        dets = []
        for bb,sc in zip(boxes, scores):
            x1,y1,x2,y2 = bb
            dets.append(([int(x1),int(y1),int(x2-x1),int(y2-y1)], sc, 'person'))  # xywh,score,label

        # update tracker
        tracks = tracker.update_tracks(dets, frame=frame)  # deep_sort_realtime API
        # tracks: list of Track objects if confirmed
        if tracks:
            for t in tracks:
                if not t.is_confirmed():
                    continue
                tid = t.track_id
                tlbr = t.to_tlbr()  # top,left,bottom,right
                x1,y1,x2,y2 = tlbr
                crop = crop_box(frame, (x1,y1,x2,y2), pad=5)

                if tid not in tracklets:
                    tracklets[tid] = {'frames':[], 'boxes':[], 'crops':[]}
                tracklets[tid]['frames'].append(frame_idx)
                tracklets[tid]['boxes'].append((x1,y1,x2,y2))
                tracklets[tid]['crops'].append(crop)

                # draw on frame
                cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
                text_x = int(x1)
                text_y = int(y2) + 20
                cv2.putText(frame, f"ID:{tid}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()

    # compute per-track appearance embedding (average of N crops)
    track_embeddings = {}
    for tid, data in tracklets.items():
        crops = data['crops']
        if len(crops)==0:
            continue
        # sample up to 8 crops evenly across the track
        indices = np.linspace(0, len(crops)-1, min(8, len(crops))).astype(int)
        feats = []
        for i in indices:
            c = crops[i]
            f = encoder.encode(c)
            if f is not None:
                feats.append(f)
        if len(feats)==0:
            continue
        avg = np.mean(np.stack(feats,axis=0), axis=0)
        avg = avg / (np.linalg.norm(avg)+1e-8)
        track_embeddings[tid] = avg
    return tracklets, track_embeddings

# -------------------
# Cross-video matching
# -------------------
def match_tracks(embA, embB, cost_thresh=0.4):
    # embA and embB are dicts tid->feat (L2-normalized)
    idsA = list(embA.keys())
    idsB = list(embB.keys())
    if len(idsA)==0 or len(idsB)==0:
        return []

    featsA = np.stack([embA[i] for i in idsA], axis=0)
    featsB = np.stack([embB[j] for j in idsB], axis=0)

    # cosine distance = 1 - cosine_similarity
    sim = featsA @ featsB.T
    cost = 1.0 - sim  # lower cost better
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    for r,c in zip(row_ind, col_ind):
        if cost[r,c] <= cost_thresh:  # threshold to accept match
            matches.append((idsA[r], idsB[c], float(cost[r,c])))
    return matches

# -------------------
# Main
# -------------------
def main(args):
    ensure_dir(args.out)

    # detector
    detector = YOLO('yolov8n.pt')  # small; swap for yolov8m/large for accuracy (requires GPU)

    # tracker: deep_sort_realtime default initialization
    trackerA = DeepSort(max_age=50,n_init = 5)
    trackerB = DeepSort(max_age=50,n_init = 5)

    encoder = AppearanceEncoder()

    # Process each video
    trackletsA, embA = process_video(args.videoA, detector, trackerA, encoder,
                                     max_frames=None,
                                     save_annotated=os.path.join(args.out))
    trackletsB, embB = process_video(args.videoB, detector, trackerB, encoder,
                                     max_frames=None,
                                     save_annotated=os.path.join(args.out))

    # Save raw tracklets so we can re-render later
    with open(os.path.join(args.out, "trackletsA.pkl"), "wb") as f:
        pickle.dump(trackletsA, f)

    with open(os.path.join(args.out, "trackletsB.pkl"), "wb") as f:
        pickle.dump(trackletsB, f)

    print("Saved tracklets for re-rendering")

    # Match
    matches = match_tracks(embA, embB, cost_thresh=args.cost_thresh)
    print(f"Found {len(matches)} matches")

    # Write CSV
    csv_path = os.path.join(args.out, 'cross_video_matches.csv')
    with open(csv_path, 'w', newline='') as cf:
        w = csv.writer(cf)
        w.writerow(['trackA','trackB','cost'])
        for a,b,c in matches:
            w.writerow([a,b,c])

    # Build global ID mapping: give each matched pair a global id, unmatched tracks get new global ids
    global_map = {}
    next_gid = 0
    matchedA = set()
    matchedB = set()
    for a,b,c in matches:
        gid = f"G{next_gid}"
        global_map[('A',a)] = gid
        global_map[('B',b)] = gid
        matchedA.add(a)
        matchedB.add(b)
        next_gid += 1
    for a in embA.keys():
        if a not in matchedA:
            global_map[('A',a)] = f"G{next_gid}"; next_gid+=1
    for b in embB.keys():
        if b not in matchedB:
            global_map[('B',b)] = f"G{next_gid}"; next_gid+=1

    # Save mapping
    map_path = os.path.join(args.out, 'global_id_map.csv')
    with open(map_path, 'w', newline='') as mf:
        w = csv.writer(mf)
        w.writerow(['video','track_id','global_id'])
        for (v,t),gid in sorted(global_map.items()):
            w.writerow([v,t,gid])

    print("Saved results to", args.out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--videoA', required=True)
    parser.add_argument('--videoB', required=True)
    parser.add_argument('--out', default='out_reid')
    parser.add_argument('--cost_thresh', type=float, default=0.45)
    args = parser.parse_args()
    main(args)
