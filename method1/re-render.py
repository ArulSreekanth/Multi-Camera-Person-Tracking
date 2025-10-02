import cv2
import csv
import os
from tqdm import tqdm

def load_global_map(csv_path):
    """Load global ID mapping from CSV"""
    global_map = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video = row['video']
            track_id = int(row['track_id'])
            global_id = row['global_id']
            global_map[(video, track_id)] = global_id
    return global_map

def re_render(video_path, out_path, video_tag, global_map, tracklets):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w,h))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total, desc=f"Rendering {out_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        # loop over all tracklets
        for tid, data in tracklets.items():
            # draw only if current frame index exists in this track
            if frame_idx in data['frames']:
                i = data['frames'].index(frame_idx)
                x1,y1,x2,y2 = data['boxes'][i]
                gid = global_map.get((video_tag, tid), f"UNK{tid}")
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.putText(frame, f"GID:{gid}", (int(x1), int(y2)+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        writer.write(frame)
        frame_idx += 1
        pbar.update(1)

    cap.release()
    writer.release()
    pbar.close()
    print(f"✅ Saved {out_path}")

if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--videoA", required=True, help="Original Video A")
    parser.add_argument("--videoB", required=True, help="Original Video B")
    parser.add_argument("--map_csv", required=True, help="global_id_map.csv file")
    parser.add_argument("--tracksA", required=True, help="Pickled track data for Video A")
    parser.add_argument("--tracksB", required=True, help="Pickled track data for Video B")
    parser.add_argument("--out_dir", default="out_global", help="Output folder")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load mapping
    global_map = load_global_map(args.map_csv)

    # Load saved track data (from process_video)
    with open(args.tracksA, "rb") as f:
        track_data_A = pickle.load(f)
    with open(args.tracksB, "rb") as f:
        track_data_B = pickle.load(f)

    # Re-render with global IDs
    re_render(args.videoA, os.path.join(args.out_dir, "revideo1.mp4"), "A", global_map, track_data_A)
    re_render(args.videoB, os.path.join(args.out_dir, "revideo2.mp4"), "B", global_map, track_data_B)
