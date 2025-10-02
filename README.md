# Multi-Camera Person Tracking

This project implements **multi-camera person tracking**: assigning a consistent **global ID** to each person across multiple cameras using YOLO detection, DeepSORT tracking, and embedding-based re-identification.

## 🚀 Features

* Person detection using **YOLO** (via `ultralytics` library)
* Local per-camera tracking using **DeepSORT**
* Two global ID assignment strategies:

  * **Method 1** (`method1/`) → Gallery-based global ID assignment
  * **Method 2** (`method2/`) → Refined cross-camera matching with thresholds
* Supports multiple input videos
* Annotated output videos with bounding boxes + global IDs
* CSV logging of tracked IDs

---

## 📂 Repository Structure

```
├── input_videos/        # Example input videos (Single1.mp4, Double1.mp4)
├── method1/             # Gallery-based approach
│   ├── detect.py        # Runs detection + tracking
│   ├── out_global/      #output videos
│   ├── out_reid/        # Re-ID embeddings/logs
│   ├── re-render.py     # Replays annotated outputs
│   └── requirements.txt
├── method2/             # Refined approach
│   ├── output/          #output videos
│   ├── track_id.py      # Cross-camera global ID assignment
│   └── requirements.txt

└── README.md            # Project documentation
```

---

## ⚙️ Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/ArulSreekanth/Multi-Camera-Person-Tracking.git
   cd Multi-Camera-Person-Tracking
   ```

2. Install dependencies:

    Move to specific folder
   ```bash
   cd method1
   (or)
   cd method2
   ```
  
   ```bash
   pip install -r requirements.txt
   ```

---
## 🔍 Approaches
- Method 1 = Offline → processes full videos, uses the gallery to match across cameras after detection & tracking are done.

- Method 2 = Online → assigns global IDs in real time while processing streams, using a buffer and thresholds.
### Method 1: Offline Gallery-Based Global ID Assignment

- Runs detection and local tracking on complete videos.

- Builds a gallery of embeddings for each local track.

- After processing, embeddings are compared across cameras to assign consistent global IDs.

- ✅ Suitable for offline analysis (all data available).

- ⚠️ Cannot be used for real-time streaming.

### Method 2: Online Buffer-Based Global ID Assignment

- Runs detection and tracking while the video/stream is being processed.

- Maintains a sliding buffer of recent embeddings across cameras.

- Assigns or refines global IDs in real time using similarity + temporal consistency.

- ✅ Suitable for real-time multi-camera tracking.

- ⚠️ May need threshold tuning for stability.

## ▶️ Usage

### Method 1: Gallery-Based Global ID

Runs YOLO + DeepSORT tracking per camera and assigns global IDs using an embedding gallery.

```bash
cd method1
python detect.py --videoA ../input_videos/Single.mp4 --videoB ../input_videos/Double.mp4
```

Re-render results with IDs:

```bash
python re-render.py --videoA ../input_videos/Single.mp4 --videoB ../input_videos/Double.mp4 --map_csv ../out_reid/global_id_map.csv --tracksA  ../out_reid/trackletsA.pkl --tracksB ../out_reid/trackletsB.pkl 
```

Outputs:

* Annotated video in `out_global/`
* Global ID CSV logs

---

### Method 2: Refined Global ID Matching

Uses a refined approach with stricter thresholds and temporal consistency.

```bash
cd method2
python track_id.py --cam1 ../input_videos/Single.mp4 --cam2 ../input_videos/Double.mp4
```

Outputs:

* Refined annotated videos in `output/`
* CSV logs with global ID assignments

---

## ⚖️ Parameters

Some important parameters (edit inside the scripts):

* `PERSON_CLASS_ID` → YOLO class ID for “person” (default = 0)
* `MATCH_THRESH` → similarity threshold for cross-camera matching
* `MIN_FRAMES_FOR_GID` → minimum frames before assigning a stable global ID
* `BUFFER_SIZE` (method2) → number of frames kept in buffer for matching

---

## 📊 Outputs

* **Per-camera tracks** (local IDs)
* **Global ID mapping** across cameras
* **Annotated videos** stored in `output`
* **CSV logs** for analysis

---

## 📌 Notes

* Ensure your input videos are placed in `input_videos/`
* GPU is recommended for real-time performance
* Threshold tuning is crucial for cross-camera consistency
* Method 1 is simpler but more error-prone, Method 2 is stricter and robust

---

## 📜 License

MIT
