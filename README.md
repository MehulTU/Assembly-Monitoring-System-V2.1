# Assembly Monitor - Software Upgrade Package (V2.1)

## Overview

This project presents an AI-supported modular assembly monitoring system developed as part of my Master's thesis at TU Darmstadt.

The software monitors manual assembly operations using computer vision and YOLO object detection. The system has been designed with a modular architecture so that new detection methods can be integrated without changing the overall software structure.

## What is in this package

```
Assembly_monitor/            <- REPLACE these 4 files in your project
    main.py                     settings-driven YOLO, throttled terminal output
    config/settings.py          new YOLO section (SS4/SS10)
    processing/object_detector.py   per-class confidence statistics + rolling avg
    visualization/display.py        confidence-coded boxes + DETECTIONS panel

scripts/                     <- ADD to your scripts folder
    train_yolo.py               NEW: training + evaluation + auto-deployment
    record_dataset_v2_2.py      fast trial recorder (session workflow)
```

## Project Structure

```text
Assembly-Monitoring-System-V2.1
│
├── Assembly_monitor/
├── datasets/
├── scripts/
├── weights/
├── recordings/
├── experiments/
├── README.md
```

## Technologies

- Python
- OpenCV
- YOLO
- Ultralytics
- NumPy
- PyTorch

## The full workflow (record -> train -> run)

1. RECORD trials:
       python scripts/record_dataset_v2_2.py

2. Extract frames, review, finalize, ANNOTATE all images in CVAT,
   download the YOLO export (your existing pipeline).

3. PREPARE the dataset (auto-split by trial, no code edits):
       python scripts/prepare_yolo_dataset.py

4. TRAIN + EVALUATE + DEPLOY in one command:
       python scripts/train_yolo.py

   This trains on datasets/yolo_dataset_v1, evaluates on the test
   split (per-class precision/recall/mAP), backs up the old
   weights/best.pt to weights/backups/, and copies the new best.pt
   to weights/best.pt.

5. RUN the live system:
       python Assembly_monitor/main.py

   main.py loads weights/best.pt via YOLO_MODEL_PATH in
   config/settings.py -> the newly trained objects are detected
   with NO software changes. This is the modular integration point.

## Why your current confidence is low (0.25 - 0.45)

Most likely causes, in order of impact:

1. UNANNOTATED IMAGES. Your last dataset had 105 labels for 304
   images. Every unannotated image that actually contains a marker
   or adapter teaches the model "this is background" - directly
   suppressing confidence. Annotate everything before retraining.

2. SMALL DATASET. 15 trials of one scene is little data. Record
   more trials with varied positions, distances, rotations, and
   lighting (your recorder makes this fast now).

3. NO HARD NEGATIVES. The model called your face "marker" - it has
   never seen faces/hands/bedroom background labeled as empty.
   Include empty_workspace and hand_interaction trials WITH correct
   (empty or partial) labels.

4. TRAIN/INFERENCE MISMATCH. Live inference now uses the same
   imgsz (640) as training (YOLO_IMGSZ in settings).

After retraining with full annotations, expect the DETECTIONS panel
to move from red/yellow into green (>60%) for clearly visible objects.


## Future Improvements

- Additional object classes
- ROS integration
- Multi-camera support
- Real-time performance optimization

## Author

Mehul Patil

Master's Thesis

Brazil - TU Darmstadt