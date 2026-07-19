# Experiment 001 – Two-Object Detection Baseline

## 1. Purpose

The purpose of this experiment is to develop and validate the first complete custom YOLO object-detection workflow of Prototype V2.

This experiment is a learning and pipeline-validation experiment.

It is not the final thesis experiment.

The experiment verifies the complete workflow:

Data Acquisition
→ Frame Extraction
→ Image Quality Analysis
→ Near-Duplicate Analysis
→ Dataset Review
→ Image Annotation
→ YOLO Dataset Creation
→ Model Training
→ Model Validation
→ Inference

---

## 2. Research Question

Can a custom YOLO object-detection model be trained using images collected with the Prototype V2 data pipeline to detect two known objects under controlled workspace conditions?

---

## 3. Object Classes

Class 0:

object_a

Class 1:

object_b

The actual physical objects used in the experiment must be documented before data collection.

Example:

object_a = marker

object_b = power_adapter

---

## 4. Camera

Initial camera:

Logitech RGB camera

The same camera configuration should be used throughout the initial experiment.

---

## 5. Workspace

The experiment will use a controlled tabletop workspace.

The camera should remain approximately fixed during each recording trial.

---

## 6. Data Collection Conditions

The dataset should contain variation in:

- Object position
- Object orientation
- Object distance from camera
- One-object scenes
- Two-object scenes
- Empty workspace scenes
- Partial object occlusion
- Hand-object interaction
- Different object arrangements

---

## 7. Trial Plan

The first dataset will contain approximately 15 short recording trials.

Trial duration:

Approximately 10–20 seconds.

Planned trials:

T01 – Empty workspace

T02 – Object A at left position

T03 – Object A at center position

T04 – Object A at right position

T05 – Object A with orientation changes

T06 – Object B at left position

T07 – Object B at center position

T08 – Object B at right position

T09 – Object B with orientation changes

T10 – Both objects separated

T11 – Both objects close together

T12 – Object A partially occluded

T13 – Object B partially occluded

T14 – Hand interacting with both objects

T15 – Continuous object movement and rearrangement

---

## 8. Data Preparation

Recorded videos will be processed using the Prototype V2 dataset-preparation pipeline.

Processing sequence:

record_dataset.py

→

extract_frames.py

→

check_dataset_quality.py

→

detect_near_duplicates.py

→

build_review_manifest.py

→

Human Review

→

Annotation

---

## 9. Annotation

Each visible instance of object_a and object_b will be annotated using bounding boxes.

Class IDs:

0 = object_a

1 = object_b

Objects should be annotated consistently across all images.

Empty images will contain no bounding-box labels.

---

## 10. Dataset Split

The dataset will later be divided into:

Training set

Validation set

Test set

Important:

Images originating from the same recorded trial should not be randomly distributed across training, validation, and test sets.

The split should be performed by trial to reduce data leakage from highly similar video frames.

---

## 11. Initial Model

The first custom detector will use a small pretrained YOLO model.

Initial model:

YOLO11n

The model will be fine-tuned using the custom two-object dataset.

---

## 12. Evaluation

The trained detector will be evaluated using:

- Precision
- Recall
- mAP50
- mAP50-95
- Confusion matrix
- Precision-recall curves
- Visual inspection of predictions
- False-positive analysis
- False-negative analysis

---

## 13. Success Criteria

The first experiment is considered technically successful if:

1. The complete data pipeline executes successfully.

2. The custom dataset can be annotated and converted into YOLO format.

3. YOLO training completes successfully.

4. Validation and inference execute successfully.

5. The model can detect both custom objects on unseen test trials.

6. Model limitations and failure cases can be identified and documented.

---

## 14. Limitations

This experiment does not demonstrate:

- Assembly-step recognition
- Correct assembly sequence detection
- Screw-to-hole compatibility reasoning
- Final thesis-system performance
- General industrial robustness

The experiment is intended to validate the custom YOLO development workflow before applying it to the final assembly-monitoring use case.