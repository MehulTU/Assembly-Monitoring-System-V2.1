"""
processing/object_detector.py

Object Detection Module (Processing Subsystem)

Uses YOLO to identify assembly objects.

V2.1 confidence upgrade:
    - Per-class confidence statistics for every frame
      (best, mean, detection count per class).
    - Rolling average confidence per class over the last N frames,
      so the display can show a STABLE confidence value instead of
      a number that flickers every frame.
    - Configuration comes from config/settings.py instead of being
      hard-coded in main.py.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ultralytics import YOLO


@dataclass
class DetectedObject:
    class_name: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]


@dataclass
class ClassConfidenceStats:
    """Confidence statistics for one class in one frame."""
    class_name: str
    count: int                 # detections of this class in the frame
    best_confidence: float     # highest confidence this frame
    mean_confidence: float     # mean confidence this frame
    rolling_confidence: float  # mean of best confidence over last N frames


@dataclass
class ObjectDetectionResult:
    objects: List[DetectedObject] = field(default_factory=list)

    # class_name -> ClassConfidenceStats (filled by ObjectDetector)
    class_stats: Dict[str, ClassConfidenceStats] = field(
        default_factory=dict
    )

    def best_per_class(self) -> Dict[str, DetectedObject]:
        """Return the highest-confidence detection for each class."""
        best = {}
        for obj in self.objects:
            current = best.get(obj.class_name)
            if current is None or obj.confidence > current.confidence:
                best[obj.class_name] = obj
        return best


class ObjectDetector:

    def __init__(
        self,
        model_path,
        confidence=0.25,
        rolling_window=30,
        imgsz=640,
        device=None,
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device

        # class_name -> deque of best-confidence values (last N frames
        # in which the class was detected)
        self._rolling: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=rolling_window)
        )

    def process(self, frame) -> ObjectDetectionResult:

        results = self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detected_objects = []

        for result in results:

            names = result.names

            for box in result.boxes:

                cls = int(box.cls[0])
                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detected_objects.append(
                    DetectedObject(
                        class_name=names[cls],
                        confidence=conf,
                        bounding_box=(x1, y1, x2, y2),
                    )
                )

        # ----------------------------------------------------
        # PER-CLASS CONFIDENCE STATISTICS
        # ----------------------------------------------------

        per_class_confidences: Dict[str, List[float]] = defaultdict(list)

        for obj in detected_objects:
            per_class_confidences[obj.class_name].append(obj.confidence)

        class_stats: Dict[str, ClassConfidenceStats] = {}

        for class_name, confidences in per_class_confidences.items():

            best = max(confidences)
            mean = sum(confidences) / len(confidences)

            # Update rolling history with this frame's best confidence
            self._rolling[class_name].append(best)

            history = self._rolling[class_name]
            rolling = sum(history) / len(history)

            class_stats[class_name] = ClassConfidenceStats(
                class_name=class_name,
                count=len(confidences),
                best_confidence=best,
                mean_confidence=mean,
                rolling_confidence=rolling,
            )

        return ObjectDetectionResult(
            objects=detected_objects,
            class_stats=class_stats,
        )
