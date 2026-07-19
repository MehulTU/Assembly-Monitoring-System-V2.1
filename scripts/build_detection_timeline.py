"""
build_detection_timeline.py

HOW TO RUN THE FILE:
    python scripts\build_detection_timeline.py  (TYPE THIS IN TERMINAL)

Prototype V2 - Temporal Detection Timeline Builder

Purpose:
    Convert raw frame-level YOLO detections into a complete and temporally
    filtered detection timeline.

Why this script is needed:
    YOLO processes every video frame independently.

    Because of confidence changes, motion blur, occlusion, or detection
    uncertainty, an object may temporarily disappear from YOLO predictions
    for one or several frames.

    Example:

        Frame 100 -> marker detected
        Frame 101 -> marker detected
        Frame 102 -> marker missing
        Frame 103 -> marker detected
        Frame 104 -> marker detected

    The Analytics Model should normally not interpret Frame 102 as a real
    object-removal event.

    This script therefore:

        1. Reads raw_detections.csv.
        2. Reads inference_summary.csv.
        3. Reconstructs EVERY processed video frame.
        4. Aggregates multiple YOLO boxes belonging to the same class.
        5. Creates raw object-presence signals.
        6. Applies temporal majority filtering.
        7. Produces one CSV row per video frame.

Input:

    datasets/analytics/raw_detections.csv

    datasets/analytics/inference_summary.csv

Output:

    datasets/analytics/detection_timeline.csv

Important:
    This script does NOT determine assembly states or process events.

    It only converts noisy frame-level YOLO detections into stable object
    presence signals for the State Machine.
"""

from pathlib import Path
import csv
from collections import defaultdict, deque


# ======================================================================
# PROJECT CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANALYTICS_FOLDER = PROJECT_ROOT / "datasets" / "analytics"

RAW_DETECTION_FILE = ANALYTICS_FOLDER / "raw_detections.csv"

INFERENCE_SUMMARY_FILE = ANALYTICS_FOLDER / "inference_summary.csv"

TIMELINE_FILE = ANALYTICS_FOLDER / "detection_timeline.csv"


# ======================================================================
# TEMPORAL FILTER CONFIGURATION
# ======================================================================

# Number of frames used by the rolling majority filter.
#
# At approximately 30 FPS:
#
# 5 frames ≈ 0.17 seconds.

WINDOW_SIZE = 5


# Object must be detected in at least this many frames inside the
# rolling window to be considered stably present.

MINIMUM_POSITIVE_FRAMES = 3


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def print_header(title):
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


def text_to_int(value):
    return int(float(value))


def text_to_float(value):
    return float(value)


def load_inference_summary():
    """
    Load information about the previously analysed video.
    """

    if not INFERENCE_SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Inference summary not found:\n{INFERENCE_SUMMARY_FILE}"
        )

    with open(
        INFERENCE_SUMMARY_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        rows = list(csv.DictReader(file))

    if len(rows) != 1:
        raise ValueError(
            "inference_summary.csv must contain exactly one data row."
        )

    return rows[0]


def load_raw_detections():
    """
    Load all raw YOLO detections.

    Returns:
        Dictionary structure:

        detections[frame_number][class_name] = list of detection rows
    """

    if not RAW_DETECTION_FILE.exists():
        raise FileNotFoundError(
            f"Raw detection file not found:\n{RAW_DETECTION_FILE}"
        )

    detections = defaultdict(lambda: defaultdict(list))

    total_rows = 0

    with open(
        RAW_DETECTION_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            frame_number = text_to_int(row["frame_number"])

            class_name = row["class_name"].strip()

            detections[frame_number][class_name].append(row)

            total_rows += 1

    return detections, total_rows


def get_best_detection(frame_detections, class_name):
    """
    Return the highest-confidence detection for one object class
    in one video frame.

    If the object was not detected, return None.
    """

    class_detections = frame_detections.get(class_name, [])

    if not class_detections:
        return None

    return max(
        class_detections,
        key=lambda row: text_to_float(row["confidence"])
    )


def majority_filter(history):
    """
    Convert recent raw presence values into a stable presence value.

    Example:

        [1, 1, 0, 1, 1]

    Positive frames = 4

    4 >= 3

    Stable presence = True
    """

    positive_count = sum(history)

    return positive_count >= MINIMUM_POSITIVE_FRAMES


# ======================================================================
# MAIN PROCESS
# ======================================================================

def main():

    print_header(
        "PROTOTYPE V2 - TEMPORAL DETECTION TIMELINE BUILDER"
    )

    print(f"Raw detections:\n{RAW_DETECTION_FILE}")
    print()

    print(f"Inference summary:\n{INFERENCE_SUMMARY_FILE}")
    print()

    print(f"Timeline output:\n{TIMELINE_FILE}")
    print()

    print(f"Rolling window size       : {WINDOW_SIZE} frames")
    print(
        f"Minimum positive frames   : "
        f"{MINIMUM_POSITIVE_FRAMES}"
    )

    # ------------------------------------------------------------------
    # VERIFY CONFIGURATION
    # ------------------------------------------------------------------

    print_header("VERIFYING TEMPORAL FILTER CONFIGURATION")

    if WINDOW_SIZE <= 0:
        raise ValueError("WINDOW_SIZE must be greater than zero.")

    if MINIMUM_POSITIVE_FRAMES <= 0:
        raise ValueError(
            "MINIMUM_POSITIVE_FRAMES must be greater than zero."
        )

    if MINIMUM_POSITIVE_FRAMES > WINDOW_SIZE:
        raise ValueError(
            "MINIMUM_POSITIVE_FRAMES cannot exceed WINDOW_SIZE."
        )

    print("Temporal filter configuration is valid.")

    # ------------------------------------------------------------------
    # LOAD INPUT DATA
    # ------------------------------------------------------------------

    print_header("LOADING INPUT DATA")

    summary = load_inference_summary()

    detections, raw_detection_rows = load_raw_detections()

    source_video = summary["source_video"]

    processed_frames = text_to_int(summary["processed_frames"])

    video_fps = text_to_float(summary["video_fps"])

    if processed_frames <= 0:
        raise ValueError("Processed frame count must be positive.")

    if video_fps <= 0:
        raise ValueError("Video FPS must be positive.")

    print(f"Source video          : {source_video}")
    print(f"Processed frames      : {processed_frames}")
    print(f"Video FPS             : {video_fps:.3f}")
    print(f"Raw detection rows    : {raw_detection_rows}")
    print(f"Frames represented    : {len(detections)}")

    # ------------------------------------------------------------------
    # PREPARE FILTER HISTORIES
    # ------------------------------------------------------------------

    marker_history = deque(maxlen=WINDOW_SIZE)

    power_adapter_history = deque(maxlen=WINDOW_SIZE)

    # ------------------------------------------------------------------
    # PREPARE OUTPUT
    # ------------------------------------------------------------------

    timeline_columns = [

        "source_video",

        "frame_number",
        "timestamp_seconds",

        "marker_raw_present",
        "marker_raw_confidence",

        "power_adapter_raw_present",
        "power_adapter_raw_confidence",

        "marker_stable_present",
        "power_adapter_stable_present",

        "marker_window_positive_count",
        "power_adapter_window_positive_count",

        "temporal_window_frames",
        "minimum_positive_frames"
    ]

    raw_marker_positive_frames = 0
    raw_adapter_positive_frames = 0

    stable_marker_positive_frames = 0
    stable_adapter_positive_frames = 0

    marker_changed_by_filter = 0
    adapter_changed_by_filter = 0

    both_stable_frames = 0
    empty_stable_frames = 0

    # ------------------------------------------------------------------
    # BUILD COMPLETE TIMELINE
    # ------------------------------------------------------------------

    print_header("BUILDING TEMPORAL DETECTION TIMELINE")

    with open(
        TIMELINE_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=timeline_columns
        )

        writer.writeheader()

        for frame_number in range(processed_frames):

            timestamp_seconds = frame_number / video_fps

            frame_detections = detections.get(frame_number, {})

            # ----------------------------------------------------------
            # SELECT BEST DETECTION FOR EACH CLASS
            # ----------------------------------------------------------

            marker_detection = get_best_detection(
                frame_detections,
                "marker"
            )

            adapter_detection = get_best_detection(
                frame_detections,
                "power_adapter"
            )

            marker_raw_present = marker_detection is not None

            adapter_raw_present = adapter_detection is not None

            marker_raw_confidence = (
                text_to_float(marker_detection["confidence"])
                if marker_detection is not None
                else 0.0
            )

            adapter_raw_confidence = (
                text_to_float(adapter_detection["confidence"])
                if adapter_detection is not None
                else 0.0
            )

            # ----------------------------------------------------------
            # UPDATE TEMPORAL HISTORIES
            # ----------------------------------------------------------

            marker_history.append(
                1 if marker_raw_present else 0
            )

            power_adapter_history.append(
                1 if adapter_raw_present else 0
            )

            # ----------------------------------------------------------
            # TEMPORAL FILTERING
            # ----------------------------------------------------------
            #
            # During startup, the rolling history contains fewer than
            # WINDOW_SIZE frames.
            #
            # We preserve the raw state until a complete window exists.
            # This avoids forcing the first frames to False simply because
            # the history has not yet filled.
            # ----------------------------------------------------------

            if len(marker_history) < WINDOW_SIZE:

                marker_stable_present = marker_raw_present

            else:

                marker_stable_present = majority_filter(
                    marker_history
                )

            if len(power_adapter_history) < WINDOW_SIZE:

                adapter_stable_present = adapter_raw_present

            else:

                adapter_stable_present = majority_filter(
                    power_adapter_history
                )

            marker_window_positive_count = sum(marker_history)

            adapter_window_positive_count = sum(
                power_adapter_history
            )

            # ----------------------------------------------------------
            # STATISTICS
            # ----------------------------------------------------------

            if marker_raw_present:
                raw_marker_positive_frames += 1

            if adapter_raw_present:
                raw_adapter_positive_frames += 1

            if marker_stable_present:
                stable_marker_positive_frames += 1

            if adapter_stable_present:
                stable_adapter_positive_frames += 1

            if marker_raw_present != marker_stable_present:
                marker_changed_by_filter += 1

            if adapter_raw_present != adapter_stable_present:
                adapter_changed_by_filter += 1

            if (
                marker_stable_present
                and adapter_stable_present
            ):
                both_stable_frames += 1

            if (
                not marker_stable_present
                and not adapter_stable_present
            ):
                empty_stable_frames += 1

            # ----------------------------------------------------------
            # SAVE TIMELINE ROW
            # ----------------------------------------------------------

            writer.writerow({

                "source_video":
                    source_video,

                "frame_number":
                    frame_number,

                "timestamp_seconds":
                    round(timestamp_seconds, 6),

                "marker_raw_present":
                    marker_raw_present,

                "marker_raw_confidence":
                    round(marker_raw_confidence, 6),

                "power_adapter_raw_present":
                    adapter_raw_present,

                "power_adapter_raw_confidence":
                    round(adapter_raw_confidence, 6),

                "marker_stable_present":
                    marker_stable_present,

                "power_adapter_stable_present":
                    adapter_stable_present,

                "marker_window_positive_count":
                    marker_window_positive_count,

                "power_adapter_window_positive_count":
                    adapter_window_positive_count,

                "temporal_window_frames":
                    WINDOW_SIZE,

                "minimum_positive_frames":
                    MINIMUM_POSITIVE_FRAMES
            })

            if (frame_number + 1) % 100 == 0:

                print(
                    f"Processed "
                    f"{frame_number + 1}/{processed_frames} frames"
                )

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------

    print_header("TEMPORAL TIMELINE SUMMARY")

    print(f"Source video                     : {source_video}")
    print(f"Timeline rows                    : {processed_frames}")

    print()

    print(
        f"Raw marker-positive frames       : "
        f"{raw_marker_positive_frames}"
    )

    print(
        f"Stable marker-positive frames    : "
        f"{stable_marker_positive_frames}"
    )

    print(
        f"Marker states changed by filter  : "
        f"{marker_changed_by_filter}"
    )

    print()

    print(
        f"Raw adapter-positive frames      : "
        f"{raw_adapter_positive_frames}"
    )

    print(
        f"Stable adapter-positive frames   : "
        f"{stable_adapter_positive_frames}"
    )

    print(
        f"Adapter states changed by filter : "
        f"{adapter_changed_by_filter}"
    )

    print()

    print(
        f"Stable BOTH-present frames        : "
        f"{both_stable_frames}"
    )

    print(
        f"Stable EMPTY frames               : "
        f"{empty_stable_frames}"
    )

    print()

    print(f"Timeline saved to:\n{TIMELINE_FILE}")

    print_header(
        "STATUS: TEMPORAL DETECTION TIMELINE CREATED SUCCESSFULLY"
    )

    print(
        "Next stage:\n"
        "Convert stable object-presence signals into assembly states, "
        "state transitions, and process events."
    )


if __name__ == "__main__":
    main()