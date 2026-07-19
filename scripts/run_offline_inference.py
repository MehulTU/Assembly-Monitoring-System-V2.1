"""
run_offline_inference.py

HOW TO RUN THE FILE:


MANUAL MODE:
    python scripts\run_offline_inference.py

DIRECT VIDEO MODE:
    python scripts\run_offline_inference.py --video "path_to_video.mp4"

OPTIONAL CUSTOM OUTPUT DIRECTORY:
    python scripts\run_offline_inference.py ^
        --video "path_to_video.mp4" ^
        --output-dir "path_to_output_folder"

PURPOSE:

Run the trained YOLO object-detection model on one video and convert
frame-by-frame predictions into structured detection data.

The script supports:

1. Manual mode for the original Prototype V2 workflow.
2. Direct-video mode for automated validation experiments.
3. Custom analytics output directories for isolated experiment results.

OUTPUTS:

    raw_detections.csv
    inference_summary.csv
"""

from pathlib import Path
import argparse
import csv
import cv2

from ultralytics import YOLO


# ======================================================================
# PROJECT CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_VIDEO_FOLDER = PROJECT_ROOT / "datasets" / "raw" / "videos"

DEFAULT_ANALYTICS_FOLDER = PROJECT_ROOT / "datasets" / "analytics"

MODEL_PATH = PROJECT_ROOT / "weights" / "best.pt"

CONFIDENCE_THRESHOLD = 0.25

IMAGE_SIZE = 640

DEVICE = 0


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def print_header(title):

    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


def parse_arguments():

    parser = argparse.ArgumentParser(

        description=(
            "Run Prototype V2 offline YOLO inference "
            "on one video."
        )
    )

    parser.add_argument(

        "--video",

        type=str,

        default=None,

        help=(
            "Optional path to the exact video to analyse. "
            "If omitted, manual video selection is used."
        ),
    )

    parser.add_argument(

        "--output-dir",

        type=str,

        default=None,

        help=(
            "Optional output directory for raw_detections.csv "
            "and inference_summary.csv."
        ),
    )

    return parser.parse_args()


def get_video_files():

    return sorted(RAW_VIDEO_FOLDER.glob("*.mp4"))


def select_video(video_files):

    print_header("AVAILABLE RECORDED VIDEOS")

    for index, video_path in enumerate(video_files, start=1):

        print(f"{index:3d}. {video_path.name}")

    print()

    while True:

        selection = input(
            "Enter video number to analyse: "
        ).strip()

        try:

            selection = int(selection)

            if 1 <= selection <= len(video_files):

                return video_files[selection - 1]

        except ValueError:

            pass

        print(
            "Invalid selection. "
            "Enter one of the listed video numbers."
        )


def resolve_video(video_argument):

    if video_argument is not None:

        selected_video = Path(video_argument).expanduser().resolve()

        if not selected_video.exists():

            raise FileNotFoundError(
                f"Requested video not found:\n{selected_video}"
            )

        if not selected_video.is_file():

            raise FileNotFoundError(
                f"Requested video path is not a file:\n"
                f"{selected_video}"
            )

        return selected_video


    video_files = get_video_files()

    if not video_files:

        raise FileNotFoundError(
            f"No MP4 videos found inside:\n{RAW_VIDEO_FOLDER}"
        )

    print(f"Recorded videos found: {len(video_files)}")

    return select_video(video_files)


def resolve_output_directory(output_argument):

    if output_argument is None:

        return DEFAULT_ANALYTICS_FOLDER

    return Path(output_argument).expanduser().resolve()


# ======================================================================
# MAIN PROCESS
# ======================================================================

def main():

    args = parse_arguments()

    selected_video = resolve_video(args.video)

    analytics_folder = resolve_output_directory(args.output_dir)

    raw_detection_file = (
        analytics_folder / "raw_detections.csv"
    )

    inference_summary_file = (
        analytics_folder / "inference_summary.csv"
    )


    print_header(
        "PROTOTYPE V2 - OFFLINE YOLO INFERENCE "
        "AND RAW DETECTION LOGGING"
    )

    print(f"Project root:\n{PROJECT_ROOT}")

    print()

    print(f"Selected video:\n{selected_video}")

    print()

    print(f"YOLO model:\n{MODEL_PATH}")

    print()

    print(f"Analytics output directory:\n{analytics_folder}")

    print()

    print(f"Raw detection output:\n{raw_detection_file}")

    print()

    print(f"Inference summary:\n{inference_summary_file}")


    # ------------------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # ------------------------------------------------------------------

    analytics_folder.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ------------------------------------------------------------------
    # VERIFY MODEL
    # ------------------------------------------------------------------

    print_header("VERIFYING MODEL")

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"YOLO model not found:\n{MODEL_PATH}"
        )

    print("YOLO model found.")


    # ------------------------------------------------------------------
    # VERIFY VIDEO
    # ------------------------------------------------------------------

    print_header("VERIFYING INPUT VIDEO")

    if not selected_video.exists():

        raise FileNotFoundError(
            f"Video not found:\n{selected_video}"
        )

    print("Input video found.")

    print(f"Video selected:\n{selected_video}")


    # ------------------------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------------------------

    print_header("LOADING YOLO MODEL")

    model = YOLO(str(MODEL_PATH))

    print("YOLO model loaded successfully.")


    # ------------------------------------------------------------------
    # OPEN VIDEO
    # ------------------------------------------------------------------

    print_header("OPENING VIDEO")

    video_capture = cv2.VideoCapture(str(selected_video))

    if not video_capture.isOpened():

        raise RuntimeError(
            f"Could not open video:\n{selected_video}"
        )

    fps = video_capture.get(cv2.CAP_PROP_FPS)

    total_frames = int(
        video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    frame_width = int(
        video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    frame_height = int(
        video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    if fps <= 0:

        video_capture.release()

        raise RuntimeError("Invalid video FPS.")

    video_duration_seconds = total_frames / fps

    print(f"Video FPS       : {fps:.3f}")

    print(f"Total frames    : {total_frames}")

    print(f"Resolution      : {frame_width} x {frame_height}")

    print(
        f"Video duration  : "
        f"{video_duration_seconds:.3f} seconds"
    )


    # ------------------------------------------------------------------
    # PREPARE CSV
    # ------------------------------------------------------------------

    detection_columns = [

        "source_video",
        "frame_number",
        "timestamp_seconds",

        "class_id",
        "class_name",

        "confidence",

        "x1",
        "y1",
        "x2",
        "y2",

        "box_center_x",
        "box_center_y",

        "box_width",
        "box_height",
    ]


    # ------------------------------------------------------------------
    # RUN INFERENCE
    # ------------------------------------------------------------------

    print_header("RUNNING FRAME-BY-FRAME YOLO INFERENCE")

    frame_number = 0

    processed_frames = 0

    frames_with_detections = 0

    total_detections = 0

    class_counts = {}


    with open(

        raw_detection_file,

        "w",

        newline="",

        encoding="utf-8",

    ) as detection_csv:

        writer = csv.DictWriter(

            detection_csv,

            fieldnames=detection_columns,
        )

        writer.writeheader()


        while True:

            read_success, frame = video_capture.read()

            if not read_success:

                break


            timestamp_seconds = frame_number / fps


            results = model.predict(

                source=frame,

                imgsz=IMAGE_SIZE,

                conf=CONFIDENCE_THRESHOLD,

                device=DEVICE,

                verbose=False,
            )


            result = results[0]

            frame_detection_count = 0


            if result.boxes is not None:

                for box in result.boxes:

                    class_id = int(box.cls.item())

                    class_name = model.names[class_id]

                    confidence = float(box.conf.item())


                    x1, y1, x2, y2 = (

                        box.xyxy[0].cpu().tolist()
                    )


                    box_center_x = (x1 + x2) / 2

                    box_center_y = (y1 + y2) / 2

                    box_width = x2 - x1

                    box_height = y2 - y1


                    writer.writerow({

                        "source_video":
                            selected_video.name,

                        "frame_number":
                            frame_number,

                        "timestamp_seconds":
                            round(timestamp_seconds, 6),

                        "class_id":
                            class_id,

                        "class_name":
                            class_name,

                        "confidence":
                            round(confidence, 6),

                        "x1":
                            round(x1, 3),

                        "y1":
                            round(y1, 3),

                        "x2":
                            round(x2, 3),

                        "y2":
                            round(y2, 3),

                        "box_center_x":
                            round(box_center_x, 3),

                        "box_center_y":
                            round(box_center_y, 3),

                        "box_width":
                            round(box_width, 3),

                        "box_height":
                            round(box_height, 3),
                    })


                    total_detections += 1

                    frame_detection_count += 1


                    class_counts[class_name] = (

                        class_counts.get(class_name, 0) + 1
                    )


            if frame_detection_count > 0:

                frames_with_detections += 1


            processed_frames += 1


            if processed_frames % 100 == 0:

                print(

                    f"Processed "
                    f"{processed_frames}/{total_frames} frames"
                )


            frame_number += 1


    video_capture.release()


    # ------------------------------------------------------------------
    # SAVE SUMMARY
    # ------------------------------------------------------------------

    print_header("SAVING INFERENCE SUMMARY")


    frames_without_detections = (

        processed_frames - frames_with_detections
    )


    detection_rate_percent = (

        frames_with_detections
        / processed_frames
        * 100

        if processed_frames > 0

        else 0
    )


    summary_columns = [

        "source_video",

        "processed_frames",

        "frames_with_detections",

        "frames_without_detections",

        "detection_rate_percent",

        "total_detections",

        "marker_detections",

        "power_adapter_detections",

        "video_fps",

        "video_duration_seconds",

        "confidence_threshold",

        "model_path",
    ]


    with open(

        inference_summary_file,

        "w",

        newline="",

        encoding="utf-8",

    ) as summary_csv:

        writer = csv.DictWriter(

            summary_csv,

            fieldnames=summary_columns,
        )

        writer.writeheader()


        writer.writerow({

            "source_video":
                selected_video.name,

            "processed_frames":
                processed_frames,

            "frames_with_detections":
                frames_with_detections,

            "frames_without_detections":
                frames_without_detections,

            "detection_rate_percent":
                round(detection_rate_percent, 3),

            "total_detections":
                total_detections,

            "marker_detections":
                class_counts.get("marker", 0),

            "power_adapter_detections":
                class_counts.get("power_adapter", 0),

            "video_fps":
                round(fps, 6),

            "video_duration_seconds":
                round(video_duration_seconds, 6),

            "confidence_threshold":
                CONFIDENCE_THRESHOLD,

            "model_path":
                str(MODEL_PATH),
        })


    # ------------------------------------------------------------------
    # FINAL REPORT
    # ------------------------------------------------------------------

    print_header("OFFLINE INFERENCE SUMMARY")

    print(f"Video analysed             : {selected_video.name}")

    print(f"Processed frames           : {processed_frames}")

    print(
        f"Frames with detections     : "
        f"{frames_with_detections}"
    )

    print(
        f"Frames without detections  : "
        f"{frames_without_detections}"
    )

    print(
        f"Detection frame rate       : "
        f"{detection_rate_percent:.2f}%"
    )

    print(f"Total detections           : {total_detections}")

    print(
        f"Marker detections          : "
        f"{class_counts.get('marker', 0)}"
    )

    print(
        f"Power adapter detections   : "
        f"{class_counts.get('power_adapter', 0)}"
    )

    print()

    print(f"Raw detections saved to:\n{raw_detection_file}")

    print()

    print(
        f"Inference summary saved to:\n"
        f"{inference_summary_file}"
    )

    print_header(
        "STATUS: OFFLINE YOLO INFERENCE COMPLETED SUCCESSFULLY"
    )

    print(

        "Next stage:\n"

        "Build the temporal detection timeline and suppress "
        "frame-level prediction flicker."
    )


if __name__ == "__main__":

    main()