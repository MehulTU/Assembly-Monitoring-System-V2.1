"""
extract_frames.py

HOW TO RUN THE FILE:
    python extract_frames.py  (TYPE THIS IN TERMINAL)


Prototype V2 - Frame Extraction Tool

Purpose:
    Convert recorded raw videos into candidate images
    for later dataset cleaning, annotation, and YOLO training.

Input:
    datasets/raw/videos/*.mp4

Outputs:
    1. Extracted image files.
    2. CSV manifest containing traceability information.

Important:
    This script does NOT create the final YOLO dataset.

    It creates candidate images that must later be:
        - reviewed,
        - cleaned,
        - selected,
        - annotated,
        - and split into train/validation/test sets.
"""

from pathlib import Path
import csv

import cv2


# ============================================================
# CONFIGURATION
# ============================================================

# Extract one frame after this many source-video frames.
#
# Example:
#
# Source video = 30 FPS
# FRAME_INTERVAL = 30
#
# Approximately one image per second.

FRAME_INTERVAL = 30


# Image format used for extracted frames.

IMAGE_EXTENSION = ".jpg"


# JPEG quality:
#
# 0   = lowest quality
# 100 = highest quality

JPEG_QUALITY = 95


# ============================================================
# PROJECT PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parent


# Raw videos created by record_dataset.py

RAW_VIDEO_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "videos"
)


# Candidate extracted images

EXTRACTED_IMAGE_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "extracted"
    / "images"
)


# Manifest folder

MANIFEST_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "extracted"
    / "metadata"
)


# Manifest file

MANIFEST_FILE = (
    MANIFEST_DIR
    / "frame_manifest.csv"
)


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

EXTRACTED_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# MANIFEST COLUMNS
# ============================================================

MANIFEST_COLUMNS = [
    "image_filename",
    "source_video",
    "trial_id",
    "source_frame_number",
    "source_time_seconds",
    "source_video_fps",
    "frame_width",
    "frame_height",
]


# ============================================================
# CREATE MANIFEST IF REQUIRED
# ============================================================

if not MANIFEST_FILE.exists():

    with MANIFEST_FILE.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=MANIFEST_COLUMNS,
        )

        writer.writeheader()


# ============================================================
# CHECK WHETHER IMAGE ALREADY EXISTS IN MANIFEST
# ============================================================

def load_existing_images():

    existing_images = set()

    if not MANIFEST_FILE.exists():

        return existing_images


    with MANIFEST_FILE.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        for row in reader:

            existing_images.add(
                row["image_filename"]
            )


    return existing_images


# ============================================================
# SAVE MANIFEST ROW
# ============================================================

def save_manifest_row(metadata):

    with MANIFEST_FILE.open(
        mode="a",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=MANIFEST_COLUMNS,
        )

        writer.writerow(metadata)


# ============================================================
# EXTRACT FRAMES FROM ONE VIDEO
# ============================================================

def extract_frames_from_video(
    video_path,
    existing_images,
):

    print()

    print("=" * 70)

    print(
        f"PROCESSING VIDEO: "
        f"{video_path.name}"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # OPEN VIDEO
    # --------------------------------------------------------

    video = cv2.VideoCapture(
        str(video_path)
    )


    if not video.isOpened():

        print(
            f"ERROR: Could not open video: "
            f"{video_path}"
        )

        return 0


    # --------------------------------------------------------
    # READ VIDEO INFORMATION
    # --------------------------------------------------------

    source_fps = video.get(
        cv2.CAP_PROP_FPS
    )


    total_frames = int(
        video.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    frame_width = int(
        video.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )


    frame_height = int(
        video.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    print(
        f"Reported FPS     : "
        f"{source_fps:.3f}"
    )

    print(
        f"Total frames     : "
        f"{total_frames}"
    )

    print(
        f"Resolution       : "
        f"{frame_width}x{frame_height}"
    )

    print(
        f"Extraction interval: "
        f"every {FRAME_INTERVAL} frames"
    )


    # --------------------------------------------------------
    # TRIAL ID
    # --------------------------------------------------------

    # Example:
    #
    # T_20260708_183045_123456.mp4
    #
    # becomes:
    #
    # T_20260708_183045_123456

    trial_id = video_path.stem


    # --------------------------------------------------------
    # PROCESS VIDEO
    # --------------------------------------------------------

    source_frame_number = 0

    extracted_count = 0


    while True:

        success, frame = video.read()


        if not success:

            break


        # ----------------------------------------------------
        # SELECT FRAME FOR EXTRACTION
        # ----------------------------------------------------

        if source_frame_number % FRAME_INTERVAL == 0:


            # ------------------------------------------------
            # CREATE TRACEABLE IMAGE NAME
            # ------------------------------------------------

            image_filename = (

                f"{trial_id}"
                f"_frame_"
                f"{source_frame_number:06d}"
                f"{IMAGE_EXTENSION}"

            )


            image_path = (
                EXTRACTED_IMAGE_DIR
                / image_filename
            )


            # ------------------------------------------------
            # AVOID DUPLICATE MANIFEST ENTRIES
            # ------------------------------------------------

            if image_filename in existing_images:

                source_frame_number += 1

                continue


            # ------------------------------------------------
            # SAVE IMAGE
            # ------------------------------------------------

            save_success = cv2.imwrite(

                str(image_path),

                frame,

                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    JPEG_QUALITY,
                ],
            )


            if not save_success:

                print(
                    f"WARNING: Could not save image: "
                    f"{image_path}"
                )

                source_frame_number += 1

                continue


            # ------------------------------------------------
            # CALCULATE SOURCE TIME
            # ------------------------------------------------

            if source_fps > 0:

                source_time_seconds = (
                    source_frame_number
                    / source_fps
                )

            else:

                source_time_seconds = -1.0


            # ------------------------------------------------
            # CREATE MANIFEST ENTRY
            # ------------------------------------------------

            metadata = {

                "image_filename":
                    image_filename,

                "source_video":
                    video_path.name,

                "trial_id":
                    trial_id,

                "source_frame_number":
                    source_frame_number,

                "source_time_seconds":
                    round(
                        source_time_seconds,
                        6,
                    ),

                "source_video_fps":
                    round(
                        source_fps,
                        6,
                    ),

                "frame_width":
                    frame_width,

                "frame_height":
                    frame_height,
            }


            # ------------------------------------------------
            # SAVE MANIFEST ENTRY
            # ------------------------------------------------

            save_manifest_row(metadata)


            existing_images.add(
                image_filename
            )


            extracted_count += 1


        source_frame_number += 1


    # --------------------------------------------------------
    # CLOSE VIDEO
    # --------------------------------------------------------

    video.release()


    print(
        f"New images extracted: "
        f"{extracted_count}"
    )


    return extracted_count


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 70)

    print(
        "PROTOTYPE V2 - FRAME EXTRACTION TOOL"
    )

    print("=" * 70)


    print(
        f"Raw video folder : "
        f"{RAW_VIDEO_DIR}"
    )


    print(
        f"Image output folder: "
        f"{EXTRACTED_IMAGE_DIR}"
    )


    print(
        f"Manifest file    : "
        f"{MANIFEST_FILE}"
    )


    print(
        f"Frame interval   : "
        f"{FRAME_INTERVAL}"
    )


    print("-" * 70)


    # --------------------------------------------------------
    # CHECK INPUT DIRECTORY
    # --------------------------------------------------------

    if not RAW_VIDEO_DIR.exists():

        print(
            "ERROR: Raw video folder does not exist."
        )

        print(
            "Run record_dataset.py first."
        )

        return


    # --------------------------------------------------------
    # FIND VIDEOS
    # --------------------------------------------------------

    video_files = sorted(

        RAW_VIDEO_DIR.glob("*.mp4")

    )


    if not video_files:

        print(
            "ERROR: No MP4 videos found."
        )

        print(
            "Record at least one trial first."
        )

        return


    print(
        f"Videos found: "
        f"{len(video_files)}"
    )


    # --------------------------------------------------------
    # LOAD EXISTING MANIFEST DATA
    # --------------------------------------------------------

    existing_images = (
        load_existing_images()
    )


    print(
        f"Images already registered: "
        f"{len(existing_images)}"
    )


    # --------------------------------------------------------
    # PROCESS ALL VIDEOS
    # --------------------------------------------------------

    total_new_images = 0


    for video_path in video_files:

        extracted_count = (
            extract_frames_from_video(
                video_path,
                existing_images,
            )
        )


        total_new_images += (
            extracted_count
        )


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "FRAME EXTRACTION COMPLETED"
    )

    print("=" * 70)


    print(
        f"Videos processed: "
        f"{len(video_files)}"
    )


    print(
        f"New images extracted: "
        f"{total_new_images}"
    )


    print(
        f"Images saved to: "
        f"{EXTRACTED_IMAGE_DIR}"
    )


    print(
        f"Manifest saved to: "
        f"{MANIFEST_FILE}"
    )


    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()