"""
detect_near_duplicates.py

HOW TO RUN THE FILE:
    python detect_near_duplicates.py  (TYPE THIS IN TERMINAL)

Prototype V2 - Near-Duplicate Image Detection

Purpose:
    Detect visually similar extracted frames originating from
    the same recorded trial.

Why this is required:
    SHA-256 detects only files with exactly identical bytes.

    Consecutive video frames can look almost identical while
    still having different file contents.

Method used in Version 1:
    1. Load frame traceability information from frame_manifest.csv.
    2. Group images by trial.
    3. Sort images by source frame number.
    4. Compare each image with a limited number of following images.
    5. Resize and convert images to grayscale.
    6. Calculate Structural Similarity Index (SSIM).
    7. Flag highly similar image pairs.
    8. Save results to CSV.

Important:
    This script DOES NOT delete images.

    It creates candidate near-duplicate pairs for later review.
"""

from pathlib import Path
import csv

import cv2
from skimage.metrics import structural_similarity


# ============================================================
# CONFIGURATION
# ============================================================

# Images are resized before comparison.
#
# We do not need full 1280x720 resolution to estimate global
# visual similarity.
#
# Smaller images make comparison significantly faster.

COMPARISON_WIDTH = 320

COMPARISON_HEIGHT = 180


# SSIM ranges approximately from:
#
# -1 = very different
#  0 = low similarity
#  1 = identical structure
#
# This is an INITIAL threshold only.
#
# It must later be calibrated using actual thesis data.

SIMILARITY_THRESHOLD = 0.95


# Number of following images compared with each image.
#
# Example:
#
# LOCAL_WINDOW_SIZE = 3
#
# Image 1 compared with:
#     Image 2
#     Image 3
#     Image 4
#
# Image 2 compared with:
#     Image 3
#     Image 4
#     Image 5

LOCAL_WINDOW_SIZE = 3


# ============================================================
# PROJECT PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parent


IMAGE_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "extracted"
    / "images"
)


FRAME_MANIFEST_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "extracted"
    / "metadata"
    / "frame_manifest.csv"
)


QUALITY_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "quality"
)


NEAR_DUPLICATE_REPORT_FILE = (
    QUALITY_DIR
    / "near_duplicate_report.csv"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

QUALITY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# REPORT COLUMNS
# ============================================================

REPORT_COLUMNS = [
    "trial_id",
    "image_a",
    "image_b",
    "source_frame_a",
    "source_frame_b",
    "frame_distance",
    "similarity_score",
    "is_near_duplicate",
]


# ============================================================
# LOAD FRAME MANIFEST
# ============================================================

def load_frame_manifest():
    """
    Load frame_manifest.csv and group image records by trial ID.
    """

    trials = {}


    if not FRAME_MANIFEST_FILE.exists():

        print(
            f"ERROR: Frame manifest does not exist:\n"
            f"{FRAME_MANIFEST_FILE}"
        )

        return trials


    with FRAME_MANIFEST_FILE.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        reader = csv.DictReader(csv_file)


        for row in reader:

            trial_id = row["trial_id"]

            image_filename = row["image_filename"]

            source_frame_number = int(
                row["source_frame_number"]
            )


            record = {
                "image_filename": image_filename,
                "source_frame_number": source_frame_number,
            }


            if trial_id not in trials:

                trials[trial_id] = []


            trials[trial_id].append(record)


    # Sort every trial by original video frame number.

    for trial_id in trials:

        trials[trial_id].sort(
            key=lambda record:
            record["source_frame_number"]
        )


    return trials


# ============================================================
# LOAD IMAGE FOR COMPARISON
# ============================================================

def load_comparison_image(image_path):
    """
    Read image, convert to grayscale, and resize for faster
    structural-similarity comparison.
    """

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )


    if image is None:

        return None


    resized_image = cv2.resize(
        image,
        (
            COMPARISON_WIDTH,
            COMPARISON_HEIGHT,
        ),
        interpolation=cv2.INTER_AREA,
    )


    return resized_image


# ============================================================
# CALCULATE SSIM
# ============================================================

def calculate_similarity(image_a, image_b):
    """
    Calculate Structural Similarity Index between two images.
    """

    score = structural_similarity(
        image_a,
        image_b,
        data_range=255,
    )


    return float(score)


# ============================================================
# ANALYSE ONE TRIAL
# ============================================================

def analyse_trial(
    trial_id,
    records,
):
    """
    Compare temporally nearby images within one trial.
    """

    results = []


    print()

    print("=" * 70)

    print(
        f"ANALYSING TRIAL: {trial_id}"
    )

    print(
        f"Images in trial: {len(records)}"
    )

    print("=" * 70)


    # Cache loaded images so we do not repeatedly read the
    # same image from disk.

    image_cache = {}


    def get_image(image_filename):

        if image_filename not in image_cache:

            image_path = (
                IMAGE_DIR
                / image_filename
            )


            image_cache[image_filename] = (
                load_comparison_image(
                    image_path
                )
            )


        return image_cache[image_filename]


    # --------------------------------------------------------
    # LOCAL TEMPORAL COMPARISON
    # --------------------------------------------------------

    for index_a in range(len(records)):

        record_a = records[index_a]

        image_a = get_image(
            record_a["image_filename"]
        )


        if image_a is None:

            print(
                f"WARNING: Could not read "
                f"{record_a['image_filename']}"
            )

            continue


        comparison_end = min(
            index_a
            + LOCAL_WINDOW_SIZE
            + 1,

            len(records),
        )


        for index_b in range(
            index_a + 1,
            comparison_end,
        ):

            record_b = records[index_b]


            image_b = get_image(
                record_b["image_filename"]
            )


            if image_b is None:

                print(
                    f"WARNING: Could not read "
                    f"{record_b['image_filename']}"
                )

                continue


            similarity_score = (
                calculate_similarity(
                    image_a,
                    image_b,
                )
            )


            is_near_duplicate = (
                similarity_score
                >=
                SIMILARITY_THRESHOLD
            )


            result = {

                "trial_id":
                    trial_id,

                "image_a":
                    record_a[
                        "image_filename"
                    ],

                "image_b":
                    record_b[
                        "image_filename"
                    ],

                "source_frame_a":
                    record_a[
                        "source_frame_number"
                    ],

                "source_frame_b":
                    record_b[
                        "source_frame_number"
                    ],

                "frame_distance":
                    record_b[
                        "source_frame_number"
                    ]
                    -
                    record_a[
                        "source_frame_number"
                    ],

                "similarity_score":
                    round(
                        similarity_score,
                        6,
                    ),

                "is_near_duplicate":
                    is_near_duplicate,
            }


            results.append(result)


    return results


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(results):
    """
    Save complete comparison results to CSV.
    """

    with NEAR_DUPLICATE_REPORT_FILE.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=REPORT_COLUMNS,
        )


        writer.writeheader()

        writer.writerows(results)


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    trials,
    results,
):

    total_trials = len(trials)

    total_images = sum(
        len(records)
        for records in trials.values()
    )


    total_comparisons = len(results)


    near_duplicate_pairs = sum(
        row["is_near_duplicate"]
        for row in results
    )


    if results:

        similarity_scores = [

            row["similarity_score"]

            for row in results

        ]


        minimum_similarity = min(
            similarity_scores
        )


        maximum_similarity = max(
            similarity_scores
        )


        mean_similarity = (
            sum(similarity_scores)
            /
            len(similarity_scores)
        )


    else:

        minimum_similarity = 0.0

        maximum_similarity = 0.0

        mean_similarity = 0.0


    print()

    print("=" * 70)

    print(
        "NEAR-DUPLICATE DETECTION REPORT"
    )

    print("=" * 70)


    print(
        f"Trials analysed            : "
        f"{total_trials}"
    )


    print(
        f"Images registered          : "
        f"{total_images}"
    )


    print(
        f"Image pairs compared       : "
        f"{total_comparisons}"
    )


    print(
        f"Near-duplicate pairs       : "
        f"{near_duplicate_pairs}"
    )


    print(
        f"Minimum similarity         : "
        f"{minimum_similarity:.6f}"
    )


    print(
        f"Maximum similarity         : "
        f"{maximum_similarity:.6f}"
    )


    print(
        f"Mean similarity            : "
        f"{mean_similarity:.6f}"
    )


    print(
        f"Current threshold          : "
        f"{SIMILARITY_THRESHOLD:.6f}"
    )


    print("-" * 70)


    if total_comparisons == 0:

        print(
            "STATUS: NOT ENOUGH IMAGES FOR COMPARISON"
        )


    elif near_duplicate_pairs == 0:

        print(
            "STATUS: NO NEAR-DUPLICATE PAIRS FLAGGED "
            "USING CURRENT THRESHOLD"
        )


    else:

        print(
            "STATUS: REVIEW FLAGGED NEAR-DUPLICATE PAIRS"
        )


    print("=" * 70)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 70)

    print(
        "PROTOTYPE V2 - NEAR-DUPLICATE IMAGE DETECTION"
    )

    print("=" * 70)


    print(
        f"Image folder : "
        f"{IMAGE_DIR}"
    )


    print(
        f"Frame manifest: "
        f"{FRAME_MANIFEST_FILE}"
    )


    print(
        f"Report file  : "
        f"{NEAR_DUPLICATE_REPORT_FILE}"
    )


    print(
        f"Threshold    : "
        f"{SIMILARITY_THRESHOLD}"
    )


    print(
        f"Local window : "
        f"{LOCAL_WINDOW_SIZE}"
    )


    print("-" * 70)


    # --------------------------------------------------------
    # CHECK INPUTS
    # --------------------------------------------------------

    if not IMAGE_DIR.exists():

        print(
            "ERROR: Extracted image directory does not exist."
        )

        return


    # --------------------------------------------------------
    # LOAD MANIFEST
    # --------------------------------------------------------

    trials = load_frame_manifest()


    if not trials:

        print(
            "ERROR: No trial records were loaded."
        )

        return


    print(
        f"Trials loaded: "
        f"{len(trials)}"
    )


    # --------------------------------------------------------
    # ANALYSE ALL TRIALS
    # --------------------------------------------------------

    all_results = []


    for trial_id, records in trials.items():

        trial_results = analyse_trial(
            trial_id,
            records,
        )


        all_results.extend(
            trial_results
        )


    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    save_report(all_results)


    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print_summary(
        trials,
        all_results,
    )


    print()

    print(
        "Detailed report saved to:"
    )

    print(
        NEAR_DUPLICATE_REPORT_FILE
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()