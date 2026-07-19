"""
check_dataset_quality.py

HOW TO RUN THE FILE:
    python check_dataset_quality.py  (TYPE THIS IN TERMINAL)

Prototype V2 - Dataset Image Quality Analysis

Purpose:
    Analyse candidate images extracted from recorded assembly videos.

The script checks:
    1. Whether each image can be opened.
    2. Image resolution.
    3. Sharpness / blur score.
    4. Mean brightness.
    5. Contrast.
    6. Simple exact file duplicates using SHA-256 hashes.
    7. Generates a CSV quality report.
    8. Prints a complete dataset summary.

Important:
    This script DOES NOT delete images.

    Images are measured and flagged for later review.
"""

from pathlib import Path
import csv
import hashlib

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_WIDTH = 1280
EXPECTED_HEIGHT = 720


# ------------------------------------------------------------
# INITIAL QUALITY THRESHOLDS
# ------------------------------------------------------------
#
# These values are starting thresholds only.
#
# They are NOT universal scientific limits.
#
# We will later inspect real assembly images and adjust them
# based on the actual camera, objects, lighting, and workspace.
# ------------------------------------------------------------

BLUR_THRESHOLD = 100.0

DARK_THRESHOLD = 50.0

BRIGHT_THRESHOLD = 205.0

LOW_CONTRAST_THRESHOLD = 25.0


# ============================================================
# PROJECT PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parent


# Images created by extract_frames.py

IMAGE_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "extracted"
    / "images"
)


# Quality-analysis output folder

QUALITY_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "quality"
)


# CSV report

QUALITY_REPORT_FILE = (
    QUALITY_DIR
    / "image_quality_report.csv"
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
    "image_filename",
    "read_success",
    "width",
    "height",
    "resolution_ok",
    "blur_score",
    "is_blurry",
    "mean_brightness",
    "is_too_dark",
    "is_too_bright",
    "contrast_score",
    "is_low_contrast",
    "sha256",
    "is_exact_duplicate",
    "duplicate_of",
    "warning_count",
    "review_required",
]


# ============================================================
# CALCULATE SHA-256 FILE HASH
# ============================================================

def calculate_sha256(file_path):
    """
    Calculate a SHA-256 hash from the exact file contents.

    Two files with exactly the same bytes will have the same
    hash and can therefore be identified as exact duplicates.
    """

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:

        while True:

            data = file.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


# ============================================================
# CALCULATE IMAGE QUALITY METRICS
# ============================================================

def analyse_image(image_path, known_hashes):
    """
    Analyse one image and return a dictionary containing
    quality measurements and warning flags.
    """

    # --------------------------------------------------------
    # DEFAULT RESULT
    # --------------------------------------------------------

    result = {
        "image_filename": image_path.name,
        "read_success": False,
        "width": "",
        "height": "",
        "resolution_ok": False,
        "blur_score": "",
        "is_blurry": False,
        "mean_brightness": "",
        "is_too_dark": False,
        "is_too_bright": False,
        "contrast_score": "",
        "is_low_contrast": False,
        "sha256": "",
        "is_exact_duplicate": False,
        "duplicate_of": "",
        "warning_count": 0,
        "review_required": False,
    }


    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image = cv2.imread(str(image_path))


    if image is None:

        result["warning_count"] = 1

        result["review_required"] = True

        return result


    result["read_success"] = True


    # --------------------------------------------------------
    # IMAGE DIMENSIONS
    # --------------------------------------------------------

    height, width = image.shape[:2]

    result["width"] = width

    result["height"] = height


    resolution_ok = (
        width == EXPECTED_WIDTH
        and
        height == EXPECTED_HEIGHT
    )


    result["resolution_ok"] = resolution_ok


    # --------------------------------------------------------
    # CONVERT TO GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )


    # --------------------------------------------------------
    # BLUR / SHARPNESS SCORE
    # --------------------------------------------------------
    #
    # Variance of the Laplacian:
    #
    # Higher value:
    #     more edge detail / generally sharper image.
    #
    # Lower value:
    #     less edge detail / potentially blurry image.
    #
    # Important:
    #     This score depends on the scene and should not be
    #     treated as a universal blur measurement.
    # --------------------------------------------------------

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    ).var()


    is_blurry = (
        blur_score < BLUR_THRESHOLD
    )


    result["blur_score"] = round(
        float(blur_score),
        3,
    )

    result["is_blurry"] = is_blurry


    # --------------------------------------------------------
    # MEAN BRIGHTNESS
    # --------------------------------------------------------
    #
    # Grayscale range:
    #
    # 0   = black
    # 255 = white
    #
    # Mean brightness gives a simple global measurement of
    # the image's average intensity.
    # --------------------------------------------------------

    mean_brightness = float(
        np.mean(gray)
    )


    is_too_dark = (
        mean_brightness < DARK_THRESHOLD
    )


    is_too_bright = (
        mean_brightness > BRIGHT_THRESHOLD
    )


    result["mean_brightness"] = round(
        mean_brightness,
        3,
    )

    result["is_too_dark"] = is_too_dark

    result["is_too_bright"] = is_too_bright


    # --------------------------------------------------------
    # CONTRAST SCORE
    # --------------------------------------------------------
    #
    # Standard deviation of grayscale intensities.
    #
    # Low value:
    #     pixel intensities are relatively similar.
    #
    # Higher value:
    #     stronger intensity variation.
    # --------------------------------------------------------

    contrast_score = float(
        np.std(gray)
    )


    is_low_contrast = (
        contrast_score
        < LOW_CONTRAST_THRESHOLD
    )


    result["contrast_score"] = round(
        contrast_score,
        3,
    )

    result["is_low_contrast"] = (
        is_low_contrast
    )


    # --------------------------------------------------------
    # EXACT DUPLICATE CHECK
    # --------------------------------------------------------

    file_hash = calculate_sha256(
        image_path
    )


    result["sha256"] = file_hash


    if file_hash in known_hashes:

        result["is_exact_duplicate"] = True

        result["duplicate_of"] = (
            known_hashes[file_hash]
        )

    else:

        known_hashes[file_hash] = (
            image_path.name
        )


    # --------------------------------------------------------
    # WARNING COUNT
    # --------------------------------------------------------

    warning_flags = [
        not resolution_ok,
        is_blurry,
        is_too_dark,
        is_too_bright,
        is_low_contrast,
        result["is_exact_duplicate"],
    ]


    warning_count = sum(
        bool(flag)
        for flag in warning_flags
    )


    result["warning_count"] = (
        warning_count
    )


    result["review_required"] = (
        warning_count > 0
    )


    return result


# ============================================================
# SAVE COMPLETE REPORT
# ============================================================

def save_quality_report(results):
    """
    Overwrite the previous report with the current complete
    dataset-quality analysis.
    """

    with QUALITY_REPORT_FILE.open(
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
# PRINT DATASET SUMMARY
# ============================================================

def print_summary(results):

    total_images = len(results)

    readable_images = sum(
        row["read_success"]
        for row in results
    )


    corrupted_images = (
        total_images
        - readable_images
    )


    resolution_problems = sum(

        row["read_success"]
        and
        not row["resolution_ok"]

        for row in results
    )


    blurry_images = sum(
        row["is_blurry"]
        for row in results
    )


    dark_images = sum(
        row["is_too_dark"]
        for row in results
    )


    bright_images = sum(
        row["is_too_bright"]
        for row in results
    )


    low_contrast_images = sum(
        row["is_low_contrast"]
        for row in results
    )


    exact_duplicates = sum(
        row["is_exact_duplicate"]
        for row in results
    )


    review_required = sum(
        row["review_required"]
        for row in results
    )


    clean_images = (
        total_images
        - review_required
    )


    print()

    print("=" * 70)

    print(
        "DATASET QUALITY REPORT"
    )

    print("=" * 70)


    print(
        f"Images found              : "
        f"{total_images}"
    )


    print(
        f"Images read successfully  : "
        f"{readable_images}"
    )


    print(
        f"Corrupted / unreadable    : "
        f"{corrupted_images}"
    )


    print(
        f"Resolution problems       : "
        f"{resolution_problems}"
    )


    print(
        f"Blurry image warnings     : "
        f"{blurry_images}"
    )


    print(
        f"Dark image warnings       : "
        f"{dark_images}"
    )


    print(
        f"Bright image warnings     : "
        f"{bright_images}"
    )


    print(
        f"Low contrast warnings     : "
        f"{low_contrast_images}"
    )


    print(
        f"Exact duplicate files     : "
        f"{exact_duplicates}"
    )


    print(
        f"Images requiring review   : "
        f"{review_required}"
    )


    print(
        f"Images with no warnings   : "
        f"{clean_images}"
    )


    print("-" * 70)


    if total_images == 0:

        print(
            "STATUS: NO IMAGES AVAILABLE"
        )


    elif corrupted_images > 0:

        print(
            "STATUS: DATASET REQUIRES ATTENTION"
        )


    elif review_required == 0:

        print(
            "STATUS: NO WARNINGS FOUND "
            "USING CURRENT THRESHOLDS"
        )


    else:

        print(
            "STATUS: REVIEW FLAGGED IMAGES "
            "BEFORE ANNOTATION"
        )


    print("=" * 70)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 70)

    print(
        "PROTOTYPE V2 - DATASET IMAGE QUALITY ANALYSIS"
    )

    print("=" * 70)


    print(
        f"Image folder : "
        f"{IMAGE_DIR}"
    )


    print(
        f"Report file  : "
        f"{QUALITY_REPORT_FILE}"
    )


    print("-" * 70)


    # --------------------------------------------------------
    # CHECK IMAGE DIRECTORY
    # --------------------------------------------------------

    if not IMAGE_DIR.exists():

        print(
            "ERROR: Extracted image folder does not exist."
        )

        print(
            "Run extract_frames.py first."
        )

        return


    # --------------------------------------------------------
    # FIND SUPPORTED IMAGES
    # --------------------------------------------------------

    image_files = sorted(

        list(IMAGE_DIR.glob("*.jpg"))
        +
        list(IMAGE_DIR.glob("*.jpeg"))
        +
        list(IMAGE_DIR.glob("*.png"))

    )


    if not image_files:

        print(
            "ERROR: No supported images found."
        )

        return


    print(
        f"Images found: "
        f"{len(image_files)}"
    )


    print()

    print(
        "Analysing images..."
    )


    # --------------------------------------------------------
    # ANALYSE DATASET
    # --------------------------------------------------------

    known_hashes = {}

    results = []


    for index, image_path in enumerate(
        image_files,
        start=1,
    ):

        result = analyse_image(
            image_path,
            known_hashes,
        )


        results.append(result)


        if (
            index % 100 == 0
            or
            index == len(image_files)
        ):

            print(
                f"Processed "
                f"{index}/{len(image_files)} images"
            )


    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    save_quality_report(results)


    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print_summary(results)


    print()

    print(
        f"Detailed report saved to:"
    )

    print(
        QUALITY_REPORT_FILE
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()