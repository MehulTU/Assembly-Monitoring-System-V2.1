"""
===============================================================================
Prototype V2.3 - Interactive Dataset Human Review Tool
===============================================================================

File:
    review_dataset.py

HOW TO RUN THE FILE:
    python review_dataset.py  (TYPE THIS IN TERMINAL)

Author:
    Mehul Patil

Project:
    AI-Supported Ergonomic and Productivity Analysis
    Vision-Based Assembly Monitoring Prototype (V2)

===============================================================================
WHAT IS THIS PROGRAM?
===============================================================================

This program provides a simple visual interface for reviewing images before
they are used for annotation and YOLO training.

The previous dataset-processing stages automatically analyse image quality
and near-duplicate similarity. However, automatic warnings cannot reliably
decide whether an image is useful for the final dataset.

Therefore, this program performs the HUMAN REVIEW stage.

The program:

    1. Reads dataset_review_manifest.csv.
    2. Finds the corresponding image automatically.
    3. Displays the image in an OpenCV window.
    4. Displays useful automatic quality information.
    5. Allows the user to accept or reject the image using keyboard controls.
    6. Saves the human decision automatically into the CSV file.
    7. Moves automatically to the next image.
    8. Preserves completed decisions so the review can be resumed later.

The user does NOT need to:

    • Search for image filenames manually.
    • Open images individually.
    • Edit the CSV file manually.
    • Delete original extracted images.

===============================================================================
REVIEW ORDER
===============================================================================

Images are reviewed in the following order:

    1. UNDECIDED images with automatic recommendation = REVIEW

    2. UNDECIDED images with automatic recommendation = NO_WARNING

    3. Already-decided images can still be visited using the navigation keys.

This allows suspicious images to be inspected first.

===============================================================================
KEYBOARD CONTROLS
===============================================================================

    K
        Mark current image as KEEP.
        Save the CSV.
        Automatically move to the next undecided image.

    R
        Mark current image as REJECT.
        Save the CSV.
        Automatically move to the next undecided image.

    U
        Change the current image back to UNDECIDED.
        Save the CSV.

    N or RIGHT ARROW
        Move to the next image.

    P or LEFT ARROW
        Move to the previous image.

    Q or ESC
        Save the CSV and quit safely.

===============================================================================
IMPORTANT
===============================================================================

This program does NOT delete images.

REJECT only records the human decision in dataset_review_manifest.csv.

A later dataset-building script will create the approved dataset using only
images marked KEEP.

This preserves the original extracted dataset and maintains traceability.

===============================================================================
"""

from pathlib import Path
import shutil

import cv2
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

WINDOW_NAME = "Prototype V2.3 - Dataset Human Review"

MANIFEST_FILENAME = "dataset_review_manifest.csv"

VALID_DECISIONS = {"KEEP", "REJECT", "UNDECIDED"}

REVIEW_FIRST_RECOMMENDATION = "REVIEW"

NO_WARNING_RECOMMENDATION = "NO_WARNING"

# Maximum display size. Large images are scaled down only for viewing.
MAX_DISPLAY_WIDTH = 1500
MAX_DISPLAY_HEIGHT = 850

# Save a backup before changing the manifest.
CREATE_BACKUP = True


# =============================================================================
# PROJECT PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parent

DATASETS_DIR = PROJECT_ROOT / "datasets"

IMAGE_DIR = DATASETS_DIR / "extracted" / "images"

REVIEW_DIR = DATASETS_DIR / "review"

MANIFEST_FILE = REVIEW_DIR / MANIFEST_FILENAME

BACKUP_FILE = REVIEW_DIR / "dataset_review_manifest_backup.csv"


# =============================================================================
# REQUIRED CSV COLUMNS
# =============================================================================

REQUIRED_COLUMNS = [
    "image_filename",
    "trial_id",
    "source_time_seconds",
    "is_blurry",
    "near_duplicate_warning",
    "highest_local_similarity",
    "automatic_recommendation",
    "human_decision",
    "human_notes",
]


# =============================================================================
# SMALL HELPERS
# =============================================================================

def normalize_text(value):
    """
    Convert a CSV value into clean text.

    NaN values become an empty string.
    """

    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_boolean(value):
    """
    Convert common CSV boolean values into TRUE or FALSE text.
    """

    text = normalize_text(value).lower()

    if text in {"true", "1", "yes"}:
        return "TRUE"

    if text in {"false", "0", "no"}:
        return "FALSE"

    return text.upper() if text else "UNKNOWN"


def normalize_decision(value):
    """
    Return KEEP, REJECT, or UNDECIDED.

    Unknown or empty values are treated as UNDECIDED.
    """

    decision = normalize_text(value).upper()

    if decision not in VALID_DECISIONS:
        return "UNDECIDED"

    return decision


def safe_float_text(value, decimal_places=4):
    """
    Format a numeric CSV value safely.
    """

    try:
        return f"{float(value):.{decimal_places}f}"

    except (TypeError, ValueError):
        return "N/A"


def shorten_text(text, maximum_length):
    """
    Shorten long text for the display overlay.
    """

    text = normalize_text(text)

    if len(text) <= maximum_length:
        return text

    return text[: maximum_length - 3] + "..."


# =============================================================================
# VALIDATE PROJECT INPUTS
# =============================================================================

def validate_inputs():
    """
    Check that the manifest, image folder, and required columns exist.
    """

    if not MANIFEST_FILE.exists():
        print()
        print("=" * 75)
        print("ERROR: REVIEW MANIFEST NOT FOUND")
        print("=" * 75)
        print(MANIFEST_FILE)
        print()
        print("Run build_review_manifest.py first.")
        print("=" * 75)

        return False

    if not IMAGE_DIR.exists():
        print()
        print("=" * 75)
        print("ERROR: EXTRACTED IMAGE FOLDER NOT FOUND")
        print("=" * 75)
        print(IMAGE_DIR)
        print("=" * 75)

        return False

    try:
        dataframe = pd.read_csv(MANIFEST_FILE)

    except Exception as error:
        print()
        print("=" * 75)
        print("ERROR: COULD NOT READ REVIEW MANIFEST")
        print("=" * 75)
        print(error)
        print("=" * 75)

        return False

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        print()
        print("=" * 75)
        print("ERROR: REQUIRED CSV COLUMNS ARE MISSING")
        print("=" * 75)

        for column in missing_columns:
            print(f"Missing: {column}")

        print("=" * 75)

        return False

    return True


# =============================================================================
# LOAD MANIFEST
# =============================================================================

def load_manifest():
    """
    Load the review manifest and normalize human decisions.
    """

    dataframe = pd.read_csv(MANIFEST_FILE)

    dataframe["human_decision"] = (
        dataframe["human_decision"]
        .apply(normalize_decision)
    )

    dataframe["human_notes"] = (
        dataframe["human_notes"]
        .fillna("")
        .astype(str)
    )

    return dataframe


# =============================================================================
# CREATE BACKUP
# =============================================================================

def create_backup():
    """
    Create one backup copy before review begins.

    The backup is overwritten each time the review program starts.
    """

    if not CREATE_BACKUP:
        return

    try:
        shutil.copy2(MANIFEST_FILE, BACKUP_FILE)

        print(f"Backup created: {BACKUP_FILE}")

    except Exception as error:
        print(f"WARNING: Could not create backup: {error}")


# =============================================================================
# SAVE MANIFEST
# =============================================================================

def save_manifest(dataframe):
    """
    Save the current human decisions to the review manifest.

    A temporary file is used first to reduce the risk of corrupting the CSV
    if the program is interrupted while writing.
    """

    temporary_file = REVIEW_DIR / "dataset_review_manifest_temp.csv"

    try:
        dataframe.to_csv(
            temporary_file,
            index=False,
            encoding="utf-8",
        )

        temporary_file.replace(MANIFEST_FILE)

    except Exception as error:
        print()
        print("=" * 75)
        print("ERROR: COULD NOT SAVE REVIEW MANIFEST")
        print("=" * 75)
        print(error)
        print("=" * 75)

        raise


# =============================================================================
# BUILD REVIEW ORDER
# =============================================================================

def build_review_order(dataframe):
    """
    Build the image-review order.

    Priority:

        1. Undecided REVIEW images.
        2. Undecided NO_WARNING images.
        3. Other undecided images.
        4. Already-decided images.

    The returned values are original DataFrame row indexes.

    This is important because decisions must be written back to the correct
    CSV rows.
    """

    undecided_review = []

    undecided_no_warning = []

    undecided_other = []

    already_decided = []

    for row_index, row in dataframe.iterrows():

        decision = normalize_decision(row["human_decision"])

        recommendation = normalize_text(
            row["automatic_recommendation"]
        ).upper()

        if decision == "UNDECIDED":

            if recommendation == REVIEW_FIRST_RECOMMENDATION:
                undecided_review.append(row_index)

            elif recommendation == NO_WARNING_RECOMMENDATION:
                undecided_no_warning.append(row_index)

            else:
                undecided_other.append(row_index)

        else:
            already_decided.append(row_index)

    return (
        undecided_review
        + undecided_no_warning
        + undecided_other
        + already_decided
    )


# =============================================================================
# FIND FIRST IMAGE TO REVIEW
# =============================================================================

def find_start_position(dataframe, review_order):
    """
    Start at the first UNDECIDED image.

    If every image has already been reviewed, start at the first image.
    """

    for position, row_index in enumerate(review_order):

        decision = normalize_decision(
            dataframe.at[row_index, "human_decision"]
        )

        if decision == "UNDECIDED":
            return position

    return 0


# =============================================================================
# FIND NEXT UNDECIDED IMAGE
# =============================================================================

def find_next_undecided_position(
    dataframe,
    review_order,
    current_position,
):
    """
    Search forward for the next UNDECIDED image.

    Wrap around to the beginning if necessary.

    Return None if all images have been decided.
    """

    total_images = len(review_order)

    for offset in range(1, total_images + 1):

        position = (current_position + offset) % total_images

        row_index = review_order[position]

        decision = normalize_decision(
            dataframe.at[row_index, "human_decision"]
        )

        if decision == "UNDECIDED":
            return position

    return None


# =============================================================================
# REVIEW STATISTICS
# =============================================================================

def calculate_statistics(dataframe):
    """
    Count KEEP, REJECT, and UNDECIDED decisions.
    """

    decisions = dataframe["human_decision"].apply(normalize_decision)

    keep_count = int((decisions == "KEEP").sum())

    reject_count = int((decisions == "REJECT").sum())

    undecided_count = int((decisions == "UNDECIDED").sum())

    return keep_count, reject_count, undecided_count


# =============================================================================
# RESIZE IMAGE FOR DISPLAY
# =============================================================================

def resize_for_display(image):
    """
    Scale the image down if it is larger than the configured display area.

    The original image file is NOT modified.
    """

    height, width = image.shape[:2]

    width_scale = MAX_DISPLAY_WIDTH / width

    height_scale = MAX_DISPLAY_HEIGHT / height

    scale = min(width_scale, height_scale, 1.0)

    if scale == 1.0:
        return image.copy()

    new_width = int(width * scale)

    new_height = int(height * scale)

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


# =============================================================================
# DRAW TEXT WITH BACKGROUND
# =============================================================================

def draw_text_line(
    image,
    text,
    position,
    font_scale=0.55,
    thickness=1,
):
    """
    Draw readable white text over a dark background rectangle.
    """

    x, y = position

    font = cv2.FONT_HERSHEY_SIMPLEX

    text_size, baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    text_width, text_height = text_size

    cv2.rectangle(
        image,
        (x - 5, y - text_height - 7),
        (x + text_width + 5, y + baseline + 5),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        image,
        text,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


# =============================================================================
# CREATE REVIEW DISPLAY
# =============================================================================

def create_review_display(
    image,
    row,
    position,
    total_images,
    keep_count,
    reject_count,
    undecided_count,
):
    """
    Create the visual review screen.

    The source image is resized only for display.

    Dataset files are never modified.
    """

    display_image = resize_for_display(image)

    image_filename = shorten_text(
        row["image_filename"],
        70,
    )

    trial_id = shorten_text(
        row["trial_id"],
        50,
    )

    source_time = safe_float_text(
        row["source_time_seconds"],
        2,
    )

    recommendation = normalize_text(
        row["automatic_recommendation"]
    ).upper()

    decision = normalize_decision(
        row["human_decision"]
    )

    blurry = normalize_boolean(
        row["is_blurry"]
    )

    near_duplicate = normalize_boolean(
        row["near_duplicate_warning"]
    )

    similarity = safe_float_text(
        row["highest_local_similarity"],
        4,
    )

    lines = [
        f"Image {position + 1} / {total_images}",
        f"File: {image_filename}",
        f"Trial: {trial_id} | Source time: {source_time} s",
        (
            f"Automatic: {recommendation} | "
            f"Blurry: {blurry} | "
            f"Near duplicate: {near_duplicate} | "
            f"Similarity: {similarity}"
        ),
        (
            f"Human decision: {decision} | "
            f"KEEP: {keep_count} | "
            f"REJECT: {reject_count} | "
            f"UNDECIDED: {undecided_count}"
        ),
        "K = KEEP | R = REJECT | U = UNDECIDED | N/P = Navigate | Q = Save and Quit",
    ]

    y = 30

    for line in lines:
        draw_text_line(
            display_image,
            line,
            (15, y),
            font_scale=0.50,
            thickness=1,
        )

        y += 30

    return display_image


# =============================================================================
# PRINT REVIEW SUMMARY
# =============================================================================

def print_review_summary(dataframe):
    """
    Print current review progress.
    """

    keep_count, reject_count, undecided_count = calculate_statistics(dataframe)

    total_images = len(dataframe)

    reviewed_count = keep_count + reject_count

    print()
    print("=" * 75)
    print("DATASET HUMAN REVIEW SUMMARY")
    print("=" * 75)
    print(f"Total images : {total_images}")
    print(f"Reviewed     : {reviewed_count}")
    print(f"KEEP         : {keep_count}")
    print(f"REJECT       : {reject_count}")
    print(f"UNDECIDED    : {undecided_count}")
    print("=" * 75)


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main():

    print()
    print("=" * 75)
    print("PROTOTYPE V2.3 - INTERACTIVE DATASET HUMAN REVIEW")
    print("=" * 75)
    print(f"Image folder : {IMAGE_DIR}")
    print(f"Review file  : {MANIFEST_FILE}")
    print("-" * 75)

    if not validate_inputs():
        return

    dataframe = load_manifest()

    if dataframe.empty:
        print("ERROR: Review manifest contains no images.")
        return

    create_backup()

    review_order = build_review_order(dataframe)

    if not review_order:
        print("ERROR: No images are available for review.")
        return

    current_position = find_start_position(
        dataframe,
        review_order,
    )

    print_review_summary(dataframe)

    print()
    print("CONTROLS")
    print("-" * 75)
    print("K = KEEP")
    print("R = REJECT")
    print("U = UNDECIDED")
    print("N / RIGHT ARROW = Next image")
    print("P / LEFT ARROW  = Previous image")
    print("Q / ESC         = Save and quit")
    print("-" * 75)

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    try:

        while True:

            row_index = review_order[current_position]

            row = dataframe.loc[row_index]

            image_filename = normalize_text(
                row["image_filename"]
            )

            image_path = IMAGE_DIR / image_filename

            image = cv2.imread(str(image_path))

            if image is None:

                print()
                print("WARNING: Could not load image:")
                print(image_path)

                next_position = find_next_undecided_position(
                    dataframe,
                    review_order,
                    current_position,
                )

                if next_position is None:
                    current_position = (
                        current_position + 1
                    ) % len(review_order)

                else:
                    current_position = next_position

                continue

            keep_count, reject_count, undecided_count = (
                calculate_statistics(dataframe)
            )

            display_image = create_review_display(
                image=image,
                row=row,
                position=current_position,
                total_images=len(review_order),
                keep_count=keep_count,
                reject_count=reject_count,
                undecided_count=undecided_count,
            )

            cv2.imshow(
                WINDOW_NAME,
                display_image,
            )

            key = cv2.waitKeyEx(0)

            # ---------------------------------------------------------
            # KEEP
            # ---------------------------------------------------------

            if key in (ord("k"), ord("K")):

                dataframe.at[
                    row_index,
                    "human_decision",
                ] = "KEEP"

                save_manifest(dataframe)

                print(
                    f"KEEP   | "
                    f"{image_filename}"
                )

                next_position = find_next_undecided_position(
                    dataframe,
                    review_order,
                    current_position,
                )

                if next_position is None:

                    print()
                    print("=" * 75)
                    print("ALL IMAGES HAVE BEEN REVIEWED")
                    print("=" * 75)

                    break

                current_position = next_position

            # ---------------------------------------------------------
            # REJECT
            # ---------------------------------------------------------

            elif key in (ord("r"), ord("R")):

                dataframe.at[
                    row_index,
                    "human_decision",
                ] = "REJECT"

                save_manifest(dataframe)

                print(
                    f"REJECT | "
                    f"{image_filename}"
                )

                next_position = find_next_undecided_position(
                    dataframe,
                    review_order,
                    current_position,
                )

                if next_position is None:

                    print()
                    print("=" * 75)
                    print("ALL IMAGES HAVE BEEN REVIEWED")
                    print("=" * 75)

                    break

                current_position = next_position

            # ---------------------------------------------------------
            # UNDECIDED
            # ---------------------------------------------------------

            elif key in (ord("u"), ord("U")):

                dataframe.at[
                    row_index,
                    "human_decision",
                ] = "UNDECIDED"

                save_manifest(dataframe)

                print(
                    f"UNDECIDED | "
                    f"{image_filename}"
                )

            # ---------------------------------------------------------
            # NEXT IMAGE
            #
            # Windows OpenCV arrow-key codes can vary between builds,
            # therefore N is the reliable navigation control.
            # ---------------------------------------------------------

            elif key in (
                ord("n"),
                ord("N"),
                2555904,
                65363,
            ):

                current_position = (
                    current_position + 1
                ) % len(review_order)

            # ---------------------------------------------------------
            # PREVIOUS IMAGE
            # ---------------------------------------------------------

            elif key in (
                ord("p"),
                ord("P"),
                2424832,
                65361,
            ):

                current_position = (
                    current_position - 1
                ) % len(review_order)

            # ---------------------------------------------------------
            # QUIT
            # ---------------------------------------------------------

            elif key in (
                ord("q"),
                ord("Q"),
                27,
            ):

                print("Quit requested.")

                break

    finally:

        save_manifest(dataframe)

        cv2.destroyAllWindows()

        print_review_summary(dataframe)

        print()
        print("Review progress saved successfully.")
        print(f"Review file: {MANIFEST_FILE}")


# =============================================================================
# PROGRAM ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()