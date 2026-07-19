"""
build_review_manifest.py

HOW TO RUN THE FILE:
    python build_review_manifest.py  (TYPE THIS IN TERMINAL)

Prototype V2 - Dataset Review Manifest Builder

Purpose:
    Combine frame traceability information, image-quality
    measurements, and near-duplicate information into one
    central dataset review table.

Inputs:
    1. frame_manifest.csv
    2. image_quality_report.csv
    3. near_duplicate_report.csv

Output:
    dataset_review_manifest.csv

Important:
    - The script DOES NOT delete images.
    - Automatic measurements and flags are kept separate from
      the final human review decision.
    - Existing human_decision and human_notes values are
      preserved when the script is run again.
"""

from pathlib import Path
import csv


# ============================================================
# PROJECT PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parent


FRAME_MANIFEST_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "extracted"
    / "metadata"
    / "frame_manifest.csv"
)


QUALITY_REPORT_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "quality"
    / "image_quality_report.csv"
)


NEAR_DUPLICATE_REPORT_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "quality"
    / "near_duplicate_report.csv"
)


REVIEW_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "review"
)


REVIEW_MANIFEST_FILE = (
    REVIEW_DIR
    / "dataset_review_manifest.csv"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

REVIEW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# OUTPUT COLUMNS
# ============================================================

OUTPUT_COLUMNS = [

    # --------------------------------------------------------
    # TRACEABILITY
    # --------------------------------------------------------

    "image_filename",
    "source_video",
    "trial_id",
    "source_frame_number",
    "source_time_seconds",
    "source_video_fps",
    "frame_width",
    "frame_height",


    # --------------------------------------------------------
    # QUALITY MEASUREMENTS
    # --------------------------------------------------------

    "read_success",
    "resolution_ok",
    "blur_score",
    "is_blurry",
    "mean_brightness",
    "is_too_dark",
    "is_too_bright",
    "contrast_score",
    "is_low_contrast",
    "is_exact_duplicate",
    "duplicate_of",
    "quality_warning_count",


    # --------------------------------------------------------
    # NEAR-DUPLICATE INFORMATION
    # --------------------------------------------------------

    "near_duplicate_pair_count",
    "highest_local_similarity",
    "most_similar_image",
    "near_duplicate_warning",


    # --------------------------------------------------------
    # AUTOMATIC REVIEW INFORMATION
    # --------------------------------------------------------

    "automatic_warning_count",
    "automatic_recommendation",


    # --------------------------------------------------------
    # HUMAN REVIEW INFORMATION
    # --------------------------------------------------------

    "human_decision",
    "human_notes",
]


# ============================================================
# LOAD CSV FILE
# ============================================================

def load_csv_rows(file_path):
    """
    Load a CSV file and return all rows.
    """

    with file_path.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        return list(
            csv.DictReader(csv_file)
        )


# ============================================================
# CONVERT TEXT TO BOOLEAN
# ============================================================

def text_to_bool(value):
    """
    Convert CSV text representation to a Python boolean.
    """

    return str(value).strip().lower() == "true"


# ============================================================
# LOAD EXISTING HUMAN REVIEW INFORMATION
# ============================================================

def load_existing_human_reviews():
    """
    Load previously saved human review decisions and notes.

    Purpose:
        Automatic measurements may be recalculated many times.

        Human decisions must survive those recalculations.

    Returns:
        Dictionary:

        {
            "image_filename": {
                "human_decision": "...",
                "human_notes": "...",
            }
        }
    """

    existing_reviews = {}


    # --------------------------------------------------------
    # FIRST EXECUTION
    # --------------------------------------------------------
    #
    # If the review manifest does not exist yet, there are no
    # previous human decisions to preserve.
    # --------------------------------------------------------

    if not REVIEW_MANIFEST_FILE.exists():

        return existing_reviews


    # --------------------------------------------------------
    # LOAD EXISTING REVIEW MANIFEST
    # --------------------------------------------------------

    rows = load_csv_rows(
        REVIEW_MANIFEST_FILE
    )


    for row in rows:

        image_filename = row.get(
            "image_filename",
            "",
        )


        if not image_filename:

            continue


        human_decision = row.get(
            "human_decision",
            "UNDECIDED",
        )


        human_notes = row.get(
            "human_notes",
            "",
        )


        # ----------------------------------------------------
        # NORMALIZE EMPTY DECISIONS
        # ----------------------------------------------------

        human_decision = (
            human_decision.strip().upper()
        )


        if not human_decision:

            human_decision = "UNDECIDED"


        # ----------------------------------------------------
        # VALIDATE HUMAN DECISION
        # ----------------------------------------------------
        #
        # Only three values are accepted:
        #
        # KEEP
        # REJECT
        # UNDECIDED
        #
        # Invalid values are converted to UNDECIDED.
        # ----------------------------------------------------

        valid_decisions = {
            "KEEP",
            "REJECT",
            "UNDECIDED",
        }


        if human_decision not in valid_decisions:

            print(
                f"WARNING: Invalid human decision "
                f"'{human_decision}' for "
                f"{image_filename}."
            )

            print(
                "Decision changed to UNDECIDED."
            )

            human_decision = "UNDECIDED"


        existing_reviews[
            image_filename
        ] = {

            "human_decision":
                human_decision,

            "human_notes":
                human_notes,
        }


    return existing_reviews


# ============================================================
# LOAD FRAME MANIFEST
# ============================================================

def load_frame_manifest():

    rows = load_csv_rows(
        FRAME_MANIFEST_FILE
    )


    frames = {}


    for row in rows:

        frames[
            row["image_filename"]
        ] = row


    return frames


# ============================================================
# LOAD QUALITY REPORT
# ============================================================

def load_quality_report():

    rows = load_csv_rows(
        QUALITY_REPORT_FILE
    )


    quality_data = {}


    for row in rows:

        quality_data[
            row["image_filename"]
        ] = row


    return quality_data


# ============================================================
# BUILD NEAR-DUPLICATE SUMMARY PER IMAGE
# ============================================================

def load_near_duplicate_summary():

    rows = load_csv_rows(
        NEAR_DUPLICATE_REPORT_FILE
    )


    image_summary = {}


    def ensure_image(image_filename):

        if image_filename not in image_summary:

            image_summary[image_filename] = {

                "near_duplicate_pair_count": 0,

                "highest_local_similarity": 0.0,

                "most_similar_image": "",

                "near_duplicate_warning": False,
            }


    for row in rows:

        image_a = row["image_a"]

        image_b = row["image_b"]


        similarity = float(
            row["similarity_score"]
        )


        is_near_duplicate = text_to_bool(
            row["is_near_duplicate"]
        )


        ensure_image(image_a)

        ensure_image(image_b)


        # ----------------------------------------------------
        # UPDATE IMAGE A
        # ----------------------------------------------------

        if (
            similarity
            >
            image_summary[
                image_a
            ][
                "highest_local_similarity"
            ]
        ):

            image_summary[
                image_a
            ][
                "highest_local_similarity"
            ] = similarity


            image_summary[
                image_a
            ][
                "most_similar_image"
            ] = image_b


        # ----------------------------------------------------
        # UPDATE IMAGE B
        # ----------------------------------------------------

        if (
            similarity
            >
            image_summary[
                image_b
            ][
                "highest_local_similarity"
            ]
        ):

            image_summary[
                image_b
            ][
                "highest_local_similarity"
            ] = similarity


            image_summary[
                image_b
            ][
                "most_similar_image"
            ] = image_a


        # ----------------------------------------------------
        # COUNT FLAGGED NEAR-DUPLICATE PAIRS
        # ----------------------------------------------------

        if is_near_duplicate:

            image_summary[
                image_a
            ][
                "near_duplicate_pair_count"
            ] += 1


            image_summary[
                image_b
            ][
                "near_duplicate_pair_count"
            ] += 1


            image_summary[
                image_a
            ][
                "near_duplicate_warning"
            ] = True


            image_summary[
                image_b
            ][
                "near_duplicate_warning"
            ] = True


    return image_summary


# ============================================================
# CHECK REQUIRED INPUT FILES
# ============================================================

def validate_inputs():

    required_files = [

        FRAME_MANIFEST_FILE,

        QUALITY_REPORT_FILE,

        NEAR_DUPLICATE_REPORT_FILE,
    ]


    missing_files = [

        file_path

        for file_path in required_files

        if not file_path.exists()

    ]


    if missing_files:

        print(
            "ERROR: Required input files are missing."
        )


        for file_path in missing_files:

            print(
                f"Missing: {file_path}"
            )


        return False


    return True


# ============================================================
# BUILD REVIEW MANIFEST
# ============================================================

def build_review_manifest():

    # --------------------------------------------------------
    # LOAD CURRENT AUTOMATIC DATA
    # --------------------------------------------------------

    frames = load_frame_manifest()


    quality_data = load_quality_report()


    near_duplicate_data = (
        load_near_duplicate_summary()
    )


    # --------------------------------------------------------
    # LOAD PREVIOUS HUMAN REVIEW INFORMATION
    # --------------------------------------------------------

    existing_human_reviews = (
        load_existing_human_reviews()
    )


    results = []


    # --------------------------------------------------------
    # PROCESS EVERY REGISTERED IMAGE
    # --------------------------------------------------------

    for image_filename, frame in frames.items():


        # ----------------------------------------------------
        # FIND QUALITY INFORMATION
        # ----------------------------------------------------

        quality = quality_data.get(
            image_filename
        )


        if quality is None:

            print(
                f"WARNING: No quality information for "
                f"{image_filename}"
            )

            continue


        # ----------------------------------------------------
        # FIND NEAR-DUPLICATE INFORMATION
        # ----------------------------------------------------

        near_duplicate = (
            near_duplicate_data.get(

                image_filename,

                {
                    "near_duplicate_pair_count": 0,

                    "highest_local_similarity": 0.0,

                    "most_similar_image": "",

                    "near_duplicate_warning": False,
                },
            )
        )


        # ----------------------------------------------------
        # QUALITY WARNING COUNT
        # ----------------------------------------------------

        quality_warning_count = int(
            quality["warning_count"]
        )


        # ----------------------------------------------------
        # NEAR-DUPLICATE WARNING
        # ----------------------------------------------------

        near_duplicate_warning = bool(
            near_duplicate[
                "near_duplicate_warning"
            ]
        )


        # ----------------------------------------------------
        # AUTOMATIC WARNING COUNT
        # ----------------------------------------------------

        automatic_warning_count = (
            quality_warning_count
            +
            int(near_duplicate_warning)
        )


        # ----------------------------------------------------
        # AUTOMATIC RECOMMENDATION
        # ----------------------------------------------------

        if automatic_warning_count > 0:

            automatic_recommendation = (
                "REVIEW"
            )

        else:

            automatic_recommendation = (
                "NO_WARNING"
            )


        # ----------------------------------------------------
        # RESTORE EXISTING HUMAN REVIEW
        # ----------------------------------------------------

        existing_review = (
            existing_human_reviews.get(

                image_filename,

                {
                    "human_decision":
                        "UNDECIDED",

                    "human_notes":
                        "",
                },
            )
        )


        human_decision = (
            existing_review[
                "human_decision"
            ]
        )


        human_notes = (
            existing_review[
                "human_notes"
            ]
        )


        # ----------------------------------------------------
        # CREATE OUTPUT ROW
        # ----------------------------------------------------

        output_row = {

            # =================================================
            # TRACEABILITY
            # =================================================

            "image_filename":
                image_filename,

            "source_video":
                frame["source_video"],

            "trial_id":
                frame["trial_id"],

            "source_frame_number":
                frame["source_frame_number"],

            "source_time_seconds":
                frame["source_time_seconds"],

            "source_video_fps":
                frame["source_video_fps"],

            "frame_width":
                frame["frame_width"],

            "frame_height":
                frame["frame_height"],


            # =================================================
            # QUALITY
            # =================================================

            "read_success":
                quality["read_success"],

            "resolution_ok":
                quality["resolution_ok"],

            "blur_score":
                quality["blur_score"],

            "is_blurry":
                quality["is_blurry"],

            "mean_brightness":
                quality["mean_brightness"],

            "is_too_dark":
                quality["is_too_dark"],

            "is_too_bright":
                quality["is_too_bright"],

            "contrast_score":
                quality["contrast_score"],

            "is_low_contrast":
                quality["is_low_contrast"],

            "is_exact_duplicate":
                quality[
                    "is_exact_duplicate"
                ],

            "duplicate_of":
                quality["duplicate_of"],

            "quality_warning_count":
                quality_warning_count,


            # =================================================
            # NEAR DUPLICATES
            # =================================================

            "near_duplicate_pair_count":
                near_duplicate[
                    "near_duplicate_pair_count"
                ],

            "highest_local_similarity":
                round(
                    float(
                        near_duplicate[
                            "highest_local_similarity"
                        ]
                    ),
                    6,
                ),

            "most_similar_image":
                near_duplicate[
                    "most_similar_image"
                ],

            "near_duplicate_warning":
                near_duplicate_warning,


            # =================================================
            # AUTOMATIC REVIEW
            # =================================================

            "automatic_warning_count":
                automatic_warning_count,

            "automatic_recommendation":
                automatic_recommendation,


            # =================================================
            # HUMAN REVIEW
            # =================================================

            "human_decision":
                human_decision,

            "human_notes":
                human_notes,
        }


        results.append(
            output_row
        )


    return results


# ============================================================
# SAVE REVIEW MANIFEST
# ============================================================

def save_review_manifest(results):

    with REVIEW_MANIFEST_FILE.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=OUTPUT_COLUMNS,
        )


        writer.writeheader()

        writer.writerows(results)


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(results):

    total_images = len(results)


    # --------------------------------------------------------
    # AUTOMATIC REVIEW COUNTS
    # --------------------------------------------------------

    images_for_review = sum(

        row[
            "automatic_recommendation"
        ]
        ==
        "REVIEW"

        for row in results
    )


    images_without_warnings = sum(

        row[
            "automatic_recommendation"
        ]
        ==
        "NO_WARNING"

        for row in results
    )


    near_duplicate_warnings = sum(

        bool(
            row[
                "near_duplicate_warning"
            ]
        )

        for row in results
    )


    # --------------------------------------------------------
    # HUMAN REVIEW COUNTS
    # --------------------------------------------------------

    keep_decisions = sum(

        row[
            "human_decision"
        ].strip().upper()
        ==
        "KEEP"

        for row in results
    )


    reject_decisions = sum(

        row[
            "human_decision"
        ].strip().upper()
        ==
        "REJECT"

        for row in results
    )


    undecided_decisions = sum(

        row[
            "human_decision"
        ].strip().upper()
        ==
        "UNDECIDED"

        for row in results
    )


    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "DATASET REVIEW MANIFEST SUMMARY"
    )

    print("=" * 70)


    print(
        f"Images registered          : "
        f"{total_images}"
    )


    print(
        f"Images flagged for review  : "
        f"{images_for_review}"
    )


    print(
        f"Images without warnings    : "
        f"{images_without_warnings}"
    )


    print(
        f"Near-duplicate warnings    : "
        f"{near_duplicate_warnings}"
    )


    print("-" * 70)


    print(
        f"Human decisions KEEP       : "
        f"{keep_decisions}"
    )


    print(
        f"Human decisions REJECT     : "
        f"{reject_decisions}"
    )


    print(
        f"Human decisions UNDECIDED  : "
        f"{undecided_decisions}"
    )


    print("-" * 70)


    print(
        "Automatic flags are screening information only."
    )


    print(
        "Final KEEP/REJECT decisions require human review."
    )


    print("=" * 70)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 70)

    print(
        "PROTOTYPE V2 - DATASET REVIEW MANIFEST BUILDER"
    )

    print("=" * 70)


    print(
        f"Output file: "
        f"{REVIEW_MANIFEST_FILE}"
    )


    print("-" * 70)


    # --------------------------------------------------------
    # VALIDATE INPUT FILES
    # --------------------------------------------------------

    if not validate_inputs():

        return


    # --------------------------------------------------------
    # BUILD UPDATED REVIEW MANIFEST
    # --------------------------------------------------------

    results = build_review_manifest()


    # --------------------------------------------------------
    # SAVE UPDATED REVIEW MANIFEST
    # --------------------------------------------------------

    save_review_manifest(
        results
    )


    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print_summary(
        results
    )


    print()

    print(
        "Review manifest saved to:"
    )


    print(
        REVIEW_MANIFEST_FILE
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()