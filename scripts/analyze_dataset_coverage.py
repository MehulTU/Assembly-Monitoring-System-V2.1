"""
===============================================================================
Prototype V2 - Dataset Statistics and Experiment Coverage Analysis
===============================================================================

File:
    analyze_dataset_coverage.py

HOW TO RUN THE FILE: 
    python scripts\analyze_dataset_coverage.py  (TYPE THIS IN TERMINAL)

Project:
    Vision-Based Assembly Monitoring System - Prototype V2

WHAT IS THIS PROGRAM?

The dataset pipeline has already:

    1. Recorded controlled experiment trials.
    2. Stored trial information in trials.csv.
    3. Extracted images from the recorded videos.
    4. Analysed image quality.
    5. Detected near-duplicate images.
    6. Created a combined review manifest.
    7. Allowed human review of every extracted image.
    8. Created a curated dataset containing only KEEP images.

Before starting YOLO annotation, we need to understand whether the curated
dataset represents the controlled experiment conditions properly.

This program performs that DATASET COVERAGE ANALYSIS.


WHAT DOES THIS PROGRAM DO?

The program reads:

    datasets/raw/trials.csv

and:

    datasets/review/dataset_review_manifest.csv

and:

    datasets/curated/curated_manifest.csv


The program then combines information using:

    trial_id


It calculates:

    - Number of recorded trials.
    - Number of extracted images.
    - Number of KEEP images.
    - Number of REJECT images.
    - Retention rate for every trial.
    - Dataset contribution of every trial.
    - Number of remaining blurry KEEP images.
    - Number of remaining near-duplicate KEEP images.
    - Number of images containing no object.
    - Number of images containing marker only.
    - Number of images containing power adapter only.
    - Number of images containing both objects.
    - Coverage of static, occlusion, interaction, and movement conditions.
    - Largest and smallest trial contributions.
    - Missing trials.
    - Unknown trial IDs.
    - Basic warnings about dataset imbalance.

The program creates:

    datasets/curated/dataset_coverage_report.csv


IMPORTANT

This program does NOT:

    - Delete images.
    - Copy images.
    - Change KEEP / REJECT decisions.
    - Modify trials.csv.
    - Modify curated_manifest.csv.
    - Train YOLO.

It only analyses the current dataset and creates a report.
===============================================================================
"""


# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path

import pandas as pd


# =============================================================================
# PROJECT PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parent


TRIALS_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    /"metadata"
    / "trials.csv"
)


REVIEW_MANIFEST = (
    PROJECT_ROOT
    / "datasets"
    / "review"
    / "dataset_review_manifest.csv"
)


CURATED_MANIFEST = (
    PROJECT_ROOT
    / "datasets"
    / "curated"
    / "curated_manifest.csv"
)


COVERAGE_REPORT = (
    PROJECT_ROOT
    / "datasets"
    / "curated"
    / "dataset_coverage_report.csv"
)


# =============================================================================
# REQUIRED COLUMNS
# =============================================================================

TRIAL_REQUIRED_COLUMNS = {
    "trial_id",
    "trial_number",
    "trial_condition",
    "objects_present",
    "frame_count",
    "duration_seconds",
}


REVIEW_REQUIRED_COLUMNS = {
    "image_filename",
    "trial_id",
    "human_decision",
    "is_blurry",
    "near_duplicate_warning",
}


CURATED_REQUIRED_COLUMNS = {
    "image_filename",
    "trial_id",
    "human_decision",
    "is_blurry",
    "near_duplicate_warning",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_text(value):
    """
    Convert a CSV value to clean lowercase text.

    Missing values become an empty string.
    """

    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def normalize_upper_text(value):
    """
    Convert a CSV value to clean uppercase text.

    Missing values become an empty string.
    """

    if pd.isna(value):
        return ""

    return str(value).strip().upper()


def normalize_boolean_series(series):
    """
    Convert common CSV boolean values into Python True / False values.

    Examples interpreted as True:

        True
        TRUE
        1
        yes
        Y

    All other values become False.
    """

    return (
        series
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def safe_percentage(numerator, denominator):
    """
    Calculate a percentage safely.

    Return 0.0 when denominator is zero.
    """

    if denominator == 0:
        return 0.0

    return round((numerator / denominator) * 100.0, 2)


def check_required_columns(dataframe, required_columns, file_label):
    """
    Verify that required columns exist.
    """

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:

        raise ValueError(
            f"\n{file_label} is missing required columns:\n"
            f"{sorted(missing_columns)}"
        )


# =============================================================================
# LOAD INPUT FILES
# =============================================================================

def load_input_files():

    print("Loading input files...")

    # -------------------------------------------------------------------------
    # CHECK FILES
    # -------------------------------------------------------------------------

    for file_path, file_label in [

        (TRIALS_FILE, "trials.csv"),

        (REVIEW_MANIFEST, "dataset_review_manifest.csv"),

        (CURATED_MANIFEST, "curated_manifest.csv"),

    ]:

        if not file_path.exists():

            raise FileNotFoundError(
                f"\nRequired file not found:\n"
                f"{file_path}\n\n"
                f"Missing input: {file_label}"
            )


    # -------------------------------------------------------------------------
    # LOAD CSV FILES
    # -------------------------------------------------------------------------

    trials_dataframe = pd.read_csv(TRIALS_FILE)

    review_dataframe = pd.read_csv(REVIEW_MANIFEST)

    curated_dataframe = pd.read_csv(CURATED_MANIFEST)


    # -------------------------------------------------------------------------
    # CHECK COLUMNS
    # -------------------------------------------------------------------------

    check_required_columns(
        trials_dataframe,
        TRIAL_REQUIRED_COLUMNS,
        "trials.csv",
    )


    check_required_columns(
        review_dataframe,
        REVIEW_REQUIRED_COLUMNS,
        "dataset_review_manifest.csv",
    )


    check_required_columns(
        curated_dataframe,
        CURATED_REQUIRED_COLUMNS,
        "curated_manifest.csv",
    )


    # -------------------------------------------------------------------------
    # NORMALIZE IMPORTANT VALUES
    # -------------------------------------------------------------------------

    review_dataframe["human_decision"] = (
        review_dataframe["human_decision"]
        .apply(normalize_upper_text)
    )


    curated_dataframe["human_decision"] = (
        curated_dataframe["human_decision"]
        .apply(normalize_upper_text)
    )


    review_dataframe["is_blurry_bool"] = normalize_boolean_series(
        review_dataframe["is_blurry"]
    )


    review_dataframe["near_duplicate_bool"] = normalize_boolean_series(
        review_dataframe["near_duplicate_warning"]
    )


    curated_dataframe["is_blurry_bool"] = normalize_boolean_series(
        curated_dataframe["is_blurry"]
    )


    curated_dataframe["near_duplicate_bool"] = normalize_boolean_series(
        curated_dataframe["near_duplicate_warning"]
    )


    print(f"Recorded trials loaded : {len(trials_dataframe)}")

    print(f"Review rows loaded     : {len(review_dataframe)}")

    print(f"Curated rows loaded    : {len(curated_dataframe)}")


    return (
        trials_dataframe,
        review_dataframe,
        curated_dataframe,
    )


# =============================================================================
# VALIDATE DATASET RELATIONSHIPS
# =============================================================================

def validate_dataset_relationships(
    trials_dataframe,
    review_dataframe,
    curated_dataframe,
):

    print()
    print("=" * 75)
    print("DATASET RELATIONSHIP CHECK")
    print("=" * 75)


    # -------------------------------------------------------------------------
    # DUPLICATE TRIAL IDS
    # -------------------------------------------------------------------------

    duplicate_trial_ids = (

        trials_dataframe[
            trials_dataframe["trial_id"].duplicated(keep=False)
        ]

        ["trial_id"]

        .dropna()

        .unique()

        .tolist()
    )


    # -------------------------------------------------------------------------
    # DUPLICATE IMAGE NAMES
    # -------------------------------------------------------------------------

    duplicate_review_images = int(
        review_dataframe["image_filename"].duplicated().sum()
    )


    duplicate_curated_images = int(
        curated_dataframe["image_filename"].duplicated().sum()
    )


    # -------------------------------------------------------------------------
    # UNKNOWN TRIAL IDS
    # -------------------------------------------------------------------------

    known_trial_ids = set(
        trials_dataframe["trial_id"].dropna()
    )


    review_trial_ids = set(
        review_dataframe["trial_id"].dropna()
    )


    curated_trial_ids = set(
        curated_dataframe["trial_id"].dropna()
    )


    unknown_review_trial_ids = sorted(
        review_trial_ids - known_trial_ids
    )


    unknown_curated_trial_ids = sorted(
        curated_trial_ids - known_trial_ids
    )


    # -------------------------------------------------------------------------
    # MISSING TRIAL COVERAGE
    # -------------------------------------------------------------------------

    trials_without_extracted_images = sorted(
        known_trial_ids - review_trial_ids
    )


    trials_without_curated_images = sorted(
        known_trial_ids - curated_trial_ids
    )


    print(f"Duplicate trial IDs in trials.csv       : {len(duplicate_trial_ids)}")

    print(f"Duplicate images in review manifest      : {duplicate_review_images}")

    print(f"Duplicate images in curated manifest     : {duplicate_curated_images}")

    print(f"Unknown trial IDs in review manifest      : {len(unknown_review_trial_ids)}")

    print(f"Unknown trial IDs in curated manifest     : {len(unknown_curated_trial_ids)}")

    print(f"Trials without extracted images           : {len(trials_without_extracted_images)}")

    print(f"Trials without curated images             : {len(trials_without_curated_images)}")

    print("=" * 75)


    if duplicate_trial_ids:

        print()
        print("WARNING: Duplicate trial IDs were found in trials.csv:")

        for trial_id in duplicate_trial_ids:
            print(f"  - {trial_id}")


    if unknown_review_trial_ids:

        print()
        print("WARNING: Review manifest contains unknown trial IDs:")

        for trial_id in unknown_review_trial_ids:
            print(f"  - {trial_id}")


    if unknown_curated_trial_ids:

        print()
        print("WARNING: Curated manifest contains unknown trial IDs:")

        for trial_id in unknown_curated_trial_ids:
            print(f"  - {trial_id}")


    if trials_without_extracted_images:

        print()
        print("WARNING: Some recorded trials have no extracted images:")

        for trial_id in trials_without_extracted_images:
            print(f"  - {trial_id}")


    if trials_without_curated_images:

        print()
        print("WARNING: Some recorded trials have no KEEP images:")

        for trial_id in trials_without_curated_images:
            print(f"  - {trial_id}")


# =============================================================================
# BUILD PER-TRIAL COVERAGE REPORT
# =============================================================================

def build_trial_coverage_report(
    trials_dataframe,
    review_dataframe,
    curated_dataframe,
):

    report_rows = []


    total_curated_images = len(curated_dataframe)


    for _, trial_row in trials_dataframe.iterrows():


        trial_id = trial_row["trial_id"]


        # ---------------------------------------------------------------------
        # SELECT REVIEW IMAGES FROM THIS TRIAL
        # ---------------------------------------------------------------------

        trial_review = review_dataframe[
            review_dataframe["trial_id"] == trial_id
        ]


        # ---------------------------------------------------------------------
        # SELECT CURATED IMAGES FROM THIS TRIAL
        # ---------------------------------------------------------------------

        trial_curated = curated_dataframe[
            curated_dataframe["trial_id"] == trial_id
        ]


        # ---------------------------------------------------------------------
        # COUNTS
        # ---------------------------------------------------------------------

        extracted_images = len(trial_review)


        kept_images = len(trial_curated)


        rejected_images = int(
            (
                trial_review["human_decision"]
                == "REJECT"
            ).sum()
        )


        undecided_images = int(
            (
                trial_review["human_decision"]
                == "UNDECIDED"
            ).sum()
        )


        kept_blurry_images = int(
            trial_curated["is_blurry_bool"].sum()
        )


        kept_near_duplicate_images = int(
            trial_curated["near_duplicate_bool"].sum()
        )


        # ---------------------------------------------------------------------
        # PERCENTAGES
        # ---------------------------------------------------------------------

        retention_rate_percent = safe_percentage(
            kept_images,
            extracted_images,
        )


        dataset_share_percent = safe_percentage(
            kept_images,
            total_curated_images,
        )


        # ---------------------------------------------------------------------
        # REPORT ROW
        # ---------------------------------------------------------------------

        report_rows.append({

            "experiment_id":
                trial_row.get("experiment_id", ""),

            "trial_number":
                trial_row["trial_number"],

            "trial_id":
                trial_id,

            "trial_condition":
                trial_row["trial_condition"],

            "objects_present":
                trial_row["objects_present"],

            "duration_seconds":
                trial_row["duration_seconds"],

            "recorded_frames":
                trial_row["frame_count"],

            "extracted_images":
                extracted_images,

            "kept_images":
                kept_images,

            "rejected_images":
                rejected_images,

            "undecided_images":
                undecided_images,

            "retention_rate_percent":
                retention_rate_percent,

            "kept_blurry_images":
                kept_blurry_images,

            "kept_near_duplicate_images":
                kept_near_duplicate_images,

            "dataset_share_percent":
                dataset_share_percent,

        })


    coverage_dataframe = pd.DataFrame(report_rows)


    if not coverage_dataframe.empty:

        coverage_dataframe = coverage_dataframe.sort_values(
            by=["trial_number", "trial_id"]
        ).reset_index(drop=True)


    return coverage_dataframe


# =============================================================================
# CLASSIFY OBJECT CONTENT
# =============================================================================

def classify_object_content(objects_present):

    """
    Convert the objects_present metadata into one of four simple groups:

        EMPTY
        MARKER_ONLY
        POWER_ADAPTER_ONLY
        BOTH_OBJECTS

    Unknown values become:

        UNKNOWN
    """


    text = normalize_text(objects_present)


    # -------------------------------------------------------------------------
    # EMPTY WORKSPACE
    # -------------------------------------------------------------------------

    if text in {
        "",
        "none",
        "empty",
        "no_object",
        "no objects",
    }:

        return "EMPTY"


    # -------------------------------------------------------------------------
    # SPLIT OBJECT NAMES
    # -------------------------------------------------------------------------

    objects = {

        item.strip()

        for item in text.split(",")

        if item.strip()

    }


    has_marker = "marker" in objects

    has_power_adapter = "power_adapter" in objects


    if has_marker and has_power_adapter:
        return "BOTH_OBJECTS"


    if has_marker:
        return "MARKER_ONLY"


    if has_power_adapter:
        return "POWER_ADAPTER_ONLY"


    return "UNKNOWN"


# =============================================================================
# CLASSIFY CONDITION TYPE
# =============================================================================

def classify_condition_type(trial_condition):

    """
    Group trial conditions into broader experiment categories.

    The grouping is based on the 15-condition pilot experiment.

    EMPTY:
        empty_workspace

    STATIC:
        marker_center
        marker_left
        marker_right
        marker_rotated
        power_adapter_center
        power_adapter_left
        power_adapter_right
        power_adapter_rotated
        marker_and_power_adapter_separated
        marker_and_power_adapter_close

    OCCLUSION:
        marker_partial_occlusion
        power_adapter_partial_occlusion

    INTERACTION:
        hand_interaction

    MOVEMENT:
        continuous_movement

    Unknown conditions become OTHER.
    """


    condition = normalize_text(trial_condition)


    if condition == "empty_workspace":
        return "EMPTY"


    if condition in {

        "marker_center",
        "marker_left",
        "marker_right",
        "marker_rotated",

        "power_adapter_center",
        "power_adapter_left",
        "power_adapter_right",
        "power_adapter_rotated",

        "marker_and_power_adapter_separated",
        "marker_and_power_adapter_close",

    }:

        return "STATIC"


    if condition in {

        "marker_partial_occlusion",
        "power_adapter_partial_occlusion",

    }:

        return "OCCLUSION"


    if condition == "hand_interaction":
        return "INTERACTION"


    if condition == "continuous_movement":
        return "MOVEMENT"


    return "OTHER"


# =============================================================================
# ADD COVERAGE GROUPS TO CURATED DATA
# =============================================================================

def add_coverage_groups(
    trials_dataframe,
    curated_dataframe,
):

    """
    Join trial metadata onto the curated dataset.

    This lets us count curated images by:

        objects_present

    and:

        trial_condition
    """


    trial_metadata = (

        trials_dataframe[

            [
                "trial_id",
                "trial_number",
                "trial_condition",
                "objects_present",
            ]

        ]

        .copy()

    )


    enriched_curated = curated_dataframe.merge(

        trial_metadata,

        on="trial_id",

        how="left",

        validate="many_to_one",

    )


    enriched_curated["object_content_group"] = (

        enriched_curated["objects_present"]

        .apply(classify_object_content)

    )


    enriched_curated["condition_type_group"] = (

        enriched_curated["trial_condition"]

        .apply(classify_condition_type)

    )


    return enriched_curated


# =============================================================================
# PRINT OVERALL DATASET SUMMARY
# =============================================================================

def print_overall_summary(
    trials_dataframe,
    review_dataframe,
    curated_dataframe,
    coverage_dataframe,
    enriched_curated,
):

    total_trials = len(trials_dataframe)

    total_extracted_images = len(review_dataframe)

    total_curated_images = len(curated_dataframe)


    total_rejected_images = int(
        (
            review_dataframe["human_decision"]
            == "REJECT"
        ).sum()
    )


    total_undecided_images = int(
        (
            review_dataframe["human_decision"]
            == "UNDECIDED"
        ).sum()
    )


    overall_retention_rate = safe_percentage(
        total_curated_images,
        total_extracted_images,
    )


    kept_blurry_images = int(
        curated_dataframe["is_blurry_bool"].sum()
    )


    kept_near_duplicate_images = int(
        curated_dataframe["near_duplicate_bool"].sum()
    )


    print()
    print("=" * 75)
    print("OVERALL DATASET SUMMARY")
    print("=" * 75)

    print(f"Recorded trials                  : {total_trials}")

    print(f"Extracted images                 : {total_extracted_images}")

    print(f"Curated KEEP images              : {total_curated_images}")

    print(f"Rejected images                  : {total_rejected_images}")

    print(f"Undecided images                 : {total_undecided_images}")

    print(f"Overall retention rate           : {overall_retention_rate:.2f}%")

    print(f"Blurry warnings among KEEP       : {kept_blurry_images}")

    print(f"Near-duplicate warnings among KEEP: {kept_near_duplicate_images}")

    print("=" * 75)


    # -------------------------------------------------------------------------
    # OBJECT CONTENT COVERAGE
    # -------------------------------------------------------------------------

    print()
    print("=" * 75)
    print("OBJECT CONTENT COVERAGE")
    print("=" * 75)


    object_order = [

        "EMPTY",
        "MARKER_ONLY",
        "POWER_ADAPTER_ONLY",
        "BOTH_OBJECTS",
        "UNKNOWN",

    ]


    object_counts = (

        enriched_curated["object_content_group"]

        .value_counts()

    )


    for group_name in object_order:

        count = int(object_counts.get(group_name, 0))

        percentage = safe_percentage(
            count,
            total_curated_images,
        )

        print(
            f"{group_name:<25}: "
            f"{count:>4} images "
            f"({percentage:>6.2f}%)"
        )


    print("=" * 75)


    # -------------------------------------------------------------------------
    # CONDITION TYPE COVERAGE
    # -------------------------------------------------------------------------

    print()
    print("=" * 75)
    print("EXPERIMENT CONDITION COVERAGE")
    print("=" * 75)


    condition_order = [

        "EMPTY",
        "STATIC",
        "OCCLUSION",
        "INTERACTION",
        "MOVEMENT",
        "OTHER",

    ]


    condition_counts = (

        enriched_curated["condition_type_group"]

        .value_counts()

    )


    for group_name in condition_order:

        count = int(condition_counts.get(group_name, 0))

        percentage = safe_percentage(
            count,
            total_curated_images,
        )

        print(
            f"{group_name:<25}: "
            f"{count:>4} images "
            f"({percentage:>6.2f}%)"
        )


    print("=" * 75)


    # -------------------------------------------------------------------------
    # PER-TRIAL SUMMARY
    # -------------------------------------------------------------------------

    print()
    print("=" * 75)
    print("PER-TRIAL COVERAGE")
    print("=" * 75)


    display_columns = [

        "trial_number",
        "trial_condition",
        "extracted_images",
        "kept_images",
        "rejected_images",
        "retention_rate_percent",
        "dataset_share_percent",

    ]


    print(

        coverage_dataframe[display_columns]

        .to_string(index=False)

    )


    print("=" * 75)


    # -------------------------------------------------------------------------
    # LARGEST AND SMALLEST TRIAL CONTRIBUTIONS
    # -------------------------------------------------------------------------

    if not coverage_dataframe.empty:


        largest_trial = coverage_dataframe.loc[

            coverage_dataframe["kept_images"].idxmax()

        ]


        smallest_trial = coverage_dataframe.loc[

            coverage_dataframe["kept_images"].idxmin()

        ]


        print()
        print("=" * 75)
        print("TRIAL CONTRIBUTION EXTREMES")
        print("=" * 75)


        print(

            "Largest contribution:"
        )

        print(

            f"  Trial {largest_trial['trial_number']} | "
            f"{largest_trial['trial_condition']} | "
            f"{largest_trial['kept_images']} images | "
            f"{largest_trial['dataset_share_percent']:.2f}%"

        )


        print()


        print(

            "Smallest contribution:"
        )

        print(

            f"  Trial {smallest_trial['trial_number']} | "
            f"{smallest_trial['trial_condition']} | "
            f"{smallest_trial['kept_images']} images | "
            f"{smallest_trial['dataset_share_percent']:.2f}%"

        )


        print("=" * 75)


# =============================================================================
# PRINT AUTOMATIC COVERAGE WARNINGS
# =============================================================================

def print_coverage_warnings(
    coverage_dataframe,
    enriched_curated,
):

    print()
    print("=" * 75)
    print("AUTOMATIC COVERAGE WARNINGS")
    print("=" * 75)


    warnings = []


    total_curated_images = len(enriched_curated)


    # -------------------------------------------------------------------------
    # DOMINANT TRIAL
    # -------------------------------------------------------------------------

    if not coverage_dataframe.empty:

        maximum_dataset_share = (

            coverage_dataframe["dataset_share_percent"]

            .max()

        )


        dominant_trials = coverage_dataframe[

            coverage_dataframe["dataset_share_percent"]

            > 20.0

        ]


        for _, row in dominant_trials.iterrows():

            warnings.append(

                "Trial "
                f"{row['trial_number']} "
                f"({row['trial_condition']}) contributes "
                f"{row['dataset_share_percent']:.2f}% "
                "of all curated images."

            )


    # -------------------------------------------------------------------------
    # LOW TRIAL COVERAGE
    # -------------------------------------------------------------------------

    low_coverage_trials = coverage_dataframe[

        coverage_dataframe["kept_images"]

        < 5

    ]


    for _, row in low_coverage_trials.iterrows():

        warnings.append(

            "Trial "
            f"{row['trial_number']} "
            f"({row['trial_condition']}) has only "
            f"{row['kept_images']} curated images."

        )


    # -------------------------------------------------------------------------
    # NO KEEP IMAGES
    # -------------------------------------------------------------------------

    zero_coverage_trials = coverage_dataframe[

        coverage_dataframe["kept_images"]

        == 0

    ]


    for _, row in zero_coverage_trials.iterrows():

        warnings.append(

            "Trial "
            f"{row['trial_number']} "
            f"({row['trial_condition']}) has no curated images."

        )


    # -------------------------------------------------------------------------
    # HIGH NEAR-DUPLICATE WARNING RATE
    # -------------------------------------------------------------------------

    near_duplicate_count = int(

        enriched_curated["near_duplicate_bool"]

        .sum()

    )


    near_duplicate_rate = safe_percentage(

        near_duplicate_count,

        total_curated_images,

    )


    if near_duplicate_rate > 30.0:

        warnings.append(

            f"{near_duplicate_rate:.2f}% of curated images still have "
            "near-duplicate warnings."

        )


    # -------------------------------------------------------------------------
    # HIGH BLUR WARNING RATE
    # -------------------------------------------------------------------------

    blurry_count = int(

        enriched_curated["is_blurry_bool"]

        .sum()

    )


    blurry_rate = safe_percentage(

        blurry_count,

        total_curated_images,

    )


    if blurry_rate > 10.0:

        warnings.append(

            f"{blurry_rate:.2f}% of curated images still have blur warnings."

        )


    # -------------------------------------------------------------------------
    # UNKNOWN OBJECT CONTENT
    # -------------------------------------------------------------------------

    unknown_object_count = int(

        (
            enriched_curated["object_content_group"]
            == "UNKNOWN"
        )

        .sum()

    )


    if unknown_object_count > 0:

        warnings.append(

            f"{unknown_object_count} curated images have unknown "
            "objects_present metadata."

        )


    # -------------------------------------------------------------------------
    # UNKNOWN CONDITION GROUP
    # -------------------------------------------------------------------------

    unknown_condition_count = int(

        (
            enriched_curated["condition_type_group"]
            == "OTHER"
        )

        .sum()

    )


    if unknown_condition_count > 0:

        warnings.append(

            f"{unknown_condition_count} curated images belong to "
            "unrecognized trial conditions."

        )


    # -------------------------------------------------------------------------
    # PRINT RESULT
    # -------------------------------------------------------------------------

    if not warnings:

        print("No automatic coverage warnings were generated.")

    else:

        for warning_number, warning in enumerate(
            warnings,
            start=1,
        ):

            print(f"{warning_number}. {warning}")


    print("=" * 75)


# =============================================================================
# SAVE COVERAGE REPORT
# =============================================================================

def save_coverage_report(coverage_dataframe):

    coverage_dataframe.to_csv(

        COVERAGE_REPORT,

        index=False,

        encoding="utf-8",

    )


    print()
    print("Coverage report saved to:")

    print(COVERAGE_REPORT)


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main():

    print()
    print("=" * 75)
    print("PROTOTYPE V2 - DATASET STATISTICS AND EXPERIMENT COVERAGE")
    print("=" * 75)

    print()
    print("Trials file:")
    print(TRIALS_FILE)

    print()
    print("Review manifest:")
    print(REVIEW_MANIFEST)

    print()
    print("Curated manifest:")
    print(CURATED_MANIFEST)

    print()
    print("Coverage report:")
    print(COVERAGE_REPORT)

    print("-" * 75)


    # -------------------------------------------------------------------------
    # LOAD FILES
    # -------------------------------------------------------------------------

    (
        trials_dataframe,
        review_dataframe,
        curated_dataframe,

    ) = load_input_files()


    # -------------------------------------------------------------------------
    # VALIDATE DATA RELATIONSHIPS
    # -------------------------------------------------------------------------

    validate_dataset_relationships(

        trials_dataframe,
        review_dataframe,
        curated_dataframe,

    )


    # -------------------------------------------------------------------------
    # BUILD PER-TRIAL REPORT
    # -------------------------------------------------------------------------

    coverage_dataframe = build_trial_coverage_report(

        trials_dataframe,
        review_dataframe,
        curated_dataframe,

    )


    # -------------------------------------------------------------------------
    # ADD EXPERIMENT COVERAGE GROUPS
    # -------------------------------------------------------------------------

    enriched_curated = add_coverage_groups(

        trials_dataframe,
        curated_dataframe,

    )


    # -------------------------------------------------------------------------
    # PRINT DATASET SUMMARY
    # -------------------------------------------------------------------------

    print_overall_summary(

        trials_dataframe,
        review_dataframe,
        curated_dataframe,
        coverage_dataframe,
        enriched_curated,

    )


    # -------------------------------------------------------------------------
    # PRINT WARNINGS
    # -------------------------------------------------------------------------

    print_coverage_warnings(

        coverage_dataframe,
        enriched_curated,

    )


    # -------------------------------------------------------------------------
    # SAVE REPORT
    # -------------------------------------------------------------------------

    save_coverage_report(

        coverage_dataframe

    )


    # -------------------------------------------------------------------------
    # FINAL STATUS
    # -------------------------------------------------------------------------

    print()
    print("=" * 75)
    print("STATUS: DATASET COVERAGE ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 75)

    print()
    print("Next stage:")
    print("Review the coverage results and decide whether targeted data")
    print("collection is needed before starting YOLO annotation.")

    print()


# =============================================================================
# PROGRAM ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()