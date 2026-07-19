"""
build_curated_dataset.py

HOW TO RUN THE CODE:
    python scripts\build_curated_dataset.py  (TYPE THIS IN TERMINAL)


Prototype V2 - Curated Dataset Builder

WHAT IS THIS CODE?

During the previous stages of Prototype V2, the system:

1. Recorded controlled experiment videos.
2. Extracted images from those videos.
3. Checked image quality.
4. Checked for near-duplicate images.
5. Created one combined dataset review file.
6. Allowed the user to visually inspect every extracted image.
7. Saved the human decision for each image as:

       KEEP
       REJECT
       UNDECIDED

However, all extracted images still remain inside:

    datasets/extracted/images/

This includes both useful images and rejected images.

The purpose of this script is to create a clean CURATED DATASET
containing only the images that were manually accepted by the user.


WHAT DOES THIS SCRIPT DO?

The script:

1. Reads:

       datasets/review/dataset_review_manifest.csv

2. Checks the "human_decision" column.

3. Selects only images marked:

       KEEP

4. Finds those images inside:

       datasets/extracted/images/

5. Copies the accepted images into:

       datasets/curated/images/

6. Creates:

       datasets/curated/curated_manifest.csv

The curated manifest preserves the metadata of the accepted images.


WHY DO WE COPY THE IMAGES?

The original extracted dataset should remain unchanged.

We do NOT delete rejected images from the extracted dataset because:

- the experiment remains reproducible,
- review decisions can be changed later,
- quality-analysis results remain traceable,
- mistakes during dataset curation can be corrected.

The curated dataset is therefore a DERIVED dataset.

Original extracted images:

    datasets/extracted/images/

                 ↓

Human review decisions

                 ↓

Accepted images only

                 ↓

    datasets/curated/images/


IMPORTANT

Running this script again will rebuild the curated image folder.

Existing files inside:

    datasets/curated/images/

will be removed before the new curated dataset is created.

The original extracted images are NEVER deleted or modified.


INPUTS

Review manifest:

    datasets/review/dataset_review_manifest.csv

Extracted images:

    datasets/extracted/images/


OUTPUTS

Curated images:

    datasets/curated/images/

Curated metadata:

    datasets/curated/curated_manifest.csv
"""


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import shutil

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

# Location of this Python script:
#
# SOFTWARE V2 AI PROTOTYPE/scripts/build_curated_dataset.py

SCRIPT_DIR = Path(__file__).resolve().parent


# Project root:
#
# SOFTWARE V2 AI PROTOTYPE/

PROJECT_ROOT = SCRIPT_DIR.parent


# Input CSV containing automatic analysis + human decisions.

REVIEW_MANIFEST = (
    PROJECT_ROOT
    / "datasets"
    / "review"
    / "dataset_review_manifest.csv"
)


# Folder containing all extracted images.

EXTRACTED_IMAGES_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "extracted"
    / "images"
)


# Output folder.

CURATED_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "curated"
)


# Output image folder.

CURATED_IMAGES_DIR = (
    CURATED_DIR
    / "images"
)


# Output curated CSV file.

CURATED_MANIFEST = (
    CURATED_DIR
    / "curated_manifest.csv"
)


# ============================================================
# REQUIRED CSV COLUMNS
# ============================================================

REQUIRED_COLUMNS = {

    "image_filename",
    "human_decision",

}


# ============================================================
# LOAD REVIEW MANIFEST
# ============================================================

def load_review_manifest():

    print("Loading dataset review manifest...")

    if not REVIEW_MANIFEST.exists():

        raise FileNotFoundError(

            "\nReview manifest was not found:\n"
            f"{REVIEW_MANIFEST}\n\n"
            "Run build_review_manifest.py first."

        )


    dataframe = pd.read_csv(REVIEW_MANIFEST)


    print(f"Rows loaded: {len(dataframe)}")


    # --------------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------------------------------------

    missing_columns = (

        REQUIRED_COLUMNS
        -
        set(dataframe.columns)

    )


    if missing_columns:

        raise ValueError(

            "\nThe review manifest is missing required columns:\n"
            f"{sorted(missing_columns)}"

        )


    return dataframe


# ============================================================
# NORMALIZE HUMAN DECISIONS
# ============================================================

def normalize_human_decisions(dataframe):

    """

    Human decisions should normally contain:

        KEEP
        REJECT
        UNDECIDED

    This function makes the processing safer.

    For example:

        keep
        Keep
         KEEP

    will all become:

        KEEP

    Empty values become:

        UNDECIDED

    """

    dataframe = dataframe.copy()


    dataframe["human_decision"] = (

        dataframe["human_decision"]

        .fillna("UNDECIDED")

        .astype(str)

        .str.strip()

        .str.upper()

    )


    return dataframe


# ============================================================
# SHOW REVIEW SUMMARY
# ============================================================

def print_review_summary(dataframe):

    total_images = len(dataframe)

    keep_count = (

        dataframe["human_decision"]
        .eq("KEEP")
        .sum()

    )

    reject_count = (

        dataframe["human_decision"]
        .eq("REJECT")
        .sum()

    )

    undecided_count = (

        dataframe["human_decision"]
        .eq("UNDECIDED")
        .sum()

    )


    known_decisions = {

        "KEEP",
        "REJECT",
        "UNDECIDED",

    }


    unknown_decision_mask = (

        ~dataframe["human_decision"]
        .isin(known_decisions)

    )


    unknown_count = unknown_decision_mask.sum()


    print()
    print("=" * 70)
    print("HUMAN REVIEW SUMMARY")
    print("=" * 70)

    print(f"Total reviewed rows        : {total_images}")
    print(f"KEEP decisions             : {keep_count}")
    print(f"REJECT decisions           : {reject_count}")
    print(f"UNDECIDED decisions        : {undecided_count}")
    print(f"Unknown decision values    : {unknown_count}")

    print("=" * 70)


    if undecided_count > 0:

        print()
        print(
            "WARNING: Some images are still marked UNDECIDED."
        )

        print(
            "These images will NOT be copied into the curated dataset."
        )


    if unknown_count > 0:

        print()
        print(
            "WARNING: Some rows contain unknown human_decision values."
        )

        print(
            "These rows will NOT be copied into the curated dataset."
        )

        print()

        print("Unknown values:")

        unknown_values = (

            dataframe.loc[
                unknown_decision_mask,
                "human_decision"
            ]
            .value_counts()
        )

        print(unknown_values.to_string())


# ============================================================
# PREPARE CURATED DIRECTORY
# ============================================================

def prepare_curated_directory():

    """

    Rebuild the curated images folder.

    IMPORTANT:

    Only:

        datasets/curated/images/

    is removed.

    The original extracted images remain untouched.

    """


    CURATED_DIR.mkdir(

        parents=True,
        exist_ok=True

    )


    if CURATED_IMAGES_DIR.exists():

        print()
        print("Existing curated image folder found.")

        print("Removing old curated images...")

        shutil.rmtree(CURATED_IMAGES_DIR)


    CURATED_IMAGES_DIR.mkdir(

        parents=True,
        exist_ok=True

    )


# ============================================================
# BUILD CURATED DATASET
# ============================================================

def build_curated_dataset(dataframe):

    """

    Select KEEP rows and copy their images.

    """


    keep_dataframe = (

        dataframe[

            dataframe["human_decision"]
            ==
            "KEEP"

        ]

        .copy()

    )


    print()
    print("=" * 70)
    print("BUILDING CURATED DATASET")
    print("=" * 70)

    print(
        f"Images marked KEEP         : {len(keep_dataframe)}"
    )


    if keep_dataframe.empty:

        print()
        print("No images are marked KEEP.")

        print(
            "The curated dataset will be empty."
        )

        return keep_dataframe, []


    copied_rows = []

    missing_images = []


    total_keep_images = len(keep_dataframe)


    # --------------------------------------------------------
    # COPY KEEP IMAGES
    # --------------------------------------------------------

    for number, (_, row) in enumerate(

        keep_dataframe.iterrows(),
        start=1

    ):


        image_filename = str(

            row["image_filename"]

        ).strip()


        source_image = (

            EXTRACTED_IMAGES_DIR
            /
            image_filename

        )


        destination_image = (

            CURATED_IMAGES_DIR
            /
            image_filename

        )


        # ----------------------------------------------------
        # CHECK SOURCE IMAGE
        # ----------------------------------------------------

        if not source_image.exists():

            missing_images.append(

                image_filename

            )

            print(

                f"[MISSING] "
                f"{image_filename}"

            )

            continue


        # ----------------------------------------------------
        # COPY IMAGE
        # ----------------------------------------------------

        shutil.copy2(

            source_image,
            destination_image

        )


        copied_rows.append(

            row.to_dict()

        )


        # ----------------------------------------------------
        # PROGRESS DISPLAY
        # ----------------------------------------------------

        print(

            f"\rCopied images: "
            f"{number}/{total_keep_images}",

            end="",

            flush=True

        )


    print()


    # --------------------------------------------------------
    # CREATE DATAFRAME CONTAINING ONLY SUCCESSFULLY COPIED ROWS
    # --------------------------------------------------------

    curated_dataframe = pd.DataFrame(

        copied_rows,

        columns=dataframe.columns

    )


    return curated_dataframe, missing_images


# ============================================================
# SAVE CURATED MANIFEST
# ============================================================

def save_curated_manifest(curated_dataframe):


    curated_dataframe.to_csv(

        CURATED_MANIFEST,

        index=False

    )


    print()
    print(

        "Curated manifest saved to:"

    )

    print(

        CURATED_MANIFEST

    )


# ============================================================
# VERIFY CURATED DATASET
# ============================================================

def verify_curated_dataset(

    curated_dataframe,
    missing_images

):

    """

    Perform simple consistency checks.

    """


    copied_file_count = len(

        list(

            CURATED_IMAGES_DIR.glob("*")

        )

    )


    manifest_row_count = len(

        curated_dataframe

    )


    print()
    print("=" * 70)
    print("CURATED DATASET VERIFICATION")
    print("=" * 70)

    print(

        f"Curated manifest rows      : "
        f"{manifest_row_count}"

    )

    print(

        f"Files in curated folder    : "
        f"{copied_file_count}"

    )

    print(

        f"Missing source images      : "
        f"{len(missing_images)}"

    )


    # --------------------------------------------------------
    # CONSISTENCY CHECK
    # --------------------------------------------------------

    if copied_file_count == manifest_row_count:

        print(

            "Manifest / image count     : MATCH"

        )

    else:

        print(

            "Manifest / image count     : MISMATCH"

        )


    print("=" * 70)


    # --------------------------------------------------------
    # MISSING IMAGE REPORT
    # --------------------------------------------------------

    if missing_images:

        print()
        print("MISSING IMAGES:")

        for image_filename in missing_images:

            print(

                f"  - {image_filename}"

            )


# ============================================================
# SHOW CURATED DATASET DISTRIBUTION
# ============================================================

def print_dataset_distribution(curated_dataframe):

    """

    Show simple information about the curated dataset.

    At this stage we mainly want to know:

    - how many trials remain,
    - how many images came from each trial.

    """


    if curated_dataframe.empty:

        return


    print()
    print("=" * 70)
    print("CURATED DATASET DISTRIBUTION")
    print("=" * 70)


    # --------------------------------------------------------
    # TRIAL COUNT
    # --------------------------------------------------------

    if "trial_id" in curated_dataframe.columns:

        trial_count = (

            curated_dataframe["trial_id"]
            .nunique()

        )

        print(

            f"Unique trials represented  : "
            f"{trial_count}"

        )


        print()
        print("Images per trial:")


        trial_distribution = (

            curated_dataframe["trial_id"]

            .value_counts()

            .sort_index()

        )


        print(

            trial_distribution.to_string()

        )


    print("=" * 70)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():


    print()
    print("=" * 70)
    print("PROTOTYPE V2 - CURATED DATASET BUILDER")
    print("=" * 70)

    print()
    print("Review manifest:")
    print(REVIEW_MANIFEST)

    print()
    print("Extracted images:")
    print(EXTRACTED_IMAGES_DIR)

    print()
    print("Curated images:")
    print(CURATED_IMAGES_DIR)

    print()
    print("Curated manifest:")
    print(CURATED_MANIFEST)

    print("-" * 70)


    # --------------------------------------------------------
    # LOAD REVIEW DATA
    # --------------------------------------------------------

    dataframe = load_review_manifest()


    # --------------------------------------------------------
    # NORMALIZE DECISIONS
    # --------------------------------------------------------

    dataframe = normalize_human_decisions(

        dataframe

    )


    # --------------------------------------------------------
    # SHOW REVIEW SUMMARY
    # --------------------------------------------------------

    print_review_summary(

        dataframe

    )


    # --------------------------------------------------------
    # PREPARE OUTPUT DIRECTORY
    # --------------------------------------------------------

    prepare_curated_directory()


    # --------------------------------------------------------
    # BUILD DATASET
    # --------------------------------------------------------

    curated_dataframe, missing_images = (

        build_curated_dataset(

            dataframe

        )

    )


    # --------------------------------------------------------
    # SAVE MANIFEST
    # --------------------------------------------------------

    save_curated_manifest(

        curated_dataframe

    )


    # --------------------------------------------------------
    # VERIFY DATASET
    # --------------------------------------------------------

    verify_curated_dataset(

        curated_dataframe,
        missing_images

    )


    # --------------------------------------------------------
    # SHOW DATASET DISTRIBUTION
    # --------------------------------------------------------

    print_dataset_distribution(

        curated_dataframe

    )


    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    print()
    print("=" * 70)

    if missing_images:

        print(

            "STATUS: CURATED DATASET CREATED WITH WARNINGS"

        )

    else:

        print(

            "STATUS: CURATED DATASET CREATED SUCCESSFULLY"

        )

    print("=" * 70)

    print()
    print(

        "Next stage:"
    )

    print(

        "Analyse curated dataset statistics and experiment coverage."

    )

    print()


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()