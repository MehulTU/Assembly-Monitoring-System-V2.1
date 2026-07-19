"""
finalize_dataset.py

HOW TO RUN THE FILE:
    python scripts\finalize_dataset.py  (TYPE THIS IN TERMINAL)

Prototype V2 - Final Annotation Dataset Builder

PURPOSE
-------
This program creates the final image dataset that will later be annotated
and used for YOLO training.

The previous stages of the dataset pipeline already performed:

    1. Controlled video recording.
    2. Trial metadata storage.
    3. Frame extraction from recorded videos.
    4. Automatic image-quality analysis.
    5. Near-duplicate image detection.
    6. Human review of extracted images.
    7. Creation of the curated dataset.
    8. Dataset coverage analysis.

However, the curated dataset can still contain too many visually similar
images from the same trial.

For example, a long movement trial may produce many images while another
trial may contain only a few images.

Training YOLO with many nearly identical images can unnecessarily increase
dataset imbalance without adding much new visual information.

This program therefore creates a separate FINAL dataset.

WHAT THE PROGRAM DOES
---------------------
1. Loads curated_manifest.csv.

2. Reads the curated images.

3. Processes every trial separately.

4. Protects trials containing only a small number of images.

5. Limits very large trials to a configurable maximum number of images.

6. Uses image similarity to prefer visually different images instead of
   simply choosing images randomly.

7. Creates:

       datasets/final/images/

       datasets/final/final_manifest.csv

       datasets/final/finalization_report.csv

8. Verifies that the number of copied images matches the final manifest.

IMPORTANT
---------
The curated dataset is NEVER modified or deleted.

The final dataset is a separate annotation-ready dataset.

The final dataset created by this script will become the input for the
next stage:

    YOLO ANNOTATION
"""


# ======================================================================
# IMPORTS
# ======================================================================

from pathlib import Path
import shutil

import cv2
import numpy as np
import pandas as pd


# ======================================================================
# PROJECT PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CURATED_MANIFEST = (
    PROJECT_ROOT
    / "datasets"
    / "curated"
    / "curated_manifest.csv"
)

CURATED_IMAGE_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "curated"
    / "images"
)

FINAL_DATASET_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "final"
)

FINAL_IMAGE_DIR = (
    FINAL_DATASET_DIR
    / "images"
)

FINAL_MANIFEST = (
    FINAL_DATASET_DIR
    / "final_manifest.csv"
)

FINALIZATION_REPORT = (
    FINAL_DATASET_DIR
    / "finalization_report.csv"
)


# ======================================================================
# FINALIZATION SETTINGS
# ======================================================================

# Trials containing this number of images or fewer are protected.
# All their images will be kept.
PROTECTED_TRIAL_SIZE = 5


# Maximum number of images allowed from one trial in the final dataset.
#
# The largest trial in our current pilot dataset contains 27 images.
# Limiting large trials reduces excessive contribution from one condition.
MAX_IMAGES_PER_TRIAL = 15


# Size used when comparing images.
#
# Images are reduced before similarity comparison because comparing full
# 1280 x 720 images would be unnecessarily expensive.
SIMILARITY_IMAGE_SIZE = (160, 90)


# ======================================================================
# HELPER FUNCTION
# READ IMAGE FOR SIMILARITY COMPARISON
# ======================================================================

def load_similarity_image(image_path):
    """
    Load an image and prepare a small grayscale representation.

    The smaller grayscale image is used only for similarity comparison.

    The original image is never modified.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(
        gray,
        SIMILARITY_IMAGE_SIZE
    )

    return resized.astype(np.float32)


# ======================================================================
# HELPER FUNCTION
# CALCULATE IMAGE SIMILARITY
# ======================================================================

def calculate_similarity(image_a, image_b):
    """
    Calculate normalized correlation between two images.

    Result is approximately:

        1.0  = extremely similar
        0.0  = very different

    Higher value means greater visual similarity.
    """

    if image_a is None or image_b is None:
        return 0.0

    similarity = cv2.matchTemplate(
        image_a,
        image_b,
        cv2.TM_CCOEFF_NORMED
    )[0][0]

    return float(similarity)


# ======================================================================
# DIVERSITY-BASED IMAGE SELECTION
# ======================================================================

def select_diverse_images(trial_dataframe, maximum_images):
    """
    Select a visually diverse subset of images from one trial.

    Selection strategy:

    1. Sort images according to source frame number.

    2. Keep the first image.

    3. Keep the last image.

    4. Repeatedly select the image that is least similar to the images
       already selected.

    This is called greedy farthest-point selection.

    It is preferable to random sampling because it attempts to preserve
    different visual states from the trial.
    """

    trial_dataframe = trial_dataframe.sort_values(
        "source_frame_number"
    ).reset_index(drop=True)

    total_images = len(trial_dataframe)

    # No reduction required.
    if total_images <= maximum_images:
        return trial_dataframe.copy()

    print(
        f"  Selecting {maximum_images} diverse images "
        f"from {total_images} images..."
    )

    similarity_images = []

    for _, row in trial_dataframe.iterrows():

        image_path = (
            CURATED_IMAGE_DIR
            / row["image_filename"]
        )

        prepared_image = load_similarity_image(image_path)

        similarity_images.append(prepared_image)


    # Always keep first and last images.
    selected_indices = [0]

    if total_images > 1:
        selected_indices.append(total_images - 1)


    remaining_indices = set(
        range(total_images)
    ) - set(selected_indices)


    # --------------------------------------------------------------
    # GREEDY DIVERSITY SELECTION
    # --------------------------------------------------------------

    while (
        len(selected_indices) < maximum_images
        and remaining_indices
    ):

        best_candidate = None

        lowest_maximum_similarity = float("inf")


        for candidate_index in remaining_indices:

            candidate_image = similarity_images[candidate_index]

            similarities_to_selected = []

            for selected_index in selected_indices:

                selected_image = similarity_images[selected_index]

                similarity = calculate_similarity(
                    candidate_image,
                    selected_image
                )

                similarities_to_selected.append(similarity)


            # We measure how similar this candidate is to its closest
            # already-selected image.
            maximum_similarity = max(
                similarities_to_selected
            )


            # Prefer the image whose closest selected neighbour is
            # least similar.
            if maximum_similarity < lowest_maximum_similarity:

                lowest_maximum_similarity = maximum_similarity

                best_candidate = candidate_index


        if best_candidate is None:
            break


        selected_indices.append(best_candidate)

        remaining_indices.remove(best_candidate)


    selected_indices = sorted(selected_indices)


    return trial_dataframe.iloc[
        selected_indices
    ].copy()


# ======================================================================
# MAIN PROGRAM
# ======================================================================

def main():

    print()
    print("=" * 75)
    print("PROTOTYPE V2 - FINAL ANNOTATION DATASET BUILDER")
    print("=" * 75)

    print()
    print("Curated manifest:")
    print(CURATED_MANIFEST)

    print()
    print("Curated images:")
    print(CURATED_IMAGE_DIR)

    print()
    print("Final images:")
    print(FINAL_IMAGE_DIR)

    print()
    print("Final manifest:")
    print(FINAL_MANIFEST)

    print()
    print("Finalization report:")
    print(FINALIZATION_REPORT)

    print("-" * 75)


    # ==================================================================
    # CHECK REQUIRED INPUTS
    # ==================================================================

    if not CURATED_MANIFEST.exists():

        print()
        print("ERROR: Curated manifest does not exist.")
        print()
        print("Run build_curated_dataset.py first.")

        return


    if not CURATED_IMAGE_DIR.exists():

        print()
        print("ERROR: Curated image folder does not exist.")

        return


    # ==================================================================
    # LOAD CURATED MANIFEST
    # ==================================================================

    print()
    print("Loading curated manifest...")

    curated_df = pd.read_csv(CURATED_MANIFEST)

    print(f"Curated rows loaded: {len(curated_df)}")


    # ==================================================================
    # VALIDATE REQUIRED COLUMNS
    # ==================================================================

    required_columns = [
        "image_filename",
        "trial_id",
        "source_frame_number"
    ]

    missing_columns = [

        column

        for column in required_columns

        if column not in curated_df.columns
    ]


    if missing_columns:

        print()
        print("ERROR: Required columns are missing:")
        print(missing_columns)

        return


    # ==================================================================
    # PREPARE OUTPUT FOLDER
    # ==================================================================

    FINAL_DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    if FINAL_IMAGE_DIR.exists():

        print()
        print("Existing final image folder found.")
        print("Removing old final images...")

        shutil.rmtree(FINAL_IMAGE_DIR)


    FINAL_IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ==================================================================
    # PROCESS EACH TRIAL
    # ==================================================================

    print()
    print("=" * 75)
    print("FINAL DATASET SELECTION")
    print("=" * 75)


    selected_trial_dataframes = []

    report_rows = []


    grouped_trials = curated_df.groupby(
        "trial_id",
        sort=False
    )


    for trial_id, trial_df in grouped_trials:

        trial_df = trial_df.sort_values(
            "source_frame_number"
        ).reset_index(drop=True)


        original_count = len(trial_df)


        print()
        print(f"Trial: {trial_id}")
        print(f"Curated images: {original_count}")


        # --------------------------------------------------------------
        # SMALL TRIAL PROTECTION
        # --------------------------------------------------------------

        if original_count <= PROTECTED_TRIAL_SIZE:

            selected_df = trial_df.copy()

            selection_method = "PROTECTED_KEEP_ALL"

            print(
                "  Small trial protected -> keeping all images."
            )


        # --------------------------------------------------------------
        # LARGE TRIAL REDUCTION
        # --------------------------------------------------------------

        elif original_count > MAX_IMAGES_PER_TRIAL:

            selected_df = select_diverse_images(
                trial_df,
                MAX_IMAGES_PER_TRIAL
            )

            selection_method = "DIVERSITY_SELECTION"

            print(
                f"  Reduced from {original_count} "
                f"to {len(selected_df)} images."
            )


        # --------------------------------------------------------------
        # NORMAL TRIAL
        # --------------------------------------------------------------

        else:

            selected_df = trial_df.copy()

            selection_method = "KEEP_ALL"

            print(
                "  Trial within allowed size -> keeping all images."
            )


        selected_trial_dataframes.append(
            selected_df
        )


        final_count = len(selected_df)


        report_rows.append({

            "trial_id": trial_id,

            "curated_image_count": original_count,

            "final_image_count": final_count,

            "removed_image_count":
                original_count - final_count,

            "selection_method":
                selection_method
        })


    # ==================================================================
    # COMBINE FINAL SELECTION
    # ==================================================================

    final_df = pd.concat(
        selected_trial_dataframes,
        ignore_index=True
    )


    # ==================================================================
    # COPY FINAL IMAGES
    # ==================================================================

    print()
    print("=" * 75)
    print("COPYING FINAL DATASET IMAGES")
    print("=" * 75)


    copied_images = 0

    missing_images = []


    for index, row in final_df.iterrows():

        image_filename = row["image_filename"]

        source_image = (
            CURATED_IMAGE_DIR
            / image_filename
        )

        destination_image = (
            FINAL_IMAGE_DIR
            / image_filename
        )


        if not source_image.exists():

            missing_images.append(
                image_filename
            )

            continue


        shutil.copy2(
            source_image,
            destination_image
        )


        copied_images += 1


        if (
            copied_images % 25 == 0
            or copied_images == len(final_df)
        ):

            print(
                f"Copied images: "
                f"{copied_images}/{len(final_df)}"
            )


    # ==================================================================
    # ADD FINAL DATASET METADATA
    # ==================================================================

    final_df = final_df.copy()

    final_df["final_dataset_selected"] = True

    final_df["final_dataset_version"] = "V1"


    # ==================================================================
    # SAVE FINAL MANIFEST
    # ==================================================================

    final_df.to_csv(
        FINAL_MANIFEST,
        index=False
    )


    # ==================================================================
    # SAVE FINALIZATION REPORT
    # ==================================================================

    report_df = pd.DataFrame(
        report_rows
    )


    report_df.to_csv(
        FINALIZATION_REPORT,
        index=False
    )


    # ==================================================================
    # VERIFY FINAL DATASET
    # ==================================================================

    files_in_final_folder = list(
        FINAL_IMAGE_DIR.glob("*.jpg")
    )


    print()
    print("=" * 75)
    print("FINAL DATASET VERIFICATION")
    print("=" * 75)

    print(
        f"Curated dataset images       : "
        f"{len(curated_df)}"
    )

    print(
        f"Final manifest rows          : "
        f"{len(final_df)}"
    )

    print(
        f"Files in final image folder  : "
        f"{len(files_in_final_folder)}"
    )

    print(
        f"Images removed               : "
        f"{len(curated_df) - len(final_df)}"
    )

    print(
        f"Missing source images        : "
        f"{len(missing_images)}"
    )


    manifest_matches_folder = (
        len(final_df)
        == len(files_in_final_folder)
        == copied_images
    )


    print(
        "Manifest / image count       : "
        + (
            "MATCH"
            if manifest_matches_folder
            else "MISMATCH"
        )
    )


    # ==================================================================
    # FINAL DATASET DISTRIBUTION
    # ==================================================================

    print()
    print("=" * 75)
    print("FINAL DATASET DISTRIBUTION")
    print("=" * 75)


    final_trial_counts = (

        final_df

        .groupby("trial_id")

        .size()

        .sort_values()
    )


    print()
    print("Images per trial:")

    print(final_trial_counts)


    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================

    print()
    print("=" * 75)
    print("FINALIZATION SUMMARY")
    print("=" * 75)

    print(
        f"Original curated images : {len(curated_df)}"
    )

    print(
        f"Final selected images   : {len(final_df)}"
    )

    print(
        f"Images removed          : "
        f"{len(curated_df) - len(final_df)}"
    )

    print(
        f"Trials represented      : "
        f"{final_df['trial_id'].nunique()}"
    )


    if missing_images:

        print()
        print("WARNING: Some source images were missing.")

        print("Missing images:")

        for filename in missing_images:

            print(filename)


    if manifest_matches_folder and not missing_images:

        print()
        print("=" * 75)
        print(
            "STATUS: FINAL ANNOTATION DATASET CREATED SUCCESSFULLY"
        )
        print("=" * 75)

        print()
        print("Next stage:")
        print("Inspect final dataset statistics and freeze Dataset Version 1.")
        print("After dataset verification, begin YOLO annotation.")

    else:

        print()
        print("=" * 75)
        print("STATUS: FINAL DATASET CREATED WITH ERRORS")
        print("=" * 75)

        print()
        print(
            "Do not start annotation until the dataset errors "
            "are corrected."
        )


# ======================================================================
# PROGRAM ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()