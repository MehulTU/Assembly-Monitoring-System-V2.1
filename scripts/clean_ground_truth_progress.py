"""
clean_ground_truth_progress.py

PROTOTYPE V2 - GROUND-TRUTH PROGRESS CLEANER

HOW TO RUN THE FILE:
    python scripts\clean_ground_truth_progress.py

What this code does:
    This script cleans the manually recorded ground-truth change points
    created during validation-video annotation.

    During manual annotation, the same assembly state may accidentally be
    entered several times even though the physical state of the workspace
    has not changed.

    Example:

        Frame 0   -> EMPTY
        Frame 39  -> EMPTY
        Frame 76  -> EMPTY
        Frame 180 -> MARKER_ONLY
        Frame 199 -> MARKER_ONLY

    For frame-level ground truth, only the first frame where a new state
    begins is required.

    Therefore, the cleaned change points become:

        Frame 0   -> EMPTY
        Frame 180 -> MARKER_ONLY

    The script performs the following steps:

        1. Loads the original ground-truth progress CSV.

        2. Creates a backup copy of the original progress CSV.

        3. Sorts all recorded change points by frame number.

        4. Removes consecutive duplicate assembly-state entries.

        5. Saves the cleaned change points to a new CSV file.

        6. Reads the validation video to determine the exact number of frames
           and FPS.

        7. Expands the cleaned change points into frame-level ground truth.

        8. Saves the regenerated final ground-truth CSV.

        9. Verifies that:
               - frame numbering starts at 0,
               - every video frame has exactly one ground-truth row,
               - no frame numbers are duplicated,
               - no frames are missing,
               - the final CSV row count matches the video frame count.

Important:
    The original progress file is NOT deleted.

    A backup copy is created before cleaning.

    The cleaned progress file is saved separately.

    The final frame-level ground-truth CSV is regenerated from the cleaned
    change points.

Run from the project root:

    python scripts\clean_ground_truth_progress.py
"""


from pathlib import Path
import shutil

import cv2
import pandas as pd


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


VALIDATION_VIDEO = (
    PROJECT_ROOT
    / "datasets"
    / "validation"
    / "videos"
    / "validation_01_correct_sequence.mp4"
)


GROUND_TRUTH_DIRECTORY = (
    PROJECT_ROOT
    / "datasets"
    / "validation"
    / "ground_truth"
)


PROGRESS_FILE = (
    GROUND_TRUTH_DIRECTORY
    / "validation_01_ground_truth_progress.csv"
)


BACKUP_FILE = (
    GROUND_TRUTH_DIRECTORY
    / "validation_01_ground_truth_progress_backup.csv"
)


CLEANED_PROGRESS_FILE = (
    GROUND_TRUTH_DIRECTORY
    / "validation_01_ground_truth_progress_cleaned.csv"
)


FINAL_GROUND_TRUTH_FILE = (
    GROUND_TRUTH_DIRECTORY
    / "validation_01_ground_truth.csv"
)


# ============================================================================
# VALID ASSEMBLY STATES
# ============================================================================

VALID_STATES = {
    "EMPTY": {
        "marker_present_gt": 0,
        "power_adapter_present_gt": 0,
        "assembly_state_id_gt": 0,
    },

    "MARKER_ONLY": {
        "marker_present_gt": 1,
        "power_adapter_present_gt": 0,
        "assembly_state_id_gt": 1,
    },

    "POWER_ADAPTER_ONLY": {
        "marker_present_gt": 0,
        "power_adapter_present_gt": 1,
        "assembly_state_id_gt": 2,
    },

    "BOTH_PRESENT": {
        "marker_present_gt": 1,
        "power_adapter_present_gt": 1,
        "assembly_state_id_gt": 3,
    },
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_header(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def find_column(dataframe, possible_names):
    """
    Finds the first matching column name.

    This allows the cleaner to work even if the annotation script used
    slightly different names for frame number or assembly state.
    """

    for column_name in possible_names:

        if column_name in dataframe.columns:
            return column_name

    raise ValueError(
        "\nCould not find any of these columns:\n"
        f"{possible_names}\n\n"
        f"Available columns:\n{list(dataframe.columns)}"
    )


def load_video_information(video_path):
    """
    Opens the validation video and returns:

        FPS
        frame count
        frame width
        frame height
    """

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open validation video:\n{video_path}"
        )

    fps = float(capture.get(cv2.CAP_PROP_FPS))

    frame_count = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    frame_width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    frame_height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    capture.release()

    if fps <= 0:
        raise ValueError(
            f"Invalid video FPS detected: {fps}"
        )

    if frame_count <= 0:
        raise ValueError(
            f"Invalid video frame count detected: {frame_count}"
        )

    return fps, frame_count, frame_width, frame_height


def clean_consecutive_duplicate_states(
    progress_dataframe,
    frame_column,
    state_column,
):
    """
    Removes consecutive duplicate assembly-state entries.

    Example:

        0   EMPTY
        39  EMPTY
        76  EMPTY
        180 MARKER_ONLY
        199 MARKER_ONLY
        344 BOTH_PRESENT

    becomes:

        0   EMPTY
        180 MARKER_ONLY
        344 BOTH_PRESENT

    The first occurrence of every new active state is preserved.
    """

    cleaned_rows = []

    previous_state = None

    for _, row in progress_dataframe.iterrows():

        current_state = str(row[state_column]).strip().upper()

        if current_state != previous_state:

            cleaned_rows.append(row.copy())

            previous_state = current_state

    cleaned_dataframe = pd.DataFrame(cleaned_rows)

    cleaned_dataframe.reset_index(drop=True, inplace=True)

    return cleaned_dataframe


def create_frame_level_ground_truth(
    cleaned_progress,
    frame_column,
    state_column,
    total_frames,
    fps,
    video_filename,
):
    """
    Expands change-point annotations into one ground-truth row per video frame.

    A recorded change point defines the active state starting from that frame
    and continuing until the next recorded change point.
    """

    ground_truth_rows = []

    change_points = cleaned_progress[
        [frame_column, state_column]
    ].to_dict("records")


    current_change_index = 0

    current_state = str(
        change_points[0][state_column]
    ).strip().upper()


    for frame_number in range(total_frames):

        while (
            current_change_index + 1 < len(change_points)
            and frame_number
            >= int(
                change_points[
                    current_change_index + 1
                ][frame_column]
            )
        ):

            current_change_index += 1

            current_state = str(
                change_points[
                    current_change_index
                ][state_column]
            ).strip().upper()


        state_information = VALID_STATES[current_state]


        ground_truth_rows.append(
            {
                "experiment_id": "validation_01",

                "video_filename": video_filename,

                "frame_number": frame_number,

                "time_seconds": round(
                    frame_number / fps,
                    6,
                ),

                "marker_present_gt":
                    state_information["marker_present_gt"],

                "power_adapter_present_gt":
                    state_information["power_adapter_present_gt"],

                "assembly_state_gt":
                    current_state,

                "assembly_state_id_gt":
                    state_information["assembly_state_id_gt"],
            }
        )


    return pd.DataFrame(ground_truth_rows)


# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main():

    print_header(
        "PROTOTYPE V2 - GROUND-TRUTH PROGRESS CLEANER"
    )


    print("\nOriginal progress file:")
    print(PROGRESS_FILE)

    print("\nBackup file:")
    print(BACKUP_FILE)

    print("\nCleaned progress file:")
    print(CLEANED_PROGRESS_FILE)

    print("\nFinal frame-level ground truth:")
    print(FINAL_GROUND_TRUTH_FILE)

    print("\nValidation video:")
    print(VALIDATION_VIDEO)


    # ========================================================================
    # VERIFY REQUIRED INPUT FILES
    # ========================================================================

    print_header(
        "VERIFYING INPUT FILES"
    )


    if not PROGRESS_FILE.exists():

        raise FileNotFoundError(
            "\nGround-truth progress file was not found:\n"
            f"{PROGRESS_FILE}"
        )


    if not VALIDATION_VIDEO.exists():

        raise FileNotFoundError(
            "\nValidation video was not found:\n"
            f"{VALIDATION_VIDEO}"
        )


    print("Found progress file.")
    print("Found validation video.")


    # ========================================================================
    # LOAD ORIGINAL PROGRESS FILE
    # ========================================================================

    print_header(
        "LOADING ORIGINAL GROUND-TRUTH PROGRESS"
    )


    progress_dataframe = pd.read_csv(PROGRESS_FILE)


    print(
        f"Original progress rows : "
        f"{len(progress_dataframe)}"
    )


    if progress_dataframe.empty:

        raise ValueError(
            "The progress CSV contains no recorded change points."
        )


    print("\nAvailable columns:")

    for column in progress_dataframe.columns:
        print(f"  - {column}")


    # ========================================================================
    # DETECT REQUIRED COLUMNS
    # ========================================================================

    frame_column = find_column(
        progress_dataframe,
        [
            "frame_number",
            "frame",
            "source_frame_number",
            "change_frame",
        ],
    )


    state_column = find_column(
        progress_dataframe,
        [
            "assembly_state_gt",
            "assembly_state",
            "state",
            "ground_truth_state",
        ],
    )


    print("\nDetected columns:")

    print(
        f"Frame column : {frame_column}"
    )

    print(
        f"State column : {state_column}"
    )


    # ========================================================================
    # NORMALIZE DATA
    # ========================================================================

    print_header(
        "VALIDATING RECORDED CHANGE POINTS"
    )


    progress_dataframe[frame_column] = pd.to_numeric(
        progress_dataframe[frame_column],
        errors="raise",
    ).astype(int)


    progress_dataframe[state_column] = (
        progress_dataframe[state_column]
        .astype(str)
        .str.strip()
        .str.upper()
    )


    progress_dataframe.sort_values(
        by=frame_column,
        inplace=True,
    )


    progress_dataframe.reset_index(
        drop=True,
        inplace=True,
    )


    duplicate_frame_count = int(
        progress_dataframe[frame_column]
        .duplicated()
        .sum()
    )


    invalid_states = sorted(
        set(progress_dataframe[state_column])
        - set(VALID_STATES.keys())
    )


    print(
        f"Duplicate frame numbers : "
        f"{duplicate_frame_count}"
    )

    print(
        f"Invalid assembly states : "
        f"{len(invalid_states)}"
    )


    if duplicate_frame_count > 0:

        raise ValueError(
            "\nDuplicate frame numbers were found in the progress file.\n"
            "The cleaner stopped to avoid silently changing ambiguous "
            "ground-truth annotations."
        )


    if invalid_states:

        raise ValueError(
            "\nUnknown assembly states were found:\n"
            f"{invalid_states}"
        )


    # ========================================================================
    # LOAD VIDEO INFORMATION
    # ========================================================================

    print_header(
        "VALIDATION VIDEO INFORMATION"
    )


    (
        fps,
        total_frames,
        frame_width,
        frame_height,

    ) = load_video_information(
        VALIDATION_VIDEO
    )


    duration_seconds = total_frames / fps


    print(
        f"FPS          : {fps:.3f}"
    )

    print(
        f"Resolution   : "
        f"{frame_width} x {frame_height}"
    )

    print(
        f"Total frames : {total_frames}"
    )

    print(
        f"Duration     : "
        f"{duration_seconds:.3f} seconds"
    )


    # ========================================================================
    # VERIFY CHANGE-POINT FRAME RANGE
    # ========================================================================

    minimum_frame = int(
        progress_dataframe[frame_column].min()
    )

    maximum_frame = int(
        progress_dataframe[frame_column].max()
    )


    if minimum_frame < 0:

        raise ValueError(
            "A negative frame number was found."
        )


    if maximum_frame >= total_frames:

        raise ValueError(
            "\nA recorded change point is outside the video frame range.\n\n"
            f"Maximum recorded frame : {maximum_frame}\n"
            f"Final valid video frame: {total_frames - 1}"
        )


    if minimum_frame != 0:

        raise ValueError(
            "\nThe first recorded ground-truth change point must be frame 0.\n"
            f"Current first frame: {minimum_frame}"
        )


    print("All recorded frame numbers are valid.")


    # ========================================================================
    # CREATE BACKUP
    # ========================================================================

    print_header(
        "CREATING BACKUP"
    )


    GROUND_TRUTH_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


    shutil.copy2(
        PROGRESS_FILE,
        BACKUP_FILE,
    )


    print("Original progress file backed up successfully.")

    print("\nBackup saved to:")
    print(BACKUP_FILE)


    # ========================================================================
    # CLEAN CONSECUTIVE DUPLICATE STATES
    # ========================================================================

    print_header(
        "CLEANING GROUND-TRUTH CHANGE POINTS"
    )


    cleaned_progress = clean_consecutive_duplicate_states(
        progress_dataframe,
        frame_column,
        state_column,
    )


    original_count = len(progress_dataframe)

    cleaned_count = len(cleaned_progress)

    removed_count = original_count - cleaned_count


    print(
        f"Original recorded change points : "
        f"{original_count}"
    )

    print(
        f"Redundant entries removed       : "
        f"{removed_count}"
    )

    print(
        f"Final change points              : "
        f"{cleaned_count}"
    )


    print("\nCLEANED CHANGE POINTS")

    print("-" * 75)


    for _, row in cleaned_progress.iterrows():

        print(
            f"Frame "
            f"{int(row[frame_column]):5d}"
            f" -> "
            f"{row[state_column]}"
        )


    # ========================================================================
    # SAVE CLEANED PROGRESS FILE
    # ========================================================================

    cleaned_progress.to_csv(
        CLEANED_PROGRESS_FILE,
        index=False,
    )


    print("\nCleaned progress file saved to:")

    print(CLEANED_PROGRESS_FILE)


    # ========================================================================
    # REGENERATE FRAME-LEVEL GROUND TRUTH
    # ========================================================================

    print_header(
        "REGENERATING FRAME-LEVEL GROUND TRUTH"
    )


    final_ground_truth = create_frame_level_ground_truth(

        cleaned_progress=cleaned_progress,

        frame_column=frame_column,

        state_column=state_column,

        total_frames=total_frames,

        fps=fps,

        video_filename=VALIDATION_VIDEO.name,
    )


    final_ground_truth.to_csv(
        FINAL_GROUND_TRUTH_FILE,
        index=False,
    )


    print(
        f"Ground-truth rows created : "
        f"{len(final_ground_truth)}"
    )


    # ========================================================================
    # FINAL VERIFICATION
    # ========================================================================

    print_header(
        "VERIFYING REGENERATED GROUND TRUTH"
    )


    row_count_matches = (
        len(final_ground_truth)
        == total_frames
    )


    first_frame_correct = (
        int(final_ground_truth["frame_number"].iloc[0])
        == 0
    )


    final_frame_correct = (
        int(final_ground_truth["frame_number"].iloc[-1])
        == total_frames - 1
    )


    duplicate_frames = int(
        final_ground_truth["frame_number"]
        .duplicated()
        .sum()
    )


    expected_frames = set(
        range(total_frames)
    )


    actual_frames = set(
        final_ground_truth["frame_number"].tolist()
    )


    missing_frames = sorted(
        expected_frames - actual_frames
    )


    unknown_final_states = sorted(
        set(final_ground_truth["assembly_state_gt"])
        - set(VALID_STATES.keys())
    )


    print(
        f"Video frames               : "
        f"{total_frames}"
    )

    print(
        f"Ground-truth rows          : "
        f"{len(final_ground_truth)}"
    )

    print(
        f"Cleaned change points      : "
        f"{cleaned_count}"
    )

    print(
        f"First frame is 0           : "
        f"{'YES' if first_frame_correct else 'NO'}"
    )

    print(
        f"Final frame is correct     : "
        f"{'YES' if final_frame_correct else 'NO'}"
    )

    print(
        f"Duplicate frame rows       : "
        f"{duplicate_frames}"
    )

    print(
        f"Missing frame rows         : "
        f"{len(missing_frames)}"
    )

    print(
        f"Unknown final states       : "
        f"{len(unknown_final_states)}"
    )

    print(
        f"Video / ground-truth match : "
        f"{'YES' if row_count_matches else 'NO'}"
    )


    # ========================================================================
    # STATE DISTRIBUTION
    # ========================================================================

    print_header(
        "GROUND-TRUTH STATE DISTRIBUTION"
    )


    state_counts = (
        final_ground_truth["assembly_state_gt"]
        .value_counts()
        .reindex(
            VALID_STATES.keys(),
            fill_value=0,
        )
    )


    for state_name, count in state_counts.items():

        percentage = (
            count / total_frames * 100
        )

        print(
            f"{state_name:20s}: "
            f"{count:5d} frames "
            f"({percentage:6.2f}%)"
        )


    # ========================================================================
    # FINAL STATUS
    # ========================================================================

    verification_passed = all(
        [
            row_count_matches,
            first_frame_correct,
            final_frame_correct,
            duplicate_frames == 0,
            len(missing_frames) == 0,
            len(unknown_final_states) == 0,
        ]
    )


    print_header(
        "FINAL STATUS"
    )


    if verification_passed:

        print(
            "STATUS: GROUND-TRUTH CLEANING COMPLETED SUCCESSFULLY"
        )

        print("\nOriginal progress file preserved.")

        print(
            f"Original change points : "
            f"{original_count}"
        )

        print(
            f"Redundant entries removed : "
            f"{removed_count}"
        )

        print(
            f"Final change points : "
            f"{cleaned_count}"
        )

        print(
            f"Frame-level rows : "
            f"{len(final_ground_truth)}"
        )

        print("\nNext stage:")

        print(
            "Inspect the cleaned change-point list and verify that the "
            "remaining transitions match the physical events in the video."
        )

        print(
            "After that, compare frame-level ground truth against the "
            "saved validation predictions and calculate quantitative metrics."
        )


    else:

        raise RuntimeError(
            "\nGROUND-TRUTH VERIFICATION FAILED.\n"
            "Inspect the generated files before continuing."
        )


# ============================================================================
# RUN PROGRAM
# ============================================================================

if __name__ == "__main__":
    main()