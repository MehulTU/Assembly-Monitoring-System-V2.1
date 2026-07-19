"""
===========================================================================
AI-SUPPORTED ERGONOMIC AND PRODUCTIVITY ANALYSIS
VISION-BASED ASSEMBLY MONITORING PROTOTYPE V2
===========================================================================

PROGRAM:
    analyze_validation_errors.py

HOW TO RUN:
    python scripts\analyze_validation_errors.py

PURPOSE:
    Diagnose the quantitative validation errors produced by Prototype V2.

THIS PROGRAM:
    1. Loads frame-level validation comparison results.
    2. Loads raw/stable detection timeline.
    3. Loads state-machine timeline.
    4. Diagnoses every incorrect frame.
    5. Classifies errors into:
           DETECTION_ERROR
           TEMPORAL_FILTER_ERROR
           STATE_MACHINE_ERROR
           TRANSITION_BOUNDARY_ERROR
           UNRESOLVED_ERROR
    6. Identifies the object contributing to each error.
    7. Groups consecutive errors into diagnostic segments.
    8. Analyzes predicted transitions.
    9. Saves detailed CSV reports.
    10. Saves a text summary.
    11. Saves a diagnostic plot.

OUTPUT FOLDER:

    datasets\\validation\\results\\validation_01\\evaluation

OUTPUT FILES:

    validation_error_diagnosis.csv
    validation_error_segment_diagnosis.csv
    validation_error_diagnosis_summary.csv
    validation_object_error_summary.csv
    validation_false_transition_diagnosis.csv
    validation_error_diagnosis_summary.txt
    validation_error_diagnosis.png

===========================================================================
"""

from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt


# ===========================================================================
# PROJECT PATHS
# ===========================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_NAME = "validation_01"

VALIDATION_RESULT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "validation"
    / "results"
    / EXPERIMENT_NAME
)

EVALUATION_DIR = VALIDATION_RESULT_DIR / "evaluation"

DETECTION_TIMELINE_PATH = (
    VALIDATION_RESULT_DIR / "detection_timeline.csv"
)

STATE_TIMELINE_PATH = (
    VALIDATION_RESULT_DIR / "state_timeline.csv"
)

FRAME_COMPARISON_PATH = (
    EVALUATION_DIR / "validation_frame_comparison.csv"
)

ERROR_SEGMENTS_PATH = (
    EVALUATION_DIR / "validation_error_segments.csv"
)

TRANSITION_COMPARISON_PATH = (
    EVALUATION_DIR / "validation_transition_comparison.csv"
)


# ===========================================================================
# OUTPUT PATHS
# ===========================================================================

FRAME_DIAGNOSIS_PATH = (
    EVALUATION_DIR / "validation_error_diagnosis.csv"
)

SEGMENT_DIAGNOSIS_PATH = (
    EVALUATION_DIR / "validation_error_segment_diagnosis.csv"
)

CAUSE_SUMMARY_PATH = (
    EVALUATION_DIR / "validation_error_diagnosis_summary.csv"
)

OBJECT_SUMMARY_PATH = (
    EVALUATION_DIR / "validation_object_error_summary.csv"
)

FALSE_TRANSITION_PATH = (
    EVALUATION_DIR / "validation_false_transition_diagnosis.csv"
)

TEXT_SUMMARY_PATH = (
    EVALUATION_DIR / "validation_error_diagnosis_summary.txt"
)

PLOT_PATH = (
    EVALUATION_DIR / "validation_error_diagnosis.png"
)


# ===========================================================================
# CONFIGURATION
# ===========================================================================

TRANSITION_BOUNDARY_TOLERANCE_FRAMES = 10

DEFAULT_FPS = 30.0


# ===========================================================================
# VALID ASSEMBLY STATES
# ===========================================================================

VALID_STATES = {
    "EMPTY",
    "MARKER_ONLY",
    "POWER_ADAPTER_ONLY",
    "BOTH_PRESENT",
}


# ===========================================================================
# BASIC HELPER FUNCTIONS
# ===========================================================================


def print_header(title):
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


def find_column(dataframe, possible_names, required=True):

    column_lookup = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for name in possible_names:

        normalized_name = str(name).strip().lower()

        if normalized_name in column_lookup:
            return column_lookup[normalized_name]

    if required:

        raise ValueError(
            "\nCould not find any of these columns:\n"
            f"{possible_names}\n\n"
            "Available columns:\n"
            f"{list(dataframe.columns)}"
        )

    return None


def normalize_boolean(value):

    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    value = str(value).strip().lower()

    return value in {
        "true",
        "1",
        "yes",
        "y",
        "present",
        "detected",
    }


def normalize_state(value):

    return str(value).strip().upper()


def state_to_objects(state):

    state = normalize_state(state)

    if state == "EMPTY":
        return False, False

    if state == "MARKER_ONLY":
        return True, False

    if state == "POWER_ADAPTER_ONLY":
        return False, True

    if state == "BOTH_PRESENT":
        return True, True

    raise ValueError(
        f"Unknown assembly state: {state}"
    )


# ===========================================================================
# COLUMN DETECTION
# ===========================================================================


def get_timeline_columns(detection_df):

    frame_col = find_column(
        detection_df,
        [
            "frame_number",
            "frame",
            "frame_id",
        ],
    )

    raw_marker_col = find_column(
        detection_df,
        [
            "marker_raw_present",
            "raw_marker_present",
            "marker_present_raw",
            "raw_marker_positive",
        ],
    )

    stable_marker_col = find_column(
        detection_df,
        [
            "marker_stable_present",
            "stable_marker_present",
            "marker_present_stable",
            "filtered_marker_present",
        ],
    )

    raw_adapter_col = find_column(
        detection_df,
        [
            "power_adapter_raw_present",
            "raw_power_adapter_present",
            "raw_adapter_present",
            "power_adapter_present_raw",
            "adapter_raw_present",
            "raw_adapter_positive",
        ],
    )

    stable_adapter_col = find_column(
        detection_df,
        [
            "power_adapter_stable_present",
            "stable_power_adapter_present",
            "stable_adapter_present",
            "power_adapter_present_stable",
            "adapter_stable_present",
            "filtered_adapter_present",
        ],
    )

    return {
        "frame": frame_col,
        "raw_marker": raw_marker_col,
        "stable_marker": stable_marker_col,
        "raw_adapter": raw_adapter_col,
        "stable_adapter": stable_adapter_col,
    }


def get_state_columns(state_df):

    frame_col = find_column(
        state_df,
        [
            "frame_number",
            "frame",
            "frame_id",
        ],
    )

    state_col = find_column(
        state_df,
        [
            "assembly_state",
            "predicted_state",
            "state",
            "stable_state",
        ],
    )

    return {
        "frame": frame_col,
        "state": state_col,
    }


def get_comparison_columns(comparison_df):

    frame_col = find_column(
        comparison_df,
        [
            "frame_number",
            "frame",
            "frame_id",
        ],
    )

    gt_state_col = find_column(
        comparison_df,
        [
            "assembly_state_gt",
            "ground_truth_state",
            "gt_state",
            "true_state",
            "actual_state",
        ],
    )

    predicted_state_col = find_column(
        comparison_df,
        [
            "assembly_state_pred",
            "predicted_state",
            "prediction_state",
            "assembly_state",
            "system_state",
        ],
    )

    correct_col = find_column(
        comparison_df,
        [
            "state_correct",
            "correct_state",
            "is_correct",
            "correct",
        ],
        required=False,
    )

    return {
        "frame": frame_col,
        "gt_state": gt_state_col,
        "predicted_state": predicted_state_col,
        "correct": correct_col,
    }


# ===========================================================================
# GROUND-TRUTH TRANSITION FUNCTIONS
# ===========================================================================


def get_ground_truth_transition_frames(
    frame_comparison_df,
    frame_col,
    gt_state_col,
):

    sorted_df = (
        frame_comparison_df
        .sort_values(frame_col)
        .copy()
    )

    sorted_df[gt_state_col] = (
        sorted_df[gt_state_col]
        .apply(normalize_state)
    )

    previous_state = sorted_df[gt_state_col].shift(1)

    transition_mask = (
        sorted_df[gt_state_col]
        != previous_state
    )

    transitions = (
        sorted_df.loc[
            transition_mask,
            frame_col,
        ]
        .astype(int)
        .tolist()
    )

    # First row is initialization, not a physical transition.
    if transitions:
        transitions = transitions[1:]

    return sorted(transitions)


def distance_to_nearest_transition(
    frame_number,
    transition_frames,
):

    if not transition_frames:
        return None

    return min(
        abs(
            int(frame_number)
            - int(transition_frame)
        )
        for transition_frame in transition_frames
    )


# ===========================================================================
# ERROR CLASSIFICATION
# ===========================================================================


def classify_error(
    gt_marker,
    gt_adapter,
    raw_marker,
    raw_adapter,
    stable_marker,
    stable_adapter,
    predicted_marker,
    predicted_adapter,
    near_transition,
):

    raw_correct = (
        raw_marker == gt_marker
        and raw_adapter == gt_adapter
    )

    stable_correct = (
        stable_marker == gt_marker
        and stable_adapter == gt_adapter
    )

    predicted_correct = (
        predicted_marker == gt_marker
        and predicted_adapter == gt_adapter
    )

    # Stable object signals are correct,
    # but final assembly state is wrong.
    if stable_correct and not predicted_correct:
        return "STATE_MACHINE_ERROR"

    # Raw detections are correct,
    # but temporal filtering changed them incorrectly.
    if raw_correct and not stable_correct:
        return "TEMPORAL_FILTER_ERROR"

    # Raw detection error survives into stable signals.
    if not raw_correct and not stable_correct:
        return "DETECTION_ERROR"

    # Remaining unexplained error near a GT transition.
    if near_transition:
        return "TRANSITION_BOUNDARY_ERROR"

    return "UNRESOLVED_ERROR"


def identify_responsible_object(
    gt_marker,
    gt_adapter,
    raw_marker,
    raw_adapter,
    stable_marker,
    stable_adapter,
    predicted_marker,
    predicted_adapter,
):

    marker_error = (
        raw_marker != gt_marker
        or stable_marker != gt_marker
        or predicted_marker != gt_marker
    )

    adapter_error = (
        raw_adapter != gt_adapter
        or stable_adapter != gt_adapter
        or predicted_adapter != gt_adapter
    )

    if marker_error and adapter_error:
        return "BOTH_OBJECTS"

    if marker_error:
        return "MARKER"

    if adapter_error:
        return "POWER_ADAPTER"

    return "NONE"


# ===========================================================================
# ERROR SEGMENT ANALYSIS
# ===========================================================================


def summarize_segment(segment_id, segment_df):

    start_frame = int(
        segment_df["frame_number"].min()
    )

    end_frame = int(
        segment_df["frame_number"].max()
    )

    frame_count = len(segment_df)

    fps = float(segment_df["fps"].iloc[0])

    duration_seconds = (
        frame_count / fps
        if fps > 0
        else 0.0
    )

    transition_distances = (
        segment_df["transition_distance_frames"]
        .dropna()
    )

    if transition_distances.empty:
        minimum_transition_distance = None
    else:
        minimum_transition_distance = int(
            transition_distances.min()
        )

    return {
        "segment_id": segment_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_count": frame_count,
        "duration_seconds": duration_seconds,

        "ground_truth_state":
            segment_df["ground_truth_state"].iloc[0],

        "predicted_state":
            segment_df["predicted_state"].iloc[0],

        "diagnosed_cause":
            segment_df["diagnosed_cause"].iloc[0],

        "responsible_object":
            segment_df["responsible_object"].iloc[0],

        "near_ground_truth_transition":
            bool(
                segment_df[
                    "near_ground_truth_transition"
                ].any()
            ),

        "minimum_transition_distance_frames":
            minimum_transition_distance,

        "raw_marker_accuracy":
            segment_df[
                "raw_marker_correct"
            ].mean(),

        "stable_marker_accuracy":
            segment_df[
                "stable_marker_correct"
            ].mean(),

        "raw_adapter_accuracy":
            segment_df[
                "raw_adapter_correct"
            ].mean(),

        "stable_adapter_accuracy":
            segment_df[
                "stable_adapter_correct"
            ].mean(),
    }


def create_segment_diagnosis(frame_diagnosis_df):

    if frame_diagnosis_df.empty:
        return pd.DataFrame()

    df = (
        frame_diagnosis_df
        .sort_values("frame_number")
        .copy()
    )

    segment_rows = []

    current_segment = []

    segment_id = 1

    previous_row = None

    for _, row in df.iterrows():

        start_new_segment = False

        if previous_row is None:

            start_new_segment = True

        else:

            if (
                int(row["frame_number"])
                !=
                int(previous_row["frame_number"]) + 1
            ):
                start_new_segment = True

            elif (
                row["ground_truth_state"]
                != previous_row["ground_truth_state"]
            ):
                start_new_segment = True

            elif (
                row["predicted_state"]
                != previous_row["predicted_state"]
            ):
                start_new_segment = True

            elif (
                row["diagnosed_cause"]
                != previous_row["diagnosed_cause"]
            ):
                start_new_segment = True

            elif (
                row["responsible_object"]
                != previous_row["responsible_object"]
            ):
                start_new_segment = True

        if start_new_segment and current_segment:

            segment_df = pd.DataFrame(
                current_segment
            )

            segment_rows.append(
                summarize_segment(
                    segment_id,
                    segment_df,
                )
            )

            segment_id += 1

            current_segment = []

        current_segment.append(
            row.to_dict()
        )

        previous_row = row

    if current_segment:

        segment_df = pd.DataFrame(
            current_segment
        )

        segment_rows.append(
            summarize_segment(
                segment_id,
                segment_df,
            )
        )

    return pd.DataFrame(segment_rows)


# ===========================================================================
# PREDICTED TRANSITION ANALYSIS
# ===========================================================================


def analyze_false_transitions(
    frame_comparison_df,
    frame_diagnosis_df,
    transition_frames,
):

    comparison_columns = get_comparison_columns(
        frame_comparison_df
    )

    frame_col = comparison_columns["frame"]

    predicted_state_col = (
        comparison_columns["predicted_state"]
    )

    df = (
        frame_comparison_df
        .sort_values(frame_col)
        .copy()
    )

    df[predicted_state_col] = (
        df[predicted_state_col]
        .apply(normalize_state)
    )

    df["previous_predicted_state"] = (
        df[predicted_state_col].shift(1)
    )

    predicted_transitions = df[
        df[predicted_state_col]
        != df["previous_predicted_state"]
    ].copy()

    # Remove initialization frame.
    if not predicted_transitions.empty:
        predicted_transitions = (
            predicted_transitions.iloc[1:].copy()
        )

    rows = []

    for _, row in predicted_transitions.iterrows():

        frame_number = int(row[frame_col])

        distance = distance_to_nearest_transition(
            frame_number,
            transition_frames,
        )

        matched_gt_transition = (
            distance is not None
            and
            distance
            <= TRANSITION_BOUNDARY_TOLERANCE_FRAMES
        )

        if matched_gt_transition:

            transition_type = (
                "MATCHED_GT_TRANSITION"
            )

            likely_cause = (
                "EXPECTED_TRANSITION"
            )

        else:

            transition_type = (
                "FALSE_PREDICTED_TRANSITION"
            )

            nearby_error_rows = frame_diagnosis_df[
                (
                    frame_diagnosis_df[
                        "frame_number"
                    ]
                    >=
                    frame_number
                    -
                    TRANSITION_BOUNDARY_TOLERANCE_FRAMES
                )
                &
                (
                    frame_diagnosis_df[
                        "frame_number"
                    ]
                    <=
                    frame_number
                    +
                    TRANSITION_BOUNDARY_TOLERANCE_FRAMES
                )
            ]

            if nearby_error_rows.empty:

                likely_cause = (
                    "UNRESOLVED_TRANSITION_CAUSE"
                )

            else:

                cause_counts = (
                    nearby_error_rows[
                        "diagnosed_cause"
                    ]
                    .value_counts()
                )

                likely_cause = (
                    cause_counts.index[0]
                )

        rows.append(
            {
                "predicted_transition_frame":
                    frame_number,

                "previous_predicted_state":
                    row[
                        "previous_predicted_state"
                    ],

                "new_predicted_state":
                    row[predicted_state_col],

                "nearest_gt_transition_distance_frames":
                    distance,

                "matched_ground_truth_transition":
                    matched_gt_transition,

                "transition_type":
                    transition_type,

                "likely_cause":
                    likely_cause,
            }
        )

    return pd.DataFrame(rows)


# ===========================================================================
# DIAGNOSTIC PLOT
# ===========================================================================


def create_diagnostic_plot(cause_summary_df):

    if cause_summary_df.empty:

        print(
            "No error diagnosis data available "
            "for plot."
        )

        return

    plot_df = (
        cause_summary_df
        .sort_values(
            "error_frames",
            ascending=False,
        )
    )

    plt.figure(figsize=(11, 7))

    bars = plt.bar(
        plot_df["diagnosed_cause"],
        plot_df["error_frames"],
    )

    plt.xlabel(
        "Diagnosed Error Cause"
    )

    plt.ylabel(
        "Incorrect Frames"
    )

    plt.title(
        f"{EXPERIMENT_NAME} - "
        "Validation Error Diagnosis"
    )

    plt.xticks(
        rotation=25,
        ha="right",
    )

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x()
            +
            bar.get_width() / 2,

            height,

            f"{int(height)}",

            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    plt.savefig(
        PLOT_PATH,
        dpi=300,
    )

    plt.close()


# ===========================================================================
# MAIN PROGRAM
# ===========================================================================


def main():

    print_header(
        "PROTOTYPE V2 - VALIDATION ERROR DIAGNOSIS"
    )

    print(
        "This program diagnoses the incorrect validation frames\n"
        "and determines whether the errors originate mainly from:\n\n"
        "  1. YOLO object detection\n"
        "  2. Temporal filtering\n"
        "  3. State-machine logic\n"
        "  4. Ground-truth transition-boundary uncertainty\n"
        "  5. Unresolved pipeline behavior\n"
    )

    print("Experiment:")
    print(EXPERIMENT_NAME)

    print()
    print("Evaluation directory:")
    print(EVALUATION_DIR)


    # =======================================================================
    # VERIFY INPUT FILES
    # =======================================================================

    print_header(
        "VERIFYING INPUT FILES"
    )

    required_files = [
        DETECTION_TIMELINE_PATH,
        STATE_TIMELINE_PATH,
        FRAME_COMPARISON_PATH,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                "Required input file not found:\n"
                f"{path}"
            )

        print(f"Found: {path.name}")

    if ERROR_SEGMENTS_PATH.exists():

        print(
            f"Found: "
            f"{ERROR_SEGMENTS_PATH.name}"
        )

    else:

        print(
            "Optional file not found: "
            f"{ERROR_SEGMENTS_PATH.name}"
        )

    if TRANSITION_COMPARISON_PATH.exists():

        print(
            f"Found: "
            f"{TRANSITION_COMPARISON_PATH.name}"
        )

    else:

        print(
            "Optional file not found: "
            f"{TRANSITION_COMPARISON_PATH.name}"
        )

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # =======================================================================
    # LOAD DATA
    # =======================================================================

    print_header(
        "LOADING VALIDATION DATA"
    )

    detection_df = pd.read_csv(
        DETECTION_TIMELINE_PATH
    )

    state_df = pd.read_csv(
        STATE_TIMELINE_PATH
    )

    comparison_df = pd.read_csv(
        FRAME_COMPARISON_PATH
    )

    print(
        f"Detection timeline rows : "
        f"{len(detection_df)}"
    )

    print(
        f"State timeline rows     : "
        f"{len(state_df)}"
    )

    print(
        f"Comparison rows         : "
        f"{len(comparison_df)}"
    )


    # =======================================================================
    # DETECT COLUMNS
    # =======================================================================

    print_header(
        "DETECTING DATA COLUMNS"
    )

    detection_columns = (
        get_timeline_columns(
            detection_df
        )
    )

    state_columns = (
        get_state_columns(
            state_df
        )
    )

    comparison_columns = (
        get_comparison_columns(
            comparison_df
        )
    )

    print(
        "Detection timeline columns:"
    )

    for key, value in (
        detection_columns.items()
    ):

        print(
            f"  {key:16s}: {value}"
        )

    print()
    print(
        "State timeline columns:"
    )

    for key, value in (
        state_columns.items()
    ):

        print(
            f"  {key:16s}: {value}"
        )

    print()
    print(
        "Comparison columns:"
    )

    for key, value in (
        comparison_columns.items()
    ):

        print(
            f"  {key:16s}: {value}"
        )


    # =======================================================================
    # PREPARE DATA
    # =======================================================================

    print_header(
        "PREPARING FRAME-LEVEL DATA"
    )

    detection_work = detection_df[
        [
            detection_columns["frame"],
            detection_columns["raw_marker"],
            detection_columns["stable_marker"],
            detection_columns["raw_adapter"],
            detection_columns["stable_adapter"],
        ]
    ].copy()

    detection_work.columns = [
        "frame_number",
        "raw_marker_present",
        "stable_marker_present",
        "raw_adapter_present",
        "stable_adapter_present",
    ]

    state_work = state_df[
        [
            state_columns["frame"],
            state_columns["state"],
        ]
    ].copy()

    state_work.columns = [
        "frame_number",
        "state_machine_state",
    ]

    comparison_columns_to_keep = [
        comparison_columns["frame"],
        comparison_columns["gt_state"],
        comparison_columns["predicted_state"],
    ]

    if comparison_columns["correct"] is not None:

        comparison_columns_to_keep.append(
            comparison_columns["correct"]
        )

    comparison_work = comparison_df[
        comparison_columns_to_keep
    ].copy()

    rename_mapping = {
        comparison_columns["frame"]:
            "frame_number",

        comparison_columns["gt_state"]:
            "ground_truth_state",

        comparison_columns["predicted_state"]:
            "predicted_state",
    }

    if comparison_columns["correct"] is not None:

        rename_mapping[
            comparison_columns["correct"]
        ] = "state_correct"

    comparison_work = (
        comparison_work.rename(
            columns=rename_mapping
        )
    )

    comparison_work[
        "ground_truth_state"
    ] = (
        comparison_work[
            "ground_truth_state"
        ].apply(normalize_state)
    )

    comparison_work[
        "predicted_state"
    ] = (
        comparison_work[
            "predicted_state"
        ].apply(normalize_state)
    )

    state_work[
        "state_machine_state"
    ] = (
        state_work[
            "state_machine_state"
        ].apply(normalize_state)
    )

    merged_df = comparison_work.merge(
        detection_work,
        on="frame_number",
        how="left",
        validate="one_to_one",
    )

    merged_df = merged_df.merge(
        state_work,
        on="frame_number",
        how="left",
        validate="one_to_one",
    )

    print(
        f"Merged frame rows: "
        f"{len(merged_df)}"
    )

    missing_detection_rows = (
        merged_df[
            "raw_marker_present"
        ].isna().sum()
    )

    missing_state_rows = (
        merged_df[
            "state_machine_state"
        ].isna().sum()
    )

    print(
        "Frames missing detection "
        "timeline data : "
        f"{missing_detection_rows}"
    )

    print(
        "Frames missing state "
        "timeline data     : "
        f"{missing_state_rows}"
    )

    if missing_detection_rows > 0:

        raise ValueError(
            "Some validation frames are missing "
            "detection timeline data."
        )

    if missing_state_rows > 0:

        raise ValueError(
            "Some validation frames are missing "
            "state timeline data."
        )


    # =======================================================================
    # VERIFY STATE-MACHINE CONSISTENCY
    # =======================================================================

    print_header(
        "VERIFYING STATE-MACHINE OUTPUT CONSISTENCY"
    )

    merged_df[
        "state_machine_matches_saved_prediction"
    ] = (
        merged_df[
            "state_machine_state"
        ]
        ==
        merged_df[
            "predicted_state"
        ]
    )

    state_output_mismatches = int(
        (
            ~merged_df[
                "state_machine_matches_saved_prediction"
            ]
        ).sum()
    )

    print(
        "State timeline / saved prediction "
        "mismatches: "
        f"{state_output_mismatches}"
    )

    if state_output_mismatches > 0:

        print()

        print(
            "WARNING: state_timeline.csv and "
            "validation_frame_comparison.csv "
            "contain different predicted states."
        )


    # =======================================================================
    # IDENTIFY INCORRECT FRAMES
    # =======================================================================

    print_header(
        "IDENTIFYING INCORRECT FRAMES"
    )

    merged_df[
        "calculated_state_correct"
    ] = (
        merged_df[
            "ground_truth_state"
        ]
        ==
        merged_df[
            "predicted_state"
        ]
    )

    incorrect_df = merged_df[
        ~merged_df[
            "calculated_state_correct"
        ]
    ].copy()

    print(
        f"Evaluated frames : "
        f"{len(merged_df)}"
    )

    print(
        f"Incorrect frames : "
        f"{len(incorrect_df)}"
    )

    if incorrect_df.empty:

        print()
        print(
            "No incorrect validation frames "
            "were found."
        )

        print(
            "Nothing to diagnose."
        )

        sys.exit(0)


    # =======================================================================
    # GROUND-TRUTH TRANSITIONS
    # =======================================================================

    print_header(
        "GROUND-TRUTH TRANSITION ANALYSIS"
    )

    # IMPORTANT CORRECTION:
    #
    # The function requires:
    #   1. comparison dataframe
    #   2. detected frame column
    #   3. detected GT state column

    transition_frames = (
        get_ground_truth_transition_frames(
            frame_comparison_df=comparison_df,
            frame_col=comparison_columns["frame"],
            gt_state_col=comparison_columns["gt_state"],
        )
    )

    print(
        "Ground-truth transitions detected: "
        f"{len(transition_frames)}"
    )

    print(
        "Transition frames:"
    )

    for frame in transition_frames:

        print(
            f"  {frame}"
        )


    # =======================================================================
    # DIAGNOSE INCORRECT FRAMES
    # =======================================================================

    print_header(
        "DIAGNOSING INCORRECT FRAMES"
    )

    diagnosis_rows = []

    for _, row in incorrect_df.iterrows():

        frame_number = int(
            row["frame_number"]
        )

        ground_truth_state = (
            normalize_state(
                row["ground_truth_state"]
            )
        )

        predicted_state = (
            normalize_state(
                row["predicted_state"]
            )
        )

        if (
            ground_truth_state
            not in VALID_STATES
        ):

            raise ValueError(
                "Unknown ground-truth state: "
                f"{ground_truth_state}"
            )

        if (
            predicted_state
            not in VALID_STATES
        ):

            raise ValueError(
                "Unknown predicted state: "
                f"{predicted_state}"
            )

        gt_marker, gt_adapter = (
            state_to_objects(
                ground_truth_state
            )
        )

        (
            predicted_marker,
            predicted_adapter,
        ) = state_to_objects(
            predicted_state
        )

        raw_marker = normalize_boolean(
            row["raw_marker_present"]
        )

        stable_marker = normalize_boolean(
            row["stable_marker_present"]
        )

        raw_adapter = normalize_boolean(
            row["raw_adapter_present"]
        )

        stable_adapter = normalize_boolean(
            row["stable_adapter_present"]
        )

        transition_distance = (
            distance_to_nearest_transition(
                frame_number,
                transition_frames,
            )
        )

        near_transition = (
            transition_distance is not None
            and
            transition_distance
            <=
            TRANSITION_BOUNDARY_TOLERANCE_FRAMES
        )

        diagnosed_cause = classify_error(
            gt_marker=gt_marker,
            gt_adapter=gt_adapter,
            raw_marker=raw_marker,
            raw_adapter=raw_adapter,
            stable_marker=stable_marker,
            stable_adapter=stable_adapter,
            predicted_marker=predicted_marker,
            predicted_adapter=predicted_adapter,
            near_transition=near_transition,
        )

        responsible_object = (
            identify_responsible_object(
                gt_marker=gt_marker,
                gt_adapter=gt_adapter,
                raw_marker=raw_marker,
                raw_adapter=raw_adapter,
                stable_marker=stable_marker,
                stable_adapter=stable_adapter,
                predicted_marker=predicted_marker,
                predicted_adapter=predicted_adapter,
            )
        )

        diagnosis_rows.append(
            {
                "frame_number":
                    frame_number,

                "ground_truth_state":
                    ground_truth_state,

                "predicted_state":
                    predicted_state,

                "state_machine_state":
                    row[
                        "state_machine_state"
                    ],

                "gt_marker_present":
                    gt_marker,

                "gt_adapter_present":
                    gt_adapter,

                "raw_marker_present":
                    raw_marker,

                "stable_marker_present":
                    stable_marker,

                "raw_adapter_present":
                    raw_adapter,

                "stable_adapter_present":
                    stable_adapter,

                "predicted_marker_present":
                    predicted_marker,

                "predicted_adapter_present":
                    predicted_adapter,

                "raw_marker_correct":
                    raw_marker == gt_marker,

                "stable_marker_correct":
                    stable_marker == gt_marker,

                "raw_adapter_correct":
                    raw_adapter == gt_adapter,

                "stable_adapter_correct":
                    stable_adapter == gt_adapter,

                "transition_distance_frames":
                    transition_distance,

                "near_ground_truth_transition":
                    near_transition,

                "diagnosed_cause":
                    diagnosed_cause,

                "responsible_object":
                    responsible_object,

                "fps":
                    DEFAULT_FPS,
            }
        )

    frame_diagnosis_df = pd.DataFrame(
        diagnosis_rows
    )

    print(
        "Diagnosed incorrect frames: "
        f"{len(frame_diagnosis_df)}"
    )


    # =======================================================================
    # CAUSE SUMMARY
    # =======================================================================

    print_header(
        "ERROR CAUSE SUMMARY"
    )

    cause_summary_df = (
        frame_diagnosis_df
        .groupby(
            "diagnosed_cause"
        )
        .size()
        .reset_index(
            name="error_frames"
        )
        .sort_values(
            "error_frames",
            ascending=False,
        )
    )

    cause_summary_df[
        "percentage_of_error_frames"
    ] = (
        cause_summary_df[
            "error_frames"
        ]
        /
        len(frame_diagnosis_df)
        *
        100.0
    )

    print(
        cause_summary_df.to_string(
            index=False,

            formatters={
                "percentage_of_error_frames":
                    lambda value:
                    f"{value:.2f}%"
            },
        )
    )


    # =======================================================================
    # RESPONSIBLE OBJECT SUMMARY
    # =======================================================================

    print_header(
        "RESPONSIBLE OBJECT SUMMARY"
    )

    object_summary_df = (
        frame_diagnosis_df
        .groupby(
            "responsible_object"
        )
        .size()
        .reset_index(
            name="error_frames"
        )
        .sort_values(
            "error_frames",
            ascending=False,
        )
    )

    object_summary_df[
        "percentage_of_error_frames"
    ] = (
        object_summary_df[
            "error_frames"
        ]
        /
        len(frame_diagnosis_df)
        *
        100.0
    )

    print(
        object_summary_df.to_string(
            index=False,

            formatters={
                "percentage_of_error_frames":
                    lambda value:
                    f"{value:.2f}%"
            },
        )
    )


    # =======================================================================
    # ERROR SEGMENTS
    # =======================================================================

    print_header(
        "CREATING ERROR SEGMENT DIAGNOSIS"
    )

    segment_diagnosis_df = (
        create_segment_diagnosis(
            frame_diagnosis_df
        )
    )

    print(
        "Diagnostic error segments: "
        f"{len(segment_diagnosis_df)}"
    )

    if not segment_diagnosis_df.empty:

        segment_cause_counts = (
            segment_diagnosis_df[
                "diagnosed_cause"
            ]
            .value_counts()
        )

        print()
        print(
            "Segments per diagnosed cause:"
        )

        for (
            cause,
            count,
        ) in (
            segment_cause_counts.items()
        ):

            print(
                f"  {cause:30s}: "
                f"{count}"
            )


    # =======================================================================
    # PREDICTED TRANSITIONS
    # =======================================================================

    print_header(
        "ANALYZING PREDICTED TRANSITIONS"
    )

    false_transition_df = (
        analyze_false_transitions(
            frame_comparison_df=comparison_df,
            frame_diagnosis_df=frame_diagnosis_df,
            transition_frames=transition_frames,
        )
    )

    total_predicted_transitions = (
        len(false_transition_df)
    )

    if false_transition_df.empty:

        matched_transitions = 0

    else:

        matched_transitions = int(
            false_transition_df[
                "matched_ground_truth_transition"
            ].sum()
        )

    false_transitions = (
        total_predicted_transitions
        -
        matched_transitions
    )

    print(
        "Predicted transitions analyzed : "
        f"{total_predicted_transitions}"
    )

    print(
        "Transitions near GT transition : "
        f"{matched_transitions}"
    )

    print(
        "False predicted transitions    : "
        f"{false_transitions}"
    )

    if not false_transition_df.empty:

        false_only = false_transition_df[
            false_transition_df[
                "transition_type"
            ]
            ==
            "FALSE_PREDICTED_TRANSITION"
        ]

        if not false_only.empty:

            print()
            print(
                "Likely causes of "
                "false transitions:"
            )

            transition_cause_counts = (
                false_only[
                    "likely_cause"
                ]
                .value_counts()
            )

            for (
                cause,
                count,
            ) in (
                transition_cause_counts.items()
            ):

                print(
                    f"  {cause:30s}: "
                    f"{count}"
                )


    # =======================================================================
    # SAVE OUTPUTS
    # =======================================================================

    print_header(
        "SAVING DIAGNOSTIC OUTPUTS"
    )

    frame_diagnosis_df.to_csv(
        FRAME_DIAGNOSIS_PATH,
        index=False,
    )

    segment_diagnosis_df.to_csv(
        SEGMENT_DIAGNOSIS_PATH,
        index=False,
    )

    cause_summary_df.to_csv(
        CAUSE_SUMMARY_PATH,
        index=False,
    )

    object_summary_df.to_csv(
        OBJECT_SUMMARY_PATH,
        index=False,
    )

    false_transition_df.to_csv(
        FALSE_TRANSITION_PATH,
        index=False,
    )

    print(
        f"Saved: "
        f"{FRAME_DIAGNOSIS_PATH.name}"
    )

    print(
        f"Saved: "
        f"{SEGMENT_DIAGNOSIS_PATH.name}"
    )

    print(
        f"Saved: "
        f"{CAUSE_SUMMARY_PATH.name}"
    )

    print(
        f"Saved: "
        f"{OBJECT_SUMMARY_PATH.name}"
    )

    print(
        f"Saved: "
        f"{FALSE_TRANSITION_PATH.name}"
    )


    # =======================================================================
    # CREATE PLOT
    # =======================================================================

    print_header(
        "CREATING DIAGNOSTIC PLOT"
    )

    create_diagnostic_plot(
        cause_summary_df
    )

    print(
        f"Saved: {PLOT_PATH.name}"
    )


    # =======================================================================
    # PRIMARY FAILURE MODE
    # =======================================================================

    print_header(
        "PRIMARY FAILURE MODE"
    )

    primary_cause_row = (
        cause_summary_df.iloc[0]
    )

    primary_cause = (
        primary_cause_row[
            "diagnosed_cause"
        ]
    )

    primary_cause_frames = int(
        primary_cause_row[
            "error_frames"
        ]
    )

    primary_cause_percentage = float(
        primary_cause_row[
            "percentage_of_error_frames"
        ]
    )

    primary_object_row = (
        object_summary_df.iloc[0]
    )

    primary_object = (
        primary_object_row[
            "responsible_object"
        ]
    )

    primary_object_frames = int(
        primary_object_row[
            "error_frames"
        ]
    )

    print(
        f"Primary diagnosed cause  : "
        f"{primary_cause}"
    )

    print(
        f"Error frames             : "
        f"{primary_cause_frames}"
    )

    print(
        f"Percentage of errors     : "
        f"{primary_cause_percentage:.2f}%"
    )

    print(
        f"Primary responsible object: "
        f"{primary_object}"
    )

    print(
        f"Object-related error frames: "
        f"{primary_object_frames}"
    )


    # =======================================================================
    # TEXT SUMMARY
    # =======================================================================

    summary_lines = []

    summary_lines.append(
        "PROTOTYPE V2 - "
        "VALIDATION ERROR DIAGNOSIS SUMMARY"
    )

    summary_lines.append(
        "=" * 75
    )

    summary_lines.append(
        f"Experiment: "
        f"{EXPERIMENT_NAME}"
    )

    summary_lines.append(
        f"Evaluated frames: "
        f"{len(merged_df)}"
    )

    summary_lines.append(
        f"Incorrect frames: "
        f"{len(frame_diagnosis_df)}"
    )

    summary_lines.append(
        f"Ground-truth transitions: "
        f"{len(transition_frames)}"
    )

    summary_lines.append(
        f"Predicted transitions analyzed: "
        f"{total_predicted_transitions}"
    )

    summary_lines.append(
        f"False predicted transitions: "
        f"{false_transitions}"
    )

    summary_lines.append("")

    summary_lines.append(
        "ERROR CAUSE SUMMARY"
    )

    summary_lines.append(
        "-" * 75
    )

    for _, row in (
        cause_summary_df.iterrows()
    ):

        summary_lines.append(
            f"{row['diagnosed_cause']}: "
            f"{int(row['error_frames'])} frames "
            f"("
            f"{row['percentage_of_error_frames']:.2f}%"
            f")"
        )

    summary_lines.append("")

    summary_lines.append(
        "RESPONSIBLE OBJECT SUMMARY"
    )

    summary_lines.append(
        "-" * 75
    )

    for _, row in (
        object_summary_df.iterrows()
    ):

        summary_lines.append(
            f"{row['responsible_object']}: "
            f"{int(row['error_frames'])} frames "
            f"("
            f"{row['percentage_of_error_frames']:.2f}%"
            f")"
        )

    summary_lines.append("")

    summary_lines.append(
        "PRIMARY FAILURE MODE"
    )

    summary_lines.append(
        "-" * 75
    )

    summary_lines.append(
        f"Primary diagnosed cause: "
        f"{primary_cause}"
    )

    summary_lines.append(
        f"Primary responsible object: "
        f"{primary_object}"
    )

    summary_lines.append("")

    summary_lines.append(
        "INTERPRETATION"
    )

    summary_lines.append(
        "-" * 75
    )

    if (
        primary_cause
        ==
        "DETECTION_ERROR"
    ):

        summary_lines.append(
            "The dominant validation failure "
            "originates from frame-level object "
            "detection instability."
        )

        summary_lines.append(
            "The next engineering stage should "
            "inspect detector failures, confidence "
            "scores, object motion, occlusion, and "
            "training-data coverage before modifying "
            "the state machine."
        )

    elif (
        primary_cause
        ==
        "TEMPORAL_FILTER_ERROR"
    ):

        summary_lines.append(
            "The dominant validation failure "
            "originates from temporal filtering "
            "behavior."
        )

        summary_lines.append(
            "The next engineering stage should "
            "improve temporal stabilization and "
            "compare the baseline and improved "
            "filter using the same validation "
            "ground truth."
        )

    elif (
        primary_cause
        ==
        "STATE_MACHINE_ERROR"
    ):

        summary_lines.append(
            "The dominant validation failure "
            "originates from state-machine logic."
        )

        summary_lines.append(
            "The next engineering stage should "
            "inspect state-transition rules and "
            "verify consistency between stable "
            "object signals and final assembly "
            "state predictions."
        )

    elif (
        primary_cause
        ==
        "TRANSITION_BOUNDARY_ERROR"
    ):

        summary_lines.append(
            "The dominant validation errors occur "
            "close to manually annotated "
            "ground-truth transitions."
        )

        summary_lines.append(
            "The next engineering stage should "
            "inspect transition timing, annotation "
            "uncertainty, and pipeline latency."
        )

    else:

        summary_lines.append(
            "The dominant validation errors could "
            "not be fully explained using the "
            "available pipeline signals."
        )

        summary_lines.append(
            "The next engineering stage should "
            "inspect the detailed frame and "
            "segment diagnosis reports."
        )

    summary_text = "\n".join(
        summary_lines
    )

    TEXT_SUMMARY_PATH.write_text(
        summary_text,
        encoding="utf-8",
    )

    print()

    print(
        f"Saved: "
        f"{TEXT_SUMMARY_PATH.name}"
    )


    # =======================================================================
    # FINAL SUMMARY
    # =======================================================================

    print_header(
        "VALIDATION ERROR DIAGNOSIS SUMMARY"
    )

    print(
        f"Experiment                 : "
        f"{EXPERIMENT_NAME}"
    )

    print(
        f"Evaluated frames           : "
        f"{len(merged_df)}"
    )

    print(
        f"Incorrect frames           : "
        f"{len(frame_diagnosis_df)}"
    )

    print(
        f"Diagnostic error segments  : "
        f"{len(segment_diagnosis_df)}"
    )

    print(
        f"Ground-truth transitions   : "
        f"{len(transition_frames)}"
    )

    print(
        f"Predicted transitions      : "
        f"{total_predicted_transitions}"
    )

    print(
        f"False predicted transitions: "
        f"{false_transitions}"
    )

    print(
        f"Primary diagnosed cause    : "
        f"{primary_cause}"
    )

    print(
        f"Primary responsible object : "
        f"{primary_object}"
    )

    print()

    print(
        "Detailed outputs saved to:"
    )

    print(
        EVALUATION_DIR
    )

    print_header(
        "STATUS"
    )

    print(
        "STATUS: VALIDATION ERROR DIAGNOSIS "
        "COMPLETED SUCCESSFULLY"
    )

    print()

    print(
        "Next stage:"
    )

    print(
        "Use the diagnosed error distribution "
        "to choose the first evidence-based "
        "system improvement."
    )

    print()

    print(
        "Do not retrain YOLO or modify the "
        "temporal filter until the diagnosis "
        "results have been inspected."
    )


# ===========================================================================
# RUN PROGRAM
# ===========================================================================


if __name__ == "__main__":
    main()