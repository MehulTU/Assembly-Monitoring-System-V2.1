"""
process_analytics.py

HOW TO RUN THE FILE:
    python scripts\process_analytics.py  (TYPE THIS IN TERMINAL)

Prototype V2 - Process Analytics Report Generator

Purpose:
    Convert the frame-level assembly state timeline and event log into
    compact process-level analytics.

Previous stages produced:

    1. raw_detections.csv
       Raw YOLO detections for every processed video frame.

    2. detection_timeline.csv
       Temporally filtered object-presence information.

    3. state_timeline.csv
       Assembly state assigned to every processed frame.

    4. event_log.csv
       Detected state transitions and process events.

This script summarizes those results into one process analytics report.

The report contains:

    - video duration
    - processed frame count
    - time spent in each assembly state
    - percentage of time spent in each state
    - number of state transitions
    - number of process events
    - sequence violation count
    - assembly completion event count
    - first and final assembly state
    - dominant assembly state
    - whether the expected assembly sequence was observed

Important:
    The current prototype analyzes one video at a time.

    The generated metrics describe the detector/state-machine output.
    They must not automatically be interpreted as true manufacturing
    productivity or ergonomic performance.

Output:

    datasets/analytics/process_analytics_report.csv
"""

from pathlib import Path
import csv
from collections import Counter


# ======================================================================
# PROJECT CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANALYTICS_FOLDER = PROJECT_ROOT / "datasets" / "analytics"

STATE_TIMELINE_FILE = (
    ANALYTICS_FOLDER / "state_timeline.csv"
)

EVENT_LOG_FILE = (
    ANALYTICS_FOLDER / "event_log.csv"
)

OUTPUT_REPORT_FILE = (
    ANALYTICS_FOLDER / "process_analytics_report.csv"
)


# ======================================================================
# STATE DEFINITIONS
# ======================================================================

STATE_EMPTY = "EMPTY"
STATE_MARKER_ONLY = "MARKER_ONLY"
STATE_POWER_ADAPTER_ONLY = "POWER_ADAPTER_ONLY"
STATE_BOTH_PRESENT = "BOTH_PRESENT"

ALL_STATES = [
    STATE_EMPTY,
    STATE_MARKER_ONLY,
    STATE_POWER_ADAPTER_ONLY,
    STATE_BOTH_PRESENT,
]


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def print_header(title):

    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


def text_to_bool(value):

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def safe_percentage(part, total):

    if total == 0:
        return 0.0

    return (part / total) * 100.0


def round_value(value):

    return round(value, 6)


# ======================================================================
# MAIN PROCESS
# ======================================================================

def main():

    print_header(
        "PROTOTYPE V2 - PROCESS ANALYTICS REPORT GENERATOR"
    )

    print(f"State timeline:\n{STATE_TIMELINE_FILE}")

    print()

    print(f"Event log:\n{EVENT_LOG_FILE}")

    print()

    print(f"Output report:\n{OUTPUT_REPORT_FILE}")


    # ------------------------------------------------------------------
    # VERIFY INPUT FILES
    # ------------------------------------------------------------------

    print_header("VERIFYING INPUT FILES")

    if not STATE_TIMELINE_FILE.exists():

        raise FileNotFoundError(
            f"State timeline not found:\n{STATE_TIMELINE_FILE}"
        )

    if not EVENT_LOG_FILE.exists():

        raise FileNotFoundError(
            f"Event log not found:\n{EVENT_LOG_FILE}"
        )

    print("State timeline found.")
    print("Event log found.")


    # ------------------------------------------------------------------
    # LOAD INPUT DATA
    # ------------------------------------------------------------------

    print_header("LOADING ANALYTICS DATA")

    with open(
        STATE_TIMELINE_FILE,
        "r",
        newline="",
        encoding="utf-8",
    ) as state_file:

        state_rows = list(csv.DictReader(state_file))


    with open(
        EVENT_LOG_FILE,
        "r",
        newline="",
        encoding="utf-8",
    ) as event_file:

        event_rows = list(csv.DictReader(event_file))


    if not state_rows:

        raise ValueError(
            "State timeline contains no data rows."
        )


    print(f"State timeline rows loaded : {len(state_rows)}")
    print(f"Event rows loaded          : {len(event_rows)}")


    # ------------------------------------------------------------------
    # VERIFY SOURCE VIDEO
    # ------------------------------------------------------------------

    source_videos = {
        row["source_video"]
        for row in state_rows
    }

    if len(source_videos) != 1:

        raise ValueError(
            "Expected exactly one source video in state timeline."
        )

    source_video = next(iter(source_videos))

    print(f"Source video               : {source_video}")


    # ------------------------------------------------------------------
    # BASIC VIDEO INFORMATION
    # ------------------------------------------------------------------

    processed_frames = len(state_rows)

    first_timestamp = float(
        state_rows[0]["timestamp_seconds"]
    )

    last_timestamp = float(
        state_rows[-1]["timestamp_seconds"]
    )

    video_duration_seconds = max(
        0.0,
        last_timestamp - first_timestamp,
    )


    # ------------------------------------------------------------------
    # COUNT FRAMES PER STATE
    # ------------------------------------------------------------------

    state_counter = Counter(
        row["assembly_state"]
        for row in state_rows
    )

    state_frame_counts = {
        state: state_counter.get(state, 0)
        for state in ALL_STATES
    }


    # ------------------------------------------------------------------
    # ESTIMATE SAMPLE PERIOD
    # ------------------------------------------------------------------

    timestamp_differences = []

    for previous_row, current_row in zip(
        state_rows,
        state_rows[1:],
    ):

        previous_time = float(
            previous_row["timestamp_seconds"]
        )

        current_time = float(
            current_row["timestamp_seconds"]
        )

        difference = current_time - previous_time

        if difference > 0:

            timestamp_differences.append(difference)


    if timestamp_differences:

        average_sample_period = (
            sum(timestamp_differences)
            / len(timestamp_differences)
        )

    else:

        average_sample_period = 0.0


    estimated_processing_fps = (

        1.0 / average_sample_period

        if average_sample_period > 0

        else 0.0
    )


    # ------------------------------------------------------------------
    # CALCULATE TIME PER STATE
    # ------------------------------------------------------------------

    state_duration_seconds = {

        state:
            state_frame_counts[state]
            * average_sample_period

        for state in ALL_STATES
    }


    state_percentages = {

        state:
            safe_percentage(
                state_frame_counts[state],
                processed_frames,
            )

        for state in ALL_STATES
    }


    # ------------------------------------------------------------------
    # FIRST, FINAL, AND DOMINANT STATE
    # ------------------------------------------------------------------

    first_state = state_rows[0]["assembly_state"]

    final_state = state_rows[-1]["assembly_state"]

    dominant_state = max(
        ALL_STATES,
        key=lambda state: state_frame_counts[state],
    )


    # ------------------------------------------------------------------
    # EVENT STATISTICS
    # ------------------------------------------------------------------

    state_transition_count = len(event_rows)

    sequence_violation_count = sum(

        text_to_bool(row["sequence_violation"])

        for row in event_rows
    )

    assembly_completion_event_count = sum(

        text_to_bool(row["assembly_completed"])

        for row in event_rows
    )


    event_type_counter = Counter(

        row["event_type"]

        for row in event_rows
    )


    # ------------------------------------------------------------------
    # EXPECTED SEQUENCE CHECK
    # ------------------------------------------------------------------

    observed_states = [
        row["assembly_state"]
        for row in state_rows
    ]


    expected_sequence_observed = False


    try:

        empty_index = observed_states.index(
            STATE_EMPTY
        )

        marker_index = observed_states.index(
            STATE_MARKER_ONLY,
            empty_index + 1,
        )

        both_index = observed_states.index(
            STATE_BOTH_PRESENT,
            marker_index + 1,
        )

        expected_sequence_observed = True

    except ValueError:

        expected_sequence_observed = False


    # ------------------------------------------------------------------
    # PROCESS STATUS
    # ------------------------------------------------------------------

    if sequence_violation_count > 0:

        process_status = "SEQUENCE_VIOLATIONS_DETECTED"

    elif expected_sequence_observed:

        process_status = "EXPECTED_SEQUENCE_OBSERVED"

    elif final_state == STATE_BOTH_PRESENT:

        process_status = "FINAL_ASSEMBLY_STATE_REACHED"

    else:

        process_status = "EXPECTED_SEQUENCE_NOT_CONFIRMED"


    # ------------------------------------------------------------------
    # CREATE OUTPUT ROW
    # ------------------------------------------------------------------

    report_row = {

        "source_video":
            source_video,

        "processed_frames":
            processed_frames,

        "video_duration_seconds":
            round_value(video_duration_seconds),

        "average_sample_period_seconds":
            round_value(average_sample_period),

        "estimated_processing_fps":
            round_value(estimated_processing_fps),

        "empty_frames":
            state_frame_counts[STATE_EMPTY],

        "marker_only_frames":
            state_frame_counts[STATE_MARKER_ONLY],

        "power_adapter_only_frames":
            state_frame_counts[STATE_POWER_ADAPTER_ONLY],

        "both_present_frames":
            state_frame_counts[STATE_BOTH_PRESENT],

        "empty_duration_seconds":
            round_value(
                state_duration_seconds[STATE_EMPTY]
            ),

        "marker_only_duration_seconds":
            round_value(
                state_duration_seconds[STATE_MARKER_ONLY]
            ),

        "power_adapter_only_duration_seconds":
            round_value(
                state_duration_seconds[STATE_POWER_ADAPTER_ONLY]
            ),

        "both_present_duration_seconds":
            round_value(
                state_duration_seconds[STATE_BOTH_PRESENT]
            ),

        "empty_percentage":
            round_value(
                state_percentages[STATE_EMPTY]
            ),

        "marker_only_percentage":
            round_value(
                state_percentages[STATE_MARKER_ONLY]
            ),

        "power_adapter_only_percentage":
            round_value(
                state_percentages[
                    STATE_POWER_ADAPTER_ONLY
                ]
            ),

        "both_present_percentage":
            round_value(
                state_percentages[STATE_BOTH_PRESENT]
            ),

        "first_state":
            first_state,

        "final_state":
            final_state,

        "dominant_state":
            dominant_state,

        "state_transition_count":
            state_transition_count,

        "process_event_count":
            len(event_rows),

        "sequence_violation_count":
            sequence_violation_count,

        "assembly_completion_event_count":
            assembly_completion_event_count,

        "expected_sequence_observed":
            expected_sequence_observed,

        "process_status":
            process_status,
    }


    # ------------------------------------------------------------------
    # SAVE REPORT
    # ------------------------------------------------------------------

    print_header("SAVING PROCESS ANALYTICS REPORT")

    OUTPUT_REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_REPORT_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=list(report_row.keys()),
        )

        writer.writeheader()

        writer.writerow(report_row)


    # ------------------------------------------------------------------
    # DISPLAY SUMMARY
    # ------------------------------------------------------------------

    print_header("PROCESS ANALYTICS SUMMARY")

    print(f"Source video                 : {source_video}")

    print(f"Processed frames             : {processed_frames}")

    print(
        f"Video duration               : "
        f"{video_duration_seconds:.3f} seconds"
    )

    print(
        f"Estimated processing FPS     : "
        f"{estimated_processing_fps:.3f}"
    )

    print()

    print(f"First assembly state         : {first_state}")

    print(f"Final assembly state         : {final_state}")

    print(f"Dominant assembly state      : {dominant_state}")

    print()

    print(
        f"State transitions            : "
        f"{state_transition_count}"
    )

    print(
        f"Sequence violations          : "
        f"{sequence_violation_count}"
    )

    print(
        f"Assembly completion events   : "
        f"{assembly_completion_event_count}"
    )

    print(
        f"Expected sequence observed   : "
        f"{expected_sequence_observed}"
    )

    print(f"Process status               : {process_status}")

    print()

    print("State distribution:")

    for state in ALL_STATES:

        print(
            f"  {state:20s}: "
            f"{state_frame_counts[state]:5d} frames | "
            f"{state_duration_seconds[state]:8.3f} s | "
            f"{state_percentages[state]:6.2f}%"
        )


    print()

    print("Event type counts:")

    if event_type_counter:

        for event_type, count in sorted(
            event_type_counter.items()
        ):

            print(
                f"  {event_type:40s}: {count}"
            )

    else:

        print("  No process events detected.")


    print()

    print(f"Report saved to:\n{OUTPUT_REPORT_FILE}")


    print_header(
        "STATUS: PROCESS ANALYTICS REPORT CREATED SUCCESSFULLY"
    )

    print(
        "Next stage:\n"
        "Create the final visualization/output video showing "
        "YOLO detections, stable object presence, assembly state, "
        "and detected process events."
    )


if __name__ == "__main__":

    main()