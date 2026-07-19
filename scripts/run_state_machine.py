"""
run_state_machine.py

HOW TO RUN THE FILE:
    python scripts\run_state_machine.py  (TYPE THIS IN TERMINAL)

Prototype V2 - Assembly State Machine and Process Event Detection

Purpose:
    Convert temporally filtered object-presence signals into meaningful
    assembly states, state transitions, and process events.

Why this script is needed:
    YOLO tells us which objects are visible.

    The temporal timeline tells us which objects are stably present.

    However, the assembly-monitoring system needs higher-level process
    information such as:

        - What is the current assembly state?
        - When did the assembly state change?
        - Which object was added?
        - Which object was removed?
        - Did the expected assembly sequence occur?
        - Was the assembly completed?

This script therefore acts as the core rule-based Analytics Model.

Expected prototype sequence:

    EMPTY
        ↓
    MARKER_ONLY
        ↓
    BOTH_PRESENT

State definitions:

    EMPTY
        marker absent
        power adapter absent

    MARKER_ONLY
        marker present
        power adapter absent

    POWER_ADAPTER_ONLY
        marker absent
        power adapter present

    BOTH_PRESENT
        marker present
        power adapter present

Outputs:

    datasets/analytics/state_timeline.csv

        One row per processed frame containing the assembly state.

    datasets/analytics/event_log.csv

        One row per detected state transition / process event.

Important:
    The current expected sequence is a prototype rule set.

    Later experiments can replace these rules with assembly-specific
    process definitions without changing the YOLO detector.
"""

from pathlib import Path
import csv


# ======================================================================
# PROJECT CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANALYTICS_FOLDER = PROJECT_ROOT / "datasets" / "analytics"

DETECTION_TIMELINE_FILE = (
    ANALYTICS_FOLDER / "detection_timeline.csv"
)

STATE_TIMELINE_FILE = (
    ANALYTICS_FOLDER / "state_timeline.csv"
)

EVENT_LOG_FILE = (
    ANALYTICS_FOLDER / "event_log.csv"
)


# ======================================================================
# STATE DEFINITIONS
# ======================================================================

STATE_EMPTY = "EMPTY"

STATE_MARKER_ONLY = "MARKER_ONLY"

STATE_POWER_ADAPTER_ONLY = "POWER_ADAPTER_ONLY"

STATE_BOTH_PRESENT = "BOTH_PRESENT"


EXPECTED_STATE_ORDER = {

    STATE_EMPTY: 0,

    STATE_MARKER_ONLY: 1,

    STATE_BOTH_PRESENT: 2
}


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
        "yes"
    }


def determine_state(marker_present, adapter_present):

    if marker_present and adapter_present:

        return STATE_BOTH_PRESENT

    if marker_present:

        return STATE_MARKER_ONLY

    if adapter_present:

        return STATE_POWER_ADAPTER_ONLY

    return STATE_EMPTY


def determine_transition_event(previous_state, current_state):

    """
    Determine the process event caused by one confirmed state transition.

    Returns:

        event_type
        event_description
        sequence_violation
        assembly_completed
    """

    # --------------------------------------------------------------
    # NORMAL EXPECTED TRANSITIONS
    # --------------------------------------------------------------

    if (
        previous_state == STATE_EMPTY
        and current_state == STATE_MARKER_ONLY
    ):

        return (
            "MARKER_ADDED",
            "Marker entered the workspace.",
            False,
            False
        )

    if (
        previous_state == STATE_MARKER_ONLY
        and current_state == STATE_BOTH_PRESENT
    ):

        return (
            "POWER_ADAPTER_ADDED",
            "Power adapter added after marker.",
            False,
            True
        )

    # --------------------------------------------------------------
    # OBJECT REMOVAL TRANSITIONS
    # --------------------------------------------------------------

    if (
        previous_state == STATE_MARKER_ONLY
        and current_state == STATE_EMPTY
    ):

        return (
            "MARKER_REMOVED",
            "Marker removed from workspace.",
            False,
            False
        )

    if (
        previous_state == STATE_POWER_ADAPTER_ONLY
        and current_state == STATE_EMPTY
    ):

        return (
            "POWER_ADAPTER_REMOVED",
            "Power adapter removed from workspace.",
            False,
            False
        )

    if (
        previous_state == STATE_BOTH_PRESENT
        and current_state == STATE_MARKER_ONLY
    ):

        return (
            "POWER_ADAPTER_REMOVED",
            "Power adapter removed while marker remained.",
            False,
            False
        )

    if (
        previous_state == STATE_BOTH_PRESENT
        and current_state == STATE_POWER_ADAPTER_ONLY
    ):

        return (
            "MARKER_REMOVED",
            "Marker removed while power adapter remained.",
            False,
            False
        )

    # --------------------------------------------------------------
    # UNEXPECTED ASSEMBLY SEQUENCE
    # --------------------------------------------------------------

    if (
        previous_state == STATE_EMPTY
        and current_state == STATE_POWER_ADAPTER_ONLY
    ):

        return (
            "POWER_ADAPTER_ADDED_OUT_OF_SEQUENCE",
            "Power adapter appeared before marker.",
            True,
            False
        )

    if (
        previous_state == STATE_POWER_ADAPTER_ONLY
        and current_state == STATE_BOTH_PRESENT
    ):

        return (
            "MARKER_ADDED_AFTER_ADAPTER",
            "Marker added after power adapter.",
            True,
            True
        )

    if (
        previous_state == STATE_EMPTY
        and current_state == STATE_BOTH_PRESENT
    ):

        return (
            "BOTH_OBJECTS_APPEARED_DIRECTLY",
            "Both objects appeared without the expected intermediate state.",
            True,
            True
        )

    # --------------------------------------------------------------
    # REVERSE / REWORK TRANSITIONS
    # --------------------------------------------------------------

    if (
        previous_state == STATE_BOTH_PRESENT
        and current_state == STATE_EMPTY
    ):

        return (
            "ALL_OBJECTS_REMOVED",
            "Both objects disappeared from the workspace.",
            False,
            False
        )

    if (
        previous_state == STATE_MARKER_ONLY
        and current_state == STATE_POWER_ADAPTER_ONLY
    ):

        return (
            "MARKER_REPLACED_BY_ADAPTER",
            "Marker disappeared while power adapter appeared.",
            True,
            False
        )

    if (
        previous_state == STATE_POWER_ADAPTER_ONLY
        and current_state == STATE_MARKER_ONLY
    ):

        return (
            "ADAPTER_REPLACED_BY_MARKER",
            "Power adapter disappeared while marker appeared.",
            True,
            False
        )

    # --------------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------------

    return (
        "UNCLASSIFIED_STATE_TRANSITION",
        f"State changed from {previous_state} to {current_state}.",
        True,
        False
    )


# ======================================================================
# MAIN PROCESS
# ======================================================================

def main():

    print_header(
        "PROTOTYPE V2 - ASSEMBLY STATE MACHINE AND EVENT DETECTION"
    )

    print(f"Detection timeline:\n{DETECTION_TIMELINE_FILE}")

    print()

    print(f"State timeline output:\n{STATE_TIMELINE_FILE}")

    print()

    print(f"Event log output:\n{EVENT_LOG_FILE}")

    # ------------------------------------------------------------------
    # VERIFY INPUT
    # ------------------------------------------------------------------

    print_header("VERIFYING INPUT DATA")

    if not DETECTION_TIMELINE_FILE.exists():

        raise FileNotFoundError(

            f"Detection timeline not found:\n"
            f"{DETECTION_TIMELINE_FILE}"
        )

    print("Detection timeline found.")

    # ------------------------------------------------------------------
    # LOAD TIMELINE
    # ------------------------------------------------------------------

    print_header("LOADING DETECTION TIMELINE")

    with open(
        DETECTION_TIMELINE_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as input_file:

        timeline_rows = list(
            csv.DictReader(input_file)
        )

    if not timeline_rows:

        raise ValueError(
            "Detection timeline contains no data rows."
        )

    print(f"Timeline rows loaded: {len(timeline_rows)}")

    source_videos = {

        row["source_video"]

        for row in timeline_rows
    }

    if len(source_videos) != 1:

        raise ValueError(

            "The current Analytics Model expects exactly "
            "one source video per timeline."
        )

    source_video = next(iter(source_videos))

    print(f"Source video: {source_video}")

    # ------------------------------------------------------------------
    # OUTPUT COLUMNS
    # ------------------------------------------------------------------

    state_timeline_columns = [

        "source_video",

        "frame_number",

        "timestamp_seconds",

        "marker_stable_present",

        "power_adapter_stable_present",

        "assembly_state",

        "state_changed",

        "previous_state",

        "current_state_duration_frames",

        "current_state_duration_seconds"
    ]


    event_log_columns = [

        "source_video",

        "event_number",

        "frame_number",

        "timestamp_seconds",

        "previous_state",

        "new_state",

        "event_type",

        "event_description",

        "sequence_violation",

        "assembly_completed"
    ]


    # ------------------------------------------------------------------
    # STATE MACHINE VARIABLES
    # ------------------------------------------------------------------

    previous_state = None

    state_start_frame = 0

    state_start_timestamp = 0.0

    event_number = 0

    transition_count = 0

    sequence_violation_count = 0

    completion_event_count = 0

    state_frame_counts = {

        STATE_EMPTY: 0,

        STATE_MARKER_ONLY: 0,

        STATE_POWER_ADAPTER_ONLY: 0,

        STATE_BOTH_PRESENT: 0
    }


    # ------------------------------------------------------------------
    # RUN STATE MACHINE
    # ------------------------------------------------------------------

    print_header("RUNNING ASSEMBLY STATE MACHINE")

    with open(
        STATE_TIMELINE_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as state_file, open(
        EVENT_LOG_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as event_file:

        state_writer = csv.DictWriter(

            state_file,

            fieldnames=state_timeline_columns
        )

        event_writer = csv.DictWriter(

            event_file,

            fieldnames=event_log_columns
        )

        state_writer.writeheader()

        event_writer.writeheader()


        for row_index, row in enumerate(timeline_rows):

            frame_number = int(row["frame_number"])

            timestamp_seconds = float(
                row["timestamp_seconds"]
            )

            marker_present = text_to_bool(

                row["marker_stable_present"]
            )

            adapter_present = text_to_bool(

                row["power_adapter_stable_present"]
            )

            current_state = determine_state(

                marker_present,

                adapter_present
            )


            # ----------------------------------------------------------
            # FIRST FRAME
            # ----------------------------------------------------------

            if previous_state is None:

                state_changed = False

                previous_state_for_output = ""

                state_start_frame = frame_number

                state_start_timestamp = timestamp_seconds


            # ----------------------------------------------------------
            # STATE TRANSITION
            # ----------------------------------------------------------

            elif current_state != previous_state:

                state_changed = True

                previous_state_for_output = previous_state

                transition_count += 1

                event_number += 1


                (
                    event_type,

                    event_description,

                    sequence_violation,

                    assembly_completed

                ) = determine_transition_event(

                    previous_state,

                    current_state
                )


                if sequence_violation:

                    sequence_violation_count += 1


                if assembly_completed:

                    completion_event_count += 1


                event_writer.writerow({

                    "source_video":
                        source_video,

                    "event_number":
                        event_number,

                    "frame_number":
                        frame_number,

                    "timestamp_seconds":
                        round(timestamp_seconds, 6),

                    "previous_state":
                        previous_state,

                    "new_state":
                        current_state,

                    "event_type":
                        event_type,

                    "event_description":
                        event_description,

                    "sequence_violation":
                        sequence_violation,

                    "assembly_completed":
                        assembly_completed
                })


                state_start_frame = frame_number

                state_start_timestamp = timestamp_seconds


            # ----------------------------------------------------------
            # NO STATE TRANSITION
            # ----------------------------------------------------------

            else:

                state_changed = False

                previous_state_for_output = previous_state


            # ----------------------------------------------------------
            # STATE DURATION
            # ----------------------------------------------------------

            current_state_duration_frames = (

                frame_number - state_start_frame + 1
            )

            current_state_duration_seconds = (

                timestamp_seconds
                - state_start_timestamp
            )


            # ----------------------------------------------------------
            # STATISTICS
            # ----------------------------------------------------------

            state_frame_counts[current_state] += 1


            # ----------------------------------------------------------
            # SAVE STATE TIMELINE
            # ----------------------------------------------------------

            state_writer.writerow({

                "source_video":
                    source_video,

                "frame_number":
                    frame_number,

                "timestamp_seconds":
                    round(timestamp_seconds, 6),

                "marker_stable_present":
                    marker_present,

                "power_adapter_stable_present":
                    adapter_present,

                "assembly_state":
                    current_state,

                "state_changed":
                    state_changed,

                "previous_state":
                    previous_state_for_output,

                "current_state_duration_frames":
                    current_state_duration_frames,

                "current_state_duration_seconds":
                    round(
                        current_state_duration_seconds,
                        6
                    )
            })


            previous_state = current_state


            if (row_index + 1) % 100 == 0:

                print(

                    f"Processed "
                    f"{row_index + 1}/"
                    f"{len(timeline_rows)} frames"
                )


    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------

    print_header("STATE MACHINE SUMMARY")

    print(f"Source video                  : {source_video}")

    print(f"Processed frames              : {len(timeline_rows)}")

    print(f"State transitions detected    : {transition_count}")

    print(f"Process events recorded       : {event_number}")

    print(
        f"Sequence violations detected  : "
        f"{sequence_violation_count}"
    )

    print(
        f"Assembly completion events    : "
        f"{completion_event_count}"
    )

    print()

    print("Frames per assembly state:")

    for state_name, frame_count in state_frame_counts.items():

        percentage = (

            frame_count
            / len(timeline_rows)
            * 100
        )

        print(

            f"  {state_name:20s}: "
            f"{frame_count:5d} frames "
            f"({percentage:6.2f}%)"
        )

    print()

    print(f"State timeline saved to:\n{STATE_TIMELINE_FILE}")

    print()

    print(f"Event log saved to:\n{EVENT_LOG_FILE}")

    print_header(
        "STATUS: ASSEMBLY STATE MACHINE COMPLETED SUCCESSFULLY"
    )

    print(

        "Next stage:\n"
        "Calculate process-level analytics including state durations, "
        "cycle information, event counts, sequence violations, "
        "and completion status."
    )


if __name__ == "__main__":

    main()