"""
===============================================================================
Prototype V2.3 - Guided Controlled Dataset Acquisition System
===============================================================================

File:
    record_dataset.py

HOW TO RUN THE FILE:
    python record_dataset.py  (TYPE THIS IN TERMINAL)

Author:
    Mehul Patil

Project:
    AI-Supported Ergonomic and Productivity Analysis
    Vision-Based Assembly Monitoring Prototype (V2)

===============================================================================
WHAT IS THIS PROGRAM?
===============================================================================

This program records controlled camera trials for building a structured dataset
for computer vision and YOLO training.

It is designed for one experiment session where the user records several trials
in a planned order. Every trial is saved as a video and documented in a CSV
file with useful metadata.

The recorder does not try to detect objects or analyze the video. It only
collects clean, traceable training data.

===============================================================================
WHY IS THIS PROGRAM IMPORTANT?
===============================================================================

A good AI model depends on a good dataset.

If the recordings are random, hard to trace, or inconsistent, the later model
training becomes confusing and unreliable.

This recorder keeps every trial organized by:

    • Experiment ID
    • Trial Number
    • Trial Condition
    • Objects Present
    • Notes
    • Video Filename
    • Trial ID
    • Start and End Time
    • Frame Count
    • Camera Settings

===============================================================================
WHAT DOES THIS PROGRAM DO?
===============================================================================

The program works in a simple loop:

1.  It opens the camera and shows a live preview.
2.  It asks for the experiment ID only once.
3.  It automatically continues with the next trial number.
4.  It shows the next planned trial from the built-in 15-trial protocol.
5.  It records the trial when the user presses a key.
6.  It saves the video.
7.  It stores all trial details in trials.csv.
8.  It shows the next trial and continues until all planned trials are done.

===============================================================================
PLANNED 15-TRIAL PROTOCOL
===============================================================================

The first experiment is built directly into this file so the user does not need
to remember the trial order.

Trial 01: Empty workspace
Trial 02: Marker at center
Trial 03: Marker at left
Trial 04: Marker at right
Trial 05: Marker rotated
Trial 06: Power adapter at center
Trial 07: Power adapter at left
Trial 08: Power adapter at right
Trial 09: Power adapter rotated
Trial 10: Marker and adapter separated
Trial 11: Marker and adapter close
Trial 12: Marker partially occluded
Trial 13: Power adapter partially occluded
Trial 14: Hand interaction with both objects
Trial 15: Continuous movement of both objects

If you later want to change the experiment, you only need to edit the
TRIAL_PROTOCOL table below.

===============================================================================
OUTPUT FILES
===============================================================================

Video recordings:
    datasets/raw/videos/

Metadata CSV:
    datasets/raw/metadata/trials.csv

===============================================================================
KEYBOARD CONTROLS
===============================================================================

In the camera window:

    R = Prepare and start a recording
    S = Stop the current recording
    Q = Quit the program

After stopping a trial:

    Y = Start the next trial immediately
    ENTER = Return to live view
    Q = Quit

===============================================================================
NOTES
===============================================================================

This program does NOT perform:

    • Frame extraction
    • Image quality analysis
    • Dataset cleaning
    • Annotation
    • YOLO training
    • Object detection

Those stages are handled by other scripts in Prototype V2.

===============================================================================
"""

from pathlib import Path
from datetime import datetime
import csv
import time

import cv2


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

REQUESTED_FPS = 30.0

VIDEO_CODEC = "mp4v"

WINDOW_NAME = "Prototype V2.3 - Dataset Acquisition"

DEFAULT_EXPERIMENT_ID = "experiment_001_two_object_detection"

DEFAULT_STARTING_TRIAL_NUMBER = 1

DEFAULT_OBJECTS_PRESENT = "marker,power_adapter"


# ============================================================
# PLANNED TRIAL PROTOCOL
# ============================================================
#
# The first 15 trials are built into the code so the recording
# session becomes guided and fast.
#
# Each entry contains:
#   trial_condition -> short internal code
#   label           -> human-readable name
#   objects_present -> what should be in the workspace
#   instruction     -> what the user should do for the trial
# ============================================================

TRIAL_PROTOCOL = {
    1: {
        "trial_condition": "empty_workspace",
        "label": "Empty workspace",
        "objects_present": "none",
        "instruction": "Clear the table completely and record an empty workspace.",
    },
    2: {
        "trial_condition": "marker_center",
        "label": "Marker at center",
        "objects_present": "marker",
        "instruction": "Place the marker in the center of the workspace and keep it visible.",
    },
    3: {
        "trial_condition": "marker_left",
        "label": "Marker at left",
        "objects_present": "marker",
        "instruction": "Move the marker to the left side of the workspace.",
    },
    4: {
        "trial_condition": "marker_right",
        "label": "Marker at right",
        "objects_present": "marker",
        "instruction": "Move the marker to the right side of the workspace.",
    },
    5: {
        "trial_condition": "marker_rotated",
        "label": "Marker rotated",
        "objects_present": "marker",
        "instruction": "Rotate the marker while keeping it on the table.",
    },
    6: {
        "trial_condition": "power_adapter_center",
        "label": "Power adapter at center",
        "objects_present": "power_adapter",
        "instruction": "Place the power adapter in the center of the workspace.",
    },
    7: {
        "trial_condition": "power_adapter_left",
        "label": "Power adapter at left",
        "objects_present": "power_adapter",
        "instruction": "Move the power adapter to the left side of the workspace.",
    },
    8: {
        "trial_condition": "power_adapter_right",
        "label": "Power adapter at right",
        "objects_present": "power_adapter",
        "instruction": "Move the power adapter to the right side of the workspace.",
    },
    9: {
        "trial_condition": "power_adapter_rotated",
        "label": "Power adapter rotated",
        "objects_present": "power_adapter",
        "instruction": "Rotate the power adapter while keeping it visible.",
    },
    10: {
        "trial_condition": "marker_and_power_adapter_separated",
        "label": "Marker and adapter separated",
        "objects_present": "marker,power_adapter",
        "instruction": "Place the marker and the power adapter far apart in the workspace.",
    },
    11: {
        "trial_condition": "marker_and_power_adapter_close",
        "label": "Marker and adapter close",
        "objects_present": "marker,power_adapter",
        "instruction": "Place the marker and the power adapter close together.",
    },
    12: {
        "trial_condition": "marker_partial_occlusion",
        "label": "Marker partially occluded",
        "objects_present": "marker",
        "instruction": "Partially hide the marker with your hand or another object.",
    },
    13: {
        "trial_condition": "power_adapter_partial_occlusion",
        "label": "Power adapter partially occluded",
        "objects_present": "power_adapter",
        "instruction": "Partially hide the power adapter with your hand or another object.",
    },
    14: {
        "trial_condition": "hand_interaction",
        "label": "Hand interaction with both",
        "objects_present": "marker,power_adapter",
        "instruction": "Pick up, move, or reposition both objects with your hand.",
    },
    15: {
        "trial_condition": "continuous_movement",
        "label": "Continuous movement",
        "objects_present": "marker,power_adapter",
        "instruction": "Move, rotate, pick, and place both objects continuously.",
    },
}

TOTAL_PLANNED_TRIALS = len(TRIAL_PROTOCOL)

# Used only when the operator chooses a CUSTOM condition.
CUSTOM_CONDITION_KEY = "c"


# ============================================================
# PROJECT PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATASETS_DIR = PROJECT_ROOT / "datasets"
RAW_DATA_DIR = DATASETS_DIR / "raw"
VIDEO_DIR = RAW_DATA_DIR / "videos"
METADATA_DIR = RAW_DATA_DIR / "metadata"
METADATA_FILE = METADATA_DIR / "trials.csv"

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# METADATA CSV COLUMNS
# ============================================================

CSV_COLUMNS = [
    "experiment_id",
    "trial_number",
    "trial_condition",
    "objects_present",
    "notes",
    "trial_id",
    "video_filename",
    "start_timestamp",
    "end_timestamp",
    "duration_seconds",
    "frame_count",
    "frame_width",
    "frame_height",
    "requested_fps",
    "actual_fps",
    "camera_index",
]


# ============================================================
# CREATE / VALIDATE METADATA CSV
# ============================================================

def initialize_metadata_file():
    """
    Create trials.csv if it does not exist.

    If an existing file uses a different header (old schema),
    the program stops instead of silently mixing schemas.
    """

    if not METADATA_FILE.exists():
        with METADATA_FILE.open(
            mode="w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=CSV_COLUMNS,
            )
            writer.writeheader()

        print("Created new metadata file:")
        print(METADATA_FILE)
        return True

    with METADATA_FILE.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.reader(csv_file)
        existing_header = next(reader, None)

    if existing_header != CSV_COLUMNS:
        print()
        print("=" * 70)
        print("ERROR: EXISTING METADATA FILE USES AN INCOMPATIBLE SCHEMA")
        print("=" * 70)
        print("File:")
        print(METADATA_FILE)
        print()
        print("The current recorder expects these columns:")
        for column in CSV_COLUMNS:
            print(f"  {column}")
        print()
        print("Do not mix old and new CSV schemas.")
        print()
        print("Rename the existing trials.csv, for example:")
        print("  trials_prototype_test_backup.csv")
        print()
        print("Then run record_dataset.py again.")
        print("=" * 70)
        return False

    return True


# ============================================================
# SMALL HELPERS
# ============================================================

def generate_trial_id():
    """Example: T_20260712_183045_123456"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"T_{timestamp}"


def shorten_text(text, maximum_length=35):
    """Shorten long text for the OpenCV overlay. Full text stays in the CSV."""
    if len(text) <= maximum_length:
        return text
    return text[: maximum_length - 3] + "..."


def ask_required_text(prompt, default_value=None):
    """
    Ask repeatedly until the user enters non-empty text.

    If default_value is provided, ENTER accepts the default.
    """

    while True:
        if default_value is not None:
            value = input(f"{prompt} [{default_value}]: ").strip()
            if not value:
                return default_value
        else:
            value = input(f"{prompt}: ").strip()

        if value:
            return value

        print("ERROR: This field cannot be empty.")


def ask_optional_text(prompt, default_value=""):
    """Ask for optional text. ENTER accepts the default (usually empty)."""
    suffix = f" [{default_value}]" if default_value else " (ENTER to skip)"
    value = input(f"{prompt}{suffix}: ").strip()
    return value if value else default_value


def format_trial_position(trial_number):
    """Return a compact Trial X/Y string for the planned experiment."""
    if trial_number <= TOTAL_PLANNED_TRIALS:
        return f"Trial {trial_number}/{TOTAL_PLANNED_TRIALS}"
    return f"Trial {trial_number}"


def get_planned_trial(trial_number):
    """
    Return the planned trial dictionary for a trial number.

    If the number is not in the built-in protocol, return None.
    """
    trial = TRIAL_PROTOCOL.get(trial_number)
    if trial is None:
        return None

    return {
        "trial_number": trial_number,
        "trial_condition": trial["trial_condition"],
        "label": trial["label"],
        "objects_present": trial["objects_present"],
        "instruction": trial["instruction"],
    }


# ============================================================
# READ EXISTING TRIALS  (auto trial numbering)
# ============================================================

def get_next_trial_number(experiment_id):
    """
    Determine the next available trial number for this experiment
    from trials.csv, so a session can be resumed safely.

    Returns DEFAULT_STARTING_TRIAL_NUMBER if the experiment has
    no recorded trials yet.
    """

    if not METADATA_FILE.exists():
        return DEFAULT_STARTING_TRIAL_NUMBER

    max_trial_number = 0

    with METADATA_FILE.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            if row.get("experiment_id", "").strip() != experiment_id:
                continue

            try:
                trial_number = int(row.get("trial_number", "").strip())
            except (ValueError, AttributeError):
                continue

            if trial_number > max_trial_number:
                max_trial_number = trial_number

    if max_trial_number > 0:
        return max_trial_number + 1

    return DEFAULT_STARTING_TRIAL_NUMBER


def count_experiment_trials(experiment_id):
    """Count how many trials already exist for this experiment."""

    if not METADATA_FILE.exists():
        return 0

    count = 0

    with METADATA_FILE.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row.get("experiment_id", "").strip() == experiment_id:
                count += 1

    return count


# ============================================================
# SESSION SETUP  (asked ONCE at startup)
# ============================================================

def collect_session_information():
    """
    Collect information that stays the same for the whole session.
    ENTER accepts the default experiment ID.

    Returns a session dict, or None if cancelled.
    """

    print()
    print("=" * 70)
    print("SESSION INFORMATION")
    print("=" * 70)

    experiment_id = ask_required_text(
        "Experiment ID",
        DEFAULT_EXPERIMENT_ID,
    )

    existing_trials = count_experiment_trials(experiment_id)
    next_trial_number = get_next_trial_number(experiment_id)

    print()
    print("-" * 70)
    print("SESSION SUMMARY")
    print("-" * 70)
    print(f"Experiment ID    : {experiment_id}")

    if existing_trials > 0:
        print(f"Existing trials  : {existing_trials}")
        print(f"Resuming at trial: {next_trial_number}")
    else:
        print("Existing trials  : 0 (new experiment)")
        print(f"Starting at trial: {next_trial_number}")

    print("-" * 70)

    confirmation = input(
        "Use these session settings? ENTER = yes, X = cancel: "
    ).strip().lower()

    if confirmation == "x":
        print("Session setup cancelled.")
        return None

    return {
        "experiment_id": experiment_id,
        "next_trial_number": next_trial_number,
    }


# ============================================================
# TRIAL CONDITION MENU  (manual fallback only)
# ============================================================

TRIAL_CONDITION_OPTIONS = {
    1: ("empty_workspace", "Empty workspace", "none"),
    2: ("marker_only", "Marker only", "marker"),
    3: ("power_adapter_only", "Power adapter only", "power_adapter"),
    4: ("marker_and_power_adapter_separated", "Marker and power adapter separated", "marker,power_adapter"),
    5: ("marker_and_power_adapter_close", "Marker and power adapter close", "marker,power_adapter"),
    6: ("marker_rotated", "Marker rotated", "marker"),
    7: ("power_adapter_rotated", "Power adapter rotated", "power_adapter"),
    8: ("partial_occlusion", "Partial occlusion", "marker,power_adapter"),
    9: ("hand_interaction", "Hand interaction", "marker,power_adapter"),
    10: ("continuous_movement", "Continuous movement", "marker,power_adapter"),
}

CUSTOM_CONDITION_KEY = "c"


def print_trial_condition_menu(previous_condition=None):
    print()
    print("-" * 70)
    print("MANUAL TRIAL CONDITION MENU")
    print("-" * 70)

    for number, (code, label, objects) in TRIAL_CONDITION_OPTIONS.items():
        print(f"{number:>3}  {label:<36} ({code} | objects: {objects})")

    print(f"  {CUSTOM_CONDITION_KEY.upper()}  Custom condition (type manually)")
    print("  X  Cancel this trial")

    if previous_condition is not None:
        print(f"  ENTER = reuse previous condition ({previous_condition[0]})")

    print("-" * 70)


def ask_trial_condition(previous_condition=None):
    """
    Select a trial condition from the manual menu.

    previous_condition:
        Optional (condition_code, objects_present) tuple.
        ENTER reuses it - fast when repeating the same condition.

    Returns (condition_code, objects_present) or None if cancelled.
    """

    print_trial_condition_menu(previous_condition)

    while True:
        value = input("Condition number: ").strip().lower()

        if not value:
            if previous_condition is not None:
                code, objects = previous_condition
                print(f"Reusing previous condition: {code}")
                return previous_condition

            print("ERROR: Please select a trial condition.")
            continue

        if value == "x":
            return None

        if value == CUSTOM_CONDITION_KEY:
            condition_code = ask_required_text("Custom trial condition")
            objects_present = ask_required_text(
                "Objects present",
                DEFAULT_OBJECTS_PRESENT,
            )
            return condition_code, objects_present

        try:
            choice = int(value)
        except ValueError:
            print("ERROR: Enter a menu number, "
                  f"'{CUSTOM_CONDITION_KEY.upper()}' for custom, or 'X' to cancel.")
            continue

        if choice not in TRIAL_CONDITION_OPTIONS:
            print("ERROR: Invalid menu number.")
            continue

        condition_code, condition_label, objects_present = TRIAL_CONDITION_OPTIONS[choice]

        print(f"Selected: {condition_label} ({condition_code} | objects: {objects_present})")

        return condition_code, objects_present


# ============================================================
# COLLECT TRIAL INFORMATION
# ============================================================

def collect_manual_trial_information(session, previous_condition=None):
    """
    Manual fallback when the trial number is beyond the built-in
    15-trial protocol or when the user chooses to customize a
    planned trial.
    """

    experiment_id = session["experiment_id"]
    trial_number = session["next_trial_number"]

    print()
    print("=" * 70)
    print(f"NEW TRIAL | {experiment_id} | Trial {trial_number}")
    print("=" * 70)

    chosen = ask_trial_condition(previous_condition)

    if chosen is None:
        print("Trial preparation cancelled.")
        return None

    trial_condition, objects_present = chosen

    notes = ask_optional_text("Trial notes")

    print()
    print("-" * 70)
    print("TRIAL SUMMARY")
    print("-" * 70)
    print(f"Experiment ID  : {experiment_id}")
    print(f"Trial number   : {trial_number}")
    print(f"Trial condition: {trial_condition}")
    print(f"Objects present: {objects_present}")
    print(f"Notes          : {notes if notes else '(none)'}")
    print("-" * 70)

    confirmation = input(
        "Start this trial? ENTER = start, X = cancel: "
    ).strip().lower()

    if confirmation == "x":
        print("Trial preparation cancelled.")
        return None

    return {
        "experiment_id": experiment_id,
        "trial_number": trial_number,
        "trial_condition": trial_condition,
        "objects_present": objects_present,
        "notes": notes,
    }


def collect_planned_trial_information(session, planned_trial):
    """
    Show the built-in planned trial, ask for optional notes,
    and start the recording when the user presses ENTER.
    """

    experiment_id = session["experiment_id"]
    trial_number = session["next_trial_number"]

    print()
    print("=" * 70)
    print(f"PLANNED TRIAL | {format_trial_position(trial_number)}")
    print("=" * 70)
    print(f"Experiment ID  : {experiment_id}")
    print(f"Trial number   : {trial_number}")
    print(f"Condition      : {planned_trial['label']}")
    print(f"Condition code : {planned_trial['trial_condition']}")
    print(f"Objects present: {planned_trial['objects_present']}")
    print(f"Instruction    : {planned_trial['instruction']}")
    print("-" * 70)

    notes = ask_optional_text("Trial notes")

    print()
    print("-" * 70)
    print("TRIAL SUMMARY")
    print("-" * 70)
    print(f"Experiment ID  : {experiment_id}")
    print(f"Trial number   : {trial_number}")
    print(f"Trial condition: {planned_trial['trial_condition']}")
    print(f"Objects present: {planned_trial['objects_present']}")
    print(f"Notes          : {notes if notes else '(none)'}")
    print("-" * 70)

    confirmation = input(
        "Start this trial? ENTER = start, C = customize, X = cancel: "
    ).strip().lower()

    if confirmation == "x":
        print("Trial preparation cancelled.")
        return None

    if confirmation == "c":
        return collect_manual_trial_information(session)

    return {
        "experiment_id": experiment_id,
        "trial_number": trial_number,
        "trial_condition": planned_trial["trial_condition"],
        "objects_present": planned_trial["objects_present"],
        "notes": notes,
    }


def collect_trial_information(session, previous_condition=None):
    """
    Collect trial information.

    For trials 1-15:
        Use the built-in guided protocol.

    After that:
        Use the manual fallback menu.
    """

    planned_trial = get_planned_trial(session["next_trial_number"])

    if planned_trial is not None:
        return collect_planned_trial_information(session, planned_trial)

    return collect_manual_trial_information(session, previous_condition)


# ============================================================
# SAVE TRIAL METADATA
# ============================================================

def save_trial_metadata(
    trial_information,
    trial_id,
    video_filename,
    start_timestamp,
    end_timestamp,
    duration_seconds,
    frame_count,
    frame_width,
    frame_height,
    requested_fps,
    actual_fps,
    camera_index,
):
    """Append one completed trial to trials.csv."""

    metadata = {
        "experiment_id": trial_information["experiment_id"],
        "trial_number": trial_information["trial_number"],
        "trial_condition": trial_information["trial_condition"],
        "objects_present": trial_information["objects_present"],
        "notes": trial_information["notes"],
        "trial_id": trial_id,
        "video_filename": video_filename,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "duration_seconds": round(duration_seconds, 3),
        "frame_count": frame_count,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "requested_fps": requested_fps,
        "actual_fps": round(actual_fps, 3),
        "camera_index": camera_index,
    }

    with METADATA_FILE.open(
        mode="a",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writerow(metadata)


# ============================================================
# PRINT SAVED-TRIAL SUMMARY
# ============================================================

def print_trial_saved(trial_information, trial_id, frame_count, duration_seconds):
    print()
    print("=" * 70)
    print("TRIAL SAVED")
    print("=" * 70)
    print(f"Experiment : {trial_information['experiment_id']}")
    print(f"Trial      : {trial_information['trial_number']}")
    print(f"Condition  : {trial_information['trial_condition']}")
    print(f"Objects    : {trial_information['objects_present']}")
    print(f"Trial ID   : {trial_id}")
    print(f"Frames     : {frame_count}")
    print(f"Duration   : {duration_seconds:.1f} seconds")
    print("=" * 70)


# ============================================================
# LIVE VIEW OVERLAYS
# ============================================================

def draw_recording_overlay(display_frame, trial_information, elapsed_time, frame_count):
    """Draw a multi-line recording status overlay on the live view."""

    experiment_display = shorten_text(
        trial_information["experiment_id"], maximum_length=30
    )

    condition_display = shorten_text(
        trial_information["trial_condition"], maximum_length=30
    )

    lines = [
        (f"Experiment: {experiment_display}", 0.60),
        (f"{format_trial_position(trial_information['trial_number'])} | {condition_display}", 0.60),
        (f"RECORDING...  {elapsed_time:.1f} s  |  {frame_count} frames", 0.65),
    ]

    y = 35
    for text, scale in lines:
        cv2.putText(
            display_frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        y += 32

    height, width = display_frame.shape[:2]
    cv2.circle(display_frame, (width - 40, 40), 12, (0, 0, 255), -1)


def draw_ready_overlay(display_frame, session):
    """Draw the READY status overlay including the next trial number."""

    experiment_display = shorten_text(
        session["experiment_id"], maximum_length=30
    )

    planned_trial = get_planned_trial(session["next_trial_number"])

    if planned_trial is not None:
        next_trial_text = f"Next: {format_trial_position(session['next_trial_number'])} | {planned_trial['label']}"
        instruction_text = f"Instruction: {shorten_text(planned_trial['instruction'], 52)}"
    else:
        next_trial_text = f"Next: {format_trial_position(session['next_trial_number'])} | Manual trial"
        instruction_text = "Press R to configure the next trial."

    line_1 = f"READY | {experiment_display}"
    line_2 = next_trial_text
    line_3 = instruction_text

    cv2.putText(
        display_frame,
        line_1,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        display_frame,
        line_2,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        display_frame,
        line_3,
        (20, 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


# ============================================================
# ASK "RECORD NEXT TRIAL?" AFTER STOPPING
# ============================================================

def ask_next_action():
    """
    After a trial is saved, ask what to do next.

    Returns:
        "next"  -> immediately prepare the next trial
        "ready" -> return to the live READY view
        "quit"  -> quit the program
    """

    print()
    print("Next action?")
    print("Y = Next trial | ENTER = Back to live view | Q = Quit")

    choice = input("> ").strip().lower()

    if choice == "y":
        return "next"

    if choice == "q":
        return "quit"

    return "ready"


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 70)
    print("PROTOTYPE V2.3 - GUIDED CONTROLLED DATASET ACQUISITION")
    print("=" * 70)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Video folder : {VIDEO_DIR}")
    print(f"Metadata file: {METADATA_FILE}")
    print("-" * 70)

    if not initialize_metadata_file():
        return

    session = collect_session_information()
    if session is None:
        return

    previous_condition = None

    print()
    print("CONTROLS (camera window)")
    print("R = Prepare and start a new recording")
    print("S = Stop recording and save metadata")
    print("Q = Quit")
    print("-" * 70)

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print(f"ERROR: Could not open camera index {CAMERA_INDEX}.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, REQUESTED_FPS)

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = camera.get(cv2.CAP_PROP_FPS)

    print(
        f"Camera opened successfully: "
        f"{actual_width}x{actual_height} at reported {actual_fps:.2f} FPS"
    )

    is_recording = False
    video_writer = None
    trial_information = None
    trial_id = None
    video_filename = None
    recording_start_time = None
    start_timestamp = None
    recorded_frame_count = 0

    prepare_next_trial = False

    def start_new_recording():
        """
        Collect trial info and create the video writer.
        Returns True if recording started, False if cancelled/failed.
        """
        nonlocal is_recording, video_writer, trial_information
        nonlocal trial_id, video_filename
        nonlocal recording_start_time, start_timestamp, recorded_frame_count
        nonlocal previous_condition

        info = collect_trial_information(session, previous_condition)

        if info is None:
            return False

        new_trial_id = generate_trial_id()
        new_video_filename = f"{new_trial_id}.mp4"
        video_path = VIDEO_DIR / new_video_filename

        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)

        writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            REQUESTED_FPS,
            (actual_width, actual_height),
        )

        if not writer.isOpened():
            print("ERROR: Could not create video file:")
            print(video_path)
            return False

        trial_information = info
        trial_id = new_trial_id
        video_filename = new_video_filename
        video_writer = writer

        recording_start_time = time.perf_counter()
        start_timestamp = datetime.now().isoformat(timespec="milliseconds")
        recorded_frame_count = 0
        is_recording = True

        print()
        print("=" * 70)
        print("RECORDING STARTED")
        print("=" * 70)
        print(f"Experiment : {info['experiment_id']}")
        print(f"Trial      : {info['trial_number']}")
        print(f"Condition  : {info['trial_condition']}")
        print(f"Objects    : {info['objects_present']}")
        print(f"Trial ID   : {new_trial_id}")
        print(f"Video      : {video_path}")
        print("=" * 70)
        print("Press S in the camera window to stop.")

        return True

    def stop_and_save_recording():
        """Close the video, write metadata, advance trial number."""
        nonlocal is_recording, video_writer, trial_information
        nonlocal trial_id, video_filename
        nonlocal recording_start_time, start_timestamp, recorded_frame_count
        nonlocal previous_condition

        end_time = time.perf_counter()
        end_timestamp = datetime.now().isoformat(timespec="milliseconds")
        duration_seconds = end_time - recording_start_time

        video_writer.release()
        video_writer = None
        is_recording = False

        save_trial_metadata(
            trial_information=trial_information,
            trial_id=trial_id,
            video_filename=video_filename,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            duration_seconds=duration_seconds,
            frame_count=recorded_frame_count,
            frame_width=actual_width,
            frame_height=actual_height,
            requested_fps=REQUESTED_FPS,
            actual_fps=actual_fps,
            camera_index=CAMERA_INDEX,
        )

        print_trial_saved(
            trial_information,
            trial_id,
            recorded_frame_count,
            duration_seconds,
        )

        previous_condition = (
            trial_information["trial_condition"],
            trial_information["objects_present"],
        )

        session["next_trial_number"] += 1

        trial_information = None
        trial_id = None
        video_filename = None
        recording_start_time = None
        start_timestamp = None
        recorded_frame_count = 0

    try:
        while True:

            if prepare_next_trial:
                prepare_next_trial = False
                start_new_recording()

            success, frame = camera.read()

            if not success or frame is None:
                print("WARNING: Invalid camera frame received.")
                continue

            if is_recording and video_writer is not None:
                video_writer.write(frame)
                recorded_frame_count += 1

            display_frame = frame.copy()

            if is_recording:
                elapsed_time = time.perf_counter() - recording_start_time
                draw_recording_overlay(
                    display_frame,
                    trial_information,
                    elapsed_time,
                    recorded_frame_count,
                )
            else:
                draw_ready_overlay(display_frame, session)

            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):

                if is_recording:
                    print("WARNING: A trial is already being recorded.")
                    continue

                start_new_recording()

            elif key == ord("s"):

                if not is_recording:
                    print("WARNING: No active recording to stop.")
                    continue

                stop_and_save_recording()

                action = ask_next_action()

                if action == "next":
                    prepare_next_trial = True
                elif action == "quit":
                    print("Quit requested.")
                    break

            elif key == ord("q"):

                print("Quit requested.")
                break

    finally:

        if is_recording and video_writer is not None:

            end_time = time.perf_counter()
            end_timestamp = datetime.now().isoformat(timespec="milliseconds")
            duration_seconds = end_time - recording_start_time

            video_writer.release()

            save_trial_metadata(
                trial_information=trial_information,
                trial_id=trial_id,
                video_filename=video_filename,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                duration_seconds=duration_seconds,
                frame_count=recorded_frame_count,
                frame_width=actual_width,
                frame_height=actual_height,
                requested_fps=REQUESTED_FPS,
                actual_fps=actual_fps,
                camera_index=CAMERA_INDEX,
            )

            print("Active recording was safely stopped and metadata was saved.")

        camera.release()
        cv2.destroyAllWindows()

        print("Camera released.")
        print("Program closed safely.")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()