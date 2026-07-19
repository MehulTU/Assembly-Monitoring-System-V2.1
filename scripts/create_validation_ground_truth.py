"""
HOW TO RUN:
    python scripts/create_validation_ground_truth.py

Prototype V2 - Frame-Level Validation Ground-Truth Creator

PURPOSE
-------
This program creates frame-level ground truth for a controlled validation
video.

Ground truth means the manually defined correct answer for every video frame.

The program is designed to avoid manually writing hundreds or thousands of
CSV rows.

Instead, the user:

    1. Selects a validation video.
    2. Moves through the video.
    3. Finds the exact frame where the real assembly state changes.
    4. Presses a number key to assign the correct state.
    5. Repeats this only when the real state changes.
    6. Finishes the annotation.
    7. The program automatically creates one ground-truth row per video frame.

SUPPORTED STATES
----------------

    0 = EMPTY

    1 = MARKER_ONLY

    2 = POWER_ADAPTER_ONLY

    3 = BOTH_PRESENT


CONTROLS
--------

    SPACE       Play / Pause video

    A           Move backward 1 frame

    D           Move forward 1 frame

    J           Move backward 30 frames

    L           Move forward 30 frames

    0           Set EMPTY starting at current frame

    1           Set MARKER_ONLY starting at current frame

    2           Set POWER_ADAPTER_ONLY starting at current frame

    3           Set BOTH_PRESENT starting at current frame

    U           Undo the most recently recorded state change

    P           Print all recorded state changes in the terminal

    S           Save annotation progress

    Q           Finish annotation and create frame-level ground truth


OUTPUT
------

datasets/validation/ground_truth/<experiment_id>_ground_truth.csv


PROGRESS FILE
-------------

datasets/validation/ground_truth/<experiment_id>_ground_truth_progress.csv


IMPORTANT
---------

The program starts PAUSED.

The first state change MUST be assigned at frame 0.

Only press a state key when the real state changes.

Example:

    Frame 0     -> EMPTY
    Frame 120   -> MARKER_ONLY
    Frame 310   -> BOTH_PRESENT
    Frame 700   -> POWER_ADAPTER_ONLY

The program automatically assigns the correct state to every frame between
these change points.
"""


from pathlib import Path
import csv

import cv2


# ======================================================================
# PROJECT CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALIDATION_VIDEO_FOLDER = (
    PROJECT_ROOT
    / "datasets"
    / "validation"
    / "videos"
)

GROUND_TRUTH_FOLDER = (
    PROJECT_ROOT
    / "datasets"
    / "validation"
    / "ground_truth"
)


# ======================================================================
# STATE DEFINITIONS
# ======================================================================

STATE_DEFINITIONS = {
    0: {
        "state": "EMPTY",
        "marker_present": False,
        "power_adapter_present": False,
    },
    1: {
        "state": "MARKER_ONLY",
        "marker_present": True,
        "power_adapter_present": False,
    },
    2: {
        "state": "POWER_ADAPTER_ONLY",
        "marker_present": False,
        "power_adapter_present": True,
    },
    3: {
        "state": "BOTH_PRESENT",
        "marker_present": True,
        "power_adapter_present": True,
    },
}


# ======================================================================
# VISUALIZATION SETTINGS
# ======================================================================

WINDOW_NAME = "Prototype V2 - Validation Ground Truth Creator"

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def print_header(title):
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


def print_startup_instructions():
    print_header("GROUND-TRUTH CREATOR - READ THIS FIRST")

    print("What this program does:")
    print("  It helps you create the correct answer for every frame of a")
    print("  controlled validation video.")
    print()
    print("How to use it:")
    print("  1) Select the validation video from the list.")
    print("  2) The video opens paused at frame 0.")
    print("  3) Press 0, 1, 2, or 3 only when the real state changes.")
    print("  4) Use SPACE to play or pause.")
    print("  5) Use A/D to move 1 frame backward or forward.")
    print("  6) Use J/L to move 30 frames backward or forward.")
    print("  7) Use U to undo the most recent state change.")
    print("  8) Use P to print the recorded change points in the terminal.")
    print("  9) Use S to save progress.")
    print(" 10) Use Q to finish and create the final CSV.")
    print()
    print("Important:")
    print("  - Do NOT keep pressing the same state key repeatedly on the")
    print("    same frame.")
    print("  - Press the key once when the state really changes.")
    print("  - The first state must be entered at frame 0.")
    print("  - If you are paused on the same frame, use A/D or SPACE to move.")
    print()
    input("Press Enter to continue...")


def discover_validation_videos():
    if not VALIDATION_VIDEO_FOLDER.exists():
        raise FileNotFoundError(
            f"Validation video folder not found:\n"
            f"{VALIDATION_VIDEO_FOLDER}"
        )

    videos = sorted(
        list(VALIDATION_VIDEO_FOLDER.glob("*.mp4"))
        + list(VALIDATION_VIDEO_FOLDER.glob("*.avi"))
        + list(VALIDATION_VIDEO_FOLDER.glob("*.mov"))
        + list(VALIDATION_VIDEO_FOLDER.glob("*.mkv"))
    )

    return videos


def choose_validation_video(videos):
    print_header("AVAILABLE VALIDATION VIDEOS")

    if not videos:
        raise FileNotFoundError(
            "No validation videos were found in:\n"
            f"{VALIDATION_VIDEO_FOLDER}"
        )

    for index, video_path in enumerate(videos, start=1):
        print(f"{index:3d}. {video_path.name}")

    while True:
        print()
        user_input = input(
            "Enter validation video number: "
        ).strip()

        try:
            selected_number = int(user_input)
        except ValueError:
            print("Please enter a valid video number.")
            continue

        if not (1 <= selected_number <= len(videos)):
            print("Selected video number is outside the available range.")
            continue

        return videos[selected_number - 1]


def determine_experiment_id(video_path):
    """
    Example:

        validation_01_correct_sequence.mp4

    becomes:

        validation_01
    """
    parts = video_path.stem.split("_")

    if (
        len(parts) >= 2
        and parts[0].lower() == "validation"
        and parts[1].isdigit()
    ):
        return f"{parts[0]}_{parts[1]}"

    return video_path.stem


def read_video_frame(
    video_capture,
    frame_number,
):
    video_capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number,
    )

    read_success, frame = video_capture.read()

    return read_success, frame


def save_progress(
    progress_file,
    source_video_name,
    total_frames,
    fps,
    change_points,
):
    GROUND_TRUTH_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        progress_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        fieldnames = [
            "source_video",
            "total_video_frames",
            "video_fps",
            "change_frame",
            "state_id",
            "assembly_state_gt",
            "marker_present_gt",
            "power_adapter_present_gt",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for frame_number in sorted(change_points):
            state_id = change_points[frame_number]

            state_information = STATE_DEFINITIONS[state_id]

            writer.writerow({
                "source_video": source_video_name,
                "total_video_frames": total_frames,
                "video_fps": round(fps, 6),
                "change_frame": frame_number,
                "state_id": state_id,
                "assembly_state_gt": state_information["state"],
                "marker_present_gt": state_information["marker_present"],
                "power_adapter_present_gt": state_information["power_adapter_present"],
            })


def load_progress(
    progress_file,
    source_video_name,
    total_frames,
):
    change_points = {}

    if not progress_file.exists():
        return change_points

    with open(
        progress_file,
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    if not rows:
        return change_points

    saved_video_name = rows[0].get("source_video", "").strip()

    if saved_video_name != source_video_name:
        raise ValueError(
            "Existing progress file belongs to a different video.\n\n"
            f"Progress video:\n{saved_video_name}\n\n"
            f"Selected video:\n{source_video_name}"
        )

    for row in rows:
        frame_number = int(row["change_frame"])
        state_id = int(row["state_id"])

        if not (0 <= frame_number < total_frames):
            raise ValueError(
                "Progress file contains an invalid frame number:\n"
                f"{frame_number}"
            )

        if state_id not in STATE_DEFINITIONS:
            raise ValueError(
                "Progress file contains an invalid state ID:\n"
                f"{state_id}"
            )

        change_points[frame_number] = state_id

    return change_points


def print_change_points(change_points):
    print_header("RECORDED GROUND-TRUTH STATE CHANGES")

    if not change_points:
        print("No state changes recorded yet.")
        return

    for frame_number in sorted(change_points):
        state_id = change_points[frame_number]
        state_name = STATE_DEFINITIONS[state_id]["state"]

        print(
            f"Frame {frame_number:6d} -> "
            f"{state_id} | "
            f"{state_name}"
        )

    print()
    print(f"Total recorded change points: {len(change_points)}")


def get_active_state(
    frame_number,
    change_points,
):
    valid_change_frames = [
        change_frame
        for change_frame in change_points
        if change_frame <= frame_number
    ]

    if not valid_change_frames:
        return None

    active_change_frame = max(valid_change_frames)

    return change_points[active_change_frame]


def draw_text_with_background(
    frame,
    text,
    x,
    y,
    font_scale=0.65,
    thickness=2,
):
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        FONT,
        font_scale,
        thickness,
    )

    cv2.rectangle(
        frame,
        (
            x - 5,
            y - text_height - 7,
        ),
        (
            x + text_width + 5,
            y + baseline + 5,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        FONT,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def create_display_frame(
    original_frame,
    frame_number,
    total_frames,
    fps,
    playing,
    active_state_id,
    change_points,
):
    display_frame = original_frame.copy()

    height, width = display_frame.shape[:2]

    overlay = display_frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (width, 155),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.65,
        display_frame,
        0.35,
        0,
        display_frame,
    )

    time_seconds = frame_number / fps

    play_status = "PLAYING" if playing else "PAUSED"

    if active_state_id is None:
        state_text = "UNASSIGNED - SET FRAME 0 STATE"
    else:
        state_text = STATE_DEFINITIONS[active_state_id]["state"]

    cv2.putText(
        display_frame,
        "Prototype V2 Validation Ground Truth",
        (20, 30),
        FONT,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        display_frame,
        (
            f"Frame: {frame_number}/{total_frames - 1} | "
            f"Time: {time_seconds:.3f} s | "
            f"{play_status}"
        ),
        (20, 65),
        FONT,
        0.60,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        display_frame,
        f"TRUE STATE: {state_text}",
        (20, 105),
        FONT,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        display_frame,
        f"Recorded change points: {len(change_points)}",
        (20, 140),
        FONT,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    controls_text = (
        "0 EMPTY | 1 MARKER | 2 ADAPTER | 3 BOTH | "
        "SPACE Play/Pause | A/D +/-1 | J/L +/-30 | "
        "U Undo | S Save | P Print | Q Finish"
    )

    draw_text_with_background(
        display_frame,
        controls_text,
        15,
        height - 20,
        font_scale=0.48,
        thickness=1,
    )

    return display_frame


def create_ground_truth_csv(
    output_file,
    source_video_name,
    total_frames,
    fps,
    change_points,
):
    if 0 not in change_points:
        raise ValueError(
            "Frame 0 does not have a ground-truth state.\n\n"
            "Go to frame 0 and press one of:\n"
            "0 = EMPTY\n"
            "1 = MARKER_ONLY\n"
            "2 = POWER_ADAPTER_ONLY\n"
            "3 = BOTH_PRESENT"
        )

    sorted_change_frames = sorted(change_points)

    GROUND_TRUTH_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "source_video",
        "frame_number",
        "time_seconds",
        "marker_present_gt",
        "power_adapter_present_gt",
        "assembly_state_gt",
        "state_id_gt",
    ]

    rows_written = 0
    active_change_index = 0
    active_state_id = change_points[sorted_change_frames[0]]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for frame_number in range(total_frames):
            while (
                active_change_index + 1 < len(sorted_change_frames)
                and frame_number >= sorted_change_frames[active_change_index + 1]
            ):
                active_change_index += 1
                active_state_id = change_points[
                    sorted_change_frames[active_change_index]
                ]

            state_information = STATE_DEFINITIONS[active_state_id]

            writer.writerow({
                "source_video": source_video_name,
                "frame_number": frame_number,
                "time_seconds": round(frame_number / fps, 6),
                "marker_present_gt": state_information["marker_present"],
                "power_adapter_present_gt": state_information["power_adapter_present"],
                "assembly_state_gt": state_information["state"],
                "state_id_gt": active_state_id,
            })

            rows_written += 1

    return rows_written


def verify_ground_truth_csv(
    output_file,
    source_video_name,
    total_frames,
):
    with open(
        output_file,
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    if len(rows) != total_frames:
        raise ValueError(
            "Ground-truth row count does not match video frame count.\n\n"
            f"Video frames      : {total_frames}\n"
            f"Ground-truth rows : {len(rows)}"
        )

    expected_frames = list(range(total_frames))

    actual_frames = [
        int(row["frame_number"])
        for row in rows
    ]

    if actual_frames != expected_frames:
        raise ValueError(
            "Ground-truth frame numbering is incomplete or out of order."
        )

    source_videos = {
        row["source_video"]
        for row in rows
    }

    if source_videos != {source_video_name}:
        raise ValueError(
            "Ground-truth CSV contains an unexpected source video."
        )

    return len(rows)


def record_state_change(
    change_points,
    current_frame_number,
    state_id,
):
    previous_value = change_points.get(current_frame_number)

    if previous_value == state_id:
        return "UNCHANGED"

    change_points[current_frame_number] = state_id

    if previous_value is None:
        return "RECORDED"

    return "UPDATED"


# ======================================================================
# MAIN PROCESS
# ======================================================================

def main():
    print_startup_instructions()

    print_header(
        "PROTOTYPE V2 - VALIDATION GROUND-TRUTH CREATOR"
    )

    print(
        f"Validation video folder:\n{VALIDATION_VIDEO_FOLDER}"
    )

    print()

    print(
        f"Ground-truth folder:\n{GROUND_TRUTH_FOLDER}"
    )

    # ==================================================================
    # SELECT VALIDATION VIDEO
    # ==================================================================

    videos = discover_validation_videos()

    selected_video = choose_validation_video(videos)

    experiment_id = determine_experiment_id(selected_video)

    output_file = (
        GROUND_TRUTH_FOLDER / f"{experiment_id}_ground_truth.csv"
    )

    progress_file = (
        GROUND_TRUTH_FOLDER / f"{experiment_id}_ground_truth_progress.csv"
    )

    print_header("SELECTED VALIDATION EXPERIMENT")

    print(f"Experiment ID:\n{experiment_id}")

    print()

    print(f"Validation video:\n{selected_video}")

    print()

    print(f"Ground-truth output:\n{output_file}")

    print()

    print(f"Progress file:\n{progress_file}")

    # ==================================================================
    # OPEN VIDEO
    # ==================================================================

    video_capture = cv2.VideoCapture(str(selected_video))

    if not video_capture.isOpened():
        raise RuntimeError(
            f"Could not open validation video:\n{selected_video}"
        )

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        video_capture.release()
        raise ValueError(f"Invalid video FPS: {fps}")

    if total_frames <= 0:
        video_capture.release()
        raise ValueError(f"Invalid video frame count: {total_frames}")

    print_header("VALIDATION VIDEO INFORMATION")

    print(f"FPS          : {fps:.3f}")
    print(f"Resolution   : {width} x {height}")
    print(f"Total frames : {total_frames}")
    print(f"Duration     : {total_frames / fps:.3f} seconds")

    # ==================================================================
    # LOAD EXISTING PROGRESS
    # ==================================================================

    change_points = load_progress(
        progress_file,
        selected_video.name,
        total_frames,
    )

    if change_points:
        print_header("EXISTING ANNOTATION PROGRESS FOUND")
        print(f"Loaded change points: {len(change_points)}")
        print_change_points(change_points)
    else:
        print_header("STARTING NEW GROUND-TRUTH ANNOTATION")
        print("No previous annotation progress was found.")
        print()
        print("The video starts PAUSED.")
        print()
        print("IMPORTANT: Set the true state at frame 0 first.")
        print()
        print("Recommended flow:")
        print("  1) Press 0, 1, 2, or 3 at frame 0.")
        print("  2) Press SPACE to play.")
        print("  3) Pause near a change with SPACE.")
        print("  4) Use A/D or J/L to move.")
        print("  5) Press the new state key once at the change frame.")
        print("  6) Press Q when finished.")

    # ==================================================================
    # ANNOTATION LOOP
    # ==================================================================

    current_frame_number = 0
    playing = False
    annotation_finished = False

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    while True:
        read_success, frame = read_video_frame(
            video_capture,
            current_frame_number,
        )

        if not read_success:
            print()
            print(
                f"WARNING: Could not read frame {current_frame_number}."
            )
            break

        active_state_id = get_active_state(
            current_frame_number,
            change_points,
        )

        display_frame = create_display_frame(
            original_frame=frame,
            frame_number=current_frame_number,
            total_frames=total_frames,
            fps=fps,
            playing=playing,
            active_state_id=active_state_id,
            change_points=change_points,
        )

        cv2.imshow(
            WINDOW_NAME,
            display_frame,
        )

        wait_time = (
            max(1, int(round(1000 / fps)))
            if playing
            else 0
        )

        key = cv2.waitKey(wait_time) & 0xFF

        # ==============================================================
        # PLAY / PAUSE
        # ==============================================================

        if key == ord(" "):
            playing = not playing

        # ==============================================================
        # MOVE BACKWARD ONE FRAME
        # ==============================================================

        elif key in (ord("a"), ord("A")):
            playing = False
            current_frame_number = max(0, current_frame_number - 1)

        # ==============================================================
        # MOVE FORWARD ONE FRAME
        # ==============================================================

        elif key in (ord("d"), ord("D")):
            playing = False
            current_frame_number = min(total_frames - 1, current_frame_number + 1)

        # ==============================================================
        # MOVE BACKWARD 30 FRAMES
        # ==============================================================

        elif key in (ord("j"), ord("J")):
            playing = False
            current_frame_number = max(0, current_frame_number - 30)

        # ==============================================================
        # MOVE FORWARD 30 FRAMES
        # ==============================================================

        elif key in (ord("l"), ord("L")):
            playing = False
            current_frame_number = min(total_frames - 1, current_frame_number + 30)

        # ==============================================================
        # ASSIGN GROUND-TRUTH STATE
        # ==============================================================

        elif key in (ord("0"), ord("1"), ord("2"), ord("3")):
            state_id = int(chr(key))
            state_name = STATE_DEFINITIONS[state_id]["state"]

            action = record_state_change(
                change_points,
                current_frame_number,
                state_id,
            )

            if action != "UNCHANGED":
                save_progress(
                    progress_file=progress_file,
                    source_video_name=selected_video.name,
                    total_frames=total_frames,
                    fps=fps,
                    change_points=change_points,
                )

                print()
                if action == "RECORDED":
                    print(
                        f"Recorded state change: Frame {current_frame_number} -> {state_name}"
                    )
                else:
                    print(
                        f"Updated state change: Frame {current_frame_number} -> {state_name}"
                    )
            else:
                print()
                print(
                    f"Frame {current_frame_number} is already marked as {state_name}."
                )
                print(
                    "Move to a different frame with A/D or J/L, or press SPACE to play."
                )

            # Keep the video paused after marking so the user can verify
            # the selected frame before moving on.
            playing = False

        # ==============================================================
        # UNDO MOST RECENT CHANGE POINT
        # ==============================================================

        elif key in (ord("u"), ord("U")):
            playing = False

            if not change_points:
                print()
                print("Nothing to undo.")
            else:
                most_recent_frame = max(change_points)
                removed_state_id = change_points.pop(most_recent_frame)
                removed_state_name = STATE_DEFINITIONS[removed_state_id]["state"]

                save_progress(
                    progress_file=progress_file,
                    source_video_name=selected_video.name,
                    total_frames=total_frames,
                    fps=fps,
                    change_points=change_points,
                )

                current_frame_number = most_recent_frame

                print()
                print(
                    f"Removed state change: Frame {most_recent_frame} -> {removed_state_name}"
                )

        # ==============================================================
        # PRINT CHANGE POINTS
        # ==============================================================

        elif key in (ord("p"), ord("P")):
            playing = False
            print_change_points(change_points)

        # ==============================================================
        # SAVE PROGRESS
        # ==============================================================

        elif key in (ord("s"), ord("S")):
            playing = False

            save_progress(
                progress_file=progress_file,
                source_video_name=selected_video.name,
                total_frames=total_frames,
                fps=fps,
                change_points=change_points,
            )

            print()
            print(f"Annotation progress saved:\n{progress_file}")

        # ==============================================================
        # FINISH ANNOTATION
        # ==============================================================

        elif key in (ord("q"), ord("Q")):
            playing = False

            if 0 not in change_points:
                print()
                print("ANNOTATION CANNOT FINISH.")
                print()
                print("Frame 0 must have a true state.")
                print()
                print("Move to frame 0 and press 0, 1, 2, or 3.")
                continue

            print_change_points(change_points)

            print()
            confirmation = input(
                "Create final frame-level ground truth? [Y/N]: "
            ).strip().lower()

            if confirmation == "y":
                annotation_finished = True
                break

            print()
            print("Returning to annotation.")

        # ==============================================================
        # AUTOMATIC PLAYBACK ADVANCE
        # ==============================================================

        if playing:
            if current_frame_number < total_frames - 1:
                current_frame_number += 1
            else:
                playing = False

    # ==================================================================
    # CLOSE VIDEO
    # ==================================================================

    video_capture.release()
    cv2.destroyAllWindows()

    # ==================================================================
    # SAVE PROGRESS BEFORE EXIT
    # ==================================================================

    save_progress(
        progress_file=progress_file,
        source_video_name=selected_video.name,
        total_frames=total_frames,
        fps=fps,
        change_points=change_points,
    )

    if not annotation_finished:
        print_header("ANNOTATION SESSION ENDED")
        print("Final ground truth was not created.")
        print()
        print(f"Annotation progress was saved to:\n{progress_file}")
        print()
        print("Run the script again to continue.")
        return

    # ==================================================================
    # CREATE FRAME-LEVEL GROUND TRUTH
    # ==================================================================

    print_header("CREATING FRAME-LEVEL GROUND TRUTH")

    rows_written = create_ground_truth_csv(
        output_file=output_file,
        source_video_name=selected_video.name,
        total_frames=total_frames,
        fps=fps,
        change_points=change_points,
    )

    print(f"Ground-truth rows created: {rows_written}")

    # ==================================================================
    # VERIFY GROUND TRUTH
    # ==================================================================

    print_header("VERIFYING GROUND-TRUTH DATASET")

    verified_rows = verify_ground_truth_csv(
        output_file=output_file,
        source_video_name=selected_video.name,
        total_frames=total_frames,
    )

    print(f"Video frames              : {total_frames}")
    print(f"Ground-truth rows         : {verified_rows}")
    print(f"Recorded state changes    : {len(change_points)}")
    print("Frame numbering complete  : YES")
    print("Video / ground-truth match : YES")

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================

    print_header("GROUND-TRUTH CREATION SUMMARY")

    print(f"Experiment              : {experiment_id}")
    print(f"Validation video        : {selected_video.name}")
    print(f"Video frames            : {total_frames}")
    print(f"Ground-truth rows       : {verified_rows}")
    print(f"State change points     : {len(change_points)}")
    print()
    print(f"Ground-truth file saved to:\n{output_file}")

    print_header("STATUS: VALIDATION GROUND TRUTH CREATED SUCCESSFULLY")

    print("Next stage:")
    print(
        "Compare frame-level ground truth against the saved validation predictions and calculate quantitative metrics."
    )


# ======================================================================
# SCRIPT ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()