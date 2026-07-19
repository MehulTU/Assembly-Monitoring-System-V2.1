"""
run_validation_experiment.py

HOW TO RUN:

    python scripts\run_validation_experiment.py

PURPOSE:

Run one controlled validation video through the complete existing
Prototype V2 offline process-monitoring pipeline.

IMPORTANT:

The validation video is passed directly to run_offline_inference.py.

The script does NOT:

    - overwrite any raw training/data-acquisition video
    - modify the YOLO training dataset
    - retrain YOLO
    - require manual video selection

The script:

1. Verifies the validation video.
2. Backs up the normal analytics directory.
3. Clears the normal analytics directory.
4. Runs YOLO inference directly on the validation video.
5. Runs the existing temporal timeline script.
6. Runs the existing state machine.
7. Runs the existing process analytics script.
8. Runs the existing visualization script.
9. Copies all outputs into the validation experiment result directory.
10. Saves experiment metadata.
11. Restores the original analytics directory.
"""

from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime


# ===========================================================================
# PROJECT PATHS
# ===========================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCRIPTS_DIR = PROJECT_ROOT / "scripts"

VALIDATION_DIR = PROJECT_ROOT / "datasets" / "validation"

VALIDATION_VIDEO_DIR = VALIDATION_DIR / "videos"

VALIDATION_RESULTS_DIR = VALIDATION_DIR / "results"

ANALYTICS_DIR = PROJECT_ROOT / "datasets" / "analytics"


# ===========================================================================
# VALIDATION EXPERIMENT SETTINGS
# ===========================================================================

EXPERIMENT_NAME = "validation_01"

VALIDATION_VIDEO_NAME = "validation_01_correct_sequence.mp4"

VALIDATION_VIDEO_PATH = (
    VALIDATION_VIDEO_DIR / VALIDATION_VIDEO_NAME
)

EXPERIMENT_RESULTS_DIR = (
    VALIDATION_RESULTS_DIR / EXPERIMENT_NAME
)


# ===========================================================================
# PIPELINE SCRIPTS
# ===========================================================================

OFFLINE_INFERENCE_SCRIPT = "run_offline_inference.py"

FOLLOWING_PIPELINE_SCRIPTS = [

    "build_detection_timeline.py",

    "run_state_machine.py",

    "process_analytics.py",

    "visualize_process_video.py",
]


# ===========================================================================
# EXPECTED OUTPUT FILES
# ===========================================================================

EXPECTED_OUTPUTS = [

    "raw_detections.csv",

    "inference_summary.csv",

    "detection_timeline.csv",

    "state_timeline.csv",

    "event_log.csv",

    "process_visualization.mp4",
]


# ===========================================================================
# TEMPORARY BACKUP
# ===========================================================================

TEMP_DIR = VALIDATION_DIR / "_temporary_validation_backup"

ANALYTICS_BACKUP_DIR = TEMP_DIR / "analytics_backup"


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def print_section(title):

    print()

    print("=" * 75)

    print(title)

    print("=" * 75)


def clear_directory(directory):

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for item in directory.iterdir():

        if item.is_file():

            item.unlink()

        elif item.is_dir():

            shutil.rmtree(item)


def copy_directory_contents(source, destination):

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not source.exists():

        return


    for item in source.iterdir():

        destination_path = destination / item.name

        if item.is_file():

            shutil.copy2(
                item,
                destination_path,
            )

        elif item.is_dir():

            shutil.copytree(
                item,
                destination_path,
            )


def run_command(command, stage_name):

    print()

    print(f"Running stage: {stage_name}")

    print("-" * 75)

    print("Command:")

    print(" ".join(str(value) for value in command))

    print()


    result = subprocess.run(

        command,

        cwd=PROJECT_ROOT,

        check=False,
    )


    if result.returncode != 0:

        raise RuntimeError(

            f"Validation pipeline failed during stage:\n"
            f"{stage_name}"
        )


    print()

    print(f"Completed stage: {stage_name}")


def verify_script(script_name):

    script_path = SCRIPTS_DIR / script_name

    if not script_path.exists():

        raise FileNotFoundError(

            f"Pipeline script not found:\n{script_path}"
        )

    return script_path


# ===========================================================================
# MAIN PROGRAM
# ===========================================================================

def main():

    print_section(
        "PROTOTYPE V2 - CONTROLLED VALIDATION EXPERIMENT RUNNER"
    )


    print(f"Experiment:\n{EXPERIMENT_NAME}")

    print()

    print(f"Validation video:\n{VALIDATION_VIDEO_PATH}")

    print()

    print(f"Experiment results:\n{EXPERIMENT_RESULTS_DIR}")


    # =======================================================================
    # VERIFY VALIDATION INPUT
    # =======================================================================

    print_section("VERIFYING VALIDATION INPUTS")


    if not VALIDATION_VIDEO_PATH.exists():

        raise FileNotFoundError(

            f"Validation video not found:\n"
            f"{VALIDATION_VIDEO_PATH}"
        )


    print("Validation video found.")


    offline_inference_path = verify_script(
        OFFLINE_INFERENCE_SCRIPT
    )


    print(
        f"Found pipeline script: "
        f"{OFFLINE_INFERENCE_SCRIPT}"
    )


    for script_name in FOLLOWING_PIPELINE_SCRIPTS:

        verify_script(script_name)

        print(f"Found pipeline script: {script_name}")


    # =======================================================================
    # PREPARE TEMPORARY BACKUP
    # =======================================================================

    print_section("BACKING UP NORMAL ANALYTICS OUTPUTS")


    if TEMP_DIR.exists():

        shutil.rmtree(TEMP_DIR)


    ANALYTICS_BACKUP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    if ANALYTICS_DIR.exists():

        copy_directory_contents(

            ANALYTICS_DIR,

            ANALYTICS_BACKUP_DIR,
        )


    print("Normal analytics outputs protected.")


    # =======================================================================
    # PREPARE RESULT DIRECTORY
    # =======================================================================

    print_section("PREPARING VALIDATION RESULT DIRECTORY")


    if EXPERIMENT_RESULTS_DIR.exists():

        print(
            "Removing previous results for this experiment."
        )

        shutil.rmtree(EXPERIMENT_RESULTS_DIR)


    EXPERIMENT_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # =======================================================================
    # CLEAR NORMAL ANALYTICS DIRECTORY
    # =======================================================================

    print_section("PREPARING ANALYTICS WORKSPACE")


    clear_directory(ANALYTICS_DIR)


    print("Analytics workspace cleared.")


    # =======================================================================
    # RUN PIPELINE
    # =======================================================================

    pipeline_success = False

    pipeline_error = None


    try:

        print_section("RUNNING COMPLETE VALIDATION PIPELINE")


        # -------------------------------------------------------------------
        # STAGE 1 - OFFLINE YOLO INFERENCE
        # -------------------------------------------------------------------

        run_command(

            [

                sys.executable,

                str(offline_inference_path),

                "--video",

                str(VALIDATION_VIDEO_PATH),
            ],

            "1/5 - OFFLINE YOLO INFERENCE",
        )


        # -------------------------------------------------------------------
        # STAGES 2 TO 5
        # -------------------------------------------------------------------

        total_following_stages = len(
            FOLLOWING_PIPELINE_SCRIPTS
        )


        for index, script_name in enumerate(

            FOLLOWING_PIPELINE_SCRIPTS,

            start=2,

        ):

            script_path = SCRIPTS_DIR / script_name


            run_command(

                [

                    sys.executable,

                    str(script_path),
                ],

                (
                    f"{index}/"
                    f"{total_following_stages + 1} - "
                    f"{script_name}"
                ),
            )


        # ===================================================================
        # VERIFY AND COPY OUTPUTS
        # ===================================================================

        print_section("VERIFYING AND COPYING VALIDATION RESULTS")


        missing_outputs = []

        copied_outputs = 0


        for filename in EXPECTED_OUTPUTS:

            source_path = ANALYTICS_DIR / filename

            destination_path = (
                EXPERIMENT_RESULTS_DIR / filename
            )


            if source_path.exists():

                shutil.copy2(

                    source_path,

                    destination_path,
                )


                copied_outputs += 1


                print(f"Copied: {filename}")


            else:

                missing_outputs.append(filename)

                print(f"MISSING: {filename}")


        if missing_outputs:

            raise RuntimeError(

                "Validation pipeline completed, but expected "
                "output files are missing:\n"
                + "\n".join(missing_outputs)
            )


        # ===================================================================
        # SAVE EXPERIMENT INFORMATION
        # ===================================================================

        print_section("SAVING EXPERIMENT INFORMATION")


        experiment_info_path = (

            EXPERIMENT_RESULTS_DIR / "experiment_info.txt"
        )


        with experiment_info_path.open(

            "w",

            encoding="utf-8",

        ) as file:

            file.write(
                "PROTOTYPE V2 VALIDATION EXPERIMENT\n"
            )

            file.write("=" * 60 + "\n\n")


            file.write(
                f"Experiment name: {EXPERIMENT_NAME}\n"
            )


            file.write(
                f"Validation video: {VALIDATION_VIDEO_NAME}\n"
            )


            file.write(
                f"Validation video path: "
                f"{VALIDATION_VIDEO_PATH}\n"
            )


            file.write(
                f"Execution time: "
                f"{datetime.now().isoformat(timespec='seconds')}\n"
            )


            file.write(
                f"YOLO model path: "
                f"{MODEL_PATH_IF_AVAILABLE()}\n"
            )


            file.write(
                "Training dataset modified: NO\n"
            )


            file.write(
                "YOLO model retrained: NO\n"
            )


            file.write(
                "Manual video selection required: NO\n"
            )


            file.write(
                "Raw source video overwritten: NO\n"
            )


            file.write(
                f"Pipeline outputs copied: "
                f"{copied_outputs}\n"
            )


            file.write(
                "Missing expected outputs: 0\n"
            )


        print(f"Saved: {experiment_info_path.name}")


        pipeline_success = True


    except Exception as error:

        pipeline_error = error

        raise


    finally:

        # ===================================================================
        # RESTORE NORMAL ANALYTICS OUTPUTS
        # ===================================================================

        print_section("RESTORING NORMAL ANALYTICS OUTPUTS")


        clear_directory(ANALYTICS_DIR)


        if ANALYTICS_BACKUP_DIR.exists():

            copy_directory_contents(

                ANALYTICS_BACKUP_DIR,

                ANALYTICS_DIR,
            )


        print("Normal analytics outputs restored.")


        # ===================================================================
        # REMOVE TEMPORARY BACKUP
        # ===================================================================

        if TEMP_DIR.exists():

            shutil.rmtree(TEMP_DIR)


        print("Temporary backup directory removed.")


    # =======================================================================
    # FINAL SUMMARY
    # =======================================================================

    print_section("VALIDATION EXPERIMENT SUMMARY")


    print(f"Experiment              : {EXPERIMENT_NAME}")

    print(f"Validation video        : {VALIDATION_VIDEO_NAME}")

    print(f"Results directory       : {EXPERIMENT_RESULTS_DIR}")

    print(f"Pipeline completed      : {pipeline_success}")


    print()


    if pipeline_success:

        print(
            "STATUS: CONTROLLED VALIDATION EXPERIMENT "
            "COMPLETED SUCCESSFULLY"
        )

        print()

        print("Generated validation outputs:")


        for filename in EXPECTED_OUTPUTS:

            print(f"  - {filename}")


        print("  - experiment_info.txt")


        print()

        print("Next stage:")

        print(
            "Inspect process_visualization.mp4 and create "
            "frame-level ground truth for quantitative validation."
        )


# ===========================================================================
# MODEL PATH METADATA HELPER
# ===========================================================================

def MODEL_PATH_IF_AVAILABLE():

    MODEL_PATH = PROJECT_ROOT / "weights" / "best.pt"

    return str(MODEL_PATH)


# ===========================================================================
# PROGRAM ENTRY POINT
# ===========================================================================

if __name__ == "__main__":

    main()