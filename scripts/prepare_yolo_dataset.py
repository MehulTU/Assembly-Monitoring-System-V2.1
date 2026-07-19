'''
HOW TO RUN THE FILE:
    python scripts/prepare_yolo_dataset.py (TYPE THIS IN TERMINAL)

Prototype V2 - Trial-Based YOLO Dataset Preparation (Auto-Split)

What changed vs. the previous version:
    - The hard-coded TRIAL_ID_TO_NUMBER dictionary is GONE.
    - The hard-coded TRAIN/VAL/TEST trial number sets are GONE.
    - Trials are discovered automatically from final_manifest.csv.
    - Trials are split into train/val/test by ratio (70/15/15).

The script now works with 15, 30, 60, or 200 trials without
editing the code again.

Splitting stays TRIAL-BASED: all images from one trial always land
in the same split, so there is no trial leakage between splits.
'''

from pathlib import Path
import csv
import shutil
import yaml
from collections import Counter, defaultdict


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FINAL_DIR = PROJECT_ROOT / "datasets" / "final"
FINAL_IMAGES_DIR = FINAL_DIR / "images"
FINAL_MANIFEST_PATH = FINAL_DIR / "final_manifest.csv"

ANNOTATION_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "annotations_export"
    / "extracted"
)

OUTPUT_DIR = PROJECT_ROOT / "datasets" / "yolo_dataset_v1"

DATA_YAML_PATH = OUTPUT_DIR / "data.yaml"


# ============================================================
# DATASET SETTINGS
# ============================================================

CLASS_NAMES = {
    0: "marker",
    1: "power_adapter",
}


# ============================================================
# AUTOMATIC TRIAL SPLITTING
# ============================================================
# Trials are discovered automatically from the manifest and
# split by these ratios. All images of one trial always stay
# in the same split (no trial leakage).

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(title):
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


def load_csv(path):
    with open(
        path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        return list(csv.DictReader(file))


def find_column(rows, candidates):
    if not rows:
        raise ValueError("CSV file is empty.")

    columns = rows[0].keys()

    for candidate in candidates:
        if candidate in columns:
            return candidate

    raise ValueError(
        f"Could not find any of these columns: {candidates}\n"
        f"Available columns: {list(columns)}"
    )


def get_image_files():
    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    return {
        path.name: path
        for path in FINAL_IMAGES_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in extensions
    }


def find_label_files():
    label_files = {}

    for path in ANNOTATION_DIR.rglob("*.txt"):

        if path.name.lower() in {
            "train.txt",
            "valid.txt",
            "test.txt",
            "obj.names",
        }:
            continue

        label_files[path.stem] = path

    return label_files


def determine_split(trial_index, total_trials):
    """
    Assign a trial to train / val / test based on its index
    in the sorted list of unique trial IDs.

    Because trial IDs are timestamp-based, sorting them gives
    chronological order, and the split is fully deterministic:
    running the script twice always produces the same split.
    """

    train_end = int(total_trials * TRAIN_RATIO)
    val_end = train_end + int(total_trials * VAL_RATIO)

    if trial_index < train_end:
        return "train"

    elif trial_index < val_end:
        return "val"

    else:
        return "test"


def count_boxes(label_path):
    class_counts = Counter()
    total_boxes = 0

    if not label_path.exists():
        return total_boxes, class_counts

    with open(label_path, "r", encoding="utf-8") as file:

        for line in file:

            stripped = line.strip()

            if not stripped:
                continue

            parts = stripped.split()

            if len(parts) != 5:
                raise ValueError(
                    f"Malformed YOLO annotation:\n{label_path}\n"
                    f"Line: {line}"
                )

            class_id = int(parts[0])

            total_boxes += 1
            class_counts[class_id] += 1

    return total_boxes, class_counts


def create_output_structure():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(
            parents=True,
            exist_ok=True,
        )

        (OUTPUT_DIR / "labels" / split).mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "PROTOTYPE V2 - TRIAL-BASED YOLO DATASET PREPARATION"
    )

    print(f"Final manifest:\n{FINAL_MANIFEST_PATH}")
    print()
    print(f"Final images:\n{FINAL_IMAGES_DIR}")
    print()
    print(f"Annotations:\n{ANNOTATION_DIR}")
    print()
    print(f"Output dataset:\n{OUTPUT_DIR}")

    # --------------------------------------------------------
    # LOAD INPUT DATA
    # --------------------------------------------------------

    print_header("LOADING DATA")

    manifest_rows = load_csv(FINAL_MANIFEST_PATH)

    image_column = find_column(
        manifest_rows,
        [
            "image_filename",
            "image_name",
            "filename",
            "file_name",
        ],
    )

    trial_id_column = find_column(
        manifest_rows,
        ["trial_id"],
    )

    # --------------------------------------------------------
    # AUTOMATIC TRIAL DISCOVERY
    # --------------------------------------------------------

    unique_trial_ids = sorted(
        {
            row[trial_id_column].strip()
            for row in manifest_rows
        }
    )

    trial_id_to_index = {
        trial_id: index
        for index, trial_id in enumerate(unique_trial_ids)
    }

    image_files = get_image_files()
    label_files = find_label_files()

    print(f"Manifest rows       : {len(manifest_rows)}")
    print(f"Detected trials     : {len(unique_trial_ids)}")
    print(f"Final image files   : {len(image_files)}")
    print(f"Exported label files: {len(label_files)}")

    print(f"Image column        : {image_column}")
    print(f"Trial ID column     : {trial_id_column}")

    # --------------------------------------------------------
    # CREATE DATASET
    # --------------------------------------------------------

    print_header("CREATING TRAIN / VAL / TEST DATASET")

    create_output_structure()

    split_image_counts = Counter()
    split_box_counts = Counter()

    split_class_counts = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }

    split_trial_ids = {
        "train": set(),
        "val": set(),
        "test": set(),
    }

    split_trial_numbers = {
        "train": set(),
        "val": set(),
        "test": set(),
    }

    empty_images = Counter()

    missing_images = []
    copied_images = set()

    for row in manifest_rows:

        image_name = Path(row[image_column]).name

        trial_id = row[trial_id_column].strip()

        trial_index = trial_id_to_index[trial_id]

        split = determine_split(
            trial_index,
            len(unique_trial_ids),
        )

        trial_number = trial_index + 1

        image_path = image_files.get(image_name)

        if image_path is None:
            missing_images.append(image_name)
            continue

        if image_name in copied_images:
            raise ValueError(
                f"Duplicate image in manifest: {image_name}"
            )

        copied_images.add(image_name)

        split_trial_ids[split].add(trial_id)
        split_trial_numbers[split].add(trial_number)

        destination_image = (
            OUTPUT_DIR
            / "images"
            / split
            / image_name
        )

        destination_label = (
            OUTPUT_DIR
            / "labels"
            / split
            / f"{image_path.stem}.txt"
        )

        shutil.copy2(
            image_path,
            destination_image,
        )

        source_label = label_files.get(image_path.stem)

        if source_label is None:

            destination_label.touch()
            empty_images[split] += 1

        else:

            shutil.copy2(
                source_label,
                destination_label,
            )

        total_boxes, class_counts = count_boxes(
            destination_label
        )

        split_image_counts[split] += 1
        split_box_counts[split] += total_boxes

        split_class_counts[split].update(
            class_counts
        )

    # --------------------------------------------------------
    # VERIFY TRIAL LEAKAGE
    # --------------------------------------------------------

    print_header("TRIAL LEAKAGE CHECK")

    train_val_overlap = (
        split_trial_ids["train"]
        & split_trial_ids["val"]
    )

    train_test_overlap = (
        split_trial_ids["train"]
        & split_trial_ids["test"]
    )

    val_test_overlap = (
        split_trial_ids["val"]
        & split_trial_ids["test"]
    )

    leakage_detected = bool(
        train_val_overlap
        or train_test_overlap
        or val_test_overlap
    )

    print(
        f"Train / Val trial overlap : "
        f"{len(train_val_overlap)}"
    )

    print(
        f"Train / Test trial overlap: "
        f"{len(train_test_overlap)}"
    )

    print(
        f"Val / Test trial overlap  : "
        f"{len(val_test_overlap)}"
    )

    print(
        f"Trial leakage detected    : "
        f"{'YES' if leakage_detected else 'NO'}"
    )

    # --------------------------------------------------------
    # CREATE DATA.YAML
    # --------------------------------------------------------

    yaml_data = {
        "path": str(OUTPUT_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASS_NAMES,
    }

    with open(
        DATA_YAML_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        yaml.safe_dump(
            yaml_data,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    # --------------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------------

    print_header("DATASET SPLIT SUMMARY")

    for split in ["train", "val", "test"]:

        print()
        print(split.upper())

        print(
            f"  Trials       : "
            f"{sorted(split_trial_numbers[split])}"
        )

        print(
            f"  Trial count  : "
            f"{len(split_trial_ids[split])}"
        )

        print(
            f"  Images       : "
            f"{split_image_counts[split]}"
        )

        print(
            f"  Empty images : "
            f"{empty_images[split]}"
        )

        print(
            f"  Total boxes  : "
            f"{split_box_counts[split]}"
        )

        for class_id, class_name in CLASS_NAMES.items():

            print(
                f"  {class_name:<13}: "
                f"{split_class_counts[split][class_id]} boxes"
            )

    # --------------------------------------------------------
    # FINAL VERIFICATION
    # --------------------------------------------------------

    print_header("FINAL VERIFICATION")

    total_output_images = sum(
        split_image_counts.values()
    )

    total_output_boxes = sum(
        split_box_counts.values()
    )

    total_empty_images = sum(
        empty_images.values()
    )

    print(f"Input manifest rows      : {len(manifest_rows)}")
    print(f"Detected trials          : {len(unique_trial_ids)}")
    print(f"Output dataset images    : {total_output_images}")
    print(f"Output bounding boxes    : {total_output_boxes}")
    print(f"Images without labels    : {total_empty_images}")
    print(f"Missing source images    : {len(missing_images)}")
    print(f"Trial leakage detected   : {'YES' if leakage_detected else 'NO'}")
    print(f"Dataset YAML             : {DATA_YAML_PATH}")

    critical_problem = (
        len(missing_images) > 0
        or total_output_images != len(manifest_rows)
        or leakage_detected
    )

    # --------------------------------------------------------
    # ANNOTATION COVERAGE WARNING
    # --------------------------------------------------------
    # Empty label files are legitimate for genuine negative
    # images (e.g. empty_workspace trials). But if MOST images
    # have no labels, the annotations are probably out of date
    # and training would produce a weak model.

    if (
        total_output_images > 0
        and total_empty_images / total_output_images > 0.30
    ):

        print()
        print("WARNING: More than 30% of the dataset images have")
        print("no annotations. If these are not intentional negative")
        print("images, re-export the updated annotations from CVAT")
        print("before training.")

    print_header("STATUS")

    if critical_problem:

        print("STATUS: DATASET PREPARATION FAILED")
        print()
        print("Do not start YOLO training.")

    else:

        print("STATUS: YOLO DATASET CREATED SUCCESSFULLY")

        print()
        print("Next stage:")
        print("Install Ultralytics YOLO and train the baseline model.")


if __name__ == "__main__":
    main()