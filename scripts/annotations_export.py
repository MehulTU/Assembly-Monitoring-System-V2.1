'''
HOW TO RUN THE FILE:
    python scripts\annotations_export.py (TYPE THIS IN TERMINAL)
'''

from pathlib import Path
import csv
import shutil
import zipfile
import yaml


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FINAL_DIR = PROJECT_ROOT / "datasets" / "final"
FINAL_IMAGES_DIR = FINAL_DIR / "images"
FINAL_MANIFEST_PATH = FINAL_DIR / "final_manifest.csv"

EXPORT_DIR = PROJECT_ROOT / "datasets" / "annotations_export"
EXTRACT_DIR = EXPORT_DIR / "extracted"

REPORT_PATH = EXPORT_DIR / "annotation_verification_report.csv"


# ============================================================
# EXPECTED DATASET SETTINGS
# ============================================================

EXPECTED_IMAGE_COUNT = 110

EXPECTED_CLASSES = {
    0: "marker",
    1: "power_adapter",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(title):
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


def find_zip_file():
    zip_files = list(EXPORT_DIR.glob("*.zip"))

    if len(zip_files) == 0:
        raise FileNotFoundError(
            f"No ZIP file found inside:\n{EXPORT_DIR}"
        )

    if len(zip_files) > 1:
        print("WARNING: More than one ZIP file found.")
        print("Using the newest ZIP file.")

        zip_files.sort(
            key=lambda path: path.stat().st_mtime,
            reverse=True
        )

    return zip_files[0]


def extract_zip(zip_path):
    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(EXTRACT_DIR)


def find_data_yaml():
    yaml_files = (
        list(EXTRACT_DIR.rglob("data.yaml"))
        + list(EXTRACT_DIR.rglob("dataset.yaml"))
    )

    if not yaml_files:
        return None

    return yaml_files[0]


def load_yaml_classes(yaml_path):
    if yaml_path is None:
        return {}

    with open(yaml_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    names = data.get("names", {})

    if isinstance(names, list):
        return {
            class_id: class_name
            for class_id, class_name in enumerate(names)
        }

    if isinstance(names, dict):
        return {
            int(class_id): class_name
            for class_id, class_name in names.items()
        }

    return {}


def find_label_files():
    all_txt_files = list(EXTRACT_DIR.rglob("*.txt"))

    label_files = []

    for path in all_txt_files:

        if path.name.lower() in {
            "obj.names",
            "train.txt",
            "valid.txt",
            "test.txt",
        }:
            continue

        label_files.append(path)

    return label_files


def load_final_manifest():
    if not FINAL_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Final manifest not found:\n{FINAL_MANIFEST_PATH}"
        )

    with open(
        FINAL_MANIFEST_PATH,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        return list(csv.DictReader(file))


def get_manifest_image_column(rows):
    if not rows:
        raise ValueError("Final manifest is empty.")

    columns = rows[0].keys()

    preferred_columns = [
        "image_name",
        "filename",
        "file_name",
        "image_filename",
        "curated_filename",
        "final_filename",
        "destination_filename",
    ]

    for column in preferred_columns:
        if column in columns:
            return column

    for column in columns:
        values = [
            str(row.get(column, "")).lower()
            for row in rows[:20]
        ]

        if any(
            value.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
            for value in values
        ):
            return column

    raise ValueError(
        "Could not automatically determine the image filename "
        "column in final_manifest.csv."
    )


def get_image_files():
    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    return sorted(
        path
        for path in FINAL_IMAGES_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in supported_extensions
    )


def parse_label_file(label_path):
    boxes = []
    errors = []

    with open(label_path, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            stripped = line.strip()

            if not stripped:
                continue

            parts = stripped.split()

            if len(parts) != 5:
                errors.append(
                    f"Line {line_number}: expected 5 values, "
                    f"found {len(parts)}"
                )
                continue

            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

            except ValueError:
                errors.append(
                    f"Line {line_number}: contains non-numeric values"
                )
                continue

            box = {
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height,
            }

            boxes.append(box)

    return boxes, errors


def box_is_valid(box):
    return (
        0.0 <= box["x_center"] <= 1.0
        and 0.0 <= box["y_center"] <= 1.0
        and 0.0 < box["width"] <= 1.0
        and 0.0 < box["height"] <= 1.0
        and box["x_center"] - box["width"] / 2 >= -1e-6
        and box["x_center"] + box["width"] / 2 <= 1.0 + 1e-6
        and box["y_center"] - box["height"] / 2 >= -1e-6
        and box["y_center"] + box["height"] / 2 <= 1.0 + 1e-6
    )


def boxes_are_duplicates(box_a, box_b, tolerance=1e-6):
    if box_a["class_id"] != box_b["class_id"]:
        return False

    keys = [
        "x_center",
        "y_center",
        "width",
        "height",
    ]

    return all(
        abs(box_a[key] - box_b[key]) <= tolerance
        for key in keys
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print_header(
        "PROTOTYPE V2 - YOLO ANNOTATION VERIFICATION"
    )

    print(f"Final images:\n{FINAL_IMAGES_DIR}")
    print()
    print(f"Final manifest:\n{FINAL_MANIFEST_PATH}")
    print()
    print(f"Annotation export folder:\n{EXPORT_DIR}")
    print()
    print(f"Verification report:\n{REPORT_PATH}")

    # --------------------------------------------------------
    # FIND AND EXTRACT ZIP
    # --------------------------------------------------------

    print_header("LOADING CVAT EXPORT")

    zip_path = find_zip_file()

    print(f"ZIP file found:\n{zip_path.name}")

    print()
    print("Extracting annotation ZIP...")

    extract_zip(zip_path)

    print(f"Extracted to:\n{EXTRACT_DIR}")

    # --------------------------------------------------------
    # INSPECT EXPORTED CLASSES
    # --------------------------------------------------------

    print_header("CLASS MAPPING CHECK")

    yaml_path = find_data_yaml()

    if yaml_path:
        print(f"Dataset YAML found:\n{yaml_path}")

        exported_classes = load_yaml_classes(yaml_path)

        print()
        print("Exported classes:")

        for class_id, class_name in sorted(exported_classes.items()):
            print(f"  {class_id}: {class_name}")

    else:
        exported_classes = {}

        print("WARNING: No data.yaml or dataset.yaml found.")

    class_mapping_match = exported_classes == EXPECTED_CLASSES

    print()
    print(f"Expected classes : {EXPECTED_CLASSES}")
    print(f"Class mapping    : "
          f"{'MATCH' if class_mapping_match else 'CHECK REQUIRED'}")

    # --------------------------------------------------------
    # LOAD FINAL DATASET
    # --------------------------------------------------------

    print_header("FINAL DATASET CHECK")

    manifest_rows = load_final_manifest()
    image_column = get_manifest_image_column(manifest_rows)
    image_files = get_image_files()

    print(f"Manifest rows              : {len(manifest_rows)}")
    print(f"Files in final image folder: {len(image_files)}")
    print(f"Detected manifest column   : {image_column}")

    final_image_names = {
        path.name for path in image_files
    }

    manifest_image_names = {
        Path(row[image_column]).name
        for row in manifest_rows
    }

    manifest_missing_files = (
        manifest_image_names - final_image_names
    )

    files_missing_manifest_rows = (
        final_image_names - manifest_image_names
    )

    print(
        f"Manifest images missing from folder : "
        f"{len(manifest_missing_files)}"
    )

    print(
        f"Folder images missing from manifest : "
        f"{len(files_missing_manifest_rows)}"
    )

    # --------------------------------------------------------
    # FIND YOLO LABEL FILES
    # --------------------------------------------------------

    print_header("ANNOTATION FILE CHECK")

    label_files = find_label_files()

    label_by_stem = {
        path.stem: path
        for path in label_files
    }

    print(f"YOLO label files found : {len(label_files)}")

    # --------------------------------------------------------
    # VERIFY EVERY IMAGE
    # --------------------------------------------------------

    print_header("VERIFYING ANNOTATIONS")

    total_boxes = 0

    class_counts = {
        class_id: 0
        for class_id in EXPECTED_CLASSES
    }

    images_with_annotations = 0
    images_without_annotations = 0

    missing_label_files = 0
    invalid_boxes = 0
    unknown_class_boxes = 0
    malformed_lines = 0
    duplicate_boxes = 0

    report_rows = []

    for image_path in image_files:

        image_name = image_path.name
        image_stem = image_path.stem

        label_path = label_by_stem.get(image_stem)

        image_status = "OK"
        notes = []

        box_count = 0
        marker_count = 0
        power_adapter_count = 0

        image_invalid_boxes = 0
        image_unknown_classes = 0
        image_malformed_lines = 0
        image_duplicate_boxes = 0

        if label_path is None:

            missing_label_files += 1
            images_without_annotations += 1

            image_status = "NO_LABEL_FILE"
            notes.append(
                "No YOLO label file. Valid only if image is "
                "intentionally empty."
            )

        else:

            boxes, parse_errors = parse_label_file(label_path)

            image_malformed_lines = len(parse_errors)
            malformed_lines += image_malformed_lines

            if parse_errors:
                image_status = "CHECK_REQUIRED"
                notes.extend(parse_errors)

            box_count = len(boxes)

            if box_count == 0:
                images_without_annotations += 1

            else:
                images_with_annotations += 1

            total_boxes += box_count

            for box in boxes:

                class_id = box["class_id"]

                if class_id not in EXPECTED_CLASSES:

                    unknown_class_boxes += 1
                    image_unknown_classes += 1

                    image_status = "CHECK_REQUIRED"

                    notes.append(
                        f"Unknown class ID: {class_id}"
                    )

                else:

                    class_counts[class_id] += 1

                    if class_id == 0:
                        marker_count += 1

                    elif class_id == 1:
                        power_adapter_count += 1

                if not box_is_valid(box):

                    invalid_boxes += 1
                    image_invalid_boxes += 1

                    image_status = "CHECK_REQUIRED"

                    notes.append(
                        "Invalid or out-of-bounds YOLO box."
                    )

            for index_a in range(len(boxes)):

                for index_b in range(
                    index_a + 1,
                    len(boxes)
                ):

                    if boxes_are_duplicates(
                        boxes[index_a],
                        boxes[index_b]
                    ):

                        duplicate_boxes += 1
                        image_duplicate_boxes += 1

                        image_status = "CHECK_REQUIRED"

                        notes.append(
                            "Exact duplicate annotation detected."
                        )

        report_rows.append({
            "image_name": image_name,
            "label_file": (
                label_path.name
                if label_path is not None
                else ""
            ),
            "box_count": box_count,
            "marker_count": marker_count,
            "power_adapter_count": power_adapter_count,
            "invalid_boxes": image_invalid_boxes,
            "unknown_class_boxes": image_unknown_classes,
            "malformed_lines": image_malformed_lines,
            "duplicate_boxes": image_duplicate_boxes,
            "status": image_status,
            "notes": " | ".join(notes),
        })

    # --------------------------------------------------------
    # CHECK UNUSED LABEL FILES
    # --------------------------------------------------------

    final_image_stems = {
        path.stem for path in image_files
    }

    unused_label_files = [
        path
        for path in label_files
        if path.stem not in final_image_stems
    ]

    # --------------------------------------------------------
    # SAVE REPORT
    # --------------------------------------------------------

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        fieldnames = [
            "image_name",
            "label_file",
            "box_count",
            "marker_count",
            "power_adapter_count",
            "invalid_boxes",
            "unknown_class_boxes",
            "malformed_lines",
            "duplicate_boxes",
            "status",
            "notes",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(report_rows)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print_header("ANNOTATION VERIFICATION SUMMARY")

    print(f"Expected final images          : {EXPECTED_IMAGE_COUNT}")
    print(f"Actual final images            : {len(image_files)}")
    print(f"YOLO label files found         : {len(label_files)}")
    print(f"Images with annotations        : {images_with_annotations}")
    print(f"Images without annotations     : {images_without_annotations}")
    print(f"Missing label files            : {missing_label_files}")
    print(f"Unused label files             : {len(unused_label_files)}")
    print(f"Total bounding boxes           : {total_boxes}")

    print()

    for class_id, class_name in EXPECTED_CLASSES.items():
        print(
            f"{class_name:<25}: "
            f"{class_counts[class_id]} boxes"
        )

    print()

    print(f"Malformed annotation lines     : {malformed_lines}")
    print(f"Unknown class boxes            : {unknown_class_boxes}")
    print(f"Invalid YOLO boxes             : {invalid_boxes}")
    print(f"Exact duplicate boxes          : {duplicate_boxes}")

    print()

    print(
        f"Manifest images missing folder : "
        f"{len(manifest_missing_files)}"
    )

    print(
        f"Folder images missing manifest : "
        f"{len(files_missing_manifest_rows)}"
    )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    critical_problems = (
        len(image_files) != EXPECTED_IMAGE_COUNT
        or len(manifest_rows) != EXPECTED_IMAGE_COUNT
        or len(manifest_missing_files) > 0
        or len(files_missing_manifest_rows) > 0
        or len(unused_label_files) > 0
        or malformed_lines > 0
        or unknown_class_boxes > 0
        or invalid_boxes > 0
        or duplicate_boxes > 0
        or not class_mapping_match
    )

    print_header("FINAL STATUS")

    if critical_problems:

        print("STATUS: CHECK REQUIRED")
        print()
        print(
            "One or more annotation or dataset consistency "
            "problems were detected."
        )

        print()
        print(
            "Inspect the terminal summary and "
            "annotation_verification_report.csv."
        )

    else:

        print("STATUS: STRUCTURAL VERIFICATION PASSED")

        print()
        print(
            "The YOLO annotation export is structurally valid."
        )

        print()
        print(
            "Next stage:"
        )

        print(
            "Generate visual annotation samples and inspect "
            "bounding-box quality before dataset splitting."
        )

    print()
    print(f"Verification report saved to:\n{REPORT_PATH}")


if __name__ == "__main__":
    main()