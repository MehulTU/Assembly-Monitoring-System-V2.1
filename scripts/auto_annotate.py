"""
===============================================================================
Prototype V2 - Automatic Pre-Annotation (Model-Assisted Labeling)
===============================================================================

File:
    auto_annotate.py

WHERE TO PUT THIS FILE:
    Put it in the  scripts/  folder, next to extract_frames.py,
    prepare_yolo_dataset.py and train_yolo.py.

HOW TO RUN THE FILE:
    python scripts\\auto_annotate.py            (TYPE THIS IN TERMINAL)

    Optional:
    python scripts\\auto_annotate.py --conf 0.40
    python scripts\\auto_annotate.py --overwrite

WHEN TO RUN THIS FILE:
    AFTER  finalize_dataset.py   (the final images exist)
    BEFORE you annotate in CVAT.

-------------------------------------------------------------------------------
FOR THE FUTURE STUDENT - WHAT THIS SCRIPT DOES, IN SIMPLE WORDS
-------------------------------------------------------------------------------

Annotating means drawing a box around every object in every image.
Doing this by hand for hundreds of images takes many hours.

This script uses the CURRENT trained model (weights/best.pt) to draw
FIRST-DRAFT boxes on every image automatically. Instead of drawing
every box yourself, you only CORRECT the drafts in CVAT:

    - move boxes that are slightly wrong
    - delete boxes that are false (for example a face detected
      as "marker")
    - add boxes the model missed

Correcting is 3 to 5 times faster than drawing from zero.

VERY IMPORTANT RULE:
    NEVER train directly on these draft labels without checking
    them. The current model makes mistakes, and training on wrong
    boxes makes the next model WORSE. The rule is always:

        MACHINE DRAFTS  ->  HUMAN CORRECTS  ->  THEN TRAIN.

WHAT THE OUTPUT LOOKS LIKE:

    datasets/pre_annotations/
        obj.names                one class name per line
        labels/
            <image_name>.txt     one draft label file per image

    Each .txt file is in YOLO format. One line per box:
        <class_id> <x_center> <y_center> <width> <height>
    All values are relative (0.0 to 1.0), which is exactly the
    format CVAT and YOLO training use.

HOW TO USE THE DRAFTS IN CVAT:

    1. Create your CVAT task with the final images as usual.
    2. In the task, choose:  Actions -> Upload annotations
       and select the format "YOLO 1.1".
    3. Upload a zip that contains the labels/ folder and
       obj.names from datasets/pre_annotations/.
    4. The draft boxes appear on the images. Now go image by
       image and CORRECT them (move / delete / add).
    5. Export as "YOLO 1.1" as usual and continue the normal
       pipeline (annotations_export.py).
"""

from pathlib import Path
import argparse

import cv2
from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================
# parents[1] = one folder up from scripts/ = the project root.

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The images that need annotation (created by finalize_dataset.py).
FINAL_IMAGES_DIR = PROJECT_ROOT / "datasets" / "final" / "images"

# Where the draft labels are written. This folder is separate
# from the real annotations on purpose, so drafts can never be
# confused with human-verified labels.
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "pre_annotations"
OUTPUT_LABELS_DIR = OUTPUT_DIR / "labels"

# The current trained model used to create the drafts.
MODEL_PATH = PROJECT_ROOT / "weights" / "best.pt"


# ============================================================
# SETTINGS
# ============================================================

# Minimum confidence for a draft box.
#
# For PRE-annotation we use a slightly HIGHER threshold than the
# live system (0.35 instead of 0.25). Reason: it is faster for a
# human to ADD a missing box than to DELETE many wrong boxes, so
# we prefer fewer, more reliable drafts.
DEFAULT_CONFIDENCE = 0.35

# Image size for inference. Keep the same value that was used
# for training (see train_yolo.py and settings.py).
IMAGE_SIZE = 640

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def print_header(title):
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


# ============================================================
# CONVERT ONE DETECTION TO A YOLO LABEL LINE
# ============================================================

def to_yolo_line(class_id, x1, y1, x2, y2, image_width, image_height):
    """
    Convert absolute pixel corners (x1, y1, x2, y2) into the
    YOLO text format:

        <class_id> <x_center> <y_center> <width> <height>

    where every value is RELATIVE to the image size (0.0 - 1.0).
    Example: a box in the middle of the image covering half the
    width and height becomes:  0 0.500000 0.500000 0.500000 0.500000
    """

    x_center = ((x1 + x2) / 2.0) / image_width
    y_center = ((y1 + y2) / 2.0) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height

    # Clamp to 0..1 so a box touching the image border never
    # produces an invalid value like 1.0000001.
    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    width = min(max(width, 0.0), 1.0)
    height = min(max(height, 0.0), 1.0)

    return (
        f"{class_id} "
        f"{x_center:.6f} {y_center:.6f} "
        f"{width:.6f} {height:.6f}"
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Create draft YOLO annotations with the current "
                    "model, for correction in CVAT."
    )

    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE,
                        help="Minimum confidence for a draft box "
                             f"(default {DEFAULT_CONFIDENCE})")

    parser.add_argument("--overwrite", action="store_true",
                        help="Also re-create draft files that already "
                             "exist (default: skip them)")

    args = parser.parse_args()

    print_header("PROTOTYPE V2 - AUTOMATIC PRE-ANNOTATION")

    print(f"Model        : {MODEL_PATH}")
    print(f"Input images : {FINAL_IMAGES_DIR}")
    print(f"Output       : {OUTPUT_LABELS_DIR}")
    print(f"Confidence   : {args.conf}")

    # --------------------------------------------------------
    # SAFETY CHECKS
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        print()
        print("ERROR: No trained model found at:")
        print(f"    {MODEL_PATH}")
        print()
        print("You need at least one trained model before drafts can")
        print("be created. Train one first with:")
        print("    python scripts\\train_yolo.py")
        return

    if not FINAL_IMAGES_DIR.exists():
        print()
        print("ERROR: Final images folder not found:")
        print(f"    {FINAL_IMAGES_DIR}")
        print()
        print("Run the dataset pipeline first (finalize_dataset.py).")
        return

    image_paths = sorted(
        path
        for path in FINAL_IMAGES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print()
        print("ERROR: No images found in the final images folder.")
        return

    print(f"Images found : {len(image_paths)}")

    # --------------------------------------------------------
    # LOAD MODEL AND PREPARE OUTPUT FOLDER
    # --------------------------------------------------------

    model = YOLO(str(MODEL_PATH))

    OUTPUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # Write obj.names: one class name per line, in class-id order.
    # CVAT uses this file to know which id is which class.
    class_names = [
        model.names[class_id]
        for class_id in sorted(model.names.keys())
    ]

    names_path = OUTPUT_DIR / "obj.names"
    with names_path.open("w", encoding="utf-8") as names_file:
        for name in class_names:
            names_file.write(name + "\n")

    print(f"Classes      : {', '.join(class_names)}")

    # --------------------------------------------------------
    # PROCESS EVERY IMAGE
    # --------------------------------------------------------

    print_header("CREATING DRAFT ANNOTATIONS")

    images_processed = 0
    images_skipped = 0
    images_with_boxes = 0
    images_empty = 0
    total_boxes = 0
    boxes_per_class = {name: 0 for name in class_names}

    for image_path in image_paths:

        label_path = OUTPUT_LABELS_DIR / f"{image_path.stem}.txt"

        # Skip images that already have a draft file, so the
        # script can be safely re-run after adding new images.
        if label_path.exists() and not args.overwrite:
            images_skipped += 1
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"WARNING: Could not read image, skipping: "
                  f"{image_path.name}")
            continue

        image_height, image_width = image.shape[:2]

        # Run the model on this single image.
        results = model.predict(
            image,
            conf=args.conf,
            imgsz=IMAGE_SIZE,
            verbose=False,
        )

        lines = []

        for result in results:
            for box in result.boxes:

                class_id = int(box.cls[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0])

                lines.append(
                    to_yolo_line(
                        class_id,
                        x1, y1, x2, y2,
                        image_width, image_height,
                    )
                )

                total_boxes += 1
                boxes_per_class[model.names[class_id]] += 1

        # Write the label file. An EMPTY file is written for
        # images with no detections - in YOLO format an empty
        # label file means "this image contains no objects",
        # which is exactly right for empty-workspace frames.
        with label_path.open("w", encoding="utf-8") as label_file:
            label_file.write("\n".join(lines))
            if lines:
                label_file.write("\n")

        images_processed += 1

        if lines:
            images_with_boxes += 1
        else:
            images_empty += 1

        # Progress message every 50 images so long runs do not
        # look frozen.
        if images_processed % 50 == 0:
            print(f"  ... {images_processed} images done")

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print_header("PRE-ANNOTATION SUMMARY")

    print(f"Images processed      : {images_processed}")
    print(f"Images skipped        : {images_skipped} "
          f"(draft already existed)")
    print(f"Images with boxes     : {images_with_boxes}")
    print(f"Images with no boxes  : {images_empty}")
    print(f"Total draft boxes     : {total_boxes}")

    for name in class_names:
        print(f"  {name:<15}: {boxes_per_class[name]} boxes")

    print()
    print(f"Draft labels folder:")
    print(f"    {OUTPUT_LABELS_DIR}")
    print(f"Class names file:")
    print(f"    {names_path}")

    print_header("NEXT STEPS")

    print("1. Zip the contents of datasets/pre_annotations/")
    print("   (the labels/ folder and obj.names together).")
    print("2. In CVAT open your task and use:")
    print("       Actions -> Upload annotations -> YOLO 1.1")
    print("   and upload that zip.")
    print("3. CORRECT every image by hand:")
    print("       - move boxes that sit wrong")
    print("       - delete false boxes")
    print("       - add boxes the model missed")
    print("4. Export from CVAT as YOLO 1.1 and continue the normal")
    print("   pipeline:")
    print("       python scripts\\annotations_export.py")
    print("       python scripts\\prepare_yolo_dataset.py")
    print("       python scripts\\train_yolo.py")
    print()
    print("REMEMBER: drafts are only a starting point. Never train")
    print("on drafts that a human has not checked.")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
