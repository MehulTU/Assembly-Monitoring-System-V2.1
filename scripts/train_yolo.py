"""
train_yolo.py

HOW TO RUN THE FILE:
    python scripts/train_yolo.py                 (TYPE THIS IN TERMINAL)
    python scripts/train_yolo.py --model yolo11s.pt --epochs 150

PURPOSE:

Train the YOLO object-detection model on the prepared dataset
(datasets/yolo_dataset_v1) and INTEGRATE the result into the
Assembly_monitor software automatically.

Workflow:
    1. Train on datasets/yolo_dataset_v1/data.yaml
       (hyperparameters tuned for a small, single-camera,
       fixed-workspace dataset).
    2. Evaluate the trained model on the TEST split and print
       per-class precision / recall / mAP.
    3. Back up the previous weights/best.pt (if it exists).
    4. Deploy the new best.pt to weights/best.pt.

Step 4 is the integration point: Assembly_monitor/main.py loads
YOLO_MODEL_PATH = "weights/best.pt" (config/settings.py), so after
this script finishes, running main.py automatically uses the newly
trained model. No software changes needed - this is the modular
design working as intended.

WHY THESE HYPERPARAMETERS (small custom dataset):
    - More epochs + early-stopping patience: small datasets need
      more passes, but patience stops training when val mAP stops
      improving, so it never overfits blindly.
    - Moderate color augmentation (hsv): makes the model robust to
      lighting changes between recording sessions.
    - Small rotation/translation/scale augmentation: the camera is
      fixed, so extreme geometric augmentation would create
      unrealistic training views and HURT confidence.
    - mosaic enabled but close_mosaic at the end: mosaic helps a
      small dataset generalize, but the final epochs train on
      realistic full frames, which raises real-world confidence.
"""

from pathlib import Path
from datetime import datetime
import argparse
import shutil

from ultralytics import YOLO


# ======================================================================
# PROJECT CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_YAML = PROJECT_ROOT / "datasets" / "yolo_dataset_v1" / "data.yaml"

RUNS_DIR = PROJECT_ROOT / "runs" / "detect"

DEPLOY_WEIGHTS = PROJECT_ROOT / "weights" / "best.pt"

WEIGHTS_BACKUP_DIR = PROJECT_ROOT / "weights" / "backups"


# ======================================================================
# DEFAULT TRAINING SETTINGS
# ======================================================================

DEFAULT_BASE_MODEL = "yolo11n.pt"   # try "yolo11s.pt" if GPU allows
DEFAULT_EPOCHS = 150
DEFAULT_IMGSZ = 640                 # must match YOLO_IMGSZ in
                                    # Assembly_monitor/config/settings.py
DEFAULT_BATCH = 16                  # lower to 8 or 4 if out of memory
DEFAULT_PATIENCE = 30               # early stopping
DEFAULT_DEVICE = None               # None = auto (GPU if available)


def print_header(title):
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


# ======================================================================
# TRAINING
# ======================================================================

def train(args):

    print_header("YOLO TRAINING")
    print(f"Base model : {args.model}")
    print(f"Dataset    : {DATA_YAML}")
    print(f"Epochs     : {args.epochs} (patience {args.patience})")
    print(f"Image size : {args.imgsz}")
    print(f"Batch      : {args.batch}")

    if not DATA_YAML.exists():
        print()
        print("ERROR: data.yaml not found.")
        print("Run scripts/prepare_yolo_dataset.py first.")
        return None

    run_name = datetime.now().strftime("assembly_v%Y%m%d_%H%M%S")

    model = YOLO(args.model)

    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,

        project=str(RUNS_DIR),
        name=run_name,

        # ------------------------------------------------------
        # AUGMENTATION (tuned for fixed camera, small dataset)
        # ------------------------------------------------------
        hsv_h=0.015,      # slight hue shift
        hsv_s=0.60,       # saturation - lighting robustness
        hsv_v=0.50,       # brightness - lighting robustness
        degrees=5.0,      # small rotations only (fixed camera)
        translate=0.10,
        scale=0.30,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,       # workspace is never upside down
        fliplr=0.5,       # horizontal flip is realistic
        mosaic=1.0,
        close_mosaic=15,  # last 15 epochs on realistic full frames
        mixup=0.0,        # mixup hurts small 2-class datasets

        # ------------------------------------------------------
        # OPTIMIZATION
        # ------------------------------------------------------
        optimizer="auto",
        lr0=0.01,
        cos_lr=True,      # cosine LR decay - smoother convergence

        seed=42,          # reproducible for the thesis
        deterministic=True,

        plots=True,
        val=True,
    )

    run_dir = RUNS_DIR / run_name
    best_weights = run_dir / "weights" / "best.pt"

    if not best_weights.exists():
        print("ERROR: Training finished but best.pt was not found:")
        print(best_weights)
        return None

    print()
    print(f"Training run folder: {run_dir}")

    return best_weights


# ======================================================================
# EVALUATION ON THE TEST SPLIT
# ======================================================================

def evaluate(best_weights, args):

    print_header("EVALUATION ON TEST SPLIT")

    model = YOLO(str(best_weights))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=args.imgsz,
        device=args.device,
        plots=True,
    )

    print()
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} "
          f"{'mAP50':>10} {'mAP50-95':>10}")
    print("-" * 60)

    class_names = list(metrics.names.values())

    for i, name in enumerate(class_names):
        p, r, map50, map5095 = metrics.class_result(i)
        print(f"{name:<15} {p:>10.3f} {r:>10.3f} "
              f"{map50:>10.3f} {map5095:>10.3f}")

    print("-" * 60)
    print(f"{'ALL':<15} "
          f"{metrics.box.mp:>10.3f} {metrics.box.mr:>10.3f} "
          f"{metrics.box.map50:>10.3f} {metrics.box.map:>10.3f}")

    print()
    if metrics.box.map50 < 0.60:
        print("NOTE: mAP50 is below 0.60. Before blaming the model,")
        print("check the dataset first: annotation coverage (empty")
        print("labels), per-class box counts, and whether the test")
        print("trials look like the training trials.")

    return metrics


# ======================================================================
# DEPLOY WEIGHTS INTO THE ASSEMBLY MONITOR SOFTWARE
# ======================================================================

def deploy(best_weights):

    print_header("DEPLOYING WEIGHTS TO ASSEMBLY MONITOR")

    DEPLOY_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)

    # Back up the previous model so experiments are reversible
    if DEPLOY_WEIGHTS.exists():
        WEIGHTS_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = WEIGHTS_BACKUP_DIR / f"best_{stamp}.pt"
        shutil.copy2(DEPLOY_WEIGHTS, backup_path)
        print(f"Previous model backed up to:")
        print(f"  {backup_path}")

    shutil.copy2(best_weights, DEPLOY_WEIGHTS)

    print(f"New model deployed to:")
    print(f"  {DEPLOY_WEIGHTS}")
    print()
    print("Integration complete. Assembly_monitor/main.py loads this")
    print("path (YOLO_MODEL_PATH in config/settings.py), so the live")
    print("system now uses the newly trained model:")
    print()
    print("  python Assembly_monitor/main.py")


# ======================================================================
# MAIN
# ======================================================================

def main():

    parser = argparse.ArgumentParser(
        description="Train YOLO and deploy it into Assembly_monitor."
    )
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL,
                        help="Base model (yolo11n.pt, yolo11s.pt, ...)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--device", default=DEFAULT_DEVICE,
                        help="0 for GPU, cpu for CPU, default auto")
    parser.add_argument("--no-deploy", action="store_true",
                        help="Train and evaluate only; do not replace "
                             "weights/best.pt")

    args = parser.parse_args()

    best_weights = train(args)

    if best_weights is None:
        print_header("STATUS")
        print("STATUS: TRAINING FAILED")
        return

    evaluate(best_weights, args)

    if args.no_deploy:
        print_header("STATUS")
        print("STATUS: TRAINED (not deployed, --no-deploy was set)")
        print(f"Best weights: {best_weights}")
        return

    deploy(best_weights)

    print_header("STATUS")
    print("STATUS: TRAINED, EVALUATED, AND DEPLOYED SUCCESSFULLY")


if __name__ == "__main__":
    main()
