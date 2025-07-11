import os
import shutil
import torch
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Albumentations
import albumentations as A

torch.cuda.empty_cache()  # Frees unreferenced cached memory
torch.cuda.ipc_collect()  # Releases inter-process memory
torch.cuda.set_per_process_memory_fraction(0.6, device=0)

# === Setup ===
data_yaml_path = '/mnt/c/Kaggle/train_data/rail_cars/Railroad-Cars-9_filtered/data.yaml'
target_model_dir = '/mnt/c/Kaggle/models/rail_cars13'
project_path = os.path.join(target_model_dir, 'runs/detect')
os.makedirs(target_model_dir, exist_ok=True)


def plot_losses_from_csv(csv_path, run_name):
    """Plot YOLOv8 train/val losses (box, cls, dfl) in 3 stacked subplots, with min markers."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # Clean column names

    epochs = df.index + 1  # Epochs start at 1

    # Extract all relevant losses
    train_box_loss = df['train/box_loss']
    val_box_loss = df['val/box_loss']
    train_cls_loss = df['train/cls_loss']
    val_cls_loss = df['val/cls_loss']
    train_dfl_loss = df['train/dfl_loss']
    val_dfl_loss = df['val/dfl_loss']

    # Set up subplots: 3 rows, 1 column
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    loss_groups = [
        ("Box Loss", train_box_loss, val_box_loss, 'r'),
        ("Cls Loss", train_cls_loss, val_cls_loss, 'orange'),
        ("DFL Loss", train_dfl_loss, val_dfl_loss, 'g')
    ]

    for ax, (label, train_series, val_series, color) in zip(axes, loss_groups):
        # Plot lines
        ax.plot(epochs, train_series, linestyle='--', color=color, alpha=0.7, label=f"Train {label}")
        ax.plot(epochs, val_series, linestyle='-', color=color, alpha=1.0, label=f"Val {label}")

        # === Train Min ===
        train_min_val = train_series.min()
        train_min_epoch = train_series.idxmin() + 1

        ax.axhline(y=train_min_val, linestyle='dotted', color=color, alpha=0.4)
        ax.axvline(x=train_min_epoch, linestyle='dotted', color=color, alpha=0.4)
        ax.text(epochs[-1], train_min_val, f"Train min: {train_min_val:.4f}",
                color=color, fontsize=8, ha='right', va='bottom', alpha=0.8)
        ax.text(train_min_epoch, ax.get_ylim()[1], f"{train_min_epoch}",
                rotation=90, va='top', ha='center', fontsize=8, color=color, alpha=0.8)

        # === Val Min ===
        val_min_val = val_series.min()
        val_min_epoch = val_series.idxmin() + 1

        ax.axhline(y=val_min_val, linestyle='dotted', color=color, alpha=0.8)
        ax.axvline(x=val_min_epoch, linestyle='dotted', color=color, alpha=0.8)
        ax.text(epochs[-1], val_min_val, f"Val min: {val_min_val:.4f}",
                color=color, fontsize=9, ha='right', va='bottom')
        ax.text(val_min_epoch, ax.get_ylim()[1], f"{val_min_epoch}",
                rotation=90, va='top', ha='center', fontsize=9, color=color)

        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel("Epoch")
    fig.suptitle(f"Train/Val Losses over Epochs: {run_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save and show
    plot_path = os.path.join(target_model_dir, f"{run_name}_losses.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"📊 Loss plot saved to: {plot_path}")


# === Training Loop ===
# for epochs in [10, 60, 130]:
for model_path in [
    # 'yolov8s.pt',
    'yolov8m.pt',
    'yolov8l.pt'
]:
    prev_model_path = None
    pretrained_epochs = 0
    for epochs in [250]:
        model_base = os.path.splitext(model_path)[0]
        run_name = f"{model_base}_railway_model_{epochs + pretrained_epochs}epoch"
        dest_best_path = os.path.join(target_model_dir, f"{run_name}.pt")

        if os.path.exists(dest_best_path):
            prev_model_path = dest_best_path
            pretrained_epochs += epochs
            continue
        if prev_model_path:
            model = YOLO(prev_model_path)
        else:
            model = YOLO(model_path)

        # === Run training ===
        model.train(
            data=data_yaml_path,  # Path to data YAML file (contains class names and train/val image dirs)
            epochs=epochs,  # Number of training epochs
            imgsz=640,  # Input image size (will be resized to 640x640 before training)
            batch=1,  # Batch size (images per GPU during training)
            workers=2 if model_path not in ['yolov8x.pt'] else 1,
            # Number of dataloader worker threads (reduce for large models)
            device=0,  # CUDA device ID to use (0 = first GPU)

            name=run_name,  # Name of the training run (used in output directory naming)
            project=project_path,  # Root directory to store training results and logs

            # --- Data Augmentation Parameters ---
            fliplr=0.5,  # Probability of horizontal flip (left/right)
            flipud=0.0,  # Probability of vertical flip (up/down)
            hsv_h=0,  # 0.015,                 # HSV hue augmentation (fractional change)
            hsv_s=0,  # 0.7,                   # HSV saturation augmentation
            hsv_v=0,  # 0.4,                   # HSV value (brightness) augmentation
            scale=0.8,  # Image scale range (zoom in/out)
            translate=0.1,  # Image translation as a fraction of image size
            shear=2.0,  # Shear angle in degrees
            perspective=0.0005,  # Perspective transform distortion
            mosaic=0,  # 0.8,                  # Probability of using 4-image mosaic augmentation
            mixup=0,  # 0.1,                   # Probability of using mixup (combining images)
            degrees=4.0,  # Random rotation in degrees
            save_period=5,
            exist_ok=True,
        )

        # === Parse training results from results.csv and plot ===
        log_csv_path = os.path.join(project_path, run_name, "results.csv")
        if os.path.exists(log_csv_path):
            plot_losses_from_csv(log_csv_path, run_name)
        else:
            print(f"⚠️ Log not found: {log_csv_path}")

        # === Save final model ===
        source_best_path = os.path.join(project_path, run_name, "weights", "best.pt")
        if os.path.exists(source_best_path):
            shutil.copy(source_best_path, dest_best_path)
            print(f"✅ Model saved: {dest_best_path}")
            prev_model_path = dest_best_path
        else:
            print(f"❌ best.pt not found: {source_best_path}")

        del model
        torch.cuda.empty_cache()  # Frees unreferenced cached memory
        torch.cuda.ipc_collect()  # Releases inter-process memory
