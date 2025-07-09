import os
import shutil
from PIL import Image

dataset_path = '/mnt/c/Kaggle/train_data/rail_cars/Railroad-Cars-8'
filtered_path = dataset_path + '_filtered'
min_size = 16  # minimum width/height in pixels

# Remove filtered_path folder if exists
if os.path.exists(filtered_path):
    shutil.rmtree(filtered_path)

# Copy entire directory tree from dataset_path to filtered_path
shutil.copytree(dataset_path, filtered_path)

# Image extensions to consider
image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

removed_images = []

def filter_labels_and_images(labels_dir, images_dir):
    for label_fname in os.listdir(labels_dir):
        if not label_fname.endswith('.txt'):
            continue
        label_path = os.path.join(labels_dir, label_fname)
        image_name = label_fname[:-4]  # remove .txt extension

        # Find corresponding image file
        image_file = None
        for ext in image_exts:
            candidate = os.path.join(images_dir, image_name + ext)
            if os.path.isfile(candidate):
                image_file = candidate
                break

        if image_file is None:
            print(f"[Warning] No image found for label {label_fname}, skipping")
            continue

        # Load image size
        with Image.open(image_file) as im:
            w_img, h_img = im.size

        # Read label lines
        with open(label_path, 'r') as f:
            lines = f.readlines()

        total_boxes = len(lines)
        filtered_lines = []
        removed_boxes = 0

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x_c, y_c, w, h = parts
            w, h = float(w), float(h)

            w_px = w * w_img
            h_px = h * h_img

            if w_px >= min_size and h_px >= min_size:
                filtered_lines.append(line)
            else:
                removed_boxes += 1

        if total_boxes == 0:
            # No boxes originally, keep image and label as is
            print(f"Image {os.path.basename(image_file)} originally had no boxes — kept as is.")
            continue

        if len(filtered_lines) == 0:
            # Boxes existed originally but all removed — delete both files
            os.remove(label_path)
            os.remove(image_file)
            removed_images.append(os.path.basename(image_file))
            print(f"Removed image and label: {os.path.basename(image_file)} (all {total_boxes} boxes removed)")
        else:
            # Overwrite label file with filtered boxes
            with open(label_path, 'w') as f:
                f.writelines(filtered_lines)
            print(f"Processed image: {os.path.basename(image_file)} - boxes found: {total_boxes}, boxes removed: {removed_boxes}")

# Walk filtered_path to find all 'labels' folders and corresponding 'images' folders
for root, dirs, files in os.walk(filtered_path):
    if os.path.basename(root) == 'labels':
        labels_dir = root
        images_dir = os.path.join(os.path.dirname(root), 'images')
        if os.path.exists(images_dir):
            print(f"Filtering labels in {labels_dir} with images from {images_dir}")
            filter_labels_and_images(labels_dir, images_dir)
        else:
            print(f"[Warning] No images directory found for labels folder: {labels_dir}")

print("Filtering complete!")
print("\nImages removed during filtering:")
for img_name in removed_images:
    print(img_name)


