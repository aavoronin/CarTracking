import os
import shutil
import random
import numpy as np
import cv2
from PIL import Image

# ========== CONFIGURATION ==========
dataset_path = '/mnt/c/Kaggle/train_data/rail_cars/Railroad-Cars-9'
filtered_path = dataset_path + '_filtered'
fixtures_path = '/mnt/c/Kaggle/train_data/rail_cars/railroad-posts'

image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

FACTOR = 4.0
MIN_WIDTH_NORM = 0.025 * FACTOR
MIN_HEIGHT_NORM = 0.025 * FACTOR

MIN_WIDTH_RATIO = 1 / 32
MAX_WIDTH_RATIO = 1 / 6
MIN_HEIGHT_RATIO = 1 / 16
MAX_HEIGHT_RATIO = 1 / 3
N_NOISE_IMAGES = 20
NOISE_COPIES_PER_IMAGE = 2

# ========== FUNCTIONS ==========
def filter_labels_and_images(labels_dir, images_dir):
    removed_images = []
    for label_fname in os.listdir(labels_dir):
        if not label_fname.endswith('.txt'):
            continue
        label_path = os.path.join(labels_dir, label_fname)
        image_name = label_fname[:-4]

        image_file = None
        for ext in image_extensions:
            candidate = os.path.join(images_dir, image_name + ext)
            if os.path.isfile(candidate):
                image_file = candidate
                break

        if image_file is None:
            print(f"[Warning] No image found for label {label_fname}, skipping")
            continue

        with Image.open(image_file) as im:
            w_img, h_img = im.size

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

            if w >= MIN_WIDTH_NORM and h >= MIN_HEIGHT_NORM:
                filtered_lines.append(line)
            else:
                removed_boxes += 1

        if total_boxes == 0:
            print(f"Image {os.path.basename(image_file)} originally had no boxes — kept as is.")
            continue

        if len(filtered_lines) == 0:
            os.remove(label_path)
            os.remove(image_file)
            removed_images.append(os.path.basename(image_file))
            print(f"Removed image and label: {os.path.basename(image_file)} (all {total_boxes} boxes removed)")
        else:
            with open(label_path, 'w') as f:
                f.writelines(filtered_lines)
            print(f"Processed image: {os.path.basename(image_file)} - boxes found: {total_boxes}, boxes removed: {removed_boxes}")
    return removed_images

def find_bounding_boxes(pil_image, white_thresh=220):
    image_np = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, white_thresh, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(cnt) for cnt in contours]

def load_object_patches():
    object_patches = []
    image_paths = [
        os.path.join(fixtures_path, fname)
        for fname in os.listdir(fixtures_path)
        if fname.lower().endswith(image_extensions)
    ]
    for path in image_paths:
        try:
            image = Image.open(path).convert("RGB")
            bboxes = find_bounding_boxes(image)
            filtered_bboxes = [box for box in bboxes if box[2] * box[3] >= 300]
            for (x, y, w, h) in filtered_bboxes:
                obj = image.crop((x, y, x + w, y + h))
                object_patches.append(obj)
            print(f"{path}: {len(filtered_bboxes)} objects kept")
        except Exception as e:
            print(f"Error processing {path}: {e}")
    return object_patches

def apply_noise_to_image(img, object_patches):
    img = img.convert("RGBA")
    overlay = img.copy()

    for _ in range(N_NOISE_IMAGES):
        if not object_patches:
            break
        obj = random.choice(object_patches)

        orig_w, orig_h = obj.size
        min_width = img.width * MIN_WIDTH_RATIO
        max_width = img.width * MAX_WIDTH_RATIO
        min_height = img.height * MIN_HEIGHT_RATIO
        max_height = img.height * MAX_HEIGHT_RATIO

        scale_w_min = min_width / orig_w
        scale_w_max = max_width / orig_w
        scale_h_min = min_height / orig_h
        scale_h_max = max_height / orig_h
        scale_min = max(scale_w_min, scale_h_min)
        scale_max = min(scale_w_max, scale_h_max)

        if scale_min >= scale_max:
            continue

        scale = random.uniform(scale_min, scale_max)
        resized_w = int(orig_w * scale)
        resized_h = int(orig_h * scale)
        obj_resized = obj.resize((resized_w, resized_h), Image.LANCZOS)

        obj_np = np.array(obj_resized.convert("RGBA"))
        r, g, b, a = obj_np[..., 0], obj_np[..., 1], obj_np[..., 2], obj_np[..., 3]
        white_mask = (r > 230) & (g > 230) & (b > 230)
        obj_np[..., 3] = np.where(white_mask, 0, 255)
        obj_rgba = Image.fromarray(obj_np)

        max_x = max(1, img.width - obj_rgba.width)
        max_y = max(1, img.height - obj_rgba.height)
        pos_x = random.randint(0, max_x)
        pos_y = random.randint(0, max_y)

        overlay.paste(obj_rgba, (pos_x, pos_y), obj_rgba)

    # Gaussian noise with 50% chance
    if random.random() < 0.5:
        noisy_img = np.array(overlay.convert("RGB"), dtype=np.float32)
        std_dev = random.uniform(10, 50)
        noise = np.random.normal(0, std_dev, noisy_img.shape)
        noisy_img += noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

    return overlay.convert("RGB")

# ========== EXECUTION ==========

# Remove filtered folder if exists
if os.path.exists(filtered_path):
    shutil.rmtree(filtered_path)
    print(f"Removed existing folder: {filtered_path}")

# Copy dataset folder to filtered folder
shutil.copytree(dataset_path, filtered_path)
print(f"Copied dataset from {dataset_path} to {filtered_path}")

# Filter labels and images in filtered folder
total_removed_images = []
for root, dirs, files in os.walk(filtered_path):
    if os.path.basename(root) == 'labels':
        labels_dir = root
        images_dir = os.path.join(os.path.dirname(root), 'images')
        if os.path.exists(images_dir):
            print(f"Filtering labels in {labels_dir} with images from {images_dir}")
            removed = filter_labels_and_images(labels_dir, images_dir)
            total_removed_images.extend(removed)
        else:
            print(f"[Warning] No images directory found for labels folder: {labels_dir}")

print(f"Filtering complete! Removed {len(total_removed_images)} images.")

# Load noise patches
print("Loading fixture noise patches...")
object_patches = load_object_patches()

# Augment images by adding noisy copies
for split in ['train', 'valid', 'test']:
    images_dir = os.path.join(filtered_path, split, 'images')
    labels_dir = os.path.join(filtered_path, split, 'labels')

    if not (os.path.exists(images_dir) and os.path.exists(labels_dir)):
        print(f"Skipping {split}, missing images or labels folder.")
        continue

    images = [f for f in os.listdir(images_dir) if f.lower().endswith(image_extensions)]
    print(f"Processing {len(images)} images in {split} split")

    for img_name in images:
        base_name, ext = os.path.splitext(img_name)
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, base_name + '.txt')

        if not os.path.isfile(label_path):
            print(f"[Warning] Missing label for {img_name}, skipping augmentation.")
            continue

        with Image.open(img_path) as img_orig:
            for i in range(1, NOISE_COPIES_PER_IMAGE + 1):
                noisy_img = apply_noise_to_image(img_orig, object_patches)
                new_img_name = f"{base_name}_aug{i}{ext}"
                new_img_path = os.path.join(images_dir, new_img_name)
                noisy_img.save(new_img_path)

                new_label_path = os.path.join(labels_dir, f"{base_name}_aug{i}.txt")
                shutil.copyfile(label_path, new_label_path)

        print(f"Augmented {img_name} with {NOISE_COPIES_PER_IMAGE} noisy copies.")

print("Augmentation done!")
