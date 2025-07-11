import os
import shutil
import random
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# ========== CONFIGURATION ==========
dataset_path = '/mnt/c/Kaggle/train_data/rail_cars/Railroad-Cars-10'
filtered_path = dataset_path + '_filtered'
fixtures_path = '/mnt/c/Kaggle/train_data/rail_cars/railroad-posts'

image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

FACTOR = 2.0
MIN_WIDTH_NORM = 0.025 * FACTOR
MIN_HEIGHT_NORM = 0.025 * FACTOR

MIN_WIDTH_RATIO = 1 / 32
MAX_WIDTH_RATIO = 1 / 6
MIN_HEIGHT_RATIO = 1 / 16
MAX_HEIGHT_RATIO = 1 / 3
N_NOISE_IMAGES = 8
NOISE_COPIES_PER_IMAGE = 1
GAUSS_NOISE_PROBABILITY = 0.5
GAUSS_NOISE_STD_MIN = 5
GAUSS_NOISE_STD_MAX = 30
MAX_IMAGES_TO_PLOT = 40
ENLARGE_SCALE = 6


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
            image_aspect_ratio = w_img / h_img

        with open(label_path, 'r') as f:
            lines = f.readlines()

        total_boxes = len(lines)
        filtered_lines = []
        removed_boxes = 0

        bboxes = []  # Store normalized bboxes for analysis

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x_c, y_c, w, h = parts
            x_c, y_c, w, h = map(float, [x_c, y_c, w, h])

            if w >= MIN_WIDTH_NORM and h >= MIN_HEIGHT_NORM:
                filtered_lines.append(line)
                bboxes.append((x_c, y_c, w, h))
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

            # ---- Compute bbox stats ----
            norm_aspect_ratios = []
            lefts, rights, tops, bottoms = [], [], [], []

            for x_c, y_c, w, h in bboxes:
                if w <= 0 or h <= 0:
                    continue
                if w < h:
                    norm_w, norm_h = 1.0, h / w
                else:
                    norm_w, norm_h = w / h, 1.0
                norm_aspect_ratios.append((norm_w, norm_h))

                lefts.append(x_c - w / 2)
                rights.append(x_c + w / 2)
                tops.append(y_c - h / 2)
                bottoms.append(y_c + h / 2)

            if norm_aspect_ratios:
                avg_w = np.mean([w for w, _ in norm_aspect_ratios])
                avg_h = np.mean([h for _, h in norm_aspect_ratios])

                left_margin = np.min(lefts)
                right_margin = 1.0 - np.max(rights)
                top_margin = np.min(tops)
                bottom_margin = 1.0 - np.max(bottoms)

                print(f"Processed image: {os.path.basename(image_file)} - "
                      f"boxes found: {total_boxes}, boxes removed: {removed_boxes}, "
                      f"img aspect: {image_aspect_ratio:.2f}, "
                      f"avg bbox aspect (w,h): ({avg_w:.2f}, {avg_h:.2f}), "
                      f"margins (L,R,T,B): ({left_margin:.2f}, {right_margin:.2f}, {top_margin:.2f}, {bottom_margin:.2f})")
            else:
                print(
                    f"Processed image: {os.path.basename(image_file)} - boxes found: {total_boxes}, boxes removed: {removed_boxes}, "
                    f"img aspect: {image_aspect_ratio:.2f}")

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
    if random.random() < GAUSS_NOISE_PROBABILITY:
        noisy_img = np.array(overlay.convert("RGB"), dtype=np.float32)
        std_dev = random.uniform(GAUSS_NOISE_STD_MIN, GAUSS_NOISE_STD_MAX)
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

# ========== PLOT SAMPLE NOISY IMAGES WITH BBOXES ==========

print("Collecting noisy images for plotting...")
noisy_images_paths = []

for split in ['train', 'valid', 'test']:
    images_dir = os.path.join(filtered_path, split, 'images')
    if not os.path.exists(images_dir):
        continue
    for fname in os.listdir(images_dir):
        if '_aug' in fname and fname.lower().endswith(image_extensions):
            noisy_images_paths.append(os.path.join(images_dir, fname))

if len(noisy_images_paths) >= MAX_IMAGES_TO_PLOT:
    sample_paths = random.sample(noisy_images_paths, MAX_IMAGES_TO_PLOT)
else:
    sample_paths = noisy_images_paths

fig, axes = plt.subplots(
    nrows=len(sample_paths),
    figsize=(ENLARGE_SCALE * 6, ENLARGE_SCALE * len(sample_paths)),
    dpi=100
)

# Ensure axes is iterable
if len(sample_paths) == 1:
    axes = [axes]

for ax, img_path in zip(axes, sample_paths):
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size
    ax.imshow(img)

    # Derive label path
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    split = img_path.split(os.sep)[-3]
    label_path = os.path.join(filtered_path, split, 'labels', base_name + '.txt')

    if os.path.isfile(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x_c, y_c, w, h = map(float, parts)
            x_c *= img_w
            y_c *= img_h
            w *= img_w
            h *= img_h
            x1 = x_c - w / 2
            y1 = y_c - h / 2
            rect = plt.Rectangle(
                (x1, y1), w, h,
                edgecolor='lime', facecolor='none', linewidth=2
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, f'{int(cls)}', color='lime', fontsize=10, backgroundcolor='black')

    ax.axis('off')
    ax.set_title(os.path.basename(img_path), fontsize=8)

plt.tight_layout()
plt.show()


