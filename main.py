"""
# OpenCV for video and image processing
#!pip install opencv-python

# PyTorch for deep learning and GPU acceleration
# Use the appropriate command from https://pytorch.org/get-started/locally/
# Here's a general CPU-only install:
#!pip install torch torchvision torchaudio

# Ultralytics YOLOv8 for object detection
#!pip install ultralytics

#!pip uninstall moviepy -y
#!pip install moviepy
#!pip install imageio-ffmpeg

#!pip install cucim
#!pip install moviepy==1.0.3

#import moviepy.editor as mp
#!pip install lap>=0.5.12

"""
import cv2
import torch
import os
import time
import math
import platform
from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips
from moviepy import VideoFileClip
from collections import defaultdict, deque
from datetime import timedelta
from ultralytics import YOLO
from scipy.signal import savgol_filter
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA


class YeloVideoProcessor:
    def __init__(self, model_classes, n_history=200):
        self.model_classes = model_classes
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Using device: {self.device}")

        self.models = []
        self.track_histories = defaultdict(lambda: deque(maxlen=n_history))
        self._load_models()
        self.frames_back = 100
        self.frames_forward = 120
        self.min_bboxes = 8

    def normalize_path(self, path):
        if 'microsoft' in platform.uname().release.lower() and path.startswith('C:\\'):
            path = path.replace('C:\\', '/mnt/c/').replace('\\', '/')
        return path

    def _load_models(self):
        for entry in self.model_classes:
            model_path = entry['model']
            classes = set(entry['classes'])
            model_path = self.normalize_path(model_path)
            print(f"✅ Loading model: {model_path}")
            model = YOLO(model_path).to(self.device)
            self.models.append({'model': model, 'classes': classes})
            print(f"✅ Loaded model: {model_path} with classes: {classes}")

    def get_text_height(self):
        (_, h), baseline = cv2.getTextSize('W', cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        return h + 2

    def preprocess_frame(self, detect_resolution, frame):
        return cv2.resize(frame, detect_resolution)

    def _draw_track_history(self, frame, history, color):
        prev_i = 0
        prev_line_length = 0
        for i in range(1, len(history)):
            (x1_prev, y1_prev, x2_prev, y2_prev) = history[prev_i]
            (x1_curr, y1_curr, x2_curr, y2_curr) = history[i]
            cx_prev = int((x1_prev + x2_prev) / 2)
            cy_prev = int((y1_prev + y2_prev) / 2)
            cx_curr = int((x1_curr + x2_curr) / 2)
            cy_curr = int((y1_curr + y2_curr) / 2)
            line_length = math.hypot(cx_curr - cx_prev, cy_curr - cy_prev)
            if prev_line_length + line_length > 5 or i in (1, len(history)):
                cv2.line(frame, (cx_prev, cy_prev), (cx_curr, cy_curr), color, 2)
                prev_line_length = line_length
                prev_i = i

    def determine_object_validity(self, track_id, detection_index, all_detections, model_idx,
                              screen_size):

        bboxes = self.collect_bboxes(all_detections, detection_index, model_idx, track_id)

        if len(bboxes) < self.min_bboxes:
            return False

        dx = bboxes[-1][0] - bboxes[0][0]
        dy = bboxes[-1][1] - bboxes[0][1]
        movement = math.hypot(dx, dy)
        screen_diagonal = math.hypot(*screen_size)

        return movement >= (screen_diagonal / 10)

    def collect_bboxes(self, all_detections, detection_index, model_idx, track_id):
        bboxes = []
        start = max(0, detection_index - self.frames_back)
        end = min(len(all_detections), detection_index + self.frames_forward)
        for i in range(start, end):
            frame_detections = all_detections[i]
            for record in frame_detections:
                if record['model_idx'] != model_idx:
                    continue
                if record['track_id'] != track_id:
                    continue

                x1, y1, x2, y2 = record['bbox']
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                bboxes.append((cx, cy))
        return bboxes

    def draw_data_on_frame(self, box_color, frame, valid_records, all_detections, frame_detections,
                       detection_index, crossing_lines, crossing_stats):

        self.draw_crossing_lines(frame, crossing_lines, box_color)

        # === Draw object bounding boxes and labels ===
        for rec in valid_records:
            x1, y1, x2, y2 = rec['bbox_smooth']
            conf = rec['conf_smooth']
            label = f"ID {rec['track_id']} | {rec['class_name']} {conf:.2f}"
            model_idx = rec['model_idx']
            track_id = rec['track_id']

            bboxes = self.collect_bboxes(all_detections, detection_index, model_idx, track_id)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        # === Draw crossing statistics in upper-right corner ===
        h, w = frame.shape[:2]
        x_base = 30  # Initial horiz offset
        y_base = 30             # Initial vertical offset
        line_height = 25        # Space between lines

        # Get stats for current frame
        if crossing_stats and detection_index < len(crossing_stats):
            frame_stats = crossing_stats[detection_index]
            for line_name, class_counts in frame_stats.items():
                for class_name, count in class_counts.items():
                    stat_text = f"{line_name}--{class_name}: {count}"
                    cv2.putText(
                        frame,
                        stat_text,
                        (x_base, y_base),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        box_color,
                        2,
                        cv2.LINE_AA
                    )
                    y_base += line_height

    def process_detections(self, all_detections, crossing_lines):
        def compute_section_averages(lst):
            n = len(lst)
            third = n // 3

            first_third = lst[:third]
            center_third = lst[third:2*third]
            last_third = lst[2*third:]

            avg_all = sum(lst) / n
            avg_first = sum(first_third) / len(first_third)
            avg_center = sum(center_third) / len(center_third)
            avg_last = sum(last_third) / len(last_third)

            return [avg_all, avg_first, avg_center, avg_last]

        def bbox_centers_linearity_metric(frame_map):
            """
            Calculates a scale-invariant metric of how close bounding box centers
            are to a straight line.

            Args:
                frame_map (dict): Dictionary with values containing 'bbox' key.
                                  Each bbox is [x1, y1, x2, y2].

            Returns:
                float: normalized average perpendicular distance of bbox centers
                       to best-fit line. Smaller means closer to a line.
                       Returns 0 if not enough points.
            """
            centers = []
            for record in frame_map.values():
                bbox = record['bbox']
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                centers.append([cx, cy])

            centers = np.array(centers)
            if len(centers) < 2:
                # Not enough points to define a line
                return 0.0

            x = centers[:, 0]
            y = centers[:, 1]

            # Fit line y = mx + b
            A = np.vstack([x, np.ones(len(x))]).T
            m, b = np.linalg.lstsq(A, y, rcond=None)[0]

            # Compute perpendicular distances
            distances = np.abs(m*x - y + b) / np.sqrt(m**2 + 1)
            avg_distance = distances.mean()

            # Normalize by the range of the centers along the major axis (line direction)
            # Calculate projection of points onto the line direction vector
            direction = np.array([1, m])
            direction = direction / np.linalg.norm(direction)  # unit vector

            projections = centers @ direction  # dot product of each center with direction

            range_proj = projections.max() - projections.min()
            if range_proj == 0:
                # Points are all at the same projection => no scale, consider perfect line
                return 0.0

            normalized_metric = avg_distance / range_proj
            return normalized_metric

        # Set to keep track of already processed track_ids
        used = set()

        # Initialize a new list of detections, one sublist per frame
        new_all_detections = [[] for _ in all_detections]

        # Iterate over all detections in each frame
        for detections in all_detections:
            for rec in detections:
                track_id = rec['track_id']

                # Skip if this track_id has already been processed
                if track_id in used:
                    continue
                used.add(track_id)

                # Extract identifiers for filtering
                class_name = rec['class_name']
                model_idx = rec['model_idx']

                # Collect all records with the same track_id, class, and model
                all_track_id_records = [
                    r for dets in all_detections for r in dets
                    if r['track_id'] == track_id and
                       r['class_name'] == class_name and
                       r['model_idx'] == model_idx
                ]

                # Sort these records by frame index
                all_track_id_records.sort(key=lambda r: r['frame_index'])

                # Skip if no records found
                if not all_track_id_records:
                    continue

                # Determine the full range of frame indices this track_id appears in
                frame_indices = [r['frame_index'] for r in all_track_id_records]
                min_f, max_f = min(frame_indices), max(frame_indices)

                # Build a quick lookup of frame_index -> record
                frame_map = {r['frame_index']: r for r in all_track_id_records}

                # Interpolate missing records between min and max frame_index
                for f in range(min_f, max_f + 1):
                    if f in frame_map:
                        # If frame already exists, add the original record
                        new_all_detections[f].append(frame_map[f])
                    else:
                        # Find nearest records before and after the missing frame
                        before = max(
                            (r for r in all_track_id_records if r['frame_index'] < f),
                            key=lambda r: r['frame_index'],
                            default=None
                        )
                        after = min(
                            (r for r in all_track_id_records if r['frame_index'] > f),
                            key=lambda r: r['frame_index'],
                            default=None
                        )

                        # Only interpolate if both before and after records exist
                        if before and after:
                            # Compute interpolation weight
                            alpha = (f - before['frame_index']) / (after['frame_index'] - before['frame_index'])

                            # Linearly interpolate bbox values
                            interp_bbox = [
                                before['bbox'][i] + alpha * (after['bbox'][i] - before['bbox'][i])
                                for i in range(4)
                            ]

                            # Linearly interpolate confidence value
                            interp_conf = before['conf'] + alpha * (after['conf'] - before['conf'])

                            # Create a new interpolated record
                            filled_record = {
                                'track_id': track_id,
                                'model_idx': model_idx,
                                'class_name': class_name,
                                'frame_index': f,
                                'bbox': interp_bbox,
                                'conf': interp_conf,
                                'interpolated': True
                            }

                            # Append the interpolated record to the correct frame
                            new_all_detections[f].append(filled_record)
                            frame_map[f] = filled_record

                self.create_smooth_detections(min_f, max_f, frame_map)

                # Debug output showing the number of frames this track covers
                if len(frame_map) >= 10:
                    interpolated_count = sum(1 for record in frame_map.values() if record.get('interpolated'))
                    aka_line = bbox_centers_linearity_metric(frame_map)
                    avs = compute_section_averages([frame_map[k].get('conf') for k in frame_map.keys()])
                    print(f"track_id {track_id}: {max_f - min_f + 1} total frames ({min_f} to {max_f}) inter: {interpolated_count} " + \
                          f"{interpolated_count/(max_f - min_f + 1):.3f} aka_line: {aka_line:.3f} {avs[0]:.2f} " + \
                          f"{avs[1]:.2f} {avs[2]:.2f} {avs[3]:.2f}")

        # Return the updated list of detections, with interpolated entries
        return new_all_detections


    def create_smooth_detections(self, min_f, max_f, frame_map):
        total_frames = max_f - min_f + 1

        # If too few frames, skip smoothing – just copy original values
        if total_frames < 4:
            for f in range(min_f, max_f + 1):
                rec = frame_map[f]
                rec['bbox_smooth'] = rec['bbox']
                rec['conf_smooth'] = rec['conf']
                rec['frame_map'] = frame_map
                rec['min_f'] = min_f
                rec['max_f'] = max_f
            return

        # Extract bbox and conf series
        x1_list, y1_list, x2_list, y2_list, conf_list = [], [], [], [], []
        frame_order = []

        for f in range(min_f, max_f + 1):
            rec = frame_map[f]
            x1, y1, x2, y2 = rec['bbox']
            conf = rec['conf']
            x1_list.append(x1)
            y1_list.append(y1)
            x2_list.append(x2)
            y2_list.append(y2)
            conf_list.append(conf)
            frame_order.append(f)

        # Convert to NumPy arrays
        x1_np = np.array(x1_list)
        y1_np = np.array(y1_list)
        x2_np = np.array(x2_list)
        y2_np = np.array(y2_list)
        conf_np = np.array(conf_list)

        # Choose window size (must be odd and <= total_frames)
        window = min(7, total_frames if total_frames % 2 == 1 else total_frames - 1)
        if window < 3:
            window = 3  # minimum size for savgol_filter

        # Apply smoothing filter
        x1_smooth = savgol_filter(x1_np, window_length=window, polyorder=2)
        y1_smooth = savgol_filter(y1_np, window_length=window, polyorder=2)
        x2_smooth = savgol_filter(x2_np, window_length=window, polyorder=2)
        y2_smooth = savgol_filter(y2_np, window_length=window, polyorder=2)
        conf_smooth = savgol_filter(conf_np, window_length=window, polyorder=2)

        # Assign smoothed values back to records
        for i, f in enumerate(frame_order):
            rec = frame_map[f]
            rec['bbox_smooth'] = [x1_smooth[i], y1_smooth[i], x2_smooth[i], y2_smooth[i]]
            rec['conf_smooth'] = float(conf_smooth[i])
            rec['frame_map'] = frame_map
            rec['min_f'] = min_f
            rec['max_f'] = max_f


    def collect_crossing_stats(self, all_detections, crossing_lines):
        def unique_key(rec):
            return (rec['track_id'], rec['model_idx'], rec['class_name'])

        def bbox_center(bbox):
            x1, y1, x2, y2 = bbox
            return ((x1 + x2) / 2, (y1 + y2) / 2)

        def point_side_of_line(px, py, x1, y1, x2, y2):
            return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

        def normalize(v):
            norm = np.linalg.norm(v)
            return v / norm if norm > 0 else v

        def compute_motion_vector(points, linearity_threshold=0.95):
            if len(points) < 2:
                return None

            data = np.array(points)

            # Center the data
            centered = data - np.mean(data, axis=0)

            # Run PCA
            pca = PCA(n_components=2)
            pca.fit(centered)

            # Check how linear the motion is (first PC must explain most variance)
            if pca.explained_variance_ratio_[0] < linearity_threshold:
                return None

            # Dominant direction (unit vector)
            direction = pca.components_[0]

            # Project data onto direction and compute total spread (motion magnitude)
            projections = centered @ direction
            magnitude = projections.max() - projections.min()

            # Final motion vector
            motion_vector = direction * magnitude
            #print(motion_vector)
            return motion_vector.tolist()


        def fit_motion_vector_normalized(direction):
            if direction is None:
                return None
            return normalize(direction)

        def line_direction(x1, y1, x2, y2):
            return normalize(np.array([x2 - x1, y2 - y1]))

        def dot(a, b):
            return np.dot(a, b)

        if len(all_detections) == 0:
            return None

        # --- Preparation ---
        max_frame_index = max(rec['frame_index'] for frame in all_detections for rec in frame)
        object_tracks = defaultdict(list)
        for frame_dets in all_detections:
            for rec in frame_dets:
                object_tracks[unique_key(rec)].append(rec)

        # For frame-by-frame stats
        per_frame_stats = []
        already_counted = set()

        for current_frame in range(max_frame_index + 1):
            frame_stats = defaultdict(lambda: defaultdict(int))  # line_name -> class_name -> count

            for obj_id, track in object_tracks.items():
                if obj_id in already_counted:
                    continue

                # Filter past detections up to current frame
                past_track = [r for r in track if r['frame_index'] <= current_frame]
                if len(past_track) < self.min_bboxes:
                    continue

                past_track = sorted(past_track, key=lambda r: r['frame_index'])
                centers = [bbox_center(r['bbox_smooth']) for r in past_track]

                for line_name, (lx1, ly1, lx2, ly2) in crossing_lines.items():
                    left, right = 0, 0
                    for cx, cy in centers:
                        side = point_side_of_line(cx, cy, lx1, ly1, lx2, ly2)
                        if side < 0:
                            left += 1
                        elif side > 0:
                            right += 1

                    if left < 3 or right < 3:
                        continue

                    line_vec = line_direction(lx1, ly1, lx2, ly2)

                    motion_vector = compute_motion_vector(centers)
                    motion_vector_norm = fit_motion_vector_normalized(motion_vector)
                    if motion_vector_norm is None:
                        continue

                    dot_product = dot(motion_vector_norm, line_vec)
                    if dot_product <= 0:
                        continue

                    vector_len = math.hypot(motion_vector[0], motion_vector[0])
                    print(f"{track[0].get('track_id')} crossed '{line_name}' " +
                          f"(dot_product: {dot_product:.3f}) vector_len: {vector_len:.3f}")

                    # Valid crossing
                    class_name = obj_id[2]
                    frame_stats[line_name][class_name] += 1
                    already_counted.add(obj_id)
                    break  # only one line per object

            # Combine current frame stats with previous frame's totals
            if per_frame_stats:
                prev_stats = per_frame_stats[-1]
                combined_stats = defaultdict(lambda: defaultdict(int))
                for line in crossing_lines:
                    for cls in prev_stats[line]:
                        combined_stats[line][cls] = prev_stats[line][cls]
                for line in frame_stats:
                    for cls in frame_stats[line]:
                        combined_stats[line][cls] += frame_stats[line][cls]
                per_frame_stats.append(combined_stats)
            else:
                per_frame_stats.append(frame_stats)
        return per_frame_stats

    def draw_crossing_lines(self, frame, crossing_lines, color):
        for direction, coords in crossing_lines.items():
            x1, y1, x2, y2 = coords

            # Draw dotted line: interpolate points between (x1, y1) and (x2, y2)
            num_dots = 30
            for i in range(num_dots):
                alpha = i / num_dots
                beta = (i + 0.5) / num_dots

                px1 = int(x1 * (1 - alpha) + x2 * alpha)
                py1 = int(y1 * (1 - alpha) + y2 * alpha)
                px2 = int(x1 * (1 - beta) + x2 * beta)
                py2 = int(y1 * (1 - beta) + y2 * beta)

                cv2.line(frame, (px1, py1), (px2, py2), color, 1)

            # Draw arrowhead
            cv2.arrowedLine(frame, (x1, y1), (x2, y2), color, 2, tipLength=0.02)

            # Compute label position at 75% from start to end (1/4 from the end)
            alpha = 0.75
            label_x = int(x1 * (1 - alpha) + x2 * alpha)
            label_y = int(y1 * (1 - alpha) + y2 * alpha)

            """
            # Draw direction label
            cv2.putText(
                frame,
                direction,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA
            )
            """


    def run_detection(self, input_files, output_file, detection_threshold=0.3, box_color=(0, 255, 0),
                  detect_resolution=(1280, 720), target_resolution=(1280, 720), crossing_lines = dict(),
                      limit_on_frames=9999999):

        input_files = [self.normalize_path(p) for p in input_files]
        output_file = self.normalize_path(output_file)
        text_height = self.get_text_height()

        all_detections = []  # list of lists of DetectionRecords
        video_fps = None
        frame_index = 0
        frame_idx = 0
        video_start_time = time.time()
        last_report_time = video_start_time

        print("\n🚀 FIRST PASS: Running detection...")
        for video_path in input_files:
            if not os.path.exists(video_path):
                print(f"❌ File not found: {video_path}")
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Couldn't open: {video_path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps is None:
                video_fps = fps

            print(f"🔍 Processing: {video_path} (fps: {fps})")

            while True:
                ret, frame = cap.read()
                if not ret or frame_index > limit_on_frames:
                    break

                detect_frame = self.preprocess_frame(detect_resolution, frame)
                detections_this_frame = []

                for model_idx, model_entry in enumerate(self.models):
                    model = model_entry['model']
                    classes = model_entry['classes']
                    result = model.track(detect_frame, persist=True, verbose=False)[0]

                    for box in result.boxes:
                        if box.id is None:
                            continue
                        cls_id = int(box.cls)
                        class_name = model.names.get(cls_id, "unknown")
                        if class_name not in classes:
                            continue
                        if float(box.conf) < detection_threshold:
                            continue

                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        track_id = int(box.id) + (model_idx + 1) * 1_000_000
                        if box.conf is None:
                            continue

                        conf = float(box.conf)

                        detections_this_frame.append({
                            'frame_index': frame_index,
                            'track_id': track_id,
                            'model_idx': model_idx,
                            'class_name': class_name,
                            'bbox': (x1, y1, x2, y2),
                            "conf": conf,
                        })

                all_detections.append(detections_this_frame)
                frame_index += 1

                if time.time() - last_report_time >= 2:
                    elapsed = timedelta(seconds=int(time.time() - video_start_time))
                    print(f"🕒 Processed {frame_index:6} frames | Elapsed time: {elapsed}")
                    last_report_time = time.time()

            total_elapsed = timedelta(seconds=int(time.time() - video_start_time))
            print(f"✅ Finished {video_path}: {frame_index} frames in {total_elapsed}.")
            cap.release()
            if frame_index > limit_on_frames:
                break

        all_detections = self.process_detections(all_detections, crossing_lines)
        crossing_stats = self.collect_crossing_stats(all_detections, crossing_lines)
        #print(crossing_stats[50:54])

        # === SECOND PASS ===
        print("\n✏️ SECOND PASS: Drawing and writing final output...")
        video_writer = None
        detection_index = 0
        frame_index = 0

        for video_path in input_files:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue

            while True:
                ret, frame = cap.read()
                if not ret or detection_index >= len(all_detections) or frame_index > limit_on_frames:
                    break

                valid_records = []
                screen_size = target_resolution
                frame_detections = all_detections[detection_index]

                for record in frame_detections:
                    if self.determine_object_validity(
                            track_id=record['track_id'],
                            detection_index=detection_index,
                            all_detections=all_detections,
                            model_idx=record['model_idx'],
                            screen_size=screen_size):
                        valid_records.append(record)

                # Draw detections
                frame = cv2.resize(frame, target_resolution)
                self.draw_data_on_frame(box_color, frame, valid_records, all_detections,
                                        frame_detections, detection_index, crossing_lines, crossing_stats)

                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(output_file, fourcc, video_fps, target_resolution)

                video_writer.write(frame)
                detection_index += 1
                frame_index += 1

                if time.time() - last_report_time >= 2:
                    elapsed = timedelta(seconds=int(time.time() - video_start_time))
                    print(f"🕒 Processed {frame_index:6} frames | Elapsed time: {elapsed}")
                    last_report_time = time.time()

            total_elapsed = timedelta(seconds=int(time.time() - video_start_time))
            print(f"✅ Finished {video_path}: {frame_index} frames in {total_elapsed}.")
            cap.release()

            if frame_index > limit_on_frames:
                break

        if video_writer:
            video_writer.release()
            print(f"\n✅ Output saved to: {output_file}")
        else:
            print("⚠️ No video was written.")


    def post_process_video(self, input_file, intervals=None, compression=3):
        base, ext = os.path.splitext(input_file)
        output_file = self.normalize_path(f"{base}_processed{ext}")
        clip = VideoFileClip(self.normalize_path(input_file))

        if intervals:
            subclips = []
            for start, end in intervals:
                try:
                    subclip = clip[start:end]
                    subclips.append(subclip)
                except Exception as e:
                    print(f"⚠️ Skipping invalid interval ({start} - {end}): {e}")
            if not subclips:
                print("❌ No valid intervals provided.")
                return
            final_clip = concatenate_videoclips(subclips)
        else:
            final_clip = clip

        if compression == 1:
            final_clip.write_videofile(output_file, codec='libx264', audio_codec='aac')
        elif compression == 2:
            final_clip.write_videofile(output_file, codec='libx265', audio_codec='aac',
                                       bitrate='0', ffmpeg_params=['-crf', '18', '-preset', 'slow'])
        elif compression == 3:
            final_clip.write_videofile(output_file, codec='libx265', audio_codec='aac',
                                       ffmpeg_params=['-crf', '0'])
        print(f"✅ Video saved to: {output_file}")


# === Example usage ===
if __name__ == "__main__":

    detection_threshold = 0.1
    box_color = (0, 0, 255)

    allowed_classes = ['car', 'truck', 'bus', 'person', "dog", 'motorcycle']
    model_classes = [
        #{'model': "/mnt/c/Kaggle/models/rail_cars/yolov8n_railway_model_50epoch.pt", 'classes': ['railway-car']},
        #{'model': "/mnt/c/Kaggle/models/rail_cars2/yolov8m_railway_model_200epoch.pt", 'classes': ['railway-car']},
        #{'model': "C:\Kaggle/models/rail_cars8/yolov8s_railway_model_50epoch.pt", 'classes': ['railcar']},
        #{'model': "C:\Kaggle/models/rail_cars7/yolov8m_railway_model_50epoch.pt", 'classes': ['railcar']},
        #{'model': "C:\Kaggle/models/rail_cars10/yolov8s_railway_model_50epoch.pt", 'classes': ['railcar']},
        #{'model': "C:\Kaggle/models/rail_cars11/best.pt", 'classes': ['railcar']},
        {'model': "C:\Kaggle/models/rail_cars12/yolov8m_railway_model_150epoch.pt", 'classes': ['railcar']},
        {'model': "yolov8x.pt", 'classes': ['car', 'truck', 'bus', 'person', "dog", 'motorcycle']},
    ]

    processor = YeloVideoProcessor(model_classes)

    tr=(1280, 720)

    input_files = [
        fr"C:\recordings\Glendale_Static_Ohio_{i:04}.mp4" for i in range(1, 6)
    ]
    output_file = r"C:\Kaggle\Video\Tracking\Glendale_Static_Ohio_9.mp4"
    tr=(1280, 720)
    crossing_lines = { "left-to-right": [int(tr[0] * 0.42), 0, int(tr[0] * 0.37), tr[1] - 1],
                       "right-to-left": [int(tr[0] * 0.35), tr[1] - 1, int(tr[0] * 0.40), 0] }

    processor.run_detection(input_files, output_file, detection_threshold=detection_threshold, box_color=box_color,
                            detect_resolution=(1280, 720),
                            target_resolution=(1280, 720),
                            crossing_lines=crossing_lines, limit_on_frames=700)
    processor.post_process_video(output_file, compression=1)


    output_file = r"C:\Kaggle\Video\Tracking\Tehachapi_Live_Train_Cams_32.mp4"
    input_files = [
        fr"C:\recordings\Tehachapi_Live_Train_Cams3_{i:04}.mp4" for i in range(14, 16)
    ]
    processor.run_detection(input_files, output_file, detection_threshold=detection_threshold, box_color=box_color,
                            detect_resolution=(1280, 720),
                            target_resolution=tr,
                            crossing_lines=crossing_lines, limit_on_frames=1000000)
    processor.post_process_video(output_file, compression=1) #, intervals=[('00:25', '03:50')

    crossing_lines = { "right-to-left": [int(tr[0] * 0.75), 0, int(tr[0] * 0.70), tr[1] - 1],
                       "left-to-right": [int(tr[0] * 0.75), tr[1] - 1, int(tr[0] * 0.80), 0] }
    crossing_lines = { "right-to-left": [int(tr[0] * 0.55), 0, int(tr[0] * 0.50), tr[1] - 1],
                       "left-to-right": [int(tr[0] * 0.55), tr[1] - 1, int(tr[0] * 0.60), 0] }

    output_file = r"C:\Kaggle\Video\Tracking\Tehachapi_Live_Train_Cams_33.mp4"
    input_files = [
        fr"C:\recordings\Tehachapi_Live_Train_Cams_{i:04}.mp4" for i in range(14, 16)
    ]

    processor.run_detection(input_files, output_file, detection_threshold=detection_threshold, box_color=box_color,
                            detect_resolution=(1280, 720),
                            target_resolution=tr,
                            crossing_lines=crossing_lines, limit_on_frames=1000)
    processor.post_process_video(output_file, compression=1) #, intervals=[('00:25', '03:50')

    output_file = r"C:\Kaggle\Video\Tracking\Tehachapi_Live_Train_Cams_34.mp4"
    input_files = [
        fr"C:\recordings\Tehachapi_Live_Train_Cams4_{i:04}.mp4" for i in range(1, 3)
    ]+[
        fr"C:\recordings\Tehachapi_Live_Train_Cams3_{i:04}.mp4" for i in range(1, 3)
    ]+[
        fr"C:\recordings\Tehachapi_Live_Train_Cams3_{i:04}.mp4" for i in range(33, 36)
    ]+[
        fr"C:\recordings\Tehachapi_Live_Train_Cams5_{i:04}.mp4" for i in range(1, 5)
    ]

    processor.run_detection(input_files, output_file, detection_threshold=detection_threshold, box_color=box_color,
                            detect_resolution=(1280, 720),
                            target_resolution=tr,
                            crossing_lines=crossing_lines, limit_on_frames=10000000)
    processor.post_process_video(output_file, compression=1) #, intervals=[('00:25', '03:50')


    '''
    input_files = [
        # r"C:\recordings\Safari_Kenya_0001.mp4"
        fr"C:\recordings\Tehachapi_Live_Train_Cams3_{i:04}.mp4" for i in range(1, 13)
    ]
    output_file = r"C:\Kaggle\Video\Tracking\Tehachapi_Live_Train_Cams_13.mp4"
    processor.run_detection(input_files, output_file, detection_threshold=detection_threshold, box_color=box_color,
                            detect_resolution=(1280, 720),
                            target_resolution=(1280, 720))
    processor.post_process_video(output_file, compression=1)
    #processor.post_process_video(output_file, compression=1, intervals=[('00:25', '01:25')])
    '''

