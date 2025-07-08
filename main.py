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


class YeloVideoProcessor:
    def __init__(self, model_classes, n_history=200):
        self.model_classes = model_classes
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Using device: {self.device}")

        self.models = []
        self.track_histories = defaultdict(lambda: deque(maxlen=n_history))
        self._load_models()

    def normalize_path(self, path):
        if 'microsoft' in platform.uname().release.lower() and path.startswith('C:\\'):
            path = path.replace('C:\\', '/mnt/c/').replace('\\', '/')
        return path

    def _load_models(self):
        for entry in self.model_classes:
            model_path = entry['model']
            classes = set(entry['classes'])
            model = YOLO(self.normalize_path(model_path)).to(self.device)
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

    def post_process_frame(self, frame, results, detect_resolution, target_resolution, text_height, box_color):
        target_frame = cv2.resize(frame, target_resolution)
        scale_x = target_resolution[0] / detect_resolution[0]
        scale_y = target_resolution[1] / detect_resolution[1]
        total_counts = defaultdict(int)

        for model_idx, result_obj in enumerate(results):
            model = result_obj['model']
            result = result_obj['result']
            allowed_classes = result_obj['allowed_classes']
            id_offset = (model_idx + 1) * 1_000_000  # offset for model-based ID disambiguation

            for box in result.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                raw_track_id = int(box.id) if box.id is not None else None
                track_id = raw_track_id + id_offset if raw_track_id is not None else None
                class_name = model.names.get(cls_id, "unknown")

                if conf < result_obj.get('threshold', 0.3) or class_name not in allowed_classes:
                    continue

                total_counts[class_name] += 1
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1 *= scale_x
                y1 *= scale_y
                x2 *= scale_x
                y2 *= scale_y

                if track_id is not None:
                    self.track_histories[track_id].append((x1, y1, x2, y2))
                    self._draw_track_history(target_frame, self.track_histories[track_id], box_color)

                label = f"ID {track_id} | {class_name} {conf:.2f}" if track_id is not None else f"{class_name} {conf:.2f}"
                cv2.rectangle(target_frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 1)
                cv2.putText(target_frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

        for i, label in enumerate(sorted(total_counts.keys())):
            y_pos = 10 + (i + 1) * text_height
            cv2.putText(target_frame, f"{label}: {total_counts[label]}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, box_color, 2)

        return target_frame

    def determine_object_validity(self, track_id, detection_index, all_detections, model_idx,
                              screen_size, frames_back=100, frames_forward=120, min_bboxes=10):
        bboxes = []

        start = max(0, detection_index - frames_back)
        end = min(len(all_detections), detection_index + frames_forward)

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

        if len(bboxes) < min_bboxes:
            return False

        dx = bboxes[-1][0] - bboxes[0][0]
        dy = bboxes[-1][1] - bboxes[0][1]
        movement = math.hypot(dx, dy)
        screen_diagonal = math.hypot(*screen_size)

        return movement >= (screen_diagonal / 8)


    def run_detection(self, input_files, output_file, detection_threshold=0.3, box_color=(0, 255, 0),
                  detect_resolution=(1280, 720), target_resolution=(1280, 720)):

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
                if not ret:
                    break

                detect_frame = self.preprocess_frame(detect_resolution, frame)
                detections_this_frame = []

                for model_idx, model_entry in enumerate(self.models):
                    model = model_entry['model']
                    classes = model_entry['classes']
                    #model_name = os.path.basename(model_entry['model'].weights.name)
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

                        detections_this_frame.append({
                            'frame_index': frame_index,
                            'track_id': track_id,
                            'model_idx': model_idx,
                            #'model_name': model_name,
                            'class_name': class_name,
                            'bbox': (x1, y1, x2, y2)
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
            print(f"✅ Finished processing: {video_path}")

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
                if not ret or detection_index >= len(all_detections):
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
                for rec in valid_records:
                    x1, y1, x2, y2 = rec['bbox']
                    label = f"ID {rec['track_id']} | {rec['class_name']}"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

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
        {'model': "C:\Kaggle/models/rail_cars5/yolov8n_railway_model_100epoch.pt", 'classes': ['railcar']},
        {'model': "yolov8x.pt", 'classes': ['car', 'truck', 'bus', 'person', "dog", 'motorcycle']},
    ]

    processor = YeloVideoProcessor(model_classes)

    input_files = [
        # r"C:\recordings\Safari_Kenya_0001.mp4"
        fr"C:\recordings\Tehachapi_Live_Train_Cams_{i:04}.mp4" for i in range(14, 16)
    ]
    output_file = r"C:\Kaggle\Video\Tracking\Tehachapi_Live_Train_Cams_6.mp4"
    processor.run_detection(input_files, output_file, detection_threshold=detection_threshold, box_color=box_color,
                            detect_resolution=(1280, 720),
                            target_resolution=(1280, 720))
    processor.post_process_video(output_file, compression=1)
    #processor.post_process_video(output_file, compression=1, intervals=[('00:20', '01:25')])


