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


    def run_detection(self, input_files, output_file, detection_threshold=0.3, box_color=(0, 255, 0),
                      detect_resolution=(3840, 2160), target_resolution=(1920, 1080)):

        input_files = [self.normalize_path(p) for p in input_files]
        output_file = self.normalize_path(output_file)
        text_height = self.get_text_height()

        # Store all detections across all input videos
        all_detections = []  # List of (video_path, frame_index, detection_result)
        video_frames_info = []  # List of (video_path, frame_idx) to help during second pass
        video_fps = None

        # === FIRST PASS ===
        print("\n🚀 FIRST PASS: Running detection across all input videos...")
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
                video_fps = fps  # Use FPS from first valid video
            print(f"\n🔍 Processing (detection only): {video_path} (fps: {fps})")

            frame_idx = 0
            video_start_time = time.time()
            last_report_time = video_start_time
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                detect_frame = self.preprocess_frame(detect_resolution, frame)
                frame_results = []

                for model_entry in self.models:
                    model = model_entry['model']
                    classes = model_entry['classes']
                    if frame_idx == 0:
                        print(classes)
                    result = model.track(detect_frame, persist=True, verbose=False)[0]
                    frame_results.append({
                        'model': model,
                        'result': result,
                        'allowed_classes': classes,
                        'threshold': detection_threshold
                    })

                all_detections.append(frame_results)
                video_frames_info.append((video_path, frame_idx))
                frame_idx += 1

                if time.time() - last_report_time >= 2:
                    elapsed = timedelta(seconds=int(time.time() - video_start_time))
                    print(f"🕒 Processed {frame_idx:6} frames | Elapsed time: {elapsed}")
                    last_report_time = time.time()

            total_elapsed = timedelta(seconds=int(time.time() - video_start_time))
            print(f"✅ Finished {video_path}: {frame_idx} frames in {total_elapsed}.")
            cap.release()

        # === SECOND PASS ===
        print("\n✏️ SECOND PASS: Drawing results and writing final output...")
        video_writer = None
        detection_index = 0

        video_start_time = time.time()
        last_report_time = video_start_time
        for video_path in input_files:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Couldn't reopen: {video_path}")
                continue

            frame_idx = 0
            print(f"\n🖼️ Processing video: {video_path}")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if detection_index >= len(all_detections):
                    print(f"⚠️ No stored detection for frame {detection_index}")
                    break

                frame_detections = all_detections[detection_index]

                processed_frame = self.post_process_frame(
                    frame,
                    frame_detections,
                    detect_resolution,
                    target_resolution,
                    text_height,
                    box_color
                )

                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(output_file, fourcc, video_fps, target_resolution)

                video_writer.write(processed_frame)
                detection_index += 1
                frame_idx += 1
                if time.time() - last_report_time >= 2:
                    elapsed = timedelta(seconds=int(time.time() - video_start_time))
                    print(f"🕒 Processed {frame_idx:6} frames | Elapsed time: {elapsed}")
                    last_report_time = time.time()

            total_elapsed = timedelta(seconds=int(time.time() - video_start_time))
            print(f"✅ Finished {video_path}: {frame_idx} frames in {total_elapsed}.")

            cap.release()

        if video_writer:
            video_writer.release()
            print(f"\n✅ Final output saved to: {output_file}")
        else:
            print("⚠️ No video frames were written.")


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


