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
    def __init__(self, model_path="yolov8x.pt", n_history=200):
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Using device: {self.device}")
        self.model = YOLO(self.model_path).to(self.device)
        self.track_histories = defaultdict(lambda: deque(maxlen=n_history))
        self.standard_resolutions = [
            (256, 144, "144p (Low)"),
            (426, 240, "240p (Low)"),
            (640, 360, "360p (SD)"),
            (854, 480, "480p (SD)"),
            (1280, 720, "720p (HD)"),
            (1366, 768, "WXGA (HD+)"),
            (1600, 900, "900p (HD+)"),
            (1920, 1080, "1080p (Full HD)"),
            (2048, 1080, "2K (DCI)"),
            (2560, 1440, "1440p (QHD)"),
            (3200, 1800, "1800p (QHD+)"),
            (3840, 2160, "2160p (4K UHD)"),
            (4096, 2160, "4K (DCI)"),
            (5120, 2880, "5K"),
            (7680, 4320, "4320p (8K UHD)"),
        ]

    def normalize_path(self, path):
        if 'microsoft' in platform.uname().release.lower() and path.startswith('C:\\'):
            path = path.replace('C:\\', '/mnt/c/').replace('\\', '/')
        return path

    def time_to_seconds(self, t):
        parts = list(map(int, t.strip().split(":")))
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        elif len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        raise ValueError(f"Invalid time format: {t}")

    def get_text_height(self):
        (_, h), baseline = cv2.getTextSize('W', cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        return h + 2

    def run_detection(self, input_files, output_file, allowed_classes, detection_threshold, box_color,
                      detect_resolution=(3840, 2160), target_resolution=(1920, 1080)):
        input_files = [self.normalize_path(p) for p in input_files]
        output_file = self.normalize_path(output_file)
        video_writer = None
        text_height = self.get_text_height()

        for video_path in input_files:
            if not os.path.exists(video_path):
                print(f"❌ File not found: {video_path}")
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"❌ Couldn't open: {video_path}")
                continue

            print(f"\n📂 Processing: {video_path}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_file, fourcc, fps, target_resolution)

            frame_count = 0
            video_start_time = time.time()
            last_report_time = video_start_time

            while True:
                #step 1 start
                ret, frame = cap.read()
                if not ret:
                    break
                #step 2 start
                detect_frame = self.preprocess_frame(detect_resolution, frame)
                #step 3 start
                results = self.model.track(detect_frame, persist=True, verbose=False)[0]
                #step 4 start
                target_frame = self.post_process_frame(allowed_classes, box_color, detect_resolution,
                                                       detection_threshold, frame, results, target_resolution,
                                                       text_height)
                #step 5 start
                video_writer.write(target_frame)
                frame_count += 1

                if time.time() - last_report_time >= 2:
                    elapsed = timedelta(seconds=int(time.time() - video_start_time))
                    print(f"🕒 Processed {frame_count:6} frames | Elapsed time: {elapsed}")
                    last_report_time = time.time()

            total_elapsed = timedelta(seconds=int(time.time() - video_start_time))
            print(f"✅ Finished {video_path}: {frame_count} frames in {total_elapsed}.")
            cap.release()

        if video_writer:
            video_writer.release()
            print(f"✅ Final output saved to: {output_file}")
        else:
            print("⚠️ No video was processed.")

    def post_process_frame(self, allowed_classes, box_color, detect_resolution, detection_threshold, frame, results,
                           target_resolution, text_height):
        target_frame = cv2.resize(frame, target_resolution)
        scale_x = target_resolution[0] / detect_resolution[0]
        scale_y = target_resolution[1] / detect_resolution[1]
        counts = defaultdict(int)
        for box in results.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            track_id = int(box.id) if box.id is not None else None
            class_name = self.model.names.get(cls_id, "unknown")

            if conf < detection_threshold or class_name not in allowed_classes:
                continue

            counts[class_name] += 1

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
        for i, label in enumerate(allowed_classes):
            summary = f"{label}: {counts[label]}"
            y_pos = 10 + (i + 1) * text_height
            cv2.putText(target_frame, summary, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.2, box_color, 2)
        return target_frame

    def preprocess_frame(self, detect_resolution, frame):
        detect_frame = cv2.resize(frame, detect_resolution)
        return detect_frame

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
    processor = YeloVideoProcessor()

    detection_threshold = 0.1
    box_color = (0, 0, 255)

    allowed_classes = ["cow", "horse", "zebra"]  # , "bird", "cat", "dog", "sheep", "elephant", "bear", "giraffe"]
    input_files = [
        # r"C:\recordings\Safari_Kenya_0001.mp4"
        fr"C:\recordings\Safari_Kenya_{i:04}.mp4" for i in range(1, 2)
    ]
    output_file = r"C:\Kaggle\Video\Tracking\Safari_Kenya_3.mp4"
    processor.run_detection(input_files, output_file, allowed_classes, detection_threshold=detection_threshold, box_color=box_color,
                            detect_resolution=(1280, 720),
                            target_resolution=(1280, 720))
    processor.post_process_video(output_file, compression=1, intervals=[('00:20', '01:25')])

    allowed_classes = ['car', 'truck', 'bus', 'person', 'bird', 'dog', 'cat']
    allowed_classes = ['car', 'truck', 'bus', 'motorcycle']

    input_files = [
        r"C:\recordings\4_Corners_Camera_Downtown_0013.mp4"
    ]
    output_file = r"C:\Kaggle\Video\Tracking\4_Corners_Camera_3.mp4"

    # processor.run_detection(input_files, output_file, allowed_classes, detection_threshold=detection_threshold, box_color=box_color,
    #                         detect_resolution=(1920, 1080),
    #                         target_resolution=(1280, 720))
    # processor.post_process_video(output_file, compression=1, intervals=[('00:20', '01:25')])

    input_files = [
        # r"C:\recordings\4_Corners_Camera_Downtown_0013.mp4"
        fr"C:\recordings\CanadianInspectionLanes_{i:04}.mp4" for i in range(8, 25)
    ]
    output_file = r"C:\Kaggle\Video\Tracking\4_Corners_Camera_2.mp4"

    # processor.run_detection(input_files, output_file, allowed_classes, detection_threshold=detection_threshold, box_color=box_color,
    #                         detect_resolution=(1920, 1080),
    #                         target_resolution=(1280, 720))
    # processor.post_process_video(output_file, compression=1)

    input_files = [r"C:\recordings\Moscow Cars 20250702.mp4"]
    output_file = r"C:\Kaggle\Video\Tracking\MoscowCars_2.mp4"

    # processor.run_detection(input_files, output_file, allowed_classes, detection_threshold=detection_threshold, box_color=box_color)
    # processor.post_process_video(output_file, compression=1)

    input_files = [
        # fr"C:\recordings\CanadianInspectionLanes_{i:04}.mp4" for i in range(3, 30)
        fr"C:\recordings\CanadianInspectionLanes_{i:04}.mp4" for i in range(5, 6)
    ]

    output_file = r"C:\Kaggle\Video\Tracking\car_tracking2_11.mp4"
    allowed_classes = ['car', 'truck', 'bus', 'person', 'bird', 'dog', 'cat']
    allowed_classes = ['car', 'truck', 'bus', 'motorcycle']
    detection_threshold = 0.1
    box_color = (0, 0, 255)

    # processor.run_detection(input_files, output_file, allowed_classes, detection_threshold=detection_threshold, box_color=box_color,
    #                         target_resolution=(1280, 720))
    # output_file = '/mnt/c/Kaggle/Video/Tracking/car_tracking2_4.mp4'
    # processor.post_process_video(output_file, [('00:00', '00:59')], 3)
    # processor.post_process_video(output_file, [('00:00', '00:59')], 2)
    # processor.post_process_video(output_file, [('00:00', '00:30')], 2)

    input_files = [
        # fr"C:\recordings\CanadianInspectionLanes_{i:04}.mp4" for i in range(3, 30)
        fr"C:\recordings\timessquare_{i:04}.mp4" for i in range(1, 6)
    ]

    output_file = r"C:\Kaggle\Video\Tracking\timessquare_1.mp4"
    allowed_classes = ['car', 'truck', 'bus', 'person']
    detection_threshold = 0.1
    box_color = (0, 0, 255)

    # processor.run_detection(input_files, output_file, allowed_classes, detection_threshold=detection_threshold, box_color=box_color)
    # processor.post_process_video(output_file)

    input_files = [
        fr"C:\recordings\timessquare_{i:04}.mp4" for i in range(1, 2)
    ]
    output_file = r"C:\Kaggle\Video\Tracking\timessquare_2.mp4"
    # processor.run_detection(input_files, output_file, allowed_classes, detection_threshold=detection_threshold, box_color=box_color)
    # processor.post_process_video(output_file, [('00:00', '00:59')])
