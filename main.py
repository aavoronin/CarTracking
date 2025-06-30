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


def normalize_path(path):
    # Convert Windows path to WSL-compatible path if needed
    if 'microsoft' in platform.uname().release.lower() and path.startswith('C:\\'):
        path = path.replace('C:\\', '/mnt/c/')
        path = path.replace('\\', '/')
    return path


def time_to_seconds(t):
    """Convert time string (HH:MM:SS or MM:SS) to total seconds."""
    parts = list(map(int, t.strip().split(":")))
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid time format: {t}")

def get_text_height():
    (text_width, text_height), baseline = cv2.getTextSize(
        text='W',
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.2,
        thickness=2
    )
    return text_height + 2


def run_detection(input_files, output_file, allowed_classes, detection_threshold, box_color,
                  detect_resolution=(3840, 2160),
                  target_resolution=(1920, 1080),
                  n_history=200):
    input_files = [normalize_path(p) for p in input_files]
    output_file = normalize_path(output_file)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {device}")

    model = YOLO("yolov8m.pt")
    model.to(device)

    video_writer = None

    # Tracks object histories: {track_id: deque([(x1, y1, x2, y2), ...])}
    track_histories = defaultdict(lambda: deque(maxlen=n_history))

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

        text_height = get_text_height()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detect_frame = cv2.resize(frame, detect_resolution)

            # Use tracking-aware model call
            results = model.track(detect_frame, persist=True, verbose=False)[0]

            target_frame = cv2.resize(frame, target_resolution)
            scale_x = target_resolution[0] / detect_resolution[0]
            scale_y = target_resolution[1] / detect_resolution[1]

            counts = defaultdict(int)

            for box in results.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                track_id = int(box.id) if box.id is not None else None

                class_name = model.names.get(cls_id, "unknown")
                if conf < detection_threshold or class_name not in allowed_classes:
                    continue

                counts[class_name] += 1

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Scale boxes to match target_frame
                x1 *= scale_x
                y1 *= scale_y
                x2 *= scale_x
                y2 *= scale_y

                if track_id is not None:
                    track_histories[track_id].append((x1, y1, x2, y2))

                    # Draw track history
                    prev_line_length = 0
                    prev_i = 0
                    prev_line_length = 0
                    for i in range(1, len(track_histories[track_id])):
                        (x1_prev, y1_prev, x2_prev, y2_prev) = track_histories[track_id][prev_i]
                        (x1_curr, y1_curr, x2_curr, y2_curr) = track_histories[track_id][i]

                        cx_prev = int((x1_prev + x2_prev) / 2)
                        cy_prev = int((y1_prev + y2_prev) / 2)
                        cx_curr = int((x1_curr + x2_curr) / 2)
                        cy_curr = int((y1_curr + y2_curr) / 2)

                        line_length = math.hypot(cx_curr - cx_prev, cy_curr - cy_prev)
                        if prev_line_length + line_length > 5 or i in (1, len(track_histories[track_id])):
                            cv2.line(target_frame, (cx_prev, cy_prev), (cx_curr, cy_curr), box_color, 2)
                            prev_line_length = line_length
                            prev_i = i

                # Draw current box
                cv2.rectangle(target_frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                label = f"{class_name} {conf:.2f}"
                if track_id is not None:
                    label = f"ID {track_id} | {label}"
                cv2.putText(target_frame, label, (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

            x_offset, y_offset = 10, text_height
            for i, label in enumerate(allowed_classes):
                summary = f"{label}: {counts[label]}"
                y_pos = y_offset + i * text_height
                cv2.putText(target_frame, summary, (x_offset, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.2, box_color, 2)

            video_writer.write(target_frame)
            frame_count += 1

            if time.time() - last_report_time >= 2:
                elapsed = timedelta(seconds=int(time.time() - video_start_time))
                print(f"🕒 Processed {frame_count:6} frames | Elapsed time: {elapsed}")
                last_report_time = time.time()

            #if frame_count > 200:
            #    break

        total_elapsed = timedelta(seconds=int(time.time() - video_start_time))
        print(f"✅ Finished {video_path}: {frame_count} frames in {total_elapsed}.")
        cap.release()

    if video_writer:
        video_writer.release()
        print(f"✅ Final output saved to: {output_file}")
    else:
        print("⚠️ No video was processed.")


def post_process_video(input_file, intervals=None):
    """
    Extract specified time intervals from a video, concatenate, and save the result.

    Parameters:
        input_file (str): Path to the input video.
        intervals (list of tuple): List of (start_time, end_time) in format 'HH:MM:SS' or 'MM:SS'.
    """
    # Derive output filename
    base, ext = os.path.splitext(input_file)
    output_file = normalize_path(f"{base}_processed{ext}")

    # Load the video
    clip = VideoFileClip(normalize_path(input_file))

    if intervals:

        # Extract and concatenate subclips
        subclips = []
        for start, end in intervals:
            try:
                subclip = clip[start:end]
                subclips.append(subclip)
            except Exception as e:
                print(f"⚠️ Skipping invalid interval ({start} - {end}): {e}")
                raise e

        if not subclips:
            print("❌ No valid intervals provided.")
            return

        final_clip = concatenate_videoclips(subclips)
    else:
        final_clip = clip

    # Export the processed video
    final_clip.write_videofile(output_file, codec='libx264', audio_codec='aac')

    print(f"✅ Video saved to: {output_file}")


# === Example usage ===
if __name__ == "__main__":
    input_files = [
        # fr"C:\recordings\CanadianInspectionLanes_{i:04}.mp4" for i in range(3, 30)
        fr"C:\recordings\CanadianInspectionLanes_{i:04}.mp4" for i in range(5, 6)
    ]
    print(input_files)

    output_file = r"C:\Kaggle\Video\Tracking\car_tracking2_8.mp4"
    allowed_classes = ['car', 'truck', 'bus', 'person', 'bird', 'dog', 'cat']
    allowed_classes = ['car', 'truck', 'bus', 'motorcycle']
    detection_threshold = 0.1
    box_color = (0, 0, 255)

    run_detection(input_files, output_file, allowed_classes, detection_threshold, box_color)
    # output_file = '/mnt/c/Kaggle/Video/Tracking/car_tracking2_4.mp4'
    post_process_video(output_file)
    # post_process_video(output_file, [('00:10', '00:20'), ('00:30', '00:40')])
