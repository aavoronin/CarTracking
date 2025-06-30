import cv2
import os
import time
import torch
import platform
from ultralytics import YOLO
from collections import defaultdict
from datetime import timedelta
from moviepy.video.io.VideoFileClip import VideoFileClip
#from moviepy.video.fx.resize import resize
#from moviepy.video.fx.all import resize



def normalize_path(path):
    # Convert Windows path to WSL-compatible path if needed
    if 'microsoft' in platform.uname().release.lower() and path.startswith('C:\\'):
        path = path.replace('C:\\', '/mnt/c/')
        path = path.replace('\\', '/')
    return path

def run_detection(input_files, output_file, allowed_classes, detection_threshold, box_color,
                  detect_resolution=(3840, 2160), #(2560, 1440)
                  target_resolution=(1920, 1080)):

    # Normalize all paths for WSL if needed
    input_files = [normalize_path(p) for p in input_files]
    output_file = normalize_path(output_file)

    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {device}")

    model = YOLO('yolov8n.pt')
    model.to(device)

    def draw_boxes(frame, results, scale_x, scale_y):
        counts = defaultdict(int)

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            if label not in allowed_classes or conf < detection_threshold:
                continue

            counts[label] += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            text = f"{label} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, box_color, 1, cv2.LINE_AA)

        x_offset, y_offset = 10, 20
        for i, label in enumerate(allowed_classes):
            summary = f"{label}: {counts[label]}"
            y_pos = y_offset + i * 20
            cv2.putText(frame, summary, (x_offset, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, box_color, 2, cv2.LINE_AA)

        return frame

    video_writer = None

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
            ret, frame = cap.read()
            if not ret:
                break

            detect_frame = cv2.resize(frame, detect_resolution)
            results = model(detect_frame, verbose=False)[0]

            target_frame = cv2.resize(frame, target_resolution)

            scale_x = target_resolution[0] / detect_resolution[0]
            scale_y = target_resolution[1] / detect_resolution[1]

            annotated = draw_boxes(target_frame, results, scale_x, scale_y)
            video_writer.write(annotated)

            frame_count += 1
            current_time = time.time()

            if current_time - last_report_time >= 2:
                elapsed = timedelta(seconds=int(current_time - video_start_time))
                print(f"🕒 Processed {frame_count:6} frames | Elapsed time: {elapsed}")
                last_report_time = current_time

        total_elapsed = timedelta(seconds=int(time.time() - video_start_time))
        print(f"✅ Finished {video_path}: {frame_count} frames in {total_elapsed}.")
        cap.release()

    if video_writer:
        video_writer.release()
        print(f"✅ Final output saved to: {output_file}")
    else:
        print("⚠️ No video was processed.")


def post_process_video(input_file):
    # Derive output filename by appending _processed before the extension
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_processed{ext}"

    # Load the video
    clip = VideoFileClip(input_file)

    # Resize to 1920x1080 using the resize effect
    #resized_clip = resize(clip, newsize=(1920, 1080))

    # Export the resized video
    clip.write_videofile(output_file, codec='libx264', audio_codec='aac')

    print(f"✅ Video saved to: {output_file}")


# === Example usage ===
if __name__ == "__main__":
    input_files = [
        r"C:\recordings\parque-nacional-donana_0002.mp4",
        r"C:\recordings\parque-nacional-donana_0003.mp4"
    ]
    output_file = r"C:\Kaggle\Video\Tracking\bird_tracking6.mp4"
    allowed_classes = ['car', 'truck', 'bus', 'person', 'bird', 'dog', 'cat']
    detection_threshold = 0.32
    box_color = (0, 0, 255)

    run_detection(input_files, output_file, allowed_classes, detection_threshold, box_color)
    post_process_video(output_file)

    input_files = [
        r"C:\recordings\hawick_0010.mp4",
        r"C:\recordings\hawick_0018.mp4"
    ]
    output_file = r"C:\Kaggle\Video\Tracking\car_tracking6.mp4"
    run_detection(input_files, output_file, allowed_classes, detection_threshold, box_color)
    post_process_video(output_file)
