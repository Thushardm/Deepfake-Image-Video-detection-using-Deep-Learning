import os
import cv2

def extract_frames_from_video(video_path, output_dir, label_prefix, max_frames=50):
    """Extract frames from a video and save them."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break
        filename = f"{label_prefix}_frame_{frame_count}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), frame)
        frame_count += 1

    cap.release()
    print(f"[INFO] Extracted {frame_count} frames from {video_path}")

def process_video_list(list_path, root_video_dir, output_root):
    with open(list_path, "r") as file:
        lines = file.readlines()

    for index, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 2:
            continue  # Skip malformed lines

        label = "real" if parts[0] == "1" else "fake"
        rel_path = parts[1].replace("/", os.sep)
        video_path = os.path.join(root_video_dir, rel_path)

        if not os.path.exists(video_path):
            print(f"[WARNING] File not found: {video_path}")
            continue

        output_dir = os.path.join(output_root, label)
        label_prefix = f"{label}_{os.path.splitext(os.path.basename(video_path))[0]}"
        extract_frames_from_video(video_path, output_dir, label_prefix)

process_video_list(
    list_path="Celeb-DF/list/List_of_testing_videos.txt",
    root_video_dir="Celeb-DF/videos",
    output_root="Celeb-DF/processed_frames"
)