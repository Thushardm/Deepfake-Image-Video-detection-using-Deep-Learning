import os
import cv2
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def extract_frames_batch(video_paths, output_dirs, label_prefixes, max_frames=50):
    """Extract frames from multiple videos in parallel"""
    
    def process_single_video(args):
        video_path, output_dir, label_prefix = args
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frames_extracted = 0
        frame_count = 0
        
        while frames_extracted < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            filename = f"{label_prefix}_frame_{frames_extracted}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            frames_extracted += 1
            frame_count += 1
            
        cap.release()
        return frames_extracted
    
    # Prepare arguments for parallel processing
    args_list = list(zip(video_paths, output_dirs, label_prefixes))
    
    # Process videos in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_single_video, args_list))
    
    print(f"Processed {len(video_paths)} videos, extracted {sum(results)} total frames")

def process_video_list_batch(list_path, root_video_dir, output_root, batch_size=10):
    """Process video list in batches"""
    with open(list_path, "r") as file:
        lines = file.readlines()
    
    video_paths = []
    output_dirs = []
    label_prefixes = []
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 2:
            continue
            
        label = "real" if parts[0] == "1" else "fake"
        rel_path = parts[1].replace("/", os.sep)
        video_path = os.path.join(root_video_dir, rel_path)
        
        if os.path.exists(video_path):
            video_paths.append(video_path)
            output_dirs.append(os.path.join(output_root, label))
            label_prefixes.append(f"{label}_{os.path.splitext(os.path.basename(video_path))[0]}")
    
    # Process in batches
    for i in range(0, len(video_paths), batch_size):
        batch_videos = video_paths[i:i+batch_size]
        batch_outputs = output_dirs[i:i+batch_size]
        batch_labels = label_prefixes[i:i+batch_size]
        
        print(f"Processing batch {i//batch_size + 1}/{(len(video_paths)-1)//batch_size + 1}")
        extract_frames_batch(batch_videos, batch_outputs, batch_labels)

if __name__ == "__main__":
    process_video_list_batch(
        list_path="../data/Celeb-DF/list/List_of_testing_videos.txt",
        root_video_dir="../data/Celeb-DF/videos",
        output_root="../data/Celeb-DF/processed_frames"
    )
