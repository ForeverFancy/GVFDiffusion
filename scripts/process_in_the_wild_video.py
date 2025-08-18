import os
import cv2
import argparse
from tqdm import tqdm
from pathlib import Path
import numpy as np
from PIL import Image

def center_crop_square(image):
    h, w = image.shape[:2]
    if h == w:
        return image
    
    # Calculate dimensions for center crop
    size = min(h, w)
    y_start = (h - size) // 2
    x_start = (w - size) // 2
    
    return image[y_start:y_start + size, x_start:x_start + size]

def pad_to_square(image):
    """Pad image to make it square without cropping any content."""
    h, w = image.shape[:2]
    if h == w:
        return image
    
    # Find the larger dimension to determine square size
    size = max(h, w)
    
    # Create a black square image
    if len(image.shape) == 3:  # Color image (has channels)
        result = np.ones((size, size, image.shape[2]), dtype=image.dtype) * 255
    else:  # Grayscale image
        result = np.ones((size, size), dtype=image.dtype) * 255
    
    # Calculate padding
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    
    # Place the original image in the center of the square
    result[y_offset:y_offset + h, x_offset:x_offset + w] = image
    
    return result

def process_video(video_path, output_dir, target_size=512, pad_instead_of_crop=True):
    # Create output directory for this video
    video_name = Path(video_path).stem
    frames_dir = os.path.join(output_dir, video_name)
    os.makedirs(frames_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return None
    
    frame_count = 0
    
    # Process frames
    with tqdm(desc=f"Processing {video_name}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Make square by padding or cropping
            if pad_instead_of_crop:
                frame = pad_to_square(frame)
            else:
                frame = center_crop_square(frame)
            
            # Convert to PIL Image and resize
            frame_pil = Image.fromarray(frame)
            frame_pil = frame_pil.resize((target_size, target_size), Image.LANCZOS)
            
            # Save frame
            frame_path = os.path.join(frames_dir, f"{frame_count:06d}.png")
            frame_pil.save(frame_path)
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    return video_name if frame_count > 0 else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input videos')
    parser.add_argument('--target_size', type=int, default=512,
                        help='Target size for output frames')
    parser.add_argument('--pad_instead_of_crop', action='store_true',
                        help='Pad images to square instead of center cropping')
    args = parser.parse_args()
    
    # Create output directory
    base_dir = os.path.dirname(os.path.normpath(args.input_dir))
    output_dir = os.path.join(os.path.dirname(os.path.normpath(args.input_dir)), 'frames')
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    
    # Get all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(Path(args.input_dir).glob(f'*{ext}')))
    
    if not video_files:
        print(f"No video files found in {args.input_dir}")
        return
    
    # Process each video
    processed_videos = []
    for video_path in sorted(video_files):
        video_name = process_video(str(video_path), args.output_dir, args.target_size, args.pad_instead_of_crop)
        if video_name:
            processed_videos.append(video_name)
    
    # Save video names to txt file
    if processed_videos:
        txt_path = os.path.join(base_dir, 'video_names.txt')
        with open(txt_path, 'w') as f:
            for video_name in processed_videos:
                f.write(f"{video_name} 0 1\n")  # Default start frame as 0
        print(f"\nProcessed videos list saved to: {txt_path}")
    
    print(f"\nProcessed {len(processed_videos)} videos")

if __name__ == '__main__':
    main()
