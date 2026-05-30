import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from main_config_merged import main_config as config  

class EchoNetPreprocessor:
    def __init__(self):
        self.config = config
        self._create_directories()
        self.filelist_df = pd.read_csv(config.FILELIST_PATH)
        
    def _create_directories(self):
        """Create the required output directories."""
        os.makedirs(config.NPZ_DIR, exist_ok=True)
        for split in config.SPLITS.values():
            os.makedirs(os.path.join(config.NPZ_DIR, split.lower()), exist_ok=True)
    
    def _read_video(self, video_path):
        """Read the video file, extract frames, and return a frame array with shape (T, H, W, C)."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()
        return np.array(frames)
    
    def _resize_frames(self, frames, target_size):
        """Resize frames from (T, H, W, C) to (T, target_H, target_W, C)."""
        resized = []
        for frame in frames:
            resized_frame = cv2.resize(frame, target_size)
            resized.append(resized_frame)
        return np.array(resized)
    
    def _crop_to_motion(self, frames, threshold=10, min_frames=16):
        """
        Automatically crop static or black-screen segments at the beginning and end of the video
        based on inter-frame differences.
        Only the effective cardiac motion interval is retained.
        
        Args:
            frames: np.array (T, H, W, C), pixel range 0-255
            threshold: Mean frame-difference threshold; values below this are treated as static
            min_frames: Minimum number of frames to keep
        
        Returns:
            cropped_frames: Cropped frame array
        """
        if len(frames) < min_frames:
            return frames
        
        # Convert to grayscale and compute differences between adjacent frames
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
        diffs = []
        for i in range(1, len(gray_frames)):
            diff = np.mean(np.abs(gray_frames[i].astype(np.float32) - 
                                  gray_frames[i-1].astype(np.float32)))
            diffs.append(diff)
        diffs = np.array(diffs)
        
        # Find the start and end indices of motion
        moving = diffs > threshold
        if not np.any(moving):
            # If motion is very weak throughout the video, keep all frames
            return frames
        
        start_idx = np.argmax(moving)
        end_idx = len(moving) - np.argmax(moving[::-1]) - 1
        
        # Ensure that at least min_frames frames are retained
        if end_idx - start_idx + 1 < min_frames:
            mid = (start_idx + end_idx) // 2
            half = min_frames // 2
            start_idx = max(0, mid - half)
            end_idx = min(len(frames)-1, mid + half)
        
        return frames[start_idx:end_idx+1]
    
    def _temporal_resample(self, frames, target_length):
        """
        Resample the video to a fixed number of frames using linear interpolation,
        completely avoiding zero padding.
        
        Args:
            frames: np.array (T, H, W, C)
            target_length: Target number of frames
        
        Returns:
            resampled: (target_length, H, W, C)
        """
        T, H, W, C = frames.shape
        if T == target_length:
            return frames
        
        # Create original frame indices (0 to T-1) and target frame indices (0 to target_length-1)
        orig_idx = np.linspace(0, T-1, T)
        target_idx = np.linspace(0, T-1, target_length)
        
        resampled = np.zeros((target_length, H, W, C), dtype=frames.dtype)
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    resampled[:, h, w, c] = np.interp(target_idx, orig_idx, frames[:, h, w, c])
        
        return resampled.astype(frames.dtype)
    
    def process_single_video(self, row):
        """Process a single video (new version: zero padding -> motion cropping + interpolation resampling)."""
        try:
            filename = row['FileName']
            split = row['Split']
            edv = row['EDV']
            esv = row['ESV']
            ef = row['EF']
            
            video_path = os.path.join(self.config.VIDEOS_DIR, f"{filename}.avi")
            if not os.path.exists(video_path):
                print(f"Warning: video file does not exist: {video_path}")
                return None
            
            # 1. Read raw frames
            frames = self._read_video(video_path)
            if len(frames) == 0:
                print(f"Warning: unable to read video: {video_path}")
                return None
            
            # 2. Automatically crop static or black-screen segments at the beginning and end
            frames_cropped = self._crop_to_motion(frames)
            
            # 3. Spatial resizing
            frames_resized = self._resize_frames(frames_cropped, self.config.IMG_SIZE)
            
            # 4. Temporal resampling to a fixed length without zero padding
            frames_normalized = self._temporal_resample(frames_resized, self.config.NUM_FRAMES)
            
            # 5. Convert to (T, C, H, W) for direct use in later training
            # Original shape is (T, H, W, C) -> transpose to (T, C, H, W)
            frames_normalized = np.transpose(frames_normalized, (0, 3, 1, 2))
            
            # Prepare labels
            labels = {
                'EDV': edv,
                'ESV': esv,
                'EF': ef,
                'filename': filename,
                'original_length': len(frames_cropped)  # Record the original length after cropping (optional)
            }
            
            # Save as an NPZ file
            output_path = os.path.join(self.config.NPZ_DIR, split.lower(), f"{filename}.npz")
            np.savez_compressed(output_path,
                                video=frames_normalized,
                                labels=labels)
            
            return filename, True
            
        except Exception as e:
            print(f"Error while processing video {row.get('FileName', 'unknown')}: {e}")
            return row.get('FileName', 'unknown'), False
    
    def process_all_videos(self, num_workers=4):
        """Process all videos using multiprocessing."""
        print("Start processing all videos (new version: motion cropping + interpolation resampling, no zero padding)...")
        
        valid_data = self.filelist_df.dropna(subset=['EDV', 'ESV', 'EF'])
        print(f"Number of valid samples: {len(valid_data)}")
        
        for split_name in self.config.SPLITS.values():
            split_data = valid_data[valid_data['Split'] == split_name]
            print(f"{split_name} set: {len(split_data)} samples")
        
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(self.process_single_video,
                                          [row for _, row in valid_data.iterrows()]),
                              total=len(valid_data),
                              desc="Processing videos"))
        
        successful = [r for r in results if r and r[1]]
        failed = [r for r in results if r and not r[1]]
        
        print(f"\nProcessing completed!")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        self._save_processing_log(successful, failed)
    
    def _save_processing_log(self, successful, failed):
        log_path = os.path.join(self.config.OUTPUT_DIR, "preprocessing_log.txt")
        with open(log_path, 'w') as f:
            f.write("Video preprocessing log (new version: no zero padding)\n")
            f.write("="*50 + "\n")
            f.write(f"Successfully processed: {len(successful)}\n")
            f.write(f"Failed to process: {len(failed)}\n\n")
            if failed:
                f.write("List of failed files:\n")
                for filename, _ in failed:
                    f.write(f"{filename}\n")

def main():
    preprocessor = EchoNetPreprocessor()
    preprocessor.process_all_videos(num_workers=4)

if __name__ == "__main__":
    main()