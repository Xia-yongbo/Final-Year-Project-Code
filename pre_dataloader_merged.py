import os
import numpy as np
import torch
import torch.nn.functional as F
import random
from torch.utils.data import Dataset, DataLoader
from pre_config_merged import pre_config_merged  # Note: may need to be changed to main_config in actual use

# Use main_config here for compatibility with the main experiment (make sure it is imported correctly)
try:
    from main_config_merged import main_config as cfg
except ImportError:
    cfg = pre_config_merged  # Fallback handling

class PreEchoNetDatasetMerged(Dataset):
    """Dataset supporting adaptive pooling for dynamic frame lengths"""
    def __init__(self, split, transform=None, is_training=True, max_samples=None, data_seed=42):
        self.split = split
        self.is_training = is_training
        self.transform = transform
        self.data_dir = os.path.join(cfg.NPZ_DIR, split)
        self.target_frames = cfg.NUM_FRAMES
        
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        if max_samples is not None and len(all_files) > max_samples:
            np.random.seed(data_seed)
            self.file_list = np.random.choice(all_files, max_samples, replace=False).tolist()
        else:
            self.file_list = all_files
        
        print(f"Loaded {split} set: {len(self.file_list)} samples (seed: {data_seed})")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = np.load(file_path, allow_pickle=True)
        video = data['video']          # Original shape: (T_orig, H, W, C) or (T, C, H, W)
        labels = data['labels'].item()
        original_length = video.shape[0]
        
        # Fix dimensions to (T, C, H, W)
        video = self._fix_video_dimensions(video)
        
        # Convert to tensor and normalize to [0, 1]
        video = torch.from_numpy(video).float() / 255.0
        
        # ========== Adaptive temporal pooling (replacing zero padding) ==========
        # Keep the first min(original_length, 64) frames to avoid overly long sequences
        max_keep = 64
        if video.shape[0] > max_keep:
            video = video[:max_keep]
        
        # Use adaptive average pooling to standardize the sequence to the target number of frames
        # Input: (T, C, H, W) -> reshape to (1, C, T, H, W) for 3D pooling
        video = video.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
        video = F.adaptive_avg_pool3d(video, (self.target_frames, cfg.IMG_SIZE[0], cfg.IMG_SIZE[1]))
        video = video.squeeze(0).permute(1, 0, 2, 3)    # Convert back to (T, C, H, W)
        # =================================================
        
        # Data augmentation before standardization
        if self.transform:
            video = self.transform(video)
        else:
            mean = torch.tensor(cfg.ECHO_MEAN).view(1, 3, 1, 1)
            std = torch.tensor(cfg.ECHO_STD).view(1, 3, 1, 1)
            video = (video - mean) / std
        
        edv = torch.tensor(labels['EDV'], dtype=torch.float32)
        esv = torch.tensor(labels['ESV'], dtype=torch.float32)
        ef = torch.tensor(labels['EF'], dtype=torch.float32)
        
        return {
            'video': video,
            'EDV': edv,
            'ESV': esv,
            'EF': ef,
            'filename': labels['filename'],
            'length': original_length
        }
    
    def _fix_video_dimensions(self, video):
        if len(video.shape) == 4:
            if video.shape[-1] == 3:
                video = np.transpose(video, (0, 3, 1, 2))
            elif video.shape[1] == 3:
                pass
            else:
                print(f"Warning: unrecognized video format: {video.shape}")
        return video


class ModerateTransform:
    """Moderate data augmentation to avoid excessive distortion"""
    def __init__(self, is_training=True):
        self.is_training = is_training
        self.mean = torch.tensor(cfg.ECHO_MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg.ECHO_STD).view(1, 3, 1, 1)
    
    def __call__(self, video):
        # video: (T, C, H, W), range [0, 1]
        T, C, H, W = video.shape
        
        if self.is_training:
            # 1. Random horizontal flip (probability 0.5)
            if random.random() > 0.5:
                video = torch.flip(video, dims=[3])
            
            # 2. Gamma correction (range narrowed to [0.7, 1.3])
            gamma = random.uniform(0.7, 1.3)
            video = torch.clamp(video, 0.0, 1.0) ** (1.0 / gamma)
            
            # 3. Random scaling (factor 0.9 to 1.1)
            scale = random.uniform(0.9, 1.1)
            if scale != 1.0:
                new_H = int(H * scale)
                new_W = int(W * scale)
                video_scaled = video.permute(1, 0, 2, 3)  # (C, T, H, W)
                video_scaled = F.interpolate(video_scaled, size=(new_H, new_W),
                                             mode='bilinear', align_corners=False)
                video_scaled = video_scaled.permute(1, 0, 2, 3)
                
                if scale < 1.0:
                    pad_h = (H - new_H) // 2
                    pad_w = (W - new_W) // 2
                    padded = torch.zeros(T, C, H, W, dtype=video.dtype, device=video.device)
                    padded[:, :, pad_h:pad_h+new_H, pad_w:pad_w+new_W] = video_scaled
                    video = padded
                else:
                    start_h = (new_H - H) // 2
                    start_w = (new_W - W) // 2
                    video = video_scaled[:, :, start_h:start_h+H, start_w:start_w+W]
            
            # 4. Random translation (range reduced to [-5, 5] pixels)
            dx = random.randint(-5, 5)
            dy = random.randint(-5, 5)
            if dx != 0 or dy != 0:
                translated = torch.roll(video, shifts=(dy, dx), dims=(2, 3))
                if dx > 0:
                    translated[:, :, :, :dx] = 0
                elif dx < 0:
                    translated[:, :, :, dx:] = 0
                if dy > 0:
                    translated[:, :, :dy, :] = 0
                elif dy < 0:
                    translated[:, :, dy:, :] = 0
                video = translated
        
        # Standardization
        video = (video - self.mean) / self.std
        return video


def get_main_dataloaders(data_seed=42):
    """Main experiment data loaders (using the full dataset)"""
    from torch.utils.data import DataLoader
    
    train_dataset = PreEchoNetDatasetMerged(
        'train',
        transform=ModerateTransform(is_training=True),
        max_samples=cfg.TRAIN_SAMPLES,
        data_seed=data_seed
    )
    val_dataset = PreEchoNetDatasetMerged(
        'val',
        transform=ModerateTransform(is_training=False),
        max_samples=cfg.VAL_SAMPLES,
        data_seed=data_seed
    )
    test_dataset = PreEchoNetDatasetMerged(
        'test',
        transform=ModerateTransform(is_training=False),
        max_samples=cfg.TEST_SAMPLES,
        data_seed=data_seed
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return train_loader, val_loader, test_loader