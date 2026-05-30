import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import experiment modules
from pre_models_merged import (
    SimpleR3D18, SimpleCNNLSTM, SimpleCNNTransformer,
    EfficientNet3D, AttentionLSTM, GraphConvNet
)
from pre_dataloader_merged import PreEchoNetDatasetMerged, SimpleTransformMerged
from pre_config_merged import pre_config_merged

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------- GradCAM Implementation --------------------
class GradCAM:
    """
    Grad-CAM implementation for any model.
    Args:
        model: torch.nn.Module
        target_layer: target layer name (string) or layer module
        device: device
    """
    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        # Try to get target module by name
        named_modules = dict(self.model.named_modules())
        if self.target_layer not in named_modules:
            # Print available modules for debugging
            print(f"Target layer '{self.target_layer}' not found in model.")
            print("Available layers (first 50):")
            for i, (name, _) in enumerate(named_modules.items()):
                if i >= 50:
                    print("... (truncated)")
                    break
                print(f"  {name}")
            raise KeyError(f"Target layer '{self.target_layer}' not found. Please choose a valid layer from the list above.")
        
        target_module = named_modules[self.target_layer]
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, target_score=None):
        """
        Generate Grad-CAM heatmap.
        Args:
            input_tensor: input tensor (1, T, C, H, W) or (1, C, T, H, W) depending on model
            target_score: if None, uses the model output (scalar)
        Returns:
            heatmap: numpy array (H, W)
        """
        self.model.zero_grad()
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        if target_score is None:
            target_score = output.squeeze()
        
        # Backward pass
        self.model.zero_grad()
        target_score.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients  # [1, C, T, H, W] or [1, C, H, W]
        activations = self.activations  # same shape
        
        # Global average pooling of gradients over spatial-temporal dimensions
        # For 3D: pool over (T, H, W) -> channel weights
        if gradients.dim() == 5:  # (B, C, T, H, W)
            pooled_gradients = torch.mean(gradients, dim=[2, 3, 4], keepdim=True)  # (B, C, 1, 1, 1)
        elif gradients.dim() == 4:  # (B, C, H, W)
            pooled_gradients = torch.mean(gradients, dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
        else:
            raise ValueError(f"Unexpected activation dimensions: {activations.shape}")
        
        # Weighted sum of activations
        cam = torch.sum(pooled_gradients * activations, dim=1, keepdim=True)  # (B, 1, T, H, W) or (B, 1, H, W)
        cam = F.relu(cam)  # apply ReLU to keep positive influences
        
        # Normalize to [0,1] per sample
        cam = cam.squeeze().cpu().numpy()
        if cam.ndim == 3:  # (T, H, W) – average over time for visualization
            cam = np.mean(cam, axis=0)  # (H, W)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# -------------------- Helper to get target layer per model --------------------
def get_target_layer(model_name):
    """
    Returns a suggested target layer name for each model.
    These are common last convolutional layers; if they fail, the script will print available layers.
    """
    # Updated based on typical architectures; may need adjustment if models differ.
    layer_map = {
        'R3D-18': 'backbone.layer4.1.conv2',      # R3D-18 uses BasicBlock with two convs (conv1, conv2)
        'CNN-LSTM': 'cnn.7.1.conv2',              # ResNet18 last conv in layer4.1
        'CNN-Transformer': 'cnn.7.1.conv2',        # same as CNN-LSTM
        'EfficientNet-3D': 'conv3d_layers.6',      # last conv layer in the custom 3D layers (index may need check)
        'Attention-LSTM': 'cnn.7.1.conv2',         # same as CNN-LSTM
        'Graph-CNN': 'graph_conv.3'                 # last conv in graph_conv (index may need check)
    }
    return layer_map.get(model_name, None)

# -------------------- Load model --------------------
def load_trained_model(model_name, experiment_dir, device):
    """
    Load a trained model from experiment directory.
    """
    # Initialize model
    model_constructors = {
        'R3D-18': SimpleR3D18,
        'CNN-LSTM': SimpleCNNLSTM,
        'CNN-Transformer': SimpleCNNTransformer,
        'EfficientNet-3D': EfficientNet3D,
        'Attention-LSTM': AttentionLSTM,
        'Graph-CNN': GraphConvNet
    }
    if model_name not in model_constructors:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model_constructors[model_name]().to(device)
    
    # Load weights
    weight_path = Path(experiment_dir) / f"{model_name}_model.pth"
    if not weight_path.exists():
        weight_path = Path(experiment_dir) / f"{model_name}_trainer.pth"
        if weight_path.exists():
            checkpoint = torch.load(weight_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded trainer checkpoint for {model_name}")
        else:
            raise FileNotFoundError(f"No weights found for {model_name} in {experiment_dir}")
    else:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"Loaded model weights for {model_name}")
    
    model.eval()
    return model

# -------------------- Data utilities --------------------
def denormalize(tensor, mean, std):
    """
    Denormalize a tensor using mean and std.
    tensor: (C, H, W) or (T, C, H, W) normalized to approx [-1,1] (since mean=0.5, std=0.5)
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    if tensor.dim() == 4:  # (T, C, H, W)
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    denorm = tensor * std + mean
    return denorm.clamp(0, 1)

def get_sample(dataset, idx):
    """
    Retrieve a sample from dataset and return video tensor and info.
    """
    sample = dataset[idx]
    video = sample['video']  # (T, C, H, W) already normalized and standardized?
    # Note: The dataset applies SimpleTransformMerged which standardizes (mean, std) = (0.5,0.5,0.5)
    # So video is in range [-1,1] approximately.
    ef = sample['EF']
    filename = sample['filename']
    return video.unsqueeze(0).to(device), ef, filename  # add batch dim

# -------------------- Visualization --------------------
def overlay_heatmap(frame, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap on a single frame.
    frame: (H, W, 3) numpy array in [0,255] (uint8)
    heatmap: (H, W) numpy array in [0,1] (will be resized to frame size inside this function)
    """
    # Resize heatmap to match frame dimensions
    H, W = frame.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)
    
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap_color + (1 - alpha) * frame).astype(np.uint8)
    return overlay

def visualize_gradcam(model_name, video_tensor, heatmap, ef_true, ef_pred, output_path, frame_idx=0):
    """
    Generate and save Grad-CAM visualization.
    Args:
        model_name: name of the model
        video_tensor: (1, T, C, H, W) tensor
        heatmap: (H_heat, W_heat) numpy array (original heatmap size)
        ef_true: true EF value
        ef_pred: predicted EF value
        output_path: path to save image
        frame_idx: index of frame to visualize (default 0)
    """
    # Denormalize video tensor to [0,1]
    video_denorm = denormalize(video_tensor.squeeze(0).cpu(), 
                               pre_config_merged.ECHO_MEAN, 
                               pre_config_merged.ECHO_STD)
    # Select frame
    frame = video_denorm[frame_idx]  # (C, H, W)
    frame_np = frame.permute(1,2,0).numpy()  # (H, W, C)
    frame_np = (frame_np * 255).astype(np.uint8)
    
    # Overlay heatmap (resizing is now inside overlay_heatmap)
    overlay = overlay_heatmap(frame_np, heatmap, alpha=0.5)
    
    # Also resize heatmap for the middle subplot (to match frame size for better visualization)
    H, W = frame_np.shape[:2]
    heatmap_display = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(frame_np)
    axes[0].set_title(f'Original Frame (t={frame_idx})', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(heatmap_display, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay\nTrue EF: {ef_true:.1f}%, Pred EF: {ef_pred:.1f}%', fontsize=12)
    axes[2].axis('off')
    
    plt.suptitle(f'{model_name} - Grad-CAM Explanation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Grad-CAM visualization to {output_path}")

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description='Grad-CAM for EchoNet models')
    # Set default values so you can run directly without command-line arguments
    parser.add_argument('--experiment_dir', type=str, 
                        default='saved_models/experiment_20260228_181642',
                        help='Path to experiment directory containing saved models')
    parser.add_argument('--model', type=str, 
                        default='R3D-18',
                        choices=['R3D-18', 'CNN-LSTM', 'CNN-Transformer', 
                                 'EfficientNet-3D', 'Attention-LSTM', 'Graph-CNN'],
                        help='Model name')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to use')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Index of sample in the dataset')
    parser.add_argument('--frame_idx', type=int, default=0,
                        help='Frame index within the video to visualize')
    parser.add_argument('--output_dir', type=str, default='gradcam_results',
                        help='Directory to save visualizations')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load dataset (test split by default)
    print(f"Loading dataset split: {args.split}")
    dataset = PreEchoNetDatasetMerged(
        split=args.split,
        transform=SimpleTransformMerged(is_training=False),
        max_samples=None,  # use all samples
        data_seed=pre_config_merged.DATA_SEED
    )
    print(f"Dataset size: {len(dataset)}")
    
    # Get sample
    video_tensor, ef_true, filename = get_sample(dataset, args.sample_idx)
    print(f"Sample: {filename}, EF: {ef_true:.2f}")
    
    # Load model
    print(f"Loading model: {args.model}")
    model = load_trained_model(args.model, args.experiment_dir, device)
    
    # Get prediction
    with torch.no_grad():
        ef_pred = model(video_tensor).item()
    print(f"Predicted EF: {ef_pred:.2f}")
    
    # Get target layer (suggestion)
    target_layer = get_target_layer(args.model)
    if target_layer is None:
        raise ValueError(f"Target layer not defined for {args.model}. Please specify manually.")
    print(f"Suggested target layer: {target_layer}")
    
    # Generate Grad-CAM (will print available layers if suggestion fails)
    try:
        gradcam = GradCAM(model, target_layer, device)
    except KeyError as e:
        print("\nIf you see a KeyError above, please choose a valid layer from the printed list.")
        print("You can manually set the correct layer in the code (see get_target_layer function).")
        raise e
    
    heatmap = gradcam.generate_cam(video_tensor, target_score=None)  # uses model output
    
    # Visualize
    out_path = output_dir / f"{args.model}_sample{args.sample_idx}_frame{args.frame_idx}_gradcam.png"
    visualize_gradcam(args.model, video_tensor, heatmap, ef_true, ef_pred, out_path, frame_idx=args.frame_idx)
    
    print("Done.")

if __name__ == "__main__":
    main()