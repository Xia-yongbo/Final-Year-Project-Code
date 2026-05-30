import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from main_config_merged import main_config as cfg
from pre_dataloader_merged import PreEchoNetDatasetMerged, ModerateTransform
from pre_models_merged import (
    SimpleR3D18,
    EfficientNet3D,
    AttentionLSTM,
    SimpleCNNLSTM,
    SimpleCNNTransformer,
    GraphConvNet
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================================
# Grad-CAM
# ============================================================
class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.handles = []

        self._disable_inplace_relu()
        self._register_hooks()

    def _disable_inplace_relu(self):
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

    def _register_hooks(self):
        modules = dict(self.model.named_modules())

        if self.target_layer_name not in modules:
            print(f"\nTarget layer not found: {self.target_layer_name}")
            print("\nAvailable layers:")
            for name in modules.keys():
                print(name)
            raise KeyError(f"Layer {self.target_layer_name} not found.")

        target_layer = modules[self.target_layer_name]

        def forward_hook(module, input, output):
            self.activations = output.detach().clone()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach().clone()

        self.handles.append(target_layer.register_forward_hook(forward_hook))
        self.handles.append(target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.handles:
            h.remove()

    def generate(self, input_tensor):
        self.model.zero_grad()

        with torch.backends.cudnn.flags(enabled=False):
            self.model.eval()

            output = self.model(input_tensor)
            score = output.squeeze()

            self.model.zero_grad()
            score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM failed: gradients or activations are None.")

        gradients = self.gradients
        activations = self.activations

        if gradients.dim() == 5:
            weights = torch.mean(gradients, dim=(2, 3, 4), keepdim=True)
        elif gradients.dim() == 4:
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        else:
            raise ValueError(f"Unsupported gradient shape: {gradients.shape}")

        cam = torch.sum(weights * activations, dim=1)
        cam = F.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()

        # If it is a 3D CAM: (T, H, W), retain the time dimension
        if cam.ndim == 3:
            cam_3d = cam
            cam_2d = np.mean(cam, axis=0)
        else:
            cam_3d = None
            cam_2d = cam

        cam_2d = normalize_cam(cam_2d)

        if cam_3d is not None:
            cam_3d = np.array([normalize_cam(c) for c in cam_3d])

        return cam_2d, cam_3d, output.detach().cpu().item()


# ============================================================
# Utility
# ============================================================
def normalize_cam(cam):
    cam_min = cam.min()
    cam_max = cam.max()
    return (cam - cam_min) / (cam_max - cam_min + 1e-8)


def denormalize_video(video_tensor):
    mean = torch.tensor(cfg.ECHO_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(cfg.ECHO_STD).view(1, 3, 1, 1)

    video = video_tensor.cpu() * std + mean
    video = torch.clamp(video, 0, 1)

    return video


def overlay_heatmap(frame_rgb, heatmap, alpha=0.45):
    H, W = frame_rgb.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (W, H))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = alpha * heatmap_color + (1 - alpha) * frame_rgb
    overlay = np.uint8(overlay)

    return overlay, heatmap_resized


def tensor_frame_to_uint8(video_tensor, frame_idx):
    video_denorm = denormalize_video(video_tensor.squeeze(0))
    T = video_denorm.shape[0]
    frame_idx = min(frame_idx, T - 1)

    frame = video_denorm[frame_idx]
    frame_np = frame.permute(1, 2, 0).numpy()
    frame_np = np.uint8(frame_np * 255)

    return frame_np


# ============================================================
# Load Data / Model
# ============================================================
def find_latest_experiment_dir():
    saved_dir = Path("saved_models")

    if not saved_dir.exists():
        raise FileNotFoundError("Cannot find saved_models folder.")

    experiment_dirs = list(saved_dir.glob("main_experiment_*"))

    if len(experiment_dirs) == 0:
        raise FileNotFoundError("Cannot find main_experiment_* folder in saved_models.")

    latest_dir = sorted(experiment_dirs, key=lambda p: p.stat().st_mtime)[-1]

    print(f"Using experiment folder: {latest_dir}")

    return latest_dir


def load_model(model_name, experiment_dir):
    model_classes = {
        "R3D-18": SimpleR3D18,
        "EfficientNet-3D": EfficientNet3D,
        "Attention-LSTM": AttentionLSTM,
        "CNN-LSTM": SimpleCNNLSTM,
        "CNN-Transformer": SimpleCNNTransformer,
        "Graph-CNN": GraphConvNet
    }

    model = model_classes[model_name]().to(device)

    trainer_path = Path(experiment_dir) / f"{model_name}_trainer.pth"
    model_path = Path(experiment_dir) / f"{model_name}_model.pth"

    if trainer_path.exists():
        checkpoint = torch.load(trainer_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"Loaded trainer checkpoint: {trainer_path}")
    elif model_path.exists():
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded model weights: {model_path}")
    else:
        raise FileNotFoundError(f"Cannot find weights for {model_name} in {experiment_dir}")

    model.eval()
    return model


def get_test_dataset():
    dataset = PreEchoNetDatasetMerged(
        split="test",
        transform=ModerateTransform(is_training=False),
        max_samples=cfg.TEST_SAMPLES,
        data_seed=cfg.DATA_SEED
    )

    print(f"Loaded test dataset: {len(dataset)} samples")
    return dataset


def get_sample(dataset, sample_idx):
    sample = dataset[sample_idx]

    video_tensor = sample["video"]
    true_ef = sample["EF"]
    filename = sample["filename"]

    true_ef = true_ef.item() if torch.is_tensor(true_ef) else float(true_ef)
    video_tensor = video_tensor.unsqueeze(0).to(device)

    return video_tensor, true_ef, filename


# ============================================================
# Figure 1: Single-case three-model comparison
# ============================================================
def save_single_case_comparison(results, video_tensor, true_ef, filename, output_path, frame_idx=0):
    n_models = len(results)

    fig, axes = plt.subplots(n_models, 3, figsize=(15, 5 * n_models))

    if n_models == 1:
        axes = np.expand_dims(axes, axis=0)

    frame_np = tensor_frame_to_uint8(video_tensor, frame_idx)

    for row, result in enumerate(results):
        model_name = result["model_name"]
        pred_ef = result["pred_ef"]
        heatmap = result["heatmap_2d"]

        overlay, heatmap_resized = overlay_heatmap(frame_np, heatmap)

        axes[row, 0].imshow(frame_np)
        axes[row, 0].set_title(f"{model_name}\nOriginal")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(heatmap_resized, cmap="jet")
        axes[row, 1].set_title("Grad-CAM Heatmap")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(f"Overlay\nTrue EF: {true_ef:.2f}% | Pred EF: {pred_ef:.2f}%")
        axes[row, 2].axis("off")

    plt.suptitle(
        f"Single-case Grad-CAM Comparison\nSample: {filename}",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved comparison figure: {output_path}")


# ============================================================
# Figure 2: Multi-frame figure
# ============================================================
def save_multiframe_gradcam(model_name, video_tensor, heatmap_2d, true_ef, pred_ef, filename, output_path):
    frame_indices = [0, 8, 16, 24, 32, 40, 48, 56]
    video_denorm = denormalize_video(video_tensor.squeeze(0))
    T = video_denorm.shape[0]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for ax, idx in zip(axes.flatten(), frame_indices):
        idx = min(idx, T - 1)

        frame = video_denorm[idx]
        frame_np = frame.permute(1, 2, 0).numpy()
        frame_np = np.uint8(frame_np * 255)

        overlay, _ = overlay_heatmap(frame_np, heatmap_2d)

        ax.imshow(overlay)
        ax.set_title(f"Frame {idx}")
        ax.axis("off")

    plt.suptitle(
        f"{model_name} Multi-frame Grad-CAM\nTrue EF: {true_ef:.2f}% | Pred EF: {pred_ef:.2f}% | {filename}",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved multi-frame figure: {output_path}")


# ============================================================
# Figure 3: Grad-CAM GIF
# ============================================================
def save_gradcam_gif(model_name, video_tensor, heatmap_2d, output_path):
    video_denorm = denormalize_video(video_tensor.squeeze(0))
    frames = []

    for idx in range(video_denorm.shape[0]):
        frame = video_denorm[idx]
        frame_np = frame.permute(1, 2, 0).numpy()
        frame_np = np.uint8(frame_np * 255)

        overlay, _ = overlay_heatmap(frame_np, heatmap_2d)
        frames.append(overlay)

    imageio.mimsave(output_path, frames, fps=8)
    print(f"Saved Grad-CAM GIF: {output_path}")


# ============================================================
# Figure 4: Population Mean CAM
# ============================================================
def save_population_mean_cam(model_name, cams, output_path):
    mean_cam = np.mean(np.stack(cams, axis=0), axis=0)
    mean_cam = normalize_cam(mean_cam)

    plt.figure(figsize=(6, 6))
    plt.imshow(mean_cam, cmap="jet")
    plt.title(f"{model_name} Population-level Mean Grad-CAM")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved population mean CAM: {output_path}")


# ============================================================
# Main
# ============================================================
def main():
    # -----------------------------
    # modifiable parametes
    # -----------------------------
    sample_idx = 0
    frame_idx = 0

    population_sample_count = 20

    output_dir = Path("gradcam_advanced_results")
    output_dir.mkdir(exist_ok=True)

    experiment_dir = find_latest_experiment_dir()
    dataset = get_test_dataset()

    models_to_analyze = [
        "R3D-18",
        "EfficientNet-3D",
        "Attention-LSTM"
    ]

    target_layers = {
        "R3D-18": "backbone.layer4.1.conv2",
        "EfficientNet-3D": "conv3d_layers.7",
        "Attention-LSTM": "cnn.7.1.conv2"
    }

    summary_rows = []

    # ========================================================
    # Part 1: Single-case comparison
    # ========================================================
    video_tensor, true_ef, filename = get_sample(dataset, sample_idx)

    single_case_results = []

    for model_name in models_to_analyze:
        print("\n" + "=" * 60)
        print(f"Single-case Grad-CAM: {model_name}")
        print("=" * 60)

        model = load_model(model_name, experiment_dir)
        gradcam = GradCAM(model, target_layers[model_name])

        heatmap_2d, heatmap_3d, pred_ef = gradcam.generate(video_tensor)
        gradcam.remove_hooks()

        single_case_results.append({
            "model_name": model_name,
            "heatmap_2d": heatmap_2d,
            "heatmap_3d": heatmap_3d,
            "pred_ef": pred_ef
        })

        summary_rows.append({
            "sample_idx": sample_idx,
            "filename": filename,
            "model": model_name,
            "true_EF": true_ef,
            "pred_EF": pred_ef,
            "abs_error": abs(pred_ef - true_ef)
        })

        save_multiframe_gradcam(
            model_name=model_name,
            video_tensor=video_tensor,
            heatmap_2d=heatmap_2d,
            true_ef=true_ef,
            pred_ef=pred_ef,
            filename=filename,
            output_path=output_dir / f"{model_name}_sample{sample_idx}_multiframe.png"
        )

        save_gradcam_gif(
            model_name=model_name,
            video_tensor=video_tensor,
            heatmap_2d=heatmap_2d,
            output_path=output_dir / f"{model_name}_sample{sample_idx}_gradcam.gif"
        )

    save_single_case_comparison(
        results=single_case_results,
        video_tensor=video_tensor,
        true_ef=true_ef,
        filename=filename,
        output_path=output_dir / f"single_case_model_comparison_sample{sample_idx}.png",
        frame_idx=frame_idx
    )

    # ========================================================
    # Part 2: Population-level Mean CAM
    # ========================================================
    population_indices = list(range(min(population_sample_count, len(dataset))))

    for model_name in models_to_analyze:
        print("\n" + "=" * 60)
        print(f"Population Mean CAM: {model_name}")
        print("=" * 60)

        model = load_model(model_name, experiment_dir)
        gradcam = GradCAM(model, target_layers[model_name])

        cams = []

        for idx in population_indices:
            try:
                video_tensor_i, true_ef_i, filename_i = get_sample(dataset, idx)

                heatmap_2d, heatmap_3d, pred_ef_i = gradcam.generate(video_tensor_i)
                cams.append(heatmap_2d)

                summary_rows.append({
                    "sample_idx": idx,
                    "filename": filename_i,
                    "model": model_name,
                    "true_EF": true_ef_i,
                    "pred_EF": pred_ef_i,
                    "abs_error": abs(pred_ef_i - true_ef_i)
                })

                print(f"[{model_name}] sample {idx}: true={true_ef_i:.2f}, pred={pred_ef_i:.2f}")

            except Exception as e:
                print(f"Skipped sample {idx} for {model_name}: {e}")

        gradcam.remove_hooks()

        if len(cams) > 0:
            save_population_mean_cam(
                model_name=model_name,
                cams=cams,
                output_path=output_dir / f"{model_name}_population_mean_cam.png"
            )

    # ========================================================
    # Save CSV
    # ========================================================
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = output_dir / "gradcam_prediction_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    print("\nAdvanced Grad-CAM analysis completed.")
    print(f"Results saved to: {output_dir.resolve()}")
    print(f"Summary CSV saved to: {summary_csv_path}")


if __name__ == "__main__":
    main()