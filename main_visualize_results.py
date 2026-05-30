import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

warnings.filterwarnings("ignore")

# -------------------- Import project modules --------------------
sys.path.append(str(Path(__file__).parent))

from main_config_merged import main_config
from main_dataloader_merged import get_main_dataloaders
from pre_models_merged import (
    SimpleR3D18,
    SimpleCNNLSTM,
    SimpleCNNTransformer,
    EfficientNet3D,
    AttentionLSTM,
    GraphConvNet
)

# -------------------- Global Plot Style --------------------
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (14, 8),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3
})


class MainExperimentVisualizer:
    def __init__(self, experiment_dir=None):
        self.experiment_dir = self._locate_experiment_dir(experiment_dir)
        self.results = None
        self.time_col = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_groups = {
            "3D Models": ["R3D-18", "EfficientNet-3D", "Graph-CNN"],
            "Temporal Models": ["CNN-LSTM", "CNN-Transformer", "Attention-LSTM"]
        }

        self.model_classes = {
            "R3D-18": SimpleR3D18,
            "CNN-LSTM": SimpleCNNLSTM,
            "CNN-Transformer": SimpleCNNTransformer,
            "EfficientNet-3D": EfficientNet3D,
            "Attention-LSTM": AttentionLSTM,
            "Graph-CNN": GraphConvNet
        }

        self.best_model_name = None
        self.best_y_true = None
        self.best_y_pred = None

        self._load_results()
        print(f"Loaded experiment directory: {self.experiment_dir}")

    def _locate_experiment_dir(self, experiment_dir):
        if experiment_dir is not None:
            return Path(experiment_dir)

        default_path = Path("saved_models") / "main_experiment_20260420_164520"
        if default_path.exists():
            return default_path

        saved_models = Path("saved_models")
        if not saved_models.exists():
            raise FileNotFoundError(
                "saved_models directory not found. Please specify the correct experiment directory."
            )

        main_experiments = list(saved_models.glob("main_experiment_*"))
        if not main_experiments:
            raise FileNotFoundError(
                "No main_experiment_* directory found. Please specify the experiment directory."
            )

        latest = sorted(main_experiments, key=lambda p: p.stat().st_mtime)[-1]
        print(f"Automatically selected latest experiment directory: {latest}")
        return latest

    def _load_results(self):
        csv_path = self.experiment_dir / "results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"results.csv not found: {csv_path}")

        self.results = pd.read_csv(csv_path)
        self.results.columns = self.results.columns.str.strip()

        possible_time_cols = [
            "Training_Time",
            "Training_Time(s)",
            "Training Time",
            "Training Time(s)",
            "Train_Time",
            "Train_Time(s)",
            "train_time"
        ]

        self.time_col = None
        for col in possible_time_cols:
            if col in self.results.columns:
                self.time_col = col
                break

        if self.time_col is None:
            raise ValueError(
                f"Training time column not found.\n"
                f"Current columns: {list(self.results.columns)}"
            )

        required_columns = ["Model", "MAE", "RMSE", "Pearson_R", "R2"]
        missing_columns = [col for col in required_columns if col not in self.results.columns]
        if missing_columns:
            raise ValueError(
                f"results.csv is missing required columns: {missing_columns}\n"
                f"Current columns: {list(self.results.columns)}"
            )

        best_row = self.results.sort_values("MAE", ascending=True).iloc[0]
        self.best_model_name = best_row["Model"]

        print("Detected columns:", list(self.results.columns))
        print(f"Training time column detected as: {self.time_col}")
        print(f"Best model selected by MAE: {self.best_model_name}")
        print(f"Loaded results file with {len(self.results)} model(s)")

    def _highlight_best_bar(self, bars, values, higher_is_better=True):
        if len(values) == 0:
            return
        best_idx = np.argmax(values) if higher_is_better else np.argmin(values)
        bars[best_idx].set_edgecolor("black")
        bars[best_idx].set_linewidth(2.2)
        bars[best_idx].set_alpha(1.0)

    def _load_best_model_predictions(self):
        if self.best_y_true is not None and self.best_y_pred is not None:
            return self.best_y_true, self.best_y_pred

        if self.best_model_name not in self.model_classes:
            raise ValueError(f"Unknown model name: {self.best_model_name}")

        model_class = self.model_classes[self.best_model_name]
        model = model_class().to(self.device)

        trainer_path = self.experiment_dir / f"{self.best_model_name}_trainer.pth"
        model_path = self.experiment_dir / f"{self.best_model_name}_model.pth"

        checkpoint_loaded = False

        if trainer_path.exists():
            try:
                checkpoint = torch.load(trainer_path, map_location=self.device)
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                    checkpoint_loaded = True
            except Exception as e:
                print(f"Warning: failed to load trainer checkpoint for {self.best_model_name}: {e}")

        if not checkpoint_loaded and model_path.exists():
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict, strict=False)
                checkpoint_loaded = True
            except Exception as e:
                print(f"Warning: failed to load model weights for {self.best_model_name}: {e}")

        if not checkpoint_loaded:
            raise FileNotFoundError(
                f"Could not load saved weights for {self.best_model_name} from:\n"
                f"  {trainer_path}\n"
                f"  {model_path}"
            )

        model.eval()
        _, _, test_loader = get_main_dataloaders(data_seed=main_config.DATA_SEED)

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                videos = batch["video"].to(self.device)
                ef_labels = batch["EF"].to(self.device).float().unsqueeze(1)
                outputs = model(videos)

                all_preds.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(ef_labels.cpu().numpy().flatten())

        self.best_y_pred = np.array(all_preds)
        self.best_y_true = np.array(all_targets)

        pred_df = pd.DataFrame({
            "True_EF": self.best_y_true,
            "Predicted_EF": self.best_y_pred,
            "Residual": self.best_y_pred - self.best_y_true
        })
        pred_df.to_csv(self.experiment_dir / "best_model_test_predictions.csv", index=False)
        print(f"Saved best model predictions to: {self.experiment_dir / 'best_model_test_predictions.csv'}")

        return self.best_y_true, self.best_y_pred

    def create_all_visualizations(self, output_dir=None):
        if output_dir is None:
            output_dir = self.experiment_dir / "visualizations"
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        print(f"\nFigures will be saved to: {output_dir}\n")

        self._load_best_model_predictions()

        self.plot_model_performance_comparison(output_dir)
        self.plot_training_time_comparison(output_dir)
        self.plot_real_prediction_vs_ground_truth(output_dir)
        self.plot_residual_distribution(output_dir)
        self.plot_correlation_metrics(output_dir)
        self.plot_model_group_comparison(output_dir)
        self.plot_training_validation_curves(output_dir)
        self.plot_bland_altman(output_dir)

        print("\nAll 8 final figures have been generated successfully!")

    # -------------------- Figure 1 --------------------
    def plot_model_performance_comparison(self, output_dir):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Model Performance Comparison Across Metrics", fontsize=18, fontweight="bold")

        metrics = ["MAE", "RMSE", "Pearson_R", "R2"]
        titles = [
            "Mean Absolute Error (MAE)",
            "Root Mean Square Error (RMSE)",
            "Pearson Correlation (R)",
            "R-squared Score (R²)"
        ]
        ylabels = [
            "MAE (Lower is Better)",
            "RMSE (Lower is Better)",
            "Correlation (Higher is Better)",
            "R² (Higher is Better)"
        ]

        for idx, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
            ax = axes[idx // 2, idx % 2]
            ascending = metric in ["MAE", "RMSE"]
            sorted_df = self.results.sort_values(by=metric, ascending=ascending).reset_index(drop=True)

            bars = ax.bar(
                sorted_df["Model"],
                sorted_df[metric],
                color=plt.cm.Set3(np.linspace(0, 1, len(sorted_df))),
                alpha=0.9
            )

            self._highlight_best_bar(
                bars,
                sorted_df[metric].values,
                higher_is_better=not ascending
            )

            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlabel("Model")
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=35)

            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9
                )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_dir / "01_model_performance_comparison.png")
        plt.close()
        print("  [1/8] Model performance comparison figure generated")

    # -------------------- Figure 2 --------------------
    def plot_training_time_comparison(self, output_dir):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("Training Time Analysis", fontsize=18, fontweight="bold")

        sorted_df = self.results.sort_values(self.time_col).reset_index(drop=True)

        bars = ax1.bar(
            sorted_df["Model"],
            sorted_df[self.time_col],
            color=plt.cm.viridis(np.linspace(0.25, 0.9, len(sorted_df))),
            alpha=0.9
        )

        self._highlight_best_bar(
            bars,
            sorted_df[self.time_col].values,
            higher_is_better=False
        )

        ax1.set_title("Training Time by Model", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Model")
        ax1.set_ylabel("Time (seconds)")
        ax1.tick_params(axis="x", rotation=35)

        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

        ax2.scatter(
            self.results[self.time_col],
            self.results["MAE"],
            s=180,
            alpha=0.8,
            edgecolors="black",
            linewidth=1
        )

        for _, row in self.results.iterrows():
            ax2.annotate(
                row["Model"],
                (row[self.time_col], row["MAE"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9
            )

        ax2.set_title("Training Time vs MAE", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Training Time (seconds)")
        ax2.set_ylabel("MAE")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_dir / "02_training_time_comparison.png")
        plt.close()
        print("  [2/8] Training time analysis figure generated")

    # -------------------- Figure 3 --------------------
    def plot_real_prediction_vs_ground_truth(self, output_dir):
        y_true, y_pred = self.best_y_true, self.best_y_pred

        fig, ax = plt.subplots(figsize=(8, 8))
        fig.suptitle(
            f"Real Prediction vs Ground Truth ({self.best_model_name})",
            fontsize=18,
            fontweight="bold"
        )

        ax.scatter(y_true, y_pred, alpha=0.7, s=40)

        min_val = min(y_true.min(), y_pred.min()) - 2
        max_val = max(y_true.max(), y_pred.max()) + 2

        ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.5, label="Ideal Line")

        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        x_sorted = np.sort(y_true)
        ax.plot(x_sorted, p(x_sorted), lw=2, label="Best Fit")

        corr = np.corrcoef(y_true, y_pred)[0, 1]
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

        ax.set_title(f"Pearson R = {corr:.3f}, R² = {r2:.3f}", fontsize=13, fontweight="bold")
        ax.set_xlabel("True EF (%)")
        ax.set_ylabel("Predicted EF (%)")
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_dir / "03_real_prediction_vs_ground_truth.png")
        plt.close()
        print("  [3/8] Real prediction vs ground truth figure generated")

    # -------------------- Figure 4 --------------------
    def plot_residual_distribution(self, output_dir):
        y_true, y_pred = self.best_y_true, self.best_y_pred
        residuals = y_pred - y_true

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(
            f"Residual Analysis ({self.best_model_name})",
            fontsize=18,
            fontweight="bold"
        )

        ax1.hist(residuals, bins=20, edgecolor="black", alpha=0.8)
        ax1.axvline(np.mean(residuals), color="red", linestyle="--", linewidth=1.5, label="Mean Residual")
        ax1.set_title("Residual Distribution", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Residual (Predicted - True)")
        ax1.set_ylabel("Frequency")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.scatter(y_pred, residuals, alpha=0.7, s=40, edgecolors="black", linewidth=0.5)
        ax2.axhline(0, color="red", linestyle="--", linewidth=1.5)
        ax2.set_title("Residuals vs Predicted Values", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Predicted EF (%)")
        ax2.set_ylabel("Residual (Predicted - True)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_dir / "04_residual_distribution.png")
        plt.close()
        print("  [4/8] Residual distribution figure generated")

    # -------------------- Figure 5 --------------------
    def plot_correlation_metrics(self, output_dir):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("Correlation Metrics Analysis", fontsize=18, fontweight="bold")

        x = np.arange(len(self.results))
        width = 0.36

        bars1 = ax1.bar(
            x - width / 2,
            self.results["Pearson_R"],
            width,
            label="Pearson R",
            color="lightgreen",
            edgecolor="black",
            alpha=0.9
        )
        bars2 = ax1.bar(
            x + width / 2,
            self.results["R2"],
            width,
            label="R²",
            color="gold",
            edgecolor="black",
            alpha=0.9
        )

        ax1.set_title("Correlation Metrics by Model", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Model")
        ax1.set_ylabel("Value")
        ax1.set_ylim(0, 1.1)
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.results["Model"], rotation=35, ha="right")
        ax1.axhline(y=1.0, color="red", linestyle="--", alpha=0.4)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

        metrics = ["MAE", "RMSE", "Pearson_R", "R2", self.time_col]
        corr = self.results[metrics].corr(numeric_only=True)

        im = ax2.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        for i in range(len(corr)):
            for j in range(len(corr)):
                text_color = "white" if abs(corr.iloc[i, j]) > 0.5 else "black"
                ax2.text(
                    j, i, f"{corr.iloc[i, j]:.2f}",
                    ha="center", va="center",
                    color=text_color, fontsize=9
                )

        display_labels = ["MAE", "RMSE", "Pearson_R", "R2", "Training_Time"]
        ax2.set_xticks(range(len(metrics)))
        ax2.set_yticks(range(len(metrics)))
        ax2.set_xticklabels(display_labels, rotation=35, ha="right")
        ax2.set_yticklabels(display_labels)
        ax2.set_title("Metric Correlation Matrix", fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax2, shrink=0.85)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_dir / "05_correlation_metrics.png")
        plt.close()
        print("  [5/8] Correlation metrics figure generated")

    # -------------------- Figure 6 --------------------
    def plot_model_group_comparison(self, output_dir):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Model Architecture Group Comparison", fontsize=18, fontweight="bold")

        metrics = ["MAE", "Pearson_R", "R2", self.time_col]
        titles = [
            "Mean Absolute Error",
            "Pearson Correlation",
            "R-squared Score",
            "Training Time"
        ]
        colors = ["skyblue", "lightcoral"]

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            group_data = []
            group_labels = []

            for group_name, models in self.model_groups.items():
                group_vals = self.results[self.results["Model"].isin(models)][metric].values
                if len(group_vals) > 0:
                    group_data.append(group_vals)
                    group_labels.append(group_name)

            if len(group_data) == 0:
                ax.text(
                    0.5, 0.5,
                    "No matching grouped model data found",
                    ha="center", va="center",
                    transform=ax.transAxes
                )
                ax.set_title(title)
                continue

            bp = ax.boxplot(group_data, labels=group_labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors[:len(group_data)]):
                patch.set_facecolor(color)

            for i, data in enumerate(group_data):
                x_jitter = np.random.normal(i + 1, 0.04, size=len(data))
                ax.scatter(x_jitter, data, alpha=0.7, s=50, edgecolors="black", linewidth=0.5)

            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("Model Group")
            ax.set_ylabel("Time (s)" if metric == self.time_col else metric)
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_dir / "06_model_group_comparison.png")
        plt.close()
        print("  [6/8] Model group comparison figure generated")

    # -------------------- Figure 7 --------------------
    def plot_training_validation_curves(self, output_dir):
        trainer_files = list(self.experiment_dir.glob("*_trainer.pth"))
        if not trainer_files:
            print("  [7/8] Warning: no trainer files found, skipping loss curves")
            return

        model_names = [f.stem.replace("_trainer", "") for f in trainer_files]
        valid_models = [m for m in self.results["Model"] if m in model_names]

        if not valid_models:
            print("  [7/8] Warning: no matching trainer files for models in results.csv")
            return

        n_models = len(valid_models)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        fig.suptitle("Training and Validation Loss Curves", fontsize=18, fontweight="bold")

        axes = np.array(axes).reshape(-1)

        for idx, model_name in enumerate(valid_models):
            ax = axes[idx]
            trainer_path = self.experiment_dir / f"{model_name}_trainer.pth"

            if not trainer_path.exists():
                ax.text(
                    0.5, 0.5,
                    f"No trainer data for {model_name}",
                    ha="center", va="center",
                    transform=ax.transAxes
                )
                ax.set_title(model_name)
                continue

            try:
                checkpoint = torch.load(trainer_path, map_location="cpu")
                train_losses = checkpoint.get("train_losses", [])
                val_losses = checkpoint.get("val_losses", [])
            except Exception as e:
                ax.text(
                    0.5, 0.5,
                    f"Error loading data:\n{str(e)}",
                    ha="center", va="center",
                    transform=ax.transAxes
                )
                ax.set_title(model_name)
                continue

            if len(train_losses) == 0:
                ax.text(
                    0.5, 0.5,
                    f"No loss data for {model_name}",
                    ha="center", va="center",
                    transform=ax.transAxes
                )
                ax.set_title(model_name)
                continue

            epochs = range(1, len(train_losses) + 1)
            ax.plot(epochs, train_losses, label="Train Loss", lw=2)
            if len(val_losses) > 0:
                ax.plot(epochs, val_losses, label="Validation Loss", lw=2)

            ax.set_title(model_name, fontsize=12, fontweight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

        for j in range(idx + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_dir / "07_training_validation_curves.png")
        plt.close()
        print("  [7/8] Training and validation loss curves generated")

    # -------------------- Figure 8 --------------------
    def plot_bland_altman(self, output_dir):
        y_true, y_pred = self.best_y_true, self.best_y_pred

        means = (y_true + y_pred) / 2
        diffs = y_pred - y_true

        bias = np.mean(diffs)
        sd = np.std(diffs, ddof=1)
        loa_upper = bias + 1.96 * sd
        loa_lower = bias - 1.96 * sd

        fig, ax = plt.subplots(figsize=(9, 7))
        fig.suptitle(
            f"Bland-Altman Plot ({self.best_model_name})",
            fontsize=18,
            fontweight="bold"
        )

        ax.scatter(means, diffs, alpha=0.7, s=40)
        ax.axhline(bias, color="red", linestyle="--", linewidth=1.5, label=f"Bias = {bias:.2f}")
        ax.axhline(loa_upper, color="gray", linestyle="--", linewidth=1.5, label=f"+1.96 SD = {loa_upper:.2f}")
        ax.axhline(loa_lower, color="gray", linestyle="--", linewidth=1.5, label=f"-1.96 SD = {loa_lower:.2f}")

        ax.set_title("Agreement Between Predicted and True EF", fontsize=14, fontweight="bold")
        ax.set_xlabel("Mean of True and Predicted EF (%)")
        ax.set_ylabel("Difference (Predicted - True)")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(output_dir / "08_bland_altman_plot.png")
        plt.close()
        print("  [8/8] Bland-Altman plot generated")


def main():
    parser = argparse.ArgumentParser(description="Final paper-ready visualization for the main experiment")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="Experiment directory path; if omitted, use default path or auto-detect latest"
    )
    args = parser.parse_args()

    try:
        visualizer = MainExperimentVisualizer(args.experiment_dir)
        visualizer.create_all_visualizations()
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()