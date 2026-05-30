# End-to-End LVEF Prediction from Echocardiography Videos

This repository contains the code for my Final Year Project (FYP), which focuses on automatic prediction of left ventricular ejection fraction (LVEF) from echocardiography videos using deep learning.

The project compares multiple deep learning architectures under a unified experimental framework and includes model performance evaluation, visualization, and Grad-CAM-based interpretability analysis.

---

## Project Overview

Left ventricular ejection fraction (LVEF) is an important clinical indicator for assessing cardiac systolic function. Traditional LVEF measurement usually relies on manual or semi-automated tracing methods, which can be time-consuming and operator-dependent.

This project aims to develop and evaluate deep learning models that can predict LVEF directly from echocardiographic video inputs in an end-to-end manner.

The main model investigated in this project is a 3D convolutional network based on **R3D-18**, which is compared with several alternative architectures.

---

## Models Compared

The following six models are included in the comparison:

- **R3D-18**
- **CNN-LSTM**
- **CNN-Transformer**
- **EfficientNet-3D**
- **Attention-LSTM**
- **Graph-CNN**

All models are trained and evaluated using a consistent experimental pipeline to allow fair comparison.

---

## Main Features

- Echocardiography video preprocessing
- Motion-based video cropping
- Temporal resampling without zero padding
- Unified training and evaluation framework
- Comparison of six deep learning architectures
- Evaluation using MAE, RMSE, Pearson correlation, and R²
- Training time analysis
- Prediction and residual visualization
- Bland-Altman analysis
- Grad-CAM interpretability analysis

---

## Repository Structure

```text
.
├── README.md
├── preprocessing.py
├── main.py
├── main_config_merged.py
├── main_dataloader_merged.py
├── main_models_merged.py
├── main_trainer_merged.py
├── main_experiment_merged.py
├── main_visualize_results.py
├── main_interpret_gradcam_merged.py
├── pre_config_merged.py
├── pre_dataloader_merged.py
├── pre_models_merged.py
├── pre_trainer_merged.py
├── pre_experiment_merged.py
├── pre_run_training_merged.py
├── pre_visualize_results.py
└── pre_interpret_gradcam_merged.py

```

## File Descriptions

### Preprocessing

- `preprocessing.py`  
  Preprocesses the EchoNet-Dynamic video files. It reads raw `.avi` videos, removes static or black-screen segments using motion-based cropping, resizes frames, applies temporal resampling, and saves the processed videos as `.npz` files.

- `main.py`  
  Runs the preprocessing pipeline and checks whether the generated data loaders work correctly.

### Configuration

- `main_config_merged.py`  
  Contains the configuration for the main experiment, including dataset paths, image size, number of frames, batch size, learning rate, random seeds, early stopping settings, and training parameters.

- `pre_config_merged.py`  
  Contains the configuration for the pre-experiment, which uses a smaller subset of the data for testing the training pipeline.

### Data Loading

- `main_dataloader_merged.py`  
  Builds the training, validation, and test data loaders for the main experiment.

- `pre_dataloader_merged.py`  
  Defines the dataset class and data loading process. It supports adaptive temporal pooling, data augmentation, normalization, and loading of processed `.npz` video files.

### Model Definitions

- `main_models_merged.py`  
  Imports the model definitions used in the main experiment.

- `pre_models_merged.py`  
  Defines the six deep learning architectures used in this project: R3D-18, CNN-LSTM, CNN-Transformer, EfficientNet-3D, Attention-LSTM, and Graph-CNN.

### Training

- `main_trainer_merged.py`  
  Defines the main training procedure, including Smooth L1 loss, Adam optimizer, learning rate scheduling, gradient accumulation, gradient clipping, early stopping, validation, and testing.

- `pre_trainer_merged.py`  
  Defines the training procedure for the pre-experiment.

### Experiment Scripts

- `main_experiment_merged.py`  
  Runs the full main experiment on the complete dataset. It trains all six models, evaluates them, and saves model weights, trainer states, and result files.

- `pre_experiment_merged.py`  
  Runs a smaller pre-experiment to test the model training and evaluation pipeline.

- `pre_run_training_merged.py`  
  Runs the pre-experiment and saves trained models, trainer states, configuration information, and results.

### Visualization and Interpretability

- `main_visualize_results.py`  
  Generates final visualization figures, including model performance comparison, training time analysis, predicted vs. ground-truth plots, residual analysis, model group comparison, training/validation loss curves, and Bland-Altman analysis.

- `pre_visualize_results.py`  
  Generates visualization figures for the pre-experiment.

- `main_interpret_gradcam_merged.py`  
  Performs Grad-CAM interpretability analysis for selected models in the main experiment. It generates single-case comparison figures, multi-frame Grad-CAM figures, GIF visualizations, and population-level mean heatmaps.

- `pre_interpret_gradcam_merged.py`  
  Performs Grad-CAM interpretability analysis for the pre-experiment.
```
```

## Usage

### 1. Preprocess the Dataset

- `preprocessing.py`  
  Runs the preprocessing pipeline for the EchoNet-Dynamic dataset. It reads the raw `.avi` videos, applies motion-based cropping, resizes frames, performs temporal resampling, and saves the processed videos as `.npz` files.

### 2. Test the Preprocessing and Data Loader

- `main.py`  
  Runs the preprocessing workflow and checks whether the training, validation, and test data loaders are working correctly. It also prints the shape of an example batch to verify the processed video format.

### 3. Run the Pre-Experiment

- `pre_experiment_merged.py`  
  Runs a smaller pre-experiment using a limited subset of the dataset. This step is mainly used to test whether all six models, the training pipeline, and the evaluation process work correctly.

- `pre_run_training_merged.py`  
  Runs the pre-experiment and saves trained model weights, trainer states, configuration information, and experimental results.

### 4. Run the Main Experiment

- `main_experiment_merged.py`  
  Runs the full main experiment using the complete dataset. It trains and evaluates all six models under the unified experimental framework and saves model weights, trainer states, experiment metadata, and final results.

### 5. Generate Result Visualizations

- `main_visualize_results.py`  
  Generates the final result figures for the main experiment, including model performance comparison, training time analysis, predicted vs. ground-truth plots, residual analysis, model group comparison, training/validation loss curves, and Bland-Altman analysis.

- `pre_visualize_results.py`  
  Generates visualization figures for the pre-experiment results.

### 6. Run Grad-CAM Interpretability Analysis

- `main_interpret_gradcam_merged.py`  
  Performs Grad-CAM interpretability analysis for selected models in the main experiment. It generates single-case comparison figures, multi-frame Grad-CAM visualizations, GIF outputs, and population-level mean heatmaps.

- `pre_interpret_gradcam_merged.py`  
  Performs Grad-CAM interpretability analysis for the pre-experiment.

### 7. Output Directory

- `saved_models/`  
  Stores trained model weights, trainer checkpoints, experiment information, result CSV files, and visualization outputs.

- `gradcam_advanced_results/`  
  Stores Grad-CAM visualization outputs, including heatmaps, overlays, GIFs, and summary CSV files.
  ```
  ```

  ## Environment Requirements

- **Python**: >= 3.8
- **PyTorch**: >= 2.0
- **torchvision**
- **NumPy**
- **pandas**
- **matplotlib**
- **seaborn**
- **scikit-learn**
- **scipy**
- **tqdm**
- **OpenCV** (`opencv-python`)
- **imageio`**
```
```
## Author
- Yongbo Xia
- BSc Biomedical Statistics
- Final Year Project
