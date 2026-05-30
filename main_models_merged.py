import torch
import torch.nn as nn
# Import all models from the pre-experiment module
from pre_models_merged import (
    SimpleR3D18,
    SimpleCNNLSTM,
    SimpleCNNTransformer,
    EfficientNet3D,
    AttentionLSTM,
    GraphConvNet
)