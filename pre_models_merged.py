import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# ============ First ============
class SimpleR3D18(nn.Module):
    """3D ResNet-18"""
    def __init__(self, num_classes=1):
        super(SimpleR3D18, self).__init__()
        self.backbone = models.video.r3d_18(pretrained=True)
        
        for param in list(self.backbone.parameters())[:-30]:
            param.requires_grad = False
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        output = self.backbone(x)
        return output * 100.0

class SimpleCNNLSTM(nn.Module):
    def __init__(self, num_classes=1, hidden_size=128, num_layers=1):
        super(SimpleCNNLSTM, self).__init__()
        cnn_backbone = models.resnet18(pretrained=True)
        for param in list(cnn_backbone.parameters())[:-20]:
            param.requires_grad = False
        self.cnn = nn.Sequential(*list(cnn_backbone.children())[:-1])
        # Bi-GRU
        self.gru = nn.GRU(
            input_size=512, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=0.2 if num_layers>1 else 0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes), nn.Sigmoid()
        )
    def forward(self, x):
        batch_size, num_frames, C, H, W = x.shape
        cnn_features = []
        for t in range(num_frames):
            frame = x[:, t, :, :, :]
            features = self.cnn(frame).view(batch_size, -1)
            cnn_features.append(features)
        cnn_features = torch.stack(cnn_features, dim=1)
        gru_out, _ = self.gru(cnn_features)
        last_output = gru_out[:, -1, :]
        return self.classifier(last_output) * 100.0

class SimpleCNNTransformer(nn.Module):
    """CNN + Transformer"""
    def __init__(self, num_classes=1, d_model=512, nhead=8, num_layers=2):
        super(SimpleCNNTransformer, self).__init__()
        
        cnn_backbone = models.resnet18(pretrained=True)
        for param in list(cnn_backbone.parameters())[:-20]:
            param.requires_grad = False
        
        self.cnn = nn.Sequential(*list(cnn_backbone.children())[:-1])
        
        self.feature_proj = nn.Linear(512, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, d_model))
    
    def forward(self, x):
        batch_size, num_frames, C, H, W = x.shape
        
        cnn_features = []
        for t in range(num_frames):
            frame = x[:, t, :, :, :]
            features = self.cnn(frame)
            features = features.view(batch_size, -1)
            features = self.feature_proj(features)
            cnn_features.append(features)
        
        features = torch.stack(cnn_features, dim=1)
        features = features + self.positional_encoding[:, :num_frames, :]
        transformer_out = self.transformer_encoder(features)
        last_output = transformer_out[:, -1, :]
        output = self.classifier(last_output)
        return output * 100.0

# ============ Second ============
class EfficientNet3D(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientNet3D, self).__init__()
        self.backbone_2d = models.efficientnet_b0(pretrained=True)
        self.features = self.backbone_2d.features
        # Reduce 3D convolutional channels: 1280→256→128→64
        self.conv3d_layers = nn.Sequential(
            nn.Conv3d(1280, 256, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True), nn.Dropout3d(0.3),
            nn.Conv3d(256, 128, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)), nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32, num_classes), nn.Sigmoid()
        )
        self._freeze_layers()
    def _freeze_layers(self):
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.features[6:].parameters():
            param.requires_grad = True
        for param in self.features[5][-2:].parameters():
            param.requires_grad = True
    def forward(self, x):
        batch_size, num_frames, C, H, W = x.shape
        frame_features = []
        for t in range(num_frames):
            frame = x[:, t, :, :, :]
            features = self.features(frame)
            frame_features.append(features.unsqueeze(2))
        temporal_features = torch.cat(frame_features, dim=2)
        features_3d = self.conv3d_layers(temporal_features)
        output = self.classifier(features_3d)
        return output * 100.0

class AttentionLSTM(nn.Module):
    """Attention LSTM"""
    def __init__(self, num_classes=1, hidden_size=128, num_layers=2):
        super(AttentionLSTM, self).__init__()
        
        cnn_backbone = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(cnn_backbone.children())[:-1])
        
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )
        
        self._freeze_layers()
    
    def _freeze_layers(self):
        for param in list(self.cnn.parameters())[:30]:
            param.requires_grad = False
    
    def forward(self, x):
        batch_size, num_frames, C, H, W = x.shape
        
        cnn_features = []
        for t in range(num_frames):
            frame = x[:, t, :, :, :]
            features = self.cnn(frame)
            features = features.view(batch_size, -1)
            cnn_features.append(features)
        
        cnn_features = torch.stack(cnn_features, dim=1)
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)
        
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        weighted_features = torch.sum(lstm_out * attention_weights, dim=1)
        output = self.classifier(weighted_features)
        return output * 100.0

class GraphConvNet(nn.Module):
    """Graph convolutional networks are used for modeling cardiac structures"""
    def __init__(self, num_classes=1):
        super(GraphConvNet, self).__init__()
        
        cnn_backbone = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(cnn_backbone.children())[:-2])
        
        self.graph_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
            nn.Sigmoid()
        )
        
        self._freeze_layers()
    
    def _freeze_layers(self):
        for param in list(self.cnn.parameters())[:40]:
            param.requires_grad = False
    
    def forward(self, x):
        batch_size, num_frames, C, H, W = x.shape
        
        frame_features = []
        for t in range(num_frames):
            frame = x[:, t, :, :, :]
            features = self.cnn(frame)
            features = self.graph_conv(features)
            features = self.global_pool(features)
            frame_features.append(features.unsqueeze(2))
        
        temporal_features = torch.cat(frame_features, dim=2)
        pooled_features = self.temporal_pool(temporal_features)
        output = self.classifier(pooled_features)
        return output * 100.0