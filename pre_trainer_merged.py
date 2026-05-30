import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from pre_config_merged import pre_config_merged

class PreTrainerMerged:
    def __init__(self, model, train_loader, val_loader, test_loader, device, model_name):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.model_name = model_name
        
        self.criterion = nn.L1Loss()
        
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=pre_config_merged.PRE_LEARNING_RATE,
            weight_decay=1e-4
        )
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self):
        """Complete one epoch of training"""
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Training {self.model_name}')
        for batch_idx, batch in enumerate(pbar):
            videos = batch['video'].to(self.device)
            ef_labels = batch['EF'].to(self.device).float().unsqueeze(1)
            
            if batch_idx == 0:
                print(f"Input data dimension: {videos.shape}")
                print(f"Target EF range: [{ef_labels.min().item():.1f}, {ef_labels.max().item():.1f}]")
            
            self.optimizer.zero_grad()
            outputs = self.model(videos)
            
            if batch_idx == 0:
                print(f"Model output range: [{outputs.min().item():.1f}, {outputs.max().item():.1f}]")
            
            loss = self.criterion(outputs, ef_labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(self.train_loader)
        self.train_losses.append(epoch_loss)
        return epoch_loss
    
    def validate(self):
        """validation model"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                videos = batch['video'].to(self.device)
                ef_labels = batch['EF'].to(self.device).float().unsqueeze(1)
                
                outputs = self.model(videos)
                loss = self.criterion(outputs, ef_labels)
                
                running_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(ef_labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        self.val_losses.append(epoch_loss)
        
        return epoch_loss, np.array(all_preds), np.array(all_targets)
    
    def evaluate(self, loader):
        """Evaluate model performance"""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in loader:
                videos = batch['video'].to(self.device)
                ef_labels = batch['EF'].to(self.device).float().unsqueeze(1)
                
                outputs = self.model(videos)
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(ef_labels.cpu().numpy())
        
        all_preds = np.array(all_preds).flatten()
        all_targets = np.array(all_targets).flatten()
        
        mae = mean_absolute_error(all_targets, all_preds)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        r_value, _ = pearsonr(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'Pearson_R': r_value,
            'R2': r2
        }
    
    def train(self, epochs=pre_config_merged.PRE_EPOCHS):
        """Training Process"""
        print(f"Begin training {self.model_name}...")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            train_loss = self.train_epoch()
            val_loss, val_preds, val_targets = self.validate()
            
            val_mae = mean_absolute_error(val_targets, val_preds)
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
        
        print(f"{self.model_name} Training completed!")
    
    def test(self):
        """Evaluate on the test set"""
        print(f"\nTest {self.model_name}...")
        test_results = self.evaluate(self.test_loader)
        
        print(f"Result - MAE: {test_results['MAE']:.4f}, "
              f"RMSE: {test_results['RMSE']:.4f}, "
              f"Pearson R: {test_results['Pearson_R']:.4f}")
        
        return test_results