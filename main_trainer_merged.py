import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from main_config_merged import main_config

class SmoothedReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """Verification loss smoothing scheduler with exponential moving average"""
    def __init__(self, optimizer, smoothing=0.9, **kwargs):
        super().__init__(optimizer, **kwargs)
        self.smoothing = smoothing
        self.smoothed_loss = None
    
    def step(self, metrics):
        loss = metrics
        if self.smoothed_loss is None:
            self.smoothed_loss = loss
        else:
            self.smoothed_loss = self.smoothing * self.smoothed_loss + (1 - self.smoothing) * loss
        super().step(self.smoothed_loss)


class MainTrainerMerged:
    def __init__(self, model, train_loader, val_loader, test_loader, device, model_name):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.model_name = model_name
        
        #SmoothL1Loss，beta=1.0
        self.criterion = nn.SmoothL1Loss(beta=1.0)
        
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=main_config.LEARNING_RATE,
            weight_decay=main_config.WEIGHT_DECAY
        )
        
        # Smooth scheduler
        self.scheduler = SmoothedReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=main_config.LR_SCHEDULER_FACTOR,
            patience=main_config.LR_SCHEDULER_PATIENCE,
            smoothing=main_config.LR_SMOOTHING,
            verbose=True
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.early_stop_counter = 0
        self.accumulation_steps = main_config.GRADIENT_ACCUMULATION_STEPS

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f'Training {self.model_name}')
        for i, batch in enumerate(pbar):
            videos = batch['video'].to(self.device)
            ef_labels = batch['EF'].to(self.device).float().unsqueeze(1)
            
            outputs = self.model(videos)
            loss = self.criterion(outputs, ef_labels) / self.accumulation_steps
            loss.backward()
            
            # accumulation gradient
            if (i + 1) % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            running_loss += loss.item() * self.accumulation_steps
            pbar.set_postfix({'Loss': f'{loss.item() * self.accumulation_steps:.4f}'})
        
        # Handle the residual gradient
        if len(self.train_loader) % self.accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return running_loss / len(self.train_loader)

    def validate(self):
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
        return epoch_loss, np.array(all_preds), np.array(all_targets)

    def train(self, epochs=main_config.EPOCHS):
        print(f"Begin training {self.model_name} ...")
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            train_loss = self.train_epoch()
            val_loss, val_preds, val_targets = self.validate()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            val_mae = mean_absolute_error(val_targets, val_preds)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, LR: {current_lr:.2e}")
            
            # Smooth scheduler step
            self.scheduler.step(val_loss)
            
            # Save the best model
            if val_loss < self.best_val_loss - main_config.EARLY_STOPPING_MIN_DELTA:
                self.best_val_loss = val_loss
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.early_stop_counter = 0
                print(f"  -> Save the best model (val_loss={val_loss:.4f})")
            else:
                self.early_stop_counter += 1
                print(f"  -> Early stop counter: {self.early_stop_counter}/{main_config.EARLY_STOPPING_PATIENCE}")
            
            if self.early_stop_counter >= main_config.EARLY_STOPPING_PATIENCE:
                print(f"Early stop trigger, stop training")
                break
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("The best model weights have been loaded")
        print(f"{self.model_name} Training completed!")

    def evaluate(self, loader):
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
        return {'MAE': mae, 'RMSE': rmse, 'Pearson_R': r_value, 'R2': r2}

    def test(self):
        print(f"\nTest {self.model_name}...")
        results = self.evaluate(self.test_loader)
        print(f"MAE: {results['MAE']:.4f}, RMSE: {results['RMSE']:.4f}, Pearson R: {results['Pearson_R']:.4f}, R2: {results['R2']:.4f}")
        return results