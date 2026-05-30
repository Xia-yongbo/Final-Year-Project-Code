import torch
import pandas as pd
import time
import os
import json
from datetime import datetime
from main_config_merged import main_config
from main_dataloader_merged import get_main_dataloaders   # Note: this function has been defined above
from pre_models_merged import (
    SimpleR3D18, SimpleCNNLSTM, SimpleCNNTransformer,
    EfficientNet3D, AttentionLSTM, GraphConvNet
)
from main_trainer_merged import MainTrainerMerged

class MainExperimentMerged:
    def __init__(self, data_seed=main_config.DATA_SEED, model_seed=main_config.MODEL_SEED):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Main experiment device: {self.device}")
        self.data_seed = data_seed
        self.model_seed = model_seed
        self.experiment_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join("saved_models", f"main_experiment_{self.experiment_time}")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Use the modified data loader (adaptive pooling + moderate augmentation)
        self.train_loader, self.val_loader, self.test_loader = get_main_dataloaders(data_seed)
        
        self.models = {
            'R3D-18': SimpleR3D18,
            'CNN-LSTM': SimpleCNNLSTM,
            'CNN-Transformer': SimpleCNNTransformer,
            'EfficientNet-3D': EfficientNet3D,
            'Attention-LSTM': AttentionLSTM,
            'Graph-CNN': GraphConvNet
        }
        self.trainers = {}
        self.results = {}
    
    def _maybe_load_pretrained(self, model, model_name):
        if not main_config.USE_PRETRAINED:
            return
        pretrained_path = os.path.join(main_config.PRETRAINED_DIR, f"{model_name}_model.pth")
        if os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location=self.device)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights:{pretrained_path}")
        else:
            print(f"Warning: pretrained weights do not exist {pretrained_path}")

    def setup_models(self):
        for name, model_class in self.models.items():
            model = model_class().to(self.device)
            self._maybe_load_pretrained(model, name)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"{name} trainable parameters: {trainable_params:,}")
            self.trainers[name] = MainTrainerMerged(
                model=model, train_loader=self.train_loader, val_loader=self.val_loader,
                test_loader=self.test_loader, device=self.device, model_name=name
            )
    
    def train_all_models(self, epochs=main_config.EPOCHS):
        print("Start training all models for the main experiment...")
        for name, trainer in self.trainers.items():
            print(f"\n{'='*60}\nTraining model: {name}\n{'='*60}")
            torch.manual_seed(self.model_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.model_seed)
            start_time = time.time()
            trainer.train(epochs=epochs)
            training_time = time.time() - start_time
            test_results = trainer.test()
            test_results['Training_Time'] = training_time
            self.results[name] = test_results
            print(f"{name} training completed, time used: {training_time:.2f} seconds")
    
    def save_models(self):
        print(f"\nSaving models to {self.save_dir}")
        info = {
            'experiment_time': self.experiment_time,
            'data_seed': self.data_seed,
            'model_seed': self.model_seed,
            'config': {k: getattr(main_config, k) for k in dir(main_config) if not k.startswith('_')},
            'results': self.results
        }
        with open(os.path.join(self.save_dir, 'experiment_info.json'), 'w') as f:
            json.dump(info, f, indent=4, default=str)
        for name, trainer in self.trainers.items():
            model_path = os.path.join(self.save_dir, f"{name}_model.pth")
            torch.save(trainer.model.state_dict(), model_path)
            trainer_path = os.path.join(self.save_dir, f"{name}_trainer.pth")
            torch.save({
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'train_losses': trainer.train_losses,
                'val_losses': trainer.val_losses,
                'best_val_loss': trainer.best_val_loss,
                'model_name': name
            }, trainer_path)
            print(f"Saved {name}")
        df = pd.DataFrame([{'Model': name, **res} for name, res in self.results.items()])
        df.to_csv(os.path.join(self.save_dir, 'results.csv'), index=False)
        print(f"Results saved to {self.save_dir}/results.csv")
    
    def run(self):
        self.setup_models()
        self.train_all_models()
        self.save_models()
        return self.results, self.save_dir

if __name__ == "__main__":
    experiment = MainExperimentMerged()
    results, save_dir = experiment.run()
    print(f"\nMain experiment completed! Results saved in: {save_dir}")