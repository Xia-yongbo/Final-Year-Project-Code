import torch
import pandas as pd
import time
import os
import json
from datetime import datetime
from pre_models_merged import (
    SimpleR3D18, SimpleCNNLSTM, SimpleCNNTransformer,
    EfficientNet3D, AttentionLSTM, GraphConvNet
)
from pre_trainer_merged import PreTrainerMerged
from pre_dataloader_merged import get_pre_dataloaders_merged
from pre_config_merged import pre_config_merged

class PreExperimentMerged:
    def __init__(self, data_seed=42, model_seed=123):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device for merged experiment: {self.device}")
        
        # Set random seeds
        self.data_seed = data_seed
        self.model_seed = model_seed
        
        # Create the save directory
        self.experiment_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join("saved_models", f"experiment_{self.experiment_time}")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Load data
        self.train_loader, self.val_loader, self.test_loader = get_pre_dataloaders_merged(data_seed)
        
        # List of all models (6 models)
        self.models = {
            'R3D-18': SimpleR3D18(),
            'CNN-LSTM': SimpleCNNLSTM(),
            'CNN-Transformer': SimpleCNNTransformer(),
            'EfficientNet-3D': EfficientNet3D(),
            'Attention-LSTM': AttentionLSTM(),
            'Graph-CNN': GraphConvNet()
        }
        
        self.trainers = {}
        self.results = {}
    
    def setup_models(self):
        """Initialize all models"""
        for name, model in self.models.items():
            model = model.to(self.device)
            
            # Print the number of trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"{name} trainable parameters: {trainable_params:,}")
            
            self.trainers[name] = PreTrainerMerged(
                model=model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                test_loader=self.test_loader,
                device=self.device,
                model_name=name
            )
    
    def train_all_models(self, epochs=pre_config_merged.PRE_EPOCHS):
        """Train all models"""
        print("Starting training for all models in the merged experiment...")
        
        for name, trainer in self.trainers.items():
            print(f"\n{'='*60}")
            print(f"Training model: {name}")
            print(f"{'='*60}")
            
            start_time = time.time()
            
            # Set the model-specific random seed
            torch.manual_seed(self.model_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.model_seed)
            
            trainer.train(epochs=epochs)
            training_time = time.time() - start_time
            
            # Test
            test_results = trainer.test()
            test_results['Training_Time'] = training_time
            
            self.results[name] = test_results
            
            print(f"{name} training completed, time used: {training_time:.2f} seconds")
    
    def save_models(self):
        """Save all trained models"""
        print(f"\n{'='*60}")
        print("Saving trained models")
        print(f"{'='*60}")
        
        model_info = {
            'experiment_time': self.experiment_time,
            'data_seed': self.data_seed,
            'model_seed': self.model_seed,
            'config': {
                'PRE_TRAIN_SAMPLES': pre_config_merged.PRE_TRAIN_SAMPLES,
                'PRE_VAL_SAMPLES': pre_config_merged.PRE_VAL_SAMPLES,
                'PRE_TEST_SAMPLES': pre_config_merged.PRE_TEST_SAMPLES,
                'PRE_BATCH_SIZE': pre_config_merged.PRE_BATCH_SIZE,
                'PRE_EPOCHS': pre_config_merged.PRE_EPOCHS,
                'PRE_LEARNING_RATE': pre_config_merged.PRE_LEARNING_RATE,
                'IMG_SIZE': pre_config_merged.IMG_SIZE,
                'NUM_FRAMES': pre_config_merged.NUM_FRAMES
            },
            'results': self.results
        }
        
        # Save model information
        with open(os.path.join(self.save_dir, 'experiment_info.json'), 'w') as f:
            json.dump(model_info, f, indent=4, default=str)
        
        # Save each model
        for name, trainer in self.trainers.items():
            # Save model weights
            model_path = os.path.join(self.save_dir, f"{name}_model.pth")
            torch.save(trainer.model.state_dict(), model_path)
            
            # Save the complete trainer state (including optimizer state)
            trainer_path = os.path.join(self.save_dir, f"{name}_trainer.pth")
            torch.save({
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_losses': trainer.train_losses,
                'val_losses': trainer.val_losses,
                'model_name': name
            }, trainer_path)
            
            print(f"Saved {name}:")
            print(f"  Model weights -> {model_path}")
            print(f"  Trainer state -> {trainer_path}")
        
        # Save results to CSV
        results_df = pd.DataFrame([
            {
                'Model': name,
                'MAE': result['MAE'],
                'RMSE': result['RMSE'],
                'Pearson_R': result['Pearson_R'],
                'R2': result['R2'],
                'Training_Time(s)': result['Training_Time']
            }
            for name, result in self.results.items()
        ])
        results_path = os.path.join(self.save_dir, 'results.csv')
        results_df.to_csv(results_path, index=False)
        
        print(f"\nAll models have been saved to directory: {self.save_dir}")
        print(f"Experiment information: {os.path.join(self.save_dir, 'experiment_info.json')}")
        print(f"Results CSV: {results_path}")
        
        return self.save_dir
    
    def load_model(self, model_name, load_trainer=False):
        """Load a saved model"""
        if model_name not in self.models:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Reinitialize the model
        model = self.models[model_name].to(self.device)
        
        if load_trainer:
            # Load trainer state
            trainer_path = os.path.join(self.save_dir, f"{model_name}_trainer.pth")
            if os.path.exists(trainer_path):
                checkpoint = torch.load(trainer_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded trainer state for {model_name}")
                return model, checkpoint
            else:
                print(f"Warning: {trainer_path} does not exist; loading model weights only")
        
        # Load model weights only
        model_path = os.path.join(self.save_dir, f"{model_name}_model.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model weights for {model_name}")
            return model
        else:
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
    
    def compare_models(self):
        """Compare the performance of all models"""
        print("\n" + "="*100)
        print("Merged experiment - performance comparison of all models")
        print("="*100)
        
        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                'Model': name,
                'MAE': result['MAE'],
                'RMSE': result['RMSE'],
                'Pearson_R': result['Pearson_R'],
                'R2': result['R2'],
                'Training_Time(s)': result['Training_Time']
            })
        
        df = pd.DataFrame(comparison_data)
        print(df.round(4))
        
        # Display results grouped by model type
        print("\n" + "="*60)
        print("Grouped comparison by model type")
        print("="*60)
        
        # Group 1: 3D models
        print("\nGroup 1 - 3D Models:")
        print(df[df['Model'].isin(['R3D-18', 'EfficientNet-3D', 'Graph-CNN'])].round(4))
        
        # Group 2: temporal models
        print("\nGroup 2 - Temporal Models:")
        print(df[df['Model'].isin(['CNN-LSTM', 'CNN-Transformer', 'Attention-LSTM'])].round(4))
        
        # Save results
        results_path = os.path.join(self.save_dir, 'pre_experiment_merged_results.csv')
        df.to_csv(results_path, index=False)
        print(f"\nMerged experiment results have been saved to {results_path}")
        
        return df
    
    def check_data_pipeline(self):
        """Check the data pipeline"""
        print("\n" + "="*50)
        print("Data pipeline check")
        print("="*50)
        
        for batch in self.train_loader:
            video = batch['video']
            ef = batch['EF']
            
            print(f"Video data shape: {video.shape}")
            print(f"Video data range: [{video.min():.3f}, {video.max():.3f}]")
            print(f"EF label range: [{ef.min():.1f}, {ef.max():.1f}]")
            print(f"EF label mean: {ef.mean():.1f} ± {ef.std():.1f}")
            break
        
        print(f"Data pipeline check completed! (data seed: {self.data_seed})")

def main():
    print("Starting the merged pre-experiment...")
    print(f"Configuration: {pre_config_merged.PRE_TRAIN_SAMPLES} training samples, {pre_config_merged.PRE_EPOCHS} training epochs")
    print(f"Data seed: {pre_config_merged.DATA_SEED}, model seed: {pre_config_merged.MODEL_SEED}")
    
    # Create the merged experiment
    pre_experiment = PreExperimentMerged(
        data_seed=pre_config_merged.DATA_SEED,
        model_seed=pre_config_merged.MODEL_SEED
    )
    
    # Check the data pipeline
    pre_experiment.check_data_pipeline()
    
    # Set up models
    pre_experiment.setup_models()
    
    # Train all models
    pre_experiment.train_all_models(epochs=pre_config_merged.PRE_EPOCHS)
    
    # Compare results
    results_df = pre_experiment.compare_models()
    
    # Save all models
    save_dir = pre_experiment.save_models()
    
    print("\n" + "="*60)
    print("Merged experiment completed!")
    print(f"All models have been saved to: {save_dir}")
    print("You can now analyze and visualize the performance comparison of all 6 models.")
    print("="*60)
    
    return results_df, save_dir

if __name__ == "__main__":
    results_df, save_dir = main()