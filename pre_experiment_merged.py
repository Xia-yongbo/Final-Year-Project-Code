import torch
import pandas as pd
import time
import os
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
    
    def compare_models(self):
        """Compare the performance of all models"""
        print("\n" + "="*100)
        print("Merged Experiment - Performance Comparison of All Models")
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
        print("Comparison by Model Type")
        print("="*60)
        
        # Group 1: 3D models
        print("\nGroup 1 - 3D Models:")
        print(df[df['Model'].isin(['R3D-18', 'EfficientNet-3D', 'Graph-CNN'])].round(4))
        
        # Group 2: temporal models
        print("\nGroup 2 - Temporal Models:")
        print(df[df['Model'].isin(['CNN-LSTM', 'CNN-Transformer', 'Attention-LSTM'])].round(4))
        
        # Save results
        df.to_csv('pre_experiment_merged_results.csv', index=False)
        print("\nMerged experiment results have been saved to pre_experiment_merged_results.csv")
        
        return df
    
    def check_data_pipeline(self):
        """Check the data pipeline"""
        print("\n" + "="*50)
        print("Data Pipeline Check")
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
    print("Starting merged preliminary experiment...")
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
    
    print("\nMerged experiment completed!")
    print("The performance comparison of all 6 models can now be analyzed.")
    
    return results_df

if __name__ == "__main__":
    main()