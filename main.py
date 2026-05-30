import os
from preprocessing import EchoNetPreprocessor
from main_dataloader_merged import get_main_dataloaders   # Use the data loader for the main experiment
from main_config_merged import main_config as config      # Use the unified configuration for the main experiment

def main():
    print("EchoNet-Dynamic data processing pipeline (updated version: motion cropping + interpolation resampling)")
    print("=" * 60)
    
    # Check the raw video directory
    if not os.path.exists(config.VIDEOS_DIR):
        print(f"Error: video directory not found {config.VIDEOS_DIR}")
        return
    
    # Step 1: offline preprocessing
    print("\nStep 1: starting offline preprocessing...")
    preprocessor = EchoNetPreprocessor()
    preprocessor.process_all_videos(num_workers=4)
    
    # Step 2: test the data loader
    print("\nStep 2: testing the data loader...")
    train_loader, val_loader, test_loader = get_main_dataloaders(data_seed=config.DATA_SEED)
    
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    # Display the data shapes of one batch
    for batch in train_loader:
        video = batch['video']
        print(f"\nExample batch video shape: {video.shape}")   # Expected: (batch, T, C, H, W)
        print(f"EDV label shape: {batch['EDV'].shape}")
        print(f"ESV label shape: {batch['ESV'].shape}")
        print(f"EF label shape: {batch['EF'].shape}")
        break
    
    print("\n✅ Data preprocessing and data loader testing completed!")

if __name__ == "__main__":
    main()