import os

class PreConfigMerged:
    # routes configuration
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ECHONET_DIR = os.path.join(BASE_DIR, "EchoNet-Dynamic")
    VIDEOS_DIR = os.path.join(ECHONET_DIR, "Videos")
    FILELIST_PATH = os.path.join(ECHONET_DIR, "FileList.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data")
    NPZ_DIR = os.path.join(OUTPUT_DIR, "npz_files")
    
    # routes configuration
    IMG_SIZE = (112, 112)
    NUM_FRAMES = 100
    ECHO_MEAN = [0.5, 0.5, 0.5]
    ECHO_STD = [0.5, 0.5, 0.5]
    
    # The number of experimental samples
    PRE_TRAIN_SAMPLES = 200
    PRE_VAL_SAMPLES = 50  
    PRE_TEST_SAMPLES = 50
    
    # Training parameters
    PRE_BATCH_SIZE = 4
    PRE_EPOCHS = 15
    PRE_LEARNING_RATE = 1e-4
    
    # Random seeds (which can be respectively used for data partitioning and model training)
    DATA_SEED = 42  
    MODEL_SEED = 123 

pre_config_merged = PreConfigMerged()