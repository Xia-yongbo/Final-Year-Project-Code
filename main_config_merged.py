import os

class MainConfigMerged:
    # Path configuration
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ECHONET_DIR = os.path.join(BASE_DIR, "EchoNet-Dynamic")
    VIDEOS_DIR = os.path.join(ECHONET_DIR, "Videos")
    FILELIST_PATH = os.path.join(ECHONET_DIR, "FileList.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data")
    NPZ_DIR = os.path.join(OUTPUT_DIR, "npz_files")

    # Data split names
    SPLITS = {
        "TRAIN": "TRAIN",
        "VAL": "VAL",
        "TEST": "TEST"
    }

    # Data configuration -- increase resolution, reduce frame count, and match the real cardiac cycle
    IMG_SIZE = (128, 128)
    NUM_FRAMES = 64
    ECHO_MEAN = [0.5, 0.5, 0.5]
    ECHO_STD = [0.5, 0.5, 0.5]

    # Use the full dataset for the main experiment
    TRAIN_SAMPLES = None
    VAL_SAMPLES = None
    TEST_SAMPLES = None

    # Training parameters -- use gradient accumulation to increase the effective batch size and adjust the learning rate
    BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 4
    EPOCHS = 50
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 1e-4

    # Learning rate scheduler parameters using smoothed validation loss
    LR_SCHEDULER_PATIENCE = 5
    LR_SCHEDULER_FACTOR = 0.5
    LR_SMOOTHING = 0.9

    # Early stopping parameters
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MIN_DELTA = 1e-4

    # Random seeds
    DATA_SEED = 42
    MODEL_SEED = 123

    # Pretrained weights optional
    PRETRAINED_DIR = "saved_models/experiment_20260228_181642"
    USE_PRETRAINED = False

main_config = MainConfigMerged()