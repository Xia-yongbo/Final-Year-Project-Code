import sys
from pathlib import Path
from torch.utils.data import DataLoader

# Add the current file directory to the path to ensure modules in the same directory can be imported
sys.path.append(str(Path(__file__).parent))

from pre_dataloader_merged import PreEchoNetDatasetMerged, ModerateTransform
from main_config_merged import main_config


def get_main_dataloaders(data_seed=main_config.DATA_SEED):
    """
    Main experiment data loaders
    Build the train / val / test DataLoaders using the configuration in main_config
    """

    train_dataset = PreEchoNetDatasetMerged(
        split='train',
        transform=ModerateTransform(is_training=True),
        max_samples=main_config.TRAIN_SAMPLES,   # None means using all samples
        data_seed=data_seed
    )

    val_dataset = PreEchoNetDatasetMerged(
        split='val',
        transform=ModerateTransform(is_training=False),
        max_samples=main_config.VAL_SAMPLES,
        data_seed=data_seed
    )

    test_dataset = PreEchoNetDatasetMerged(
        split='test',
        transform=ModerateTransform(is_training=False),
        max_samples=main_config.TEST_SAMPLES,
        data_seed=data_seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=main_config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=main_config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=main_config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader