import torch
from CIFAR10DVS.DVS_CIFAR10_data_process.DVS_CIFAR10_dataloaders import create_datasets


def create_data(config):
    # Data set
    config.train_dataset = create_datasets(
        config.savePath,
        train=True,
        is_train_Enhanced=config.is_train_Enhanced,
        ds=config.ds,
        dt=config.dt * 1000,
        chunk_size_train=config.T,
    )

    config.test_dataset = create_datasets(
        config.savePath,
        train=False,
        ds=config.ds,
        dt=config.dt * 1000,
        chunk_size_test=config.T,
        clip=config.clip
    )
    # Data loader
    config.train_loader = torch.utils.data.DataLoader(
        config.train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=config.drop_last,
        pin_memory=config.pip_memory,
        num_workers = config.num_workers,

    )
    config.test_loader = torch.utils.data.DataLoader(
        config.test_dataset,
        batch_size=config.batch_size_test,
        shuffle=False,
        drop_last=config.drop_last,
        pin_memory=config.pip_memory,
        num_workers=config.num_workers,
    )
