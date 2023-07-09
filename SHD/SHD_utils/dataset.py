import torch
from SHD.SHD_data_process.SHD_Dataset import SHD_Dataset


def create_data(config):
    # Data set
    config.train_dataset = SHD_Dataset(
        config.savePath,
        train=True,
        is_train_Enhanced=config.is_train_Enhanced,
        ds=config.ds,
        dt=config.dt,
        T=config.T,
    )

    config.test_dataset = SHD_Dataset(
        config.savePath,
        train=False,
        ds=config.ds,
        dt=config.dt,
        T=config.T,
        clips=config.clip,
    )
    # Data loader
    config.train_loader = torch.utils.data.DataLoader(
        config.train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=config.drop_last,
        num_workers=config.num_work,
        pin_memory=config.pip_memory)
    config.test_loader = torch.utils.data.DataLoader(
        config.test_dataset,
        batch_size=config.batch_size_test,
        shuffle=False,
        drop_last=config.drop_last,
        num_workers=config.num_work,
        pin_memory=config.pip_memory)
