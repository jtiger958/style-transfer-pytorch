from src.train import Trainer
from src.test import Tester
from dataloader.dataloader import get_loader
import os
from config.config import get_config
from utils.download_dataset import download_dataset


def main(config):
    if config.checkpoint_dir is None:
        config.checkpoint_dir = 'checkpoints'

    if not os.path.exists(os.path.join(config.checkpoint_dir, config.style_image_name)):
        os.makedirs(os.path.join(config.checkpoint_dir, config.style_image_name))

    if not os.path.exists(os.path.join(config.sample_dir, config.style_image_name)):
        os.makedirs(os.path.join(config.sample_dir, config.style_image_name))

    if not os.path.exists(config.data_path) or not os.listdir(config.data_path):
        download_dataset(config)

    print(f"photo to {config.style_dir} style transfer using cnn")

    data_loader, val_data_loader = get_loader(config.data_path, config.image_size
                                              , config.batch_size, config.sample_batch_size)
    trainer = Trainer(config, data_loader)
    trainer.train()

    tester = Tester(config, val_data_loader)
    tester.test()


if __name__ == "__main__":
    config = get_config()
    main(config)
