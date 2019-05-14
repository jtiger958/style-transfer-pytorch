import torch
import os
from model.transfer_net import TransferNet
from model.vgg import VGG16
from glob import glob
from torch.optim.adam import Adam
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from util.util import load_image, gram_matrix


class Trainer:
    def __init__(self, config, data_loader):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_epoch = config.num_epoch
        self.epoch = config.epoch
        self.image_size = config.image_size
        self.data_loader = data_loader
        self.num_residual = config.num_residual
        self.checkpoint_dir = config.checkpoint_dir
        self.lr = config.lr
        self.content_weight = config.content_weight
        self.style_weight = config.style_weight
        self.style_image_name = config.style_image_name
        self.style_dir = config.style_dir
        self.batch_size = config.batch_size
        self.sample_dir = config.sample_dir

        self.build_model()
        self.load_feature_style()

    def train(self):
        total_step = len(self.data_loader)
        optimizer = Adam(self.transfer_net.parameters(), lr=self.lr)
        loss = nn.MSELoss()
        self.transfer_net.train()
        self.vgg.eval()

        for epoch in range(self.epoch, self.num_epoch):
            if not os.path.exists(os.path.join(self.sample_dir, self.style_image_name, f"{epoch}")):
                os.makedirs(os.path.join(self.sample_dir, self.style_image_name, f"{epoch}"))
            for step, image in enumerate(self.data_loader):

                optimizer.zero_grad()
                image = image.to(self.device)
                transformed_image = self.transfer_net(image)

                image_feature = self.vgg(image)
                transformed_image_feature = self.vgg(transformed_image)

                content_loss = self.content_weight*loss(image_feature.relu2_2, transformed_image_feature.relu2_2)

                style_loss = 0
                for ft_y, gm_s in zip(transformed_image_feature, self.gram_style):
                    gm_y = gram_matrix(ft_y)
                    style_loss += loss(gm_y, gm_s[:self.batch_size, :, :])
                style_loss *= self.style_weight

                total_loss = content_loss + style_loss

                total_loss.backward(retain_graph=True)
                optimizer.step()

                if step % 10 == 0:
                    print(f"[Epoch {epoch}/{self.num_epoch}] [Batch {step}/{total_step}] "
                          f"[Style loss: {style_loss.item():.4}] [Content loss loss: {content_loss.item():.4}]")
                    if step % 100 == 0:
                        save_image(transformed_image, os.path.join(self.sample_dir, self.style_image_name, f"{epoch}", f"{step}.png"), normalize=False)

            torch.save(self.transfer_net.state_dict(), os.path.join(self.checkpoint_dir, self.style_image_name, f"TransferNet_{epoch}.pth"))

    def build_model(self):
        self.transfer_net = TransferNet(self.num_residual)
        self.transfer_net.apply(self.weights_init)
        self.transfer_net.to(self.device)
        self.vgg = VGG16(requires_grad=True)
        self.vgg.to(self.device)
        self.load_model()

    def load_model(self):
        print(f"[*] Load model from {self.checkpoint_dir}")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.listdir(self.checkpoint_dir):
            print(f"[!] No checkpoint in {self.checkpoint_dir}")
            return

        transfer_net = glob(os.path.join(self.checkpoint_dir, self.style_image_name, f'TransferNet_{self.epoch - 1}.pth'))

        if not transfer_net:
            print(f"[!] No checkpoint in epoch {self.epoch - 1}")
            return

        self.transfer_net.load_state_dict(torch.load(transfer_net[0]))

    def load_feature_style(self):
        if not os.path.exists(self.style_dir):
            os.makedirs(self.style_dir)
        if not os.listdir(self.style_dir):
            raise Exception(f"[!] No image for style transfer")

        image_name = glob(os.path.join(self.style_dir, f"{self.style_image_name}.*"))
        if not image_name:
            raise Exception(f"[!] No image for {self.style_image_name} transfer")

        image = load_image(image_name[0], size=self.image_size)
        image = transforms.Compose([
            transforms.CenterCrop(min(image.size[0], image.size[1])),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])(image)
        image = image.repeat(self.batch_size, 1, 1, 1)
        image = image.to(self.device)
        style_image = self.vgg(image)
        self.gram_style = [gram_matrix(y) for y in style_image]

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)

        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

