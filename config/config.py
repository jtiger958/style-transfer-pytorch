import argparse


parser = argparse.ArgumentParser()


parser.add_argument('--image_size', type=int, default=224, help='the height / width of the input image to network')
parser.add_argument('--batch_size', type=int, default=2, help='batch size')
parser.add_argument('--sample_batch_size', type=int, default=1, help='sample batch size')
parser.add_argument('--num_epoch', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--epoch', type=int, default=0, help='epochs in current train')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--data_path', default='dataset', help='path to dataset')
parser.add_argument('--checkpoint_dir', default='checkpoints', help="path to saved models (to continue training)")
parser.add_argument('--sample_dir', default='samples', help='folder to output images and model checkpoints')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--num_residual', type=int, default=5, help='number of residual network for style transfer')
parser.add_argument("--content_weight", type=float, default=1e5, help="weight for content-loss, default is 1e5")
parser.add_argument("--style_weight", type=float, default=1e10, help="weight for style-loss, default is 1e10")
parser.add_argument("--style_image_name", type=str, default="picabia", help="image for getting style")
parser.add_argument("--style_dir", type=str, default="style", help="image folder for getting style")


def get_config():
    return parser.parse_args()
