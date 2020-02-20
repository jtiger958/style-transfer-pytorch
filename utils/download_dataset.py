import os
import shutil
from utils.utils import download_url, unzip_zip_file


def download_dataset(config):
    urls = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip'

    if not os.path.exists('temp'):
        os.makedirs('temp')
    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    download_url(urls, os.path.join('temp', 'dataset.zip'))

    unzip_zip_file(os.path.join('temp', 'dataset.zip'), 'temp')

    images = os.listdir(os.path.join('temp', 'monet2photo', 'trainB'))

    print('[*] Move image')
    for image_name in images:
        shutil.move(os.path.join('temp', 'monet2photo', 'trainB', image_name), os.path.join('dataset', image_name))

    shutil.rmtree('temp')
