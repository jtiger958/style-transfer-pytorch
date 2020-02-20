import PIL.Image
import zipfile
import tarfile
import urllib.request
from tqdm import tqdm

def load_image(file_name, size=None):
    image = PIL.Image.open(file_name)
    if size is not None:
        Image = image.resize((size, size))
    return image

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def unzip_zip_file(zip_path, data_path):
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall(data_path)
    zip_ref.close()


def unzip_tar_file(zip_path, data_path):
    tar_ref = tarfile.open(zip_path, "r:")
    tar_ref.extractall(data_path)
    tar_ref.close()


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    print("[!] download data file")
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)