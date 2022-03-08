import os, sys, shutil
from distutils.dir_util import copy_tree
from typing import Iterable
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as tf
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
import gdown
from google_drive_downloader import GoogleDriveDownloader as gdd

from util import log_progress, DATA_DIR

DEF_IMG_H = 400
DEF_IMG_W = 400

# HELPER FUNCTIONS

def load_image(path, h=DEF_IMG_H, w=DEF_IMG_W, transform=None):
    pil_img = Image.open(path).convert("RGB")
    if transform:
        img = transform(img)
    else:
        img = F.resize(pil_img, (h, w))
        img = F.to_tensor(img)
        img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return img

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def _has_dataset(root):
    if not os.path.isdir(root):
        return False
    if not os.path.isfile(os.path.join(root, "target.tsv")):
        return False
    return True


# BASE CLASSES

class MultimodalDataset(Dataset):

    def __init__(self, dir, img_size=DEF_IMG_H, col=1, frac=None, shuffle=False):
        """Create a multimodal dataset object from directory.

        Args:
            dir: A directory that must include:
                * ./image directory containing one .jpg file per training example
                * ./texts directory containing one .txt file per example, with matching filenames
                * ./target.tsv file with filenames in col = 0 and target classes in col >= 1
            img_size: Image height (= width) after loading.
            col: Target class column in target.tsv.
            frac: Choose < 1 to subsample loaded dataset.
            shuffle: Randomly shuffle training examples.
        """
        print(dir)
        if not _has_dataset(dir):
            raise FileNotFoundError("Provided 'dir' does not contain dataset in required form.")

        self.df = pd.read_csv(os.path.join(dir, "target.tsv"), dtype={0: str}, sep="\t", header=None)
        if frac:
            self.df = self.df.sample(frac=frac)
        if shuffle:
            self.shuffle()
        self.img_dir = os.path.join(dir, "images")
        self.text_dir = os.path.join(dir, "texts")
        self.h = img_size

        # self.norm = tf.Normalize(mean=[0.485, 0.456, 0.406],
        #                         std=[0.229, 0.224, 0.225])
        # self.transform = tf.Compose([tf.Resize((self.h, self.h)), tf.ToTensor(), self.norm])

        self.classes, self.targets = np.unique(self.df.iloc[:,col], return_inverse=True)
        self.n_cls = len(self.classes)


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        id = self.df.iloc[i, 0]
        # pil_img = Image.open(os.path.join(self.img_dir, id + ".jpg")).convert("RGB")
        # img = self.transform(pil_img)
        img = load_image(os.path.join(self.img_dir, id + ".jpg"), h=self.h, w=self.h)
        text = load_text(os.path.join(self.text_dir, id + ".txt"))
        target = self.targets[i] # targets are indices, self.classes[target] to get strings

        return img, text, target

    def shuffle(self, seed=None):
        perm = np.random.permutation(len(self))
        self.df = self.df.iloc[perm, :]
        self.targets = self.targets[perm]

    def get_texts(self, *args):
        """Get (a selection of) texts and targets in dataset at once.

        Args:
            ids: An optional list of ids to retrieve.

        Returns: A tuple (texts, targets) of Ndarrays
        """

        texts = []
        targets = []
        ids_valid = (len(args) > 0 and isinstance(args[0], Iterable) and len(args[0]) > 0)
        selection = args[0] if ids_valid else range(0, len(self))

        for i in selection:
            id = self.df.iloc[i, 0]
            text = load_text(os.path.join(self.text_dir, id + ".txt"))
            texts.append(text)
            targets.append(self.targets[i])
        return np.array(texts), np.array(targets)

    @property
    def names(self):
        """Get filenames (ids) for all examples in dataset
        
        Returns: An Ndarray of names
        """
        return np.array(list(self.df.iloc[:,0]))



# INCLUDED MULTIMODAL DATASETS

# TODO: change download source from drive
def _download_dataset(source, name):
    log_progress(f"Downloading dataset to {DATA_DIR}/{name}...")
    if not os.path.isdir(DATA_DIR):
        os.mkdir(DATA_DIR)
    ds_dir = os.path.join(DATA_DIR, name)
    zip_path = os.path.join(DATA_DIR, "data.zip")
    # if not os.path.isdir(ds_dir):
    #     os.mkdir(ds_dir)
    
    gdown.download(source, zip_path, quiet=False)
    
    #urllib.request.urlretrieve(source, zip_path)

    zipfile.ZipFile(zip_path).extractall(DATA_DIR)
    os.remove(zip_path)
    #copy_tree(source, root)

class CaltechBirds(MultimodalDataset):

    def __init__(self, name="caltech-birds"):
        source = "http://dl.dropboxusercontent.com/s/um716b92ih2851f/caltech-birds.zip"
        if not _has_dataset(os.path.join(DATA_DIR, name)):
            _download_dataset(source, name)
        super(CaltechBirds, self).__init__(os.path.join(DATA_DIR, name))


class TastyRecipes(MultimodalDataset):

    def __init__(self, name="tasty-recipes"):
        source = "http://dl.dropboxusercontent.com/s/ems0r9jdkbkamtz/tasty-recipes.zip"
        if not _has_dataset(os.path.join(DATA_DIR, name)):
            _download_dataset(source, name)        
        super(TastyRecipes, self).__init__(os.path.join(DATA_DIR, name))


class Fakeddit5k(MultimodalDataset):

    def __init__(self, name="fakeddit", class_col=1):
        source = "http://dl.dropboxusercontent.com/s/fyctsd9gqyygqrw/fakeddit.zip"
        if not _has_dataset(os.path.join(DATA_DIR, name)):
            _download_dataset(source, name)
        super(Fakeddit5k, self).__init__(os.path.join(DATA_DIR, name), col=class_col)


class Fauxtography(MultimodalDataset):

    def __init__(self, name="fauxtography"):
        source = "http://dl.dropboxusercontent.com/s/m75ima4r20m1uzs/fauxtography.zip"
        if not _has_dataset(os.path.join(DATA_DIR, name)):
            _download_dataset(source, name)
        super(Fauxtography, self).__init__(os.path.join(DATA_DIR, name))
