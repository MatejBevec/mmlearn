import os, sys, shutil
from distutils.dir_util import copy_tree
from typing import Iterable
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
import gdown

from mmlearn.util import log_progress, DATA_DIR

# HELPER FUNCTIONS

def load_image(path, h=400, w=400, transform=None):
    """Load an image into a normalized pytorch Tensor.
    
    Args:
        h: Height of output Tensor.
        w: Width of output Tensor.
        transform: A custom pytorch Transform to apply to image. If None, resize and normalize.
    """

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

def load_audio(path):
    #TODO
    pass

def load_video(path):
    #TODO
    pass

def _has_dataset(root):
    if not os.path.isdir(root):
        return False
    if not os.path.isfile(os.path.join(root, "target.tsv")):
        return False
    return True

def _is_data_directory(dir, ds):
    if not os.path.isdir(dir):
        return False
    else:
        fns = sorted(list(os.listdir(dir)))
        a = sorted([fn.rsplit(".")[0] for fn in os.listdir(dir)])
        b = sorted(ds.df.iloc[:, 0])
        if not (a == b):
            log_progress(f"WARNING: path {dir} is a directory but its contents\
                            don't align with target.tsv. Skipping this modality.")
            return False
        elif not all(fn.rsplit(".")[1] == fns[0].rsplit(".")[1] for fn in fns):
            log_progress(f"WARNING: path {dir} contains files\
                            the file extensions don't match. Skipping this modality.")
            return False
        return True

# BASE CLASSES

class MultimodalDataset(Dataset):

    def __init__(self, dir, img_size=400, col=1, frac=None, shuffle=False):
        """Create a multimodal dataset object from directory.

        Args:
            dir: A directory that must include:
                * ./target.tsv file with filenames in col = 0 and target classes in col >= 1
                And a subset of the following:
                * ./images directory containing one .jpg or .png file per training example with filenames from target.tsv
                * ./texts directory containing one .txt file per example
                * ./audio directory containing one .wav or .mp3 file per example
                * ./video directory containing one .mp4 file per example
                The dataset object will output data in modalities for which a directory is provided.
            col: Target class column in target.tsv.
            frac: Choose < 1 to randomly sub-sample loaded dataset.
            shuffle: Randomly shuffle training examples.
        """

        if not _has_dataset(dir):
            raise FileNotFoundError("Provided 'dir' does not contain dataset in required form.")

        self.df = pd.read_csv(os.path.join(dir, "target.tsv"), dtype={0: str}, sep="\t", header=None)
        if frac:
            self.df = self.df.sample(frac=frac)
        if shuffle:
            self.shuffle()

        self.data_dirs, self.exts, self.modalities = {}, {}, {}
        for modality in ["images", "texts", "audio", "video"]:
            data_dir = os.path.join(dir, modality)
            if _is_data_directory(data_dir, self):
                self.data_dirs[modality] = data_dir
                self.modalities[modality] = True
                self.exts[modality] = "." + list(os.listdir(data_dir))[0].rsplit(".")[1]
            else:
                self.modalities[modality] = False

        # self.img_dir = os.path.join(dir, "images")
        # self.text_dir = os.path.join(dir, "texts")
        self.h = img_size

        self.classes, self.targets = np.unique(self.df.iloc[:,col], return_inverse=True)
        self.n_cls = len(self.classes)


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        id = self.df.iloc[i, 0]
        example = {}

        if self.modalities["images"]:
            img = load_image(os.path.join(self.data_dirs["images"], id + self.exts["images"]),
                             h=self.h, w=self.h)
            example["image"] = img
        if self.modalities["texts"]:
            text = load_text(os.path.join(self.data_dirs["texts"], id + self.exts["texts"]))
            example["text"] = text
        if self.modalities["audio"]:
            audio = load_audio(os.path.join(self.data_dirs["audio"], id + self.exts["audio"]))
            example["audio"] = audio
        if self.modalities["video"]:
            video = load_video(os.path.join(self.data_dirs["video"], id + self.exts["video"]))
            example["video"] = video

        target = self.targets[i] # targets are indices, self.classes[target] to get the string
        example["target"] = target
        return example

    def toggle_modalities(self, dict):
        """Enable or disable modalities to return. Use to prevent loading unused data.
        
        Args:
            dict: A dictionary, of form {"modality": True/False}
        """
        for modality in dict:
            if dict[modality] is True and not self.data_dirs[modality]:
                log_progress(f"WARNING: Cannot enable a modality ({modality}) that was not provided.")
            self.modalities[modality] = dict[modality]

    def shuffle(self, seed=None):
        """Randomly shuffle the dataset in place."""
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
            text = load_text(os.path.join(self.data_dirs["texts"], id + ".txt"))
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

# TODO: change download source from dropbox
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
