import os, sys, shutil
from distutils.dir_util import copy_tree
from typing import Iterable
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as VF
from PIL import Image
from torchaudio import transforms as AT
import torchaudio
import librosa
from sklearn.preprocessing import MultiLabelBinarizer
import gdown

from mmlearn.util import log_progress, DATA_DIR

# HELPER FUNCTIONS

def load_image(path, h=400, w=400, transform=None):
    """Load an image into a normalized pytorch Tensor.
    
    Args:
        path: The path to the image file. Accepted filetypes are: TODO
        h: Height of output Tensor.
        w: Width of output Tensor.
        transform: An instantiated custom pytorch Transform to apply to the image.
            The transform should match expected output type and shape.
            If None, resize and normalize. 
    
    Returns:
        A normalized 3D pytorch Tensor of size (channels, h, w).
    """

    pil_img = Image.open(path).convert("RGB")
    if transform:
        img = transform(img)
    else:
        img = VF.resize(pil_img, (h, w))
        img = VF.to_tensor(img)
        img = VF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return img

def load_text(path):
    """Load a text file into a string.
    
    Args:
        path: The path to the text file.

    Returns: A string.
    """

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def _load_clip(clip_path):
    #clip_path = os.path.join(clip_dir, name + "." + suffix)
    #t1 = time.time()
    if clip_path.rsplit(".")[-1] == "wav":
        raw_clip, raw_sr = torchaudio.load(clip_path)
    else:
        raw_clip, raw_sr = librosa.load(clip_path, sr=None, mono=True)
        raw_clip = torch.Tensor(raw_clip).unsqueeze(0)
    #print(time.time() - t1, "s elapsed")
    return raw_clip, raw_sr

def _preprocess_clip(signal, sr, target_sr, target_samples):
    
    # resample if necessary
    if target_sr and sr != target_sr:
        signal = AT.Resample(sr, target_sr)(signal)

    # to mono
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)

    # cut if necessary
    if target_samples and signal.shape[1] > target_samples:
        signal = signal[:, 0:target_samples]

    # right pad if necessary
    if target_samples and signal.shape[1] < target_samples:
        missing = target_samples - signal.shape[1]
        signal = torch.nn.functional.pad(signal, (0, missing))

    out_sr = target_sr if target_sr else sr
    return signal, out_sr

SPECTROGRAM = AT.MelSpectrogram(
            n_fft = 1024,
            hop_length = 512,
            n_mels = 64,
            normalized = False,
        )

def _to_spectrogram(clip, db_scale=True, minmax_norm=True):
    spec = SPECTROGRAM(clip)
    if db_scale:
        spec = AT.AmplitudeToDB()(spec)
    if minmax_norm:
        spec -= torch.min(spec)
        spec /= torch.max(spec)
    return spec

def load_audio(path, sample_rate=16000, mono=True, n_samples=None, normalized=False, transform=None):
    """Load an audio clip into a pytorch Tensor.
    
    Args:
        path: The path to the audio file. Accepted filetypes are: TODO.
        sample_rate: The sample rate of the output signal Tensor. Resample input if necessary.
            Keep original sample rate if None.
        mono: Transform input signal to mono by averaging channels.
        n_samples: Cut or pad the input signal to make it n_samples long.
            Keep original length if None.
        normalized: Normalize amplitude values to [0, 1].
        transform: An instantiated custom pytorch Transform to apply to the signal
            The transform must match the output type and shape.

    Returns:
        A 2D pytorch Tensor of size (1 or 2, samples).
        The output sample rate.
    """

    raw_clip, raw_sr = _load_clip(path)
    clip, sr = _preprocess_clip(raw_clip, raw_sr, sample_rate, n_samples)

    if normalized:
        clip = VF.normalize(clip, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if transform:
        clip = transform(clip)

    return clip, sr

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
        if not (set(b).issubset(set(b))):
            log_progress(f"Path {dir} is a directory but its contents\
                            don't align with target.tsv. Skipping this modality.", level="warning")
            return False
        elif not all(fn.rsplit(".")[1] == fns[0].rsplit(".")[1] for fn in fns):
            log_progress(f"Path {dir} contains files\
                            but there are multiple file extensions present. Skipping this modality.", level="warning")
            return False
        return True

def _is_torch_multimodal_dataset(dataset):
    has_len = callable(getattr(dataset, "__len__"))
    has_getitem = callable(getattr(dataset, "__getitem__"))
    if not (has_len and has_getitem):
        log_progress(f"Provided object is not a torch Dataset.", level="warning")
        return False
    example = dataset[0]
    if not(type(example) == dict and "target" in example):
        log_progress(f"Provided dataset does not return expected data.", level="warning")
        return False
    valid_dtypes = True
    for mod in example:
        if mod in ["image", "audio", "video"]:
            valid_dtypes = valid_dtypes and isinstance(example[mod], torch.Tensor)
        if mod == "text":
            valid_dtype = valid_dtypes and type(example[mod]) == str
    if not valid_dtypes:
        log_progress(f"Provided dataset does not return expected data.", level="warning")
        return False
    return True


# BASE CLASSES

class MultimodalDataset(Dataset):

    def __init__(self, dir, col=1, frac=None, shuffle=False, header=False, verbose=True,
                    img_size=400,
                    sample_rate=16000, mono=True, n_samples=None):
        """Create a multimodal dataset object from directory.

        Args:
            dir: A directory that must include:
                * ./target.tsv file with filenames in col = 0 and target classes in col >= 1
                And a subset of the following:
                * ./image directory containing one .jpg or .png file per training example with filenames from target.tsv
                * ./text directory containing one .txt file per example
                * ./audio directory containing one .wav or .mp3 file per example
                * ./video directory containing one .mp4 file per example
                The dataset object will output data in modalities for which a directory is provided.
            col: Target class column in target.tsv.
            frac: Choose < 1 to randomly sub-sample loaded dataset.
            shuffle: Randomly shuffle training examples.
            img_size: Height and width for output images.
            sample_rate: The sample rate output audio clips. Resample input if necessary.
                Keep original sample rate if None.
            mono: Transform input signal to mono by averaging channels.
            n_samples: Cut or pad the input signal to make it n_samples long.
                Take the length of the first clip if None.
        """

        # Set modality params
        self.h = img_size
        self.sample_rate = sample_rate
        self.mono = mono
        self.n_samples = n_samples

        # Load and process the dataframe
        if not _has_dataset(dir):
            raise FileNotFoundError("Provided 'dir' does not contain dataset in required form.")

        self.df = pd.read_csv(os.path.join(dir, "target.tsv"), dtype={0: str}, sep="\t",
                    header=0 if header else None)
        if frac:
            self.df = self.df.sample(frac=frac)
        if shuffle:
            self.shuffle()

        self.classes, self.targets = np.unique(self.df.iloc[:,col], return_inverse=True)
        self.n_cls = len(self.classes)

        # Initialize this dataset's modalities
        self.data_dirs, self.exts, self.av_mods = {}, {}, {}
        for modality in ["image", "text", "audio", "video"]:
            data_dir = os.path.join(dir, modality)
            if _is_data_directory(data_dir, self):
                self.data_dirs[modality] = data_dir
                self.av_mods[modality] = True
                self.exts[modality] = "." + list(os.listdir(data_dir))[0].rsplit(".")[1]
            else:
                self.av_mods[modality] = False
        self.mods = self.av_mods.copy()

        # Set default audio clip n_samples
        if self.mods["audio"] and self.n_samples is None:
            first_clip_path = os.path.join(self.data_dirs["audio"], self.names[0] + self.exts["audio"])
            first_clip, sr = load_audio(first_clip_path, self.sample_rate, self.mono)
            self.n_samples = first_clip.shape[1]


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        id = self.df.iloc[i, 0]
        example = {}

        if self.mods["image"]:
            img = load_image(os.path.join(self.data_dirs["image"], id + self.exts["image"]),
                                h=self.h, w=self.h)
            example["image"] = img
        if self.mods["text"]:
            text = load_text(os.path.join(self.data_dirs["text"], id + self.exts["text"]))
            example["text"] = text
        if self.mods["audio"]:
            audio = load_audio(os.path.join(self.data_dirs["audio"], id + self.exts["audio"]),
                                sample_rate=self.sample_rate, mono=self.mono, n_samples=self.n_samples)
            example["audio"] = audio
        if self.mods["video"]:
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
            if dict[modality] is True and not self.av_mods[modality]:
                log_progress(f"Cannot enable a modality ({modality}) that was not provided.", level="warning")
            self.mods[modality] = dict[modality]

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
        
        texts, targets = [], []
        ids_valid = (len(args) > 0 and isinstance(args[0], Iterable) and len(args[0]) > 0)
        selection = args[0] if ids_valid else range(0, len(self))

        for i in selection:
            id = self.df.iloc[i, 0]
            text = load_text(os.path.join(self.data_dirs["text"], id + ".txt"))
            texts.append(text)
            targets.append(self.targets[i])
        return np.array(texts), np.array(targets)

    @property
    def names(self):
        """Filenames (ids) for all examples in dataset
        
        Returns: An Ndarray of names
        """
        return np.array(list(self.df.iloc[:,0]))

    @property
    def modalities(self):
        """Modalities covered by this dataset

        Returns: A list of strings.
        """
        return [m for m in self.mods if self.mods[m]]


class TorchMultimodalDataset(MultimodalDataset):

    def __init__(self, torch_dataset, img_size=400, frac=None, shuffle=False):
        """Create a MultimodalDataset from a PyTorch Dataset.
            See data.from_torch_dataset().
        """
    
        if not _is_torch_multimodal_dataset(torch_dataset):
            raise TypeError("Provided is not a valid multimodal torch Dataset.")

        self.torch_dataset = torch_dataset
        targets = [torch_dataset[i]["target"] for i in range(len(torch_dataset))]
        names = [str(i) for i in range(len(targets))]
        self.df = pd.DataFrame({0: names, 1: targets})

        if frac:
            self.df = self.df.sample(frac=frac)
        if shuffle:
            self.shuffle()

        example_keys = list(torch_dataset[0].keys())
        self.av_mods = {m: m in example_keys for m in ["image", "text", "audio", "video"]}
        self.mods = self.av_mods.copy()

        self.h = img_size
        self.classes, self.targets = np.unique(self.df.iloc[:,1], return_inverse=True)
        self.n_cls = len(self.classes)

    def __getitem__(self, i):
        id = self.df.iloc[i, 0]
        example = self.torch_dataset[id]
        example = {m: example[m] for m in example if m in self.mods}
        
        return example

    def get_texts(self, *args):
        """Get (a selection of) text and targets in dataset at once.

        Args:
            ids: An optional list of ids to retrieve.

        Returns: A tuple (texts, targets) of Ndarrays.
        """

        texts, targets = [], []
        ids_valid = (len(args) > 0 and isinstance(args[0], Iterable) and len(args[0]) > 0)
        selection = args[0] if ids_valid else range(0, len(self))    

        for i in selection:
            example = self.torch_dataset[i]
            texts.append(example["text"])
            targets.append(example["target"])
        return np.array(texts), np.array(targets)   


def from_torch_dataset(torch_dataset, img_size=400, frac=None, shuffle=False):
    """Create a MultimodalDataset from a PyTorch Dataset.
    Args:
        torch_dataset: A torch Dataset object which returns a dictionary in __getitem__.
            The returned dictionary should be a subset of {"image": Tensor, "text": String,
            "audio": Tensor, "video": Tensor, "target": int}.
        frac: Choose < 1 to randomly sub-sample loaded dataset.
        shuffle: Randomly shuffle training examples.

    Returns: A TorchMultimodalDataset (subclass of MultimodalDataset).
    """

    return TorchMultimodalDataset(torch_dataset, img_size=img_size, frac=frac, shuffle=shuffle)

    
    


# INCLUDED MULTIMODAL DATASETS

# TODO: change download source from dropbox
def _download_dataset(source, name, verbose=True):
    log_progress(f"Downloading dataset to {DATA_DIR}/{name}...", verbose=verbose)

    ds_dir = os.path.join(DATA_DIR, name)
    if not os.path.isdir(ds_dir):
        os.makedirs(ds_dir)
    zip_path = os.path.join(ds_dir, "data.zip")

    gdown.download(source, zip_path, quiet=not verbose)
    #urllib.request.urlretrieve(source, zip_path)
    zipfile.ZipFile(zip_path).extractall(ds_dir)
    os.remove(zip_path)

class CaltechBirds(MultimodalDataset):

    def __init__(self, name="caltech-birds"):
        source = "https://dl.dropboxusercontent.com/s/u2cyt2c3f1clqfb/caltech-birds.zip"
        
        if not _has_dataset(os.path.join(DATA_DIR, name)):
            _download_dataset(source, name)
        super(CaltechBirds, self).__init__(os.path.join(DATA_DIR, name))


class TastyRecipes(MultimodalDataset):

    def __init__(self, name="tasty-recipes"):
        source = "http://dl.dropboxusercontent.com/s/r95zhkuqhvphym3/tasty-recipes.zip"
        if not _has_dataset(os.path.join(DATA_DIR, name)):
            _download_dataset(source, name)  
        super(TastyRecipes, self).__init__(os.path.join(DATA_DIR, name))


class Fakeddit5k(MultimodalDataset):

    def __init__(self, name="fakeddit", class_col=1):
        source = "http://dl.dropboxusercontent.com/s/i1ef9xinebp0dof/fakeddit.zip"
        if not _has_dataset(os.path.join(DATA_DIR, name)):
            _download_dataset(source, name)
        super(Fakeddit5k, self).__init__(os.path.join(DATA_DIR, name), col=class_col)


class Fauxtography(MultimodalDataset):

    def __init__(self, name="fauxtography"):
        source = "http://dl.dropboxusercontent.com/s/zl6or9lc7ph836s/fauxtography.zip"
        if not _has_dataset(os.path.join(DATA_DIR, name)):
            _download_dataset(source, name)
        super(Fauxtography, self).__init__(os.path.join(DATA_DIR, name))


class SpotifyMultimodalPop(MultimodalDataset):

    def __init__(self, name="spotify_multimodal_pop", frac=None):
        source = "https://download1336.mediafire.com/k9dl7neyh2eg/fnd60rzaj0ey4na/spotify_mm_pop.zip"
        if not _has_dataset(os.path.join(DATA_DIR, name)):
            _download_dataset(source, name)
        super(SpotifyMultimodalPop, self).__init__(os.path.join(DATA_DIR, name),
                header=True, n_samples=48000, frac=frac)


class SpotifyMultimodalVal(MultimodalDataset):

    def __init__(self, name="spotify_multimodal_val", frac=None):
        source = "https://download1497.mediafire.com/mku6selaeemg/w04p8oisw4i03i5/spotify_mm_val.zip"
        if not _has_dataset(os.path.join(DATA_DIR, name)):
            _download_dataset(source, name)
        super(SpotifyMultimodalVal, self).__init__(os.path.join(DATA_DIR, name),
                header=True, n_samples=48000, frac=frac)
    