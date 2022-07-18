import os, sys, shutil, copy
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

    #HACK
    clip.sr = sr

    return clip

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

def _is_array_multimodal_dataset(dataset):
    has_len = callable(getattr(dataset, "__len__"))
    has_getitem = callable(getattr(dataset, "__getitem__"))
    if not (has_len and has_getitem):
        log_progress(f"Provided object is not a array-like.", level="warning")
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

def _check_idx(idx, n):
    idx_valid = (isinstance(idx, Iterable) and len(idx) > 0)
    # TODO: always return ndarray
    idx = idx if idx_valid else range(0, n)
    return idx


# BASE CLASSES

class MultimodalDataset(Dataset):

    def __init__(self, dir, col=1, frac=None, shuffle=False, header=False, verbose=True, tensor=True,
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
            header: Put True if target.tsv uses a header.
            verbose: Print progress information.
            tensor: Return Tensors for images, audio, video if true, else return ndarrays.
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
        self.base_dir = dir
        if not _has_dataset(dir):
            raise FileNotFoundError("Provided 'dir' does not contain dataset in required form.")

        self.df = pd.read_csv(os.path.join(dir, "target.tsv"), dtype={0: str}, sep="\t",
                    header=0 if header else None)
        if frac:
            self.df = self.df.sample(frac=frac)

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

        self.loaders = {
            "image": self._limage,
            "text": self._ltext,
            "audio": self._laudio,
            "video": self._lvideo,
        }
        self.loaders = {m: self.loaders[m] for m in self.loaders if self.mods[m]}

        if shuffle:
            self._shuffle()

    def _limage(self, i):
        id = self.df.iloc[i, 0]
        return load_image(os.path.join(self.data_dirs["image"], id + self.exts["image"]),
                                h=self.h, w=self.h)

    def _ltext(self, i):
        id = self.df.iloc[i, 0]
        return load_text(os.path.join(self.data_dirs["text"], id + self.exts["text"]))

    def _laudio(self, i):
        id = self.df.iloc[i, 0]
        return load_audio(os.path.join(self.data_dirs["audio"], id + self.exts["audio"]),
                                sample_rate=self.sample_rate, mono=self.mono, n_samples=self.n_samples)

    def _lvideo(self, i):
        id = self.df.iloc[i, 0]
        return load_video(os.path.join(self.data_dirs["video"], id + self.exts["video"]))

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i):
        # IS THIS A GOOD IDEA?
        if isinstance(i, slice):
            return self.sample(i)

        example = {}

        # Load file for every modality
        for m in self.loaders:
            example[m] = self.loaders[m](i)

        target = self.targets[i] # targets are indices, self.classes[target] to get the string
        example["target"] = target
        return example

    def __str__(self):
        st = "\nMultimodalDataset:\n\n"
        st += f"Source dir: {self.base_dir}\n"
        st += f"Size: {len(self)}\n"
        st += f"Modalities: {self.modalities}\n"
        cl = {i: self.classes[i] for i in range(len(self.classes))}
        st += f"Classes: {cl}\n\n"
        st += f"{self.df}"
        return st


    def toggle_modalities(self, dict):
        """Enable or disable modalities to return. Use to prevent loading unused data.
        
        Args:
            dict: A dictionary, of form {"modality": True/False}
        """
        for modality in dict:
            if dict[modality] is True and not self.av_mods[modality]:
                log_progress(f"Cannot enable a modality ({modality}) that was not provided.", level="warning")
            self.mods[modality] = dict[modality]

    def clone(self):
        return copy.deepcopy(self)
    
    def _shuffle(self, seed=420):
        """Shuffle this dataset in place."""

        perm = np.random.RandomState(seed=seed).permutation(len(self))
        self.df = self.df.iloc[perm, :].reset_index(drop=True)
        self.targets = self.targets[perm]

    def shuffle(self, seed=420):
        """Return a randomly shuffled copy of the dataset."""
        
        cl = self.clone()
        cl._shuffle(self, seed=seed)
        return cl

    def sample(self, idx):
        """Return a copy of dataset, sampled according to indices in idx."""
        cl = self.clone()
        cl.df = self.df.iloc[idx, :]
        cl.targets = self.targets[idx]
        return cl

    def get_data(self, modalities, idx, tensor=True, keep_dict=False, target=True):
        """Get (a selection of) examples in chosen modalities (and targets) at once.
            Returns a dict of form ({"modality": data}, targets), the same as a DataLoader batch.
            Note that for most datasets, all images/audio/videos will not fit in memory.
            Using MultimodalDataset with a DataLoader is preferred.

        Args:
            modalities: A list of strings representing the modalities to retrieve. All if None.
            idx: An array-like of indices to retrieve.
            tensor: If True, array-like data is returned as Tensors. If False, data is returned as ndarrays.
            keep_dict: If False, the wrapping dict is omitted in case of single modality. Only the data is returned.
            target: If True, target variables are returned in a tuple.

        Returns: A (data:dict, targets) tuple if target is True, data:dict otherwise.
        """
        
        if modalities is None:
            modalities = self.modalities

        idx = _check_idx(idx, len(self))

        data = {}
        for m in modalities:
            if self.mods[m]:
                data[m] = []
                for i in idx:
                    file = self.loaders[m](i)
                    data[m].append(file)
        
        for m in data:
            if tensor:
                data[m] = data[m] if m == "text" else torch.stack(data[m], dim=0)
            else:
                np.stack(data[m], axis=0)
        
        if not keep_dict and len(data) == 1:
            data = data[list(data.keys())[0]]

        out = data
        if target:
            out = (out, self.get_targets(idx, tensor=tensor))
        return out

    def get_images(self, idx=None, tensor=True, target=True):
        """Get (a selection of) images (and targets).
            Note that for most datasets, all images won't fit in memory.
            Using MultimodalDataset with a DataLoader is preferred.

        Args:
            idx: An optional list of ids to retrieve. All if None.
            tensor: If True, image batch is returned as Tensor. If False, image batch is returned as ndarray.
            target: If True, target variables are returned in a tuple.
        
        Returns: A (images, targets) tuple if target is True, images otherwise.
        """

        return self.get_data(["image"], idx, tensor=tensor, keep_dict=False)

    def get_texts(self, idx=None, tensor=True, target=True):
        """Get (a selection of) texts (and targets).

        Args:
            idx: An optional list of ids to retrieve. All if None.
            tensor: If True, texts batch is returned as Tensor. If False, text batch is returned as ndarray.
            target: If True, target variables are returned in a tuple.
        
        Returns: A (texts, targets) tuple if target is True, texts otherwise.
        """

        return self.get_data(["text"], idx, tensor=tensor, keep_dict=False)

    def get_audio(self, idx=None, tensor=True, target=True):
        """Get (a selection of) audio clips (and targets).

        Args:
            idx: An optional list of ids to retrieve. All if None.
            tensor: If True, audio batch is returned as Tensor. If False, audio batch is returned as ndarray.
            target: If True, target variables are returned in a tuple.
        
        Returns: A (clips, targets) tuple if target is True, clips otherwise.
        """

        return self.get_data(["audio"], idx, tensor=tensor, keep_dict=False)

    def get_videos(self, idx=None, tensor=True, target=True):
        """Get (a selection of) videos (and targets).

        Args:
            idx: An optional list of ids to retrieve. All if None.
            tensor: If True, video batch is returned as Tensor. If False, video batch is returned as ndarray.
            target: If True, target variables are returned in a tuple.
        
        Returns: A (clips, targets) tuple if target is True, clips otherwise.
        """

        return self.get_data(["video"], idx, tensor=tensor, keep_dict=False)

    def get_targets(self, idx=None, tensor=True):
        """ Get (a selection of) texts and targets in dataset at once.

        Args:
            ids: An optional list of ids to retrieve. Retrieve all if None.
            tensor: If True, returns a list of targets, else return an ndarray.

        Returns: An Ndarray.
        """

        idx = _check_idx(idx, len(self))
        targets = self.targets[idx]
        if tensor:
            targets = list(targets)
        return targets


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


class ArrayMultimodalDataset(MultimodalDataset):

    def __init__(self, array_dataset, img_size=400, frac=None, shuffle=False):
        """Create a MultimodalDataset from a PyTorch Dataset.
            See data.from_array_dataset().
        """
    
        if not _is_array_multimodal_dataset(array_dataset):
            raise TypeError("Provided is not a valid multimodal torch Dataset.")

        self.array_dataset = array_dataset
        targets = [array_dataset[i]["target"] for i in range(len(array_dataset))]
        names = [str(i) for i in range(len(targets))]
        self.df = pd.DataFrame({0: names, 1: targets})
        self.base_dir = None

        if frac:
            self.df = self.df.sample(frac=frac)
        if shuffle:
            self.shuffle()

        example_keys = list(array_dataset[0].keys())
        self.av_mods = {m: m in example_keys for m in ["image", "text", "audio", "video"]}
        self.mods = self.av_mods.copy()

        self.h = img_size
        self.classes, self.targets = np.unique(self.df.iloc[:,1], return_inverse=True)
        self.n_cls = len(self.classes)

        self.loaders = {
            "image": self._limage,
            "text": self._ltext,
            "audio": self._laudio,
            "video": self._lvideo,
        }
        self.loaders = {m: self.loaders[m] for m in self.loaders if self.mods[m]}

    def _limage(self, i):
        return self.array_dataset[i]["image"]

    def _ltext(self, i):
        return self.array_dataset[i]["text"]

    def _laudio(self, i):
        return self.array_dataset[i]["audio"]

    def _lvideo(self, i):
        return self.array_dataset[i]["video"]


def from_array_dataset(array_dataset, img_size=400, frac=None, shuffle=False):
    """Create a MultimodalDataset from a PyTorch Dataset.
    Args:
        array_dataset: A array Dataset object which returns a dictionary in __getitem__.
            The returned dictionary should be a subset of {"image": Tensor, "text": String,
            "audio": Tensor, "video": Tensor, "target": int}.
        frac: Choose < 1 to randomly sub-sample loaded dataset.
        shuffle: Randomly shuffle training examples.

    Returns: A TorchMultimodalDataset, subclass of MultimodalDataset.
    """

    return ArrayMultimodalDataset(array_dataset, img_size=img_size, frac=frac, shuffle=shuffle)




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

    def __init__(self, name="caltech-birds", frac=None, shuffle=False, verbose=False, tensor=True):
        source = "https://dl.dropboxusercontent.com/s/u2cyt2c3f1clqfb/caltech-birds.zip"
        
        if not _has_dataset(os.path.join(DATA_DIR, name)):
            _download_dataset(source, name)
        super(CaltechBirds, self).__init__(os.path.join(DATA_DIR, name),
                                        frac=frac, shuffle=shuffle, verbose=verbose, tensor=tensor)


class TastyRecipes(MultimodalDataset):

    def __init__(self, name="tasty-recipes", frac=None, shuffle=False, verbose=False, tensor=True):
        source = "http://dl.dropboxusercontent.com/s/r95zhkuqhvphym3/tasty-recipes.zip"
        if not _has_dataset(os.path.join(DATA_DIR, name)):
            _download_dataset(source, name)  
        super(TastyRecipes, self).__init__(os.path.join(DATA_DIR, name),
                                    frac=frac, shuffle=shuffle, verbose=verbose, tensor=tensor)


class Fakeddit5k(MultimodalDataset):

    def __init__(self, name="fakeddit", class_col=1, frac=None, shuffle=False, verbose=False, tensor=True):
        source = "http://dl.dropboxusercontent.com/s/i1ef9xinebp0dof/fakeddit.zip"
        if not _has_dataset(os.path.join(DATA_DIR, name)):
            _download_dataset(source, name)
        super(Fakeddit5k, self).__init__(os.path.join(DATA_DIR, name), col=class_col,
                                    frac=frac, shuffle=shuffle, verbose=verbose, tensor=tensor)


class Fauxtography(MultimodalDataset):

    def __init__(self, name="fauxtography", frac=None, shuffle=False, verbose=False, tensor=True):
        source = "http://dl.dropboxusercontent.com/s/zl6or9lc7ph836s/fauxtography.zip"
        if not _has_dataset(os.path.join(DATA_DIR, name)):
            _download_dataset(source, name)
        super(Fauxtography, self).__init__(os.path.join(DATA_DIR, name),
                                    frac=frac, shuffle=shuffle, verbose=verbose, tensor=tensor)


class SpotifyMultimodalPop(MultimodalDataset):

    def __init__(self, name="spotify_multimodal_pop", frac=None, shuffle=False, verbose=False, tensor=True):
        source = "https://download1336.mediafire.com/k9dl7neyh2eg/fnd60rzaj0ey4na/spotify_mm_pop.zip"
        if not _has_dataset(os.path.join(DATA_DIR, name)):
            _download_dataset(source, name)
        super(SpotifyMultimodalPop, self).__init__(os.path.join(DATA_DIR, name),
                header=True, n_samples=48000, frac=frac, shuffle=shuffle, verbose=verbose, tensor=tensor)


class SpotifyMultimodalVal(MultimodalDataset):

    def __init__(self, name="spotify_multimodal_val", frac=None, shuffle=False, verbose=False, tensor=True):
        source = "https://download1497.mediafire.com/mku6selaeemg/w04p8oisw4i03i5/spotify_mm_val.zip"
        if not _has_dataset(os.path.join(DATA_DIR, name)):
            _download_dataset(source, name)
        super(SpotifyMultimodalVal, self).__init__(os.path.join(DATA_DIR, name),
                header=True, n_samples=48000, frac=frac, shuffle=shuffle, verbose=verbose, tensor=tensor)
    