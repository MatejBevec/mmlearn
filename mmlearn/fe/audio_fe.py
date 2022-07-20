import os
import sys
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as tf
from torch.utils.data import DataLoader
import torchvision.models as models
import pytorch_pretrained_vit
from sentence_transformers import SentenceTransformer
import openl3
from tqdm import tqdm

from mmlearn.data import MultimodalDataset, from_array_dataset
from mmlearn.util import log_progress, DEVICE, USE_CUDA, SAMPLE_RATE

AUDIO_FE_BATCH_SIZE = 4   # Batch size when extracting features from images


# HELPER FUNCTIONS
def _check_input(clips):
    t = type(clips)
    if not (t in [torch.Tensor, np.ndarray] and len(clips.shape) == 3 and clips.shape[1] <= 2):
        raise TypeError("Audio input must be a 3-dimensional Tensor or ndarray of shape (batches, channels, samples).")
    if t == np.ndarray:
        clips = torch.from_numpy(clips)
    return clips

def _extract_audio_features(fe, dataset, ids=None, verbose=False):
    if not isinstance(dataset, MultimodalDataset):
        try:
            dataset = from_array_dataset(dataset)
        except:
            raise TypeError("'dataset' must be a MultimodalDataset.")
    if not isinstance(fe, AudioExtractor):
        raise TypeError("'fe' must be a AudioExtractor from fe.audio_fe.")

    n = len(ids) if ids is not None else len(dataset)
    dl = DataLoader(dataset, batch_size=AUDIO_FE_BATCH_SIZE, sampler=ids)

    features_list = []
    pbar = tqdm(total=n, desc="Extracting audio features", disable=not verbose)
    for i, batch in enumerate(dl, 0):
        features_list.append(fe(batch["audio"]))
        pbar.update(len(batch["audio"]))
    pbar.close()
    features = np.concatenate(features_list, axis=0)

    labels = None
    if dataset.targets is not None:
        labels_list = [batch["target"] for batch in dl]
        labels = np.concatenate(labels_list, axis=0)

    return features, labels

def _check_output(out, tensor=False):
    assert isinstance(out, np.ndarray)
    if tensor:
        out = torch.from_numpy(out)   
    return out



class AudioExtractor(ABC):
    """Base audio feature extractor class.

    Args:
        tensor: If True, return the encoded batch as `Tensor`, else return `ndarray`.
        
    Attributes:
        modalities: A list of strings denoting modalities which this model operates with.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, clips, train=False, hop_length="auto", win_length="auto", combine="mean"):
        """Extracts features (embeddings) for a batch of audio clips.

        Args:
            clips: A batch of audio clips as a (bsize, 1 or 2, samples) Tensor.
            train: In feature extractor that are trainable, this indicates whether to fit the extractor in this call.
                    train=True is identical to calling fit_transform() and train=False is identical to calling transform()
                    "auto" only fits in first call after init.
                    If not trainable, this argument is irrelevant and defaults to False.
            hop_length: Most audio models embed short windows of the input clip separately.
                    This parameter sets the distance between embedding windows, if possible. Set to "auto" to use defaults.
            win_length: This parameter sets the width of the embedding window, if possible. Set to "auto" to use defaults.
            combine: Whether to combine window embeddings into one vector, representing the entire input clip.
                    Set to "mean" to return the average of window embeddings as a (bsize, dim) matrix.
                    Set to None to perform no combination and return window embeddings as a (bsize, nwindows, dim) matrix.

        Returns:
            A batch of embeddings as an `ndarray` of shape (bsize, d) or (bsize, nwindows, dim).
        """
        pass

    def extract_all(self, dataset, ids=None, verbose=False):
        """Extracts image features (embeddings) for entire dataset.
        
        Args:
            dataset: A MultimodalDataset with `audio` modality.
            ids: The indices of examples to encode. None for all.
        """

        return _extract_audio_features(self, dataset, ids, verbose)

    def fit_transform(self, X, y=None):
        """For sklearn compatibility. (Trains) and calls self."""

        return self.__call__(torch.from_numpy(X), train=True)

    def transform(self, X, y=None):
        """For sklearn compatibility. Calls self (without training)."""

        return self.__call__(torch.from_numpy(X), train=False)

    @property
    def modalities(self):
        return ["audio"]


class OpenL3(AudioExtractor):
    """Feature extractor: OpenL3 deep audio embedding.
    
    OpenL3 is a variation and implementation of the L3-Net self-supervised join audio-image embedding.
    See https://arxiv.org/abs/1705.08168.
    The model is trained to predict whether an image and audio clip are from the same video segment.
    The OpenL3 variation is specialized for music. See https://github.com/marl/openl3.
    """

    def __init__(self, input="mel128", weights="music", dim=512, hop_size=2, combine="mean", tensor=False):
        """
        Args:
            input: Type of spectrogram to use as input features. Options are: TODO
            weights: The pretrained model of choice. Options are: TODO
            dim: Output embedding size.
            hop_size: OpenL3 embeds a 1s clip every hop_size seconds and takes the mean as final output.
            tensor: Output Tensor instead of ndarray. 
        """

        self.model = openl3.models.load_audio_embedding_model(
            input_repr=input,
            content_type=weights,
            embedding_size=dim
        )
        self.hop_size = hop_size
        self.combine = combine
        self.tensor = tensor


    def __call__(self, clips, train=False):
        sr = SAMPLE_RATE
        #sr = sr[0].item()
        clips = _check_input(clips)
        emb_list = []
        for i in range(clips.shape[0]):
            clip = clips[i, :]
            clip = clip.transpose(1, 0).numpy()
            emb, ts = openl3.get_audio_embedding(clip, sr, 
                                model=self.model, hop_size=self.hop_size, verbose=False)
            if self.combine == "mean":
                emb = np.mean(emb, axis=0, keepdims=False)
            emb_list.append(emb)

        out = np.stack(emb_list, axis=0)
        return _check_output(out, self.tensor)


