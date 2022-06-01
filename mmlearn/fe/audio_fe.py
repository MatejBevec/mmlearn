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

from mmlearn.data import MultimodalDataset
from mmlearn.util import log_progress, DEVICE, USE_CUDA

AUDIO_FE_BATCH_SIZE = 4   # Batch size when extracting features from images


# HELPER FUNCTIONS
def _check_input(x):
    if not (type(x) is torch.Tensor and len(x.shape) == 3 and x.shape[1] <= 2):
        raise TypeError("Audio input must be a 3-dimensional Tensor of shape (batches, channels, samples).")

def _extract_audio_features(fe, dataset, ids=None, verbose=False):
    if not isinstance(dataset, MultimodalDataset):
        raise TypeError("'dataset' must be a MultimodalDataset.")
    if not isinstance(fe, AudioExtractor):
        raise TypeError("'fe' must be a AudioExtractor from fe.audio_fe.")

    features_list = []
    labels_list = []
    n = len(ids) if ids is not None else len(dataset)
    dl = DataLoader(dataset, batch_size=AUDIO_FE_BATCH_SIZE, sampler=ids)

    pbar = tqdm(total=n, desc="Extracting audio features", disable=not verbose)
    for i, batch in enumerate(dl, 0):
        features_list.append(fe(batch["audio"]))
        labels_list.append(batch["target"])
        pbar.update(len(batch["target"]))
    pbar.close()

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return features, labels



class AudioExtractor(ABC):
    """Base audio feature extractor class."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, imgs):
        """Extracts features (embeddings) for a batch of audio clips.

        Args:
            clips: A batch of audio clips as a (bsize, 1 or 2, samples) Tensor.

        Returns:
            A batch of embeddings as a (bsize, d) Ndarray.
        """
        pass

    def extract_all(self, dataset, ids=None, verbose=False):
        """Extracts image features (embeddings) for entire dataset."""

        return _extract_audio_features(self, dataset, ids, verbose)


class OpenL3(AudioExtractor):
    """Audio feature extractor: OpenL3 deep audio embedding.
    
    OpenL3 is a variation and implementation of the L3-Net self-supervised join audio-image embedding.
    See https://arxiv.org/abs/1705.08168.
    The model is trained to predict whether an image and audio clip are from the same video segment.
    The OpenL3 variation is specialized for music and trained accordingly. See https://github.com/marl/openl3.
    """

    def __init__(self, input="mel128", weights="music", dim=512, hop_size=2):
        """
        Args:
            input: Type of spectrogram to use as input features. Options are: TODO
            weights: The pretrained model of choice. Options are: TODO
            dim: Output embedding size.
            hop_size: OpenL3 embeds a 1s clip every hop_size seconds and takes the mean as final output.
        """

        self.model = openl3.models.load_audio_embedding_model(
            input_repr=input,
            content_type=weights,
            embedding_size=dim
        )
        self.hop_size = hop_size


    def __call__(self, batch):
        clips, sr = batch
        sr = sr[0].item()
        _check_input(clips)
        emb_list = []
        for i in range(clips.shape[0]):
            clip = clips[i, :]
            clip = clip.transpose(1, 0).numpy()
            emb_batch, ts = openl3.get_audio_embedding(clip, sr, 
                                model=self.model, hop_size=self.hop_size, verbose=False)
            emb = torch.mean(torch.from_numpy(emb_batch), dim=0, keepdim=False)
            emb_list.append(emb)

        return torch.stack(emb_list, dim=0)


