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
from tqdm import tqdm

from mmlearn.data import MultimodalDataset
from mmlearn.util import log_progress, DEVICE, USE_CUDA

IMG_FE_BATCH_SIZE = 4   # Batch size when extracting features from images


# HELPER FUNCTIONS
def _check_input(imgs):
    t = type(imgs)
    if not (t in [torch.Tensor, np.ndarray] and len(imgs.shape) == 4):
        raise TypeError("Image input must be a 4-dimensional Tensor or ndarray of shape (batches, channels, h, w).")
    if t == np.ndarray:
        imgs = torch.from_numpy(imgs)
    return imgs


def _extract_image_features(fe, dataset, ids=None, verbose=False):
    if not isinstance(dataset, MultimodalDataset):
        raise TypeError("'dataset' must be a MultimodalDataset.")
    if not isinstance(fe, ImageExtractor):
        raise TypeError("'fe' must be a ImageExtractor from fe.image")

    features_list = []
    labels_list = []
    n = len(ids) if ids is not None else len(dataset)
    dl = DataLoader(dataset, batch_size=IMG_FE_BATCH_SIZE, sampler=ids)

    pbar = tqdm(total=n, desc="Extracting image features", disable=not verbose)
    for i, batch in enumerate(dl, 0):
        features_list.append(fe(batch["image"]))
        labels_list.append(batch["target"])
        pbar.update(len(batch["target"]))
    pbar.close()

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return features, labels



class ImageExtractor(ABC):
    """Base image feature extractor class."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, imgs):
        """Extracts features (embeddings) for a batch of images.

        Args:
            imgs: A batch of images as a (bsize, h, w) Tensor.

        Returns:
            A batch of embeddings as a (bsize, d) Ndarray.
        """
        pass

    def extract_all(self, dataset, ids=None, verbose=False):
        """Extracts image features (embeddings) for entire dataset."""

        return _extract_image_features(self, dataset, ids, verbose)

    def fit_transform(self, X, y=None):
        """For sklearn compatibility. (Trains) and calls f.e."""

        return self.__call__(torch.from_numpy(X), train=True)

    def transform(self, X, y=None):
        """For sklearn compatibility. Calls f.e. (without training)."""

        return self.__call__(torch.from_numpy(X), train=False)

    @property
    def modalities(self):
        return ["image"]



class ResNet(ImageExtractor):

    def __init__(self):
        """Feature extractor: pretrained ResNet 152 without clf layer."""
        self.model = models.resnet152(pretrained=True)
        self.model.fc = nn.Identity()
        self.model.eval().to(DEVICE)

    def __call__(self, imgs, train=False):
        imgs = _check_input(imgs)
        return self.model(imgs.to(DEVICE)).detach().cpu().numpy()

class InceptionV3(ImageExtractor):

    def __init__(self):
        """Feature extractor: pretrained ResNet 152 without clf layer."""
        self.model = models.inception_v3(pretrained=True)
        self.model.fc = nn.Identity()
        self.model.eval().to(DEVICE)

    def __call__(self, imgs, train=False):
        imgs = _check_input(imgs)
        return self.model(imgs.to(DEVICE)).detach().cpu().numpy()

class MobileNetV3(ImageExtractor):

    def __init__(self):
        """Feature extractor: pretrained ResNet MobileNet V3 Large without clf layer."""
        self.model = models.mobilenet_v3_large(pretrained=True)
        self.model.classifier = nn.Identity()
        self.model.eval().to(DEVICE)

    def __call__(self, imgs, train=False):
        imgs = _check_input(imgs)
        return self.model(imgs.to(DEVICE)).detach().cpu().numpy()

class EfficientNet(ImageExtractor):

    def __init__(self):
        """Feature extractor: pretrained EfficientNet B7 without clf layer."""
        self.model = models.efficientnet_b7(pretrained=True)
        self.model.classifier = nn.Identity()
        self.model.eval().to(DEVICE)
    
    def __call__(self, imgs, train=False):
        imgs = _check_input(imgs)
        return self.model(imgs.to(DEVICE)).detach().cpu().numpy()

class ViT(ImageExtractor):

    def __init__(self):
        """Feature extractor: pretrained Vision Transformer without clf layer."""
        self.model = pytorch_pretrained_vit.ViT('B_16_imagenet1k', pretrained=True)
        self.model.norm = nn.Identity()
        self.model.fc = nn.Identity()
        self.model.eval().to(DEVICE)
        self.tf = tf.Compose([tf.Resize((384,384)), tf.Normalize(0.5, 0.5)])

    def __call__(self, imgs, train=False):
        imgs = _check_input(imgs)
        return self.model(self.tf(imgs).to(DEVICE)).detach().cpu().numpy()

class ImageCLIP(ImageExtractor):

    def __init__(self):
        """Feature extractor: CLIP joint image-text embedding model."""
        self.fe = SentenceTransformer("clip-ViT-B-32")
    
    def __call__(self, imgs, train=False):
        imgs = _check_input(imgs)
        return self.fe.encode(imgs)




if __name__ == "__main__":

    # dataset = MultimodalDataset("data/caltech-birds")
    # model = MobileNetV3()

    # imgs, text, label = dataset[0]
    # y = model.predict(imgs.unsqueeze(0))
    # print(y.shape)
    pass
