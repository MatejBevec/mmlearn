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

from mmlearn.data import MultimodalDataset, from_array_dataset
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
        try:
            dataset = from_array_dataset(dataset)
        except:
            raise TypeError("'dataset' must be a MultimodalDataset.")
    if not isinstance(fe, ImageExtractor):
        raise TypeError("'fe' must be a ImageExtractor from fe.image")

    n = len(ids) if ids is not None else len(dataset)
    dl = DataLoader(dataset, batch_size=IMG_FE_BATCH_SIZE, sampler=ids)

    features_list = []
    pbar = tqdm(total=n, desc="Extracting image features", disable=not verbose)
    for i, batch in enumerate(dl, 0):
        features_list.append(fe(batch["image"]))
        pbar.update(len(batch["image"]))
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



class ImageExtractor(ABC):
    """Base image feature extractor (image embedding) class.
    
    Args:
        tensor: If True, return the encoded batch as `Tensor`, else return `ndarray`.

    Attributes:
        modalities: A list of strings denoting modalities which this model operates with.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, imgs, train="auto", train_data=None):
        """Extracts features (embeddings) for a batch of images.

        Args:
            imgs: A batch of images as `Tensor` of shape (bsize, h, w) .
            train: In feature extractor that are trainable, this indicates whether to fit the extractor in this call.
                    train=True is identical to calling fit_transform() and train=False is identical to calling transform()
                    "auto" only fits in first call after init.
                    If not trainable, this argument is irrelevant and defaults to False.
            train_data: The custom training data, if trainable fe supports it.

        Returns:
            A batch of embeddings as an `ndarray` of shape (bsize, d).
        """
        pass

    def extract_all(self, dataset, ids=None, verbose=False):
        """Extracts image features (embeddings) for entire dataset.
        
        Args:
            dataset: A MultimodalDataset with `image` modality.
            ids: The indices of examples to encode. None for all.
        """

        return _extract_image_features(self, dataset, ids, verbose)

    def fit_transform(self, X, y=None):
        """For sklearn compatibility. (Trains) and calls self"""

        return self.__call__(torch.from_numpy(X), train=True)

    def transform(self, X, y=None):
        """For sklearn compatibility. Calls self (without training)."""

        return self.__call__(torch.from_numpy(X), train=False)

    @property
    def modalities(self):
        return ["image"]



class ResNet(ImageExtractor):
    """Feature extractor: pretrained ResNet 152 without clf layer."""

    def __init__(self, tensor=False):
        self.model = models.resnet152(pretrained=True)
        self.model.fc = nn.Identity()
        self.model.eval().to(DEVICE)
        self.tensor = tensor

    def __call__(self, imgs, train=False, train_data=None):
        imgs = _check_input(imgs)
        out = self.model(imgs.to(DEVICE)).detach().cpu().numpy()
        return _check_output(out, self.tensor)

class InceptionV3(ImageExtractor):
    """Feature extractor: pretrained ResNet 152 without clf layer."""

    def __init__(self, tensor=False):
        self.model = models.inception_v3(pretrained=True)
        self.model.fc = nn.Identity()
        self.model.eval().to(DEVICE)
        self.tensor = tensor

    def __call__(self, imgs, train=False, train_data=None):
        imgs = _check_input(imgs)
        out = self.model(imgs.to(DEVICE)).detach().cpu().numpy()
        return _check_output(out, self.tensor)

class MobileNetV3(ImageExtractor):
    """Feature extractor: pretrained ResNet MobileNet V3 Large without clf layer."""

    def __init__(self, tensor=False):
        self.model = models.mobilenet_v3_large(pretrained=True)
        self.model.classifier = nn.Identity()
        self.model.eval().to(DEVICE)
        self.tensor = tensor

    def __call__(self, imgs, train=False, train_data=None):
        imgs = _check_input(imgs)
        out = self.model(imgs.to(DEVICE)).detach().cpu().numpy()
        return _check_output(out, self.tensor)

class EfficientNet(ImageExtractor):
    """Feature extractor: pretrained EfficientNet B7 without clf layer."""

    def __init__(self, tensor=False):
        self.model = models.efficientnet_b7(pretrained=True)
        self.model.classifier = nn.Identity()
        self.model.eval().to(DEVICE)
        self.tensor = tensor
    
    def __call__(self, imgs, train=False, train_data=None):
        imgs = _check_input(imgs)
        out = self.model(imgs.to(DEVICE)).detach().cpu().numpy()
        return _check_output(out, self.tensor)

class ViT(ImageExtractor):
    """Feature extractor: pretrained Vision Transformer without clf layer."""

    def __init__(self, tensor=False):
        self.model = pytorch_pretrained_vit.ViT('B_16_imagenet1k', pretrained=True)
        self.model.norm = nn.Identity()
        self.model.fc = nn.Identity()
        self.model.eval().to(DEVICE)
        self.tf = tf.Compose([tf.Resize((384,384)), tf.Normalize(0.5, 0.5)])
        self.tensor = tensor

    def __call__(self, imgs, train=False, train_data=None):
        imgs = _check_input(imgs)
        out = self.model(self.tf(imgs).to(DEVICE)).detach().cpu().numpy()
        return _check_output(out, self.tensor)

class CLIP(ImageExtractor):
    """Feature extractor: CLIP joint image-text embedding model."""

    def __init__(self, tensor=False):
        self.fe = SentenceTransformer("clip-ViT-B-32")
        self.tensor = tensor
    
    def __call__(self, imgs, train=False, train_data=None):
        imgs = _check_input(imgs)
        out = self.fe.encode(imgs)
        return _check_output(out, self.tensor)




if __name__ == "__main__":

    # dataset = MultimodalDataset("data/caltech-birds")
    # model = MobileNetV3()

    # imgs, text, label = dataset[0]
    # y = model.predict(imgs.unsqueeze(0))
    # print(y.shape)
    pass
