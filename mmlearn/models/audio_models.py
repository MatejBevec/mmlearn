import os
import sys
from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as tf
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
import pytorch_pretrained_vit
from sentence_transformers import SentenceTransformer
import sklearn
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import scipy.special

from mmlearn.models.base_models import PredictionModel, UnimodalSkClassifier
from mmlearn.models.base_models import prepare_input, check_predicts_proba, get_classifier
from mmlearn.models import base_models
from mmlearn.fe import audio_fe
from mmlearn.util import log_progress, DEVICE, REG_PARAM_RANGE


class AudioSkClassifier(UnimodalSkClassifier):

    def __init__(self, fe="default", clf="svm_best", verbose=False):
        """Audio feature extractor + a scikit-learn classifier.
            A shorthand for UnimodalSkClassifier(fe=audio_fe.OpenL3()).
        
        Args:
            fe: A feature extractor model from 'fe.audio_fe'. Default is OpenL3.
            clf: The classifier to use. 'svm', 'lr', 'rf' or an instance of any sklearn classifer.
        """
        
        fe = audio_fe.OpenL3() if fe == "default" else fe
        super().__init__(fe=fe, clf=clf, verbose=verbose)