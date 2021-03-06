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
    """Audio feature extractor + a scikit-learn classifier.
        A shorthand for UnimodalSkClassifier(fe=audio_fe.OpenL3()).
        See superclass for details.
    
    Args:
        fe: A feature extractor model from 'fe.audio_fe'. Default is OpenL3.
        clf: The classifier to use.
    """

    def __init__(self, fe="default", clf="svm_best", verbose=False):
        fe = audio_fe.OpenL3() if fe == "default" else fe
        super().__init__(fe=fe, clf=clf, verbose=verbose)