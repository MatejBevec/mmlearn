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

from mmlearn.models.base_models import ClsModel, prepare_input, check_predicts_proba, get_classifier
from mmlearn.models import base_models
from mmlearn.fe import audio_fe
from mmlearn.util import log_progress, DEVICE, REG_PARAM_RANGE



class AudioSkClassifier():

    def __init__(self, fe="default", clf="svm_best", best_reg=True, verbose=False):
        """Audio feature extractor + a scikit-learn classifier.
        
        Args:
            fe: A feature extractor model from 'fe.audio_fe'. Default is OpenL3.
            clf: The classifier to use. 'svm', 'lr', 'rf' or an instance of any sklearn classifer.
        """
        
        self.fe = audio_fe.OpenL3() if fe == "default" else fe
        self.model = get_classifier(clf, verbose=verbose)
        self.modalities = ["audio"]
        self.verbose = verbose

    def train(self, dataset, train_ids):
        dataset, train_ids = prepare_input(dataset, train_ids)
        log_progress(f"Training {type(self.fe).__name__} + {type(self.model).__name__} classifier model...",
                        verbose=self.verbose)

        features, labels = self.fe.extract_all(dataset, train_ids, verbose=self.verbose)

        self.model.fit(features, labels)

    def predict(self, dataset, test_ids):
        dataset, test_ids = prepare_input(dataset, test_ids)
        features, labels = self.fe.extract_all(dataset, test_ids, verbose=self.verbose)
        return self.model.predict(features)

    def predict_proba(self, dataset, test_ids):
        dataset, test_ids = prepare_input(dataset, test_ids)
        features, labels = self.fe.extract_all(dataset, test_ids, verbose=self.verbose)
        check_predicts_proba(self.model)
        return self.model.predict_proba(features)
