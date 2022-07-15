from multiprocessing.spawn import prepare
import os
import sys
import time
from abc import ABC, abstractmethod
from unittest import TextTestResult

import numpy as np
import pandas as pd
import scipy
import sklearn
import sklearn.svm
import sklearn.linear_model
import sklearn.ensemble
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
import torch

from simpletransformers.classification import ClassificationModel
from tpot import TPOTClassifier
from sentence_transformers import SentenceTransformer

from mmlearn.models.base_models import PredictionModel, UnimodalSkClassifier
from mmlearn.models.base_models import prepare_input, check_predicts_proba, get_classifier
from mmlearn.fe import text_fe as textfe
from mmlearn.fe import image_fe as imgfe
from mmlearn.models import text_models
from mmlearn.models import image_models
from mmlearn.util import log_progress, DEVICE, USE_CUDA, REG_PARAM_RANGE


class LateFusion(PredictionModel):

    def __init__(self, image_model="default", text_model="default", combine="max", verbose=False):
        """
        Multimodal late fusion model. Combine prediction of an image model and a text model.

        Args:
            image_model: An image classifier from models.image. Default is ImageSkClassifier.
            text_model: A text classifier from models.text. Default is TextSkClassifier.
            combine: Method for combining predictions. "max", "sum", or "stack".
        """

        self.modalities = ["image", "text"]
        self.verbose = verbose

        self.image_model = image_models.ImageSkClassifier() if image_model == "default" else image_model
        self.text_model = text_models.TextSkClassifier(clf="lr") if text_model == "default" else text_model

    def train(self, dataset, train_ids=None):
        dataset, train_ids = prepare_input(dataset, train_ids, self)
        log_progress(f"Training {type(self).__name__} model...")

        self.image_model.train(dataset, train_ids)
        self.text_model.train(dataset, train_ids)
        _, labels = dataset.get_texts(train_ids)

        # TODO: Stacking
        if self.combine == "stack":
            pass

    def predict(self, dataset, test_ids=None):
        dataset, test_ids = prepare_input(dataset, test_ids, self)

        # TODO: All model.model-s must have a predict_proba option?
        # TODO: Combine predictions
        pass

    def predict_proba(self, dataset, test_ids=None):
        pass
 

class NaiveEarlyFusion(PredictionModel):

    def __init__(self, image_fe="default", text_fe="default", clf="svm", verbose=False):
        """Naive multimodal early fusion model.
            Image and text features are extracted, concatenated and fed to a classifier.

        Args:
            image_fe: An image feature extractor from fe.image. Default is MobileNetV3.
            text_fe: A text feature extractor from fe.text. Default is SentenceBERT.
            clf: Classifier - string shorthand or a sklearn estimator instance.
                See [todo] for list of shorthands.
        """

        self.modalities = ["image", "text"]
        self.verbose = verbose

        self.image_fe = imgfe.MobileNetV3() if image_fe == "default" else image_fe
        self.text_fe = textfe.SentenceBERT() if text_fe == "default" else text_fe
        self.clf = clf

    def _extract_features(self, dataset, ids):
        log_progress(f"Extracting features...", color="white", verbose=self.verbose)
        image_ft, labels = self.image_fe.extract_all(dataset, ids, verbose=self.verbose)
        text_ft, _ = self.text_fe.extract_all(dataset, ids, verbose=self.verbose)
        features = np.concatenate([image_ft, text_ft], axis=1)
        return features, labels

    def train(self, dataset, train_ids=None):
        dataset, train_ids = prepare_input(dataset, train_ids, self)
        self.model = get_classifier(self.clf)
        log_progress(f"Training early fusion model: \
                    ({type(self.image_fe).__name__} + {type(self.text_fe).__name__} \
                     -> {type(self.model).__name__})", verbose=self.verbose)
        
        features, labels = self._extract_features(dataset, train_ids)
        self.model.fit(features, labels)

    def predict(self, dataset, test_ids=None):
        features, _ = self._extract_features(dataset, test_ids)
        return self.model.predict(features)
    
    def predict_proba(self, dataset, test_ids=None):
        features, _ = self._extract_features(dataset, test_ids)
        check_predicts_proba(self.model)
        return self.model.predict_proba(features)