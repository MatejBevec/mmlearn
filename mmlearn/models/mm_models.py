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
    """
    Multimodal late fusion model. Combine predictions (probabilities) of an image model and a text model.
    The image and text model must implement predict_proba.

    Args:
        image_model: An image classifier from models.image. Default is ImageSkClassifier.
        text_model: A text classifier from models.text. Default is TextSkClassifier.
        combine: Method for combining predictions. "max", "sum", or "stack".
                "stack" trains a meta classifier to predict targets from output probabilites.
        stacking_clf: The sklearn-style classifier to use for final probability stacking.
                        Only relevant if combine == "stack". Must implement predict_proba.
    """

    def __init__(self, image_model="default", text_model="default", combine="max", stacking_clf="lr", verbose=False):
        self.modalities = ["image", "text"]
        self.verbose = verbose
        self.combine = combine
        self.stacking_clf = stacking_clf

        self.image_model = image_models.ImageSkClassifier(clf="lr_best") if image_model == "default" else image_model
        self.text_model = text_models.TextSkClassifier(clf="lr_best") if text_model == "default" else text_model
        self.image_model.verbose = self.verbose
        self.text_model.verbose = self.verbose

    def train(self, dataset, train_ids=None):
        dataset, train_ids = prepare_input(dataset, train_ids, self)
        log_progress(f"Training {type(self).__name__} ({self.combine}) model...")

        self.image_model.train(dataset, train_ids)
        self.text_model.train(dataset, train_ids)
        targets = dataset.get_targets(train_ids, tensor=False)

        # TODO: Stacking
        if self.combine in ["stack", "stacking"]:
            self.stacking_model = get_classifier(self.stacking_clf, verbose=self.verbose)
            
            image_proba = self.image_model.predict_proba(dataset, train_ids)
            text_proba = self.text_model.predict_proba(dataset, train_ids)
            #stack = np.concatenate([image_proba, text_proba], axis=1)
            stack = np.stack([image_proba, text_proba], axis=2).reshape((image_proba.shape[0], image_proba.shape[1]*2))

            self.stacking_model.fit(stack, targets)

    def _predict(self, dataset, test_ids=None):
        dataset, test_ids = prepare_input(dataset, test_ids, self)

        image_proba = self.image_model.predict_proba(dataset, test_ids)
        text_proba = self.text_model.predict_proba(dataset, test_ids)
        all_proba = np.stack([image_proba, text_proba], axis=0)

        if self.combine == "max":
            combined = np.max(all_proba, axis=0)
        if self.combine == "sum":
            combined = np.sum(all_proba, axis=0)
        if self.combine in ["stack", "stacking"]:
            stack = np.stack([image_proba, text_proba], axis=2).reshape((image_proba.shape[0], image_proba.shape[1]*2))
            combined = self.stacking_model.predict_proba(stack)

        combined_proba = scipy.special.softmax(combined, axis=1)
        return combined_proba

    def predict(self, dataset, test_ids=None):
        proba = self._predict(dataset, test_ids)
        return np.argmax(proba, axis=1)

    def predict_proba(self, dataset, test_ids=None):
        return self._predict(dataset, test_ids)
 

class NaiveEarlyFusion(PredictionModel):
    """Naive multimodal early fusion model.
        Image and text features are extracted, concatenated and fed to a classifier.

    Args:
        image_fe: An ImageFeatureExtractor from fe.image. Default is MobileNetV3.
        text_fe: A TextFeatureExtractor from fe.text. Default is SentenceBERT.
        clf: Classifier - string shorthand or a sklearn estimator instance.
            See :meth:`mmlearn.models.base_models.get:classifier` for list of shorthands.
    """

    def __init__(self, image_fe="default", text_fe="default", clf="svm", verbose=False):
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
        log_progress(f"Training early fusion model:\n" \
                    f"({type(self.image_fe).__name__} + {type(self.text_fe).__name__}" \
                    f"-> {type(self.model).__name__})", verbose=self.verbose)
        
        features, labels = self._extract_features(dataset, train_ids)
        self.model.fit(features, labels)

    def predict(self, dataset, test_ids=None):
        features, _ = self._extract_features(dataset, test_ids)
        return self.model.predict(features)
    
    def predict_proba(self, dataset, test_ids=None):
        features, _ = self._extract_features(dataset, test_ids)
        check_predicts_proba(self.model)
        return self.model.predict_proba(features)