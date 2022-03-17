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

from mmlearn.models.base import ClsModel, prepare_input, get_classifier
from mmlearn.fe import text as textfe
from mmlearn.fe import image as imgfe
from mmlearn.models import text
from mmlearn.models import image
from mmlearn.util import log_progress, DEVICE, USE_CUDA, REG_PARAM_RANGE


class LateFusion(ClsModel):

    def __init__(self, image_model=image.ImageSkClassifier(),
                        text_model=text.TextSkClassifier(clf="lr"),
                        combine="max"):
        """
        Multimodal late fusion model. Combine prediction of an image model and a text model.

        Args:
            image_model: An image classifier from image.
            text_model: A text classifier from text.
            combine: Method for combining predictions. "max", "sum", or "stack".
        """

        self.image_model = image_model
        self.text_model = text_model

    def train(self, dataset, train_ids):
        dataset, train_ids = prepare_input(dataset, train_ids)
        log_progress(f"Training {type(self).__name__} model...")

        self.image_model.train(dataset, train_ids)
        self.text_model.train(dataset, train_ids)
        _, labels = dataset.get_texts(train_ids)

        # TODO: Stacking
        if self.combine == "stack":
            pass

    def predict(self, dataset, test_ids):
        dataset, test_ids = prepare_input(dataset, test_ids)

        # TODO: All model.model-s must have a predict_proba option?
        # TODO: Combine predictions
        pass
 

class NaiveEarlyFusion(ClsModel):

    def __init__(self, image_fe=imgfe.MobileNetV3(), text_fe=textfe.SentenceBERT(), clf="svm"):
        """Naive multimodal early fusion model.
            Image and text features are extracted, concatenated and fed to a classifier.

        Args:
            image_fe: An image feature extractor from imgfe.
            text_fe: A text feature extractor from textfe.
            clf: Classifier - string shorthand or a sklearn estimator instance.
                See [todo] for list of shorthands.
        """

        self.image_fe = image_fe
        self.text_fe = text_fe
        self.model = get_classifier(clf)
        print(self.model)

    def train(self, dataset, train_ids):
        dataset, train_ids = prepare_input(dataset, train_ids)
        log_progress(f"Training early fusion model: \
                    ({type(self.image_fe).__name__} + {type(self.text_fe).__name__} \
                     -> {type(self.model).__name__})")

        image_ft, labels = self.image_fe.extract_all(dataset, train_ids)
        text_ft, _ = self.text_fe.extract_all(dataset, train_ids)
        features = np.concatenate([image_ft, text_ft], axis=1)

        self.model.fit(features, labels)

    def predict(self, dataset, test_ids):
        dataset, test_ids = prepare_input(dataset, test_ids)
        image_ft, _ = self.image_fe.extract_all(dataset, test_ids)
        text_ft, _ = self.text_fe.extract_all(dataset, test_ids)
        return self.model.predict(np.concatenate([image_ft, text_ft], axis=1))  