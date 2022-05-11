from abc import ABC, abstractmethod
from typing import Iterable
import logging
import itertools

import numpy as np
import pandas as pd
import scipy
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier
import tpot
import sklearn.metrics
import torch
from torch.utils.data import DataLoader, TensorDataset

from mmlearn.data import MultimodalDataset
from mmlearn.util import log_progress, DEVICE

TRAIN_BATCH_SIZE = 32   # Default batch size for training ClsModel-s
PRED_BATCH_SIZE = 16    # Default batch size for prediction using ClsModel-s

# HELPER METHODS

def prepare_input(dataset, ids):
    if not isinstance(dataset, MultimodalDataset):
        raise TypeError("'dataset' must be a MultimodalDataset.")
    if not (isinstance(ids, Iterable) and len(ids) > 0 or ids is None):
        raise TypeError("IDs must be an iterable of integer indices or None")
    if ids is None:
        ids = np.arange(len(dataset))
    return dataset, ids


def get_classifier(clf):
    """Returns one of the pre-configured sklearn-style classifiers.
        Used to parse the "clf" argument in ClsModel constructors when a shorthand string is provided.

    Args:
        clf: One of the following string abbreviations, referring to the type of classifier to return:
                "svm": sklearn.svm.LinearSVC() with default settings              
                "svm_best": Fit LinearSVC models with different reguralization params (C = [0.1:500]) and pick best performer.
                "lr": sklearn.linear_model.LogisticRegression() with default settings
                "lr_best": Fit LogisticRegression models with different reguralization params (C = [0.1:500]) and pick best performer.
                "rf": sklearn.ensemble.RandomForestClassifier() with default settings
                "nn": sklearn.neural_network.MLPClassifier() with default settings
                "tpot": sklearn.tpot.TPOTClassifier(generations=5, population_size=50,
                                        verbosity=2, random_state=42, max_time_mins=120)
    
    """

    if clf == "svm_best":
        clf = AutoLinearSVM()
    elif clf == "svm":
        clf = LinearSVC()
    elif clf == "lr_best":
        clf = AutoLogisticRegression()
    elif clf == "lr":
        clf = LogisticRegression()
    elif clf == "rf":
        clf = RandomForestClassifier()
    elif clf == "nn":
        clf = MLPClassifier()
    elif clf == "tpot":
        clf = tpot.TPOTClassifier(generations=5,
                            population_size=50,
                            verbosity=2,
                            random_state=42,
                            max_time_mins=120)
    elif not isinstance(clf, BaseEstimator):
        raise TypeError("'clf' must be one of the choice strings or sklearn estimator")
    return clf


def train_torch_nn(model, dataset, loss_func, optimizer,
                                    train_ids=None, batch_size=32, epochs=10, lr=1e-3):
    """Trains a pytorch Module using given pytorch Dataset and training parameters."""

    model = model.to(DEVICE)
    dl = DataLoader(dataset, batch_size=batch_size, sampler=train_ids)
    # 'dataset' must be a pytorch dataset 

    for epoch in range(epochs):
        batch_loss = 0
        for i, (features, labels) in enumerate(dl, 0):
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            pred = model(features) 
            loss = loss_func(pred, labels.long())
            loss.backward()
            optimizer.step()
            batch_loss += loss
            if i%10 == 0:
                #print(f"{i} batches done. Loss = {batch_loss/10}")
                batch_loss = 0

def predict_torch_nn(model, dataset, test_ids=None):
    """Performs the inference step on given pytorch Module and Dataset."""

    model = model.to(DEVICE).eval()
    dl = DataLoader(dataset, batch_size=PRED_BATCH_SIZE, sampler=test_ids)
    pred_list = []
    for i, (features, labels) in enumerate(dl, 0):
        features = features.to(DEVICE)
        out = model(features).cpu().detach().numpy()
        pred_list.append(out.argmax(axis=1))
    return np.concatenate(pred_list, axis=0).astype(int)


# HELPER SKLEARN-STYLE END CLASSIFIERS (after fe)

# class NeuralClf(BaseEstimator):
    
#     def __init__(self, model, loss_func, optimizer, batch_size=32, epochs=10, lr=1e-3):
#         self.model = model
#         pass

#     def fit(self, X, y):
#         self.model = train_torch_nn(self.model, TensorDataset(X), None, self.loss_func, self.optimizer, ...)
#         pass

#     def predict(self, X):
#         return predict_torch_nn(self.model, TensorDataset(X), None)
#         pass


class AutoSkClf(BaseEstimator):

    def __init__(self):
        self.param_sets = None # dicts of parameters
        self.models = None # instantiated sk estimators with such parameters

    def fit(self, X, y):
        split = int(len(y)*0.7)
        tr_ft, tr_lbl = X[:split], y[:split]
        test_ft, test_lbl = X[split:], y[split:]
        scores = []
        for i, m in enumerate(self.models):
            m.fit(tr_ft, tr_lbl)
            pred = m.predict(test_ft)
            score = sklearn.metrics.f1_score(test_lbl, pred, average="macro")
            log_progress(f"{self.param_sets[i]}, score = {score:.3f}", color="white")
            scores.append(score)
        best = np.argmax(scores)
        log_progress(f"Picked {self.param_sets[best]}")
        self.model = self.models[best]
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class AutoLinearSVM(AutoSkClf):

    def __init__(self):
        C_range = [0.1, 0.5, 1, 5, 10, 20, 50, 100, 500]
        self.param_sets = [{"C": C} for C in C_range]
        self.models = [LinearSVC(C=C) for C in C_range]

class AutoLogisticRegression(AutoSkClf):

    def __init__(self):
        C_range = [0.1, 0.5, 1, 5, 10, 20, 50, 100, 500]
        self.param_sets = [{"C": C} for C in C_range]
        self.models = [LogisticRegression(C=C) for C in C_range]

# BASE END-TO-END CLASSIFIERS

class ClsModel(ABC):
    """Base classification model class.
    
    A common interface for all models - text-only, image-only and multimodal.
    Any ClsModel can be trained on any MultimodalDataset.
    """

    @abstractmethod
    def __init__(self):
        """Initialize model with optional settings."""
        pass

    @abstractmethod
    def train(self, dataset, train_ids):
        """Train model on given dataset.
        
        Args:
            dataset: A MultimodalDataset instance.
            train_ids: Indices representing the training set or None (whole dataset).
        """
        pass

    @abstractmethod
    def predict(self, dataset, test_ids):
        """Predict target variables on given dataset.
        
        Args:
            dataset: A MultimodalDataset instance. Usually same as for train().
            test_ids: Indices representing the testing set or None (whole dataset).

        Returns:
            Predictions as a (len(test_ids), ) Ndarray.
        """
        pass


class RandomClassifier(ClsModel):
    """Predicts random class for every test instance."""

    def __init__(self, random_state=42):
        self.seed = random_state
    
    def train(self, dataset, train_ids=None):
        dataset, train_ids = prepare_input(dataset, train_ids)
        log_progress(f"Training {type(self).__name__} model...")
        self.dataset = dataset
        self.n_cls = len(dataset.classes)
    
    def predict(self, dataset, test_ids=None):
        dataset, test_ids = prepare_input(dataset, test_ids)
        return np.random.RandomState(seed=self.seed).randint(0, self.n_cls, len(test_ids))


class MajorityClassifier(ClsModel):
    """Predicts the most frequent class in training set for every test instance."""

    def __init__(self):
        pass

    def train(self, dataset, train_ids=None):
        dataset, train_ids = prepare_input(dataset, train_ids)
        log_progress(f"Training {type(self).__name__} model...")
        _, labels = dataset.get_texts()
        self.mode_cls = scipy.stats.mode(labels[train_ids])[0]

    def predict(self, dataset, test_ids=None):
        dataset, test_ids = prepare_input(dataset, test_ids)
        return np.repeat(self.mode_cls, len(test_ids))
