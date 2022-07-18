from abc import ABC, abstractmethod
from multiprocessing.sharedctypes import Value
from typing import Iterable
import logging
import itertools

import numpy as np
import pandas as pd
import scipy.special
import scipy.stats
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier
import tpot
import sklearn.metrics
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

from mmlearn.data import MultimodalDataset, from_array_dataset
from mmlearn.util import log_progress, DEVICE

TRAIN_BATCH_SIZE = 32   # Default batch size for training PredictionModel-s
PRED_BATCH_SIZE = 16    # Default batch size for prediction using PredictionModel-s

# HELPER METHODS

def prepare_input(dataset, ids, model):
    """Checks and prepares input to a PredictionModel's train and predict methods.
        Returns the full dataset, not sampled by ids!
    """

    if not isinstance(dataset, MultimodalDataset):
        try:
            dataset = from_array_dataset(dataset)
        except:
            raise TypeError(f"'dataset' must be a MultimodalDataset or convertable to one, but is {type(dataset)}")
    if not (isinstance(ids, Iterable) and len(ids) > 0 or ids is None):
        raise TypeError("IDs must be an iterable of integer indices or None")
    if not set(model.modalities).issubset(dataset.modalities):
        raise TypeError(f"The dataset {type(dataset).__name__}: {dataset.modalities} does not provide"\
                        "modalities required by the model {type(model).__name__}: {model.modalities}.")
    if ids is None:
        ids = np.arange(len(dataset))
    return dataset, ids


def predicts_proba(model):
    return callable(getattr(model, "predict_proba", None))

def check_predicts_proba(model):
    if not predicts_proba(model):
            raise NotImplementedError(f"The chosen classifer ({type(model).__name__}) does not implement predict_proba.")
    


def get_classifier(clf, verbose=False):
    """Returns one of the pre-configured sklearn-style classifiers.
        Used to parse the "clf" argument in PredictionModel constructors when a shorthand string is provided.

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
        clf = AutoLinearSVM(verbose=verbose)
    elif clf == "svm":
        clf = LinearSVC(max_iter=5000)
    elif clf == "lr_best":
        clf = AutoLogisticRegression(verbose=verbose)
    elif clf == "lr":
        clf = LogisticRegression(max_iter=1000)
    elif clf == "rf":
        clf = RandomForestClassifier()
    elif clf == "nn":
        clf = MLPClassifier()
    elif clf == "tpot":
        clf = tpot.TPOTClassifier(generations=5,
                            population_size=50,
                            verbosity=2 if verbose else 0,
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
    prob_list = []
    for i, (features, labels) in enumerate(dl, 0):
        features = features.to(DEVICE)
        out = model(features).cpu().detach().numpy()
        pred_list.append(out.argmax(axis=1))
        prob_list.append(out)
    pred = np.concatenate(pred_list, axis=0).astype(int)
    prob = scipy.special.softmax(np.concatenate(prob_list, axis=0), axis=1)
    return pred, prob


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

    def __init__(self, verbose=False):
        self.param_sets = None # dicts of parameters
        self.models = None # instantiated sk estimators with such parameters
        self.verbose = verbose

    def fit(self, X, y):
        split = int(len(y)*0.7)
        tr_ft, tr_lbl = X[:split], y[:split]
        test_ft, test_lbl = X[split:], y[split:]
        scores = []
        for i, m in enumerate(self.models):
            m.fit(tr_ft, tr_lbl)
            pred = m.predict(test_ft)
            score = sklearn.metrics.f1_score(test_lbl, pred, average="macro")
            log_progress(f"{self.param_sets[i]}, score = {score:.3f}", color="white", verbose=self.verbose)
            scores.append(score)
        best = np.argmax(scores)
        log_progress(f"Picked {self.param_sets[best]}", verbose=self.verbose)
        self.model = self.models[best]
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        check_predicts_proba(self.model)
        return self.model.predict_proba(X)

class AutoLinearSVM(AutoSkClf):
    """A sklearn BaseEstimator.
        Fits multiple LinearSVM models with different reguralization parameters and picks best performer.
    """

    def __init__(self, verbose=False):
        C_range = [0.1, 0.5, 1, 5, 10, 20, 50, 100, 500]
        self.param_sets = [{"C": C} for C in C_range]
        self.models = [LinearSVC(C=C, max_iter=5000) for C in C_range]
        self.verbose = verbose

class AutoLogisticRegression(AutoSkClf):
    """A sklearn BaseEstimator.
        Fits multiple LogisticRegression models with different reguralization parameters and picks best performer.
    """

    def __init__(self, verbose=False):
        C_range = [0.1, 0.5, 1, 5, 10, 20, 50, 100, 500]
        self.param_sets = [{"C": C} for C in C_range]
        self.models = [LogisticRegression(C=C, max_iter=1000) for C in C_range]
        self.verbose = verbose

# BASE END-TO-END CLASSIFIERS

class PredictionModel(BaseEstimator):
    """Base classification model class.
    
    A common interface for all models - image, text, audio, video and multimodal.
    Any PredictionModel can be trained on any MultimodalDataset.
    Subclasses implement the torch Dataset interface, and are compatible with the sklearn ecosystem (with some caveats).

    Attributes:
        modalities: A list of strings denoting modalities which this model operates with.
    """
    
    @abstractmethod
    def __init__(self):
        """Initialize model with optional settings."""
        pass

    @abstractmethod
    def train(self, dataset, train_ids=None):
        """Train model on given dataset.
        
        Args:
            dataset: A MultimodalDataset instance.
            train_ids: Indices representing the training set or None (whole dataset).
        """
        pass

    @abstractmethod
    def predict(self, dataset, test_ids=None):
        """Predict target variables on given dataset. Sklearn-compatible if test_ids are not passed.
        
        Args:
            dataset: A MultimodalDataset instance. Usually same as for train().
            test_ids: Indices representing the testing set or None (whole dataset).

        Returns:
            Predictions as a (len(test_ids), ) Ndarray.
        """
        pass

    def fit(self, dataset, y=None):
        """A sklearn-compatible wrapper for the train() method.
        
        Args:
            X: Must be a MultimodalDataset instance. Because train_ids are not passed here, X itself should be the train split.
            y: Since targets are included in X, this arguments is ignored.
        """
        self.train(dataset, train_ids=None)

    # def set_params(self, **parameters):
    #     """Set hyperparameters from dict. For compatibility with sklearn model selection."""

    #     for parameter, value in parameters.items():
    #         setattr(self, parameter, value)
    #     return self


class RandomClassifier(PredictionModel):
    """Predicts random class for every test instance."""

    def __init__(self, random_state=42, verbose=False):
        self.seed = random_state
        self.verbose = verbose
        self.modalities = []
    
    def train(self, dataset, train_ids=None):
        dataset, train_ids = prepare_input(dataset, train_ids, self)
        log_progress(f"Training {type(self).__name__} model...", verbose=self.verbose)
        self.dataset = dataset
        self.n_cls = len(dataset.classes)
    
    def predict(self, dataset, test_ids=None):
        dataset, test_ids = prepare_input(dataset, test_ids, self)
        return np.random.RandomState(seed=self.seed).randint(0, self.n_cls, len(test_ids))

    def predict_proba(self, dataset, test_ids=None):
        dataset, test_ids = prepare_input(dataset, test_ids, self)
        return np.random.RandomState(seed=self.seed).rand(len(test_ids))      


class MajorityClassifier(PredictionModel):
    """Predicts the most frequent class in training set for every test instance."""

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.modalities = []

    def train(self, dataset, train_ids=None):
        dataset, train_ids = prepare_input(dataset, train_ids, self)
        log_progress(f"Training {type(self).__name__} model...", verbose=self.verbose)
        _, labels = dataset.get_texts(tensor=False)
        self.mode_cls = scipy.stats.mode(labels[train_ids])[0]
        self.classes, self.freqs = np.unique(labels[train_ids], return_counts=True)
        self.proba = scipy.special.softmax(self.freqs)

    def predict(self, dataset, test_ids=None):
        dataset, test_ids = prepare_input(dataset, test_ids, self)
        return np.repeat(self.mode_cls, len(test_ids))

    def predict_proba(self, dataset, test_ids=None):
        dataset, test_ids = prepare_input(dataset, test_ids, self)
        return np.tile(self.proba, (len(test_ids), 1))


class UnimodalSkClassifier(PredictionModel):

    def __init__(self, fe=None, clf="svm_best", verbose=False):
        """A model consisting of a feature extractor (from mmlearn.fe) and a sklearn classifier.
        
        Args:
            fe: A feature extractor model from 'fe.image', 'fe.text', 'fe.audio' or 'fe.video'.
                The modality of the choosen f.e. must match the used dataset(s) (the dataset(s) must include it).
            clf: The classifier to use. 'svm', 'lr', 'rf' or an instance of any sklearn classifer.
                Default is base_models.AutoLinearSVM().
        """
        
        if not fe:
            raise ValueError("The feature extractor (fe) was not provided as argument.")
        self.fe = fe
        self.clf = clf
        self.modalities = self.fe.modalities
        self.verbose = verbose

    def train(self, dataset, train_ids=None):
        dataset, train_ids = prepare_input(dataset, train_ids, self)
        self.model = get_classifier(self.clf, verbose=self.verbose)
        log_progress(f"Training {type(self.fe).__name__} + {type(self.model).__name__} classifier model...",
                        verbose=self.verbose)

        features, labels = self.fe.extract_all(dataset, train_ids, verbose=self.verbose)

        self.model.fit(features, labels)

    def predict(self, dataset, test_ids=None):
        dataset, test_ids = prepare_input(dataset, test_ids, self)
        features, labels = self.fe.extract_all(dataset, test_ids, verbose=self.verbose)
        return self.model.predict(features)

    def predict_proba(self, dataset, test_ids=None):
        dataset, test_ids = prepare_input(dataset, test_ids, self)
        features, labels = self.fe.extract_all(dataset, test_ids, verbose=self.verbose)
        return self.model.predict_proba(features)