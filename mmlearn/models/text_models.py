from logging import logProcesses
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
from sklearn.base import BaseEstimator
import sklearn.metrics
import torch
from simpletransformers.classification import ClassificationModel
from tpot import TPOTClassifier
from sentence_transformers import SentenceTransformer
import autoBOTLib

from mmlearn.models.base_models import PredictionModel, UnimodalSkClassifier
from mmlearn.models.base_models import prepare_input, check_predicts_proba, get_classifier
from mmlearn.fe import text_fe as textfe
from mmlearn.util import log_progress, DEVICE, USE_CUDA



class TextSkClassifier(UnimodalSkClassifier):

    def __init__(self, fe="default", clf="svm_best", verbose=False):
        """Text feature extractor + a scikit-learn classifier.
            A shorthand for UnimodalSkClassifier(fe=text_fe.NGrams()).
        
        Args:
            fe: A feature extractor model from 'fe.text_fe'. Default is NGrams().
            clf: The classifier to use. 'svm', 'lr', 'rf' or an instance of any sklearn classifer.
        """
        
        fe = textfe.NGrams() if fe == "default" else fe
        super().__init__(fe=fe, clf=clf, verbose=verbose)


class BERT(PredictionModel):
    """A fine-tuned BERT transformer model in classifier configuration. 

    The chosen pretrained BERT model and fine-tuned on given data.
    Implemented using the Simple Transformers library.
    """

    def __init__(self, weights='bert-base-uncased', epochs=20, verbose=False):
        """Initialize BERT model.

        Args:
            weights: The pretrained weights to load. Chose among 'bert-base-uncased', [todo], and more.
                    See huggingface.com/[todo]
            epochs: Number of epochs for fine-tuning.
        """
        self.weights = weights
        self.epochs = epochs
        self.modalities = ["text"]
        self.verbose = verbose
        
    def train(self, dataset, train_ids=None):
        dataset, train_ids = prepare_input(dataset, train_ids, self)
        log_progress(f"Training {type(self).__name__} model...", verbose=self.verbose)

        self.dataset = dataset
        texts, labels = dataset.get_texts(train_ids, tensor=False)
        train_df = pd.DataFrame()
        train_df["text"] = texts
        train_df["labels"] = labels

        model_args = {
            'num_train_epochs': self.epochs,
            'max_sequence_length': 512,
            'save_eval_checkpoints': False,
            'save_model_every_epoch': False,
            'overwrite_output_dir': True
        }
        self.model = ClassificationModel('bert',
                                    self.weights,
                                    num_labels=len(set(labels)),
                                    args=model_args,
                                    use_cuda=USE_CUDA)

        self.model.train_model(train_df, verbose=self.verbose)

    def _predict(self, dataset, test_ids=None):
        dataset, test_ids = prepare_input(dataset, test_ids, self)
        texts, _ = dataset.get_texts(test_ids, tensor=False)
        pred, raw_outputs = self.model.predict(list(texts))
        return pred, scipy.special.softmax(np.array(raw_outputs), axis=1)

    def predict(self, dataset, test_ids=None):
        return self._predict(dataset, test_ids)[0]

    def predict_proba(self, dataset, test_ids=None):
        return self._predict(dataset, test_ids)[1]


class TextTPOT(PredictionModel):
    """N-gram features into TPOT, a tree-based sklearn AutoML model.
    
    POT is anAutoML tool that automatically evolves scikit-learn pipelines based on tree ensemble learners.
    """

    def __init__(self, random_state=42, verbose=False):
        self.seed = random_state
        self.modalities = ["text"]
        self.verbose = verbose

    def train(self, dataset, train_ids=None):
        dataset, train_ids = prepare_input(dataset, train_ids, self)
        log_progress(f"Training {type(self).__name__} model...", verbose=self.verbose)

        self.dataset = dataset
        texts, labels = dataset.get_texts(train_ids, tensor=False)

        max_features = 1000
        vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=max_features)
        feat_list = [('word', vectorizer)]
        clf = TPOTClassifier(generations=5,
                            population_size=50,
                            verbosity=2,
                            random_state=self.seed,
                            config_dict="TPOT sparse")

        self.model = sklearn.pipeline.Pipeline([('union',
                                FeatureUnion(transformer_list=feat_list)),
                                ('scale', Normalizer()), ('tpot', clf)])

        self.model.fit(texts[train_ids], labels[train_ids])

    def predict(self, test_ids=None):
        dataset, test_ids = prepare_input(dataset, test_ids, self)
        texts, labels = dataset.get_texts(test_ids, tensor=False)
        return self.model.predict(texts)

    def predict_proba(self, test_ids=None):
        dataset, test_ids = prepare_input(dataset, test_ids, self)
        texts, labels = dataset.get_texts(test_ids, tensor=False)
        check_predicts_proba(self.model)
        return self.model.predict_proba(texts)


class AutoBOT(PredictionModel):
    """AutoBOT: representation evolution AutoML model.
    
    AutoML tool where an evolutionary algorithm automates the representation selection process.
    See the paper "autoBOT: evolving neuro‑symbolic representations for explainable low resource text classifcation" for more details.
    """

    def __init__(self, time_constraint=1, num_cpu="all", device="cpu",
                latent_dim=512, upsample=False, random_state=42, verbose=False):
        self.seed = random_state
        self.modalities = ["text"]
        self.verbose = verbose
        self.time_constraint = time_constraint
        self.num_cpu = num_cpu
        self.device = device
        self.latent_dim = latent_dim
        self.upsample = upsample
        self.random_state = random_state

    def train(self, dataset, train_ids=None):
        dataset, train_ids = prepare_input(dataset, train_ids, self)
        log_progress(f"Training {type(self).__name__} model...", verbose=self.verbose)

        self.dataset = dataset
        texts, labels = dataset.get_texts(train_ids, tensor=False)

        self.model = autoBOTLib.GAlearner(
            texts,
            labels,
            time_constraint=self.time_constraint,
            num_cpu=self.num_cpu,
            device=self.device,
            latent_dim=self.latent_dim,
            upsample=self.upsample,
            random_seed=self.random_state,
            verbose=1 if self.verbose else 0
        )

        self.model.evolve()

    def predict(self, test_ids=None):
        dataset, test_ids = prepare_input(dataset, test_ids, self)
        texts, labels = dataset.get_texts(test_ids, tensor=False)
        pred = self.model.predict(texts)
        return(pred)

    def predict_proba(self, test_ids=None):
        dataset, test_ids = prepare_input(dataset, test_ids, self)
        texts, labels = dataset.get_texts(test_ids, tensor=False)
        check_predicts_proba(self.model)
        return self.model.predict_proba(texts)