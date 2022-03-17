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

from mmlearn.models.base import ClsModel, prepare_input, get_classifier, REG_PARAM_RANGE
from mmlearn.fe import text as textfe
from mmlearn.util import log_progress, DEVICE, USE_CUDA



class TextSkClassifier(ClsModel):

    def __init__(self, fe=textfe.NGrams(), clf="svm_best"):
        """Word and character n-gram features + sklearn classifier.
        
        Args:
            fe: A feature extractor model from 'textfe'. Default is textfe.NGrams.
            clf: The classifier to use. 'svm', 'lr', 'rf' or an instance of any sklearn classifer.
        """
        
        self.fe = fe
        self.model = get_classifier(clf)

    def train(self, dataset, train_ids):
        dataset, train_ids = prepare_input(dataset, train_ids)

        log_progress(f"Training {type(self).__name__} model...")
        texts, labels = dataset.get_texts(train_ids)

        for i, tid in enumerate(train_ids):
            #print(tid, dataset.classes[labels[i]], dataset.names[tid])
            pass
        train_ft = self.fe(texts, train=True)
        #print(train_ft)
        self.model.fit(train_ft, labels)

    def predict(self, dataset, test_ids):
        dataset, test_ids = prepare_input(dataset, test_ids)
        texts, labels = dataset.get_texts(test_ids)
        return self.model.predict(self.fe(texts, train=False))


class BERT(ClsModel):
    """A fine-tuned BERT transformer model in classifier configuration. 

    The chosen pretrained BERT model and fine-tuned on given data.
    Implemented using the Simple Transformers library.
    """

    def __init__(self, weights='bert-base-uncased', epochs=20):
        """Initialize BERT model.

        Args:
            weights: The pretrained weights to load. Chose among 'bert-base-uncased', [todo], and more.
                    See huggingface.com/[todo]
            epochs: Number of epochs for fine-tuning.
        """
        self.weights = weights
        self.epochs = epochs
        
    def train(self, dataset, train_ids):
        dataset, train_ids = prepare_input(dataset, train_ids)
        log_progress(f"Training {type(self).__name__} model...")

        self.dataset = dataset
        texts, labels = dataset.get_texts(train_ids)
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

        self.model.train_model(train_df)

    def predict(self, dataset, test_ids):
        dataset, test_ids = prepare_input(dataset, test_ids)
        texts, _ = dataset.get_texts(test_ids)
        pred, raw_outputs = self.model.predict(list(texts))
        return pred


class TPOT(ClsModel):
    """A tree-based sklearn AutoML model.
    
    AutoML tool that automatically evolves scikit-learn pipelines based on tree ensemble learners.
    """

    def __init__(self, random_state=42):
        self.seed = random_state

    def train(self, dataset, train_ids):
        dataset, train_ids = prepare_input(dataset, train_ids)
        log_progress(f"Training {type(self).__name__} model...")

        self.dataset = dataset
        texts, labels = dataset.get_texts(train_ids)

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

    def predict(self, test_ids):
        dataset, test_ids = prepare_input(dataset, test_ids)
        texts, labels = dataset.get_texts(test_ids)
        return self.model.predict(texts)
