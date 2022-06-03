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
from mmlearn.fe import image_fe as imgfe
from mmlearn.util import log_progress, DEVICE, REG_PARAM_RANGE

class ImageNeuralClassifier(PredictionModel):

    def __init__(self, fe="default", verbose=False):
        """Image feature extractor + a pytorch NN classifier.
            The classifier is a 2-layer fully-connected network with ReLU act., Adam optim. and cross-entropy loss. 
        
        Args:
            fe: A feature extractor model from 'fe.image'. Default is MobileNetV3.
        """
        
        self.fe = imgfe.MobileNetV3() if fe == "default" else fe
        self.modalities = ["image"]
        self.verbose = verbose

    def train(self, dataset, train_ids):
        dataset, train_ids = prepare_input(dataset, train_ids, self)
        log_progress(f"Training {type(self.fe).__name__} fe + NN classifier model...", verbose=self.verbose)

        features, labels = self.fe.extract_all(dataset, train_ids, verbose=self.verbose)
        ft_dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
        n_cls = len(dataset.classes)
        
        self.model = nn.Sequential(
            nn.Linear(features.shape[1], 256),
            nn.ReLU(),
            nn.Linear(256, n_cls)
        )
        lr = 1e-3
        loss_fn = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(self.model.parameters(), lr=lr)

        base_models.train_torch_nn(self.model, ft_dataset, loss_fn, optim, lr=lr)

    def _predict(self, dataset, test_ids):
        dataset, test_ids = prepare_input(dataset, test_ids, self)
        features, labels = self.fe.extract_all(dataset, test_ids)
        ft_dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
        return base_models.predict_torch_nn(self.model, ft_dataset)

    def predict(self, dataset, test_ids):
        return self._predict(dataset,test_ids)[0]

    def predict_proba(self, dataset, test_ids):
        return self._predict(dataset, test_ids)[1]


class ImageSkClassifier(UnimodalSkClassifier):

    def __init__(self, fe="default", clf="svm_best", verbose=False):
        """Image feature extractor + a scikit-learn classifier.
            A shorthand for UnimodalSkClassifier(fe=image_fe.MobileNetV3()).
        
        Args:
            fe: A feature extractor model from 'fe.image_fe'. Default is MobileNetV3.
            clf: The classifier to use. 'svm', 'lr', 'rf' or an instance of any sklearn classifer.
        """
        
        fe = imgfe.MobileNetV3() if fe == "default" else fe
        super(self, ImageSkClassifier).__init__(fe=fe, clf=clf, verbose=verbose)


class TunedMobileNetV3(PredictionModel):

    def __init__(self, epochs=20, verbose=False):
        """Fine-tuned pretrained MobileNet V3 Large image classification model.
            Final (classifier) layer is replaced to fit output dimension and the whole network is fine-tuned on given data.

        Args:
            epochs: Number of epochs to train for fune-tuning.
        """
        self.model = models.mobilenet_v3_large(pretrained=True)
        self.epochs = epochs
        self.modalities = ["image"]
        self.verbose = verbose
    
    def train(self, dataset, train_ids):
        self.n_cls = len(dataset.classes)
        self.model.classifier = nn.Linear(960, self.n_cls)
        self.model.to(DEVICE)

        dl = DataLoader(dataset, batch_size=64, sampler=train_ids)
        loss_func = torch.nn.CrossEntropyLoss()
        lr = 1e-3
        optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=1e-4)

        for p in self.model.features.parameters():
            p.requires_grad = False

        for epoch in range(self.epochs):
            batch_loss = 0
            log_progress(f"Training epoch {epoch}/{self.epochs}...", verbose=self.verbose)

            for i,p in enumerate(self.model.features.parameters()):
                #Each epoch we are going to unfreeze 2 parameters, the weights and bias of a layer
                if i >= 16-epoch*2:
                    #Track the gradient for the appropriate layers
                    p.requires_grad = True

            pbar = tqdm(total=len(dl), disable=not self.verbose)
            for i, batch in enumerate(dl, 0):
                imgs, labels = batch["image"], batch["target"]
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()
                pred = self.model(imgs) 
                loss = loss_func(pred, labels.long())
                loss.backward()
                optimizer.step()
                batch_loss += loss
                if i%10 == 0:
                    pbar.update(10)
                    pbar.set_description(f"Loss = {batch_loss/10}, batches done")
                    log_progress(f"{i} batches done. Loss = {batch_loss/10}")
                    batch_loss = 0

    def _predict(self, dataset, test_ids):
        self.model.eval()
        bs = 64
        dl = DataLoader(dataset, batch_size=bs, sampler=test_ids)
        pred = np.ndarray(len(test_ids))
        prob = np.ndarray((len(test_ids), len(dataset.classes)))
        loss_func = torch.nn.CrossEntropyLoss()
        for i, batch in enumerate(dl, 0):
            imgs = batch["image"].to(DEVICE)
            out = self.model(imgs).cpu().detach().numpy()
            pred[i*bs : i*bs+imgs.shape[0]] = out.argmax(axis=1)
            prob[i*bs : i*bs+imgs.shape[0], :] = out
        return pred.astype(int), scipy.special.softmax(prob, axis=1)

    def predict(self, dataset, test_ids):
        return self._predict(dataset, test_ids)[0]

    def predict_proba(self, dataset, test_ids):
        return self._predict(dataset, test_ids)[1]
