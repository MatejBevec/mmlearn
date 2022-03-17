import os
import sys
from abc import ABC, abstractmethod

import numpy as np
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

from mmlearn.models.base import ClsModel, prepare_input, get_classifier
from mmlearn.models import base
from mmlearn.fe import image as imgfe
from mmlearn.util import log_progress, DEVICE, REG_PARAM_RANGE

class ImageNeuralClassifier():

    def __init__(self, fe=imgfe.MobileNetV3()):
        """Image feature extractor + a pytorch NN classifier.
            The classifier is a 2-layer fully-connected network with ReLU act., Adam optim. and cross-entropy loss. 
        
        Args:
            fe: A feature extractor model from 'imgfe'.
        """
        
        self.fe = fe

    def train(self, dataset, train_ids):
        dataset, train_ids = prepare_input(dataset, train_ids)
        log_progress(f"Training {type(self.fe).__name__} fe + NN classifier model...")

        features, labels = self.fe.extract_all(dataset, train_ids)
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

        base.train_torch_nn(self.model, ft_dataset, loss_fn, optim, lr=lr)


    def predict(self, dataset, test_ids):
        dataset, test_ids = prepare_input(dataset, test_ids)
        features, labels = self.fe.extract_all(dataset, test_ids)
        ft_dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
        return base.predict_torch_nn(self.model, ft_dataset)


class ImageSkClassifier():

    def __init__(self, fe=imgfe.MobileNetV3(), clf="svm", best_reg=True):
        """Image feature extractor + a scikit-learn classifier.
        
        Args:
            fe: A feature extractor model from 'imgfe'.
            clf: The classifier to use. 'svm', 'lr', 'rf' or an instance of any sklearn classifer.
        """
        
        self.fe = fe
        self.model = get_classifier(clf, best_reg=best_reg)

    def train(self, dataset, train_ids):
        dataset, train_ids = prepare_input(dataset, train_ids)
        log_progress(f"Training {type(self.fe).__name__} + {type(self.model).__name__} classifier model...")

        features, labels = self.fe.extract_all(dataset, train_ids)

        if isinstance(self.model, list):
            split = int(len(features)*0.7)
            tr_ft, tr_lbl = features[:split], labels[:split]
            test_ft, test_lbl = features[split:], labels[split:]
            scores = []
            for i,m in enumerate(self.model):
                m.fit(tr_ft, tr_lbl)
                pred = m.predict(test_ft)
                score = sklearn.metrics.f1_score(test_lbl, pred, average="macro")
                log_progress(f"C = {REG_PARAM_RANGE[i]}, score = {score:.3f}", color="white")
                scores.append(score)
            best = np.argmax(scores)
            log_progress(f"Picked C = {REG_PARAM_RANGE[best]}")
            self.model = self.model[best]

        self.model.fit(features, labels)

    def predict(self, dataset, test_ids):
        dataset, test_ids = prepare_input(dataset, test_ids)
        features, _ = self.fe.extract_all(dataset, test_ids)
        return self.model.predict(features)


class TunedMobileNetV3(ClsModel):

    def __init__(self, epochs=20):
        """Fine-tuned pretrained MobileNet V3 Large image classification model.
            Final (classifier) layer is replaced to fit output dimension and the whole network is fine-tuned on given data.

        Args:
            epochs: Number of epochs to train for fune-tuning.
        """
        self.model = models.mobilenet_v3_large(pretrained=True)
        self.epochs = epochs
    
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

            for i,p in enumerate(self.model.features.parameters()):
                #Each epoch we are going to unfreeze 2 parameters, the weights and bias of a layer
                if i >= 16-epoch*2:
                    #Track the gradient for the appropriate layers
                    p.requires_grad = True

            for i, batch in enumerate(dl, 0):
                print(i)
                imgs, texts, labels = batch
                print(labels)
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()
                pred = self.model(imgs) 
                loss = loss_func(pred, labels.long())
                loss.backward()
                optimizer.step()
                batch_loss += loss
                if i%1 == 0:
                    print(f"{i} batches done. Loss = {batch_loss/10}")
                    batch_loss = 0

    def predict(self, dataset, test_ids):
        self.model.eval()
        bs = 64
        dl = DataLoader(dataset, batch_size=bs, sampler=test_ids)
        pred = np.ndarray(len(test_ids))
        loss_func = torch.nn.CrossEntropyLoss()
        for i, (imgs, _, labels) in enumerate(dl, 0):
            imgs = imgs.to(DEVICE)
            out = self.model(imgs).cpu().detach().numpy()
            pred[i*bs : i*bs+imgs.shape[0]] = out.argmax(axis=1)
        return pred.astype(int)
