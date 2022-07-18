# SCRATCH PAPER FOR TESTING MODULES

import os, sys
sys.path.insert(0, ".")
import numpy as np
import torch
import logging
import pprint

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from mmlearn import data
from mmlearn.fe import image_fe, text_fe, audio_fe
from mmlearn.models import base_models, image_models, mm_models, text_models, audio_models
from mmlearn import eval



logging.basicConfig(level=logging.INFO)

class TestDataset(Dataset):

    def __init__(self):
        n = 10
        self.texts = ["hello", "world", "this", "is", "a", "test", "of", "the", "new", "dataset", "feature"]
        self.imgs = [torch.rand(800,200) for i in range(len(self.texts))]
        self.targets = [torch.randint(0,5, (1,)).item() for i in range(len(self.texts))]
        pass

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return {"image": self.imgs[i], "text": self.texts[i], "target": self.targets[i]}

if __name__ == "__main__":
    
    # MULTIMODAL DATASETS

    dataset = data.TastyRecipes(shuffle=True)
    dataset2 = data.Fauxtography(shuffle=True)
    #spotify_dataset = data.SpotifyMultimodalVal(frac=0.2)




    # EVALUATE MULTIPLE MODELS AND DATASETS

    model= base_models.MajorityClassifier()
    model2 = text_models.TextSkClassifier(fe=text_fe.NGrams(), clf="lr_best")

    arraylike = [
        {"text": "hello world", "target": 0},
        {"text": "lalalal", "target": 1},
        {"text": "fdsafsdasfddf", "target": 0},
    ]

    latef = mm_models.LateFusion(combine="max")
    earlyf = mm_models.NaiveEarlyFusion(text_fe=text_fe.NGrams(), clf="lr_best")
    models = {"majority": model, "text": model2,
            "late fusion": latef, "early fusion": earlyf}
    results = eval.holdout_many(dataset2, models, dataframe=True)
    pprint.pprint(results)

    #eval.holdout(dataset2, model2)


    # gs = GridSearchCV(
    #     model2,
    #     {"clf": ["lr", "rf"], "fe": [text_fe.NGrams(), text_fe.SentenceBERT()]},
    #     scoring="f1",
    #     cv=2
    # )

    # gs.fit(dataset2, dataset2.get_targets())

    # print(gs.best_estimator_)
    # print(gs.best_params_)
    # print(gs.best_score_)




