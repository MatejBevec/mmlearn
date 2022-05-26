# SCRATCH PAPER FOR TESTING MODULES

import os, sys
import numpy as np
import torch
import logging

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from mmlearn import data
from mmlearn.fe import image_fe
from mmlearn.fe import text_fe
from mmlearn.models import base_models, image_models, mm_models, text_models
from mmlearn import eval

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    
    # MULTIMODAL DATASETS

    dataset = data.TastyRecipes()
    print("tr loaded")
    dataset2 = data.Fauxtography()
    print("fx loaded")

    dl = DataLoader(dataset, batch_size=4, shuffle=True)
    #imgs, texts, targets = next(iter(dl))
    batch = next(iter(dl))
    print(batch)

    # USE FEATURE EXTRACTORS SEPARATELY

    # img_fe = image_fe.MobileNetV3()
    # img_f = img_fe(batch["image"])
    # print("img features:, ", img_f)

    # text_fe = text_fe.WordNGrams()
    # text_f = text_fe(batch["text"])
    # print("text features: ", text_f)

    # CLASSIFICATION MODEL

    # model = text.TextSkClassifier(fe=fe.text.NGrams(), clf="svm_best")
    # split = int(0.7 * len(dataset))
    # perm = np.random.permutation(len(dataset))
    # train_ids = perm[:split]
    # test_ids = perm[split:]

    # model.train(dataset, train_ids)
    # y = model.predict(dataset, test_ids)
    # print(y)
    # _, labels = dataset.get_texts(test_ids)
    # ca = accuracy_score(labels, y)
    # print(ca)

    # EVALUATE MULTIPLE MODELS AND DATASETS

    model= base_models.MajorityClassifier()
    model2 = text_models.TextSkClassifier(fe=text_fe.NGrams(), clf="svm_best")
    model3 = image_models.ImageSkClassifier()
    #model3 = text.TextSkClassifier(fe=fe.text.SentenceBERT(), clf="svm_best")
    all_models = {"majority": model, "ngrams_svm": model2, "mobilenet_svm": model3}
    all_ds = {"tasty": dataset, "faux": dataset2}

    # print(eval.holdout(dataset, model2, dataframe=True))
    # print(eval.cross_validate(dataset, model2, dataframe=True))

    results = eval.holdout_many(all_ds, all_models)
    for metric in results:
        print("\n", metric, "\n", results[metric])



