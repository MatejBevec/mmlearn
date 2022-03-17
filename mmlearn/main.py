# SCRATCH PAPER FOR TESTING MODULES

# TODO: look up docstring and static typing conventions ✓
# TODO: feature extractor interface spec ✓
# TODO: implement image feature extractors ✓
# TODO: dataset interface spec (think extending modalities)
# TODO: create dataset from torch dataset (constructor or factory?)
# TODO: implement MultimodalDataset class ✓
# TODO: model interface spec ✓
# TODO: look at torch lightning interface ✓ (not applicable)
# TODO: verbose mode
# TODO: text feature extractors:
#   - keyword features
#   - POS n-grams
#   - doc2vec ✓
# TODO: trad text classifier ✓
# TODO: fine-tuned X image classifier ✓
# TODO: basic mm models: late fusion, early fusion with concat
# TODO: should models return probabilites as well?
# TODO: dataset downloads from dropbox ✓
# TODO: research types of early fusion
# TODO: global global constants ✓
# TODO: util class (log, hidden print....) ✓
# TODO: eval module ✓
#   - holdout(model, dataset, metric) ✓
#   - cross_validate(model, dataset, metric) ✓
#   - eval-all ✓

# Considerations:
# TODO: wrap Pytorch NN classifiers and "auto reg" sklearn classifiers into sklearn-like classes!!!!
# TODO: how many separate premade classifiers should be provided?
# TODO: think about the module naming (fe.image and image might clash)
# TODO: are mutable (instanciated) default arguments a problem? !!! -> looks like they are
# TODO: support for when embedding space or text space is larger than memory?
# TODO: make MultimodalDataset subscriptable?
# TODO: get_transformed(fes) in MultiModal dataset?
# TODO: helper train functions or separate Classifier, NeuralClassifier classes?
# TODO: is one MultimodalDataset for all modalities clean, or should i split the class?
# TODO: global environment variables for settings?

# Style Guide:
#   - import default modules, external libraries, internal modules in this order
#   - do not import specific classes or functions
#   - use asserts, add custom exceptions later
#   - use underscores for _internal_functions
#   - Google style docstrings
#   - no - dashes in module names


import os, sys
import numpy as np
import torch
import logging

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from mmlearn import data
from mmlearn.fe import image as imgfe 
from mmlearn.fe import text as textfe
from mmlearn.models import base, image, text, mm
from mmlearn import eval

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    
    # MULTIMODAL DATASETS

    dataset = data.TastyRecipes()
    dataset2 = data.Fauxtography()

    dl = DataLoader(dataset, batch_size=4, shuffle=True)
    imgs, texts, targets = next(iter(dl))

    # USE FEATURE EXTRACTORS SEPARATELY

    # img_fe = fe.image.MobileNetV3()
    # img_f = img_fe(imgs)
    # print("img features:, ", img_f)

    # text_fe = fe.text.WordNGrams()
    # text_f = text_fe(texts)
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

    model= base.MajorityClassifier()
    model2 = text.TextSkClassifier(fe=fe.text.NGrams(), clf="svm_best")
    #model3 = text.TextSkClassifier(fe=fe.text.SentenceBERT(), clf="svm_best")
    all_models = {"majority": model, "ngrams_svm": model2}
    all_ds = {"tasty": dataset, "faux": dataset2}

    print(eval.holdout(dataset, model2, dataframe=True))
    print(eval.cross_validate(dataset, model2, dataframe=True))

    # results = eval.holdout_many(all_ds, all_models)
    # for metric in results:
    #     print("\n", metric, "\n", results[metric])



