# SCRATCH PAPER FOR TESTING MODULES

# To-Do:
# TODO: look up docstring and static typing conventions ✓
# TODO: feature extractor interface spec ✓
# TODO: implement image feature extractors ✓
# TODO: create dataset from torch dataset (constructor or factory?)
# TODO: implement MultimodalDataset class ✓
# TODO: model interface spec ✓
# TODO: look at torch lightning interface ✓ (not applicable)
# TODO: text feature extractors:
#   - keyword features
#   - POS n-grams
#   - doc2vec ✓
# TODO: trad text classifier ✓
# TODO: fine-tuned X image classifier ✓
# TODO: dataset downloads from dropbox ✓
# TODO: research types of early fusion
# TODO: global global constants ✓
# TODO: util class (log, hidden print....) ✓
# TODO: eval module ✓
#   - holdout(model, dataset, metric) ✓
#   - cross_validate(model, dataset, metric) ✓
#   - eval-all ✓
# TODO: basic mm models: late fusion, early fusion with concat ✓
# TODO: verbose mode
# TODO: should models return probabilites as well?
# TODO: extend MultimodalDataset interface for audio and video

# Considerations:
# TODO: wrap Pytorch NN classifiers and "auto reg" sklearn classifiers into sklearn-like classes!!!!
# TODO: think about the module naming (fe.image and image might clash) !!!
#   - yes, use full names
# are mutable default arguments a problem?
#   - yes, use "default" string
# TODO: how many separate pre-configured classifiers should be provided?
# support for when embedding space or text space is larger than memory?
#   - no
# TODO: get_transformed(fes) in MultiModal dataset?
# TODO: helper train functions or separate Classifier, NeuralClassifier classes?
# is one MultimodalDataset for all modalities clean, or should i split the class?
#   - one dataset
# TODO: JSON file for settings?

# Style Guide:
#   - import default modules, external libraries, internal modules in this order
#   - use underscores for _internal_functions
#   - Google style docstrings
#   - no dashes in module names
#   - "image", "text", "fe", "mm" refer to image, text, feature extractors and multimodal in names
#   - "targets" or "labels" refer to class values, "classes" refer to class types
#   - use util.log_progress(string) to log function/model progress
#   - do not use mutables in DEFAULT arguments (don't do "model=MobileNetV3()")
#   - TODO: do not import specific classes or functions, call module.Class() instead


import os, sys
import numpy as np
import torch
import logging

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from mmlearn import data
from mmlearn.fe import image_fe as imgfe 
from mmlearn.fe import text_fe as textfe
from mmlearn.models import base_models, image_models, mm_models, text_models
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

    model= base_models.MajorityClassifier()
    model2 = text_models.TextSkClassifier(fe=textfe.NGrams(), clf="svm_best")
    model3 = image_models.ImageSkClassifier()
    #model3 = text.TextSkClassifier(fe=fe.text.SentenceBERT(), clf="svm_best")
    all_models = {"majority": model, "ngrams_svm": model2, "mobilenet_svm": model3}
    all_ds = {"tasty": dataset, "faux": dataset2}

    # print(eval.holdout(dataset, model2, dataframe=True))
    # print(eval.cross_validate(dataset, model2, dataframe=True))

    results = eval.holdout_many(all_ds, all_models)
    for metric in results:
        print("\n", metric, "\n", results[metric])



