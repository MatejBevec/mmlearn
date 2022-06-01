# SCRATCH PAPER FOR TESTING MODULES

import os, sys
sys.path.insert(0, ".")
import numpy as np
import torch
import logging

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

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

    #dataset = data.TastyRecipes()
    #dataset2 = data.Fauxtography()
    # dataset3 = data.Fakeddit5k()
    # print("fe loaded")

    spotify_dataset = data.SpotifyMultimodalPop(frac=0.2)
    example = spotify_dataset[0]
    print(type(example["audio"]))
    print(type(example["image"]))
    print(type(example["text"]))
    print(example["target"])

    # clip, sr = example["audio"]
    # batch = (clip.unsqueeze(0), sr)
    # fe = audio_fe.OpenL3()
    # emb = fe(batch)
    # print(emb)

    # dl = DataLoader(dataset, batch_size=4, shuffle=True)
    # # #imgs, texts, targets = next(iter(dl))
    # batch = next(iter(dl))
    # print(batch)

    # USE FEATURE EXTRACTORS SEPARATELYFline

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
    #model3 = audio_models.AudioSkClassifier(verbose=True)
    # # # #model3 = text.TextSkClassifier(fe=fe.text.SentenceBERT(), clf="svm_best")
    all_models = {"majority": model, "ngrams_svm": model2, "mobilenet_svm": model3}
    #all_models = {"mobilenet_svm": model3}
    ##all_ds = {"tasty": dataset, "faux": dataset2}
    all_ds = {"spotify_valence": spotify_dataset}

    # # # # print(eval.holdout(dataset, model2, dataframe=True))
    # # # # print(eval.cross_validate(dataset, model2, dataframe=True))

    results = eval.holdout_many(all_ds, all_models)
    for metric in results:
        print("\n", metric, "\n", results[metric])


    # model = mm_models.NaiveEarlyFusion(verbose=True, clf="lr")
    # split = int(0.7 * len(dataset))
    # perm = np.random.permutation(len(dataset))
    # train_ids = perm[:split]
    # test_ids = perm[split:]
    # model.train(dataset, train_ids)

    # pred = model.predict(dataset, test_ids)
    # prob = model.predict_proba(dataset, test_ids)
    # print(pred)
    # print(prob)
    # print(prob.shape)
    # print(np.sum(prob, axis=1).round(2))

