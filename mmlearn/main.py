# SCRATCH PAPER FOR TESTING MODULES

import os, sys
sys.path.insert(0, ".")
import numpy as np
import torch
import logging

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

    dataset = data.TastyRecipes()
    dataset2 = data.Fauxtography()

    spotify_dataset = data.SpotifyMultimodalVal(frac=0.2)
    # dl = DataLoader(spotify_dataset, batch_size=4, shuffle=True)
    # itr = iter(dl)
    # batch = next(itr)
    # audio_batch = batch["audio"]
    # print(audio_batch)
    # fe = audio_fe.OpenL3(combine=None)
    # emb = fe(audio_batch)
    # print(emb)


    # USE FEATURE EXTRACTORS SEPARATELY

    # dl2 = DataLoader(dataset2, batch_size=4, shuffle=True)
    # batch = next(iter(dl2))

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
    model2 = text_models.TextSkClassifier(fe=text_fe.NGrams(), clf="lr")
    # #model3 = audio_models.AudioSkClassifier(verbose=True)
    # # # # #model3 = text.TextSkClassifier(fe=fe.text.SentenceBERT(), clf="svm_best")
    # model3 = base_models.UnimodalSkClassifier(fe=audio_fe.OpenL3(), verbose=True)
    # #all_models = {"majority": model, "ngrams_svm": model2, "mobilenet_svm": model3}
    # all_models = {"openl3_svm": model3}
    # #all_ds = {"tasty": dataset, "faux": dataset2}
    # all_ds = {"spotify_valence": spotify_dataset}

    # # # # # print(eval.holdout(dataset, model2, dataframe=True))
    # # # # # print(eval.cross_validate(dataset, model2, dataframe=True))

    # results = eval.holdout_many(all_ds, all_models)
    # for metric in results:
    #     print("\n", metric, "\n", results[metric])


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
    # print(np.sum(prob, axis=1).round(2)


    # model2.fit(dataset2, None)
    # prob = model2.predict_proba(dataset2)
    # print(prob)
    # pred = model2.predict(dataset2)
    # print(pred)

    # dl2 = DataLoader(dataset2, batch_size=128, shuffle=True)
    # img_batch = next(iter(dl2))["image"].numpy()
    # img_fe = image_fe.MobileNetV3()
    # clf = LogisticRegression()
    # pl = Pipeline([("mobilenet", img_fe), ("lr", clf)])

    # pl.fit(img_batch, np.random.randint(0,2,img_batch.shape[0]))
    # pred = pl.predict(img_batch)
    # print(pred)

    arraylike = [
        {"text": "hello world", "target": 0},
        {"text": "lalalal", "target": 1},
        {"text": "fdsafsdasfddf", "target": 0},
    ]

    #print(spotify_dataset.get_audio([1,2,3]))
    
    # audio_model = audio_models.AudioSkClassifier()
    # eval.holdout(spotify_dataset, audio_model)

    #-----------
    # texts, targets = dataset2.get_texts(tensor=False)
    # fe = text_fe.CombineFeatures([text_fe.Doc2Vec(), text_fe.BERTExtractor()])
    # ft = fe(texts)
    # print(ft)
    autobot = text_models.AutoBOT()
    eval.holdout(dataset2, model)


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

    # print(dataset2)
    # dataset_sample = dataset2.sample([4,6,7])
    # print(dataset_sample)

    # print(dataset2[0:10])

    # arraylike = [
    #     {"text": "hello world", "target": 0},
    #     {"text": "lalalal", "target": 1},
    #     {"text": "fdsafsdasfddf", "target": 0},
    # ]
    # ald = data.from_torch_dataset(arraylike)
    # print(ald)

    # ald_dl = DataLoader(ald, batch_size=4, shuffle=True)
    # batch = next(iter(ald_dl))
    # print(batch)

    # model2.train(arraylike)
    # print(model2.predict(arraylike))


