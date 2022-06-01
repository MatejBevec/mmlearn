import os
import sys
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as tf
import torchvision.models as models
from torch.utils.data import DataLoader
import pytorch_pretrained_vit
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import FeatureUnion, Pipeline
import gensim.test.utils
from gensim.models import doc2vec
from tqdm import tqdm

from mmlearn.data import MultimodalDataset
from mmlearn.util import log_progress, DEVICE, USE_CUDA, SAVE_DIR

TEXT_FE_BATCH_SIZE = 4  # Batch size when extracting features from text


# HELPER FUNCTIONS
def _check_input(texts):
    if not (isinstance(texts, Iterable) and len(texts) > 0 and isinstance(texts[0], str)):
        raise TypeError("Text input must be a list or array of strings.")
    # Check if first run - maybe move elsewhere?

def _extract_text_features(fe, dataset, ids=None, verbose=False):
    if not isinstance(dataset, MultimodalDataset):
        raise TypeError("'dataset' must be a MultimodalDataset.")
    if not isinstance(fe, TextExtractor):
        raise TypeError("'fe' must be a TextExtractor from fe.text")

    features_list = []
    labels_list = []
    n = len(ids) if ids is not None else len(dataset)
    dl = DataLoader(dataset, batch_size=TEXT_FE_BATCH_SIZE, sampler=ids)

    pbar = tqdm(total=n, desc="Extracting text features", disable=not verbose)
    for i, batch in enumerate(dl, 0):
        features_list.append(fe(batch["text"]))
        labels_list.append(batch["target"])
        pbar.update(len(batch["target"]))
    pbar.close()

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return features, labels



class TextExtractor(ABC):
    """Base text feature extractor class."""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, texts, trained=False):
        """Extracts features for a batch of texts.

        Args:
            imgs: An (bsize,) array or list of strings.
            trained: In feature extractor that are trainable, this indicates whether to fit the extractor in this call.
                    Use True if using in training step and False othewise. "auto" only fits in first call after init.
                    In non-trainable, this argument is irrelevant and deafults to False.

        Returns:
            A batch of embeddings as a (bsize, d) Ndarray.
        """
        pass

    def extract_all(self, dataset, ids=None, verbose=False):
        """Extracts image features (embeddings) for entire dataset."""
        
        return _extract_text_features(self, dataset, ids, verbose)

    @property
    def modalities(self):
        return ["text"]


class NGrams(TextExtractor):

    def __init__(self, word_n=(1, 3), char_n=(2, 4), max_features=2000):
        """Extract TfIdf features from word and character n-grams.

        Args:
            word_n: Word n-gram range (n).
            char_n: Character n-gram range (n).
            max_features: Limit output features to 'max_features' dimensions.
        """

        self.trained = False
        max_features = int(max_features/2)
        word_vec = TfidfVectorizer(ngram_range=word_n, max_features=max_features)
        char_vec = TfidfVectorizer(analyzer="char", ngram_range=char_n, max_features=max_features)
        union = FeatureUnion(transformer_list=[("char", char_vec), ("word", word_vec)])
        self.pipeline = Pipeline([("union", union), ("norm", Normalizer())])

    def __call__(self, texts, train="auto"):
        _check_input(texts)
        if train is False or (train == "auto" and self.trained):
            return self.pipeline.transform(texts).toarray()
        else:
            self.trained = True
            return self.pipeline.fit_transform(texts).toarray()


class WordNGrams(TextExtractor):

    def __init__(self, n=(1,3), max_features=1000):
        """Extract TfIdf features from word n-grams.

        Args:
            n: Word n-gram range.
            max_features: Limit output features to 'max_features' dimensions.
        """

        self.trained = False
        fe = TfidfVectorizer(ngram_range=n, max_features=max_features)
        norm = Normalizer()
        self.pipeline = Pipeline([("word", fe), ("norm", norm)])

    def __call__(self, texts, train="auto"):
        _check_input(texts)
        if train is False or (train == "auto" and self.trained):
            return self.pipeline.transform(texts).toarray()
        else:
            self.trained = True
            return self.pipeline.fit_transform(texts).toarray()


class CharNGrams(TextExtractor):

    def __init__(self, n=(2,4), max_features=1000):
        """Extract TfIdf features from character n-grams.

        Args:
            n: Character n-gram range.
            max_features: Limit output features to 'max_features' dimensions.
        """

        self.trained = False
        TfidfVectorizer(analyzer="char", ngram_range=n, max_features=max_features)
        norm = Normalizer()
        self.pipeline = Pipeline([("char", fe), ("norm", norm)])

    def __call__(self, texts, train="auto"):
        _check_input(texts)
        if train is False or (train == "auto" and self.trained):
            return self.pipeline.transform(texts).toarray()
        else:
            self.trained = True
            return self.pipeline.fit_transform(texts).toarray()


class SentenceBERT(TextExtractor):

    def __init__(self, model='paraphrase-multilingual-mpnet-base-v2'):
        """Extract text features with a pretrained SentenceBERT document embedding model.
        
        Args:
            model: The pretrained BERT model to use. See [TODO] for more details.
        """

        # TODO: change cache folder?
        self.fe = SentenceTransformer(model, device=None)

    def __call__(self, texts, train=False):
        _check_input(texts)
        return self.fe.encode(texts, show_progress_bar=True,
                                device=("cuda" if USE_CUDA else "cpu"))


class TextCLIP(TextExtractor):

    def __init__(self):
        """Extract text features with a pretrained CLIP joint document-image embedding model."""

        self.fe = SentenceTransformer("clip-ViT-B-32")
    
    def __call__(self, texts, train=False):
        _check_input(texts)
        return self.fe.encode(texts, show_progress_bar=False)


class Doc2Vec(TextExtractor):

    def __init__(self, train=False, train_data=None, dim=64, window=2, min_count=1):
        """Doc2Vec text feature extractor. See base class for details.

        Args:
            train_data: An optional list of string documents.
                        Trained on Gensim's "common_texts" if None.
        """

        if train_data is None:
            train_data = gensim.test.utils.common_texts

        save_path = os.path.join(SAVE_DIR, "doc2vec.tmp")
        if os.path.isfile(save_path):
            self.model = doc2vec.Doc2Vec.load(save_path) #TODO: don't load if different parameters
        else:
            train_corpus = [doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(train_data)]       
            self.model = doc2vec.Doc2Vec(train_corpus,
                                    vector_size=dim, window=window, min_count=min_count, workers=4)
            self.model.save(save_path)


    def __call__(self, texts, train=False):
        _check_input(texts)
        docs = [text.split(" ") for text in texts]
        return np.array([self.model.infer_vector(doc) for doc in docs])

        

if __name__ == "__main__":

    # dataset = MultimodalDataset("data/caltech-birds")
    # model = MobileNetV3()

    # imgs, text, label = dataset[0]
    # y = model.predict(imgs.unsqueeze(0))
    # print(y.shape)
    pass
