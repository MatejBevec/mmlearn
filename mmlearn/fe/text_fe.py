import os
import sys
import io
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
import autoBOTLib

from mmlearn.data import MultimodalDataset
from mmlearn.util import log_progress, DEVICE, USE_CUDA, SAVE_DIR

TEXT_FE_BATCH_SIZE = 4  # Batch size when extracting features from text


# HELPER FUNCTIONS
def _check_input(texts):
    if not (isinstance(texts, Iterable) and len(texts) > 0 and isinstance(texts[0], str)):
        raise TypeError("Text input must be an iterable of strings.")
    return np.array(texts)

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

def _check_output(out, tensor=False):
    assert isinstance(out, np.ndarray)
    if tensor:
        out = torch.from_numpy(out)   
    return out


class TextExtractor(ABC):
    """Base text feature extractor (document embedding) class.
    
    Args:
        tensor: If True, return the encoded batch as `Tensor`, else return `ndarray`.
        
    Attributes:
        modalities: A list of strings denoting modalities which this model operates with.
    """

    @abstractmethod
    def __init__(self, tensor=False):
        pass

    @abstractmethod
    def __call__(self, texts, train="auto", train_data=None):
        """Extracts features for a batch of texts.

        Args:
            imgs: An array or list of strings of length (bsize).
            train: In feature extractor that are trainable, this indicates whether to fit the extractor in this call.
                    train=True is identical to calling fit_transform() and train=False is identical to calling transform()
                    "auto" only fits in first call after init.
                    If not trainable, this argument is irrelevant and defaults to False.
            train_data: The custom training data, if trainable fe supports it.

        Returns:
            A batch of embeddings as an `ndarray` of shape (bsize, d).
        """
        pass

    def extract_all(self, dataset, ids=None, verbose=False):
        """Extracts text features (embeddings) for entire dataset.
        
        Args:
            dataset: A MultimodalDataset with `text` modality.
            ids: The indices of examples to encode. None for all.
        """
        
        return _extract_text_features(self, dataset, ids, verbose)

    def fit_transform(self, X, y=None):
        """For sklearn compatibility. (Trains) and calls self."""

        return self.__call__(torch.from_numpy(X), train=True)

    def transform(self, X, y=None):
        """For sklearn compatibility. Calls self (without training)."""

        return self.__call__(torch.from_numpy(X), train=False, train_data=None)

    @property
    def modalities(self):
        return ["text"]


class NGrams(TextExtractor):
    """Extract TfIdf features from word and character n-grams.

    Args:
        word_n: Word n-gram range (n).
        char_n: Character n-gram range (n).
        max_features: Limit output features to 'max_features' dimensions.
    """

    def __init__(self, word_n=(1, 3), char_n=(2, 4), max_features=2000, tensor=False):
        self.trained = False
        self.tensor = tensor
        max_features = int(max_features/2)
        word_vec = TfidfVectorizer(ngram_range=word_n, max_features=max_features)
        char_vec = TfidfVectorizer(analyzer="char", ngram_range=char_n, max_features=max_features)
        union = FeatureUnion(transformer_list=[("char", char_vec), ("word", word_vec)])
        self.pipeline = Pipeline([("union", union), ("norm", Normalizer())])

    def __call__(self, texts, train="auto", train_data=None):
        texts = _check_input(texts)
        if train is False or (train == "auto" and self.trained):
            out = self.pipeline.transform(texts).toarray()
        else:
            self.trained = True
            out = self.pipeline.fit_transform(texts).toarray()

        return _check_output(out, self.tensor)


class WordNGrams(TextExtractor):
    """Extract TfIdf features from word n-grams.

    Args:
        n: Word n-gram range.
        max_features: Limit output features to 'max_features' dimensions.
    """

    def __init__(self, n=(1,3), max_features=1000, tensor=False):
        self.trained = False
        self.tensor = tensor
        fe = TfidfVectorizer(ngram_range=n, max_features=max_features)
        norm = Normalizer()
        self.pipeline = Pipeline([("word", fe), ("norm", norm)])  

    def __call__(self, texts, train="auto", train_data=None):
        texts = _check_input(texts)
        if train is False or (train == "auto" and self.trained):
            out = self.pipeline.transform(texts).toarray()
        else:
            self.trained = True
            out = self.pipeline.fit_transform(texts).toarray()

        return _check_output(out, self.tensor)


class CharNGrams(TextExtractor):
    """Extract TfIdf features from character n-grams.

    Args:
        n: Character n-gram range.
        max_features: Limit output features to 'max_features' dimensions.
    """

    def __init__(self, n=(2,4), max_features=1000, tensor=False):
        self.trained = False
        self.tensor = tensor
        fe = TfidfVectorizer(analyzer="char", ngram_range=n, max_features=max_features)
        norm = Normalizer()
        self.pipeline = Pipeline([("char", fe), ("norm", norm)])

    def __call__(self, texts, train="auto", train_data=None):
        texts = _check_input(texts)
        if train is False or (train == "auto" and self.trained):
            out = self.pipeline.transform(texts).toarray()
        else:
            self.trained = True
            out = self.pipeline.fit_transform(texts).toarray()

        return _check_output(out, self.tensor)

class Keywords(TextExtractor):
    """Extract keyword features.
        See the paper "autoBOT: evolving neuro-symbolic representations for explainable low resource text classifcation" for more details.
        Target labels must be bassed via train_data=labels when calling for the first time.

    Args:
        max_features: Limit output features to 'max_features' dimensions.
        tensor: If True, return Tensor, else return ndarray.
    """

    def __init__(self, max_features=500, tensor=False):
        self.trained = False
        self.tensor = tensor
        self.max_features = max_features

    def __call__(self, texts, train="auto", train_data=None):
        texts = _check_input(texts)

        if train is False or (train == "auto" and self.trained):
            out = self.model.transform(texts).toarray()
        else:
            self.trained = True
            self.model = autoBOTLib.KeywordFeatures(targets=train_data, max_features=10000)
            out = self.model.fit_transform(texts).toarray()

        return _check_output(out, self.tensor)

class TokenRelations(TextExtractor):
    """Extract token relations, i.e. features based on the average distances between pairs of tokens.
        See the paper "autoBOT: evolving neuro‑symbolic representations for explainable low resource text classifcation" for more details.

    Args:
        max_features: Limit output features to 'max_features' dimensions.
        min_token: The kind of tokens to consider. Options are "word", "unigrams", "bigrams" and "threegrams".
        tensor: If True, return Tensor, else return ndarray.
    """

    def __init__(self, max_features=10000, min_token="bigrams", targets=None, tensor=False):
        self.trained = False
        self.tensor = tensor
        self.max_features = max_features 
        self.min_token = min_token
        self.targets = targets

    def __call__(self, texts, train="auto", train_data=None):
        texts = _check_input(texts)

        if train is False or (train == "auto" and self.trained):
            out = self.model.transform(texts).toarray()
        else:
            self.trained = True
            self.model = autoBOTLib.relationExtractor(
                max_features=self.max_features,
                min_token=self.min_token,
                targets=train_data,
                verbose=False
            )
            out = self.model.fit_transform(texts, a2=None).toarray()

        return _check_output(out, self.tensor)

class AllSparseFeatures(TextExtractor):
    """A convenience class: concatenate features from (all) available sparse/symbolic feature extractors.

    Args:
        word_n: An optional list of selected extractors.
                Must be a subset of ["word_ngrams", "char_ngrams", "keywords", "token_relations"] or None to select all.
        tensor: If True, return Tensor, else return ndarray.
    """

    def __init__(self, selection=None, tensor=False):
        fes = {
            "word_ngrams": WordNGrams,
            "char_ngrams": CharNGrams,
            "keywords": Keywords,
            "token_relations": TokenRelations,
        }
        
        self.trained = False
        self.selection = selection if selection else list(fes.keys())
        self.tensor = tensor

        self.fe_list = [fes[fe](tensor=False) for fe in self.selection]

    def __call__(self, texts, train="auto", train_data=None):
        texts = _check_input(texts)

        ft_list = [fe(texts, train=train, train_data=train_data) for fe in self.fe_list]
        out = np.concatenate(ft_list, axis=1)

        return _check_output(out, self.tensor)

class CombineFeatures(TextExtractor):
    """A convenience class: concatenate features from provided ft. extractors.

    Args:
        extractors: A list of FeatureExtractor instances.
        tensor: If True, return Tensor, else return ndarray.
    """

    def __init__(self, extractors, tensor=False):
        self.trained = False
        self.extractors = extractors
        self.tensor = tensor

    def __call__(self, texts, train="auto", train_data=None):
        texts = _check_input(texts)

        ft_list = []
        for fe in self.extractors:
            ft = fe(texts, train=train, train_data=train_data) 
            ft_list.append(_check_output(ft, tensor=False))

        out = np.concatenate(ft_list, axis=1)

        return _check_output(out, self.tensor)




class SentenceTransformersExtractor(TextExtractor):
    """A generic fe wrapper: extract text features with pretrained SentenceTransformers model of choice.
    
    Args:
        model: The pretrained SentenceTransformers model (weights) to use.
            Some options include:
            [TODO]
            See [TODO] for more detail.
        tensor: If True, return Tensor, else return ndarray.
    """

    def __init__(self, model, tensor=False):
        self.model = model
        self.tensor = tensor
        self.fe = SentenceTransformer(model)

    def __call__(self, texts, train=False, train_data=None):
        texts = _check_input(texts)
        out = self.fe.encode(texts, show_progress_bar=False,
                        device=("cuda" if USE_CUDA else "cpu"))
        return _check_output(out, self.tensor)


class BERTExtractor(SentenceTransformersExtractor):
    """Extract text features with a pretrained BERT document embedding model.
        This is a shorthand for SentenceTransformersExtractor(model='paraphrase-multilingual-mpnet-base-v2').
    
    Args:
        tensor: If True, return Tensor, else return ndarray.
    """

    def __init__(self, tensor=False):
        self.tensor = tensor
        self.fe = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

class MPNETExtractor(SentenceTransformersExtractor):
    """Extract text features with a fine-tuned microsoft/mpnet-base embedding model.
        Intended for information retrieval with sentences and short paragraphs.
        See https://huggingface.co/sentence-transformers/all-mpnet-base-v2 for more detail.
        This is a shorthand for SentenceTransformersExtractor(model='all-mpnet-base-v2').
    
    Args:
        tensor: If True, return Tensor, else return ndarray.
    """

    def __init__(self, tensor=False):
        self.tensor = tensor
        self.fe = SentenceTransformer('all-mpnet-base-v2')
        

class CLIPExtractor(TextExtractor):
    """Extract text features with a pretrained CLIP joint image-text embedding model.
        This is a shorthand for SentenceTransformersExtractor(model='clip-ViT-B-32').
    
    Args:
        tensor: If True, return Tensor, else return ndarray.
    """

    def __init__(self, tensor=False):
        self.tensor = tensor
        self.fe = SentenceTransformer('clip-ViT-B-32')


class Doc2Vec(TextExtractor):
    """Doc2Vec text feature extractor.
        Pretrained on Gensim's "common_texts" by default.
        Pass fe(train=True, train_data=texts:list) to train on custom data.

    Args:
        dim: Output dimension.
        window: The window size in training.
        min_count: Ignore words with lower frequency.
    """
    
    def __init__(self, dim=64, window=2, min_count=1, tensor=False, train_data=None):
        self.dim = dim
        self.window = window
        self.min_count = min_count
        self.tensor = tensor

        if train_data is None:
            train_data = gensim.test.utils.common_texts
        
        save_dir = SAVE_DIR
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(SAVE_DIR, "doc2vec.tmp")

        save_stdout = sys.stdout
        sys.stdout = io.BytesIO()

        if os.path.isfile(save_path):
            self.model = doc2vec.Doc2Vec.load(save_path) #TODO: don't load if different parameters
        else:
            train_corpus = [doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(train_data)]       
            self.model = doc2vec.Doc2Vec(train_corpus, verbose=False,
                                    vector_size=dim, window=window, min_count=min_count, workers=4)
            self.model.save(save_path)

        sys.stdout = save_stdout    


    def __call__(self, texts, train=False, train_data=None):
        texts = _check_input(texts)
        docs = [text.split(" ") for text in texts]
        out = np.array([self.model.infer_vector(doc) for doc in docs])
        return _check_output(out, self.tensor)




        

if __name__ == "__main__":

    # dataset = MultimodalDataset("data/caltech-birds")
    # model = MobileNetV3()

    # imgs, text, label = dataset[0]
    # y = model.predict(imgs.unsqueeze(0))
    # print(y.shape)
    pass
