**Easy-to-use and flexible multimodal classification library with a sk-learn-like interface.**
Offers traditional and neural image, text and multimodal models with a standardized interface as well as feature extractors and evaluation utilities.

## Usage

### Install from repo
```
git clone https://github.com/MatejBevec/mmlearn
cd mmlearn
pip install .
```

### Quickly initialize, train and evaluate a model

```python
import mmlearn as mm
from mm import data, eval
from mm import models.mm
from mm import fe.image, fe.text

dataset = data.CaltechBirds()
model = models.mm.EarlyFusion(image_fe=fe.image.ResNet(), text_fe=fe.text.SentenceBERT(), clf="svm")

results = eval.holdout(dataset, model, ratio=0.7)
```

## Dataset interface

```python
from mm import data

my_dataset = data.MultimodalDataset("path_to_directory")
incl_dataset = data.TastyRecipes()

# MultimodalDataset provides a pytorch Dataset interface
dl = DataLoader(my_dataset, batch_size=4, shuffle=True)
imgs, texts, targets = next(iter(dl))   # Tensors, strings, integer target classes

# Alternatively, (all) examples can be accessed as Ndarrays
optional_index = [0, 7, 9]
texts, targets = my_dataset.get_texts(optional_index)
```

## Extracting only features for downstream learning
Various models are available to be used as standalone feature extractors
```python
from mm import fe.image, fe.text

# All extractors take a batch of images/texts as input and produce a (batch_size, dim) Ndarray of embeddings
# Some accept optional parameters

img_fe = fe.image.MobileNetV3()
text_fe = fe.text.NGrams(word_n=[1,3], char_n=[2,4])

imgs, texts, targets = next(iter(dl)) 

all_features = np.concatenate([ img_fe(imgs) , text_fe(texts) ])

```

## Classification
Various image-only, text-only and multimodal classification models are available

```python
import numpy as np
from sklearn.metrics import accuracy_score as ca
from mm import data
from mm import models.image, models.text
from mm import fe.image, fe.text

# All models subclass ClsModel and provide an identical interface, with "train" and "predict" methods
# Some accept optional parameters

text_model = models.text.BERT()
mm_model = models.mm.EarlyFusion(image_fe=fe.image.ViT(), text_fe=fe.text.TextCLIP(), clf="lr_best")

dataset = data.CaltechBirds()
split = int(len(dataset)*0.7)
perm = np.random.permutation(len(dataset))
train_ids, test_ids = perm[:split], perm[split:]

# One way to interface ClsModel is to provide a train and test index
text_model.train(dataset, train_ids)
pred = text_model.predict(dataset, test_ids)

# Another is to use separate datasets
mm_model.train(train_dataset)
pred = mm_model.predict(test_dataset)

_, labels = test_dataset.get_texts()
accuracy = ca(labels, pred)

```

## Evaluation
Utilities that simplify evaluation of multiple models on multiple dataset are also available.

```python
from mm import data, eval
from mm import models.image, models.text, models.mm

# Evaluate a single model with a holdout set using default metrics
mm_model = models.mm.EarlyFusion()
recipes = data.TastyRecipes()

results = eval.holdout(recipes, mm_model, ratio=0.7) # A dict of scores for every metric

# Evaluate multiple models on multiple datasets using cross-validation
datasets = {
    "recipes": recipes,
    "birds": data.CaltechBirds()
}
clf_models = {
    "early_fusion": mm_model,
    "bert": models.text.BERT()
}

results = eval.cross_validate_many(datasets, clf_models, folds=4)
# A dict of (models, datasets) DataFrames for every metric

```


## Contributing TODO

## Issues, PRs etc. TODO
