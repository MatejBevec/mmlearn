**Easy-to-use and flexible multimodal classification library with a sk-learn-like interface.**

Offers a standardized interface across image, text, audio and video modalities, including quickly deployable traditional and neural classification models and feature extractors (encoders), as well as easy-to-use evaluation utilities.

## Install from repo
```
git clone https://github.com/MatejBevec/mmlearn
pip install ./mmlearn
```

To avoid any dependency clashes, consider doing this in a **virtual environment.**



## Quickly initialize, train and evaluate a model

```python
import mmlearn as mm
from mm import data, eval
from mm.models import mm_models
from mm.fe import image_fe, text_fe

dataset = data.CaltechBirds()
model = mm_models.EarlyFusion(image_fe=image_fe.ResNet(), text_fe=text_fe.SentenceBERT(), clf="svm")

results = eval.holdout(dataset, model, ratio=0.7)
```



## Usage

### Dataset interface

MultimodalDataset offers a standardized data handling interface across modalities.

```python
from mm import data

my_dataset = data.MultimodalDataset("path_to_directory")
included_dataset = data.TastyRecipes()

# MultimodalDataset implements a pytorch Dataset interface
dl = DataLoader(my_dataset, batch_size=4, shuffle=True)
imgs, texts, targets = next(iter(dl))   # Tensors, strings, integer target classes

# Alternatively, (all) examples can be accessed directly
optional_idx = [0, 7, 9]
images, targets = my_dataset.get_images(optional_idx)
```

### Extracting only features for downstream learning
Various models are available to be used as standalone feature extractors or as components in prediction models.

```python
from mm.fe import image_fe, text_fe

# Feature extractors take a batch of images/texts/audio clips/videos as input and produce a (batch_size, dim) embeddings

img_fe = image_fe.MobileNetV3()
text_fe = text_fe.NGrams(word_n=[1,3], char_n=[2,4])
imgs, texts, targets = next(iter(dl)) 

img_features, text_features = img_fe(imgs), text_fe(texts)
```

### Classification
Various unimodal and multimodal classification models are available

```python
import numpy as np
from sklearn.metrics import accuracy_score as ca
from mm import data
from mm.models import image_models, text_models
from mm.fe import image_fe, text_fe

# All PredictionModel-s provide an identical interface, with "train" and "predict" methods
text_model = text_models.BERT()
mm_model = mm_models.EarlyFusion(image_fe=image_fe.ViT(), text_fe=text_fe.TextCLIP(), clf="lr_best")

dataset = data.CaltechBirds()
split = int(len(dataset)*0.7)
perm = np.random.permutation(len(dataset))
train_ids, test_ids = perm[:split], perm[split:]

# One way to interface ClsModel is to provide a train and test index
text_model.train(dataset, train_ids)
pred = text_model.predict(dataset, test_ids)

# Another is to use separate datasets
train_set, test_set = dataset[:split], dataset[split:] # returns subsampled copies of dataset
mm_model.train(train_set)
pred = mm_model.predict(test_set)

labels = test_set.get_targets()
accuracy = ca(labels, pred)
```

### Evaluation
Utilities that simplify evaluation of multiple models on multiple dataset are also available.

```python
from mm import data, eval
from mm.models import image_models, text_models, mm_models

# Evaluate a single model with a holdout set using default metrics
mm_model = mm_models.EarlyFusion()
recipes = data.TastyRecipes()

results = eval.holdout(recipes, mm_model, ratio=0.7)
# A dict of scores for every metric

# Evaluate multiple models on multiple datasets using cross-validation
datasets = {
    "recipes": recipes,
    "birds": data.CaltechBirds()
}
clf_models = {
    "early_fusion": mm_model,
    "bert": text_models.BERT()
}

results = eval.cross_validate_many(datasets, clf_models, folds=4)
# A dict of (models, datasets) dataframes for every metric

```

### Compatibility with scikit-learn
mmlearn bridges the gap between in-memory computing (sklearn) in the domain of text/tabular data and deep learning with images, audio and video, where batched operation is necessary.
For maximum flexibility, both FeaturesExtractor-s and PredictionModels-s are compatible with sklearn workflows. Naturally though, this is only an option for datasets that fit in memory.
The conventional usage with batches operation (as in above examples) is advised.


```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Feature extractors are valid sklearn Transformers.
fe = image_fe.MobileNetV3()
features = fe.fit_transform(imgs)
pl = Pipeline(["fe": fe, "classifier": LogisticRegression()])
pl.fit(imgs, targets)
pred = pl.predict(imgs)

# Prediction models can function as sklearn Estimators under the described constraints
model = mm_models.LateFusion()
dataset = data.TastyRecipes()
naked_data, targets = dataset.get_data()
model.fit(dataset, y) # works
model.fit(naked_data, y) # works

gs = GridSearchCV(model, {"combine": ["max", "sum", "stack"]}, scoring="f1")
gs.fit(dataset, y) # works
gs.fit(naked_data, y) # works

```


## Contributing TODO

## Issues, PRs etc. TODO
