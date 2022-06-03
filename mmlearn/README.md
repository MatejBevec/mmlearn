# Dev Whiteboard

## To-Do
- ~~look up docstring and static typing conventions~~ ✓
- ~~feature extractor interface spec~~ ✓
- ~~implement image feature extractors~~ ✓
- ~~implement MultimodalDataset class~~ ✓
- ~~model interface spec~~ ✓
- ~~look at torch lightning interface~~ ✓ (not applicable)
- text feature extractors (see issues):
  - keyword features
  - POS n-grams
  - ~~doc2vec~~ ✓
- ~~trad text classifier~~ ✓
- ~~fine-tuned X image classifier~~ ✓
- ~~dataset downloads from dropbox~~ ✓
- ~~global global constants~~ ✓
- ~~util class (log, hidden print....)~~ ✓
- ~~eval module~~ ✓
  - ~~holdout(model, dataset, metric)~~ ✓
  - ~~cross_validate(model, dataset, metric)~~ ✓
  - ~~eval-all~~ ✓
- ~~basic mm models: late fusion, early fusion with concat~~ ✓
- ~~extend MultimodalDataset interface for audio and video~~ ✓
- ~~create dataset from torch dataset (constructor or factory?)~~ ✓
- ~~verbose mode~~ ✓
- ~~models should return probabilites also~~ ✓
- ~~a multimodal dataset which includes audio~~ ✓
- research types of early fusion
- sound feature extractors |
- video feature extractors
- more image models
- docs from docstrings
- a multimodal dataset which includes video
- cut all unnecessary kwargs


## Considerations
- wrap Pytorch NN classifiers and "auto reg" sklearn classifiers into sklearn-like classes
- think about the module naming (fe.image and image might clash)
  - yes, use full names ✓
- are mutable default arguments a problem?
  - yes, use "default" string ✓
- TODO: how many separate pre-configured classifiers should be provided?
- support for when embedding space or text space is larger than memory?
  - no
- TODO: get_transformed(fes) in MultiModal dataset?
- TODO: helper train functions or separate Classifier, NeuralClassifier classes?
- is one MultimodalDataset for all modalities clean, or should i split the class?
  - one dataset
- TODO: JSON file for settings
- TODO: ! support regression?
- TODO: should the interface dtype be Ndarray instead of Tensor?
- TODO: ! exact scikit model interface? (see how skorch does it)
- TODO: is get_texts() slow?
- TODO: a more elegant way to do verbosity?
- TODO: is it strange that feature extractor take Tensors on input and output Ndarrays?
- TODO: using external libraries: how many dependencies is too many??? For example, OpenL3 installs    Tensorflow and more
- TODO: MultimodalDataset: break out modality settings into config dicts?
- TODO: ! Rethink trainable fes. Problem: you should be able to use them on batches.

## Style Guide
  - import default modules, external libraries, internal modules in this order
  - use underscores for _internal_functions
  - Google style docstrings
  - no dashes in module names
  - "image", "text", "fe", "mm" refer to image, text, feature extractors and multimodal in names
  - "targets" or "labels" refer to class values, "classes" refer to class types
  - use util.log_progress(string) to log function/model progress
  - do not use mutables in DEFAULT arguments (don't do "model=MobileNetV3()")
  - TODO: do not import specific classes or functions, call module.Class() instead
