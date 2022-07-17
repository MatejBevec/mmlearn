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
- ~~streamline interface type~~ ✓
- ~~see if sklearn predictor compatibility is possible~~ ✓
    - it is, did it, not sure if its the right design decision
- ~~subsample MultimodalDataset with splicing?~~ ✓
    - again, did it, not sure if correct design decision
- ~~easily access batches per modality~~ ✓
- final batch interface
- ~~what to do with sample rate~~ ✓
- sound feature extractors |
- video feature extractors
- pin versions
- a video dataset
- more image models (2 more end-to-end lets say)
- better model tuning
- add text extractors
- research types of early fusion
- add multimodal models
- kwargs handling for complex third-party models?
- cut all unnecessary kwargs
- docs from docstrings
- devise a task for testers
- GPU
- ~~DRY dataset variations~~ ✓
- model saving capability?
- trainable fe question
- import external lib in model inits
- standardize fe attribute naming
- FEs must fit targets to correct form on input
- later: feature selection at extractor level?
- where to put generic features extractors



## Considerations
- wrap Pytorch NN classifiers and "auto reg" sklearn classifiers into sklearn-like classes
- think about the module naming (fe.image and image might clash)
  - yes, use full names ✓
- are mutable default arguments a problem?
  - yes, use "default" string ✓
- support for when embedding space or text space is larger than memory?
  - no
- support regression?
  - not yet
- JSON file for settings
  - not yet
- is one MultimodalDataset for all modalities clean, or should i split the class?
  - one dataset
- should the interface dtype be Ndarray instead of Tensor?
  - detected input, parametrized output
- Using external high-level libraries: how many dependencies is too many?
  - using high-level libraries is OK
- TODO: get_transformed(fes) in MultiModal dataset?
- TODO: helper train functions or separate Classifier, NeuralClassifier classes?
- TODO: is get_texts() slow?
- TODO: how many separate pre-configured classifiers should be provided?
- TODO: MultimodalDataset: break out modality settings into config dicts?

- TODO: ! sklearn-compatible classifier interface? (see how skorch does it)
- TODO: ! Rethink trainable fes. Problem: you should be able to use them on batches.
- TODO: ! Where to put sample rate?


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
