import os, sys
import torch
import logging
import pprint

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import sklearn.metrics
from sklearn.model_selection import KFold

from mmlearn import data
from mmlearn.models import base_models, image_models, mm_models, text_models
from mmlearn.util import log_progress, RESULTS_DIR
from mmlearn.data import MultimodalDataset
from mmlearn.models.base_models import PredictionModel

METRICS = {
    "ca": sklearn.metrics.accuracy_score,
    "macro_f1": lambda true, pred : sklearn.metrics.f1_score(true, pred, average="macro"),
    "precision": lambda true, pred : sklearn.metrics.precision_score(true, pred, average="macro"),
    "recall": lambda true, pred : sklearn.metrics.recall_score(true, pred, average="macro")
}

def _get_predictions(dataset, model, train_ids, test_ids):
    model.train(dataset, train_ids)
    pred = model.predict(dataset, test_ids)
    targets = dataset.get_targets(test_ids, tensor=False)
    return targets, pred

def _dataframes_to_dict(datasets, models, all_results):
    metrics = list(all_results.keys())
    all_res_dict = {mname: {} for mname in metrics}
    for mname in metrics:
        for d, dname in enumerate(datasets):
            all_res_dict[mname][dname] = {}
            for m, mdname in enumerate(models):
                all_res_dict[mname][dname][mdname] = all_results[mname][m, d]
    all_results = all_res_dict

def _check_input_eval(dataset, model):
    if not isinstance(dataset, MultimodalDataset):
        raise TypeError(f"dataset must be a MultimodalDataset instance, got {type(dataset)}.")
    if not isinstance(model, PredictionModel):
        raise TypeError(f"model must be a PredictionModel instance, got {type(model)}.")
    return dataset, model

def _check_input_eval_many(datasets, models):
    if not (isinstance(datasets, dict) and isinstance(next(iter(datasets.values())), MultimodalDataset)):
        if isinstance(datasets, MultimodalDataset):
            datasets = {f"{type(datasets).__name__}": datasets}
        else:
            raise TypeError(f"datasets must be a dict of MultimodalDataset instances\
                            (or a single one), got {type(datasets)}.")
    if not (isinstance(models, dict) and isinstance(next(iter(models.values())), PredictionModel)):
        if isinstance(models, PredictionModel):
            models = {f"{type(models).__name__}": models}
        else:
            raise TypeError(f"models must be a dict of PredictionModel instances\
                            (or a single one), got {type(models)}.")
    
    return datasets, models
    

def save_results(results, save_dir=RESULTS_DIR):
    """Save a dict of result dataframes as .csv files.

    Args:
        results: A dict of result dataframes of form {'metric_name': len(models) x len(datasets) DataFrame, ...}    
    """

    for mname in results:
        results[mname].to_csv(sep="\t", index=True, header=True)


def holdout(dataset, model, metrics=METRICS, ratio=0.7, shuffle=True,
            dataframe=False, random_state=42, verbose=True):
    """Train and evaluate given model on given dataset with a holdout test set.

    Args:
        dataset: The MultimodalDataset to use.
        model: The classification model (ClsModel) to evaluate.
        metrics: A dict of sklearn style metrics to evaluate (ex. {'ca': sklearn.metrics.accuracy_score}).
                Default is [TODO].
        ratio: The holdout ratio. Use len(dataset)*ratio examples for training.
        shuffle: Shuffle the given dataset pre-training.
        dataframe: Return results as a pandas dataframe.
        random_state: Random see to use when needed. 'None' not to use a seed.

    Returns:
        A dict of scores of form {'metric_name': number, ...}.
    """

    dataset, model = _check_input_eval(dataset, model)

    log_progress(f"Evaluating (holdout) {type(model).__name__} model on {type(dataset).__name__} dataset...",
                 verbose=verbose)
    split = int(len(dataset) * ratio)
    ids = np.random.RandomState(seed=random_state).permutation(len(dataset)) if shuffle else np.arange(len(dataset))
    train_ids, test_ids = ids[:split], ids[split:]

    targets, pred = _get_predictions(dataset, model, train_ids, test_ids)
    log_progress(f"Computing metric scores...", verbose=verbose)
    results = {mname:metrics[mname](targets, pred) for mname in metrics}

    if dataframe:
        results = pd.DataFrame.from_dict({mname:[results[mname]] for mname in results})

    return results

def holdout_many(datasets, models, metrics=METRICS, ratio=0.7, shuffle=True,
                    dataframe=True, random_state=42, verbose=True):
    """Train and evaluate multiple models on multiple datasets with a holdout test set.

    Args:
        datasets: A dict of datasets of form {'dataset_name': MultimodalDataset, ...}
        models: A dict of models of form {'model_name': ClsModel, ...}
        See eval.holdout() for more details.

    Returns:
        A dict of result dataframes of form {'metric_name': len(models) x len(datasets) DataFrame, ...}
    """

    datasets, models = _check_input_eval_many(datasets, models)

    all_results = {mname: np.ndarray([len(models), len(datasets)]) for mname in metrics}
    for m, mdname in enumerate(models):
        for d, dname in enumerate(datasets):
            results = holdout(datasets[dname], models[mdname], ratio=ratio, shuffle=shuffle, verbose=verbose)
            for mname in metrics:
                all_results[mname][m, d] = results[mname]

    if dataframe:
        all_results = {mname: pd.DataFrame(all_results[mname], index=models.keys(), columns=datasets.keys()) \
                        for mname in all_results}
    else:
        all_results = _dataframes_to_dict(datasets, models, all_results)

    return all_results
    

def cross_validate(dataset, model, metrics=METRICS, folds=4, shuffle=True,
                    dataframe=False, random_state=42, verbose=True):
    """Train and evaluate given model on given dataset with k-fold cross-validation.

    Args:
        dataset: The MultimodalDataset to use.
        model: The classification model (ClsModel) to evaluate.
        metrics: A dict of sklearn style metrics to evaluate (ex. {'ca': sklearn.metrics.accuracy_score}).
                Default is [TODO].
        folds: Number of folds (k) for k-fold cross-validation.
        shuffle: Shuffle the given dataset pre-training.
        dataframe: Return results as a pandas dataframe.
        random_state: Random see to use when needed. 'None' not to use a seed.

    Returns:
        A dict of scores of form {'metric_name': number, ...}.
    """

    dataset, model = _check_input_eval(dataset, model)

    log_progress(f"Cross-validating {type(model).__name__} model on {type(dataset).__name__} dataset...", verbose=verbose)
    kfold = KFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
    all_ids = np.arange(len(dataset))
    
    results = {mname: np.ndarray([folds,]) for mname in metrics}
    for i, (train_ids, test_ids) in enumerate(kfold.split(all_ids)):
        # train_ids = sklearn.utils.shuffle(train_ids, random_state=seed)
        # test_ids = sklearn.utils.shuffle(test_ids, random_state=seed)

        # TODO: check if retraining same model is ok
        targets, pred = _get_predictions(dataset, model, train_ids, test_ids)
        log_progress(f"Computing metric scores...", verbose=verbose)
        for mname in metrics:
            results[mname][i] = metrics[mname](targets, pred)
        #fold_results = {mname:metrics[mname](targets, pred) for mname in metrics}
    
    avg_results = {mname: np.mean(results[mname]) for mname in results}
    if dataframe:
        avg_results = pd.DataFrame.from_dict({mname:[avg_results[mname]] for mname in avg_results})

    return avg_results

def cross_validate_many(datasets, models, metrics=METRICS, folds=4, shuffle=True,
                        dataframe=True, random_state=42, verbose=True):
    """Train and evaluate multiple models on multiple datasets with cross-validation.

    Args:
        datasets: A dict of datasets of form {'dataset_name': MultimodalDataset, ...}
        models: A dict of models of form {'model_name': ClsModel, ...}
        See eval.cross_validate() for more details.

    Returns:
        A dict of result dataframes of form {'metric_name': len(models) x len(datasets) DataFrame, ...}
    """

    datasets, models = _check_input_eval_many(datasets, models)

    all_results = {mname: np.ndarray([len(models), len(datasets)]) for mname in metrics}
    for m, mdname in enumerate(models):
        for d, dname in enumerate(datasets):
            results = cross_validate(datasets[dname], models[mdname], folds=folds,
                                    shuffle=shuffle, random_state=random_state, verbose=verbose)
            for mname in metrics:
                all_results[mname][m, d] = results[mname]

    if dataframe:
        all_results = {mname: pd.DataFrame(all_results[mname], index=models.keys(), columns=datasets.keys()) \
                        for mname in all_results}
    else:
        all_results = _dataframes_to_dict(datasets, models, all_results)

    return all_results

