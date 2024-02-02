import os
import pandas as pd
import pytorch_lightning as pl
from statistics import mean
from typing import Any, Callable, List
import torch
from torchmetrics import Accuracy
# local imports
from shapelearningtheory.networks import make_resnet50, make_vit_b_16, \
    make_vgg_19, \
    make_mlp_small, make_convnet_small, make_rconvnet_small, \
    make_softmaxconv_small, make_ViT_small, make_AE_small
from shapelearningtheory.networks.convnet import RectangleLikeConvNet, ColorLikeConvNet, TextureLikeConvNet, LTLikeConvNet

def create_save_path(*dirs):
    path = os.path.join(*dirs)
    os.makedirs(path, exist_ok=True)
    return path

def get_basic_networks(classes, channels, imagesize):
    return {
        "MLP": lambda: make_mlp_small(num_inputs=imagesize * imagesize * channels, num_outputs=classes),
        "ConvNet": lambda: make_convnet_small(channels=channels, classes=classes),
        "rConvNet": lambda: make_rconvnet_small(channels=channels, classes=classes),
        "spcConvNet": lambda: make_softmaxconv_small(channels=channels, classes=classes),
        "ViT": lambda: make_ViT_small(imgsize=imagesize, classes=classes),
        #"autoencoder": lambda: make_AE_small(num_inputs=imagesize * imagesize * channels, classes=classes)
    }

def get_standard_networks(classes, imagesize):
    return {
        "VGG19": lambda: make_vgg_19(classes),
        "Resnet": lambda: make_resnet50(classes),
        "ViT_b_16": lambda: make_vit_b_16(imagesize, classes)
    }

def get_featurelike_networks(classes):
    return {
        "RectangleNetLike": lambda: RectangleLikeConvNet(
            loss_fun=torch.nn.functional.cross_entropy,
            metric=Accuracy("multiclass", num_classes=classes)),
        "LTNetLike": lambda: LTLikeConvNet(
            loss_fun=torch.nn.functional.cross_entropy,
            metric=Accuracy("multiclass", num_classes=classes)),
        "ColorLike": lambda: ColorLikeConvNet(
            loss_fun=torch.nn.functional.cross_entropy,
            metric=Accuracy("multiclass", num_classes=classes)),
        "TextureLike": lambda: TextureLikeConvNet(
            loss_fun=torch.nn.functional.cross_entropy,
            metric=Accuracy("multiclass", num_classes=classes))
    }

def format_table(test_names: List[str], results: dict, cellwidth: int=15):
    """Print results of test runs as a table on the command line."""
    table = "Test results:\n"
    table += "| " + "Network".ljust(cellwidth-1) + "|"
    for testname in test_names:
        table += " " + testname.ljust(cellwidth-1) + "|"
    table += "\n" + ("|" + "-" * cellwidth) * (len(test_names)+1) + "|"
    for model, model_results in results.items():
        table += "\n" + "| " + model.ljust(cellwidth-1)
        for testname, result in model_results.items():
            r = round(mean(result["test_metric"]), ndigits=3)
            table += "|" + f"{r}".rjust(cellwidth-2) + "  "
        table += "|"
    return table

def train_and_validate(model_fun: Callable, train_data: Any,
        validation_sets: dict, repetitions: int = 5, epochs: int = 100):
    """Train the given model repeatedly on the train data and
    evaluate on all validation_sets."""
    validation_results = {}
    for _ in range(repetitions):
        trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu",
                             logger=False, enable_checkpointing=False)
        model = model_fun()
        trainer.fit(model, train_data)
        for name, valset in validation_sets.items():
            metrics = trainer.test(model, valset, verbose=False)
            if not name in validation_results:
                validation_results[name] = {k: [v] for k, v in metrics[0].items()}
            else:
                for k, v in metrics[0].items():
                    validation_results[name][k].append(v)
    return validation_results

def unpack_results(test_results):
    models = []
    datasets = []
    metrics = []
    for model, model_results in test_results.items():
        for testname, result in model_results.items():
            metric = result["test_metric"]
            n = len(metric)
            models.extend([model] * n)
            datasets.extend([testname] * n)
            metrics.extend(result["test_metric"])
    df = pd.DataFrame({"model": models, "dataset": datasets, "metric": metrics})
    return df