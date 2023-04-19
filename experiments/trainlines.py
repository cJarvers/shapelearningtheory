import torch
from torchmetrics import Accuracy
import pytorch_lightning as pl
import sys
# local imports
sys.path.append("..")
from shapelearningtheory.linedataset import LineDataModule
from shapelearningtheory.squaredataset import SquaresDataModule
from shapelearningtheory.linearnetworks import ShallowLinear, DeepLinear
from shapelearningtheory.mlp import MLP
from shapelearningtheory.colorcategories import Grey, RedXORBlue, NotRedXORBlue

# get data:
# training dataset
traindata = LineDataModule(15, 15, range(5, 11), horizontalcolor=RedXORBlue, verticalcolor=NotRedXORBlue)
# test datasets - parametrized slightly differently to test generalization
test_sets = {
    "short": LineDataModule(15, 15, [3], horizontalcolor=RedXORBlue, verticalcolor=NotRedXORBlue), # shorter lines, correct color
    "long": LineDataModule(15, 15, [13], horizontalcolor=RedXORBlue, verticalcolor=NotRedXORBlue), # longer lines, correct color
    "coloronly": SquaresDataModule(15, 15, [5], color1=RedXORBlue, color2=NotRedXORBlue), # correct color, but squares instead of lines (cannot classify by shape)
    "grey": LineDataModule(15, 15, [7], horizontalcolor=Grey, verticalcolor=Grey), # medium length lines, no color
    "conflict": LineDataModule(15, 15, [7], horizontalcolor=NotRedXORBlue, verticalcolor=RedXORBlue) # medium length lines, incorrect color
}

# define models
shallow_model = ShallowLinear(15 * 15 * 3, 2, loss_fun=torch.nn.functional.cross_entropy, 
    metric=Accuracy("multiclass", num_classes=2))
deep_model = DeepLinear(num_inputs=15 * 15 * 3, num_hidden=1000, num_layers=3,
    num_outputs=2, loss_fun=torch.nn.functional.cross_entropy, 
    metric=Accuracy("multiclass", num_classes=2))
mlp_model = MLP(num_inputs=15 * 15 * 3, num_hidden=1000, num_layers=3,
    num_outputs=2, loss_fun=torch.nn.functional.cross_entropy, 
    metric=Accuracy("multiclass", num_classes=2))
models = {
    "shallow_linear": shallow_model,
    "deep_linear": deep_model,
    "mlp": mlp_model
}

# initialize trainers
trainers = {name: pl.Trainer(max_epochs=100) for name in models.keys()}

# train
for name, model in models.items():
    trainers[name].fit(model, traindata)


# test generalization with shorter and longer lines
test_results = {}
for name, model in models.items():
    results = {}
    for testname, testset in test_sets.items():
        results[testname] = trainers[name].test(model, testset, verbose=False)
    test_results[name] = results
# Print test results as table
cellwidth = 15
print("Test results:")
print("| " + "Network".ljust(cellwidth-1), end="|")
for testname in test_sets.keys():
    print(" " + testname.ljust(cellwidth-1), end="|")
print("\n" + ("|" + "-" * cellwidth) * (len(test_sets)+1) + "|")
for model, results in test_results.items():
    print("| " + model.ljust(cellwidth-1), end="")
    for testname, result in results.items():
        r = round(result[0]["test_metric"], ndigits=3)
        print("|" + f"{r}".rjust(cellwidth-2), end="  ")
    print("|")