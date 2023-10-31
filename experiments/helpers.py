import pandas as pd
import pytorch_lightning as pl
from statistics import mean
from typing import Any, Callable, List

def print_table(test_names: List[str], results: dict, cellwidth: int=15):
    """Print results of test runs as a table on the command line."""
    print("Test results:")
    print("| " + "Network".ljust(cellwidth-1), end="|")
    for testname in test_names:
        print(" " + testname.ljust(cellwidth-1), end="|")
    print("\n" + ("|" + "-" * cellwidth) * (len(test_names)+1) + "|")
    for model, model_results in results.items():
        print("| " + model.ljust(cellwidth-1), end="")
        for testname, result in model_results.items():
            r = round(mean(result["test_metric"]), ndigits=3)
            print("|" + f"{r}".rjust(cellwidth-2), end="  ")
        print("|")

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