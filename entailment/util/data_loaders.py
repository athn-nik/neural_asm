import os

from config import BASE_PATH

snli_datasets = {
    "normal": {
        "train": "src-train.txt",
        "dev": "src-dev.txt",
        "test": "src-test.txt",
    },
    "one": {
        "train": "one_common_at_leasttrain.txt",
        "dev": "one_common_at_leastdev.txt",
        "test": "one_common_at_leasttest.txt",
    },
    "two": {
        "train": "two_common_at_leasttrain.txt",
        "dev": "two_common_at_leastdev.txt",
        "test": "two_common_at_leasttest.txt",
    },
    "three": {
        "train": "three_common_at_leasttrain.txt",
        "dev": "three_common_at_leastdev.txt",
        "test": "three_common_at_leasttest.txt",
    }
}


def load_snli(dataset, split):
    """

    Args:
        dataset (): choose one from ["normal", "one", "two", "three"]
        split (): choose one from ["train", "dev", "test"]

    Returns:

    """
    file = os.path.join(BASE_PATH, "data", "neural_activations",
                        "snli_processed",
                        snli_datasets[dataset][split])
    X = []
    y = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            premise, hypothesis, label = line.rstrip().split("\t")
            # premise = premise.split(" ")
            # hypothesis = hypothesis.split(" ")
            X.append((premise, hypothesis))
            y.append(label)
    return X, y
