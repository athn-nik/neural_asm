import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score
from torch.autograd import Variable

from modules.datasets import PairDataset
from modules.models import PairModel
from util.data_loaders import load_snli
from util.training import class_weigths, Trainer, Checkpoint, EarlyStop


def train_pairmodel(name, dataset, embeddings, word2idx, config):
    print("loading data...")
    train_data, train_labels = load_snli(dataset, "train")
    dev_data, dev_labels = load_snli(dataset, "dev")
    test_data, test_labels = load_snli(dataset, "test")

    print("creating datasets...")
    train_set = PairDataset(train_data, train_labels, word2idx,
                            name="_".join([dataset, "train"]))
    dev_set = PairDataset(dev_data, dev_labels, word2idx,
                          name="_".join([dataset, "dev"]))
    test_set = PairDataset(test_data, test_labels, word2idx,
                           name="_".join([dataset, "test"]))

    classes = train_set.label_encoder.classes_.size
    model = PairModel(embeddings=embeddings, out_size=classes, **config)

    print(model)

    weights = class_weigths(train_set.labels, to_pytorch=True)
    if torch.cuda.is_available():
        model.cuda()
        weights = weights.cuda()

    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters)

    metrics = {
        "acc": lambda y, y_hat: accuracy_score(y, y_hat),
        "precision": lambda y, y_hat: precision_score(y, y_hat,
                                                      average='macro'),
        "recall": lambda y, y_hat: recall_score(y, y_hat, average='macro'),
        "f1": lambda y, y_hat: f1_score(y, y_hat, average='macro'),
    }
    monitor = "acc"

    def pipeline(nn_model, curr_batch):
        # get the inputs (batch)
        premise, hypothesis, prem_lengths, hyp_lengths, labels = curr_batch

        # convert to Variables
        premise = Variable(premise)
        hypothesis = Variable(hypothesis)
        prem_lengths = Variable(prem_lengths)
        hyp_lengths = Variable(hyp_lengths)
        labels = Variable(labels)

        # convert to CUDA Variables
        if torch.cuda.is_available():
            premise = premise.cuda()
            hypothesis = hypothesis.cuda()
            prem_lengths = prem_lengths.cuda()
            hyp_lengths = hyp_lengths.cuda()
            labels = labels.cuda()

        outputs = nn_model(premise, hypothesis,
                           prem_lengths, hyp_lengths)

        loss = criterion(outputs, labels)

        return outputs, labels, None, loss

    trainer = Trainer(model=model,
                      task="clf",
                      train_set=train_set,
                      val_set=test_set,
                      config=config,
                      optimizer=optimizer,
                      pipeline=pipeline,
                      metrics=metrics,
                      train_batch_size=config["batch_train"],
                      eval_batch_size=config["batch_eval"],
                      use_exp=True,
                      inspect_weights=False,
                      checkpoint=Checkpoint(name=name,
                                            model=model,
                                            model_conf=config,
                                            keep_best=True,
                                            scorestamp=True,
                                            metric=monitor,
                                            mode="max",
                                            base=config["base"]),
                      early_stopping=EarlyStop(metric=monitor,
                                               mode="max",
                                               patience=config[
                                                   "patience"])
                      )
    return trainer
