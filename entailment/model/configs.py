"""

Model Configurations

"""

NLI_BASELINE = {
    "name": "NLI_BASELINE",
    "batch_train": 64,
    "batch_eval": 64,
    "epochs": 100,
    "embeddings_file": "glove.6B.300d.txt",
    "embeddings_dim": 300,
    "embeddings_project": False,
    "embeddings_project_dim": 100,
    "embeddings_trainable": False,
    "input_noise": 0.0,
    "input_dropout": 0.1,
    "encoder_dropout": 0.2,
    "encoder_size": 200,
    "encoder_layers": 1,
    "encoder_type": "att-rnn",
    "attention_layers": 1,
    "attention_activation": "tanh",
    "attention_dropout": 0.0,
    "rnn_type": "LSTM",
    "rnn_bidirectional": True,
    "base": None,
    "patience": 10,
    "weight_decay": 0.0,
    "clip_norm": 1,
}
