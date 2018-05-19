import torch

from torch import nn
from torch.autograd import Variable

from modules.attention import SelfAttention
from modules.modules import Embed, RecurrentEncoder


class ModelHelper:
    def _sort_by(self, lengths):
        """
        Sort batch data and labels by length.
        Useful for variable length inputs, for utilizing PackedSequences
        Args:
            lengths (nn.Tensor): tensor containing the lengths for the data

        Returns:
            - sorted lengths Tensor
            - sort (callable) which will sort a given iterable
                according to lengths
            - unsort (callable) which will revert a given iterable to its
                original order

        """
        batch_size = lengths.size(0)

        sorted_lengths, sorted_idx = lengths.sort()
        _, original_idx = sorted_idx.sort(0, descending=True)
        reverse_idx = torch.linspace(batch_size - 1, 0, batch_size).long()

        if lengths.data.is_cuda:
            reverse_idx = reverse_idx.cuda()

        sorted_lengths = sorted_lengths[reverse_idx]

        def sort(iterable):
            return iterable[sorted_idx][reverse_idx]

        def unsort(iterable):
            return iterable[reverse_idx][original_idx][reverse_idx]

        return sorted_lengths, sort, unsort

    def _get_mask(self, sequence, lengths, axis=0):
        """
        Construct mask for padded itemsteps, based on lengths
        """
        max_len = max(lengths.data)
        mask = Variable(torch.ones(sequence.size(axis))).detach()

        if sequence.data.is_cuda:
            mask = mask.cuda()

        for i, l in enumerate(lengths.data):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0
        return mask


class PairModel(nn.Module, ModelHelper):
    def __init__(self, embeddings, out_size, **kwargs):
        """
        Define the layer of the model and perform the initializations
        of the layers (wherever it is necessary)
        Args:
            embeddings (numpy.ndarray): the 2D ndarray with the word vectors
            out_size ():
        """
        super(PairModel, self).__init__()

        self.embeddings_project = kwargs.get("embeddings_project", False)
        self.embeddings_project_dim = kwargs.get("embeddings_project_dim", 100)
        embeddings_trainable = kwargs.get("embeddings_trainable", False)
        self.embeddings_skip = kwargs.get("embeddings_skip", False)

        input_noise = kwargs.get("input_noise", 0.)
        input_dropout = kwargs.get("input_dropout", 0.2)

        self.encoder_type = kwargs.get("encoder_type", "att-rnn")
        encoder_size = kwargs.get("encoder_size", 128)
        encoder_layers = kwargs.get("encoder_layers", 1)
        encoder_dropout = kwargs.get("encoder_dropout", 0.2)

        attention_layers = kwargs.get("attention_layers", 1)
        attention_dropout = kwargs.get("attention_dropout", 0.)
        attention_activation = kwargs.get("attention_activation", "tanh")
        self.attention_context = kwargs.get("attention_context", "last")

        rnn_type = kwargs.get("rnn_type", "LSTM")
        rnn_bidirectional = kwargs.get("rnn_bidirectional", False)

        ########################################################

        self.embedding = Embed(num_embeddings=embeddings.shape[0],
                               embedding_dim=embeddings.shape[1],
                               embeddings=embeddings,
                               noise=input_noise,
                               dropout=input_dropout,
                               trainable=embeddings_trainable)

        self.encoder = RecurrentEncoder(input_size=embeddings.shape[1],
                                        rnn_size=encoder_size,
                                        rnn_type=rnn_type,
                                        num_layers=encoder_layers,
                                        bidirectional=rnn_bidirectional,
                                        dropout=encoder_dropout)

        feature_size = encoder_size
        if rnn_bidirectional:
            feature_size *= 2

        if self.attention_context == "none":
            attention_size = feature_size
        else:
            attention_size = 2 * feature_size

        self.attention = SelfAttention(attention_size,
                                       layers=attention_layers,
                                       dropout=attention_dropout,
                                       non_linearity=attention_activation,
                                       batch_first=True)

        self.classifier = torch.nn.Sequential(
            nn.Linear(feature_size * 2, feature_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            # nn.Linear(feature_size, feature_size),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(in_features=feature_size,
                      out_features=out_size)
        )

    def _get_context(self, all_outputs, last_output):
        # todo: account for zero padded timesteps
        if self.attention_context == "mean":
            context = torch.mean(all_outputs, dim=1)
        elif self.attention_context == "last":
            context = last_output
        else:
            raise ValueError

        return context

    def _add_context(self, outputs, context):
        context = context.unsqueeze(1).expand(-1, outputs.size(1), -1)

        outputs = torch.cat([outputs, context], -1)

        return outputs

    def _remove_context(self, outputs, context):
        representations = outputs[:, :context.size(-1)]
        return representations

    def forward(self, a, b, len_a, len_b):
        """
        This is the heart of the model. This function, defines how the data
        passes through the network.
        Args:
            x (): the input data (the sentences)
            lengths (): the lengths of each sentence

        Returns: the logits for each class

        """
        lengths_a_sorted, sort_a, unsort_a = self._sort_by(len_a)
        lengths_b_sorted, sort_b, unsort_b = self._sort_by(len_b)

        a_sorted = sort_a(a)
        b_sorted = sort_b(b)

        emb_a = self.embedding(a_sorted)
        emb_b = self.embedding(b_sorted)

        outputs_a, last_outputs_a = self.encoder(emb_a, lengths_a_sorted)
        outputs_b, last_outputs_b = self.encoder(emb_b, lengths_b_sorted)

        context_a = self._get_context(outputs_a, last_outputs_a)
        context_b = self._get_context(outputs_b, last_outputs_b)

        context_outputs_a = self._add_context(outputs_a, context_b)
        context_outputs_b = self._add_context(outputs_b, context_a)

        representations_a, attentions_a = self.attention(context_outputs_a,
                                                         lengths_a_sorted)
        representations_b, attentions_b = self.attention(context_outputs_b,
                                                         lengths_b_sorted)

        representations_a = self._remove_context(representations_a, context_b)
        representations_b = self._remove_context(representations_b, context_a)

        representations_a = unsort_a(representations_a)
        representations_b = unsort_b(representations_b)

        representations = torch.cat([representations_a, representations_b], -1)

        logits = self.classifier(representations)

        return logits
