import math
import torch
import torch.nn as nn
from model.Prenorm_TransformerEncoder import Encoder, get_past_mask


class OhioModel(nn.Module):
    def __init__(self, d_in, num_layers, d_model, heads, d_ff, dropout, attention_dropout, single_pred=True):
        """
        Args
            d_in: int
                num of features.
            single_pred: bool
                if True, means only predict CGM and make others known
                if False, means all channels are to be predicted
        """
        super(OhioModel, self).__init__()
        if single_pred:
            predict_channels = [0]
        else:
            predict_channels = list(range(d_in))
        self.predict_channels = predict_channels
        self.encoder = Encoder(num_layers, d_model, heads, d_ff, dropout, attention_dropout)
        self.emb = nn.Linear(d_in, d_model)
        self.final_linear = nn.Linear(d_model, len(predict_channels))

    def _transformer_forward(self, x):
        """
        Args:
            x: (N, l, d_in)
        Returns:
            (N, l, len(predict_channels))
        """
        x = self.emb(x)
        mask = get_past_mask(x.shape[1]).unsqueeze(0).expand(x.shape[0], -1, -1).to(x.device)
        out = self.final_linear(self.encoder(x, mask=mask))
        return out

    def forward(self, whole_example, input_len):
        """
        Args:
            whole_example: (N, l, d_in)
            input_len: int
        Returns:
            (N, l, d_in) where self.predict_channels on position [input_len: ] has been changed by the prediction
        """
        whole_example_clone = whole_example.clone().detach()
        total_len = whole_example_clone.shape[1]
        assert input_len < total_len

        while True:
            if input_len == total_len:
                return whole_example_clone
            x = whole_example_clone[:, :input_len, :]
            y_hat = self._transformer_forward(x)
            whole_example_clone[:, input_len, self.predict_channels] = y_hat[:, -1, self.predict_channels]
            input_len += 1
