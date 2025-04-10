import logging
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from typing import Any, Literal

import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn

from alertbert.aitads import AlertDataset
from alertbert.preprocessing import BaseSequenceCollate, Vocabulary

"""This module contains classes defining masked language models and utilities for their 
use implemented in PyTorch."""


class MaskedLangModelEmbeddingLayer(nn.Module):
    """A module that represents the embedding layer for a masked language model.

    Args:
        features (list[str]): List of feature to use.
        encoding (str | None): Feature to use for positional encoding.
        vocabs (dict[str, Vocabulary]): Dictionary of feature vocabularies.
        dim (int): Dimension of the embeddings.
        init_std (float, optional): Standard deviation of the normal distribution used
            for initializing the embeddings. Defaults to 1.0.
        context_size (int, optional): Size of the context for positional encoding.
            Defaults to None.

    Attributes:
        dim (int): Dimension of the embeddings.
        features (list[str]): List of feature names.
        n_features (int): Number of features.
        encoding (str | None): Feature to use for positional encoding.
        vocabs (dict[str, Vocabulary]): Dictionary of feature vocabularies.
        embeddings (nn.ModuleDict): Module dictionary of feature embeddings.

    Methods:
        forward(**src: dict[str,torch.Tensor]) -> torch.Tensor:
            Forward pass of the module.

    """

    def __init__(
        self,
        features: list[str],
        encoding: str | None,
        vocabs: dict[str, Vocabulary],
        dim: int,
        init_std: float = 1.0,
        context_size: int = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.features = list(features)
        self.n_features = len(features)
        self.encoding = encoding
        self.vocabs = vocabs
        self.layernorm = nn.LayerNorm(dim)
        self.embeddings = nn.ModuleDict(
            {
                f: nn.Embedding(
                    num_embeddings=vocabs[f].vocab_size,
                    embedding_dim=dim,
                    max_norm=None,
                    norm_type=2.0,
                )
                for f in features
            }
        )
        for emb in self.embeddings.values():
            emb.weight = nn.init.normal_(emb.weight, mean=0, std=init_std)

        if self.encoding == "position":
            self.embeddings["position"] = nn.Embedding(
                num_embeddings=context_size, embedding_dim=dim
            )
        elif self.encoding == "raw_time":
            raise NotImplementedError("Learned time encoding is not supported")

    def forward(self, **src: dict[str, torch.Tensor]) -> torch.Tensor:
        x = [self.embeddings[k](src[k]) for k in self.embeddings]
        x = torch.sum(torch.stack(x), dim=0)
        return self.layernorm(x)


class RotaryEmbedding(nn.Module):
    """A module that represents the rotary positional encoding for a masked language model.

    Args:
        dim (int): Dimension of the embeddings.
        freqs (list[float]): List of frequencies to use for the positional encoding.

    Attributes:
        dim (int): Dimension of the embeddings.
        freqs (torch.Tensor): Tensor of frequencies to use for the positional encoding.

    Methods:
        forward(q: torch.Tensor, k: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor]: Forward pass of the module.

    """

    def __init__(self, dim: int, freqs: list[float]) -> None:
        super().__init__()
        self.dim = dim
        assert len(freqs) <= dim // 2
        if len(freqs) < dim // 2:
            freqs = freqs + [0.0] * (dim // 2 - len(freqs))
        freqs = torch.tensor(freqs, dtype=torch.float64)
        self.freqs = nn.Parameter(
            torch.cat([freqs, -freqs], dim=0), requires_grad=False
        )

    @torch.no_grad()
    def _get_rotation(self, position_ids: torch.Tensor) -> tuple[torch.Tensor]:
        """Returns the rotation matrices for the given position ids."""
        assert position_ids.dtype != torch.float32
        position_ids = position_ids.to(torch.float64)
        # position_ids: [batch_size, seqlen]
        # freqs: [head_dim]
        # cos, sin: [batch_size, seq_len, head_dim]
        cos = (
            torch.cos(position_ids.unsqueeze(-1) * self.freqs)
            .requires_grad_(False)
            .to(torch.float32)
        )
        sin = (
            torch.sin(position_ids.unsqueeze(-1) * self.freqs)
            .requires_grad_(False)
            .to(torch.float32)
        )
        return cos, sin

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((x2, x1), dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, p: torch.Tensor
    ) -> tuple[torch.Tensor]:
        cos, sin = self._get_rotation(p)
        # q, k: [batch_size, heads, seq_len, head_dim]
        # cos, sin: [batch_size, seq_len, head_dim]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k


class SelfAttention(nn.Module):
    """A module that represents the self-attention mechanism for a masked language model.

    Args:
        d_model (int): Dimension of the embeddings.
        nhead (int): Number of attention heads.
        rotary_pos_enc (bool, optional): Whether to use rotary positional encoding. Defaults to False.
        rotary_pos_enc_freqs (list[float], optional): List of frequencies to use for the positional encoding. Defaults to None.
        biases (bool, optional): Whether to use biases in the linear layers. Defaults to True.

    Attributes:
        nhead (int): Number of attention heads.
        d_model (int): Dimension of the embeddings.
        dim_head (int): Dimension of the embeddings per head.
        scale (float): Scaling factor for the attention weights.
        rotary_pos_enc (bool): Whether to use rotary positional encoding.

    Methods:
        forward(x: torch.Tensor, p: torch.Tensor = None, return_attn_weights: bool = False) -> torch.Tensor: Forward pass of the module.
            If rotary_pos_enc is set to True, the parameter p is used for rotary positional encoding.
            If return_attn_weights is set to True, the attention weights are returned instead of the output.

    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        rotary_pos_enc: bool = False,
        rotary_pos_enc_freqs: Sequence[float] | None = None,
        biases: bool = True,
    ) -> None:
        super().__init__()
        self.Wqkv = nn.Linear(d_model, d_model * 3, bias=biases)
        self.Wo = nn.Linear(d_model, d_model, bias=biases)
        self.nhead = nhead
        self.d_model = d_model
        self.dim_head = d_model // nhead
        self.scale = self.dim_head**-0.5
        self.rotary_pos_enc = rotary_pos_enc
        if rotary_pos_enc:
            self.rotary_emb = RotaryEmbedding(
                dim=self.dim_head, freqs=rotary_pos_enc_freqs
            )
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, x: torch.Tensor, p: torch.Tensor = None, return_attn_weights: bool = False
    ) -> torch.Tensor:
        # qkv: [batch_size, seqlen, 3, nheads, headdim]
        qkv = self.Wqkv(x).view(x.size(0), x.size(1), 3, self.nhead, self.dim_head)

        # query, key, value: [batch_size, heads, seq_len, head_dim]
        query, key, value = qkv.transpose(3, 1).unbind(dim=2)
        if self.rotary_pos_enc:
            query, key = self.rotary_emb(query, key, p)

        attn_weights = torch.matmul(query, key.transpose(2, 3)) * self.scale
        attn_weights = self.softmax(attn_weights)
        if return_attn_weights:
            return attn_weights

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        out = attn_output.view(x.size(0), x.size(1), self.d_model)
        return self.Wo(out)


class EncoderMLP(nn.Module):
    """A module that represents the multi-layer perceptron for a masked language model.

    Args:
        d_model (int): Dimension of the embeddings.
        dim_feedforward (int): Dimension of the feedforward network.
        activation (Literal["relu", "gelu"], optional): The activation function to be used. Defaults to "gelu".
        gated_activation (bool, optional): Whether to use gated activation. Defaults to True.
        biases (bool, optional): Whether to use biases in the linear layers. Defaults to True.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor: Forward pass of the module.

    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        activation: Literal["relu", "gelu"] = "gelu",
        gated_activation: bool = True,
        biases: bool = True,
    ) -> None:
        super().__init__()
        if gated_activation:
            self.linear1 = nn.Linear(d_model, dim_feedforward * 2, bias=biases)
        else:
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=biases)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=biases)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.gated_activation = gated_activation
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        if self.gated_activation:
            x, gate = torch.chunk(x, 2, dim=-1)
            x = self.activation(x) * gate
        else:
            x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """A module that represents an encoder layer in a masked language model.

    Args:
        d_model (int): Dimension of the embeddings.
        nhead (int): Number of attention heads.
        dim_feedforward (int): Dimension of the feedforward network.
        activation (Literal["relu", "gelu"], optional): The activation function to be used. Defaults to "gelu".
        gated_activation (bool, optional): Whether to use gated activation. Defaults to True.
        rotary_pos_enc (bool, optional): Whether to use rotary positional encoding. Defaults to False.
        rotary_pos_enc_freqs (list[float], optional): List of frequencies to use for the positional encoding. Defaults to None.
        biases (bool, optional): Whether to use biases in the linear layers. Defaults to True.

    Methods:
        forward(x: torch.Tensor, p: torch.Tensor = None) -> torch.Tensor: Forward pass of the module.

    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        activation: Literal["relu", "gelu"] = "gelu",
        gated_activation: bool = True,
        rotary_pos_enc: bool = False,
        rotary_pos_enc_freqs: Sequence[float] | None = None,
        biases: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = SelfAttention(
            d_model=d_model,
            nhead=nhead,
            rotary_pos_enc=rotary_pos_enc,
            rotary_pos_enc_freqs=rotary_pos_enc_freqs,
            biases=biases,
        )
        self.attn_norm = nn.LayerNorm(normalized_shape=d_model, bias=biases)
        self.mlp = EncoderMLP(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            activation=activation,
            gated_activation=gated_activation,
            biases=biases,
        )
        self.mlp_norm = nn.LayerNorm(normalized_shape=d_model, bias=biases)

    def forward(self, x: torch.Tensor, p: torch.Tensor = None) -> torch.Tensor:
        out = self.attn_norm(self.self_attn(x, p) + x)
        return self.mlp_norm(self.mlp(out) + out)


class MaskedLangModelEncoder(nn.Module):
    """A module that represents the encoder for a masked language model.

    Args:
        d_model (int): Dimension of the embeddings.
        nhead (int): Number of attention heads.
        num_layers (int): Number of encoder layers.
        dim_feedforward (int): Dimension of the feedforward network.
        activation (Literal["relu", "gelu"], optional): The activation function to be used. Defaults to "gelu".
        gated_activation (bool, optional): Whether to use gated activation. Defaults to True.
        rotary_pos_enc (bool, optional): Whether to use rotary positional encoding. Defaults to False.
        rotary_pos_enc_freqs (list[float], optional): List of frequencies to use for the positional encoding. Defaults to None.
        biases (bool, optional): Whether to use biases in the linear layers. Defaults to True.

    Methods:
        forward(x: torch.Tensor, p: torch.Tensor = None) -> torch.Tensor: Forward pass of the module.

    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        activation: Literal["relu", "gelu"] = "gelu",
        gated_activation: bool = True,
        rotary_pos_enc: bool = False,
        rotary_pos_enc_freqs: Sequence[float] | None = None,
        biases: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.modules.transformer._get_clones(
            EncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=activation,
                gated_activation=gated_activation,
                rotary_pos_enc=rotary_pos_enc,
                rotary_pos_enc_freqs=rotary_pos_enc_freqs,
                biases=biases,
            ),
            num_layers,
        )

    def forward(self, x: torch.Tensor, p: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, p)
        return x


class MultiClassificationHead(nn.Module):
    """Module that represents the classification head for a multi-classification task.

    Args:
        d_model (int): The input dimension of the head.
        targets (list[str]): The list of target names.
        vocabs (dict[str, Vocabulary]): The dictionary of target names and their corresponding Vocabulary objects.
        bias (bool, optional): Whether to use biases in the linear layers. Defaults to True.

    Attributes:
        targets (list[str]): The list of target names.
        vocabs (dict[str, Vocabulary]): The dictionary of target names and their corresponding Vocabulary objects.
        linear (nn.ModuleList): List of linear layers for each target.

    Methods:
        forward(x: torch.Tensor) -> tuple[torch.Tensor]: Forward pass of the module.
    """

    def __init__(
        self,
        d_model: int,
        targets: list[str],
        vocabs: dict[str, Vocabulary],
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.targets = targets
        self.vocabs = vocabs
        self.d_model = d_model
        self.linear = nn.ModuleDict(
            {
                t: nn.Linear(d_model, self.vocabs[t].vocab_size, bias=bias)
                for t in self.targets
            }
        )
        # set all biases to zero
        if bias:
            for linear in self.linear.values():
                nn.init.zeros_(linear.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return tuple(
            self.linear[t](x)[..., self.vocabs[t].offset :] for t in self.targets
        )


@torch.compile
class MaskedLanguageModel(nn.ModuleDict):
    """Module that represents a masked language model.

    Args:
        params (dict[str, Any]): Dictionary of model parameters, for more information see the documentation in `train_mlm.py`.
        vocabs (dict[str, Vocabulary]): Dictionary of feature vocabularies.

    Methods:
        forward(**src: dict[str,torch.Tensor]) -> tuple[torch.Tensor]: Forward pass of the module.

    """

    def __init__(
        self,
        params: dict[str, Any],
        vocabs: dict[str, Vocabulary],
    ) -> None:
        super().__init__(
            {
                "embedding": MaskedLangModelEmbeddingLayer(
                    features=params["features"],
                    encoding=(
                        params["encoding"] if params["enc_type"] == "learned" else None
                    ),
                    vocabs=vocabs,
                    dim=params["d_model"],
                    init_std=params["emb_init_std"],
                    context_size=params["context_size"],
                ),
                "encoder": MaskedLangModelEncoder(
                    d_model=params["d_model"],
                    nhead=params["nhead"],
                    num_layers=params["num_layers"],
                    dim_feedforward=params["dim_feedforward"],
                    activation=params["activation"],
                    gated_activation=params["gated_activation"],
                    rotary_pos_enc=params["enc_type"] == "rotary",
                    rotary_pos_enc_freqs=(
                        params["encoding_freqs"]
                        if params["enc_type"] == "rotary"
                        else None
                    ),
                    biases=params["biases"],
                ),
                "head": MultiClassificationHead(
                    d_model=params["d_model"],
                    targets=params["targets"],
                    vocabs=vocabs,
                    bias=params["head_bias"],
                ),
            }
        )

        self.rotary_pos_enc = params["enc_type"] == "rotary"
        self.encoding = params["encoding"]

        # tie weigths of the embedding layer and the output layer
        if params["tie_weights"]:
            for t in params["targets"]:
                self["head"].linear[t].weight = self["embedding"].embeddings[t].weight

    def forward(self, **src: dict[str, torch.Tensor]) -> tuple[torch.Tensor]:
        x = self.embedding(**src)
        x = self.encoder(x, p=src[self.encoding] if self.rotary_pos_enc else None)
        return self.head(x)


class MultiTargetLoss(nn.Module):
    """A module that combines multiple loss functions for multi-target learning.

    Args:
        loss_fns (list[nn.Module]): A list of loss functions to be combined.

    Methods:
        forward(output: tuple[torch.tensor], target: tuple[torch.tensor]) -> torch.tensor: Forward pass of the module.

    """

    def __init__(self, loss_fns: list[nn.Module]) -> None:
        super().__init__()
        self.loss_fns = nn.ModuleList(loss_fns)

    def forward(
        self, output: tuple[torch.tensor], target: tuple[torch.tensor]
    ) -> torch.tensor:
        return torch.mean(
            torch.stack(
                tuple(
                    loss_fn(out, tar)
                    for loss_fn, out, tar in zip(self.loss_fns, output, target)
                )
            )
        )


# wrappers to make the models work with TensorDicts


class MaskedLangModelInferenceWrapper(TensorDictModule):
    """A wrapper class for performing inference using a MaskedLanguageModel and TensorDicts.

    Args:
        model (MaskedLanguageModel): The masked language model to be used for inference.
        layers (list[str], optional): A list of layer names to be used from the model.
                                      Defaults to ["embedding", "encoder"].

    """

    def __init__(
        self,
        model: MaskedLanguageModel,
        layers: Sequence[str] = ("embedding", "encoder"),
    ) -> None:
        super().__init__(
            nn.Sequential(OrderedDict({layer: model[layer] for layer in layers})),
            in_keys={f: f for f in model["embedding"].features + [model.encoding]},
            out_keys=["output"],
        )
        self.encoding = model.encoding
        self.module.forward = self._star_forward

    @torch.no_grad()
    def _star_forward(self, **x: dict[str, torch.Tensor]) -> torch.Tensor:
        """This function is used to override the forward method of the nn.Sequential module to make it compatible with varying numbers of input arguments."""
        p = x.get(self.encoding, None)
        x = self.module[0](**x)
        for module in self.module[1:]:
            x = module(x, p)
        return x


class MaskedLangModelAttentionWrapper(MaskedLangModelInferenceWrapper):
    """Wrapper class for extracting attention weights from a MaskedLanguageModel.

    Args:
        model (MaskedLanguageModel): The masked language model to be used for extracting attention weights.

    Methods:
        forward(batch: TensorDict[str, torch.Tensor]) -> TensorDict[str, torch.Tensor]:
            Performs forward pass of the model on the given batch and extracts attention weights from the encoder layers.

    """

    def __init__(self, model: MaskedLanguageModel) -> None:
        super().__init__(model, layers=["embedding"])
        self.encoder = model["encoder"].layers

    def forward(
        self, batch: TensorDict[str, torch.Tensor]
    ) -> TensorDict[str, torch.Tensor]:
        batch = super().forward(batch)
        attention = []
        for module in self.encoder:
            attention.append(
                module.self_attn(
                    x=batch["output"],
                    p=batch[self.encoding],
                    return_attn_weights=True,
                )
            )
            batch["output"] = module(batch["output"], batch[self.encoding])
        batch["attention"] = torch.stack(attention)
        return batch


class MaskedLangModelEvalWrapper(TensorDictModule):
    """Wrapper class for evaluating MaskedLanguageModel with TensorDicts.

    Args:
        model (MaskedLanguageModel): The masked language model to be evaluated.

    Attributes:
        model (MaskedLanguageModel): The masked language model being evaluated.
        true_target_keys (List[str]): The keys for the true target values.
        masked_out_keys (List[str]): The keys for the masked output values.

    Methods:
        forward(batch: TensorDict[str, torch.Tensor]) -> TensorDict[str, torch.Tensor]:
            Performs forward pass of the model on the given batch.

    """

    def __init__(self, model: MaskedLanguageModel) -> None:
        super().__init__(
            model,
            in_keys={f"{f}_mask": f for f in model["embedding"].features}
            | {model.encoding: model.encoding},
            out_keys=[f"{t}_pred" for t in model["head"].targets],
        )
        self.true_target_keys = [f"{t}_true" for t in model["head"].targets]
        self.masked_out_keys = [f"{k}_mask" for k in self.out_keys]

    def forward(
        self, batch: TensorDict[str, torch.Tensor]
    ) -> TensorDict[str, torch.Tensor]:
        mask_idx = torch.unbind(batch["mask_index"])
        batch = super().forward(batch)
        for t, k in zip(self.module["head"].targets, self.true_target_keys):
            batch[k] = self.module["head"].vocabs[t].compute_targets(batch[t][mask_idx])
        for k, m in zip(self.out_keys, self.masked_out_keys):
            batch[m] = batch[k][mask_idx]
        return batch


class MaskedLangModelTrainWrapper(MaskedLangModelEvalWrapper):
    """Wrapper class for training MaskedLanguageModel with TensorDicts.

    Args:
        model (MaskedLanguageModel): The masked language model.
        loss_fn (MultiTargetLoss): The loss function.

    """

    def __init__(self, model: MaskedLanguageModel, loss_fn: MultiTargetLoss) -> None:
        super().__init__(model)
        self.loss_fn = loss_fn

    def forward(
        self, batch: TensorDict[str, torch.Tensor]
    ) -> TensorDict[str, torch.Tensor]:
        batch = super().forward(batch)
        batch["loss"] = self.loss_fn(
            tuple(batch[k] for k in self.masked_out_keys),
            tuple(batch[t] for t in self.true_target_keys),
        )
        return batch


# token clustering models


class AbstractClusteringModel:
    """An abstract class for clustering models that assigns tokens to clusters based on their attributes.
    Does not support batch processing, all inputs must have batch size 1!

    Methods:
        forward(batch: TensorDict[str, torch.Tensor]) -> TensorDict[str, torch.Tensor]:
            If implemented, assigns tokens to clusters based on their attributes.
        block_parallelization() -> None:
            Function to block parallelization inside the model if applicable.
    """

    def __call__(
        self, batch: TensorDict[str, torch.Tensor]
    ) -> TensorDict[str, torch.Tensor]:
        assert batch.batch_size_[0] == 1, (
            f"Batch size must be 1! Encountered batch of size: {batch.batch_size}"
        )
        return self.forward(batch)

    def forward(
        self, batch: TensorDict[str, torch.Tensor]
    ) -> TensorDict[str, torch.Tensor]:
        """If implemented, assigns tokens to clusters based on their attributes."""
        raise NotImplementedError("forward method must be implemented in subclass!")

    def block_parallelization(self) -> None:
        """Function to block parallelization inside the model if applicable."""


class BaselineClusteringModel(AbstractClusteringModel):
    """A baseline clustering model that assigns tokens to clusters based on their attributes.
    Does not support batch processing, all inputs must have batch size 1!

    Args:
        features (list[str]): List of features to be used for clustering.

    Methods:
        forward(batch: TensorDict[str, torch.Tensor]) -> TensorDict[str, torch.Tensor]:
            Assigns tokens to clusters based on their attributes.
    """

    def __init__(self, features: list[str]) -> None:
        self.features = features

    def _reverse_enumerate(self, iterable: Iterable[Any]) -> Iterable[tuple[Any, int]]:
        for element in enumerate(iterable):
            yield element[1], element[0]

    def forward(
        self, batch: TensorDict[str, torch.Tensor]
    ) -> TensorDict[str, torch.Tensor]:
        combined_features = torch.stack(
            [batch[f][0] for f in self.features], dim=-1
        ).numpy()
        combined_features = [tuple(item) for item in combined_features]
        # the construction in the line below ensures that the order of the cluster labels is the same as the appearance of the features.
        # this is relevant if the features are timestamps and one wants the labels to be in chronological order.
        labels = dict(self._reverse_enumerate(OrderedDict.fromkeys(combined_features)))
        batch["cluster"] = (
            torch.tensor([labels[item] for item in combined_features])
            .unsqueeze(0)
            .to(torch.int)
        )
        return batch


class TokenClusteringModel(AbstractClusteringModel):
    """A wrapper class for performing token clustering using a MaskedLanguageModel with TensorDicts,
    dimensionality reduction, and a clustering model.
    Does not support batch processing, all inputs must have batch size 1!

    Args:
        model (MaskedLangModelInferenceWrapper): The masked language model to be used.
        dim_reduction (BaseEstimator, optional): A dimensionality reduction module supporting the sklearn API.
        clustering (BaseEstimator): A clustering module supporting the sklearn API.
        precomputed_metric (callable, optional): A callable that computes the distance matrix to be used for clustering.
            In this case, the clustering module must be set to the "precomputed" metric.
        add_time_dim (bool, optional): Whether to add the time dimension to the input for clustering. Defaults to True.

    Methods:
        forward(batch: TensorDict[str, torch.Tensor]) -> TensorDict[str, torch.Tensor]:
            Embedds the input sequence using the specified layers of the MaskedLanguageModel, and fits and applies the dimensionality reduction and clustering models.
        block_parallelization() -> None:
            Blocks parallelization in dimensionality reduction and clustering.
    """

    def __init__(
        self,
        model: MaskedLangModelInferenceWrapper,
        dim_reduction: BaseEstimator = None,
        clustering: BaseEstimator = None,
        precomputed_metric: callable = None,
        add_time_dim: bool = True,
    ) -> None:
        self.model = model
        self.dim_reduction = dim_reduction
        self.clustering = clustering
        self.precomputed_metric = precomputed_metric
        self.add_time_dim = add_time_dim
        assert clustering is not None, "Clustering model must be specified!"

    def block_parallelization(self) -> None:
        """Function to block parallelization in dimensionality reduction and clustering."""
        if self.dim_reduction:
            self._check_jobs(self.dim_reduction)
        self._check_jobs(self.clustering)

    def _check_jobs(self, module: BaseEstimator) -> None:
        """Function to check if the module has a n_jobs attribute and set it to 1 if it is not already set to 1 or None."""
        if hasattr(module, "n_jobs") and not (
            module.n_jobs == 1 or module.n_jobs is None
        ):
            module.n_jobs = 1
            logging.warning(f"Module {module}.n_jobs={module.n_jobs} overriden to 1!")

    def _consolidate_labels(self, labels: np.ndarray) -> np.ndarray:
        """Function to consolidate the cluster labels to start from 0 and be consecutive.
        Raises a ValueError if the clustering encountered infinite (cluster label `-2`) or missing (cluster label `-3`) values.
        If the cluster labels contain the "noise" label `-1`, it is retained in the consolidated labels.
        """
        unique_labels, new_labels = np.unique(labels, return_inverse=True)
        if unique_labels[0] <= -2:
            raise ValueError(
                f"Clustering encountered infinite or missing values! minimal cluster label assigned: {unique_labels[0]}"
            )
        elif unique_labels[0] == -1:
            new_labels -= 1
        return new_labels

    def forward(
        self, batch: TensorDict[str, torch.Tensor], consolidate_labels: bool = True
    ) -> TensorDict[str, torch.Tensor]:
        batch = self.model(batch)
        output = batch["output"]
        output = output.cpu().numpy().squeeze()
        if self.dim_reduction:
            output = self.dim_reduction.fit_transform(output)
        batch["red_dim"] = torch.tensor(output).unsqueeze(0).to(batch.device)
        if self.add_time_dim:
            output = np.concatenate(
                (output, batch["raw_time"].cpu().numpy().squeeze().reshape(-1, 1)),
                axis=1,
            )
        if self.precomputed_metric:
            output = self.precomputed_metric(output)
        output = self.clustering.fit_predict(output)
        if consolidate_labels:
            output = self._consolidate_labels(output)
        batch["cluster"] = torch.tensor(output).unsqueeze(0).to(batch.device)
        return batch


class CombinedTimeCosineMetric:
    """A distance metric that is defined as the maximum of the cosine distance between the first n-1 dimensions (the embedding part) and
        the euclidean distance in the last dimension (the time part) of the input vectors.

    Args:
        theta (float, optional): A scaling factor for the cosine distance, vectors with the same time part and
        opposing embedding parts will have the distance theta. Defaults to 1.
    """

    def __init__(self, theta: float = 1.0) -> None:
        self.theta = theta

    def __call__(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        if y is None:
            y = x
        if x.ndim == 1:
            x = x.unsequeze(0)
        if y.ndim == 1:
            y = y.unsequeze(0)
        return np.maximum(
            cosine_distances(x[:, :-1], y[:, :-1]) / 2. * self.theta,
            np.abs(x[:, -1][:, None] - y[:, -1]),
        )


class TimeDeltaClusteringModel(AbstractClusteringModel):
    """A clustering model that assigns tokens to clusters based on the time differences
    between them as discussed in the paper [Dealing with Security Alert Flooding:
    Using Machine Learning for Domain-independent Alert Aggregation](https://dl.acm.org/doi/pdf/10.1145/3510581).
    A sequence of tokens is considered to be a cluster if the time difference between any two
    consecutive tokens is less than a specified delta.
    Does not support batch processing, all inputs must have batch size 1!

    Args:
        delta (float, optional): The time difference threshold for clustering. Defaults to 2.0.
    Methods:
        forward(batch: TensorDict[str, torch.Tensor]) -> TensorDict[str, torch.Tensor]:
            Assigns tokens to clusters based on the time differences between them.
    """

    def __init__(self, delta: float = 2.0) -> None:
        self.delta = delta

    def forward(
        self, batch: TensorDict[str, torch.Tensor]
    ) -> TensorDict[str, torch.Tensor]:
        cluster = batch["raw_time"].cpu().numpy().squeeze()
        cluster = np.diff(cluster, prepend=cluster[0])
        cluster = np.where(cluster >= self.delta, 1, 0)
        cluster = np.cumsum(cluster)
        batch["cluster"] = torch.tensor(cluster).unsqueeze(0).to(batch.device)
        return batch


# whole dataset clustering models
# TODO: add docstrings


class AbstractDatasetGroupingModel:
    """An abstract class for applying alert grouping models to AlertDatasets.

    Methods:
        forward(data: AlertDataset) -> np.ndarray:
            If implemented, groups the dataset based on the underlying model.
    """

    def __call__(self, data: AlertDataset) -> np.ndarray:
        return self.forward(data)

    def forward(self, data: AlertDataset) -> np.ndarray:
        """If implemented, embeds the dataset based on its attributes."""
        raise NotImplementedError("forward method must be implemented in subclass!")


class TimeDelta(AbstractDatasetGroupingModel):
    def __init__(self, delta: float = 2.0) -> None:
        self.delta = delta

    def forward(self, data: AlertDataset) -> np.ndarray:
        c = data.data["raw_time"]
        c = np.diff(c, prepend=c[0])
        c = np.where(c >= self.delta, 1, 0)
        return np.cumsum(c)


class AlertBERT(AbstractDatasetGroupingModel):
    def __init__(
        self,
        model: MaskedLangModelInferenceWrapper,
        collate_fn: BaseSequenceCollate,
        dim_reduction: int = 2,
        delta: float = 2.0,
        theta: float = 1.0,
        padding: int = 1024,
        readout: int = 2048,
    ) -> None:
        super().__init__()
        self.model = model
        self.collate_fn = collate_fn
        self.dim_reduction = PCA(n_components=dim_reduction) if dim_reduction else None
        self.delta = delta
        self.theta = theta
        self.padding = padding
        self.readout = readout
        self.timedelta = TimeDelta(delta=delta)
        self.metric = CombinedTimeCosineMetric(theta=theta)

    def forward(self, data: AlertDataset) -> np.ndarray:
        # get embeddings of full dataset
        embeddings = self.get_embeddings(data)

        # compute pre clustering
        pre_clustering = self.timedelta(data)

        # build graph adjacency matrix based on pre-clustering
        coords_0 = []
        coords_1 = []

        # this keeps track of how many alerts were in the previous pre-clusters
        alert_idx_offset = 0

        for pre_cluster in range(pre_clustering[-1] + 1):
            current_alerts = pre_clustering == pre_cluster
            pre_cluster_size = np.sum(current_alerts)
            if pre_cluster_size <= 1:
                # only one alert in the pre-cluster, no need to compute distances
                continue
            pre_cluster_raw_time = data.data["raw_time"][current_alerts]
            pre_cluster_time_length = pre_cluster_raw_time[-1] - pre_cluster_raw_time[0]

            if pre_cluster_time_length <= self.delta:
                # all alert pairs are relevant
                # we compute everything at once bc it is less complicated to implement
                distance_matrix = self.metric(
                    embeddings[alert_idx_offset : alert_idx_offset + pre_cluster_size]
                )
                distance_matrix = distance_matrix < self.delta
                np.fill_diagonal(distance_matrix, False)
                matrix_idxs = np.nonzero(distance_matrix)
                coords_0.append(matrix_idxs[0].astype(np.int32) + alert_idx_offset)
                coords_1.append(matrix_idxs[1].astype(np.int32) + alert_idx_offset)

            else:
                # not all alert pairs are relevant
                # we compute distance pairs only in a sliding window of appropriate size for self.delta to save memory while covering all relevant pairs

                # initially the sliding window size will be determined exactly to cover all relevant pairs
                # this is then adjusted in the subsequent iterations based on the actual data to save compute
                window_size = 1
                while (
                    pre_cluster_raw_time[window_size] - pre_cluster_raw_time[0]
                    < self.delta
                ):
                    window_size += 1

                # at first compute a sliding-window-sized square part of the distance matrix at once bc it is easier to implement
                distance_matrix = self.metric(
                    embeddings[alert_idx_offset : alert_idx_offset + window_size + 1]
                )
                distance_matrix = distance_matrix < self.delta
                np.fill_diagonal(distance_matrix, False)
                matrix_idxs = np.nonzero(distance_matrix)
                coords_0.append(matrix_idxs[0].astype(np.int32) + alert_idx_offset)
                coords_1.append(matrix_idxs[1].astype(np.int32) + alert_idx_offset)

                # then compute the remaining distance pairs in a sliding window fashion for one alert at a time
                # thus the values of distance matrix are computed in a fishbone pattern
                for i in range(window_size + 1, pre_cluster_size):
                    # adjust the window size
                    if (
                        pre_cluster_raw_time[i] - pre_cluster_raw_time[i - window_size]
                        < self.delta
                    ):
                        window_size += 1
                    elif (
                        pre_cluster_raw_time[i] - pre_cluster_raw_time[i - window_size]
                        > self.delta * 1.5
                    ):
                        while (
                            pre_cluster_raw_time[i]
                            - pre_cluster_raw_time[i - window_size]
                            > self.delta
                        ):
                            window_size -= 1
                        window_size += 1  # the while loop overshoots by one

                    # compute the distances for the current alert
                    distance_current_alert_to_window = self.metric(
                        embeddings[alert_idx_offset + i : alert_idx_offset + i + 1],
                        embeddings[
                            alert_idx_offset + i - window_size : alert_idx_offset + i
                        ],
                    ).flatten()
                    distance_current_alert_to_window = distance_current_alert_to_window < self.delta

                    # insert connections in the lists
                    window_idxs = np.nonzero(distance_current_alert_to_window)[0].astype(np.int32) + alert_idx_offset + i - window_size
                    current_alert_idxs = np.ones_like(window_idxs) * (alert_idx_offset + i)

                    coords_0.append(current_alert_idxs)
                    coords_1.append(window_idxs)

                    coords_0.append(window_idxs)
                    coords_1.append(current_alert_idxs)

            alert_idx_offset += pre_cluster_size

        assert alert_idx_offset == len(data)

        coords_0 = np.concatenate(coords_0)
        coords_1 = np.concatenate(coords_1)
        connections = np.ones_like(coords_0, dtype=bool)

        connections = coo_array((connections, (coords_0, coords_1),), shape=(len(data), len(data)))
        del coords_0, coords_1
        connections = connections.tocsr()

        # find connected components aka groups
        n_groups, pred = connected_components(connections, connection="strong")
        del connections
        return pred

    def get_embeddings(
        self, data: AlertDataset, apply_dim_red: bool = True, add_time: bool = True
    ) -> np.ndarray:
        if apply_dim_red:
            assert self.dim_reduction is not None, (
                "Dimensionality reduction module must be specified if apply_dim_red is True!"
            )

        num_contexts = len(data) // self.readout
        remainder = len(data) % self.readout
        embeddings = []

        for i in range(num_contexts):
            batch = self.collate_fn(
                [
                    data[
                        i * self.readout - self.padding : (i + 1) * self.readout
                        + self.padding
                    ]
                ]
            )
            batch = self.model(batch)
            embeddings.append(
                batch["output"][0, self.padding : -self.padding].cpu().numpy()
            )

        if remainder:
            assert len(data) - (i + 1) * self.readout == remainder
            batch = self.collate_fn(
                [data[ (i + 1) * self.readout - self.padding : len(data) + self.padding]]
            )
            batch = self.model(batch)
            embeddings.append(
                batch["output"][0, self.padding : -self.padding].cpu().numpy()
            )
        embeddings = np.concatenate(embeddings)

        # apply dimensionality reduction
        if apply_dim_red:
            embeddings = self.dim_reduction.fit_transform(embeddings)

        # add time dimension
        if add_time:
            embeddings = np.concatenate(
                (embeddings, data.data["raw_time"].reshape(-1, 1)), axis=1
            )

        return embeddings
