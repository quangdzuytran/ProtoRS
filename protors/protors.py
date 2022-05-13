import argparse
from sklearn.preprocessing import binarize

import torch
import torch.nn as nn

from protors.mllp import MLLP
from util.prototype_sim import FocalSimilarity, Binarization

class ProtoRS(nn.Module):
    def __init__(self, 
                 num_classes: int,
                 feature_net: torch.nn.Module,
                 args: argparse.Namespace,
                 add_on_layers: nn.Module = nn.Identity(),
                 use_not: bool = False,
                 left: float = None,
                 right: float = None,
                 estimated_grad: bool = False
                 ):
        super().__init__()
        self.num_classes = num_classes
        # Conv net
        self.net = feature_net
        self.add_on = add_on_layers
        # Prototype layer
        self.epsilon = 1e-4
        self.num_prototypes = args.num_prototypes
        self.prototype_shape = (args.W1, args.H1, args.num_features)
        self.prototype_layer = FocalSimilarity(self.num_prototypes,
                                        args.num_features,
                                        args.W1,
                                        args.H1,
                                        self.epsilon)
        self.binarize_layer = Binarization(self.num_prototypes)
        # MLLP
        n_discrete_features = self.num_prototypes
        n_continuous_features = 0
        self.rs_dim_list = [(n_discrete_features, n_continuous_features)] + \
                            list(map(int, args.structure.split('@'))) + \
                            [self.num_classes]
        self.mllp = MLLP(dim_list=self.rs_dim_list, 
                        use_not=use_not, 
                        left=left, 
                        right=right, 
                        estimated_grad=estimated_grad)
    
    @property
    def features_requires_grad(self) -> bool:
        return any([param.requires_grad for param in self._net.parameters()])

    @features_requires_grad.setter
    def features_requires_grad(self, val: bool):
        for param in self._net.parameters():
            param.requires_grad = val

    @property
    def add_on_layers_requires_grad(self) -> bool:
        return any([param.requires_grad for param in self._add_on.parameters()])

    @add_on_layers_requires_grad.setter
    def add_on_layers_requires_grad(self, val: bool):
        for param in self._add_on.parameters():
            param.requires_grad = val

    @property
    def prototypes_requires_grad(self) -> bool:
        return self.prototype_layer.prototype_vectors.requires_grad

    @prototypes_requires_grad.setter
    def prototypes_requires_grad(self, val: bool):
        self.prototype_layer.prototype_vectors.requires_grad = val

    @property
    def binarization_requires_grad(self) -> bool:
        return self.binarize_layer.thresholds.requires_grad

    @binarization_requires_grad.setter
    def binarization_requires_grad(self, val: bool):
        self.binarize_layer.thresholds.requires_grad = val

    @property
    def mllp_requires_grad(self) -> bool:
        return self.mllp.layer_list[-1].requires_grad

    @mllp_requires_grad.setter
    def mllp_requires_grad(self, val: bool):
        for layer in self.mllp.layer_list:
            layer.requires_grad = val
    
    def forward(self, 
                xs: torch.Tensor
                ) -> tuple:
        # Forward conv net
        features = self.net(xs)
        features = self.add_on(features)
        bs, D, W, H = features.shape
        # Compute similarities and binarize
        similarities = self.prototype_layer(features, W, H).view(bs, self.num_prototypes)
        binarized_similarities = self.binarize_layer(similarities)
        # Classify
        out_cont, out_disc = self.mllp(binarized_similarities)
        return out_cont, out_disc

    
