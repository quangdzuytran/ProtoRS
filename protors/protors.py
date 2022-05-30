import argparse
import os
import pickle
from sklearn.preprocessing import binarize

import torch
import torch.nn as nn

from protors.mllp import MLLP
from protors.prototype_sim import FocalSimilarity, Binarization

class ProtoRS(nn.Module):
    def __init__(self, 
                 num_classes: int,
                 feature_net: torch.nn.Module,
                 args: argparse.Namespace,
                 add_on_layers: nn.Module = nn.Identity()
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
        self.rs_dim_list = [self.num_prototypes] + \
                            list(map(int, args.structure.split('@'))) + \
                            [self.num_classes]
        self.mllp = MLLP(dim_list=self.rs_dim_list, 
                        estimated_grad=args.estimated_grad)
    
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

    def save(self, directory_path: str):
        # Make sure the target directory exists
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)
        # Save the model to the target directory
        with open(directory_path + '/model.pth', 'wb') as f:
            torch.save(self, f)

    def save_state(self, directory_path: str):
        # Make sure the target directory exists
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)
        # Save the model to the target directory
        with open(directory_path + '/model_state.pth', 'wb') as f:
            torch.save(self.state_dict(), f)
        # Save the out_map of the model to the target directory
        with open(directory_path + '/model_pickle.pkl', 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(directory_path: str):
        return torch.load(directory_path + '/model.pth')  

    def forward(self, 
                xs: torch.Tensor
                ) -> tuple:
        # Forward conv net
        features = self.net(xs)
        features = self.add_on(features)
        bs, D, W, H = features.shape
        # Compute similarities
        similarities = self.prototype_layer(features, W, H).view(bs, self.num_prototypes)
        similarities_cont = self.binarize_layer(similarities)
        similarities_disc = self.binarize_layer.binarized_forward(similarities)
        # Classify
        out_cont = self.mllp(similarities_cont)
        out_disc = self.mllp.binarized_forward(similarities_disc)
        return out_cont, out_disc

    def forward_partial(self,
                        xs: torch.Tensor
                        ) -> tuple:
        # Forward conv net
        features = self.net(xs)
        features = self.add_on(features)
        # Compute similarities
        similarities = self.prototype_layer(features, 1, 1)
        return features, similarities